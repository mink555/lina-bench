"""
평가 모듈 — 3가지 지표 산출 (Single + Parallel + No-Call 모두 지원)

Metric 1: BFCL_Score   — AST 매칭 (deterministic)
Metric 2: FC_Judgment   — 행동 판단 정확도 (deterministic)
Metric 3: FC_Quality    — 자연어 답변 품질 (LLM-as-Judge, GPT-4o)

Parallel 전용 추가 메트릭:
  - parallel_detected: 모델이 실제로 복수 tool을 호출했는가 (0/1)

No-Call 턴 (Slot Question / Relevance Detection):
  - tool을 호출하지 않아야 정답. 호출하면 전 지표 0점.
"""

from __future__ import annotations

import json
from typing import Any


# ═══════════════════════════════════════════════════════════════════
# Helper: Deep Comparison (BFCL AST 매칭 핵심)
# ═══════════════════════════════════════════════════════════════════

def deep_compare(model_val: Any, gt_val: Any) -> bool:
    """
    GT 값과 모델 출력 값을 재귀적으로 비교한다.
    - dict: GT의 모든 key가 model에 존재하고 값 일치
    - list[dict]: 순서 의존 (disclosed_conditions 등)
    - list[str/int]: 순서 무관 (coverage_interests 등)
    - scalar: 직접 비교 (int 45 == float 45.0 허용)
    """
    if isinstance(gt_val, dict) and isinstance(model_val, dict):
        return all(
            k in model_val and deep_compare(model_val[k], gt_val[k])
            for k in gt_val
        )
    if isinstance(gt_val, list) and isinstance(model_val, list):
        if len(gt_val) != len(model_val):
            return False
        if gt_val and isinstance(gt_val[0], dict):
            # list[dict] → 순서 의존
            return all(deep_compare(m, g) for m, g in zip(model_val, gt_val))
        # list[scalar] → 순서 무관 (int/float 호환 유지)
        remaining = list(model_val)
        for g in gt_val:
            found = False
            for i, m in enumerate(remaining):
                if deep_compare(m, g):
                    remaining.pop(i)
                    found = True
                    break
            if not found:
                return False
        return True
    # scalar 비교 (int/float 호환)
    if isinstance(gt_val, (int, float)) and isinstance(model_val, (int, float)):
        return abs(float(model_val) - float(gt_val)) < 1e-9
    return model_val == gt_val


# ═══════════════════════════════════════════════════════════════════
# Metric 1: BFCL_Score (AST 매칭)
# ═══════════════════════════════════════════════════════════════════

def evaluate_bfcl(
    tool_calls: list[dict],
    gt_tool_name: str,
    gt_arguments: dict,
) -> dict:
    """
    BFCL v4 스타일 AST 매칭.

    Returns:
        tool_name_acc  : 정답 tool 이름 일치 (0 or 1)
        arg_key_acc    : GT 키 중 모델이 포함한 비율
        arg_value_acc  : GT 키-값이 정확히 일치한 비율
    """
    result = {"tool_name_acc": 0.0, "arg_key_acc": 0.0, "arg_value_acc": 0.0}

    if not tool_calls:
        return result

    tc = tool_calls[0]
    fn = tc.get("function", {})
    model_name = fn.get("name", "")

    # arguments 파싱
    raw_args = fn.get("arguments", "{}")
    try:
        model_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except (json.JSONDecodeError, TypeError):
        model_args = {}

    # 1) Tool name
    name_match = model_name == gt_tool_name
    result["tool_name_acc"] = 1.0 if name_match else 0.0

    # 2-3) Arguments — tool 이름이 틀리면 arg도 0 (BFCL v4 원본 방식)
    if not name_match:
        return result

    gt_keys = set(gt_arguments.keys())
    if not gt_keys:
        result["arg_key_acc"] = 1.0
        result["arg_value_acc"] = 1.0
        return result

    model_keys = set(model_args.keys())
    matching_keys = gt_keys & model_keys
    result["arg_key_acc"] = len(matching_keys) / len(gt_keys)

    value_hits = sum(
        1 for k in gt_keys
        if k in model_args and deep_compare(model_args[k], gt_arguments[k])
    )
    result["arg_value_acc"] = value_hits / len(gt_keys)

    return result


# ═══════════════════════════════════════════════════════════════════
# Metric 1-P: BFCL_Score Parallel (AST 매칭 — 복수 tool)
# ═══════════════════════════════════════════════════════════════════

def evaluate_bfcl_parallel(
    tool_calls: list[dict],
    gt_tool_names: list[str],
    gt_arguments_list: list[dict],
) -> dict:
    """
    BFCL v4 Parallel / Parallel-Multiple 평가.

    GT 의 tool 목록과 모델 출력 tool 목록을 greedy 매칭한 뒤
    이름 일치율, key 일치율, value 일치율을 평균한다.

    Returns:
        parallel_detected : 모델이 >= N개 tool을 호출했는가 (0 or 1)
        tool_name_acc     : GT tool 중 정답 이름 비율
        arg_key_acc       : 매칭된 tool 의 arg key 일치 평균
        arg_value_acc     : 매칭된 tool 의 arg value 일치 평균
    """
    n_gt = len(gt_tool_names)
    result = {
        "parallel_detected": 0.0,
        "tool_name_acc": 0.0,
        "arg_key_acc": 0.0,
        "arg_value_acc": 0.0,
    }

    if not tool_calls or n_gt == 0:
        return result

    # parallel detection
    result["parallel_detected"] = 1.0 if len(tool_calls) >= n_gt else 0.0

    # 모델 tool call 파싱
    parsed_calls = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        raw = fn.get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            args = {}
        parsed_calls.append({"name": fn.get("name", ""), "args": args})

    # Best-match greedy: 각 GT tool → 이름 일치 + arg 유사도가 가장 높은 model call
    # (같은 이름 tool이 여러 개일 때 순서에 무관하게 최적 매칭)
    used: set[int] = set()
    name_hits = 0
    key_accs: list[float] = []
    val_accs: list[float] = []

    for gt_name, gt_args in zip(gt_tool_names, gt_arguments_list):
        best_j = None
        best_score = -1.0
        for j, pc in enumerate(parsed_calls):
            if j in used or pc["name"] != gt_name:
                continue
            # arg value 유사도 계산 → 가장 높은 후보 선택
            gt_keys = set(gt_args.keys())
            if not gt_keys:
                score = 1.0
            else:
                vh = sum(
                    1 for k in gt_keys
                    if k in pc["args"] and deep_compare(pc["args"][k], gt_args[k])
                )
                score = vh / len(gt_keys)
            if score > best_score:
                best_score = score
                best_j = j

        if best_j is not None:
            used.add(best_j)
            name_hits += 1
            model_args = parsed_calls[best_j]["args"]
            gt_keys = set(gt_args.keys())
            if not gt_keys:
                key_accs.append(1.0)
                val_accs.append(1.0)
            else:
                mk = set(model_args.keys())
                key_accs.append(len(gt_keys & mk) / len(gt_keys))
                key_accs_val = sum(
                    1 for k in gt_keys
                    if k in model_args and deep_compare(model_args[k], gt_args[k])
                )
                val_accs.append(key_accs_val / len(gt_keys))
        else:
            key_accs.append(0.0)
            val_accs.append(0.0)

    result["tool_name_acc"] = name_hits / n_gt
    result["arg_key_acc"] = sum(key_accs) / n_gt if key_accs else 0.0
    result["arg_value_acc"] = sum(val_accs) / n_gt if val_accs else 0.0
    return result


# ═══════════════════════════════════════════════════════════════════
# Metric 1-NC: BFCL_Score No-Call (Slot Question / Relevance Detection)
# ═══════════════════════════════════════════════════════════════════

def evaluate_bfcl_no_call(tool_calls: list[dict]) -> dict:
    """
    No-Call 턴의 BFCL 평가.

    BFCL은 'tool을 호출해야 하는 턴'에서만 의미가 있다.
    No-Call 턴의 미호출 정확도는 FC Judge(action_type_acc)가 커버하므로,
    BFCL 점수는 항상 0.0을 반환한다 (집계에서 제외될 마커 역할).

    Returns:
        tool_name_acc  : 0.0 (BFCL 대상 아님)
        arg_key_acc    : 0.0 (BFCL 대상 아님)
        arg_value_acc  : 0.0 (BFCL 대상 아님)
    """
    return {"tool_name_acc": 0.0, "arg_key_acc": 0.0, "arg_value_acc": 0.0}


# ═══════════════════════════════════════════════════════════════════
# Metric 2: FC_Judgment (행동 판단 — deterministic)
# ═══════════════════════════════════════════════════════════════════

def evaluate_fc_judgment(
    tool_calls: list[dict],
    gt_tool_name: str,
) -> dict:
    """
    FunctionChat 스타일 행동 판단.

    Returns:
        action_type_acc      : tool 호출을 했는가 (0 or 1)
        tool_selection_acc   : 올바른 tool을 골랐는가 (0 or 1)
        privacy_detection_acc: 개인정보 위반 턴에서 올바르게 감지했는가 (0 or 1)
    """
    has_call = bool(tool_calls)
    model_name = ""
    if has_call:
        model_name = tool_calls[0].get("function", {}).get("name", "")

    is_privacy = gt_tool_name == "privacy_violation_detector"

    return {
        "action_type_acc": 1.0 if has_call else 0.0,
        "tool_selection_acc": 1.0 if model_name == gt_tool_name else 0.0,
        "privacy_detection_acc": (
            1.0 if model_name == "privacy_violation_detector" else 0.0
        ) if is_privacy else 1.0,
    }


# ═══════════════════════════════════════════════════════════════════
# Metric 2-NC: FC_Judgment No-Call (Slot Question / Relevance Detection)
# ═══════════════════════════════════════════════════════════════════

def evaluate_fc_judgment_no_call(tool_calls: list[dict]) -> dict:
    """
    No-Call 턴의 FC Judgment.
    Tool을 호출하지 않아야 정답 (Slot Question: 추가 질문, Relevance: 거부).

    Returns:
        action_type_acc      : 1.0 if NOT called (correct), 0.0 if called
        tool_selection_acc   : 1.0 if NOT called (no tool = correct selection), 0.0 if called
        privacy_detection_acc: 1.0 (N/A — non-privacy turn)
    """
    score = 0.0 if tool_calls else 1.0
    return {
        "action_type_acc": score,
        "tool_selection_acc": score,
        "privacy_detection_acc": 1.0,
    }


# ═══════════════════════════════════════════════════════════════════
# Metric 2-P: FC_Judgment Parallel (행동 판단 — 복수 tool)
# ═══════════════════════════════════════════════════════════════════

def evaluate_fc_judgment_parallel(
    tool_calls: list[dict],
    gt_tool_names: list[str],
) -> dict:
    """
    FunctionChat 스타일 행동 판단 — Parallel 전용.

    Returns:
        action_type_acc      : 충분한 수의 tool을 호출했는가 (0 or 1)
        tool_selection_acc   : GT tool 중 올바르게 호출된 비율
        privacy_detection_acc: 개인정보 위반 tool 감지 여부
    """
    n_gt = len(gt_tool_names)
    model_names = [tc.get("function", {}).get("name", "") for tc in tool_calls]

    # 충분한 tool 호출?
    has_enough = len(tool_calls) >= n_gt

    # GT tool 커버리지 (greedy count)
    remaining = list(model_names)
    matched = 0
    for gt_name in gt_tool_names:
        if gt_name in remaining:
            remaining.remove(gt_name)
            matched += 1
    tool_coverage = matched / n_gt if n_gt else 1.0

    # Privacy detection
    is_privacy = "privacy_violation_detector" in gt_tool_names
    privacy_acc = (
        1.0 if "privacy_violation_detector" in model_names else 0.0
    ) if is_privacy else 1.0

    return {
        "action_type_acc": 1.0 if has_enough else 0.0,
        "tool_selection_acc": tool_coverage,
        "privacy_detection_acc": privacy_acc,
    }


# ═══════════════════════════════════════════════════════════════════
# Metric 3: FC_Quality (LLM-as-Judge)
# ═══════════════════════════════════════════════════════════════════

JUDGE_SYSTEM = """당신은 AI TMR 챗봇의 자연어 응답 품질을 평가하는 심사관입니다.

평가 기준 (모두 충족해야 Pass):
1. 관련성 — TMR의 요청과 관련된 응답인가?
2. 정확성 — 잘못된 정보나 환각(hallucination)이 없는가?
3. 간결성 — 바쁜 TMR이 빠르게 이해할 수 있는가?
4. 적절성 — tool 호출 맥락에 맞는 부가 설명이 있는가?

반드시 아래 JSON만 출력하세요 (다른 텍스트 금지):
{"pass": true, "reason": "판단 근거 1줄"}
또는
{"pass": false, "reason": "판단 근거 1줄"}"""


JUDGE_SYSTEM_SLOT = """당신은 AI TMR 챗봇의 Slot Question 응답을 평가하는 심사관입니다.

TMR이 정보가 부족한 요청을 했을 때, AI 챗봇은:
1. Tool을 호출하지 않고 (이미 별도 평가됨)
2. 누락된 필수 정보를 **각각 구체적으로** 질문해야 합니다.
   — "정보가 부족합니다" 같은 모호한 답변은 Fail.
   — 누락 정보 N개가 있으면 **N개 모두** 언급하고 구체적으로 물어야 Pass.

이것은 FunctionChat 벤치마크의 SLOT-all 평가 기준입니다:
"누락된 필수 정보가 여러 가지면 각각을 모두 구체적으로 요청"

반드시 아래 JSON만 출력하세요 (다른 텍스트 금지):
{"pass": true, "reason": "판단 근거 1줄", "asked_count": N}
또는
{"pass": false, "reason": "판단 근거 1줄", "asked_count": N}
(asked_count = AI가 실제로 구체적으로 물어본 누락 항목 수)"""


JUDGE_SYSTEM_RELEVANCE = """당신은 AI TMR 챗봇의 Relevance Detection 응답을 평가하는 심사관입니다.

TMR이 tool로 해결할 수 없는 요청을 했을 때, AI 챗봇은:
1. Tool을 호출하지 않고 (이미 별도 평가됨)
2. 해당 요청이 tool 범위 밖임을 설명하고
3. 가능하면 대안이나 도움이 되는 조언을 자연어로 제공해야 합니다.

반드시 아래 JSON만 출력하세요 (다른 텍스트 금지):
{"pass": true, "reason": "판단 근거 1줄"}
또는
{"pass": false, "reason": "판단 근거 1줄"}"""


def build_judge_messages(
    turn_id: str,
    user_prompt: str,
    gt_tool_name: str | list[str],
    model_text: str | None,
    *,
    gt_action: str | None = None,
    gt_missing_params: list[str] | None = None,
    has_tool_call: bool = False,
) -> list[dict] | None:
    """
    LLM-as-Judge 프롬프트를 구성한다.
    
    NL Quality 평가 로직:
    - No-Call 턴: 텍스트가 있어야 평가 가능 (없으면 None)
    - Tool-Call 턴: 텍스트가 있으면 평가, 없으면 None (tool만 호출한 경우)
    
    gt_action: "tool_call" | "slot_question" | "relevance_detection"
    gt_missing_params: Slot Question 전용 — 누락된 필수 파라미터 목록
    has_tool_call: 모델이 실제로 tool을 호출했는지 여부
    """
    # 텍스트가 없으면 평가 불가 (N/A)
    if not model_text or not model_text.strip():
        return None

    # ── Slot Question 전용 (SLOT-all) ──
    if gt_action == "slot_question" and gt_missing_params:
        params_str = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(gt_missing_params))
        return [
            {"role": "system", "content": JUDGE_SYSTEM_SLOT},
            {"role": "user", "content": (
                f"[Turn] {turn_id}\n"
                f"[TMR 입력] {user_prompt}\n"
                f"[정답 행동] tool 미호출 + 누락 정보 {len(gt_missing_params)}개를 각각 질문\n"
                f"[누락된 필수 정보]\n{params_str}\n\n"
                f"[AI 응답]\n{model_text.strip()}\n\n"
                f"위 응답이 누락 정보 {len(gt_missing_params)}개를 **모두 구체적으로** "
                f"물어보는지 판단하세요."
            )},
        ]

    # ── Relevance Detection 전용 ──
    if gt_action == "relevance_detection":
        return [
            {"role": "system", "content": JUDGE_SYSTEM_RELEVANCE},
            {"role": "user", "content": (
                f"[Turn] {turn_id}\n"
                f"[TMR 입력] {user_prompt}\n"
                f"[정답 행동] tool 미호출 + tool 범위 밖임을 설명\n"
                f"[AI 응답]\n{model_text.strip()}\n\n"
                "위 응답을 Pass/Fail로 판단하세요."
            )},
        ]

    # ── 기본: Tool Call 턴 ──
    # parallel인 경우 tool 목록 표시
    if isinstance(gt_tool_name, list):
        tool_str = " + ".join(gt_tool_name)
    else:
        tool_str = str(gt_tool_name) if gt_tool_name else "N/A"

    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": (
            f"[Turn] {turn_id}\n"
            f"[TMR 입력] {user_prompt}\n"
            f"[정답 Tool] {tool_str}\n"
            f"[AI 응답]\n{model_text.strip()}\n\n"
            "위 응답을 Pass/Fail로 판단하세요."
        )},
    ]


def parse_judge_response(raw: str) -> dict:
    """Judge 응답에서 pass/reason/asked_count 추출."""
    try:
        # JSON 블록 추출 (```json ... ``` 감싸기 대응)
        text = raw.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        obj = json.loads(text.strip())
        result = {
            "pass": bool(obj.get("pass", False)),
            "reason": str(obj.get("reason", "")),
        }
        # Slot Question 전용: asked_count (Judge가 인식한 질문 수)
        if "asked_count" in obj:
            result["asked_count"] = int(obj["asked_count"])
        return result
    except (json.JSONDecodeError, IndexError, KeyError):
        return {"pass": False, "reason": f"[parse error] {raw[:100]}"}
