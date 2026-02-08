#!/usr/bin/env python3
"""
AI TMR Assistant — Multi-Turn Benchmark Runner

사용법:
    cd my_bench
    python -m benchmark.run_benchmark              # 전체 실행
    python -m benchmark.run_benchmark --dry-run     # API 호출 없이 구조 검증
    python -m benchmark.run_benchmark --models 0 2  # 특정 모델만 (인덱스)

평가 지표 (3개, 가중치 없이 개별 산출):
    1. BFCL_Score   — tool name + args AST 매칭 (deterministic)
    2. FC_Judgment   — 행동 판단 정확도 (deterministic)
    3. FC_Quality    — 자연어 답변 품질 (LLM-as-Judge, GPT-4o)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# ── 프로젝트 루트 경로 ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from benchmark.evaluator import (
    evaluate_bfcl,
    evaluate_bfcl_parallel,
    evaluate_bfcl_no_call,
    evaluate_fc_judgment,
    evaluate_fc_judgment_parallel,
    evaluate_fc_judgment_no_call,
    build_judge_messages,
    parse_judge_response,
)

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")

# 테스트 대상 모델 5개 (OpenRouter)
MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-small-3.2-24b-instruct",
    "qwen/qwen3-32b",
    "qwen/qwen3-14b",
    "qwen/qwen3-next-80b-a3b-instruct",
]

# LLM-as-Judge 모델 (FC_Quality 전용 — OpenRouter 경유)
JUDGE_MODEL = "openai/gpt-4o"

# 시나리오별 고객 컨텍스트 (TMR이 통화 전 알고 있는 정보)
SCENARIO_CONTEXT = {
    "O1_ST1": "C001, 45세 남성, 건강보험(POL_H001)",
    "O1_ST2": "C002, 38세 여성, 실손보험(POL_I001)",
    "O1_ST3": "C003, 42세 남성, 종신보험(POL_L001)",
    "O2_ST1": "C004, 50세 남성, 암보험(POL_C001)",
    "O2_ST2": "C005, 35세 여성, 실손보험(POL_I002)",
    "O2_ST3": "C006, 47세 남성, 건강보험(POL_H002)",
}

# 시나리오 라벨 (가독성용)
SCENARIO_LABELS = {
    "O1_ST1": "청약/조건누적",
    "O1_ST2": "청약/맥락희석",
    "O1_ST3": "청약/교란주입",
    "O2_ST1": "보류/조건누적",
    "O2_ST2": "보류/맥락희석",
    "O2_ST3": "보류/교란주입",
}

SYSTEM_PROMPT = """당신은 한국 생명보험회사인 라이나생명 AI TMR 어시스턴트입니다.
TMR(텔레마케터)이 고객 통화 중 필요한 정보를 요청하면, 적절한 tool을 호출하여 답변합니다.

규칙:
1. TMR의 요청에 맞는 tool을 정확히 호출하세요.
2. 본인(전화 중인 고객) 정보만 조회 가능합니다.
   배우자/자녀/제3자 정보 요청 시 → privacy_violation_detector 호출.
3. 콜 이력, 문의 이력, 납입/청구 내역 요청 시 → privacy_violation_detector 호출.
4. 이전 대화에서 축적된 고객 정보(병력, 흡연, 예산 등)를 정확히 기억하고 반영하세요.
5. 타사 상품 비교는 불가합니다. 자사(라이나생명) 상품 간 비교만 가능합니다.
6. 정보가 부족하여 tool을 호출할 수 없으면 (상품명, 질환, 예산 등 미제공), tool을 호출하지 말고 TMR에게 필요한 정보를 먼저 물어보세요.
7. tool로 해결할 수 없는 요청(대화 기술, 감정 대응, 타사/타업종 상품 문의 등)에는 tool을 호출하지 말고 자연어로 답변하세요."""

# API 호출 설정
MAX_RETRIES = 3
RETRY_DELAY = 2.0    # seconds
CALL_DELAY = 1.0      # calls 사이 대기


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_tools() -> list[dict]:
    """tool_specs/tools_spec.py 에서 TOOLS 로드."""
    from tool_specs.tools_spec import TOOLS
    return TOOLS


def load_scenarios() -> dict[str, list[dict]]:
    """scenarios/scenarios_6_multi_turn.jsonl 로드 → {scenario_id: [turns]}"""
    path = ROOT / "scenarios" / "scenarios_6_multi_turn.jsonl"
    scenarios: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            turn = json.loads(line)
            sc = turn["scenario"]
            scenarios.setdefault(sc, []).append(turn)
    # 턴 순서 정렬
    for sc in scenarios:
        scenarios[sc].sort(key=lambda t: t["turn"])
    return scenarios


# ═══════════════════════════════════════════════════════════════════
# Mock Tool Responses (Multi-turn 대화 유지용)
# ═══════════════════════════════════════════════════════════════════

def get_mock_response(tool_name: str, arguments: dict) -> str:
    """
    각 tool에 대한 시뮬레이션 응답.
    모델이 다음 턴에서 자연스럽게 대화를 이어갈 수 있을 정도로만 구성.
    """
    mocks = {
        "customer_policy_lookup": {
            "status": "success",
            "policies": [{"policy_id": "POL_001", "product_type": "health", "status": "active"}],
        },
        "product_lookup": {
            "status": "success",
            "products": [
                {"product_id": "P_REC_01", "name": "무배당 라이나 추천상품", "monthly_premium_range": "30000-60000"},
            ],
        },
        "coverage_detail_lookup": {
            "status": "success",
            "coverages": [{"category": "requested", "amount": "3000만원", "condition": "약관 기준"}],
        },
        "rider_detail_lookup": {
            "status": "success",
            "riders": [{"name": "진단금특약", "monthly_premium": 5000, "coverage": "1000만원"}],
        },
        "underwriting_rules_lookup": {
            "status": "success",
            "rules": [{"condition": "해당 질환", "decision": "조건부 인수 가능", "surcharge": "20-50%"}],
        },
        "underwriting_policy_qa": {
            "status": "success",
            "answer": "최신 인수 기준에 따르면 해당 조건은 심사 대상입니다.",
        },
        "waiting_period_lookup": {
            "status": "success",
            "waiting_period": "90일", "reduction_period": "1년 50% 감액",
        },
        "health_declaration_guide": {
            "status": "success",
            "must_declare": True,
            "guide": "해당 조건은 고지 의무 대상입니다. 진단일, 치료 내역을 고지해야 합니다.",
        },
        "product_comparison": {
            "status": "success",
            "comparison": [{"aspect": "보장범위", "product_a": "기본", "product_b": "확대"}],
        },
        "product_change_calculator": {
            "status": "success",
            "current_surrender": 1200000, "new_premium": 45000, "recommendation": "전환 시 보장 확대",
        },
        "underwriting_eligibility_checker": {
            "status": "success",
            "eligible": True, "decision": "조건부 인수", "conditions": "할증 적용 가능",
        },
        "premium_calculator": {
            "status": "success",
            "monthly_premium": 52000,
            "breakdown": {"base": 40000, "surcharge": 12000},
        },
        "smoking_impact_calculator": {
            "status": "success",
            "non_smoker_premium": 40000, "smoker_premium": 52000, "difference": 12000,
        },
        "special_condition_lookup": {
            "status": "success",
            "conditions": [{"type": "할증", "rate": "30%", "duration": "전기간"}],
        },
        "product_recommender": {
            "status": "success",
            "recommendations": [{"product_id": "P_REC_01", "name": "추천상품", "monthly": 35000}],
        },
        "rider_recommendation": {
            "status": "success",
            "recommended_riders": [{"name": "진단금특약", "reason": "보장 보완", "monthly": 5000}],
        },
        "policy_upsell_checker": {
            "status": "success",
            "opportunities": [{"type": "cross_sell", "suggestion": "암보험 추가 추천"}],
        },
        "budget_optimizer": {
            "status": "success",
            "optimized_plan": {"product": "최적상품", "monthly": 38000, "coverages": ["암", "입원"]},
        },
        "compliance_checker": {
            "status": "success",
            "violations": [{"category": "exaggeration", "severity": "warning", "message": "표현 수정 권장"}],
        },
        "privacy_violation_detector": {
            "status": "blocked",
            "violations": [{"type": "data_access_denied", "message": "해당 정보는 개인정보 보호 규정에 의해 조회 불가합니다."}],
        },
    }
    default = {"status": "success", "message": f"{tool_name} 처리 완료"}
    return json.dumps(mocks.get(tool_name, default), ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════
# API Calls
# ═══════════════════════════════════════════════════════════════════

def create_openrouter_client() -> OpenAI:
    return OpenAI(
        api_key=OPENROUTER_KEY,
        base_url="https://openrouter.ai/api/v1",
        timeout=60.0,
    )


def create_judge_client() -> OpenAI:
    """FC_Quality 전용 — OpenRouter 경유 (OPENAI_KEY 만료 대비)."""
    return OpenAI(
        api_key=OPENROUTER_KEY,
        base_url="https://openrouter.ai/api/v1",
        timeout=60.0,
    )


def call_model(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict],
    dry_run: bool = False,
) -> dict:
    """
    모델 API 호출 + 재시도.
    Returns: {"tool_calls": [...], "content": "..."}
    """
    if dry_run:
        return {"tool_calls": [], "content": "[dry-run] 테스트 응답"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
            )
            msg = resp.choices[0].message

            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })
            return {
                "tool_calls": tool_calls,
                "content": msg.content or "",
            }
        except Exception as e:
            print(f"    [retry {attempt+1}/{MAX_RETRIES}] {model}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    return {"tool_calls": [], "content": "[error] API 호출 실패", "error": True}


def call_judge(
    client: OpenAI,
    messages: list[dict],
) -> dict:
    """LLM-as-Judge 호출 → pass/fail."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content or ""
            return parse_judge_response(raw)
        except Exception as e:
            print(f"    [judge retry {attempt+1}] {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return {"pass": False, "reason": "[error] judge 호출 실패"}


# ═══════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════

def run_scenario(
    client: OpenAI,
    judge_client: OpenAI,
    model: str,
    scenario_id: str,
    turns: list[dict],
    tools: list[dict],
    dry_run: bool = False,
) -> list[dict]:
    """
    단일 시나리오를 multi-turn으로 실행하고 턴별 결과를 반환한다.
    Single / Parallel / Parallel-Multiple 모두 지원.
    """
    # 시나리오별 system prompt 구성
    ctx = SCENARIO_CONTEXT.get(scenario_id, "")
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n[고객 정보] {ctx}"},
    ]

    results = []

    for turn in turns:
        turn_id = turn["id"]
        user_prompt = turn["user_prompt"]
        gt_tool = turn["gt_tool_name"]        # str (single) or list (parallel)
        gt_args = turn["gt_arguments"]         # dict (single) or list[dict] (parallel)
        call_type = turn.get("call_type", "single")
        gt_action = turn.get("gt_action", "tool_call")
        is_parallel = call_type in ("parallel", "parallel_multiple")
        is_no_call = call_type == "no_call"

        # 1) User 메시지 추가
        messages.append({"role": "user", "content": user_prompt})

        # 2) 모델 호출
        resp = call_model(client, model, messages, tools, dry_run=dry_run)
        tool_calls = resp["tool_calls"]
        model_text = resp["content"]

        # 3-4) 평가 분기: No-Call vs Parallel vs Single
        if is_no_call:
            bfcl = evaluate_bfcl_no_call(tool_calls)
            fc_j = evaluate_fc_judgment_no_call(tool_calls)
        elif is_parallel:
            bfcl = evaluate_bfcl_parallel(tool_calls, gt_tool, gt_args)
            fc_j = evaluate_fc_judgment_parallel(tool_calls, gt_tool)
        else:
            bfcl = evaluate_bfcl(tool_calls, gt_tool, gt_args)
            fc_j = evaluate_fc_judgment(tool_calls, gt_tool)

        # 5) Metric 3: FC_Quality (LLM-as-Judge)
        # no_call 턴에서는 gt_action + gt_missing_params를 전달 (SLOT-all 평가)
        # tool_call 턴에서도 텍스트가 있으면 평가 (없으면 N/A)
        _gt_missing = turn.get("gt_missing_params")  # Slot Question 전용
        judge_msgs = build_judge_messages(
            turn_id, user_prompt, gt_tool, model_text,
            gt_action=gt_action,
            gt_missing_params=_gt_missing,
            has_tool_call=bool(tool_calls),
        )
        if judge_msgs and not dry_run:
            fc_q = call_judge(judge_client, judge_msgs)
        elif dry_run:
            fc_q = {"pass": True, "reason": "[dry-run]"}
        else:
            fc_q = None  # 텍스트 응답 없음 → 스킵 (N/A)

        # 6) 대화 히스토리 업데이트 (모든 tool_call 포함 — multi-turn 유지)
        if is_no_call:
            # No-call 턴: tool 호출 여부와 관계없이 텍스트만 히스토리에 추가
            # (모델이 잘못 tool을 호출해도 mock response는 넣지 않음 — GT가 no_call)
            messages.append({"role": "assistant", "content": model_text or ""})
        elif tool_calls:
            tc_list = []
            for tc in tool_calls:
                tc_list.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"],
                })
            messages.append({
                "role": "assistant",
                "content": model_text if model_text else None,
                "tool_calls": tc_list,
            })
            # 각 tool_call에 대한 Mock response 추가
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": get_mock_response(fn_name, fn_args),
                })
        else:
            messages.append({"role": "assistant", "content": model_text or ""})

        # 결과 저장
        model_tool_names = [tc["function"]["name"] for tc in tool_calls] if tool_calls else []
        result = {
            "turn_id": turn_id,
            "turn": turn["turn"],
            "call_type": call_type,
            "gt_action": gt_action,
            "gt_tool": gt_tool,
            "model_tools": model_tool_names,
            "bfcl": bfcl,
            "fc_judgment": fc_j,
            "fc_quality": fc_q,
        }
        # Slot Question 전용: gt_missing_params + asked_count 기록
        if _gt_missing:
            result["gt_missing_params"] = _gt_missing
            result["gt_missing_count"] = len(_gt_missing)
            # asked_count: 모델이 tool을 잘못 호출했으면 질문 0개로 처리
            if tool_calls:
                result["slot_asked_count"] = 0
            elif fc_q and "asked_count" in fc_q:
                result["slot_asked_count"] = fc_q["asked_count"]
            else:
                result["slot_asked_count"] = 0
        results.append(result)

        # ── 턴별 진행 상태 출력 ──
        args_pct = f"{bfcl['arg_value_acc']:>4.0%}"
        qual_ch = "P" if fc_q and fc_q.get("pass") else "F" if fc_q else "-"

        if is_no_call:
            correct = not bool(tool_calls)
            ok_ch = "O" if correct else "X"
            if gt_action == "slot_question":
                action_label = "NO-CALL: 정보부족→질문"
            else:
                action_label = "NO-CALL: 범위밖→거부"
            if tool_calls:
                wrong_tool = tool_calls[0]["function"]["name"]
                mismatch = f"  -> {wrong_tool} (should NOT call)"
            else:
                mismatch = ""
            # Slot Question: asked_count 표시 (SLOT-all)
            if gt_action == "slot_question" and _gt_missing:
                if tool_calls:
                    asked = 0  # tool을 잘못 호출 → 질문 0개
                elif fc_q:
                    asked = fc_q.get("asked_count", 0)
                else:
                    asked = 0
                label_full = f"{action_label} [{asked}/{len(_gt_missing)}]"
            else:
                label_full = action_label
            print(f"    | {turn_id:<13} | [{ok_ch}] {label_full:<34} | {'N/A':>5} | {qual_ch} |{mismatch}")
        elif is_parallel:
            n_gt = len(gt_tool) if isinstance(gt_tool, list) else 1
            n_hit = round(bfcl["tool_name_acc"] * n_gt)
            gt_tools = gt_tool if isinstance(gt_tool, list) else [gt_tool]
            p_detected = "Y" if bfcl.get("parallel_detected", 0) == 1 else "N"
            tag = f"[{n_hit}/{n_gt}]"
            # 추가 tool은 테이블 오른쪽(마지막 | 뒤)에 표시 → 프레임 유지
            extra = "  +" + ", +".join(gt_tools[1:]) if len(gt_tools) > 1 else ""
            print(f"    | {turn_id:<13} | {tag} {gt_tools[0]:<30} | {args_pct} | {qual_ch} | P:{p_detected}{extra}")
        else:
            tool_ok = "O" if bfcl["tool_name_acc"] == 1 else "X"
            gt_name = gt_tool if isinstance(gt_tool, str) else gt_tool[0]
            if tool_calls:
                model_tool_name = tool_calls[0]["function"]["name"]
                mismatch = f"  -> {model_tool_name}" if model_tool_name != gt_name else ""
            else:
                mismatch = "  -> (no tool call)" if tool_ok == "X" else ""
            print(f"    | {turn_id:<13} | [{tool_ok}] {gt_name:<34} | {args_pct} | {qual_ch} |{mismatch}")

        if not dry_run:
            time.sleep(CALL_DELAY)

    return results


# ═══════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════

def _agg_bucket(turns: list[dict]) -> dict:
    """턴 리스트 → bfcl / fc_judgment / fc_quality 집계 (공통 헬퍼)."""
    bfcl_acc = defaultdict(list)
    fcj_acc = defaultdict(list)
    fcq_pass = 0
    fcq_total = 0

    for t in turns:
        for k, v in t["bfcl"].items():
            bfcl_acc[k].append(v)
        for k, v in t["fc_judgment"].items():
            fcj_acc[k].append(v)
        if t["fc_quality"] is not None:
            fcq_total += 1
            if t["fc_quality"].get("pass"):
                fcq_pass += 1

    return {
        "turns": len(turns),
        "bfcl": {k: round(sum(v)/len(v), 4) for k, v in bfcl_acc.items()} if bfcl_acc else {},
        "fc_judgment": {k: round(sum(v)/len(v), 4) for k, v in fcj_acc.items()} if fcj_acc else {},
        "fc_quality_pass_rate": round(fcq_pass / fcq_total, 4) if fcq_total else None,
    }


def _agg_split(all_turns: list[dict]) -> dict:
    """BFCL은 tool_call 턴만, FC/NL은 전체 턴에서 집계.

    Tool Acc / Arg Acc는 '호출해야 할 턴'에서만 보는 것이 정확하며,
    no_call 턴의 미호출 정확도는 FC Judge와 NC:Acc가 별도로 커버함.
    """
    tc_turns = [t for t in all_turns if t.get("call_type", "single") != "no_call"]
    tc_agg = _agg_bucket(tc_turns) if tc_turns else {
        "turns": 0, "bfcl": {}, "fc_judgment": {}, "fc_quality_pass_rate": None,
    }
    all_agg = _agg_bucket(all_turns)
    return {
        "turns": len(all_turns),
        "tool_call_turns": len(tc_turns),
        "bfcl": tc_agg["bfcl"],
        "fc_judgment": all_agg["fc_judgment"],
        "fc_quality_pass_rate": all_agg["fc_quality_pass_rate"],
    }


def aggregate(all_results: dict) -> dict:
    """
    모델별 → 시나리오별 → 전체 집계.
    Single / Parallel 분리 집계 포함.
    """
    summary = {}

    for model, scenarios in all_results.items():
        all_turns = []
        single_turns = []
        parallel_turns = []
        no_call_turns = []
        per_scenario = {}

        for sc_id, turns in scenarios.items():
            all_turns.extend(turns)

            sc_single = [t for t in turns if t.get("call_type", "single") == "single"]
            sc_parallel = [t for t in turns if t.get("call_type", "single") in ("parallel", "parallel_multiple")]
            sc_no_call = [t for t in turns if t.get("call_type", "single") == "no_call"]
            single_turns.extend(sc_single)
            parallel_turns.extend(sc_parallel)
            no_call_turns.extend(sc_no_call)

            # per-scenario: BFCL은 tool_call턴만, FC/NL은 전체턴
            per_scenario[sc_id] = _agg_split(turns)
            per_scenario[sc_id]["single"] = _agg_bucket(sc_single) if sc_single else None
            per_scenario[sc_id]["parallel"] = _agg_bucket(sc_parallel) if sc_parallel else None
            per_scenario[sc_id]["no_call"] = _agg_bucket(sc_no_call) if sc_no_call else None

        # overall: BFCL은 tool_call턴(single+parallel)만, FC/NL은 전체턴
        overall = _agg_split(all_turns)
        overall_single = _agg_bucket(single_turns)
        overall_parallel = _agg_bucket(parallel_turns)
        overall_no_call = _agg_bucket(no_call_turns)

        # parallel 전용 metric: parallel_detected 평균
        pd_vals = [t["bfcl"].get("parallel_detected", 0) for t in parallel_turns]
        parallel_detected_rate = round(sum(pd_vals) / len(pd_vals), 4) if pd_vals else None

        # no_call 전용 metric: 정확도 — FC Judge의 action_type_acc 사용
        # (BFCL은 tool_call 턴 전용이므로 no_call에는 FC Judge가 기준)
        nc_correct = sum(1 for t in no_call_turns if t["fc_judgment"]["action_type_acc"] == 1.0)
        nc_acc = round(nc_correct / len(no_call_turns), 4) if no_call_turns else None

        # no_call sub-type 정확도
        slot_q_turns = [t for t in no_call_turns if t.get("gt_action") == "slot_question"]
        rel_det_turns = [t for t in no_call_turns if t.get("gt_action") == "relevance_detection"]
        slot_q_acc = round(sum(1 for t in slot_q_turns if t["fc_judgment"]["action_type_acc"] == 1.0) / len(slot_q_turns), 4) if slot_q_turns else None
        rel_det_acc = round(sum(1 for t in rel_det_turns if t["fc_judgment"]["action_type_acc"] == 1.0) / len(rel_det_turns), 4) if rel_det_turns else None

        # SLOT-all completeness: 누락 정보 전부 물어봤는가?
        slot_complete_vals = []
        for t in slot_q_turns:
            gt_cnt = t.get("gt_missing_count", 0)
            asked = t.get("slot_asked_count")
            if gt_cnt > 0 and asked is not None:
                slot_complete_vals.append(min(asked / gt_cnt, 1.0))
        slot_completeness = round(sum(slot_complete_vals) / len(slot_complete_vals), 4) if slot_complete_vals else None

        summary[model] = {
            **overall,
            "total_turns": len(all_turns),
            "single": overall_single,
            "parallel": {**overall_parallel, "parallel_detected_rate": parallel_detected_rate},
            "no_call": {
                **overall_no_call,
                "no_call_acc": nc_acc,
                "slot_question_acc": slot_q_acc,
                "relevance_detection_acc": rel_det_acc,
                "slot_completeness": slot_completeness,
            },
            "per_scenario": per_scenario,
        }

    return summary


# ═══════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════

def _pct(val: float | None) -> str:
    """float -> 퍼센트 문자열 (5자 고정폭)."""
    if val is None:
        return " N/A "
    return f"{val:5.1%}"


def _bar(val: float, width: int = 10) -> str:
    """0.0~1.0 -> ASCII 바."""
    filled = round(val * width)
    return "#" * filled + "." * (width - filled)


def _display_width(s: str) -> int:
    """터미널 표시 너비 (CJK 한글 = 2칸, ASCII = 1칸)."""
    import unicodedata
    w = 0
    for ch in s:
        eaw = unicodedata.east_asian_width(ch)
        w += 2 if eaw in ("W", "F") else 1
    return w


def _center_cjk(s: str, width: int) -> str:
    """CJK 폭 고려 center 정렬."""
    dw = _display_width(s)
    if dw >= width:
        return s
    pad = width - dw
    left = pad // 2
    right = pad - left
    return " " * left + s + " " * right


def print_summary(summary: dict):
    """콘솔에 가독성 좋은 테이블 출력 (순수 ASCII — 정렬 보장)."""
    scenarios = list(next(iter(summary.values()))["per_scenario"].keys())

    # ── 지표 설명 ──────────────────────────────────
    print()
    print("=" * 100)
    print("  BENCHMARK RESULTS")
    print("=" * 100)
    print()
    print("  Metric 설명:")
    print("    BFCL Score  = tool 이름 + 인자 AST 매칭 (tool_call 턴에서만, 이름 틀리면 arg=0)")
    print("    FC Judgment = 행동 판단 정확도: 호출 여부, tool 선택, 개인정보 (전체 턴)")
    print("    NL Quality  = 자연어 답변 품질 (GPT-4o Pass/Fail)")
    print("    NC:Acc      = No-Call 정확도: FC Judge action_type_acc 기준")
    print()

    # ── [1] 종합 점수 ───────────────────────────────
    C = [28, 12, 12, 12, 8, 12]  # col widths
    sep = "+" + "+".join("-" * c for c in C) + "+"

    print("  [1] 종합 점수 (3 Metrics)")
    print(sep)
    print(
        f"| {'Model':<{C[0]}}|"
        f"{'BFCL':^{C[1]}}|"
        f"{'FC Judge':^{C[2]}}|"
        f"{'NL Qual':^{C[3]}}|"
        f"{'Turns':^{C[4]}}|"
        f"{'BFCL bar':^{C[5]}}|"
    )
    print(
        f"| {'(tool+args)':>{C[0]}}|"
        f"{'AST':^{C[1]}}|"
        f"{'action':^{C[2]}}|"
        f"{'GPT-4o':^{C[3]}}|"
        f"{'':^{C[4]}}|"
        f"{'':^{C[5]}}|"
    )
    print(sep)

    for model, data in summary.items():
        bfcl_avg = sum(data["bfcl"].values()) / len(data["bfcl"])
        fcj_avg = sum(data["fc_judgment"].values()) / len(data["fc_judgment"])
        fcq = data.get("fc_quality_pass_rate")
        short = model.split("/")[-1][:C[0] - 2]
        bar = _bar(bfcl_avg)

        print(
            f"| {short:<{C[0]}}|"
            f"{bfcl_avg:^{C[1]}.1%}|"
            f"{fcj_avg:^{C[2]}.1%}|"
            f"{_pct(fcq):^{C[3]}}|"
            f"{data['total_turns']:^{C[4]}}|"
            f"{bar:^{C[5]}}|"
        )

    print(sep)
    print()

    # ── [2] BFCL Sub-metrics (Single vs Parallel vs No-Call 분리) ──
    C2 = [28, 10, 10, 10, 10, 10, 10, 10]
    sep2 = "+" + "+".join("-" * c for c in C2) + "+"

    print("  [2] Tool Acc 상세 — Single vs Parallel vs No-Call")
    print("      Single = 단일 tool 호출  |  Parallel = 복수 tool 동시 호출  |  NC = tool 미호출 정답")
    print(sep2)
    print(
        f"| {'Model':<{C2[0]}}|"
        f"{'S:Tool':^{C2[1]}}|"
        f"{'S:Arg':^{C2[2]}}|"
        f"{'P:Tool':^{C2[3]}}|"
        f"{'P:Arg':^{C2[4]}}|"
        f"{'P:감지':^{C2[5]}}|"
        f"{'NC:Acc':^{C2[6]}}|"
        f"{'NC:SQ':^{C2[7]}}|"
    )
    print(sep2)

    for model, data in summary.items():
        short = model.split("/")[-1][:C2[0] - 2]
        s = data.get("single", {}).get("bfcl", {})
        p = data.get("parallel", {}).get("bfcl", {})
        pd_rate = data.get("parallel", {}).get("parallel_detected_rate")
        nc = data.get("no_call", {})

        s_name = s.get("tool_name_acc", 0)
        s_argv = s.get("arg_value_acc", 0)

        p_name = p.get("tool_name_acc", 0)
        p_argv = p.get("arg_value_acc", 0)

        nc_acc = nc.get("no_call_acc")
        nc_sq = nc.get("slot_question_acc")

        print(
            f"| {short:<{C2[0]}}|"
            f"{s_name:^{C2[1]}.1%}|"
            f"{s_argv:^{C2[2]}.1%}|"
            f"{p_name:^{C2[3]}.1%}|"
            f"{p_argv:^{C2[4]}.1%}|"
            f"{_pct(pd_rate):^{C2[5]}}|"
            f"{_pct(nc_acc):^{C2[6]}}|"
            f"{_pct(nc_sq):^{C2[7]}}|"
        )

    print(sep2)
    print("  * P:감지 = 모델이 실제로 복수 tool을 호출한 비율")
    print("  * NC:Acc = No-Call 턴 정답률 (tool 미호출 = 정답)")
    print("  * NC:SQ  = Slot Question 정답률 (정보 부족 시 tool 미호출)")
    print()

    # ── [2b] Slot Question SLOT-all ──
    print("  [2b] Slot Question: SLOT-all Completeness (누락 정보 전부 질문했는가?)")
    print("       FunctionChat 기준: 누락 필수 정보가 여러 개면 각각을 모두 구체적으로 요청해야 함")
    print()
    for model, data in summary.items():
        short = model.split("/")[-1][:28]
        nc = data.get("no_call", {})
        sq_acc = nc.get("slot_question_acc")
        sq_comp = nc.get("slot_completeness")
        rd_acc = nc.get("relevance_detection_acc")
        sq_str = f"{sq_acc:.0%}" if sq_acc is not None else "N/A"
        comp_str = f"{sq_comp:.0%}" if sq_comp is not None else "N/A"
        rd_str = f"{rd_acc:.0%}" if rd_acc is not None else "N/A"
        print(f"  {short:<28} SQ:미호출={sq_str}  SQ:SLOT-all={comp_str}  RD:미호출={rd_str}")
    print()

    # ── [3] 시나리오별 Tool Name Accuracy ──────────
    sc_w = 16
    left_w = 28
    sep3 = "+" + "-" * left_w + "+" + (("-" * sc_w + "+") * len(scenarios))

    print("  [3] 시나리오별 Tool Name Accuracy")
    print(sep3)
    # 시나리오 ID + 라벨
    header_cells = ""
    for s in scenarios:
        label = SCENARIO_LABELS.get(s, s)
        header_cells += f"{s:^{sc_w}}|"
    print(f"| {'Model':<{left_w}}|{header_cells}")
    # 라벨 행 (CJK 폭 보정)
    label_cells = ""
    for s in scenarios:
        label = SCENARIO_LABELS.get(s, "")
        label_cells += _center_cjk(label, sc_w) + "|"
    print(f"| {'':^{left_w}}|{label_cells}")
    print(sep3)

    for model, data in summary.items():
        short = model.split("/")[-1][:left_w - 2]
        cells = ""
        for s in scenarios:
            if s in data["per_scenario"]:
                v = data["per_scenario"][s]["bfcl"].get("tool_name_acc", 0)
                cells += f"{v:^{sc_w}.0%}|"
            else:
                cells += f"{'N/A':^{sc_w}}|"
        print(f"| {short:<{left_w}}|{cells}")

    print(sep3)
    print()

    # ── [4] FC Judgment Sub-metrics ─────────────────
    C4 = [28, 14, 14, 14, 14]
    sep4 = "+" + "+".join("-" * c for c in C4) + "+"

    print("  [4] FC Judgment Sub-metrics (행동 판단 상세)")
    print("       action_type  = tool을 호출했는가?")
    print("       tool_select  = 올바른 tool을 골랐는가?")
    print("       privacy_det  = 개인정보 위반 요청을 감지했는가?")
    print(sep4)
    print(
        f"| {'Model':<{C4[0]}}|"
        f"{'action_type':^{C4[1]}}|"
        f"{'tool_select':^{C4[2]}}|"
        f"{'privacy_det':^{C4[3]}}|"
        f"{'average':^{C4[4]}}|"
    )
    print(sep4)

    for model, data in summary.items():
        j = data["fc_judgment"]
        short = model.split("/")[-1][:C4[0] - 2]
        avg = sum(j.values()) / len(j)
        print(
            f"| {short:<{C4[0]}}|"
            f"{j['action_type_acc']:^{C4[1]}.1%}|"
            f"{j['tool_selection_acc']:^{C4[2]}.1%}|"
            f"{j['privacy_detection_acc']:^{C4[3]}.1%}|"
            f"{avg:^{C4[4]}.1%}|"
        )

    print(sep4)
    print()

    # ── [5] 시나리오별 2-Benchmark 상세 ─────────────
    C5 = [28, 8, 8, 8, 8, 8, 8, 7]
    sep5 = "+" + "+".join("-" * c for c in C5) + "+"

    print("  [5] 시나리오별 상세 (BFCL + FC Judgment + NL Quality)")
    print("      B:name/key/val = BFCL AST 매칭  |  F:act/sel/priv = FC 행동판단  |  NL = 답변품질")
    print()

    models_list = list(summary.keys())

    for sc_id in scenarios:
        sc_label = SCENARIO_LABELS.get(sc_id, "")
        sc_info = summary[models_list[0]]["per_scenario"].get(sc_id, {})
        sc_turns = sc_info.get("turns", "?")
        # parallel 턴 수 표시
        p_info = sc_info.get("parallel", {})
        p_count = p_info.get("turns", 0) if p_info else 0
        nc_info = sc_info.get("no_call", {})
        nc_count = nc_info.get("turns", 0) if nc_info else 0
        s_count = sc_turns - p_count - nc_count if isinstance(sc_turns, int) else "?"
        print(f"    {sc_id} [{sc_label}] {sc_turns}turns (S:{s_count}, P:{p_count}, NC:{nc_count})")
        print(f"    {sep5}")
        print(
            f"    | {'Model':<{C5[0]}}|"
            f"{'B:name':^{C5[1]}}|"
            f"{'B:key':^{C5[2]}}|"
            f"{'B:val':^{C5[3]}}|"
            f"{'F:act':^{C5[4]}}|"
            f"{'F:sel':^{C5[5]}}|"
            f"{'F:priv':^{C5[6]}}|"
            f"{'NL':^{C5[7]}}|"
        )
        print(f"    {sep5}")

        for model in models_list:
            sc_data = summary[model]["per_scenario"].get(sc_id)
            if not sc_data:
                continue
            short = model.split("/")[-1][:C5[0] - 2]
            b = sc_data["bfcl"]
            j = sc_data["fc_judgment"]
            nl = sc_data.get("fc_quality_pass_rate")

            print(
                f"    | {short:<{C5[0]}}|"
                f"{b.get('tool_name_acc', 0):^{C5[1]}.0%}|"
                f"{b.get('arg_key_acc', 0):^{C5[2]}.0%}|"
                f"{b.get('arg_value_acc', 0):^{C5[3]}.0%}|"
                f"{j.get('action_type_acc', 0):^{C5[4]}.0%}|"
                f"{j.get('tool_selection_acc', 0):^{C5[5]}.0%}|"
                f"{j.get('privacy_detection_acc', 0):^{C5[6]}.0%}|"
                f"{_pct(nl):^{C5[7]}}|"
            )

        print(f"    {sep5}")
        print()

    print()


# ═══════════════════════════════════════════════════════════════════
# Turn-Point Analysis + Interpretation
# ═══════════════════════════════════════════════════════════════════

# 분석 기준 턴 포인트 (no_call 추가로 최대 19턴)
TURN_CUTOFFS = [3, 5, 7, 10, 13, 15, 17, 19]

# 해석용 임계값
THRESHOLD_SAFE = 0.90      # 90%+ → 안정
THRESHOLD_CRITICAL = 0.85  # 85%+ → 절대 임계선
THRESHOLD_WARNING = 0.75   # 75%+ → 경고
# 75% 미만 → 위험/붕괴


def _collect_turn_data(all_results: dict) -> dict:
    """
    모델별로 모든 시나리오의 턴 결과를 turn_number 기준으로 수집.
    Returns: {model: {turn_num: [turn_result, ...]}}
    """
    data = {}
    for model, scenarios in all_results.items():
        by_turn = defaultdict(list)
        for sc_id, turns in scenarios.items():
            for t in turns:
                by_turn[t["turn"]].append(t)
        data[model] = dict(by_turn)
    return data


def _cumulative_metric(
    turn_data: dict[int, list],
    metric_path: str,
    up_to: int,
    *,
    exclude_no_call: bool = False,
) -> tuple[float | None, int]:
    """
    Turn 1~up_to 까지 누적 평균을 계산.
    metric_path: "bfcl.tool_name_acc" 등 (dot notation)
    exclude_no_call=True이면 no_call 턴 제외 (BFCL 지표용)
    Returns: (average, sample_count)
    """
    parts = metric_path.split(".")
    values = []
    for tn in range(1, up_to + 1):
        if tn not in turn_data:
            continue
        for t in turn_data[tn]:
            if exclude_no_call and t.get("call_type", "single") == "no_call":
                continue
            obj = t
            for p in parts:
                obj = obj.get(p, {}) if isinstance(obj, dict) else None
                if obj is None:
                    break
            if obj is not None and isinstance(obj, (int, float)):
                values.append(float(obj))
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def _per_turn_metric(
    turn_data: dict[int, list],
    metric_path: str,
    turn_num: int,
    *,
    exclude_no_call: bool = False,
) -> tuple[float | None, int]:
    """특정 턴 번호의 평균 metric. exclude_no_call=True이면 no_call 턴 제외."""
    if turn_num not in turn_data:
        return None, 0
    parts = metric_path.split(".")
    values = []
    for t in turn_data[turn_num]:
        if exclude_no_call and t.get("call_type", "single") == "no_call":
            continue
        obj = t
        for p in parts:
            obj = obj.get(p, {}) if isinstance(obj, dict) else None
            if obj is None:
                break
        if obj is not None and isinstance(obj, (int, float)):
            values.append(float(obj))
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def _find_threshold_turn(
    turn_data: dict[int, list],
    metric_path: str,
    threshold: float,
    *,
    exclude_no_call: bool = False,
) -> int | None:
    """누적 accuracy가 threshold 아래로 처음 떨어지는 turn 번호 반환."""
    max_turn = max(turn_data.keys()) if turn_data else 0
    for tn in range(1, max_turn + 1):
        val, cnt = _cumulative_metric(turn_data, metric_path, tn, exclude_no_call=exclude_no_call)
        if val is not None and val < threshold:
            return tn
    return None


def _zone_label(val: float | None) -> str:
    """값 기반 구간 라벨 (85% = 절대 임계선)."""
    if val is None:
        return "   -   "
    if val >= THRESHOLD_SAFE:
        return " SAFE  "
    if val >= THRESHOLD_CRITICAL:
        return " GOOD  "
    if val >= THRESHOLD_WARNING:
        return " RISK  "
    return "DANGER "


def _turn_performance(turn_result: dict) -> float:
    """턴 하나의 Performance.

    tool_call 턴: (Tool + Arg + FC) / 3
    no_call 턴:   FC Judge만 (BFCL은 '호출해야 할 턴'에서만 측정)
    """
    fcj_vals = list(turn_result["fc_judgment"].values())
    fc = sum(fcj_vals) / len(fcj_vals) if fcj_vals else 0
    if turn_result.get("call_type", "single") == "no_call":
        return fc
    tool = turn_result["bfcl"]["tool_name_acc"]
    arg = turn_result["bfcl"]["arg_value_acc"]
    return (tool + arg + fc) / 3


def _cumulative_performance(turn_data: dict[int, list], up_to: int) -> tuple[float | None, int]:
    """Turn 1~up_to 까지 누적 Performance 평균."""
    values = []
    for tn in range(1, up_to + 1):
        if tn not in turn_data:
            continue
        for t in turn_data[tn]:
            values.append(_turn_performance(t))
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def print_turnpoint_analysis(all_results: dict):
    """[6] Turn-Point 성능 추이 + [7] 해석 & 권장사항."""
    turn_data = _collect_turn_data(all_results)
    models = list(turn_data.keys())
    metric = "bfcl.tool_name_acc"
    metric_arg = "bfcl.arg_value_acc"

    # 각 cutoff에서 가용 시나리오 수 계산
    all_turn_data_first = turn_data[models[0]]
    sc_counts = {}
    for c in TURN_CUTOFFS:
        cnt = len(all_turn_data_first.get(c, []))
        sc_counts[c] = cnt

    # ── [6] Turn-Point 성능 추이 ─────────────────
    print("=" * 100)
    print("  [6] TURN-POINT ANALYSIS")
    print("=" * 100)
    print()
    print("  목적: 모델이 몇 턴까지 정확한 tool을 호출하는지 '붕괴 지점' 탐색")
    print("  방법: 시나리오를 Turn N까지 잘라서 누적 정확도를 계산")
    print()

    # -- [6a] 누적 tool_name_acc --
    cw = 8  # col width per cutoff
    mw = 28  # model col width
    sep6 = "+" + "-" * mw + "+" + (("-" * cw + "+") * len(TURN_CUTOFFS))

    print("  [6a] 누적 Tool Name Accuracy (T1~TN까지 정답 tool 선택률)")
    print(f"  {sep6}")
    # header: cutoff labels
    h = f"  | {'Model':<{mw}}|"
    for c in TURN_CUTOFFS:
        h += f"{'~T' + str(c):^{cw}}|"
    print(h)
    # sub-header: scenario counts
    sh = f"  | {'':^{mw}}|"
    for c in TURN_CUTOFFS:
        sh += f"{'(' + str(sc_counts[c]) + 'sc)':^{cw}}|"
    print(sh)
    print(f"  {sep6}")

    cumul_data = {}  # {model: {cutoff: val}}
    for model in models:
        short = model.split("/")[-1][:mw - 2]
        row = f"  | {short:<{mw}}|"
        cumul_data[model] = {}
        for c in TURN_CUTOFFS:
            val, cnt = _cumulative_metric(turn_data[model], metric, c, exclude_no_call=True)
            cumul_data[model][c] = val
            if val is not None:
                row += f"{val:^{cw}.0%}|"
            else:
                row += f"{'N/A':^{cw}}|"
        print(row)

    print(f"  {sep6}")
    print()

    # -- [6b] 누적 Arg Value Accuracy (인자 기억 유지율) --
    print("  [6b] 누적 Arg Value Accuracy (T1~TN까지 인자 정확도 — 기억 유지)")
    print(f"  {sep6}")
    h = f"  | {'Model':<{mw}}|"
    for c in TURN_CUTOFFS:
        h += f"{'~T' + str(c):^{cw}}|"
    print(h)
    print(f"  {sep6}")

    for model in models:
        short = model.split("/")[-1][:mw - 2]
        row = f"  | {short:<{mw}}|"
        for c in TURN_CUTOFFS:
            val, _ = _cumulative_metric(turn_data[model], metric_arg, c, exclude_no_call=True)
            if val is not None:
                row += f"{val:^{cw}.0%}|"
            else:
                row += f"{'N/A':^{cw}}|"
        print(row)

    print(f"  {sep6}")
    print()

    # -- [6c] 개별 턴 정확도 (해당 턴 자체의 정답률 — 붕괴 시점 탐색) --
    print("  [6c] 개별 Turn 정확도 (해당 턴 자체의 tool 정답률 — 급락 지점)")
    print(f"  {sep6}")
    h = f"  | {'Model':<{mw}}|"
    for c in TURN_CUTOFFS:
        h += f"{'T' + str(c):^{cw}}|"
    print(h)
    print(f"  {sep6}")

    for model in models:
        short = model.split("/")[-1][:mw - 2]
        row = f"  | {short:<{mw}}|"
        for c in TURN_CUTOFFS:
            val, _ = _per_turn_metric(turn_data[model], metric, c, exclude_no_call=True)
            if val is not None:
                row += f"{val:^{cw}.0%}|"
            else:
                row += f"{'N/A':^{cw}}|"
        print(row)

    print(f"  {sep6}")
    print()

    # -- [6d] 구간 판정 (SAFE / GOOD / RISK / DANGER) — Performance 기준 --
    print("  [6d] 구간 판정 (누적 Performance — tool_call: (Tool+Arg+FC)/3, no_call: FC)")
    print(f"        90%+: SAFE  | 85%+: GOOD | 75%+: RISK | <75%: DANGER  (85% = critical threshold)")
    print(f"  {sep6}")
    h = f"  | {'Model':<{mw}}|"
    for c in TURN_CUTOFFS:
        h += f"{'~T' + str(c):^{cw}}|"
    print(h)
    print(f"  {sep6}")

    # Performance 기반 누적 데이터
    perf_data: dict[str, dict] = {}
    for model in models:
        perf_data[model] = {}
        for c in TURN_CUTOFFS:
            val, _ = _cumulative_performance(turn_data[model], c)
            perf_data[model][c] = val

    for model in models:
        short = model.split("/")[-1][:mw - 2]
        row = f"  | {short:<{mw}}|"
        for c in TURN_CUTOFFS:
            val = perf_data[model].get(c)
            zone = _zone_label(val)
            row += f"{zone:^{cw}}|"
        print(row)

    print(f"  {sep6}")
    print()

    # ── [7] 해석 & 권장사항 ──────────────────────
    print("=" * 100)
    print("  [7] INTERPRETATION & RECOMMENDATION")
    print("=" * 100)
    print()

    # 모델별 분석 (Performance 기준)
    model_safe_turns = {}
    for model in models:
        short = model.split("/")[-1]
        td = turn_data[model]
        max_t = max(td.keys()) if td else 0

        # Performance 기준 90%, 85%, 75% 붕괴 지점 탐색
        drop_90 = drop_85 = drop_75 = None
        for tn in range(1, max_t + 1):
            val, cnt = _cumulative_performance(td, tn)
            if val is not None:
                if drop_90 is None and val < THRESHOLD_SAFE:
                    drop_90 = tn
                if drop_85 is None and val < THRESHOLD_CRITICAL:
                    drop_85 = tn
                if drop_75 is None and val < THRESHOLD_WARNING:
                    drop_75 = tn

        # 안전 턴 = 85% 붕괴 직전 (또는 전구간 안전이면 max turn)
        safe_t = (drop_85 - 1) if drop_85 else max_t
        model_safe_turns[model] = safe_t

        print(f"  [{short}]")
        if drop_90:
            print(f"    - T{drop_90}부터 Performance 90% 미만 (하락 시작)")
        else:
            print(f"    - 전 구간 90%+ 유지 (안정)")
        if drop_85:
            print(f"    - T{drop_85}부터 Performance 85% 미만 *** 임계선 이탈 ***")
        else:
            print(f"    - 전 구간 85%+ 유지 (양호)")
        if drop_75:
            print(f"    - T{drop_75}부터 Performance 75% 미만 (심각)")
        else:
            print(f"    - 75% 미만 구간 없음")
        print(f"    >>> 권장 최대 턴: {safe_t}턴 (Performance 85%+ 유지 구간)")
        print()

    # 종합 권장
    print("  " + "-" * 60)
    print("  [종합 권장사항]")

    # 전 모델 공통 안전 턴
    common_safe = min(model_safe_turns.values()) if model_safe_turns else 0
    print(f"    1. 실서비스 권장 최대 턴: {common_safe}턴")
    print(f"       (전 모델이 85%+ 정확도를 유지하는 구간)")

    # 최적 모델
    best_model = max(model_safe_turns, key=model_safe_turns.get) if model_safe_turns else None
    if best_model:
        best_short = best_model.split("/")[-1]
        best_safe = model_safe_turns[best_model]
        print(f"    2. 가장 오래 버티는 모델: {best_short} ({best_safe}턴)")

    # Stress Test 필요성 판단
    print()
    if common_safe >= 7:
        print("    3. Stress Test 추가 불필요")
        print("       → 전 모델 7턴 이상 안정. 실서비스 5턴 기준 충분한 여유.")
    elif common_safe >= 5:
        print("    3. Stress Test 추가 권장 (선택)")
        print("       → 실서비스 5턴 기준 안정이나, 7턴 이상 시나리오에서 검증 필요")
    else:
        print("    3. Stress Test 추가 필수")
        print("       → 5턴 미만에서도 붕괴 발생. 모델 교체 또는 프롬프트 개선 필요.")

    print()
    print("  " + "-" * 60)
    print()


def save_results(
    all_results: dict,
    summary: dict,
    run_id: str,
    elapsed_sec: float,
    models_used: list[str],
    scenario_ids: list[str],
):
    """결과를 JSON으로 저장 — 메타데이터 포함."""
    out_dir = ROOT / "benchmark" / "results"
    out_dir.mkdir(exist_ok=True)

    # 공통 메타데이터
    meta = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed_sec, 1),
        "models": models_used,
        "scenarios": scenario_ids,
        "scenario_labels": {s: SCENARIO_LABELS.get(s, s) for s in scenario_ids},
        "scenario_contexts": {s: SCENARIO_CONTEXT.get(s, "") for s in scenario_ids},
        "total_turns": sum(
            len(turns)
            for scenarios in all_results.values()
            for turns in scenarios.values()
        ) // max(len(all_results), 1),
        "judge_model": JUDGE_MODEL,
    }

    # 상세 결과 (per-turn + meta)
    detail_path = out_dir / f"detail_{run_id}.json"
    detail_out = {"_meta": meta, "results": {}}
    for model, scenarios in all_results.items():
        detail_out["results"][model] = {
            sc: turns for sc, turns in scenarios.items()
        }
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail_out, f, ensure_ascii=False, indent=2)

    # 요약 결과 (+ meta)
    summary_path = out_dir / f"summary_{run_id}.json"
    summary_out = {"_meta": meta, "summary": summary}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)

    print(f"\n  Results saved:")
    print(f"    Detail : {detail_path}")
    print(f"    Summary: {summary_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AI TMR Multi-Turn Benchmark")
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 구조 검증")
    parser.add_argument("--models", nargs="+", type=int, help="모델 인덱스 (0-based)")
    parser.add_argument("--scenarios", nargs="+", help="특정 시나리오만 (예: O1_ST1 O2_ST3)")
    args = parser.parse_args()

    # 모델 선택
    selected_models = MODELS
    if args.models:
        selected_models = [MODELS[i] for i in args.models if i < len(MODELS)]

    print("=" * 60)
    print("  AI TMR Assistant Multi-Turn Benchmark")
    print("=" * 60)
    print(f"  Models   : {len(selected_models)}")
    print(f"  Dry-run  : {args.dry_run}")

    # 데이터 로드
    tools = load_tools()
    scenarios = load_scenarios()
    if args.scenarios:
        scenarios = {k: v for k, v in scenarios.items() if k in args.scenarios}

    total_turns = sum(len(v) for v in scenarios.values())
    print(f"  Tools    : {len(tools)}")
    print(f"  Scenarios: {len(scenarios)} ({total_turns} turns)")
    print(f"  Models   : {', '.join(m.split('/')[-1] for m in selected_models)}")
    print()

    # API 클라이언트
    or_client = create_openrouter_client()
    judge_client = create_judge_client()

    # 범례 출력
    print()
    print("=" * 70)
    print("  LEGEND")
    print("  Single   : [O]=정답tool 호출  [X]=오답tool 호출")
    print("  Parallel : [N/M]=GT M개 중 N개 정답  P:Y/N=복수호출 감지")
    print("  No-Call  : [O]=tool 미호출(정답)  [X]=tool 호출(오답)")
    print("             '정보부족→질문' = 필수 정보 누락 시 tool 대신 질문해야 함 [asked/total]")
    print("             '범위밖→거부'   = 업무 범위 밖 요청을 tool 없이 거절해야 함")
    print("  Args     : GT 인자 값 일치율 (BFCL AST 매칭)")
    print("  NL       : 자연어 답변 품질 (GPT-4o Judge)")
    print("               P=Pass  F=Fail  -=텍스트 응답 없음(tool만 반환)")
    print("=" * 70)
    print()

    # 실행
    all_results: dict[str, dict[str, list]] = {}
    run_start = time.time()

    for mi, model in enumerate(selected_models):
        model_short = model.split("/")[-1]
        print(f"[{mi+1}/{len(selected_models)}] {model_short}")
        print("-" * 70)

        all_results[model] = {}

        for sc_id, turns in sorted(scenarios.items()):
            ctx_short = SCENARIO_CONTEXT.get(sc_id, "")
            sc_label = SCENARIO_LABELS.get(sc_id, "")
            p_cnt = sum(1 for t in turns if t.get("call_type", "single") in ("parallel", "parallel_multiple"))
            nc_cnt = sum(1 for t in turns if t.get("call_type", "single") == "no_call")
            print(f"    +-- {sc_id} [{sc_label}] {len(turns)}turns (P:{p_cnt}, NC:{nc_cnt}) -- {ctx_short}")
            print(f"    | {'Turn':<13} | {'Tool (GT)':<38} | {'Args':>5} | {'NL':>2} |")
            print(f"    |{'-'*15}|{'-'*40}|{'-'*7}|{'-'*4}|")

            turn_results = run_scenario(
                client=or_client,
                judge_client=judge_client,
                model=model,
                scenario_id=sc_id,
                turns=turns,
                tools=tools,
                dry_run=args.dry_run,
            )
            all_results[model][sc_id] = turn_results
            # 시나리오 소계 — tool_call 턴에서만 (no_call 제외)
            tc = [t for t in turn_results if t.get("call_type", "single") != "no_call"]
            correct = sum(1 for t in tc if t["bfcl"]["tool_name_acc"] == 1.0)
            tc_n = len(tc)
            print(f"    +-- {sc_id} done: tool {correct}/{tc_n} ({correct/tc_n:.0%})" if tc_n else f"    +-- {sc_id} done")
            print()

        print()

    elapsed = time.time() - run_start
    print(f"  Total time: {elapsed:.0f}s")

    # 집계
    summary = aggregate(all_results)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print_summary(summary)
    print_turnpoint_analysis(all_results)
    save_results(
        all_results, summary, run_id,
        elapsed_sec=elapsed,
        models_used=selected_models,
        scenario_ids=list(scenarios.keys()),
    )


if __name__ == "__main__":
    main()
