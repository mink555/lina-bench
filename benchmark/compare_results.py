#!/usr/bin/env python3
"""
Phase 4: 결과 비교 & 성능 곡선 시각화

사용법:
    python -m benchmark.compare_results                      # 최신 결과 → 리포트 생성
    python -m benchmark.compare_results --run-id 20260207    # 특정 run_id 포함 결과
    python -m benchmark.compare_results --charts             # 차트도 함께 생성
    python -m benchmark.compare_results --list               # 저장된 결과 목록

산출물 (benchmark/results/ 에 저장):
    report_{run_id}.txt                  — 텍스트 리포트
    charts/ (--charts 옵션 시)           — 성능 곡선 차트 7종 (PNG)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 서버/터미널 환경
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── 프로젝트 루트 ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmark" / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

# ── 시나리오 라벨 ───────────────────────────────────────────
SCENARIO_LABELS = {
    "O1_ST1": "청약/조건누적",
    "O1_ST2": "청약/맥락희석",
    "O1_ST3": "청약/교란주입",
    "O2_ST1": "보류/조건누적",
    "O2_ST2": "보류/맥락희석",
    "O2_ST3": "보류/교란주입",
}

TURN_CUTOFFS = [3, 5, 7, 10, 13, 15, 17, 19]

# 실무 구간 cutoff — 실제 TMR 콜은 대부분 5~7턴 이내
PRODUCTION_CUTOFF = 7

# 구간 판정 임계값 (85% = 절대 임계선)
THRESHOLD_SAFE = 0.90
THRESHOLD_CRITICAL = 0.85   # ← 절대 떨어지면 안 되는 포인트
THRESHOLD_WARNING = 0.75

# ── 모던 차트 스타일 ─────────────────────────────────────────
# 팔레트: 구분 명확한 5색 (색각이상 친화적)
PALETTE = {
    "blue":    "#3B82F6",
    "red":     "#EF4444",
    "emerald": "#10B981",
    "violet":  "#8B5CF6",
    "amber":   "#F59E0B",
}
COLORS = list(PALETTE.values())
MARKERS = ["o", "D", "s", "^", "v"]

# 글로벌 matplotlib 설정
_RC = {
    "figure.facecolor":   "#FAFAFA",
    "axes.facecolor":     "#FFFFFF",
    "axes.edgecolor":     "#E5E7EB",
    "axes.grid":          True,
    "grid.color":         "#F3F4F6",
    "grid.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelsize":     11,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#E5E7EB",
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica Neue", "Arial", "DejaVu Sans"],
}
plt.rcParams.update(_RC)


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def find_latest_detail() -> Path | None:
    """results/ 에서 가장 최근 detail JSON을 찾는다."""
    files = sorted(RESULTS_DIR.glob("detail_*.json"), reverse=True)
    return files[0] if files else None


def find_detail_by_id(run_id: str) -> Path | None:
    """run_id 문자열을 포함하는 detail JSON을 찾는다."""
    for f in sorted(RESULTS_DIR.glob("detail_*.json"), reverse=True):
        if run_id in f.name:
            return f
    return None


def list_results():
    """저장된 결과 목록을 출력한다."""
    files = sorted(RESULTS_DIR.glob("detail_*.json"))
    if not files:
        print("  저장된 결과 없음. 먼저 run_benchmark.py를 실행하세요.")
        return
    print(f"\n  저장된 벤치마크 결과 ({len(files)}건):")
    print(f"  {'─' * 70}")
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        meta = data.get("_meta", {})
        models = meta.get("models", [])
        elapsed = meta.get("elapsed_seconds", 0)
        total = meta.get("total_turns", 0)
        ts = meta.get("timestamp", "?")
        short_models = ", ".join(m.split("/")[-1][:20] for m in models[:3])
        if len(models) > 3:
            short_models += f" +{len(models)-3}"
        print(f"    {f.name}")
        print(f"      시간: {ts}  |  소요: {elapsed}s  |  턴: {total}")
        print(f"      모델: {short_models}")
        print()


def load_detail(path: Path) -> tuple[dict, dict]:
    """detail JSON → (meta, results) 반환."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("_meta", {})
    results = data.get("results", data)  # 이전 포맷 호환
    if "_meta" in results:
        del results["_meta"]
    return meta, results


# ═══════════════════════════════════════════════════════════════════
# Metric Computation (from detail)
# ═══════════════════════════════════════════════════════════════════

def compute_turnpoint(
    results: dict,
    metric_path: str,
    *,
    exclude_no_call: bool = False,
) -> dict[str, dict[int, float]]:
    """
    모델별로 각 cutoff 지점의 누적 metric을 계산.
    exclude_no_call=True이면 no_call 턴을 제외 (BFCL 지표용).
    Returns: {model: {cutoff: avg_value}}
    """
    output = {}
    for model, scenarios in results.items():
        # 턴 번호별 수집
        by_turn: dict[int, list[float]] = defaultdict(list)
        for sc_id, turns in scenarios.items():
            for t in turns:
                if exclude_no_call and t.get("call_type", "single") == "no_call":
                    continue
                parts = metric_path.split(".")
                val = t
                for p in parts:
                    val = val.get(p, {}) if isinstance(val, dict) else None
                    if val is None:
                        break
                if val is not None and isinstance(val, (int, float)):
                    by_turn[t["turn"]].append(float(val))

        # 각 cutoff에서 누적 평균
        cutoff_vals = {}
        for c in TURN_CUTOFFS:
            vals = []
            for tn in range(1, c + 1):
                vals.extend(by_turn.get(tn, []))
            if vals:
                cutoff_vals[c] = sum(vals) / len(vals)
        output[model] = cutoff_vals

    return output


def compute_per_turn(
    results: dict,
    metric_path: str,
    *,
    exclude_no_call: bool = False,
) -> dict[str, dict[int, float]]:
    """모델별로 각 개별 턴의 metric 평균.
    exclude_no_call=True이면 no_call 턴 제외 (BFCL 지표용)."""
    output = {}
    for model, scenarios in results.items():
        by_turn: dict[int, list[float]] = defaultdict(list)
        for sc_id, turns in scenarios.items():
            for t in turns:
                if exclude_no_call and t.get("call_type", "single") == "no_call":
                    continue
                parts = metric_path.split(".")
                val = t
                for p in parts:
                    val = val.get(p, {}) if isinstance(val, dict) else None
                    if val is None:
                        break
                if val is not None and isinstance(val, (int, float)):
                    by_turn[t["turn"]].append(float(val))

        output[model] = {tn: sum(v) / len(v) for tn, v in by_turn.items() if v}

    return output


def compute_single_parallel(results: dict) -> dict[str, dict]:
    """모델별 Single / Parallel / No-Call 분리 집계."""
    output = {}
    for model, scenarios in results.items():
        single_tool = []
        single_arg = []
        par_tool = []
        par_arg = []
        par_detect = []
        nc_acc = []           # no_call 정답 (tool 미호출 = 1)
        nc_slot_acc = []      # slot_question 정답
        nc_rel_acc = []       # relevance_detection 정답
        nc_nl_quality = []    # no_call 턴의 NL Quality

        for sc_id, turns in scenarios.items():
            for t in turns:
                ct = t.get("call_type", "single")
                if ct == "no_call":
                    # NC:Acc는 FC Judge의 action_type_acc 사용 (BFCL 대상 아님)
                    nc_val = t["fc_judgment"]["action_type_acc"]
                    nc_acc.append(nc_val)
                    if t.get("gt_action") == "slot_question":
                        nc_slot_acc.append(nc_val)
                    elif t.get("gt_action") == "relevance_detection":
                        nc_rel_acc.append(nc_val)
                    # No-Call 턴의 NL Quality 집계
                    if t.get("fc_quality") is not None:
                        nc_nl_quality.append(1.0 if t["fc_quality"].get("pass") else 0.0)
                elif ct == "single":
                    single_tool.append(t["bfcl"]["tool_name_acc"])
                    single_arg.append(t["bfcl"]["arg_value_acc"])
                else:
                    par_tool.append(t["bfcl"]["tool_name_acc"])
                    par_arg.append(t["bfcl"]["arg_value_acc"])
                    par_detect.append(t["bfcl"].get("parallel_detected", 0))

        # Slot Question SLOT-all completeness
        slot_complete_vals = []
        for sc_id, turns in scenarios.items():
            for t in turns:
                if t.get("gt_action") == "slot_question":
                    gt_cnt = t.get("gt_missing_count", 0)
                    asked = t.get("slot_asked_count")
                    if gt_cnt > 0 and asked is not None:
                        slot_complete_vals.append(min(asked / gt_cnt, 1.0))
        slot_completeness = sum(slot_complete_vals) / len(slot_complete_vals) if slot_complete_vals else 0

        output[model] = {
            "single_tool": sum(single_tool) / len(single_tool) if single_tool else 0,
            "single_arg": sum(single_arg) / len(single_arg) if single_arg else 0,
            "single_n": len(single_tool),
            "parallel_tool": sum(par_tool) / len(par_tool) if par_tool else 0,
            "parallel_arg": sum(par_arg) / len(par_arg) if par_arg else 0,
            "parallel_detect": sum(par_detect) / len(par_detect) if par_detect else 0,
            "parallel_n": len(par_tool),
            "nc_acc": sum(nc_acc) / len(nc_acc) if nc_acc else 0,
            "nc_slot_acc": sum(nc_slot_acc) / len(nc_slot_acc) if nc_slot_acc else 0,
            "nc_rel_acc": sum(nc_rel_acc) / len(nc_rel_acc) if nc_rel_acc else 0,
            "nc_slot_completeness": slot_completeness,
            "nc_nl_quality": sum(nc_nl_quality) / len(nc_nl_quality) if nc_nl_quality else None,
            "nc_n": len(nc_acc),
        }

    return output


def _turn_performance(t: dict) -> float:
    """턴 하나의 Performance 종합 점수.

    tool_call 턴: (Tool + Arg + FC) / 3
    no_call 턴:   FC Judge만 (BFCL은 '호출해야 할 턴'에서만 측정)
    """
    fcj_vals = list(t["fc_judgment"].values())
    fc = sum(fcj_vals) / len(fcj_vals) if fcj_vals else 0
    if t.get("call_type", "single") == "no_call":
        return fc
    tool = t["bfcl"]["tool_name_acc"]
    arg = t["bfcl"]["arg_value_acc"]
    return (tool + arg + fc) / 3


def compute_turnpoint_performance(
    results: dict,
) -> dict[str, dict[int, float]]:
    """모델별 각 cutoff 지점의 누적 Performance 종합 점수."""
    output = {}
    for model, scenarios in results.items():
        by_turn: dict[int, list[float]] = defaultdict(list)
        for sc_id, turns in scenarios.items():
            for t in turns:
                by_turn[t["turn"]].append(_turn_performance(t))

        cutoff_vals = {}
        for c in TURN_CUTOFFS:
            vals = []
            for tn in range(1, c + 1):
                vals.extend(by_turn.get(tn, []))
            if vals:
                cutoff_vals[c] = sum(vals) / len(vals)
        output[model] = cutoff_vals

    return output


def compute_turnpoint_performance_by_stress(
    results: dict,
) -> dict[str, dict[str, dict[int, float]]]:
    """모델별 Stress Type(ST1/ST2/ST3)별 각 cutoff 지점의 누적 Performance.

    Returns: {model: {"ST1": {cutoff: avg}, "ST2": ..., "ST3": ...}}
    """
    output = {}
    for model, scenarios in results.items():
        by_st_turn: dict[str, dict[int, list[float]]] = {
            st: defaultdict(list) for st in ("ST1", "ST2", "ST3")
        }
        for sc_id, turns in scenarios.items():
            st = sc_id.split("_")[1]  # "ST1", "ST2", "ST3"
            for t in turns:
                by_st_turn[st][t["turn"]].append(_turn_performance(t))

        model_result = {}
        for st in ("ST1", "ST2", "ST3"):
            cutoff_vals = {}
            for c in TURN_CUTOFFS:
                vals = []
                for tn in range(1, c + 1):
                    vals.extend(by_st_turn[st].get(tn, []))
                if vals:
                    cutoff_vals[c] = sum(vals) / len(vals)
            model_result[st] = cutoff_vals
        output[model] = model_result

    return output


def compute_turnpoint_fc(results: dict) -> dict[str, dict[int, float]]:
    """모델별 각 cutoff 지점의 누적 FC Judgment 평균."""
    output = {}
    for model, scenarios in results.items():
        by_turn: dict[int, list[float]] = defaultdict(list)
        for sc_id, turns in scenarios.items():
            for t in turns:
                fcj_vals = list(t["fc_judgment"].values())
                if fcj_vals:
                    by_turn[t["turn"]].append(sum(fcj_vals) / len(fcj_vals))

        cutoff_vals = {}
        for c in TURN_CUTOFFS:
            vals = []
            for tn in range(1, c + 1):
                vals.extend(by_turn.get(tn, []))
            if vals:
                cutoff_vals[c] = sum(vals) / len(vals)
        output[model] = cutoff_vals

    return output


def compute_overall(results: dict) -> dict[str, dict]:
    """모델별 종합 점수 (Section 1용).

    BFCL (Tool Acc, Arg Acc): tool_call 턴(single + parallel)에서만 산출
    FC Judge: 전체 턴(tool_call + no_call)에서 산출
    NL Quality: 전체 턴에서 텍스트가 있는 경우 LLM-as-Judge 평가 (pass rate)
    Performance: 턴별 _turn_performance()의 평균 (turn-point와 동일 방식)
    """
    output = {}
    for model, scenarios in results.items():
        all_turns = [t for sc in scenarios.values() for t in sc]
        # BFCL: tool을 호출해야 하는 턴에서만
        tc_turns = [t for t in all_turns if t.get("call_type", "single") != "no_call"]
        tool = sum(t["bfcl"]["tool_name_acc"] for t in tc_turns) / len(tc_turns) if tc_turns else 0
        arg = sum(t["bfcl"]["arg_value_acc"] for t in tc_turns) / len(tc_turns) if tc_turns else 0
        # FC Judge: 전체 턴 (no_call 포함)
        fcj_all = [v for t in all_turns for v in t["fc_judgment"].values()]
        fc = sum(fcj_all) / len(fcj_all) if fcj_all else 0
        # NL Quality: 텍스트가 있는 턴에서만 (fc_quality가 None이 아닌 경우)
        nl_evals = [t["fc_quality"] for t in all_turns if t.get("fc_quality") is not None]
        nl_pass = sum(1 for q in nl_evals if q.get("pass")) if nl_evals else 0
        nl_rate = nl_pass / len(nl_evals) if nl_evals else None
        # Performance: per-turn 방식 (turn-point 계산과 동일)
        # tool_call 턴: (Tool + Arg + FC) / 3
        # no_call 턴:   FC Judge만
        perf_vals = [_turn_performance(t) for t in all_turns]
        perf = sum(perf_vals) / len(perf_vals) if perf_vals else 0
        output[model] = {
            "tool": tool, "arg": arg, "fc": fc, "nl_quality": nl_rate, "performance": perf,
            "tool_call_turns": len(tc_turns), "total_turns": len(all_turns),
        }
    return output


def compute_scenario_matrix(results: dict) -> dict[str, dict[str, float]]:
    """모델별 × 시나리오별 tool_name_acc (tool_call 턴에서만)."""
    output = {}
    for model, scenarios in results.items():
        sc_acc = {}
        for sc_id, turns in scenarios.items():
            tc_turns = [t for t in turns if t.get("call_type", "single") != "no_call"]
            vals = [t["bfcl"]["tool_name_acc"] for t in tc_turns]
            sc_acc[sc_id] = sum(vals) / len(vals) if vals else 0
        output[model] = sc_acc
    return output


def compute_stress_cross_analysis(results: dict) -> dict[str, dict]:
    """Stress Type × Outcome 교차분석 (Tool Acc + Performance)."""
    output = {}
    for model, scenarios in results.items():
        by_st_tool: dict[str, list[float]] = defaultdict(list)
        by_st_perf: dict[str, list[float]] = defaultdict(list)
        by_outcome_tool: dict[str, list[float]] = defaultdict(list)
        by_outcome_perf: dict[str, list[float]] = defaultdict(list)

        for sc_id, turns in scenarios.items():
            tc_turns = [t for t in turns if t.get("call_type", "single") != "no_call"]
            tool_vals = [t["bfcl"]["tool_name_acc"] for t in tc_turns]
            perf_vals = [_turn_performance(t) for t in turns]
            tool_avg = sum(tool_vals) / len(tool_vals) if tool_vals else 0
            perf_avg = sum(perf_vals) / len(perf_vals) if perf_vals else 0

            parts = sc_id.split("_")
            outcome = parts[0]  # "O1" or "O2"
            st = parts[1]       # "ST1", "ST2", "ST3"

            by_st_tool[st].append(tool_avg)
            by_st_perf[st].append(perf_avg)
            by_outcome_tool[outcome].append(tool_avg)
            by_outcome_perf[outcome].append(perf_avg)

        output[model] = {
            "st_tool": {k: sum(v) / len(v) for k, v in sorted(by_st_tool.items())},
            "st_perf": {k: sum(v) / len(v) for k, v in sorted(by_st_perf.items())},
            "outcome_tool": {k: sum(v) / len(v) for k, v in sorted(by_outcome_tool.items())},
            "outcome_perf": {k: sum(v) / len(v) for k, v in sorted(by_outcome_perf.items())},
        }
    return output


def find_threshold_turn(
    results: dict,
    metric_path: str,
    threshold: float,
) -> dict[str, int | None]:
    """모델별로 누적 metric이 threshold 아래로 떨어지는 첫 턴."""
    output = {}
    for model, scenarios in results.items():
        by_turn: dict[int, list[float]] = defaultdict(list)
        for sc_id, turns in scenarios.items():
            for t in turns:
                parts = metric_path.split(".")
                val = t
                for p in parts:
                    val = val.get(p, {}) if isinstance(val, dict) else None
                    if val is None:
                        break
                if val is not None and isinstance(val, (int, float)):
                    by_turn[t["turn"]].append(float(val))

        max_t = max(by_turn.keys()) if by_turn else 0
        found = None
        for tn in range(1, max_t + 1):
            vals = []
            for n in range(1, tn + 1):
                vals.extend(by_turn.get(n, []))
            if vals and (sum(vals) / len(vals)) < threshold:
                found = tn
                break
        output[model] = found

    return output


# ═══════════════════════════════════════════════════════════════════
# Chart Generation
# ═══════════════════════════════════════════════════════════════════

def _short(model: str) -> str:
    return model.split("/")[-1][:25]


def chart_turnpoint_curve(
    cumul: dict[str, dict[int, float]],
    title: str,
    ylabel: str,
    save_path: Path,
):
    """Turn-point 누적 성능 곡선 — 모던 디자인."""
    fig, ax = plt.subplots(figsize=(13, 6.5))

    # 구간 배경 (파스텔)
    ax.axvspan(2, 5.5,   alpha=0.06, color="#10B981", zorder=0)
    ax.axvspan(5.5, 10.5, alpha=0.06, color="#F59E0B", zorder=0)
    ax.axvspan(10.5, 20,  alpha=0.06, color="#EF4444", zorder=0)

    # 구간 라벨
    ax.text(3.75,  1.02, "Production", ha="center", fontsize=8,
            color="#059669", fontweight="bold", alpha=0.7)
    ax.text(8.0,   1.02, "Stress", ha="center", fontsize=8,
            color="#D97706", fontweight="bold", alpha=0.7)
    ax.text(15.25, 1.02, "Extreme", ha="center", fontsize=8,
            color="#DC2626", fontweight="bold", alpha=0.7)

    # 임계선
    for y, label, color in [
        (0.90, "SAFE  90%",       "#059669"),
        (0.85, "CRITICAL  85%",   "#DC2626"),
        (0.75, "WARNING  75%",    "#9CA3AF"),
    ]:
        ax.axhline(y=y, color=color, linestyle="--", linewidth=1, alpha=0.45, zorder=1)
        ax.text(max(TURN_CUTOFFS) + 0.5, y, label, va="center",
                fontsize=7.5, color=color, alpha=0.8, fontweight="bold")

    # 모델 곡선
    for i, (model, vals) in enumerate(cumul.items()):
        xs = sorted(vals.keys())
        ys = [vals[x] for x in xs]
        c = COLORS[i % len(COLORS)]
        ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)],
                color=c, linewidth=2.5, markersize=9,
                markeredgecolor="white", markeredgewidth=1.5,
                label=_short(model), zorder=3)
        # 끝점 값 표시
        if ys:
            ax.annotate(f"{ys[-1]:.0%}", (xs[-1], ys[-1]),
                        textcoords="offset points", xytext=(8, 0),
                        fontsize=8, color=c, fontweight="bold")

    ax.set_xlabel("Turn Cutoff")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=18)
    ax.set_xticks(TURN_CUTOFFS)
    ax.set_xlim(2, max(TURN_CUTOFFS) + 2.5)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="lower left", frameon=True, borderpad=0.8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    chart: {save_path.name}")


def chart_stress_turnpoint(
    cumul_by_stress: dict[str, dict[str, dict[int, float]]],
    save_path: Path,
):
    """Stress Type(ST1/ST2/ST3)별 Performance 곡선 — 3-subplot."""
    st_labels = {"ST1": "ST1 — State Accumulation",
                 "ST2": "ST2 — Context Drift",
                 "ST3": "ST3 — Distraction Injection"}
    models = list(cumul_by_stress.keys())
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=True)

    for idx, st in enumerate(("ST1", "ST2", "ST3")):
        ax = axes[idx]

        # 구간 배경
        ax.axvspan(2, 5.5,   alpha=0.06, color="#10B981", zorder=0)
        ax.axvspan(5.5, 10.5, alpha=0.06, color="#F59E0B", zorder=0)
        ax.axvspan(10.5, 20,  alpha=0.06, color="#EF4444", zorder=0)

        # 임계선
        for y, color in [(0.90, "#059669"), (0.85, "#DC2626"), (0.75, "#9CA3AF")]:
            ax.axhline(y=y, color=color, linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)

        # 모델 곡선
        for i, model in enumerate(models):
            vals = cumul_by_stress[model].get(st, {})
            xs = sorted(vals.keys())
            ys = [vals[x] for x in xs]
            if not xs:
                continue
            c = COLORS[i % len(COLORS)]
            ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)],
                    color=c, linewidth=2.2, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.2,
                    label=_short(model), zorder=3)
            if ys:
                ax.annotate(f"{ys[-1]:.0%}", (xs[-1], ys[-1]),
                            textcoords="offset points", xytext=(6, 0),
                            fontsize=7, color=c, fontweight="bold")

        ax.set_title(st_labels[st], fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Turn Cutoff")
        ax.set_xticks(TURN_CUTOFFS)
        ax.set_xlim(2, max(TURN_CUTOFFS) + 2.5)
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        if idx == 0:
            ax.set_ylabel("Cumulative Performance")
            ax.legend(loc="lower left", frameon=True, borderpad=0.6, fontsize=8)

    fig.suptitle("Performance by Stress Type", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    chart: {save_path.name}")


def chart_per_turn(
    per_turn: dict[str, dict[int, float]],
    title: str,
    ylabel: str,
    save_path: Path,
):
    """개별 턴 정확도 곡선 — 급락 지점 탐지."""
    fig, ax = plt.subplots(figsize=(13, 6.5))

    # 위험 구간
    ax.axhspan(0, 0.7, alpha=0.04, color="#EF4444", zorder=0)

    ax.axhline(y=0.9, color="#059669", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.axhline(y=0.7, color="#DC2626", linestyle="--", linewidth=0.8, alpha=0.4)

    for i, (model, vals) in enumerate(per_turn.items()):
        xs = sorted(vals.keys())
        ys = [vals[x] for x in xs]
        c = COLORS[i % len(COLORS)]
        # 면 채우기 (연한)
        ax.fill_between(xs, ys, alpha=0.06, color=c, zorder=1)
        ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)],
                color=c, linewidth=2, markersize=7,
                markeredgecolor="white", markeredgewidth=1.2,
                label=_short(model), zorder=3, alpha=0.9)

    ax.set_xlabel("Turn")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.set_ylim(-0.05, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="lower left", frameon=True, borderpad=0.8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    chart: {save_path.name}")


def _bar_label(ax, bars, fmt="{:.0%}", offset=3, fontsize=8, color="#374151"):
    """바 위에 값 표시 (공통 헬퍼)."""
    for bar in bars:
        h = bar.get_height()
        if h > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                    fmt.format(h), ha="center", va="bottom",
                    fontsize=fontsize, color=color, fontweight="medium")


def chart_single_vs_parallel(
    sp: dict[str, dict],
    save_path: Path,
):
    """Single vs Parallel — 깔끔한 그룹 바 차트."""
    models = list(sp.keys())
    short_names = [_short(m) for m in models]
    n = len(models)
    x = np.arange(n)
    w = 0.32

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5),
                              gridspec_kw={"wspace": 0.30})

    c_s = PALETTE["blue"]
    c_p = PALETTE["amber"]
    c_d = PALETTE["emerald"]

    # --- (a) Tool Name Acc ---
    ax = axes[0]
    s_vals = [sp[m]["single_tool"] for m in models]
    p_vals = [sp[m]["parallel_tool"] for m in models]
    b1 = ax.bar(x - w / 2, s_vals, w, label="Single", color=c_s,
                edgecolor="white", linewidth=0.8, zorder=3)
    b2 = ax.bar(x + w / 2, p_vals, w, label="Parallel", color=c_p,
                edgecolor="white", linewidth=0.8, zorder=3)
    _bar_label(ax, b1, fontsize=7.5)
    _bar_label(ax, b2, fontsize=7.5)
    ax.set_ylabel("Tool Name Acc")
    ax.set_title("(a) Tool Name", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="upper right", fontsize=8)

    # --- (b) Arg Value Acc ---
    ax = axes[1]
    s_vals = [sp[m]["single_arg"] for m in models]
    p_vals = [sp[m]["parallel_arg"] for m in models]
    b1 = ax.bar(x - w / 2, s_vals, w, label="Single", color=c_s,
                edgecolor="white", linewidth=0.8, zorder=3)
    b2 = ax.bar(x + w / 2, p_vals, w, label="Parallel", color=c_p,
                edgecolor="white", linewidth=0.8, zorder=3)
    _bar_label(ax, b1, fontsize=7.5)
    _bar_label(ax, b2, fontsize=7.5)
    ax.set_ylabel("Arg Value Acc")
    ax.set_title("(b) Arg Value", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="upper right", fontsize=8)

    # --- (c) Parallel Detection ---
    ax = axes[2]
    d_vals = [sp[m]["parallel_detect"] for m in models]
    bars = ax.bar(x, d_vals, 0.48, color=c_d,
                  edgecolor="white", linewidth=0.8, zorder=3)
    _bar_label(ax, bars, fontsize=8.5, color="#065F46")
    ax.set_ylabel("Detection Rate")
    ax.set_title("(c) Parallel Detection", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.suptitle("Single vs Parallel Performance Comparison",
                 fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.88, wspace=0.30)
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    chart: {save_path.name}")


def chart_scenario_heatmap(
    matrix: dict[str, dict[str, float]],
    save_path: Path,
):
    """시나리오별 성능 히트맵 — 커스텀 컬러맵."""
    models = list(matrix.keys())
    scenarios = sorted(
        set(sc for m in models for sc in matrix[m]),
        key=lambda x: (x[:2], x[3:]),
    )
    short_names = [_short(m) for m in models]

    CHART_LABELS = {
        "O1_ST1": "Sub / Accum",
        "O1_ST2": "Sub / Drift",
        "O1_ST3": "Sub / Distract",
        "O2_ST1": "Hold / Accum",
        "O2_ST2": "Hold / Drift",
        "O2_ST3": "Hold / Distract",
    }
    sc_labels = [f"{s}\n{CHART_LABELS.get(s, '')}" for s in scenarios]

    data = np.array([
        [matrix[m].get(s, 0) for s in scenarios]
        for m in models
    ])

    # 커스텀 컬러맵: 빨강 → 노랑 → 초록 (더 부드러운 톤)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "perf",
        ["#FCA5A5", "#FDE68A", "#6EE7B7", "#059669"],
        N=256,
    )

    fig, ax = plt.subplots(
        figsize=(max(10, len(scenarios) * 1.6), max(4.5, len(models) * 1.0 + 1)),
    )

    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # 격자선
    for i in range(len(models) + 1):
        ax.axhline(y=i - 0.5, color="white", linewidth=2)
    for j in range(len(scenarios) + 1):
        ax.axvline(x=j - 0.5, color="white", linewidth=2)

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(sc_labels, fontsize=9.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(short_names, fontsize=10)

    # 셀 값 + 테두리 효과
    for i in range(len(models)):
        for j in range(len(scenarios)):
            val = data[i, j]
            txt_color = "#1F2937" if val > 0.45 else "#FAFAFA"
            weight = "bold" if val >= 0.9 or val <= 0.1 else "medium"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    color=txt_color, fontsize=12, fontweight=weight)

    ax.set_title("Tool Name Accuracy by Scenario", pad=14)
    cb = fig.colorbar(im, ax=ax, format=mticker.PercentFormatter(1.0),
                      shrink=0.8, pad=0.02)
    cb.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    chart: {save_path.name}")


# ═══════════════════════════════════════════════════════════════════
# Text Report Generation
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# Error Taxonomy
# ═══════════════════════════════════════════════════════════════════

# 6개 에러 태그 정의
ERROR_TAGS = {
    "WRONG_TOOL":  "호출해야 하는데 다른 tool 호출",
    "MISSED_CALL": "호출해야 하는데 호출 안 함",
    "FALSE_CALL":  "호출하면 안 되는데 호출",
    "ARG_MISSING": "tool 맞지만 필수 인자 누락",
    "ARG_WRONG":   "tool 맞지만 인자 값 틀림",
    "ARG_STALE":   "번복값 미갱신 (ST3 추정)",
}


def compute_error_taxonomy(results: dict) -> dict[str, dict[str, int]]:
    """모델별 에러 유형 분류.

    Returns: {model: {tag: count, ..., "_total": N, "_correct": N}}
    """
    output = {}
    for model, scenarios in results.items():
        counts = {tag: 0 for tag in ERROR_TAGS}
        total = 0
        correct = 0

        for sc_id, turns in scenarios.items():
            is_st3 = "ST3" in sc_id
            for t in turns:
                total += 1
                ct = t.get("call_type", "single")
                has_calls = bool(t.get("model_tools")) or (
                    t["bfcl"]["tool_name_acc"] > 0 or
                    t["fc_judgment"]["action_type_acc"] == 1.0
                )

                # model_tools가 없는 경우 fc_judgment로 추론
                if "model_tools" in t:
                    has_calls = bool(t["model_tools"])
                else:
                    # tool_call 턴: action_type_acc=1 → 호출함
                    # no_call 턴: action_type_acc=0 → 호출함 (오답)
                    if ct == "no_call":
                        has_calls = t["fc_judgment"]["action_type_acc"] == 0.0
                    else:
                        has_calls = t["fc_judgment"]["action_type_acc"] == 1.0

                if ct == "no_call":
                    # 미호출이 정답
                    if has_calls:
                        counts["FALSE_CALL"] += 1
                    else:
                        correct += 1
                else:
                    # 콜이 정답
                    tool_ok = t["bfcl"]["tool_name_acc"] == 1.0
                    arg_key_ok = t["bfcl"]["arg_key_acc"] == 1.0
                    arg_val_ok = t["bfcl"]["arg_value_acc"] == 1.0

                    if not has_calls:
                        counts["MISSED_CALL"] += 1
                    elif not tool_ok:
                        counts["WRONG_TOOL"] += 1
                    elif not arg_key_ok:
                        counts["ARG_MISSING"] += 1
                    elif not arg_val_ok:
                        if is_st3:
                            counts["ARG_STALE"] += 1
                        else:
                            counts["ARG_WRONG"] += 1
                    else:
                        correct += 1

        counts["_total"] = total
        counts["_correct"] = correct
        output[model] = counts
    return output


# ═══════════════════════════════════════════════════════════════════
# Git / Config Helpers
# ═══════════════════════════════════════════════════════════════════

def _get_git_rev() -> str:
    """현재 git commit hash (short). 실패 시 'unknown'."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _format_config(meta: dict) -> list[str]:
    """메타데이터에서 config 정보를 사람이 읽기 좋은 형태로."""
    cfg = meta.get("config", {})
    gen = cfg.get("generation", {})
    jdg = cfg.get("judge", {})
    lines = []
    lines.append(f"  seed={gen.get('seed', '?')}  "
                 f"temp={gen.get('temperature', '?')}  "
                 f"tool_choice={gen.get('tool_choice', '?')}")
    lines.append(f"  judge: seed={jdg.get('seed', '?')}  "
                 f"temp={jdg.get('temperature', '?')}  "
                 f"max_tokens={jdg.get('max_tokens', '?')}")
    return lines


def _zone(val: float) -> str:
    if val >= THRESHOLD_SAFE:
        return "SAFE"        # 90%+ : 안정
    if val >= THRESHOLD_CRITICAL:
        return "GOOD"        # 85~90%: 양호
    if val >= THRESHOLD_WARNING:
        return "RISK"        # 75~85%: 85% 미만 — 위험
    return "DANGER"          # 75% 미만: 사용 불가


def _zone_dot(val: float) -> str:
    """숫자% + 색상 원형 이모지.
    표시 값(반올림)과 이모지가 불일치하지 않도록 반올림 후 판정."""
    pct = round(val * 100)
    if pct >= 90:
        return f"{val:>5.0%} 🟢"
    if pct >= 85:
        return f"{val:>5.0%} 🔵"
    if pct >= 75:
        return f"{val:>5.0%} 🟡"
    return f"{val:>5.0%} 🔴"


def _turnpoint_table(w, label: str, cumul: dict, models: list, as_zone=False):
    """Turn-point 테이블 헬퍼 (값 또는 구간라벨)."""
    col_w = 9 if as_zone else 7
    hdr = f"    {'모델':<28}"
    for c in TURN_CUTOFFS:
        hdr += f" {'~T' + str(c):>{col_w}}"
    w(hdr)
    sep_unit = "─" * (col_w + 1)
    w(f"    {'─' * 28}" + sep_unit * len(TURN_CUTOFFS))
    for model in models:
        row = f"    {_short(model):<28}"
        for c in TURN_CUTOFFS:
            v = cumul[model].get(c)
            if v is None:
                row += f" {'  -  ':>{col_w}}"
            elif as_zone:
                row += f" {_zone_dot(v)}"
            else:
                row += f" {v:>6.0%} "
        w(row)


def generate_report(
    meta: dict,
    results: dict,
    save_path: Path,
):
    """텍스트 리포트."""
    lines: list[str] = []
    w = lines.append

    models = list(results.keys())
    overall = compute_overall(results)
    sp = compute_single_parallel(results)
    cumul_tool = compute_turnpoint(results, "bfcl.tool_name_acc", exclude_no_call=True)
    cumul_arg = compute_turnpoint(results, "bfcl.arg_value_acc", exclude_no_call=True)
    cumul_fc = compute_turnpoint_fc(results)
    cumul_perf = compute_turnpoint_performance(results)
    cumul_perf_st = compute_turnpoint_performance_by_stress(results)
    cross = compute_stress_cross_analysis(results)

    # 실무 구간 Performance (PRODUCTION_CUTOFF 기준)
    prod_perf: dict[str, float] = {}
    for model in models:
        prod_perf[model] = cumul_perf.get(model, {}).get(PRODUCTION_CUTOFF, 0)

    # Pre-compute safe turns (85% 임계선 기반)
    safe_turns: dict[str, int] = {}
    for model in models:
        _vals = cumul_perf[model]
        _d85 = None
        for _c in TURN_CUTOFFS:
            _v = _vals.get(_c)
            if _v is not None and _d85 is None and _v < THRESHOLD_CRITICAL:
                _d85 = _c
        if _d85:
            _idx = TURN_CUTOFFS.index(_d85)
            safe_turns[model] = TURN_CUTOFFS[_idx - 1] if _idx > 0 else 0
        else:
            safe_turns[model] = max(TURN_CUTOFFS)

    best_perf_raw = max(models, key=lambda m: overall[m]["performance"])
    best_par_model = max(models, key=lambda m: sp[m]["parallel_tool"])
    # NL Quality 1위 (N/A 제외)
    nl_candidates = [m for m in models if overall[m].get("nl_quality") is not None
                     and overall[m]["performance"] >= 0.30]
    best_nl_model = max(nl_candidates, key=lambda m: overall[m]["nl_quality"]) if nl_candidates else None

    # Agent 1위: Perf가 3%p 이내면 safe_turns → NL 순으로 종합 판단
    # (단순 Perf 평균보다 "85%+를 얼마나 유지하느냐"가 실무에서 더 중요)
    top_perf = overall[best_perf_raw]["performance"]
    contenders = [m for m in models
                  if overall[m]["performance"] >= top_perf - 0.03
                  and overall[m]["performance"] >= 0.30]
    best_perf_model = max(
        contenders,
        key=lambda m: (
            safe_turns.get(m, 0),           # 1st: 85%+ 유지 턴 수
            overall[m]["performance"],       # 2nd: Perf 평균
            overall[m].get("nl_quality") or 0,  # 3rd: NL Quality
        ),
    )

    # ── Pre-compute helper data ──
    common_safe_t = min(safe_turns.values()) if safe_turns else 0
    sorted_perf = sorted(models, key=lambda m: overall[m]["performance"], reverse=True)
    unusable = [m for m in models if overall[m]["performance"] < 0.30]
    usable = [m for m in sorted_perf if overall[m]["performance"] >= 0.30]
    best_nc_model = max(models, key=lambda m: sp[m]["nc_acc"])

    # Cross-analysis helpers
    valid_models = [m for m in models if len(cross[m]["st_tool"]) >= 3]
    st1_worst_cnt = sum(
        1 for m in valid_models
        if cross[m]["st_perf"].get("ST1", 1) <= min(cross[m]["st_perf"].values())
    )
    outcome_diffs = [
        abs(cross[m]["outcome_tool"].get("O1", 0) - cross[m]["outcome_tool"].get("O2", 0))
        for m in models
    ]
    avg_outcome_diff = sum(outcome_diffs) / len(outcome_diffs) if outcome_diffs else 0

    # ST 순서 (average performance across usable models)
    st_avg = {}
    for st in ["ST1", "ST2", "ST3"]:
        vals_st = [cross[m]["st_perf"].get(st, 0) for m in usable]
        st_avg[st] = sum(vals_st) / len(vals_st) if vals_st else 0
    st_order = sorted(st_avg, key=st_avg.get, reverse=True)
    st_names = {"ST1": "조건누적", "ST2": "맥락희석", "ST3": "교란주입"}

    # ── Error Taxonomy 사전 계산 ──
    err_tax = compute_error_taxonomy(results)

    # ════════════════════════════════════════════════════════════════
    # HEADER
    # ════════════════════════════════════════════════════════════════
    w("=" * 78)
    w("  AI TMR Assistant — 모델 성능 비교 리포트")
    w("=" * 78)
    w(f"  생성: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
      f"턴: {meta.get('total_turns', '?')} × {len(models)}모델 | "
      f"Judge: {meta.get('judge_model', 'N/A')}")
    git_rev = _get_git_rev()
    w(f"  commit: {git_rev} | run_id: {meta.get('run_id', '?')}")
    for cfg_line in _format_config(meta):
        w(f"  config: {cfg_line}")
    w("")
    w(f"  용어 정의:")
    w(f"    @T7 (실무 구간) = Turn 1~7까지의 누적 성능. TMR 영업콜의")
    w(f"    실무 턴 수(청약 ~7턴, 보류 ~5턴)에 대응하는 운영 기준선.")
    w(f"    T7 이후(T10~T19)는 스트레스 테스트 구간으로 내구도 진단용.")
    w("")

    # ════════════════════════════════════════════════════════════════
    # ★ 요약
    # ════════════════════════════════════════════════════════════════
    best_m = _short(best_perf_model)
    best_p = overall[best_perf_model]["performance"]
    best_safe = safe_turns[best_perf_model]
    best_safe_str = f"~T{best_safe}" if best_safe > 0 else "T3 미만"
    worst_st = st_order[-1] if st_order else "ST1"

    # Agent/NL 모델 권장 판단
    best_nl_short = _short(best_nl_model) if best_nl_model else "N/A"
    best_nl_rate = overall[best_nl_model]["nl_quality"] if best_nl_model else 0
    same_model = best_perf_model == best_nl_model
    if same_model:
        model_strategy = f"1모델 권장: {best_m} (Agent+답변 겸용)"
    elif best_nl_model:
        model_strategy = f"Agent: {best_m} | 답변: {best_nl_short} (2모델 분리 고려)"
    else:
        model_strategy = f"Agent: {best_m} (NL 데이터 부족)"

    best_prod = prod_perf[best_perf_model]

    # 1위 모델의 실무 구간 세부 지표 계산 (병목 분석용)
    _bp_tc_tool, _bp_tc_arg, _bp_tc_fc, _bp_nc_fc = [], [], [], []
    for _sc_id, _turns in results[best_perf_model].items():
        for _t in _turns:
            if _t["turn"] > PRODUCTION_CUTOFF:
                continue
            _fcj = _t["fc_judgment"]
            _fc_avg = sum(_fcj.values()) / len(_fcj.values()) if _fcj else 0
            if _t.get("call_type", "single") == "no_call":
                _bp_nc_fc.append(_fc_avg)
            else:
                _bp_tc_tool.append(_t["bfcl"]["tool_name_acc"])
                _bp_tc_arg.append(_t["bfcl"]["arg_value_acc"])
                _bp_tc_fc.append(_fc_avg)
    _bp_prod_tool = sum(_bp_tc_tool) / len(_bp_tc_tool) if _bp_tc_tool else 0
    _bp_prod_arg = sum(_bp_tc_arg) / len(_bp_tc_arg) if _bp_tc_arg else 0
    _bp_prod_fc = sum(_bp_tc_fc) / len(_bp_tc_fc) if _bp_tc_fc else 0
    _bp_prod_nc = sum(_bp_nc_fc) / len(_bp_nc_fc) if _bp_nc_fc else 0
    _bp_tc_n = len(_bp_tc_tool)
    _bp_nc_n = len(_bp_nc_fc)
    _bp_total_n = _bp_tc_n + _bp_nc_n

    # 병목 식별
    _bottlenecks = []
    if _bp_prod_arg < 0.85:
        _bottlenecks.append(("Arg Acc", _bp_prod_arg))
    if _bp_prod_nc < 0.50:
        _bottlenecks.append(("No-Call", _bp_prod_nc))
    _bottleneck_str = " / ".join(f"{n} {v:.0%}" for n, v in _bottlenecks)

    w("=" * 78)
    w(f"  실무 기준({PRODUCTION_CUTOFF}턴) 1위: {best_m}  Performance {best_prod:.0%}")
    w(f"  병목: {_bottleneck_str}  |  NL 1위: {best_nl_short} ({best_nl_rate:.0%})")
    w(f"  → {model_strategy}")
    w("=" * 78)
    w(f"  ※ 본 벤치마크는 최대 19턴 스트레스 테스트를 포함합니다.")
    w(f"    실무 TMR 콜은 보통 5~7턴이므로, @T{PRODUCTION_CUTOFF} 누적을 실무 성능으로 봅니다.")
    w(f"    T{PRODUCTION_CUTOFF} 이후는 내구도 진단용이며, 운영 목표 수치가 아닙니다.")
    w("")

    # ════════════════════════════════════════════════════════════════
    # 1. 핵심 성적표
    # ════════════════════════════════════════════════════════════════
    w("=" * 78)
    w("  1. 모델별 성적표")
    w("=" * 78)
    w("")
    total_t = meta.get("total_turns", 106)
    tc_example = list(overall.values())[0].get("tool_call_turns", 94) if overall else 94
    w(f"  Tool Acc = tool 호출 정답률 ({tc_example}턴)")
    w(f"  Arg Acc  = 인자 정확도 (tool name 정답일 때만)")
    w(f"  FC Judge = 행동 판단 정확도 (전체 {total_t}턴)")
    w(f"  NL Qual  = 자연어 답변 품질 (LLM Judge, 텍스트 있는 턴만)")
    w(f"  Perf     = 종합 (tool턴: (Tool+Arg+FC)/3, no-call턴: FC)")
    w(f"  ※ 실무 = ~T{PRODUCTION_CUTOFF} 누적 | 전체 = ~T{max(TURN_CUTOFFS)} 누적 (스트레스 포함)")
    w("")
    w(f"  {'모델':<28} {'Tool':>7} {'Arg':>7} {'FC':>7} {'NL':>7}"
      f" {'│ 실무':>7} {'전체':>6} {'Gap':>6}")
    w(f"  {'─' * 28} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 7}"
      f" {'─' * 7} {'─' * 6} {'─' * 6}")
    for model in models:
        o = overall[model]
        pp = prod_perf[model]
        fp = o["performance"]
        gap = pp - fp
        marks = []
        if model in unusable:
            marks.append("✗")
        else:
            if model == best_perf_model:
                marks.append("Agent1위")
            if model == best_nl_model:
                marks.append("NL1위")
        mark_str = f" ◀ {','.join(marks)}" if marks else ""
        nl_str = f"{o['nl_quality']:>6.0%}" if o["nl_quality"] is not None else "  N/A "
        w(f"  {_short(model):<28}"
          f" {o['tool']:>6.1%}"
          f" {o['arg']:>6.1%}"
          f" {o['fc']:>6.1%}"
          f" {nl_str}"
          f" │{pp:>5.0%}"
          f" {fp:>5.0%}"
          f" {gap:>+5.0%}p{mark_str}")

    w("")
    # Agent 1위 선정 근거
    if best_perf_model != best_perf_raw:
        raw_s = _short(best_perf_raw)
        raw_pp = prod_perf[best_perf_raw]
        raw_fp = overall[best_perf_raw]["performance"]
        agent_s = _short(best_perf_model)
        agent_pp = prod_perf[best_perf_model]
        agent_fp = overall[best_perf_model]["performance"]
        agent_safe = safe_turns[best_perf_model]
        agent_safe_str = f"~T{agent_safe}" if agent_safe > 0 else "T3 미만"
        w(f"  ※ 전체 Perf 1위는 {raw_s}(실무 {raw_pp:.0%} → 전체 {raw_fp:.0%})이나,")
        w(f"    {agent_s}(실무 {agent_pp:.0%} → 전체 {agent_fp:.0%})가 "
          f"85%+ {agent_safe_str}까지 유지 → Agent 1위.")
    w("")

    # ── 1위 모델 실무 구간 병목 ──
    w(f"  [1위 모델 실무 구간(@T{PRODUCTION_CUTOFF}) 세부]")
    w(f"    대상: {best_m} | 실무 턴: {_bp_total_n}턴 "
      f"(tool_call {_bp_tc_n} + no_call {_bp_nc_n})")
    def _bp_label(v, hi=0.85, lo=0.75):
        if v >= hi: return "이미 우수"
        if v >= lo: return "양호"
        return "🔴 병목"
    w(f"    Tool Acc  {_bp_prod_tool:>5.0%}  ← {_bp_label(_bp_prod_tool)}")
    w(f"    Arg Acc   {_bp_prod_arg:>5.0%}  ← {_bp_label(_bp_prod_arg)}")
    w(f"    FC Judge  {_bp_prod_fc:>5.0%}  ← {_bp_label(_bp_prod_fc)}")
    w(f"    No-Call   {_bp_prod_nc:>5.0%}  ← {_bp_label(_bp_prod_nc, hi=0.70, lo=0.50)}")
    w(f"    ─────────────────────")
    w(f"    Perf      {best_prod:>5.0%}")
    w(f"    → 개선 우선순위: "
      + " > ".join(f"{n}({v:.0%})" for n, v in
                   sorted(_bottlenecks, key=lambda x: x[1])))
    w("")

    # ════════════════════════════════════════════════════════════════
    # 2. 능력 해부 — 3개 테이블
    # ════════════════════════════════════════════════════════════════
    w("=" * 78)
    w("  2. 능력 해부 — Single / Parallel / No-Call")
    w("=" * 78)
    n_s = sp[models[0]]["single_n"] if models else 82
    n_p = sp[models[0]]["parallel_n"] if models else 12
    n_nc = sp[models[0]]["nc_n"] if models else 12
    w("")

    # ── 2a. Single ──
    w(f"  [Single — tool 1개 호출 ({n_s}턴)]")
    w(f"    {'모델':<28} {'tool 정답':>9} {'인자 정답':>10}")
    w(f"    {'─' * 28} {'─' * 9} {'─' * 10}")
    for model in models:
        d = sp[model]
        w(f"    {_short(model):<28} {d['single_tool']:>8.0%} {d['single_arg']:>9.0%}")
    w("")

    # ── 2b. Parallel ──
    w(f"  [Parallel — tool 2개 동시 호출 ({n_p}턴)]")
    w(f"    {'모델':<28} {'tool 정답':>9} {'인자 정답':>10} {'2개 인식':>9}")
    w(f"    {'─' * 28} {'─' * 9} {'─' * 10} {'─' * 9}")
    for model in models:
        d = sp[model]
        w(f"    {_short(model):<28} {d['parallel_tool']:>8.0%} {d['parallel_arg']:>9.0%} {d['parallel_detect']:>8.0%}")
    w(f"    → 최고 {sp[best_par_model]['parallel_detect']:.0%}. 실서비스에서는 1개씩 분리 호출 필요.")
    w("")

    # ── 2c. No-Call ──
    w(f"  [No-Call — tool 안 불러야 정답 ({n_nc}턴)]")
    w(f"    {'모델':<28} {'미호출 정답':>11} {'질문':>7} {'거부':>7} {'누락 전부 질문':>14} {'텍스트 품질':>12}")
    w(f"    {'─' * 28} {'─' * 11} {'─' * 7} {'─' * 7} {'─' * 14} {'─' * 12}")
    for model in models:
        d = sp[model]
        nl_str = f"{d['nc_nl_quality']:>11.0%}" if d.get("nc_nl_quality") is not None else "        N/A"
        w(f"    {_short(model):<28} {d['nc_acc']:>10.0%} {d['nc_slot_acc']:>6.0%} {d['nc_rel_acc']:>6.0%} {d['nc_slot_completeness']:>13.0%} {nl_str}")

    nc_perfect_fake = [m for m in models
                       if sp[m]["nc_acc"] >= 0.99 and overall[m]["tool"] < 0.50]
    nc_fail = [m for m in models
               if sp[m]["nc_acc"] < 0.50 and overall[m]["performance"] > 0.30]
    if nc_perfect_fake:
        w(f"    ⚠ {', '.join(_short(m) for m in nc_perfect_fake)}: "
          f"100%이지만 tool 자체를 못 불러서 높은 것 (의미 없음)")
    if nc_fail:
        w(f"    ⚠ {', '.join(_short(m) for m in nc_fail)}: "
          f"정보 부족해도 tool 호출 → 위험")
    w("")

    # ── Tool 호출 성향 분석 (trade-off) ──
    # Tool Acc 높지만 No-Call 낮은 모델 vs 그 반대
    aggressive = [m for m in usable if overall[m]["tool"] >= 0.70 and sp[m]["nc_acc"] < 0.50]
    conservative = [m for m in usable if sp[m]["nc_acc"] >= 0.80 and overall[m]["tool"] < 0.60]
    if aggressive or conservative:
        w(f"  [No-Call vs Tool 호출 — trade-off 분석]")
        w(f"    {'모델':<28} {'Tool Acc':>9} {'NC 정답':>8} {'성향':>14}")
        w(f"    {'─' * 28} {'─' * 9} {'─' * 8} {'─' * 14}")
        for model in models:
            if model in unusable:
                continue
            t_acc = overall[model]["tool"]
            nc = sp[model]["nc_acc"]
            if t_acc >= 0.70 and nc < 0.50:
                tendency = "tool 과잉"
            elif nc >= 0.80 and t_acc < 0.60:
                tendency = "tool 부족"
            elif t_acc >= 0.70 and nc >= 0.60:
                tendency = "균형"
            else:
                tendency = "-"
            w(f"    {_short(model):<28} {t_acc:>8.0%} {nc:>7.0%} {tendency:>14}")
        if aggressive:
            w(f"    → tool 과잉 ({len(aggressive)}개 모델): "
              f"No-Call 정확도가 낮아 불필요한 tool 호출 발생")
        w("")

    # ════════════════════════════════════════════════════════════════
    # 3. "몇 턴까지 버티나?" — 성능 곡선
    # ════════════════════════════════════════════════════════════════
    w("=" * 78)
    w("  3. 성능 곡선 — 몇 턴까지 85%를 유지하는가?")
    w("=" * 78)
    w("")
    w("  시나리오를 T3~T19 지점에서 잘라 누적 평균을 계산한다.")
    w("  🟢 90%+ | 🔵 85%+ | 🟡 75%+ | 🔴 <75%  (85% = 절대 임계선)")
    w("")

    w("  [3a] Performance 종합")
    _turnpoint_table(w, "Performance", cumul_perf, models, as_zone=True)
    w("")

    w("  [3b] Tool Name Acc")
    _turnpoint_table(w, "Tool", cumul_tool, models, as_zone=False)
    w("")

    w("  [3c] Arg Value Acc")
    _turnpoint_table(w, "Arg", cumul_arg, models, as_zone=False)
    w("")

    w("  [3d] FC Judgment")
    _turnpoint_table(w, "FC", cumul_fc, models, as_zone=False)
    w("")

    # [3e] Stress별 Performance 곡선
    st_names = {"ST1": "조건누적", "ST2": "맥락희석", "ST3": "교란주입"}
    w("  [3e] Stress별 Performance 곡선")
    w("  → 동일 turn-point에서 어떤 스트레스 유형이 먼저 성능을 깎는지 비교")
    w("")
    for st in ("ST1", "ST2", "ST3"):
        cumul_st = {m: cumul_perf_st[m][st] for m in models}
        w(f"    [{st} — {st_names[st]}]")
        _turnpoint_table(w, st, cumul_st, models, as_zone=True)
        w("")

    # ST 간 최대 편차가 큰 모델 식별
    w("    [Stress 민감도 요약]")
    w(f"      {'모델':<28} {'ST1':>6} {'ST2':>6} {'ST3':>6} {'최대편차':>8} {'최약':>10}")
    w(f"      {'─' * 28} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 8} {'─' * 10}")
    for model in models:
        if model in unusable:
            continue
        # cross["st_perf"]를 사용하여 4절과 동일한 소스 보장
        st_finals = cross[model]["st_perf"]
        if not st_finals:
            continue
        spread = max(st_finals.values()) - min(st_finals.values())
        worst = min(st_finals, key=st_finals.get)
        w(f"      {_short(model):<28} {st_finals.get('ST1', 0):>5.0%} {st_finals.get('ST2', 0):>5.0%}"
          f" {st_finals.get('ST3', 0):>5.0%} {spread * 100:>5.1f}%p"
          f"  {worst}({st_names.get(worst, '')})")
    w("")

    # ── tool 과잉 ↔ 교란 내성 인사이트 ──
    # ST3 no_call 비중이 낮으므로 (36턴 중 4턴=11%), tool 과잉 모델이
    # 교란(ST3)에 오히려 강할 수 있음을 분석
    aggressive_models = [m for m in usable
                         if overall[m]["tool"] >= 0.70 and sp[m]["nc_acc"] < 0.50]
    if aggressive_models:
        # ST3 vs 다른 ST의 성능 비교
        st3_stronger = []
        for m in aggressive_models:
            st_f = cross[m]["st_perf"]
            if st_f["ST3"] >= max(st_f["ST1"], st_f["ST2"]):
                st3_stronger.append(m)

        if st3_stronger:
            w("    [인사이트: tool 과잉 성향 ↔ 교란 내성]")
            w(f"      ST3 교란주입은 tool_call 턴이 ~89%를 차지한다.")
            w(f"      'tool 과잉' 모델은 no_call에 약하지만, 교란 후에도 주저 없이")
            w(f"      올바른 tool을 호출하므로 ST3 성능이 오히려 높다.")
            w("")
            w(f"      {'모델':<28} {'NC정답':>7} {'ST1':>6} {'ST2':>6} {'ST3':>6} {'ST3이 최고?':>12}")
            w(f"      {'─' * 28} {'─' * 7} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 12}")
            for m in aggressive_models:
                nc = sp[m]["nc_acc"]
                st_f = cross[m]["st_perf"]
                is_best = "✓" if m in st3_stronger else ""
                w(f"      {_short(m):<28} {nc:>6.0%} {st_f['ST1']:>5.0%}"
                  f" {st_f['ST2']:>5.0%} {st_f['ST3']:>5.0%} {is_best:>12}")
            # 반대로 균형/보수 모델의 ST3 점수
            balanced = [m for m in usable
                        if sp[m]["nc_acc"] >= 0.60 and overall[m]["tool"] >= 0.70
                        and m not in aggressive_models]
            if balanced:
                w(f"      ─── 비교: 균형 모델 ───")
                for m in balanced:
                    nc = sp[m]["nc_acc"]
                    st_f = cross[m]["st_perf"]
                    w(f"      {_short(m):<28} {nc:>6.0%} {st_f['ST1']:>5.0%}"
                      f" {st_f['ST2']:>5.0%} {st_f['ST3']:>5.0%}")
            w(f"      → tool 과잉 성향은 교란 내성에서 유리하나,")
            w(f"        no_call 정확도를 희생하는 trade-off가 존재한다.")
            w("")

    # 붕괴 순서
    fc_resilient = sum(
        1 for m in usable
        if cumul_fc[m].get(17, cumul_fc[m].get(15, 0)) > cumul_tool[m].get(17, cumul_tool[m].get(15, 0))
        and cumul_fc[m].get(17, cumul_fc[m].get(15, 0)) > 0.50
    )
    w(f"  붕괴 순서: 인자(Arg) → 도구 선택(Tool) → 행동 판단(FC) ({fc_resilient}/{len(usable)} 모델)")
    w("")

    # ════════════════════════════════════════════════════════════════
    # 4. "왜 망가지나?" — 원인 분석
    # ════════════════════════════════════════════════════════════════
    w("=" * 78)
    w("  4. 원인 분석 — 무엇이 성능을 떨어뜨리는가?")
    w("=" * 78)
    w("")
    w("  6개 시나리오(O1/O2 × ST1/ST2/ST3)에서 '어떤 스트레스가 더 치명적인가' 비교.")
    w("")

    # Stress Type table with performance
    w("  [스트레스 유형별 Performance]")
    w(f"    {'모델':<28} {'ST1(누적)':>9} {'ST2(희석)':>9} {'ST3(교란)':>9} {'편차':>8}")
    w(f"    {'─' * 28} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 8}")
    # ST별 1위 (usable)
    _st_best = {}
    for st in ["ST1", "ST2", "ST3"]:
        _st_best[st] = max(usable, key=lambda m: cross[m]["st_perf"].get(st, 0), default=None) if usable else None

    for model in models:
        d = cross[model]["st_perf"]
        vals = list(d.values())
        spread = max(vals) - min(vals) if vals else 0
        row = f"    {_short(model):<28}"
        for st in ["ST1", "ST2", "ST3"]:
            v = d.get(st, 0)
            mark = " ◀" if model == _st_best.get(st) and model in usable else ""
            row += f" {v:>8.1%}{mark}"
        row += f" {spread * 100:>5.1f}%p"
        w(row)
    w("")

    # Outcome table
    w("  [콜 유형별 Tool Acc]")
    w(f"    {'모델':<28} {'O1(청약)':>9} {'O2(보류)':>9} {'차이':>8}")
    w(f"    {'─' * 28} {'─' * 9} {'─' * 9} {'─' * 8}")
    for model in models:
        d = cross[model]["outcome_tool"]
        o1 = d.get("O1", 0)
        o2 = d.get("O2", 0)
        diff = abs(o1 - o2)
        w(f"    {_short(model):<28} {o1:>8.1%} {o2:>8.1%} {diff * 100:>5.1f}%p")
    w("")

    # 모델별 최약 ST 집계
    _worst_cnt = defaultdict(int)
    for m in usable:
        _d = cross[m]["st_perf"]
        if _d:
            _worst_cnt[min(_d, key=_d.get)] += 1
    _dominant_worst = max(_worst_cnt, key=_worst_cnt.get) if _worst_cnt else st_order[-1]
    w(f"  → 가장 치명적: {st_names[_dominant_worst]}({_dominant_worst}) "
      f"— usable {len(usable)}개 모델 중 {_worst_cnt[_dominant_worst]}개가 최약")
    w(f"    청약 vs 보류 차이: 평균 {avg_outcome_diff*100:.1f}%p(미미) "
      f"→ 변별력은 스트레스 유형(ST1/ST2/ST3)에 있음")
    w("")

    # ════════════════════════════════════════════════════════════════
    # 4b. Error Taxonomy — 에러 유형 분류
    # ════════════════════════════════════════════════════════════════
    w("=" * 78)
    w("  4b. Error Taxonomy — 에러 유형 분류")
    w("=" * 78)
    w("")
    w("  각 턴의 실패를 6개 태그로 분류하여 '어떤 종류의 실수를 하는가' 진단.")
    w("  개선 방향: 각 에러 유형의 Top 태그를 우선 개선.")
    w("")
    w("  태그 정의:")
    for tag, desc in ERROR_TAGS.items():
        w(f"    {tag:<14} {desc}")
    w("")

    # 에러 테이블
    et_cols = [28, 10, 10, 10, 10, 10, 10, 8]
    et_sep = "  " + "+".join("-" * c for c in et_cols) + "+"
    w(et_sep)
    w(f"  | {'모델':<{et_cols[0]-2}}"
      f"| {'WRONG':^{et_cols[1]-1}}"
      f"| {'MISSED':^{et_cols[2]-1}}"
      f"| {'FALSE':^{et_cols[3]-1}}"
      f"| {'ARG_MIS':^{et_cols[4]-1}}"
      f"| {'ARG_WR':^{et_cols[5]-1}}"
      f"| {'STALE':^{et_cols[6]-1}}"
      f"| {'OK':^{et_cols[7]-1}}|")
    w(f"  | {'':^{et_cols[0]-2}}"
      f"| {'_TOOL':^{et_cols[1]-1}}"
      f"| {'_CALL':^{et_cols[2]-1}}"
      f"| {'_CALL':^{et_cols[3]-1}}"
      f"| {'SING':^{et_cols[4]-1}}"
      f"| {'ONG':^{et_cols[5]-1}}"
      f"| {'(ST3)':^{et_cols[6]-1}}"
      f"| {'':^{et_cols[7]-1}}|")
    w(et_sep)

    for model in models:
        d = err_tax[model]
        short = _short(model)[:et_cols[0]-2]
        total = d["_total"]
        ok = d["_correct"]
        w(f"  | {short:<{et_cols[0]-2}}"
          f"| {d['WRONG_TOOL']:^{et_cols[1]-1}}"
          f"| {d['MISSED_CALL']:^{et_cols[2]-1}}"
          f"| {d['FALSE_CALL']:^{et_cols[3]-1}}"
          f"| {d['ARG_MISSING']:^{et_cols[4]-1}}"
          f"| {d['ARG_WRONG']:^{et_cols[5]-1}}"
          f"| {d['ARG_STALE']:^{et_cols[6]-1}}"
          f"| {ok:^{et_cols[7]-1}}|")
    w(et_sep)
    w("")

    # 모델별 Top-2 에러 태그 + 개선 방향
    w("  [모델별 Top 에러 + 개선 방향]")
    tag_fixes = {
        "WRONG_TOOL": "tool description 개선 또는 tool 선택 정확도 향상",
        "MISSED_CALL": "호출 판단 기준 강화",
        "FALSE_CALL": "No-Call 판별 정확도 향상",
        "ARG_MISSING": "필수 인자 채움 로직 보강",
        "ARG_WRONG": "인자 값 정확도 향상",
        "ARG_STALE": "대화 상태 추적/갱신 보강",
    }
    for model in models:
        if model in unusable:
            continue
        d = err_tax[model]
        errs = [(tag, d[tag]) for tag in ERROR_TAGS if d[tag] > 0]
        errs.sort(key=lambda x: x[1], reverse=True)
        top2 = errs[:2]
        short = _short(model)
        if top2:
            top_str = " > ".join(f"{tag}({cnt})" for tag, cnt in top2)
            fix = tag_fixes.get(top2[0][0], "")
            w(f"    {short:<28} {top_str}")
            w(f"    {'':28} → {fix}")
        else:
            w(f"    {short:<28} 에러 없음")
    w("")

    # ── 1위 모델 에러 인사이트 ──
    _best_err = err_tax[best_perf_model]
    _best_total = _best_err["_total"]
    _best_ok = _best_err["_correct"]
    _best_arg_total = _best_err["ARG_WRONG"] + _best_err["ARG_STALE"]
    _best_errs = [(tag, _best_err[tag]) for tag in ERROR_TAGS if _best_err[tag] > 0]
    _best_errs.sort(key=lambda x: x[1], reverse=True)

    w(f"  [1위 모델({_short(best_perf_model)}) 에러 인사이트]")
    w(f"    전체 {_best_total}턴 중 완벽 정답 {_best_ok}턴 ({_best_ok/_best_total:.0%})")
    w("")
    w(f"    ● 핵심 약점 — 인자 오류 {_best_arg_total}건 ({_best_arg_total/_best_total:.0%})")
    w(f"      ARG_WRONG({_best_err['ARG_WRONG']}) + ARG_STALE({_best_err['ARG_STALE']}):")
    w(f"      tool은 맞게 골랐지만 인자 값을 틀림.")
    if _best_err["ARG_STALE"] > 0:
        w(f"      특히 STALE {_best_err['ARG_STALE']}건은 고객이 값을 번복한 뒤")
        w(f"      이전 값을 갱신하지 못한 실수 → 대화 상태 추적 실패.")
    w("")
    if _best_err["WRONG_TOOL"] > 0:
        w(f"    ● tool 혼동 — WRONG_TOOL {_best_err['WRONG_TOOL']}건")
        w(f"      정답과 유사한 다른 tool을 호출하는 실수.")
        w("")
    if _best_err["FALSE_CALL"] > 0:
        w(f"    ● No-Call 실패 — FALSE_CALL {_best_err['FALSE_CALL']}건")
        w(f"      정보 부족/범위 밖 상황에서 tool을 호출해버림.")
        w("")
    if _best_err["MISSED_CALL"] == 0:
        w(f"    ● 강점 — MISSED_CALL 0건")
        w(f"      호출해야 할 때 빠뜨리는 일은 한 번도 없음.")
        w(f"      tool을 적극적으로 부르는 성향이 교란(ST3) 내성의 원인.")
        w("")
    # 한 줄 요약 — top 에러 기반 동적 생성
    _top_err = _best_errs[0] if _best_errs else None
    if _top_err:
        _err_to_lever = {
            "ARG_WRONG": "Arg Acc", "ARG_STALE": "Arg Acc",
            "ARG_MISSING": "Arg Acc", "WRONG_TOOL": "Tool Acc",
            "MISSED_CALL": "Tool Acc", "FALSE_CALL": "No-Call",
        }
        _lever = _err_to_lever.get(_top_err[0], _top_err[0])
        w(f"    한 줄 요약: 최다 에러는 {_top_err[0]}({_top_err[1]}건) → {_lever} 개선이 최우선.")
    w("")

    # ════════════════════════════════════════════════════════════════
    # 5. 모델별 판정 (자연어)
    # ════════════════════════════════════════════════════════════════
    w("=" * 78)
    w("  5. 모델별 판정")
    w("=" * 78)
    w("")

    # 순위 정렬 (usable → prod_perf 내림차순, unusable은 끝)
    _ranked = sorted(usable, key=lambda m: prod_perf[m], reverse=True)
    _ranked += [m for m in models if m in unusable]

    for rank_idx, model in enumerate(_ranked):
        short = _short(model)
        o = overall[model]
        d = sp[model]
        pp = prod_perf[model]
        safe_t = safe_turns[model]
        e = err_tax[model]

        if model in unusable:
            # ── 사용 불가 모델 ──
            w(f"  ❌ {short} — 사용 불가 (실무 {pp:.0%})")
            # 왜 사용 불가인지
            if o["tool"] < 0.15:
                w(f"    tool 호출 자체를 거의 못 함 (Tool {o['tool']:.0%}).")
                if d["nc_acc"] >= 0.90:
                    w(f"    No-Call {d['nc_acc']:.0%}는 tool을 못 불러서 높은 것이지,")
                    w(f"    판단이 좋은 게 아님.")
            else:
                w(f"    전체 Performance {o['performance']:.0%}로 실무 투입 기준 미달.")
            w("")
            continue

        # ── 판정 라벨 ──
        if model == best_perf_model:
            label = "권장"
            icon = "🏆"
        elif rank_idx == 1:
            label = "차선"
            icon = "  "
        elif pp >= 0.70:
            label = "조건부 사용"
            icon = "  "
        else:
            label = "비권장"
            icon = "  "

        w(f"  {icon} {short} — {label} (실무 {pp:.0%})")

        # ── 강점 ──
        strengths = []
        if model == best_perf_model:
            strengths.append(f"실무 {pp:.0%}로 1위")
        if e["MISSED_CALL"] == 0 and o["performance"] >= 0.50:
            strengths.append("호출 누락 0건 — tool을 적극적으로 부름")
        nl_q = o.get("nl_quality")
        if nl_q is not None and nl_q >= 0.80:
            strengths.append(f"NL {nl_q:.0%}로 답변 품질 우수 → Agent+답변 겸용 가능")
        if d["nc_acc"] >= 0.70:
            strengths.append(f"No-Call {d['nc_acc']:.0%}로 상황 판단이 정확 (균형형)")

        # ── 약점 ──
        weaknesses = []
        arg_err = e["ARG_WRONG"] + e["ARG_STALE"]
        if arg_err >= 15:
            stale_note = f"(번복 미갱신 {e['ARG_STALE']}건 포함)" if e["ARG_STALE"] > 5 else ""
            weaknesses.append(f"인자 오류 {arg_err}건{stale_note} — 값을 채우는 정밀도 부족")
        if e["WRONG_TOOL"] >= 15:
            weaknesses.append(f"tool 혼동 {e['WRONG_TOOL']}건 — 유사 tool 간 구분 실패")
        if d["nc_acc"] < 0.50 and o["tool"] >= 0.70:
            weaknesses.append(f"No-Call {d['nc_acc']:.0%} — 불러야/말아야 판단 부족 (tool 과잉)")
        if o["tool"] - o["arg"] > 0.35:
            weaknesses.append(f"Tool {o['tool']:.0%} vs Arg {o['arg']:.0%} — tool은 맞추지만 인자를 절반 이상 틀림")
        if d["parallel_detect"] == 0 and o["performance"] >= 0.50:
            weaknesses.append("복수호출 인식 0%")

        # ── 스트레스 약점 ──
        _st_d = cross[model]["st_perf"]
        if _st_d:
            _st_worst = min(_st_d, key=_st_d.get)
            _st_best = max(_st_d, key=_st_d.get)
            _st_spread = _st_d[_st_best] - _st_d[_st_worst]
            if _st_spread > 0.10:
                weaknesses.append(
                    f"{st_names[_st_worst]}({_st_worst})에 약함 "
                    f"({_st_d[_st_worst]:.0%}, 최강 {st_names[_st_best]} {_st_d[_st_best]:.0%}과 "
                    f"{_st_spread*100:.0f}%p 차이)")

        # ── 출력 ──
        if strengths:
            for s in strengths:
                w(f"    + {s}")
        if weaknesses:
            for wk in weaknesses:
                w(f"    - {wk}")

        # ── 비권장 사유 ──
        if label == "비권장":
            w(f"    → 실무 투입 시 리스크가 높음.")

        w("")

    # ════════════════════════════════════════════════════════════════
    # 6. 결론
    # ════════════════════════════════════════════════════════════════
    w("=" * 78)
    w("  6. 결론")
    w("=" * 78)
    w("")

    best_pp = prod_perf[best_perf_model]
    best_fp = overall[best_perf_model]["performance"]
    par_best = max(sp[m]["parallel_detect"] for m in models) if models else 0

    # ── 현재 위치 ──
    w(f"  [현재 위치 — {_short(best_perf_model)} @T{PRODUCTION_CUTOFF}]")
    w(f"    Performance {best_pp:.0%} = "
      f"Tool {_bp_prod_tool:.0%} + Arg {_bp_prod_arg:.0%} + "
      f"FC {_bp_prod_fc:.0%} (tool턴) / NC {_bp_prod_nc:.0%} (no-call턴)")
    w(f"    tool_call {_bp_tc_n}턴 ({_bp_tc_n/_bp_total_n:.0%}) + "
      f"no_call {_bp_nc_n}턴 ({_bp_nc_n/_bp_total_n:.0%})")
    w("")

    # ── 민감도 분석 ──
    w(f"  [민감도 분석 — 어디를 고치면 Performance가 가장 오르는가?]")
    sens_per10 = lambda n: (0.10 * n) / _bp_total_n  # +10%p 개선 시 Perf 변화
    sens = [
        ("Arg Acc", _bp_prod_arg, sens_per10(_bp_tc_n / 3),
         max(0, 0.95 - _bp_prod_arg)),
        ("No-Call", _bp_prod_nc, sens_per10(_bp_nc_n),
         max(0, 0.85 - _bp_prod_nc)),
        ("Tool Acc", _bp_prod_tool, sens_per10(_bp_tc_n / 3),
         max(0, 0.98 - _bp_prod_tool)),
    ]
    w(f"    {'지표':<12} {'현재':>6} {'여유':>8} {'민감도':>8} {'최대 효과':>10} {'우선순위':>8}")
    w(f"    {'─' * 12} {'─' * 6} {'─' * 8} {'─' * 8} {'─' * 10} {'─' * 8}")
    for name, cur, s_per10, headroom in sens:
        headroom_str = f"+{headroom*100:.0f}%p" if headroom > 0 else "포화"
        # 최대 효과 = 민감도 × 여유 (해당 지표를 천장까지 올렸을 때 Perf 변화)
        max_gain = s_per10 * (headroom / 0.10) if headroom > 0 else 0
        prio = "★★★" if max_gain >= 0.04 else ("★★" if max_gain >= 0.01 else "★")
        w(f"    {name:<12} {cur:>5.0%} {headroom_str:>8}"
          f" {s_per10*100:>+6.1f}%p {max_gain*100:>+8.1f}%p {prio:>8}")
    w(f"    민감도=+10%p당 Perf 변화 | 최대 효과=여유분 전부 개선 시 Perf 변화")
    # top-2 레버를 최대 효과 기준으로 동적 선택
    _sens_ranked = sorted(sens, key=lambda x: x[2] * (x[3] / 0.10) if x[3] > 0 else 0, reverse=True)
    _top_levers = [s[0] for s in _sens_ranked[:2] if s[3] > 0]
    if _top_levers:
        w(f"    → {'와 '.join(_top_levers)}이 가장 효과적인 개선 레버")
    w("")

    # ── 운영 가이드 ──
    w(f"  [운영 가이드]")
    w(f"    • 턴 제한: 실무 {PRODUCTION_CUTOFF}턴 이내 (현재 {best_pp:.0%}, 충분히 활용 가능)")
    w(f"    • T{PRODUCTION_CUTOFF} 이후 성능 하락은 스트레스 테스트 결과이며, 운영 목표 아님")
    w(f"    • 개선 후 이 벤치마크 재실행 → 달성 여부 확인")
    w("")
    w("=" * 78)

    # 파일 저장
    text = "\n".join(lines)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)

    # 콘솔에도 출력
    print(text)
    print(f"\n    report: {save_path.name}")


def generate_report_md(
    meta: dict,
    results: dict,
    save_path: Path,
):
    """GitHub 렌더링용 Markdown 리포트."""
    lines: list[str] = []
    w = lines.append

    models = list(results.keys())
    overall = compute_overall(results)
    sp = compute_single_parallel(results)
    cumul_perf = compute_turnpoint_performance(results)
    err_tax = compute_error_taxonomy(results)
    cross = compute_stress_cross_analysis(results)

    prod_perf: dict[str, float] = {}
    for model in models:
        prod_perf[model] = cumul_perf.get(model, {}).get(PRODUCTION_CUTOFF, 0)

    # safe turns
    safe_turns: dict[str, int] = {}
    for model in models:
        _vals = cumul_perf[model]
        _d85 = None
        for _c in TURN_CUTOFFS:
            _v = _vals.get(_c)
            if _v is not None and _d85 is None and _v < THRESHOLD_CRITICAL:
                _d85 = _c
        if _d85:
            _idx = TURN_CUTOFFS.index(_d85)
            safe_turns[model] = TURN_CUTOFFS[_idx - 1] if _idx > 0 else 0
        else:
            safe_turns[model] = max(TURN_CUTOFFS)

    usable = [m for m in models if overall[m]["performance"] >= 0.30]
    git_rev = _get_git_rev()

    # ── Header ──
    w("# AI TMR Assistant — 모델 성능 비교 리포트")
    w("")
    w(f"> 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
      f"턴: {meta.get('total_turns', '?')} × {len(models)}모델 | "
      f"Judge: {meta.get('judge_model', 'N/A')}  ")
    w(f"> commit: `{git_rev}` | run_id: `{meta.get('run_id', '?')}`  ")
    for cfg_line in _format_config(meta):
        w(f"> config:{cfg_line}")
    w("")

    # ── 용어 정의 ──
    w("> **@T7 (실무 구간)** = Turn 1\~7까지의 누적 성능. TMR 영업콜의 "
      "실무 턴 수(청약 \~7턴, 보류 \~5턴)에 대응하는 운영 기준선. "
      "T7 이후(T10\~T19)는 스트레스 테스트 구간(내구도 진단용).")
    w("")

    # ── 1. 성적표 ──
    w("## 1. 모델별 성적표")
    w("")
    w("| 모델 | Tool | Arg | FC | NL | 실무 Perf | 전체 Perf |")
    w("|------|------|-----|----|----|-----------|-----------|")
    for model in models:
        o = overall[model]
        pp = prod_perf[model]
        short = _short(model)
        nl_str = f"{o['nl_quality']:.0%}" if o["nl_quality"] is not None else "N/A"
        w(f"| {short} | {o['tool']:.1%} | {o['arg']:.1%} | "
          f"{o['fc']:.1%} | {nl_str} | {pp:.0%} | {o['performance']:.0%} |")
    w("")

    # ── 2. 능력 해부 ──
    w("## 2. 능력 해부 — Single / Parallel / No-Call")
    w("")
    w("| 모델 | S:Tool | S:Arg | P:Tool | P:Arg | P:감지 | NC:Acc |")
    w("|------|--------|-------|--------|-------|--------|--------|")
    for model in models:
        d = sp[model]
        w(f"| {_short(model)} | {d['single_tool']:.0%} | {d['single_arg']:.0%} | "
          f"{d['parallel_tool']:.0%} | {d['parallel_arg']:.0%} | "
          f"{d['parallel_detect']:.0%} | {d['nc_acc']:.0%} |")
    w("")

    # ── 3. 성능 곡선 ──
    w("## 3. 성능 곡선 — Turn-Point Performance")
    w("")
    w("🟢 90%+ | 🔵 85%+ | 🟡 75%+ | 🔴 <75%")
    w("")
    header = "| 모델 |" + " | ".join(f"~T{c}" for c in TURN_CUTOFFS) + " |"
    sep_row = "|------|" + " | ".join("---:" for _ in TURN_CUTOFFS) + " |"
    w(header)
    w(sep_row)
    for model in models:
        row = f"| {_short(model)} |"
        for c in TURN_CUTOFFS:
            v = cumul_perf[model].get(c)
            if v is None:
                row += " - |"
            else:
                row += f" {_zone_dot(v)} |"
        w(row)
    w("")

    # ── 4. 스트레스 민감도 ──
    w("## 4. 스트레스 민감도")
    w("")
    w("| 모델 | ST1(조건누적) | ST2(맥락희석) | ST3(교란주입) | 최약점 |")
    w("|------|-------------|-------------|-------------|--------|")
    for model in models:
        if model not in usable:
            continue
        d = cross[model]["st_perf"]
        vals = list(d.values())
        worst = min(d, key=d.get) if d else "-"
        st_names_map = {"ST1": "조건누적", "ST2": "맥락희석", "ST3": "교란주입"}
        w(f"| {_short(model)} | {d.get('ST1', 0):.0%} | {d.get('ST2', 0):.0%} | "
          f"{d.get('ST3', 0):.0%} | {worst} |")
    w("")

    # ── 4b. Error Taxonomy ──
    w("## 4b. Error Taxonomy")
    w("")
    w("각 턴의 실패를 6개 태그로 분류.")
    w("")
    w("| 태그 | 설명 |")
    w("|------|------|")
    for tag, desc in ERROR_TAGS.items():
        w(f"| `{tag}` | {desc} |")
    w("")

    w("| 모델 | WRONG_TOOL | MISSED_CALL | FALSE_CALL | ARG_MISSING | ARG_WRONG | ARG_STALE | OK |")
    w("|------|-----------|------------|-----------|------------|----------|----------|---|")
    for model in models:
        d = err_tax[model]
        ok = d["_correct"]
        w(f"| {_short(model)} | {d['WRONG_TOOL']} | {d['MISSED_CALL']} | "
          f"{d['FALSE_CALL']} | {d['ARG_MISSING']} | {d['ARG_WRONG']} | "
          f"{d['ARG_STALE']} | {ok} |")
    w("")

    # Top 에러
    tag_fixes = {
        "WRONG_TOOL": "tool 선택 정확도 향상",
        "MISSED_CALL": "호출 판단 기준 강화",
        "FALSE_CALL": "No-Call 판별 정확도 향상",
        "ARG_MISSING": "필수 인자 채움 보강",
        "ARG_WRONG": "인자 값 정확도 향상",
        "ARG_STALE": "대화 상태 추적 보강",
    }
    w("**모델별 Top 에러:**")
    w("")
    for model in models:
        if overall[model]["performance"] < 0.30:
            continue
        d = err_tax[model]
        errs = [(tag, d[tag]) for tag in ERROR_TAGS if d[tag] > 0]
        errs.sort(key=lambda x: x[1], reverse=True)
        if errs:
            top = errs[0]
            w(f"- **{_short(model)}**: `{top[0]}`({top[1]}) → {tag_fixes.get(top[0], '')}")
    w("")

    # ── 5. 운영 가이드 ──
    w("## 5. 운영 가이드")
    w("")
    best_model = max(usable, key=lambda m: prod_perf[m]) if usable else models[0]
    best_pp = prod_perf[best_model]
    best_safe = safe_turns.get(best_model, 0)
    best_safe_str = f"~T{best_safe}" if best_safe > 0 else "T3 미만"
    w(f"- **권장 모델**: {_short(best_model)} (실무 {best_pp:.0%})")
    w(f"- **권장 턴 제한**: {PRODUCTION_CUTOFF}턴 이내")
    w(f"- **85%+ 유지 구간**: {best_safe_str}")
    w(f"- 개선 후 벤치마크 재실행으로 달성 여부 확인")
    w("")

    # 저장
    text = "\n".join(lines)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"    report: {save_path.name}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 4: 결과 비교 & 성능 곡선")
    parser.add_argument("--run-id", type=str, help="분석할 run_id (부분 일치)")
    parser.add_argument("--charts", action="store_true", help="차트 생성 (기본: 리포트만)")
    parser.add_argument("--list", action="store_true", help="저장된 결과 목록")
    args = parser.parse_args()

    if args.list:
        list_results()
        return

    # 결과 파일 찾기
    if args.run_id:
        detail_path = find_detail_by_id(args.run_id)
    else:
        detail_path = find_latest_detail()

    if not detail_path:
        print("  ERROR: 결과 파일을 찾을 수 없습니다.")
        print("  먼저 python -m benchmark.run_benchmark 를 실행하세요.")
        sys.exit(1)

    print(f"\n  Loading: {detail_path.name}")

    meta, results = load_detail(detail_path)
    models = list(results.keys())
    run_id = meta.get("run_id", detail_path.stem.replace("detail_", ""))

    print(f"  Models : {len(models)}")
    print(f"  Run ID : {run_id}")
    print()

    # ── 차트 생성 (--charts 명시 시에만) ──
    if args.charts:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)

        print("  Generating charts...")

        # [1] Turn-point Performance 종합
        cumul_perf = compute_turnpoint_performance(results)
        chart_turnpoint_curve(
            cumul_perf,
            title="Turn-Point: Performance Score  (Tool + Arg + FC) / 3",
            ylabel="Cumulative Performance",
            save_path=CHARTS_DIR / f"turnpoint_performance_{run_id}.png",
        )

        # [2] Turn-point 누적 Tool Name Acc (no_call 제외)
        cumul_tool = compute_turnpoint(results, "bfcl.tool_name_acc", exclude_no_call=True)
        chart_turnpoint_curve(
            cumul_tool,
            title="Turn-Point: Cumulative Tool Name Accuracy",
            ylabel="Cumulative Tool Name Acc",
            save_path=CHARTS_DIR / f"turnpoint_tool_acc_{run_id}.png",
        )

        # [3] Turn-point 누적 Arg Value Acc (no_call 제외)
        cumul_arg = compute_turnpoint(results, "bfcl.arg_value_acc", exclude_no_call=True)
        chart_turnpoint_curve(
            cumul_arg,
            title="Turn-Point: Cumulative Arg Value Accuracy",
            ylabel="Cumulative Arg Value Acc",
            save_path=CHARTS_DIR / f"turnpoint_arg_acc_{run_id}.png",
        )

        # [4] 개별 턴 정확도 (no_call 제외)
        per_turn = compute_per_turn(results, "bfcl.tool_name_acc", exclude_no_call=True)
        chart_per_turn(
            per_turn,
            title="Per-Turn Tool Name Accuracy (Collapse Detection)",
            ylabel="Tool Name Acc at Turn N",
            save_path=CHARTS_DIR / f"per_turn_tool_acc_{run_id}.png",
        )

        # [5] Single vs Parallel
        sp = compute_single_parallel(results)
        chart_single_vs_parallel(
            sp,
            save_path=CHARTS_DIR / f"single_vs_parallel_{run_id}.png",
        )

        # [6] 시나리오 히트맵
        matrix = compute_scenario_matrix(results)
        chart_scenario_heatmap(
            matrix,
            save_path=CHARTS_DIR / f"scenario_heatmap_{run_id}.png",
        )

        # [7] Stress별 Performance 곡선 (3-subplot)
        cumul_perf_st = compute_turnpoint_performance_by_stress(results)
        chart_stress_turnpoint(
            cumul_perf_st,
            save_path=CHARTS_DIR / f"stress_performance_{run_id}.png",
        )

        print()

    # ── 텍스트 리포트 + Markdown 리포트 ──
    print("  Generating reports...")
    report_path = RESULTS_DIR / f"report_{run_id}.txt"
    generate_report(meta, results, report_path)

    report_md_path = RESULTS_DIR / f"report_{run_id}.md"
    generate_report_md(meta, results, report_md_path)


if __name__ == "__main__":
    main()
