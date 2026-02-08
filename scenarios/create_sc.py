import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ──────── Style ────────
plt.rcParams.update({
    "font.family": "AppleGothic",
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
})

turns = np.arange(1, 13)

# ──────── Scenario Definitions ────────
scenarios = [
    {
        "id": "S1",
        "name": "조건 누적",
        "eng": "State Accumulation",
        "pattern_desc": "턴마다 새 조건이 추가됨",
        "stress": np.linspace(2, 9, len(turns)),
        "color": "#2563eb",
        "bg": "#dbeafe",
        "cognitive": "Information Overload",
        "test_target": "Working Memory\n(동시 조건 추적)",
        "fail_signal": "이전 조건 누락/무시",
        "expected": "모든 조건을 동시에 반영",
    },
    {
        "id": "S2",
        "name": "맥락 희석",
        "eng": "Context Drift",
        "pattern_desc": "턴당 복잡도 일정, 대화만 길어짐",
        "stress": np.ones(len(turns)) * 5,
        "color": "#7c3aed",
        "bg": "#ede9fe",
        "cognitive": "Decay over Distance",
        "test_target": "Long-range Retention\n(장거리 맥락 유지)",
        "fail_signal": "초기 판단 변경/망각",
        "expected": "초기 판단을 끝까지 유지",
    },
    {
        "id": "S3",
        "name": "교란 주입",
        "eng": "Distraction Injection",
        "pattern_desc": "중간에 모순/혼란 정보 삽입",
        "stress": np.array([4, 4, 4, 4, 4, 4, 9, 4, 4, 4, 4, 4]),
        "color": "#dc2626",
        "bg": "#fee2e2",
        "cognitive": "Interference",
        "test_target": "Robustness\n(교란 내성)",
        "fail_signal": "교란에 판단 이탈",
        "expected": "교란 후 원래 판단으로 복귀",
    },
]

# ──────── Layout ────────
fig = plt.figure(figsize=(18, 15.5), facecolor="white")
gs = GridSpec(
    4, 3,
    height_ratios=[0.4, 3, 0.08, 3.2],
    hspace=0.22,
    wspace=0.30,
    left=0.06, right=0.94, top=0.93, bottom=0.03,
)

# ──────── Row 0: Title ────────
ax_title = fig.add_subplot(gs[0, :])
ax_title.axis("off")
ax_title.text(
    0.5, 0.75,
    "Multi-turn Stress Test : 시나리오 설계",
    fontsize=22, fontweight="bold", ha="center", va="center",
    transform=ax_title.transAxes,
)
ax_title.text(
    0.5, 0.10,
    "턴 수를 의도적으로 늘리며, 3가지 독립된 방식으로 난이도를 높여 모델의 state / 의도 / 맥락 유지 능력 한계를 관측한다",
    fontsize=13, ha="center", va="center", color="#475569",
    transform=ax_title.transAxes, style="italic",
)

# ──────── Row 1: 3 Charts ────────
for i, sc in enumerate(scenarios):
    ax = fig.add_subplot(gs[1, i])

    # Background danger zones
    ax.axhspan(0, 4, alpha=0.07, color="#22c55e", zorder=0)
    ax.axhspan(4, 7, alpha=0.07, color="#eab308", zorder=0)
    ax.axhspan(7, 10, alpha=0.07, color="#ef4444", zorder=0)

    # Zone labels (right edge)
    ax.text(12.8, 2.0, "Low",  fontsize=8, color="#16a34a", fontweight="bold", va="center", clip_on=False)
    ax.text(12.8, 5.5, "Mid",  fontsize=8, color="#ca8a04", fontweight="bold", va="center", clip_on=False)
    ax.text(12.8, 8.5, "High", fontsize=8, color="#dc2626", fontweight="bold", va="center", clip_on=False)

    # Critical threshold
    ax.axhline(y=7, color="#ef4444", linewidth=1.3, linestyle="--", alpha=0.45)
    ax.text(1.0, 7.35, "Critical Threshold", fontsize=8, color="#ef4444", alpha=0.7)

    # Stress area + line
    ax.fill_between(turns, 0, sc["stress"], alpha=0.18, color=sc["color"])
    ax.plot(
        turns, sc["stress"],
        color=sc["color"], linewidth=3.5,
        marker="o", markersize=8,
        markerfacecolor="white", markeredgewidth=2.5,
        zorder=5,
    )

    # ── S2 special: context distance overlay ──
    if sc["id"] == "S2":
        ax_dist = ax.twinx()
        context_distance = np.arange(1, len(turns) + 1)  # 1→12 linear
        ax_dist.fill_between(
            turns, 0, context_distance,
            alpha=0.08, color="#f59e0b",
        )
        ax_dist.plot(
            turns, context_distance,
            color="#d97706", linewidth=2.5, linestyle=":",
            zorder=4,
        )
        ax_dist.set_ylim(0, 14)
        ax_dist.set_ylabel(
            "초기 맥락과의 거리 (Context Distance)",
            fontsize=10, color="#d97706", fontweight="bold", labelpad=8,
        )
        ax_dist.tick_params(axis="y", labelcolor="#d97706", labelsize=9)
        ax_dist.spines["right"].set_color("#d97706")
        ax_dist.spines["right"].set_linewidth(1.5)
        ax_dist.spines["top"].set_visible(False)

    # Axes
    ax.set_xlabel("Turn", fontsize=12, fontweight="bold", labelpad=8)
    if i == 0:
        ax.set_ylabel("턴당 복잡도 (Per-turn Complexity)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylim(0, 10)
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(turns)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.grid(True, alpha=0.15, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(labelsize=10)

    # Title
    ax.set_title(
        f"{sc['id']}. {sc['name']} ({sc['eng']})\n{sc['pattern_desc']}",
        fontsize=13, fontweight="bold", pad=14, color=sc["color"],
    )

    # ── Per-scenario annotations ──
    if sc["id"] == "S1":
        ax.annotate(
            "조건 7개+ → 임계 진입",
            xy=(10, sc["stress"][9]), xytext=(5.5, 9.3),
            fontsize=9, color=sc["color"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=sc["color"], lw=1.5),
            ha="center",
        )
    elif sc["id"] == "S2":
        ax.annotate(
            "턴당 복잡도 일정 (의도적)",
            xy=(6, 5), xytext=(3, 8.5),
            fontsize=9, color=sc["color"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=sc["color"], lw=1.5),
            ha="center",
        )
        ax.annotate(
            "하지만 맥락 거리는\n계속 증가 (점선)",
            xy=(10, 5), xytext=(3, 1.8),
            fontsize=9, color="#d97706", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#d97706", lw=1.5),
            ha="center",
        )
    elif sc["id"] == "S3":
        ax.annotate(
            "교란 (Turn 7)",
            xy=(7, 9), xytext=(10, 9.3),
            fontsize=9, color=sc["color"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=sc["color"], lw=1.5),
            ha="center",
        )
        ax.annotate(
            "복귀 확인",
            xy=(8, 4), xytext=(10, 2.5),
            fontsize=9, color="#059669", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#059669", lw=1.5),
            ha="center",
        )

# ──────── Row 2: Divider ────────
ax_div = fig.add_subplot(gs[2, :])
ax_div.axis("off")
ax_div.axhline(y=0.5, color="#cbd5e1", linewidth=1.5, xmin=0.02, xmax=0.98)

# ──────── Row 3: MECE Table + Justification ────────
ax_tbl = fig.add_subplot(gs[3, :])
ax_tbl.axis("off")

table_data = [
    [
        sc["id"],
        f"{sc['name']}\n({sc['eng']})",
        sc["cognitive"],
        sc["test_target"],
        sc["fail_signal"],
        sc["expected"],
    ]
    for sc in scenarios
]

col_labels = [
    "",
    "Stress 방식",
    "인지적 원인",
    "검증 대상",
    "실패 신호\n(이러면 실패)",
    "정상 기대 행동\n(이래야 통과)",
]

tbl = ax_tbl.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="upper center",
    cellLoc="center",
    colWidths=[0.05, 0.18, 0.15, 0.18, 0.20, 0.20],
)
tbl.scale(1, 3.2)

# Style table
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#e2e8f0")
    cell.set_linewidth(1.5)

    if row == 0:  # header
        cell.set_facecolor("#1e293b")
        cell.get_text().set_color("white")
        cell.set_text_props(fontsize=10, weight="bold")
    else:
        sc = scenarios[row - 1]
        cell.set_text_props(fontsize=10)

        if col == 0:  # ID badge
            cell.set_facecolor(sc["bg"])
            cell.get_text().set_color(sc["color"])
            cell.set_text_props(fontsize=14, weight="bold")
        elif col == 1:  # stress type
            cell.get_text().set_color(sc["color"])
            cell.set_text_props(fontsize=10, weight="bold")
        elif col == 2:  # cognitive cause
            cell.get_text().set_color("#475569")
            cell.set_text_props(fontsize=10, style="italic")
        elif col == 4:  # fail
            cell.get_text().set_color("#dc2626")
            cell.set_text_props(fontsize=10, weight="bold")
        elif col == 5:  # expected
            cell.get_text().set_color("#059669")
            cell.set_text_props(fontsize=10, weight="bold")

        # subtle alternating rows
        if row % 2 == 0 and col not in [0]:
            cell.set_facecolor("#f8fafc")

# ──────── MECE Justification ────────
mece_y = 0.12
ax_tbl.text(
    0.5, mece_y,
    "MECE 분류 기준 (인지심리학 — 기억 실패의 3대 원인)",
    fontsize=12, ha="center", va="center", color="#1e293b", fontweight="bold",
    transform=ax_tbl.transAxes,
)
ax_tbl.text(
    0.5, mece_y - 0.06,
    "S1 = Information Overload  (처리할 정보가 너무 많아서)   |   "
    "S2 = Decay over Distance  (시간이 지나 희미해져서)   |   "
    "S3 = Interference  (방해 자극이 끼어들어서)",
    fontsize=10.5, ha="center", va="center", color="#475569",
    transform=ax_tbl.transAxes,
    bbox=dict(
        boxstyle="round,pad=0.6",
        facecolor="#f1f5f9",
        edgecolor="#94a3b8",
        linewidth=1.5,
    ),
)
ax_tbl.text(
    0.5, mece_y - 0.16,
    "→  세 원인은 상호 독립적이며, 멀티턴 대화에서 모델이 실패할 수 있는 주요 차원을 빠짐없이 커버",
    fontsize=10, ha="center", va="center", color="#64748b",
    transform=ax_tbl.transAxes,
)

# ──────── Save ────────
from pathlib import Path
out = Path(__file__).resolve().parent / "stress_test_summary.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Done → {out}")
