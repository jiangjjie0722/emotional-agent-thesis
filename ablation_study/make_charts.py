#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
OUT = R / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

s = pd.read_csv(R / "condition_summary.csv").set_index("condition")
order = ["B0_single_llm", "B1_single_llm_rag", "A1_without_affect", "A2_without_validator", "A4_without_need_rag", "A3_full_agent"]
labels = ["B0", "B1", "A1", "A2", "A4", "A3"]
colors = ["#9AA4B2", "#78879A", "#5C7C9E", "#D08A55", "#56A3A6", "#1F4E79"]

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.45), constrained_layout=True)
metrics = [
    ("traceability_mean", "Traceability", 100),
    ("full_compliance_mean", "Full compliance", 100),
    ("topic_alignment_f1_mean", "Topic-alignment F1", 100),
]
for ax, (col, title, scale) in zip(axes, metrics):
    vals = s.loc[order, col].to_numpy() * scale
    bars = ax.bar(np.arange(len(order)), vals, color=colors, width=.72)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(np.arange(len(order)), labels, fontsize=7.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Percent / F1 × 100")
    ax.grid(axis="y", color="#D8DEE6", linewidth=.7, alpha=.8)
    ax.set_axisbelow(True)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+2, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
fig.suptitle("Ablation outcomes on 30 frozen cases", x=.01, ha="left", fontsize=14, fontweight="bold", color="#163A5F")
fig.savefig(OUT / "outcome_bars.png", dpi=240, bbox_inches="tight")
plt.close(fig)

c = pd.read_csv(R / "paired_comparisons.csv")
sub = c[c.metric == "traceability"].copy()
display = {
    "A3_full_agent - B0_single_llm": "vs B0 Single LLM",
    "A3_full_agent - B1_single_llm_rag": "vs B1 LLM + RAG",
    "A3_full_agent - A1_without_affect": "vs A1 without Affect",
    "A3_full_agent - A2_without_validator": "vs A2 without Validator",
    "A3_full_agent - A4_without_need_rag": "vs A4 without Need/RAG",
}
sub["label"] = sub.comparison.map(display)
sub = sub.set_index("comparison").loc[list(display)].reset_index()
y = np.arange(len(sub))[::-1]
diff = sub.difference.to_numpy()*100
lo = sub.ci_low.to_numpy()*100
hi = sub.ci_high.to_numpy()*100
fig, ax = plt.subplots(figsize=(7.8, 3.6), constrained_layout=True)
ax.errorbar(diff, y, xerr=[diff-lo, hi-diff], fmt="o", color="#1F4E79", ecolor="#6F8EA8", capsize=4, markersize=6)
ax.axvline(0, color="#9AA4B2", linestyle="--", linewidth=1)
ax.set_yticks(y, sub.label)
ax.set_xlabel("Paired difference in traceability (percentage points), 95% bootstrap CI")
ax.set_title("Full agent minus each comparison condition", loc="left", fontweight="bold", color="#163A5F")
ax.grid(axis="x", color="#D8DEE6", linewidth=.7)
for x, yy in zip(diff, y):
    ax.text(x + (1.2 if x >= 0 else -1.2), yy+.13, f"{x:+.1f}", ha="left" if x >= 0 else "right", fontsize=8)
fig.savefig(OUT / "traceability_effects.png", dpi=240, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.35), constrained_layout=True)
cost_metrics = [
    ("latency_s_mean", "End-to-end latency", "Seconds"),
    ("prompt_tokens_mean", "Input tokens", "Tokens / case"),
    ("completion_tokens_mean", "Output tokens", "Tokens / case"),
]
for ax, (col, title, ylabel) in zip(axes, cost_metrics):
    vals = s.loc[order, col].to_numpy()
    bars = ax.bar(np.arange(len(order)), vals, color=colors, width=.72)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(np.arange(len(order)), labels, fontsize=7.5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D8DEE6", linewidth=.7)
    ax.set_axisbelow(True)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v + max(vals)*.025, f"{v:.1f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_ylim(0, max(vals)*1.16)
fig.suptitle("Engineering cost by condition", x=.01, ha="left", fontsize=14, fontweight="bold", color="#163A5F")
fig.savefig(OUT / "cost_bars.png", dpi=240, bbox_inches="tight")
plt.close(fig)

print("created", *(str(x) for x in sorted(OUT.glob("*.png"))), sep="\n")
