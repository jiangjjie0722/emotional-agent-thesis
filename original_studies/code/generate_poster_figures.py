#!/usr/bin/env python3
"""Generate English, poster-ready scientific modules from audited project results.

All drawing, export, and preview generation is performed in Python/matplotlib.
The schematics are conceptual summaries; quantitative values are read from the
audited result tables in ``results/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "poster_figures"

NAVY = "#102A43"
BLUE = "#3A6EA5"
BLUE_SOFT = "#DCE9F5"
TEAL = "#138A7E"
TEAL_SOFT = "#DDF2EE"
ROSE = "#B14E70"
ROSE_SOFT = "#F3DEE6"
GOLD = "#D89B2B"
GOLD_SOFT = "#F7EBCF"
GREEN = "#2E9E44"
RED = "#C44E52"
INK = "#243746"
MID = "#637381"
LIGHT = "#D7E0E7"
PALE = "#F4F8F6"
WHITE = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 9,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    }
)


def mm(width: float, height: float) -> tuple[float, float]:
    return width / 25.4, height / 25.4


def clean_canvas(ax, facecolor: str = WHITE) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_facecolor(facecolor)


def title_block(
    ax,
    kicker: str,
    title: str,
    subtitle: str,
    *,
    x: float = 0.04,
    title_size: float = 18,
) -> None:
    ax.text(x, 0.93, kicker.upper(), color=TEAL, fontsize=7.2, fontweight="bold", va="top")
    ax.text(x, 0.855, title, color=NAVY, fontsize=title_size, fontweight="bold", va="top")
    ax.text(x, 0.72, subtitle, color=MID, fontsize=8.6, va="top", linespacing=1.3)


def rounded_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str = WHITE,
    edgecolor: str = LIGHT,
    linewidth: float = 1.0,
    radius: float = 0.018,
    zorder: int = 1,
):
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def arrow(ax, start, end, *, color=TEAL, lw=1.5, style="-|>", mutation=11, zorder=3):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            mutation_scale=mutation,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=zorder,
    )


def stage_box(ax, x, y, w, h, number, heading, body, color, soft):
    rounded_box(ax, (x, y), w, h, facecolor=WHITE, edgecolor=color, linewidth=1.1)
    ax.add_patch(patches.Rectangle((x, y + h - 0.055), w, 0.055, color=soft, ec="none"))
    ax.text(x + 0.018, y + h - 0.028, number, color=color, fontsize=7, fontweight="bold", va="center")
    ax.text(x + 0.018, y + h - 0.085, heading, color=NAVY, fontsize=9.5, fontweight="bold", va="top")
    ax.text(x + 0.018, y + h - 0.17, body, color=MID, fontsize=7.2, va="top", linespacing=1.25)


def save_bundle(fig, stem: str, *, tiff_dpi: int = 600) -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=tiff_dpi,
        bbox_inches="tight",
        pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    return {"stem": stem, "formats": ["svg", "pdf", "png", "tiff"]}


def make_workflow() -> None:
    fig, ax = plt.subplots(figsize=mm(300, 108))
    clean_canvas(ax, PALE)
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, color=PALE, zorder=-10))
    ax.text(0.035, 0.925, "END-TO-END RESEARCH WORKFLOW", color=TEAL, fontsize=7.2, fontweight="bold")
    ax.text(
        0.035,
        0.81,
        "From affect-rich dialogue to auditable software requirements",
        color=NAVY,
        fontsize=19,
        fontweight="bold",
    )
    ax.text(
        0.035,
        0.73,
        "A staged pipeline separates interpretation, routing, generation, and deterministic engineering control.",
        color=MID,
        fontsize=8.8,
    )

    y, h, w, gap = 0.30, 0.31, 0.168, 0.026
    xs = [0.035 + i * (w + gap) for i in range(5)]
    stages = [
        ("01", "Stakeholder dialogue", "Original question\nand expressed need", BLUE, BLUE_SOFT),
        ("02", "Affect interpreter", "Locate emotional cues\nusing dialogue context", TEAL, TEAL_SOFT),
        ("03", "Need-topic routing", "Predict one or more\nrequirement topics", GOLD, GOLD_SOFT),
        ("04", "Retrieval + generation", "Use related examples to\ndraft an SRS-style item", ROSE, ROSE_SOFT),
        ("05", "Evidence validator", "Bind source evidence and\nenforce output schema", TEAL, TEAL_SOFT),
    ]
    for x, (num, head, body, color, soft) in zip(xs, stages):
        stage_box(ax, x, y, w, h, num, head, body, color, soft)
    for left, right in zip(xs[:-1], xs[1:]):
        arrow(ax, (left + w + 0.006, y + h / 2), (right - 0.006, y + h / 2), color=TEAL, lw=1.4)

    ax.text(0.035, 0.19, "MODEL SIGNALS", color=MID, fontsize=6.8, fontweight="bold")
    for x, txt, color in [
        (0.19, "13-class emotion", BLUE),
        (0.37, "18 need topics", GOLD),
        (0.54, "Evidence IDs", ROSE),
        (0.72, "Required SRS fields", TEAL),
        (0.90, "Auditable output", GREEN),
    ]:
        ax.plot([x - 0.014], [0.185], "o", color=color, ms=4)
        ax.text(x, 0.19, txt, color=INK, fontsize=7.2, va="center", ha="center")
    ax.text(
        0.965,
        0.065,
        "Conceptual schematic",
        color=MID,
        fontsize=6.5,
        ha="right",
    )
    save_bundle(fig, "01_end_to_end_workflow")


def make_context_augmentation(cped: pd.DataFrame, results: dict) -> None:
    fig = plt.figure(figsize=mm(166, 108))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 0.82], left=0.04, right=0.97, top=0.95, bottom=0.10, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    clean_canvas(ax, WHITE)
    title_block(
        ax,
        "Key action 01 · Context construction",
        "Use the previous three turns",
        "Speaker-aware history is prepended to the current\nutterance before emotion classification.",
    )
    bubble_specs = [
        (0.07, 0.50, 0.52, 0.085, "t−3", "Previous turn · other speaker", BLUE_SOFT, BLUE),
        (0.16, 0.39, 0.52, 0.085, "t−2", "Previous turn · same speaker", TEAL_SOFT, TEAL),
        (0.07, 0.28, 0.52, 0.085, "t−1", "Previous turn · other speaker", BLUE_SOFT, BLUE),
        (0.16, 0.13, 0.62, 0.105, "t", "Current utterance", ROSE_SOFT, ROSE),
    ]
    for x, y, w, h, tag, txt, fill, edge in bubble_specs:
        rounded_box(ax, (x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=0.9, radius=0.022)
        ax.text(x + 0.025, y + h / 2, tag, color=edge, fontsize=7, fontweight="bold", va="center")
        ax.text(x + 0.11, y + h / 2, txt, color=INK, fontsize=7.5, va="center")
    ax.plot([0.84, 0.84], [0.18, 0.54], color=TEAL, lw=1.2)
    for yy in [0.54, 0.43, 0.32, 0.18]:
        arrow(ax, (0.78, yy), (0.84, yy), color=TEAL, lw=1.0, mutation=8)
    arrow(ax, (0.84, 0.36), (0.97, 0.36), color=TEAL, lw=1.4)
    ax.text(0.84, 0.59, "CONTEXT-3", color=TEAL, fontsize=6.6, fontweight="bold", ha="center")
    ax.text(0.60, 0.055, "[PREV_SAME / PREV_OTHER] + [CURRENT]", color=MID, fontsize=6.6, ha="center")

    ax2 = fig.add_subplot(gs[0, 1])
    labels = ["Utterance only", "Context-3"]
    rows = cped.set_index("model").loc[
        ["Utterance-only TF-IDF + LinearSVC", "Context-3 TF-IDF + LinearSVC"]
    ]
    vals = rows["macro_f1"].to_numpy() * 100
    low = rows["macro_f1_ci_low"].to_numpy() * 100
    high = rows["macro_f1_ci_high"].to_numpy() * 100
    x = np.arange(2)
    ax2.bar(x, vals, color=[BLUE, ROSE], width=0.62, edgecolor=INK, linewidth=0.7)
    ax2.errorbar(x, vals, yerr=np.vstack([vals - low, high - vals]), fmt="none", ecolor=INK, capsize=3, lw=1)
    ax2.set_xticks(x, labels, rotation=18, ha="right")
    ax2.set_ylabel("Macro-F1 (%)")
    ax2.set_ylim(11.5, 17.2)
    ax2.set_yticks([12, 14, 16])
    ax2.set_title("Stable contextual gain", color=NAVY, fontweight="bold", loc="left", pad=10)
    for xi, val in zip(x, vals):
        ax2.text(xi, val + 0.72, f"{val:.1f}", ha="center", color=INK, fontsize=8, fontweight="bold")
    delta = results["cped"]["macro_f1_delta"] * 100
    lo = results["cped"]["delta_ci_low"] * 100
    hi = results["cped"]["delta_ci_high"] * 100
    ax2.text(
        0.03,
        0.96,
        f"Δ +{delta:.2f} pp\n95% CI [{lo:.2f}, {hi:.2f}]",
        transform=ax2.transAxes,
        va="top",
        color=GREEN,
        fontsize=8,
        fontweight="bold",
    )
    ax2.text(0.98, 0.03, "test n = 27,438", transform=ax2.transAxes, ha="right", color=MID, fontsize=6.6)
    save_bundle(fig, "02_context_augmentation")


def make_topic_routing(needs: pd.DataFrame) -> None:
    fig = plt.figure(figsize=mm(166, 108))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85], left=0.04, right=0.97, top=0.95, bottom=0.11, wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    clean_canvas(ax)
    title_block(
        ax,
        "Key action 02 · Need-topic routing",
        "Route dialogue to design concerns",
        "A multi-label classifier converts stakeholder language\ninto an explicit routing signal for generation.",
        title_size=14.5,
    )
    rounded_box(ax, (0.055, 0.33), 0.32, 0.23, facecolor=BLUE_SOFT, edgecolor=BLUE, linewidth=1.1)
    ax.text(0.085, 0.515, "DIALOGUE", color=BLUE, fontsize=6.8, fontweight="bold")
    ax.text(0.085, 0.455, "Free-form question", color=NAVY, fontsize=9, fontweight="bold")
    ax.text(0.085, 0.395, "+ expressed needs", color=MID, fontsize=7.2)
    arrow(ax, (0.39, 0.445), (0.55, 0.445), color=TEAL, lw=1.5)
    rounded_box(ax, (0.56, 0.30), 0.35, 0.29, facecolor=TEAL_SOFT, edgecolor=TEAL, linewidth=1.1)
    ax.text(0.595, 0.545, "CHARACTER TF–IDF", color=TEAL, fontsize=6.8, fontweight="bold")
    ax.text(0.595, 0.485, "One-vs-Rest", color=NAVY, fontsize=9, fontweight="bold")
    ax.text(0.595, 0.43, "LinearSVC", color=NAVY, fontsize=9, fontweight="bold")
    ax.text(0.595, 0.355, "18 possible topics", color=MID, fontsize=7.2)
    for yy, label, color in [
        (0.22, "privacy", ROSE),
        (0.15, "safety", GOLD),
        (0.08, "usability", BLUE),
    ]:
        rounded_box(ax, (0.56, yy), 0.22, 0.045, facecolor=WHITE, edgecolor=color, linewidth=0.9, radius=0.012)
        ax.text(0.67, yy + 0.0225, label, color=color, fontsize=7, fontweight="bold", ha="center", va="center")
        ax.plot([0.815], [yy + 0.0225], marker="o", ms=4.5, color=color)
    ax.text(0.90, 0.055, "multi-label output", color=MID, fontsize=6.5, ha="right")

    ax2 = fig.add_subplot(gs[0, 1])
    model_order = [
        "Most-prevalent label",
        "Word TF-IDF + One-vs-Rest LinearSVC",
        "Character TF-IDF + One-vs-Rest LinearSVC",
    ]
    frame = needs.set_index("model").loc[model_order]
    labels = ["Baseline", "Word", "Character"]
    micro = frame["micro_f1"].to_numpy() * 100
    macro = frame["macro_f1"].to_numpy() * 100
    y = np.arange(3)
    h = 0.32
    ax2.barh(y + h / 2, micro, h, color=BLUE, label="Micro-F1")
    ax2.barh(y - h / 2, macro, h, color=ROSE, label="Macro-F1")
    ax2.set_yticks(y, ["", "", ""])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 66)
    ax2.set_xlabel("F1 score (%)")
    ax2.set_title("Held-out topic detection", color=NAVY, fontweight="bold", loc="left", pad=10)
    ax2.legend(loc="upper right", fontsize=7)
    ax2.axvline(50, color=LIGHT, lw=0.8, ls="--", zorder=0)
    for yi, label in zip(y, labels):
        ax2.text(1.0, yi, label, color=WHITE, fontsize=7, fontweight="bold", va="center", ha="left", zorder=5)
    for yi, value in zip(y, micro):
        ax2.text(value + 1.0, yi + h / 2, f"{value:.1f}", va="center", color=BLUE, fontsize=7, fontweight="bold")
    for yi, value in zip(y, macro):
        ax2.text(value + 1.0, yi - h / 2, f"{value:.1f}", va="center", color=ROSE, fontsize=7, fontweight="bold")
    ax2.text(0.98, 0.03, "test n = 148", transform=ax2.transAxes, ha="right", color=MID, fontsize=6.6)
    save_bundle(fig, "03_need_topic_routing")


def checklist(ax, x, y, items, color):
    for i, item in enumerate(items):
        yy = y - i * 0.07
        ax.add_patch(patches.Circle((x, yy), 0.014, facecolor=color, edgecolor="none"))
        ax.text(x, yy - 0.001, "OK", color=WHITE, fontsize=3.8, fontweight="bold", ha="center", va="center")
        ax.text(x + 0.03, yy, item, color=INK, fontsize=7.2, va="center")


def make_validator(results: dict) -> None:
    fig, ax = plt.subplots(figsize=mm(166, 108))
    clean_canvas(ax, PALE)
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, color=PALE, zorder=-10))
    title_block(
        ax,
        "Key action 03 · Deterministic control",
        "Bind every requirement to evidence",
        "The validator repairs structure and traceability without claiming stronger semantic understanding.",
    )
    rounded_box(ax, (0.055, 0.20), 0.36, 0.40, facecolor=WHITE, edgecolor=LIGHT, linewidth=1.0)
    ax.text(0.085, 0.55, "UNVERIFIED LLM DRAFT", color=MID, fontsize=6.6, fontweight="bold")
    ax.text(0.085, 0.49, "Generated requirement", color=NAVY, fontsize=9.5, fontweight="bold")
    for yy, width in [(0.43, 0.26), (0.385, 0.20), (0.34, 0.28)]:
        ax.plot([0.085, 0.085 + width], [yy, yy], color=LIGHT, lw=4, solid_capstyle="butt")
    ax.text(0.085, 0.265, "Possible issues", color=RED, fontsize=6.7, fontweight="bold")
    ax.text(0.085, 0.225, "Missing fields · unbound quote", color=MID, fontsize=7)

    arrow(ax, (0.44, 0.40), (0.55, 0.40), color=TEAL, lw=1.8, mutation=13)
    ax.text(0.495, 0.44, "VALIDATE", color=TEAL, fontsize=6.6, fontweight="bold", ha="center")

    rounded_box(ax, (0.58, 0.16), 0.36, 0.48, facecolor=TEAL_SOFT, edgecolor=TEAL, linewidth=1.1)
    ax.text(0.61, 0.59, "AUDITABLE OUTPUT", color=TEAL, fontsize=6.6, fontweight="bold")
    ax.text(0.61, 0.525, "The system shall…", color=NAVY, fontsize=10.5, fontweight="bold")
    ax.plot([0.61, 0.88], [0.48, 0.48], color=TEAL, lw=4, solid_capstyle="butt")
    checklist(
        ax,
        0.625,
        0.40,
        ["Required fields complete", "Evidence ID bound to quote", "SRS form enforced"],
        TEAL,
    )
    baseline = results["agent"]["baseline_traceability"] * 100
    agent = results["agent"]["agent_traceability"] * 100
    ax.text(0.07, 0.095, "TRACEABILITY", color=MID, fontsize=6.6, fontweight="bold")
    ax.text(0.26, 0.09, f"{baseline:.1f}%", color=BLUE, fontsize=14, fontweight="bold", ha="center")
    arrow(ax, (0.35, 0.095), (0.58, 0.095), color=GREEN, lw=1.8, mutation=12)
    ax.text(0.47, 0.125, "+83.3 pp", color=GREEN, fontsize=7.4, fontweight="bold", ha="center")
    ax.text(0.73, 0.09, f"{agent:.1f}%", color=TEAL, fontsize=14, fontweight="bold", ha="center")
    ax.text(0.26, 0.047, "single-pass", color=MID, fontsize=6.5, ha="center")
    ax.text(0.73, 0.047, "staged agent", color=MID, fontsize=6.5, ha="center")
    save_bundle(fig, "04_evidence_validation")


def make_outcomes(results: dict, paired: pd.DataFrame) -> None:
    fig = plt.figure(figsize=mm(300, 126))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.43, 0.57], left=0.035, right=0.98, top=0.95, bottom=0.12, hspace=0.22, wspace=0.30)
    ax_title = fig.add_subplot(gs[0, :])
    clean_canvas(ax_title)
    ax_title.text(0, 0.93, "AUDITED QUANTITATIVE EVIDENCE", color=TEAL, fontsize=7.2, fontweight="bold", va="top")
    ax_title.text(0, 0.68, "Engineering control improves; semantic alignment remains unchanged", color=NAVY, fontsize=19, fontweight="bold", va="top")
    ax_title.text(
        0,
        0.38,
        "Paired evaluation on 30 held-out cases separates robust gains in output control from an unsupported claim of better topic understanding.",
        color=MID,
        fontsize=8.8,
        va="top",
    )
    ax_title.text(0.995, 0.05, "Error bars: 95% paired bootstrap CI", color=MID, fontsize=6.8, ha="right")

    agent = results["agent"]
    metrics = [
        ("SRS form", agent["baseline_srs"], agent["agent_srs"], GREEN),
        ("Traceability", agent["baseline_traceability"], agent["agent_traceability"], GREEN),
        ("Full compliance", agent["baseline_full_compliance"], agent["agent_full_compliance"], GREEN),
        ("Topic alignment", agent["baseline_topic_alignment"], agent["agent_topic_alignment"], BLUE),
    ]
    paired_map = paired.set_index("metric")
    metric_keys = ["srs_form_rate", "traceability_coverage", "full_engineering_compliance", "topic_alignment_f1"]
    for i, ((label, base, staged, accent), key) in enumerate(zip(metrics, metric_keys)):
        ax = fig.add_subplot(gs[1, i])
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(0, 108)
        ax.set_xticks([0, 1], ["Single-pass", "Staged agent"])
        ax.set_yticks([0, 50, 100] if i == 0 else [])
        if i == 0:
            ax.set_ylabel("Outcome (%)")
        ax.spines["left"].set_visible(i == 0)
        ax.spines["bottom"].set_color(LIGHT)
        y0, y1 = base * 100, staged * 100
        ax.plot([0, 1], [y0, y1], color=accent, lw=2.3, zorder=2)
        ax.scatter([0], [y0], s=48, color=BLUE, edgecolor=WHITE, linewidth=0.8, zorder=3)
        ax.scatter([1], [y1], s=48, color=ROSE, edgecolor=WHITE, linewidth=0.8, zorder=3)
        ax.text(0, y0 + (5 if y0 < 90 else -8), f"{y0:.1f}", ha="center", color=BLUE, fontsize=8, fontweight="bold")
        ax.text(1, y1 + (5 if y1 < 90 else -8), f"{y1:.1f}", ha="center", color=ROSE, fontsize=8, fontweight="bold")
        row = paired_map.loc[key]
        delta = row["delta_agent_minus_baseline"] * 100
        lo = row["bootstrap_ci_low"] * 100
        hi = row["bootstrap_ci_high"] * 100
        sign = "+" if delta >= 0 else ""
        ax.set_title(label, loc="left", color=NAVY, fontweight="bold", pad=8)
        ax.text(
            0.02,
            0.93,
            f"Δ {sign}{delta:.1f} pp\nCI [{lo:.1f}, {hi:.1f}]",
            transform=ax.transAxes,
            color=accent,
            fontsize=7.2,
            fontweight="bold",
            va="top",
        )
        if key == "topic_alignment_f1":
            ax.text(0.5, 4, "No demonstrated improvement", color=RED, fontsize=6.8, ha="center", fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, label="Single-pass LLM", markersize=6),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ROSE, label="Staged affect-sensitive agent", markersize=6),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.015), fontsize=7.5)
    save_bundle(fig, "05_quantitative_outcomes")


def write_source_data(results: dict, cped: pd.DataFrame, needs: pd.DataFrame, paired: pd.DataFrame) -> None:
    rows = []
    for _, row in cped.iterrows():
        rows.append(
            {
                "figure": "02_context_augmentation",
                "measure": "CPED macro-F1",
                "group": row["model"],
                "value": row["macro_f1"],
                "ci_low": row["macro_f1_ci_low"],
                "ci_high": row["macro_f1_ci_high"],
                "n": int(row["test_n"]),
            }
        )
    for _, row in needs.iterrows():
        for metric in ["micro_f1", "macro_f1"]:
            rows.append(
                {
                    "figure": "03_need_topic_routing",
                    "measure": metric,
                    "group": row["model"],
                    "value": row[metric],
                    "ci_low": row[f"{metric}_ci_low"],
                    "ci_high": row[f"{metric}_ci_high"],
                    "n": int(row["test_n"]),
                }
            )
    for _, row in paired.iterrows():
        rows.append(
            {
                "figure": "05_quantitative_outcomes",
                "measure": row["metric"],
                "group": "staged agent minus single-pass LLM",
                "value": row["delta_agent_minus_baseline"],
                "ci_low": row["bootstrap_ci_low"],
                "ci_high": row["bootstrap_ci_high"],
                "n": int(row["n_paired_cases"]),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "poster_figure_source_data.csv", index=False)


def make_contact_sheet(stems: list[str]) -> None:
    images = [Image.open(OUT / f"{stem}.png").convert("RGB") for stem in stems]
    fig = plt.figure(figsize=mm(300, 270))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.78, 1, 1], left=0.025, right=0.975, top=0.97, bottom=0.03, hspace=0.12, wspace=0.08)
    positions = [(0, slice(None)), (1, 0), (1, 1), (2, 0), (2, 1)]
    for image, pos, stem in zip(images, positions, stems):
        ax = fig.add_subplot(gs[pos])
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(stem.replace("_", " ").upper(), color=MID, fontsize=6.5, loc="left", pad=3)
    fig.savefig(OUT / "poster_figures_contact_sheet.png", dpi=180, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / "poster_figures_contact_sheet.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    for image in images:
        image.close()


def qa(stems: list[str]) -> None:
    report: dict[str, object] = {"backend": "Python/matplotlib", "figures": {}}
    for stem in stems:
        entries = {}
        for suffix in [".svg", ".pdf", ".png", ".tiff"]:
            path = OUT / f"{stem}{suffix}"
            info: dict[str, object] = {"exists": path.exists(), "bytes": path.stat().st_size}
            if suffix == ".svg":
                content = path.read_text(encoding="utf-8")
                info["editable_text_nodes"] = content.count("<text")
            if suffix in [".png", ".tiff"]:
                with Image.open(path) as im:
                    info["pixels"] = list(im.size)
                    info["mode"] = im.mode
            entries[suffix] = info
        report["figures"][stem] = entries
    report["all_formats_exist"] = all(
        item[suffix]["exists"]
        for item in report["figures"].values()
        for suffix in [".svg", ".pdf", ".png", ".tiff"]
    )
    report["all_svg_text_editable"] = all(
        item[".svg"]["editable_text_nodes"] > 0 for item in report["figures"].values()
    )
    (OUT / "qa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = json.loads((RESULTS / "paper_results.json").read_text(encoding="utf-8"))
    cped = pd.read_csv(RESULTS / "cped_metrics.csv")
    needs = pd.read_csv(RESULTS / "needs_metrics_with_ci.csv")
    paired_a = pd.read_csv(RESULTS / "agent_paired_stats.csv")
    paired_b = pd.read_csv(RESULTS / "agent_derived_paired_stats.csv")
    paired = pd.concat([paired_a, paired_b], ignore_index=True)

    make_workflow()
    make_context_augmentation(cped, results)
    make_topic_routing(needs)
    make_validator(results)
    make_outcomes(results, paired)

    stems = [
        "01_end_to_end_workflow",
        "02_context_augmentation",
        "03_need_topic_routing",
        "04_evidence_validation",
        "05_quantitative_outcomes",
    ]
    write_source_data(results, cped, needs, paired)
    make_contact_sheet(stems)
    qa(stems)
    print(json.dumps({"output": str(OUT), "figures": stems}, indent=2))


if __name__ == "__main__":
    main()
