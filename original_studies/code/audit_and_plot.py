#!/usr/bin/env python3
"""Audit experiment artifacts, derive statistics, and create Python figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score


# Mandatory editable-SVG settings from the selected Python figure workflow.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results"
FIG = ROOT / "figures"
SEED = 42
BASELINE = "Single-pass LLM"
AGENT = "Staged affect-sensitive agent"
COLORS = {
    "baseline": "#7884B4",
    "baseline_dark": "#484878",
    "agent": "#E4A9BD",
    "agent_dark": "#A94B6A",
    "neutral": "#B8B8B8",
    "accent": "#2E9E44",
}


def parse_labels(cell: object) -> set[str]:
    text = str(cell or "")
    return {x for x in text.split("|") if x}


def multilabel_arrays(true_cells: pd.Series, pred_cells: pd.Series, labels: list[str]):
    index = {label: i for i, label in enumerate(labels)}
    true = np.zeros((len(true_cells), len(labels)), dtype=int)
    pred = np.zeros_like(true)
    for row_idx, (truth, prediction) in enumerate(zip(true_cells, pred_cells)):
        for label in parse_labels(truth):
            if label in index:
                true[row_idx, index[label]] = 1
        for label in parse_labels(prediction):
            if label in index:
                pred[row_idx, index[label]] = 1
    return true, pred


def bootstrap_multilabel(
    y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 2_000
) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    micro = np.empty(n_boot)
    macro = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        micro[i] = f1_score(
            y_true[idx], y_pred[idx], average="micro", zero_division=0
        )
        macro[i] = f1_score(
            y_true[idx], y_pred[idx], average="macro", zero_division=0
        )
    return {
        "micro_f1_ci_low": float(np.quantile(micro, 0.025)),
        "micro_f1_ci_high": float(np.quantile(micro, 0.975)),
        "macro_f1_ci_low": float(np.quantile(macro, 0.025)),
        "macro_f1_ci_high": float(np.quantile(macro, 0.975)),
        "bootstrap_replicates": n_boot,
        "seed": SEED,
    }


def paired_delta(a: np.ndarray, b: np.ndarray, n_boot: int = 2_000):
    rng = np.random.default_rng(SEED)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        samples[i] = np.mean(b[idx] - a[idx])
    try:
        statistic, p_value = wilcoxon(b, a, zero_method="zsplit")
    except ValueError:
        statistic, p_value = np.nan, np.nan
    return {
        "delta_agent_minus_baseline": float(np.mean(b - a)),
        "bootstrap_ci_low": float(np.quantile(samples, 0.025)),
        "bootstrap_ci_high": float(np.quantile(samples, 0.975)),
        "bootstrap_replicates": n_boot,
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_two_sided_p": float(p_value),
        "n_paired_cases": len(a),
        "seed": SEED,
    }


def derive_metrics() -> dict[str, object]:
    needs_metrics = pd.read_csv(EXP / "needs_metrics.csv")
    needs_pred = pd.read_csv(EXP / "needs_predictions.csv")
    labels = pd.read_csv(EXP / "needs_topic_counts.csv")["topic"].tolist()
    pred_columns = {
        "Most-prevalent label": "pred_most_prevalent",
        "Word TF-IDF + One-vs-Rest LinearSVC": "pred_word",
        "Character TF-IDF + One-vs-Rest LinearSVC": "pred_character",
    }
    ci_rows = []
    for model, column in pred_columns.items():
        y_true, y_pred = multilabel_arrays(
            needs_pred["topics_true"], needs_pred[column], labels
        )
        row = {"model": model}
        row.update(bootstrap_multilabel(y_true, y_pred))
        ci_rows.append(row)
    needs_ci = pd.DataFrame(ci_rows)
    needs_ci.to_csv(EXP / "needs_bootstrap_ci.csv", index=False)
    needs_metrics = needs_metrics.merge(needs_ci, on="model", how="left")
    needs_metrics.to_csv(EXP / "needs_metrics_with_ci.csv", index=False)

    agent_cases = pd.read_csv(EXP / "agent_case_metrics.csv")
    agent_cases["full_engineering_compliance"] = (
        (agent_cases["requirement_count"] > 0)
        & (agent_cases["schema_completeness"] == 1)
        & (agent_cases["srs_form_rate"] == 1)
        & (agent_cases["traceability_coverage"] == 1)
    ).astype(float)
    agent_cases["engineering_control_score"] = agent_cases[
        ["schema_completeness", "srs_form_rate", "traceability_coverage"]
    ].mean(axis=1)
    agent_cases.to_csv(EXP / "agent_case_metrics_derived.csv", index=False)

    derived_rows = []
    for variant, group in agent_cases.groupby("variant", sort=False):
        for metric in ("full_engineering_compliance", "engineering_control_score"):
            derived_rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "n_cases": len(group),
                    "mean": group[metric].mean(),
                    "std": group[metric].std(ddof=1),
                    "median": group[metric].median(),
                }
            )
    pd.DataFrame(derived_rows).to_csv(
        EXP / "agent_derived_summary.csv", index=False
    )

    pivot = agent_cases.pivot(
        index="case_id",
        columns="variant",
        values=["full_engineering_compliance", "engineering_control_score"],
    )
    paired_rows = []
    for metric in ("full_engineering_compliance", "engineering_control_score"):
        row = {"metric": metric}
        row.update(
            paired_delta(
                pivot[metric][BASELINE].to_numpy(float),
                pivot[metric][AGENT].to_numpy(float),
            )
        )
        paired_rows.append(row)
    pd.DataFrame(paired_rows).to_csv(
        EXP / "agent_derived_paired_stats.csv", index=False
    )
    return {
        "needs_metrics": needs_metrics,
        "agent_cases": agent_cases,
        "agent_derived_paired": pd.DataFrame(paired_rows),
    }


def audit(derived: dict[str, object]) -> dict[str, object]:
    cped_pred = pd.read_csv(EXP / "cped_predictions.csv")
    needs_split = pd.read_csv(EXP / "needs_split.csv")
    agent_final = pd.read_csv(EXP / "agent_cases.csv")
    dev1 = pd.read_csv(EXP / "prompt_development" / "run1" / "agent_cases.csv")
    dev2 = pd.read_csv(EXP / "prompt_development" / "run2" / "agent_cases.csv")

    split_sets = {
        split: set(group["questionID"].astype(str))
        for split, group in needs_split.groupby("split")
    }
    needs_overlap = {
        "train_valid": len(split_sets["train"].intersection(split_sets["valid"])),
        "train_test": len(split_sets["train"].intersection(split_sets["test"])),
        "valid_test": len(split_sets["valid"].intersection(split_sets["test"])),
    }
    final_ids = set(agent_final["questionID"].astype(str))
    dev1_ids = set(dev1["questionID"].astype(str))
    dev2_ids = set(dev2["questionID"].astype(str))
    checks = {
        "cped_prediction_rows_equal_official_test": len(cped_pred) == 27_438,
        "cped_has_13_true_classes": cped_pred["true_emotion"].nunique() == 13,
        "needs_split_question_ids_unique": not needs_split["questionID"].duplicated().any(),
        "needs_no_cross_split_overlap": all(v == 0 for v in needs_overlap.values()),
        "agent_final_has_30_unique_cases": (
            len(agent_final) == 30 and agent_final["questionID"].nunique() == 30
        ),
        "agent_dev_sets_disjoint": len(dev1_ids.intersection(dev2_ids)) == 0,
        "agent_final_disjoint_from_dev": (
            len(final_ids.intersection(dev1_ids.union(dev2_ids))) == 0
        ),
        "agent_final_metrics_two_rows_per_case": (
            derived["agent_cases"].groupby("case_id").size().eq(2).all()
        ),
        "agent_final_both_variants_present": set(
            derived["agent_cases"]["variant"]
        )
        == {BASELINE, AGENT},
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "needs_overlap_counts": needs_overlap,
        "counts": {
            "cped_test_predictions": len(cped_pred),
            "needs_unique_split_rows": len(needs_split),
            "agent_prompt_development_cases": len(dev1_ids.union(dev2_ids)),
            "agent_final_cases": len(final_ids),
        },
    }
    (EXP / "quality_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if not result["all_checks_pass"]:
        raise RuntimeError(f"Experiment audit failed: {result}")
    return result


def add_panel_label(ax, label: str):
    ax.text(
        -0.16,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
    )


def percent_axis(ax, top=1.0):
    ax.set_ylim(0, top)
    ticks = ax.get_yticks()
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{x * 100:.0f}" for x in ticks])
    ax.set_ylabel("Score (%)")


def save_figure(fig, base: Path):
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_overview_figure(derived: dict[str, object]) -> None:
    cped = pd.read_csv(EXP / "cped_metrics.csv")
    needs = derived["needs_metrics"]
    agent = derived["agent_cases"]
    agent_stats = pd.read_csv(EXP / "agent_paired_stats.csv")
    derived_stats = derived["agent_derived_paired"]

    source_rows = []
    for _, row in cped.iterrows():
        source_rows.append(
            {
                "panel": "a",
                "task": "CPED emotion",
                "method": row["model"],
                "metric": "macro_f1",
                "value": row["macro_f1"],
                "ci_low": row["macro_f1_ci_low"],
                "ci_high": row["macro_f1_ci_high"],
            }
        )
    for _, row in needs.iterrows():
        for metric in ("micro_f1", "macro_f1"):
            source_rows.append(
                {
                    "panel": "b",
                    "task": "CounselChat need topics",
                    "method": row["model"],
                    "metric": metric,
                    "value": row[metric],
                    "ci_low": row[f"{metric}_ci_low"],
                    "ci_high": row[f"{metric}_ci_high"],
                }
            )
    agent_metric_map = {
        "srs_form_rate": "SRS form",
        "traceability_coverage": "Traceability",
        "full_engineering_compliance": "Full compliance",
        "topic_alignment_f1": "Topic alignment",
    }
    for metric, label in agent_metric_map.items():
        for variant, group in agent.groupby("variant"):
            source_rows.append(
                {
                    "panel": "c",
                    "task": "Agent comparison",
                    "method": variant,
                    "metric": label,
                    "value": group[metric].mean(),
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            )
    source = pd.DataFrame(source_rows)
    source.to_csv(EXP / "figure1_source_data.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 122 / 25.4))
    ax = axes[0, 0]
    labels = ["Majority", "Utterance", "Context-3"]
    values = cped["macro_f1"].to_numpy()
    errors = np.vstack(
        [
            values - cped["macro_f1_ci_low"].to_numpy(),
            cped["macro_f1_ci_high"].to_numpy() - values,
        ]
    )
    bars = ax.bar(
        np.arange(3),
        values,
        yerr=errors,
        capsize=2,
        color=[COLORS["neutral"], COLORS["baseline"], COLORS["agent"]],
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.set_xticks(np.arange(3), labels)
    percent_axis(ax, top=0.18)
    ax.set_title("CPED 13-class emotion recognition")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.009,
            f"{value * 100:.1f}",
            ha="center",
            fontsize=6.2,
        )
    add_panel_label(ax, "a")

    ax = axes[0, 1]
    x = np.arange(3)
    width = 0.34
    micro = needs["micro_f1"].to_numpy()
    macro = needs["macro_f1"].to_numpy()
    micro_err = np.vstack(
        [
            micro - needs["micro_f1_ci_low"].to_numpy(),
            needs["micro_f1_ci_high"].to_numpy() - micro,
        ]
    )
    macro_err = np.vstack(
        [
            macro - needs["macro_f1_ci_low"].to_numpy(),
            needs["macro_f1_ci_high"].to_numpy() - macro,
        ]
    )
    ax.bar(
        x - width / 2,
        micro,
        width,
        yerr=micro_err,
        capsize=2,
        label="Micro-F1",
        color=COLORS["baseline_dark"],
    )
    ax.bar(
        x + width / 2,
        macro,
        width,
        yerr=macro_err,
        capsize=2,
        label="Macro-F1",
        color=COLORS["agent"],
    )
    ax.set_xticks(x, ["Most-frequent", "Word", "Character"])
    percent_axis(ax, top=0.72)
    ax.set_title("CounselChat need-topic detection")
    ax.legend(loc="upper left", ncol=2, fontsize=6)
    add_panel_label(ax, "b")

    ax = axes[1, 0]
    metrics = list(agent_metric_map)
    labels = list(agent_metric_map.values())
    base_vals = [
        agent.loc[agent["variant"] == BASELINE, metric].mean() for metric in metrics
    ]
    agent_vals = [
        agent.loc[agent["variant"] == AGENT, metric].mean() for metric in metrics
    ]
    x = np.arange(len(metrics))
    ax.bar(
        x - width / 2,
        base_vals,
        width,
        label="Single-pass LLM",
        color=COLORS["baseline"],
    )
    ax.bar(
        x + width / 2,
        agent_vals,
        width,
        label="Staged agent",
        color=COLORS["agent_dark"],
    )
    ax.set_xticks(x, labels, rotation=15, ha="right")
    percent_axis(ax, top=1.08)
    ax.set_title("Held-out agent evaluation (n=30 paired)")
    ax.legend(loc="upper right", fontsize=6)
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    stats = pd.concat(
        [
            agent_stats[
                agent_stats["metric"].isin(
                    ["srs_form_rate", "traceability_coverage", "topic_alignment_f1"]
                )
            ],
            derived_stats[
                derived_stats["metric"].eq("full_engineering_compliance")
            ],
        ],
        ignore_index=True,
    )
    order = [
        "srs_form_rate",
        "traceability_coverage",
        "full_engineering_compliance",
        "topic_alignment_f1",
    ]
    stats = stats.set_index("metric").loc[order].reset_index()
    display = [
        "SRS form",
        "Traceability",
        "Full compliance",
        "Topic alignment",
    ]
    y = np.arange(len(stats))[::-1]
    estimate = stats["delta_agent_minus_baseline"].to_numpy()
    low = stats["bootstrap_ci_low"].to_numpy()
    high = stats["bootstrap_ci_high"].to_numpy()
    for yi, est, lo, hi in zip(y, estimate, low, high):
        color = COLORS["accent"] if lo > 0 else COLORS["baseline_dark"]
        ax.plot([lo * 100, hi * 100], [yi, yi], color=color, lw=1.5)
        ax.plot(est * 100, yi, "o", color=color, ms=4)
    ax.axvline(0, color="#777777", lw=0.8, ls="--")
    ax.set_yticks(y, display)
    ax.set_xlabel("Agent − baseline (percentage points, 95% bootstrap CI)")
    ax.set_title("Paired effect estimates")
    add_panel_label(ax, "d")

    fig.suptitle(
        "Python experiments: context, need detection, and engineering controls",
        fontsize=8,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(pad=1.2, h_pad=2.0, w_pad=2.2)
    save_figure(fig, FIG / "experiment_results_overview")


def make_confusion_figure() -> None:
    path = EXP / "cped_confusion_context_3_tf_idf_plus_linearsvc.csv"
    cm = pd.read_csv(path, index_col=0)
    row_sum = cm.sum(axis=1).replace(0, np.nan)
    normalized = cm.div(row_sum, axis=0).fillna(0)
    normalized.reset_index().to_csv(
        EXP / "figure2_confusion_source_data.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(110 / 25.4, 105 / 25.4))
    im = ax.imshow(normalized.to_numpy(), cmap="Blues", vmin=0, vmax=0.65)
    ax.set_xticks(np.arange(len(cm.columns)), cm.columns, rotation=50, ha="right")
    ax.set_yticks(np.arange(len(cm.index)), cm.index)
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_title("Context-3 CPED confusion matrix (row-normalized)")
    for i in range(len(cm.index)):
        for j in range(len(cm.columns)):
            value = normalized.iat[i, j]
            if value >= 0.10:
                ax.text(
                    j,
                    i,
                    f"{value * 100:.0f}",
                    ha="center",
                    va="center",
                    fontsize=4.8,
                    color="white" if value > 0.35 else "#222222",
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Within-class predictions (%)")
    cbar.set_ticks([0, 0.2, 0.4, 0.6])
    cbar.set_ticklabels(["0", "20", "40", "60"])
    fig.tight_layout(pad=1.0)
    save_figure(fig, FIG / "cped_context_confusion")


def figure_qa() -> dict[str, object]:
    qa = {}
    for stem in ("experiment_results_overview", "cped_context_confusion"):
        formats = {}
        for suffix in (".svg", ".pdf", ".tiff", ".png"):
            path = FIG / f"{stem}{suffix}"
            entry = {"exists": path.exists(), "bytes": path.stat().st_size}
            if suffix in (".png", ".tiff"):
                with Image.open(path) as image:
                    entry["pixels"] = list(image.size)
                    entry["mode"] = image.mode
            if suffix == ".svg":
                text = path.read_text(encoding="utf-8")
                entry["editable_text_nodes"] = text.count("<text")
            formats[suffix] = entry
        qa[stem] = formats
    qa["all_primary_svg_have_text"] = all(
        item[".svg"]["editable_text_nodes"] > 0 for item in qa.values()
    )
    (EXP / "figure_qa.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    return qa


def write_paper_results(derived: dict[str, object], audit_result: dict[str, object]):
    cped = pd.read_csv(EXP / "cped_metrics.csv").set_index("model")
    cped_delta = pd.read_csv(EXP / "cped_paired_bootstrap.csv").iloc[0]
    needs = derived["needs_metrics"].set_index("model")
    agent = derived["agent_cases"]
    agent_paired = pd.read_csv(EXP / "agent_paired_stats.csv").set_index("metric")
    derived_paired = derived["agent_derived_paired"].set_index("metric")

    def mean(variant: str, metric: str) -> float:
        return float(agent.loc[agent["variant"] == variant, metric].mean())

    results = {
        "cped": {
            "train_n": 94_187,
            "valid_n": 11_137,
            "test_n": 27_438,
            "utterance_macro_f1": float(
                cped.loc["Utterance-only TF-IDF + LinearSVC", "macro_f1"]
            ),
            "context_macro_f1": float(
                cped.loc["Context-3 TF-IDF + LinearSVC", "macro_f1"]
            ),
            "context_accuracy": float(
                cped.loc["Context-3 TF-IDF + LinearSVC", "accuracy"]
            ),
            "macro_f1_delta": float(cped_delta["observed_delta"]),
            "delta_ci_low": float(cped_delta["ci_2.5"]),
            "delta_ci_high": float(cped_delta["ci_97.5"]),
            "one_sided_bootstrap_p": float(
                cped_delta["one_sided_p_delta_le_0"]
            ),
        },
        "needs": {
            "raw_answer_rows": 1_482,
            "unique_questions": 781,
            "filtered_questions": 738,
            "train_n": 479,
            "valid_n": 111,
            "test_n": 148,
            "labels": 18,
            "best_model": "Character TF-IDF + One-vs-Rest LinearSVC",
            "micro_f1": float(
                needs.loc[
                    "Character TF-IDF + One-vs-Rest LinearSVC", "micro_f1"
                ]
            ),
            "macro_f1": float(
                needs.loc[
                    "Character TF-IDF + One-vs-Rest LinearSVC", "macro_f1"
                ]
            ),
            "exact_match": float(
                needs.loc[
                    "Character TF-IDF + One-vs-Rest LinearSVC",
                    "exact_match_accuracy",
                ]
            ),
        },
        "agent": {
            "prompt_development_n": 60,
            "final_test_n": 30,
            "baseline_srs": mean(BASELINE, "srs_form_rate"),
            "agent_srs": mean(AGENT, "srs_form_rate"),
            "srs_delta": float(
                agent_paired.loc["srs_form_rate", "delta_agent_minus_baseline"]
            ),
            "srs_ci_low": float(
                agent_paired.loc["srs_form_rate", "bootstrap_ci_low"]
            ),
            "srs_ci_high": float(
                agent_paired.loc["srs_form_rate", "bootstrap_ci_high"]
            ),
            "srs_p": float(
                agent_paired.loc["srs_form_rate", "wilcoxon_two_sided_p"]
            ),
            "baseline_traceability": mean(BASELINE, "traceability_coverage"),
            "agent_traceability": mean(AGENT, "traceability_coverage"),
            "traceability_delta": float(
                agent_paired.loc[
                    "traceability_coverage", "delta_agent_minus_baseline"
                ]
            ),
            "traceability_ci_low": float(
                agent_paired.loc["traceability_coverage", "bootstrap_ci_low"]
            ),
            "traceability_ci_high": float(
                agent_paired.loc["traceability_coverage", "bootstrap_ci_high"]
            ),
            "traceability_p": float(
                agent_paired.loc[
                    "traceability_coverage", "wilcoxon_two_sided_p"
                ]
            ),
            "baseline_full_compliance": mean(
                BASELINE, "full_engineering_compliance"
            ),
            "agent_full_compliance": mean(AGENT, "full_engineering_compliance"),
            "full_compliance_delta": float(
                derived_paired.loc[
                    "full_engineering_compliance",
                    "delta_agent_minus_baseline",
                ]
            ),
            "full_compliance_ci_low": float(
                derived_paired.loc[
                    "full_engineering_compliance", "bootstrap_ci_low"
                ]
            ),
            "full_compliance_ci_high": float(
                derived_paired.loc[
                    "full_engineering_compliance", "bootstrap_ci_high"
                ]
            ),
            "full_compliance_p": float(
                derived_paired.loc[
                    "full_engineering_compliance", "wilcoxon_two_sided_p"
                ]
            ),
            "baseline_topic_alignment": mean(BASELINE, "topic_alignment_f1"),
            "agent_topic_alignment": mean(AGENT, "topic_alignment_f1"),
            "topic_alignment_p": float(
                agent_paired.loc[
                    "topic_alignment_f1", "wilcoxon_two_sided_p"
                ]
            ),
            "baseline_latency_s": mean(BASELINE, "latency_seconds"),
            "agent_latency_s": mean(AGENT, "latency_seconds"),
            "latency_delta_s": float(
                agent_paired.loc["latency_seconds", "delta_agent_minus_baseline"]
            ),
            "baseline_input_tokens": mean(BASELINE, "input_tokens"),
            "agent_input_tokens": mean(AGENT, "input_tokens"),
            "mean_validator_repairs": float(
                agent.loc[
                    agent["variant"] == AGENT, "validator_repair_count"
                ].mean()
            ),
        },
        "audit": audit_result,
    }
    (EXP / "paper_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    derived = derive_metrics()
    audit_result = audit(derived)
    make_overview_figure(derived)
    make_confusion_figure()
    qa = figure_qa()
    if not qa["all_primary_svg_have_text"]:
        raise RuntimeError("SVG editable-text QA failed")
    write_paper_results(derived, audit_result)
    print(json.dumps({"audit": audit_result, "figure_qa": qa}, indent=2))


if __name__ == "__main__":
    main()
