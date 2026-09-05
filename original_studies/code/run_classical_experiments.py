#!/usr/bin/env python3
"""Run reproducible CPED emotion and CounselChat need-topic experiments.

The script uses only public dataset files supplied by the cited CPED GitHub
repository and the user-specified Kaggle archive. It writes all predictions,
metrics, confusion matrices, and provenance metadata needed for audit.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
CPED_DIR = ROOT / "work" / "data" / "CPED" / "data" / "CPED"
COUNSEL_PATH = (
    ROOT
    / "work"
    / "data"
    / "mental_health_kaggle"
    / "archive"
    / "counselchat-data.csv"
)
OUT = ROOT / "results"
RANDOM_SEED = 42
EMOTION_ORDER = [
    "happy",
    "grateful",
    "relaxed",
    "positive-other",
    "neutral",
    "anger",
    "sadness",
    "fear",
    "depress",
    "disgust",
    "astonished",
    "worried",
    "negative-other",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def build_context_text(df: pd.DataFrame, window: int = 3) -> pd.Series:
    """Prepend up to `window` prior utterances within the same dialogue."""
    values = []
    for _, group in df.groupby("Dialogue_ID", sort=False):
        history: list[tuple[str, str]] = []
        for row in group.itertuples(index=False):
            pieces = []
            for speaker, utterance in history[-window:]:
                relation = "SAME" if speaker == str(row.Speaker) else "OTHER"
                pieces.append(f"[PREV_{relation}] {utterance}")
            pieces.append(f"[CURRENT] {row.Utterance}")
            values.append((row.IndexOrder, " ".join(pieces)))
            history.append((str(row.Speaker), str(row.Utterance)))
    values.sort(key=lambda item: item[0])
    return pd.Series([text for _, text in values], index=df.index, dtype="string")


def prepare_cped_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(CPED_DIR / f"{name}_split.csv")
    df = df.reset_index(drop=True)
    df["IndexOrder"] = np.arange(len(df))
    df["Utterance"] = df["Utterance"].fillna("").astype(str)
    df["Speaker"] = df["Speaker"].fillna("UNKNOWN").astype(str)
    df["text_utterance"] = "[CURRENT] " + df["Utterance"]
    df["text_context3"] = build_context_text(df, window=3)
    return df


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str,
    n_boot: int = 400,
    seed: int = RANDOM_SEED,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if metric == "macro_f1":
            scores[i] = f1_score(
                y_true[idx], y_pred[idx], average="macro", zero_division=0
            )
        elif metric == "accuracy":
            scores[i] = accuracy_score(y_true[idx], y_pred[idx])
        else:
            raise ValueError(metric)
    return (
        float(np.mean(scores)),
        float(np.quantile(scores, 0.025)),
        float(np.quantile(scores, 0.975)),
    )


def paired_bootstrap_delta(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    n_boot: int = 400,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Return macro-F1(B)-macro-F1(A), percentile CI, and one-sided p."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        a = f1_score(y_true[idx], pred_a[idx], average="macro", zero_division=0)
        b = f1_score(y_true[idx], pred_b[idx], average="macro", zero_division=0)
        deltas[i] = b - a
    observed = f1_score(
        y_true, pred_b, average="macro", zero_division=0
    ) - f1_score(y_true, pred_a, average="macro", zero_division=0)
    return {
        "comparison": "Context-3 minus utterance-only",
        "metric": "macro_f1",
        "observed_delta": float(observed),
        "bootstrap_mean_delta": float(deltas.mean()),
        "ci_2.5": float(np.quantile(deltas, 0.025)),
        "ci_97.5": float(np.quantile(deltas, 0.975)),
        "one_sided_p_delta_le_0": float((np.sum(deltas <= 0) + 1) / (n_boot + 1)),
        "bootstrap_replicates": n_boot,
        "seed": seed,
    }


def run_cped() -> dict[str, object]:
    started = time.time()
    splits = {name: prepare_cped_split(name) for name in ("train", "valid", "test")}
    labels = sorted(set().union(*(set(x["Emotion"]) for x in splits.values())))
    train = splits["train"]
    valid = splits["valid"]
    test = splits["test"]
    train_valid = pd.concat([train, valid], ignore_index=True)

    all_metrics: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    prediction_frame = test[
        ["TV_ID", "Dialogue_ID", "Utterance_ID", "Speaker", "Utterance", "Emotion"]
    ].copy()
    prediction_frame = prediction_frame.rename(columns={"Emotion": "true_emotion"})
    confusion_tables: dict[str, pd.DataFrame] = {}
    model_predictions: dict[str, np.ndarray] = {}
    tuning_rows: list[dict[str, object]] = []

    majority = Counter(train_valid["Emotion"]).most_common(1)[0][0]
    dummy_pred = np.full(len(test), majority, dtype=object)
    model_predictions["Majority baseline"] = dummy_pred

    variants = [
        ("Utterance-only TF-IDF + LinearSVC", "text_utterance"),
        ("Context-3 TF-IDF + LinearSVC", "text_context3"),
    ]
    for model_name, text_col in variants:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 4),
            min_df=3,
            max_features=120_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        x_train = vectorizer.fit_transform(train[text_col])
        x_valid = vectorizer.transform(valid[text_col])
        best_c = None
        best_score = -np.inf
        for c in (0.25, 0.5, 1.0):
            clf = LinearSVC(
                C=c,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                max_iter=5_000,
            )
            clf.fit(x_train, train["Emotion"])
            val_pred = clf.predict(x_valid)
            score = f1_score(
                valid["Emotion"], val_pred, average="macro", zero_division=0
            )
            tuning_rows.append(
                {
                    "task": "CPED emotion",
                    "model": model_name,
                    "parameter": "C",
                    "value": c,
                    "validation_macro_f1": score,
                }
            )
            if score > best_score:
                best_score, best_c = score, c

        final_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 4),
            min_df=3,
            max_features=120_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        x_train_valid = final_vectorizer.fit_transform(train_valid[text_col])
        x_test = final_vectorizer.transform(test[text_col])
        final_clf = LinearSVC(
            C=float(best_c),
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=5_000,
        )
        final_clf.fit(x_train_valid, train_valid["Emotion"])
        pred = final_clf.predict(x_test)
        model_predictions[model_name] = pred

    y_true = test["Emotion"].to_numpy()
    for model_name, pred in model_predictions.items():
        macro = f1_score(y_true, pred, average="macro", zero_division=0)
        weighted = f1_score(y_true, pred, average="weighted", zero_division=0)
        acc = accuracy_score(y_true, pred)
        p_macro = precision_score(y_true, pred, average="macro", zero_division=0)
        r_macro = recall_score(y_true, pred, average="macro", zero_division=0)
        _, macro_lo, macro_hi = bootstrap_metric(
            y_true, pred, metric="macro_f1", n_boot=400
        )
        _, acc_lo, acc_hi = bootstrap_metric(
            y_true, pred, metric="accuracy", n_boot=400, seed=RANDOM_SEED + 1
        )
        all_metrics.append(
            {
                "task": "CPED 13-class emotion recognition",
                "model": model_name,
                "test_n": len(test),
                "accuracy": acc,
                "accuracy_ci_low": acc_lo,
                "accuracy_ci_high": acc_hi,
                "macro_precision": p_macro,
                "macro_recall": r_macro,
                "macro_f1": macro,
                "macro_f1_ci_low": macro_lo,
                "macro_f1_ci_high": macro_hi,
                "weighted_f1": weighted,
                "bootstrap_replicates": 400,
                "seed": RANDOM_SEED,
            }
        )
        report = classification_report(
            y_true,
            pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        for label in labels:
            row = report[label]
            per_class_rows.append(
                {
                    "model": model_name,
                    "emotion": label,
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1-score"],
                    "support": int(row["support"]),
                }
            )
        prediction_frame["pred_" + text_col.replace("text_", "")] = pred
        cm = confusion_matrix(y_true, pred, labels=EMOTION_ORDER)
        confusion_tables[model_name] = pd.DataFrame(
            cm, index=EMOTION_ORDER, columns=EMOTION_ORDER
        )

    pd.DataFrame(all_metrics).to_csv(OUT / "cped_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(OUT / "cped_per_class.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(OUT / "cped_tuning.csv", index=False)
    prediction_frame.to_csv(OUT / "cped_predictions.csv", index=False)
    for model_name, cm in confusion_tables.items():
        slug = (
            model_name.lower()
            .replace(" ", "_")
            .replace("+", "plus")
            .replace("-", "_")
        )
        cm.to_csv(OUT / f"cped_confusion_{slug}.csv", index_label="true_emotion")

    delta = paired_bootstrap_delta(
        y_true,
        model_predictions["Utterance-only TF-IDF + LinearSVC"],
        model_predictions["Context-3 TF-IDF + LinearSVC"],
        n_boot=400,
    )
    pd.DataFrame([delta]).to_csv(OUT / "cped_paired_bootstrap.csv", index=False)
    return {
        "elapsed_seconds": time.time() - started,
        "train_n": len(train),
        "valid_n": len(valid),
        "test_n": len(test),
        "labels": labels,
        "majority_label": majority,
        "paired_bootstrap": delta,
    }


def normalize_topic(topic: str) -> str:
    return " ".join(topic.strip().lower().replace("&", "and").split())


def aggregate_counsel_questions(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for question_id, group in raw.groupby("questionID", sort=False):
        topic_set: set[str] = set()
        for cell in group["topics"].dropna().astype(str):
            topic_set.update(
                normalize_topic(part)
                for part in cell.split(",")
                if normalize_topic(part)
            )
        rows.append(
            {
                "questionID": str(question_id),
                "questionTitle": str(group["questionTitle"].iloc[0]),
                "questionText": str(group["questionText"].iloc[0]),
                "topics_all": sorted(topic_set),
                "answer_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def multilabel_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, model: str
) -> dict[str, object]:
    return {
        "task": "CounselChat multi-label need-topic detection",
        "model": model,
        "test_n": len(y_true),
        "micro_precision": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "micro_recall": recall_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "exact_match_accuracy": accuracy_score(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "seed": RANDOM_SEED,
    }


def run_counsel() -> dict[str, object]:
    started = time.time()
    raw = pd.read_csv(COUNSEL_PATH)
    questions = aggregate_counsel_questions(raw)
    counts = Counter(topic for topics in questions["topics_all"] for topic in topics)
    selected_topics = sorted(topic for topic, count in counts.items() if count >= 20)
    questions["topics_selected"] = questions["topics_all"].apply(
        lambda xs: sorted(set(xs).intersection(selected_topics))
    )
    questions = questions[questions["topics_selected"].str.len() > 0].reset_index(
        drop=True
    )
    questions["primary_topic"] = questions["topics_selected"].apply(
        lambda xs: max(xs, key=lambda x: (counts[x], x))
    )
    questions["text"] = (
        questions["questionTitle"].fillna("")
        + " [SEP] "
        + questions["questionText"].fillna("")
    )

    train_val, test = train_test_split(
        questions,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=questions["primary_topic"],
    )
    train, valid = train_test_split(
        train_val,
        test_size=0.1875,
        random_state=RANDOM_SEED,
        stratify=train_val["primary_topic"],
    )
    mlb = MultiLabelBinarizer(classes=selected_topics)
    y_train = mlb.fit_transform(train["topics_selected"])
    y_valid = mlb.transform(valid["topics_selected"])
    y_test = mlb.transform(test["topics_selected"])

    metric_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    pred_frame = test[
        [
            "questionID",
            "questionTitle",
            "questionText",
            "topics_selected",
            "primary_topic",
        ]
    ].copy()
    pred_frame["topics_true"] = pred_frame["topics_selected"].apply(
        lambda xs: "|".join(xs)
    )
    pred_frame = pred_frame.drop(columns=["topics_selected"])

    prevalence = y_train.mean(axis=0)
    most_prevalent = int(np.argmax(prevalence))
    dummy_pred = np.zeros_like(y_test)
    dummy_pred[:, most_prevalent] = 1
    metric_rows.append(multilabel_metrics(y_test, dummy_pred, "Most-prevalent label"))
    pred_frame["pred_most_prevalent"] = [
        "|".join(mlb.inverse_transform(row.reshape(1, -1))[0]) for row in dummy_pred
    ]

    variants = [
        (
            "Word TF-IDF + One-vs-Rest LinearSVC",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_features=35_000,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
        (
            "Character TF-IDF + One-vs-Rest LinearSVC",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=45_000,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
    ]
    saved_decisions: dict[str, np.ndarray] = {}
    for model_name, vectorizer in variants:
        x_train = vectorizer.fit_transform(train["text"])
        x_valid = vectorizer.transform(valid["text"])
        best = None
        best_score = -np.inf
        for c in (0.25, 0.5, 1.0):
            clf = OneVsRestClassifier(
                LinearSVC(
                    C=c,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    max_iter=5_000,
                )
            )
            clf.fit(x_train, y_train)
            decision = clf.decision_function(x_valid)
            for threshold in (-0.30, -0.15, 0.0, 0.15):
                pred = (decision >= threshold).astype(int)
                empty = pred.sum(axis=1) == 0
                pred[empty, np.argmax(decision[empty], axis=1)] = 1
                score = f1_score(
                    y_valid, pred, average="micro", zero_division=0
                )
                tuning_rows.append(
                    {
                        "task": "CounselChat need topics",
                        "model": model_name,
                        "C": c,
                        "threshold": threshold,
                        "validation_micro_f1": score,
                    }
                )
                if score > best_score:
                    best_score = score
                    best = (c, threshold)

        train_valid = pd.concat([train, valid], ignore_index=True)
        y_train_valid = mlb.transform(train_valid["topics_selected"])
        final_vectorizer = vectorizer.set_params()
        x_train_valid = final_vectorizer.fit_transform(train_valid["text"])
        x_test = final_vectorizer.transform(test["text"])
        final_clf = OneVsRestClassifier(
            LinearSVC(
                C=float(best[0]),
                class_weight="balanced",
                random_state=RANDOM_SEED,
                max_iter=5_000,
            )
        )
        final_clf.fit(x_train_valid, y_train_valid)
        decision = final_clf.decision_function(x_test)
        pred = (decision >= float(best[1])).astype(int)
        empty = pred.sum(axis=1) == 0
        pred[empty, np.argmax(decision[empty], axis=1)] = 1
        saved_decisions[model_name] = decision
        metric_rows.append(multilabel_metrics(y_test, pred, model_name))
        report = classification_report(
            y_test,
            pred,
            target_names=selected_topics,
            output_dict=True,
            zero_division=0,
        )
        for topic in selected_topics:
            row = report[topic]
            per_class_rows.append(
                {
                    "model": model_name,
                    "topic": topic,
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1-score"],
                    "support": int(row["support"]),
                }
            )
        slug = "word" if model_name.startswith("Word") else "character"
        pred_frame[f"pred_{slug}"] = [
            "|".join(mlb.inverse_transform(row.reshape(1, -1))[0]) for row in pred
        ]
        pred_frame[f"top_score_{slug}"] = decision.max(axis=1)

    split_rows = []
    for split_name, frame in (("train", train), ("valid", valid), ("test", test)):
        for row in frame.itertuples(index=False):
            split_rows.append(
                {
                    "questionID": row.questionID,
                    "split": split_name,
                    "primary_topic": row.primary_topic,
                    "topics": "|".join(row.topics_selected),
                }
            )
    pd.DataFrame(metric_rows).to_csv(OUT / "needs_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(OUT / "needs_per_class.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(OUT / "needs_tuning.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUT / "needs_split.csv", index=False)
    pred_frame.to_csv(OUT / "needs_predictions.csv", index=False)
    pd.DataFrame(
        [{"topic": topic, "unique_question_count": counts[topic]} for topic in selected_topics]
    ).to_csv(OUT / "needs_topic_counts.csv", index=False)

    # Save the final test records for the later same-split agent pilot.
    test_records = test[
        ["questionID", "questionTitle", "questionText", "topics_selected", "primary_topic"]
    ].to_dict("records")
    train_records = train_val[
        ["questionID", "questionTitle", "questionText", "topics_selected", "primary_topic"]
    ].to_dict("records")
    atomic_write_json(
        OUT / "agent_data_split.json",
        {
            "selected_topics": selected_topics,
            "train_records": train_records,
            "test_records": test_records,
        },
    )
    return {
        "elapsed_seconds": time.time() - started,
        "raw_answer_rows": len(raw),
        "unique_questions_before_filter": len(aggregate_counsel_questions(raw)),
        "questions_after_topic_frequency_filter": len(questions),
        "train_n": len(train),
        "valid_n": len(valid),
        "test_n": len(test),
        "selected_topics": selected_topics,
        "label_frequency_threshold": 20,
    }


def write_manifest(cped_summary: dict[str, object], counsel_summary: dict[str, object]) -> None:
    cped_commit = (
        __import__("subprocess")
        .check_output(
            ["git", "-C", str(ROOT / "work" / "data" / "CPED"), "rev-parse", "HEAD"],
            text=True,
        )
        .strip()
    )
    rows = []
    for split in ("train", "valid", "test"):
        path = CPED_DIR / f"{split}_split.csv"
        rows.append(
            {
                "dataset": "CPED",
                "file": path.name,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "source_url": "https://github.com/scutcyr/CPED",
                "version": cped_commit,
            }
        )
    rows.append(
        {
            "dataset": "Mental Health Counseling Conversations (Kaggle)",
            "file": COUNSEL_PATH.name,
            "sha256": sha256(COUNSEL_PATH),
            "bytes": COUNSEL_PATH.stat().st_size,
            "source_url": "https://www.kaggle.com/datasets/melissamonfared/mental-health-counseling-conversations-k",
            "version": "downloaded 2026-06-27",
        }
    )
    pd.DataFrame(rows).to_csv(OUT / "dataset_manifest.csv", index=False)
    atomic_write_json(
        OUT / "run_metadata.json",
        {
            "random_seed": RANDOM_SEED,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "cped": cped_summary,
            "counsel": counsel_summary,
        },
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cped_summary = run_cped()
    counsel_summary = run_counsel()
    write_manifest(cped_summary, counsel_summary)
    print(json.dumps({"cped": cped_summary, "counsel": counsel_summary}, indent=2))


if __name__ == "__main__":
    main()
