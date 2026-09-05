#!/usr/bin/env python3
"""Compare a single-pass LLM with a staged affect-sensitive RE agent.

Both variants use the same local Qwen2.5-0.5B-Instruct checkpoint. The staged
variant adds an affect-interpreter call, a held-out need-topic classifier,
retrieval from training questions, and a deterministic evidence validator.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results"
MODEL_PATH = Path(os.environ.get("QWEN_MODEL_PATH", str(ROOT.parent / "ablation_study" / "local_assets" / "Qwen2.5-0.5B-Instruct")))
OUT = EXP
SEED = 42
N_CASES = 30
DEVELOPMENT_OFFSET = 60
REQUIRED_FIELDS = [
    "requirement_id",
    "type",
    "statement",
    "source_quote",
    "rationale",
    "acceptance_criterion",
]
ALLOWED_TYPES = {
    "functional",
    "non-functional",
    "privacy",
    "safety",
    "usability",
    "accessibility",
    "data",
}
STOP_WORDS = {
    "the",
    "system",
    "shall",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "for",
    "in",
    "on",
    "with",
    "is",
    "be",
    "that",
    "this",
    "user",
    "users",
    "provide",
    "support",
    "allow",
    "enable",
    "ensure",
    "their",
    "when",
    "from",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_space(text: object) -> str:
    return " ".join(str(text or "").replace("\u00a0", " ").split()).strip()


def extract_json(text: str) -> Any | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    starts = [i for i, c in enumerate(cleaned) if c in "[{"]
    for start in starts:
        try:
            value, _ = decoder.raw_decode(cleaned[start:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def normalize_requirements(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("requirements"), list):
        items = parsed["requirements"]
    elif isinstance(parsed, dict) and all(k in parsed for k in REQUIRED_FIELDS):
        items = [parsed]
    else:
        items = []
    return [x for x in items if isinstance(x, dict)][:1]


def content_words(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z-]{2,}", text.lower())
        if token not in STOP_WORDS
    }


def exact_quote_in_dialogue(quote: object, dialogue: str) -> bool:
    q = normalize_space(quote).lower().strip("\"'")
    d = normalize_space(dialogue).lower()
    return len(q) >= 5 and q in d


def requirement_scores(requirements: list[dict[str, Any]], dialogue: str) -> dict[str, float]:
    if not requirements:
        return {
            "requirement_count": 0.0,
            "schema_completeness": 0.0,
            "srs_form_rate": 0.0,
            "traceability_coverage": 0.0,
            "lexical_evidence_validity": 0.0,
        }
    field_scores = []
    srs_scores = []
    trace_scores = []
    evidence_scores = []
    for req in requirements:
        present = sum(bool(normalize_space(req.get(k, ""))) for k in REQUIRED_FIELDS)
        field_scores.append(present / len(REQUIRED_FIELDS))
        statement = normalize_space(req.get("statement", ""))
        source_quote = normalize_space(req.get("source_quote", ""))
        srs_scores.append(float(statement.lower().startswith("the system shall")))
        traced = exact_quote_in_dialogue(source_quote, dialogue)
        trace_scores.append(float(traced))
        overlap = content_words(statement).intersection(content_words(source_quote))
        evidence_scores.append(float(traced and bool(overlap)))
    return {
        "requirement_count": float(len(requirements)),
        "schema_completeness": float(np.mean(field_scores)),
        "srs_form_rate": float(np.mean(srs_scores)),
        "traceability_coverage": float(np.mean(trace_scores)),
        "lexical_evidence_validity": float(np.mean(evidence_scores)),
    }


def split_evidence(dialogue: str, max_items: int = 8) -> list[dict[str, str]]:
    chunks = [
        normalize_space(x)
        for x in re.split(r"(?<=[.!?])\s+|\n+", dialogue)
        if len(normalize_space(x)) >= 8
    ]
    if not chunks:
        chunks = [normalize_space(dialogue)]
    if len(chunks) > max_items:
        # Preserve both the beginning and the direct question at the end.
        chunks = chunks[: max_items - 2] + chunks[-2:]
    return [{"evidence_id": f"E{i}", "quote": text} for i, text in enumerate(chunks, 1)]


def validate_requirements(
    requirements: list[dict[str, Any]],
    dialogue: str,
    evidence_map: dict[str, str],
    default_evidence_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    accepted = []
    reasons: list[str] = []
    for idx, req in enumerate(requirements, start=1):
        statement = normalize_space(req.get("statement"))
        if not statement:
            reasons.append(f"R{idx}:missing_semantic_statement")
            continue
        if not statement.lower().startswith("the system shall"):
            statement = "The system shall " + statement.rstrip(".")
            statement += "."
            reasons.append(f"R{idx}:statement_normalized")
        req_type = normalize_space(req.get("type")).lower()
        if req_type not in ALLOWED_TYPES:
            req_type = "functional"
            reasons.append(f"R{idx}:type_normalized")
        evidence_id = normalize_space(req.get("evidence_id"))
        if evidence_id not in evidence_map:
            evidence_id = default_evidence_id
            reasons.append(f"R{idx}:evidence_id_repaired_to_{evidence_id}")
        quote = evidence_map[evidence_id]
        if normalize_space(req.get("source_quote")) != quote:
            reasons.append(f"R{idx}:source_quote_bound_to_{evidence_id}")
        requirement_id = normalize_space(req.get("requirement_id"))
        if not requirement_id:
            requirement_id = f"REQ-{idx:03d}"
            reasons.append(f"R{idx}:requirement_id_filled")
        rationale = normalize_space(req.get("rationale"))
        if not rationale:
            rationale = (
                f"This requirement is grounded in the stakeholder cue identified as "
                f"{evidence_id}."
            )
            reasons.append(f"R{idx}:rationale_filled")
        acceptance = normalize_space(req.get("acceptance_criterion"))
        if not acceptance:
            behavior = re.sub(
                r"^the system shall\s+", "", statement, flags=re.I
            ).rstrip(".")
            acceptance = (
                f"In a scripted test derived from {evidence_id}, the system can "
                f"demonstrate that it {behavior}."
            )
            reasons.append(f"R{idx}:acceptance_criterion_filled")
        accepted.append(
            {
                "requirement_id": requirement_id,
                "type": req_type,
                "statement": statement,
                "source_quote": quote,
                "rationale": rationale,
                "acceptance_criterion": acceptance,
                "evidence_id": evidence_id,
            }
        )
    return accepted, reasons


def affect_fallback(
    dialogue: str, evidence_candidates: list[dict[str, str]]
) -> dict[str, Any]:
    lexicon = {
        "anxious": ["anxious", "anxiety", "worry", "worried", "afraid", "fear"],
        "depressed": ["depressed", "depression", "worthless", "hopeless", "empty"],
        "angry": ["angry", "anger", "furious", "mad"],
        "sad": ["sad", "grief", "loss", "cry", "hurt"],
        "overwhelmed": ["overwhelmed", "stress", "stressed", "cannot cope"],
    }
    lower = dialogue.lower()
    for affect, words in lexicon.items():
        for word in words:
            pos = lower.find(word)
            if pos >= 0:
                start = max(0, dialogue.rfind(".", 0, pos) + 1)
                end = dialogue.find(".", pos)
                if end < 0:
                    end = min(len(dialogue), pos + 120)
                best = min(
                    evidence_candidates,
                    key=lambda item: (
                        0 if word in item["quote"].lower() else 1,
                        len(item["quote"]),
                    ),
                )
                return {
                    "affect": affect,
                    "evidence_id": best["evidence_id"],
                    "cue_quote": best["quote"],
                    "confidence": 0.50,
                    "fallback": True,
                }
    best = evidence_candidates[-1]
    return {
        "affect": "unclear",
        "evidence_id": best["evidence_id"],
        "cue_quote": best["quote"],
        "confidence": 0.25,
        "fallback": True,
    }


def select_balanced_cases(records: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        buckets[row["primary_topic"]].append(row)
    for rows in buckets.values():
        rng.shuffle(rows)
    selected = []
    topics = sorted(buckets)
    while len(selected) < n and any(buckets.values()):
        for topic in topics:
            if buckets[topic] and len(selected) < n:
                selected.append(buckets[topic].pop())
    return selected


def make_need_model(
    train_records: list[dict[str, Any]], topics: list[str]
) -> tuple[TfidfVectorizer, OneVsRestClassifier, MultiLabelBinarizer, float]:
    train = pd.DataFrame(train_records)
    train["text"] = train["questionTitle"] + " [SEP] " + train["questionText"]
    mlb = MultiLabelBinarizer(classes=topics)
    y = mlb.fit_transform(train["topics_selected"])
    tuning = pd.read_csv(EXP / "needs_tuning.csv")
    best = (
        tuning[tuning["model"].str.startswith("Character")]
        .sort_values("validation_micro_f1", ascending=False)
        .iloc[0]
    )
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=45_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    x = vectorizer.fit_transform(train["text"])
    clf = OneVsRestClassifier(
        LinearSVC(
            C=float(best["C"]),
            class_weight="balanced",
            random_state=SEED,
            max_iter=5_000,
        )
    )
    clf.fit(x, y)
    return vectorizer, clf, mlb, float(best["threshold"])


def predict_topics(
    text: str,
    vectorizer: TfidfVectorizer,
    clf: OneVsRestClassifier,
    mlb: MultiLabelBinarizer,
    threshold: float,
) -> tuple[list[str], np.ndarray]:
    decision = clf.decision_function(vectorizer.transform([text]))[0]
    pred = (decision >= threshold).astype(int)
    if pred.sum() == 0:
        pred[int(np.argmax(decision))] = 1
    topics = list(mlb.inverse_transform(pred.reshape(1, -1))[0])
    return topics, decision


def retrieve_examples(
    text: str,
    train_records: list[dict[str, Any]],
    vectorizer: TfidfVectorizer,
    predicted_topics: list[str],
    k: int = 2,
) -> list[dict[str, str]]:
    train = pd.DataFrame(train_records)
    train["text"] = train["questionTitle"] + " [SEP] " + train["questionText"]
    mask = train["topics_selected"].apply(
        lambda xs: bool(set(xs).intersection(predicted_topics))
    )
    candidates = train[mask].copy()
    if len(candidates) < k:
        candidates = train.copy()
    matrix = vectorizer.transform(candidates["text"])
    query = vectorizer.transform([text])
    similarities = (matrix @ query.T).toarray().ravel()
    top = np.argsort(similarities)[::-1][:k]
    result = []
    for idx in top:
        row = candidates.iloc[int(idx)]
        result.append(
            {
                "question": normalize_space(row["questionText"])[:300],
                "topics": ", ".join(row["topics_selected"]),
                "similarity": f"{similarities[int(idx)]:.3f}",
            }
        )
    return result


class LocalGenerator:
    def __init__(self) -> None:
        torch.manual_seed(SEED)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if self.device == "mps" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=dtype, local_files_only=True
        ).to(self.device)
        self.model.eval()

    def generate(
        self, system: str, user: str, max_new_tokens: int
    ) -> tuple[str, dict[str, float]]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - start
        new_tokens = generated[0, encoded["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, {
            "latency_seconds": elapsed,
            "input_tokens": float(encoded["input_ids"].shape[1]),
            "output_tokens": float(new_tokens.shape[0]),
        }


def requirement_topic_f1(
    requirements: list[dict[str, Any]],
    true_topics: list[str],
    vectorizer: TfidfVectorizer,
    clf: OneVsRestClassifier,
    mlb: MultiLabelBinarizer,
    threshold: float,
) -> float:
    if not requirements:
        return 0.0
    text = " ".join(
        normalize_space(req.get("statement", ""))
        + " "
        + normalize_space(req.get("rationale", ""))
        for req in requirements
    )
    pred_topics, _ = predict_topics(text, vectorizer, clf, mlb, threshold)
    y_true = mlb.transform([true_topics])
    y_pred = mlb.transform([pred_topics])
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


def bootstrap_delta(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2_000
) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    deltas = np.empty(n_boot)
    n = len(a)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = np.mean(b[idx] - a[idx])
    return (
        float(np.mean(b - a)),
        float(np.quantile(deltas, 0.025)),
        float(np.quantile(deltas, 0.975)),
    )


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    split = json.loads((EXP / "agent_data_split.json").read_text(encoding="utf-8"))
    train_records = split["train_records"]
    test_records = split["test_records"]
    topics = split["selected_topics"]
    selected = select_balanced_cases(
        test_records, DEVELOPMENT_OFFSET + N_CASES
    )
    cases = selected[DEVELOPMENT_OFFSET : DEVELOPMENT_OFFSET + N_CASES]
    vectorizer, need_clf, mlb, need_threshold = make_need_model(train_records, topics)
    generator = LocalGenerator()

    case_rows: list[dict[str, Any]] = []
    req_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, start=1):
        case_id = f"CASE-{case_index:03d}"
        dialogue = normalize_space(case["questionText"])
        title = normalize_space(case["questionTitle"])
        full_text = f"{title} [SEP] {dialogue}"
        evidence_candidates = split_evidence(dialogue)
        evidence_map = {
            item["evidence_id"]: item["quote"] for item in evidence_candidates
        }
        predicted_topics, topic_scores = predict_topics(
            full_text, vectorizer, need_clf, mlb, need_threshold
        )
        retrieved = retrieve_examples(
            full_text, train_records, vectorizer, predicted_topics, k=2
        )

        baseline_system = (
            "You are a software requirements analyst. Do not give therapy or medical "
            "advice. Return valid JSON only."
        )
        baseline_user = f"""
Convert the stakeholder dialogue below into exactly one software requirement.
Return a JSON object with a "requirements" array. Every requirement must contain:
requirement_id, type, statement, source_quote, rationale, acceptance_criterion.
The statement must start with "The system shall". The source_quote must be copied
verbatim from the dialogue. Allowed type values: {", ".join(sorted(ALLOWED_TYPES))}.

Dialogue:
{dialogue}
""".strip()
        baseline_raw, baseline_usage = generator.generate(
            baseline_system, baseline_user, max_new_tokens=220
        )
        baseline_parsed = extract_json(baseline_raw)
        baseline_requirements = normalize_requirements(baseline_parsed)

        affect_system = (
            "You are the Affect Interpreter in a requirements-engineering pipeline. "
            "Do not diagnose and do not give advice. Return valid JSON only."
        )
        affect_user = f"""
Identify the stakeholder's most salient expressed affect. Select one evidence_id
from the supplied candidates and return exactly:
{{"affect":"short label","evidence_id":"E1","confidence":0.0}}

Evidence candidates:
{json.dumps(evidence_candidates, ensure_ascii=False)}
""".strip()
        affect_raw, affect_usage = generator.generate(
            affect_system, affect_user, max_new_tokens=100
        )
        affect = extract_json(affect_raw)
        affect_fallback_used = False
        if (
            not isinstance(affect, dict)
            or normalize_space(affect.get("evidence_id")) not in evidence_map
        ):
            affect = affect_fallback(dialogue, evidence_candidates)
            affect_fallback_used = True
        else:
            affect["cue_quote"] = evidence_map[normalize_space(affect["evidence_id"])]

        structurer_system = (
            "You are the Requirement Structurer in an affect-sensitive multi-agent "
            "requirements pipeline. Do not give therapy or medical advice. Use only "
            "the supplied dialogue as requirement evidence. Return valid JSON only."
        )
        structurer_user = f"""
Create exactly one software requirement from the source dialogue. Focus on a
software capability or quality attribute that addresses the expressed need.
Do not turn another person's behaviour into a system rule, and do not invent
organizations, policies, or clinical diagnoses.
Return a JSON object with a "requirements" array. Every requirement must contain:
requirement_id, type, statement, evidence_id, source_quote, rationale,
acceptance_criterion.
The statement must start with "The system shall". The source_quote must be copied
verbatim from the evidence candidate selected by evidence_id, not from retrieved
examples. Allowed type values:
{", ".join(sorted(ALLOWED_TYPES))}.

Need-topic classifier output: {json.dumps(predicted_topics)}
Affect interpreter output: {json.dumps(affect, ensure_ascii=False)}
Retrieved historical patterns (for topic orientation only):
{json.dumps([{"topics": x["topics"], "similarity": x["similarity"]} for x in retrieved], ensure_ascii=False)}

Evidence candidates:
{json.dumps(evidence_candidates, ensure_ascii=False)}
""".strip()
        agent_raw, agent_usage = generator.generate(
            structurer_system, structurer_user, max_new_tokens=220
        )
        agent_parsed = extract_json(agent_raw)
        agent_prevalidation = normalize_requirements(agent_parsed)
        agent_requirements, validator_actions = validate_requirements(
            agent_prevalidation,
            dialogue,
            evidence_map,
            normalize_space(affect["evidence_id"]),
        )

        variants = [
            (
                "Single-pass LLM",
                baseline_raw,
                baseline_parsed,
                baseline_requirements,
                baseline_usage,
                [],
            ),
            (
                "Staged affect-sensitive agent",
                agent_raw,
                agent_parsed,
                agent_requirements,
                {
                    "latency_seconds": affect_usage["latency_seconds"]
                    + agent_usage["latency_seconds"],
                    "input_tokens": affect_usage["input_tokens"]
                    + agent_usage["input_tokens"],
                    "output_tokens": affect_usage["output_tokens"]
                    + agent_usage["output_tokens"],
                },
                validator_actions,
            ),
        ]
        for variant, raw, parsed, requirements, usage, rejections in variants:
            scores = requirement_scores(requirements, dialogue)
            scores["topic_alignment_f1"] = requirement_topic_f1(
                requirements,
                case["topics_selected"],
                vectorizer,
                need_clf,
                mlb,
                need_threshold,
            )
            case_rows.append(
                {
                    "case_id": case_id,
                    "questionID": case["questionID"],
                    "variant": variant,
                    "primary_topic": case["primary_topic"],
                    "true_topics": "|".join(case["topics_selected"]),
                    "predicted_topics": "|".join(predicted_topics),
                    "json_parse_success": float(parsed is not None),
                    **scores,
                    "latency_seconds": usage["latency_seconds"],
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "affect_fallback_used": (
                        float(affect_fallback_used)
                        if variant == "Staged affect-sensitive agent"
                        else np.nan
                    ),
                    "validator_action_count": (
                        len(rejections)
                        if variant == "Staged affect-sensitive agent"
                        else np.nan
                    ),
                    "validator_repair_count": (
                        sum(
                            ("repaired" in x)
                            or ("bound" in x)
                            or ("filled" in x)
                            or ("normalized" in x)
                            for x in rejections
                        )
                        if variant == "Staged affect-sensitive agent"
                        else np.nan
                    ),
                    "validator_rejection_count": (
                        sum(
                            "missing_semantic_statement" in x
                            for x in rejections
                        )
                        if variant == "Staged affect-sensitive agent"
                        else np.nan
                    ),
                }
            )
            for req_index, req in enumerate(requirements, start=1):
                req_rows.append(
                    {
                        "case_id": case_id,
                        "questionID": case["questionID"],
                        "variant": variant,
                        "requirement_index": req_index,
                        "evidence_id": normalize_space(req.get("evidence_id", "")),
                        **{k: normalize_space(req.get(k, "")) for k in REQUIRED_FIELDS},
                    }
                )
            raw_rows.append(
                {
                    "case_id": case_id,
                    "questionID": case["questionID"],
                    "variant": variant,
                    "raw_output": raw,
                    "affect_output": affect if variant.startswith("Staged") else None,
                    "retrieved_examples": retrieved if variant.startswith("Staged") else None,
                    "validator_actions": rejections,
                }
            )
        print(f"completed {case_id}/{N_CASES}", flush=True)

    case_df = pd.DataFrame(case_rows)
    req_df = pd.DataFrame(req_rows)
    case_df.to_csv(OUT / "agent_case_metrics.csv", index=False)
    req_df.to_csv(OUT / "agent_requirements.csv", index=False)
    pd.DataFrame(
        [
            {
                "case_id": f"CASE-{i:03d}",
                "questionID": row["questionID"],
                "questionTitle": row["questionTitle"],
                "questionText": normalize_space(row["questionText"]),
                "primary_topic": row["primary_topic"],
                "true_topics": "|".join(row["topics_selected"]),
            }
            for i, row in enumerate(cases, start=1)
        ]
    ).to_csv(OUT / "agent_cases.csv", index=False)
    with (OUT / "agent_raw_outputs.jsonl").open("w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = [
        "json_parse_success",
        "schema_completeness",
        "srs_form_rate",
        "traceability_coverage",
        "lexical_evidence_validity",
        "topic_alignment_f1",
        "requirement_count",
        "latency_seconds",
        "input_tokens",
        "output_tokens",
    ]
    summary_rows = []
    paired_rows = []
    for variant, group in case_df.groupby("variant", sort=False):
        for metric in metrics:
            summary_rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "n_cases": len(group),
                    "mean": group[metric].mean(),
                    "std": group[metric].std(ddof=1),
                    "median": group[metric].median(),
                }
            )
    pivot = case_df.pivot(index="case_id", columns="variant", values=metrics)
    for metric in metrics:
        a = pivot[metric]["Single-pass LLM"].to_numpy(dtype=float)
        b = pivot[metric]["Staged affect-sensitive agent"].to_numpy(dtype=float)
        delta, lo, hi = bootstrap_delta(a, b)
        try:
            statistic, p_value = wilcoxon(b, a, zero_method="zsplit")
        except ValueError:
            statistic, p_value = np.nan, np.nan
        paired_rows.append(
            {
                "metric": metric,
                "delta_agent_minus_baseline": delta,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "bootstrap_replicates": 2_000,
                "wilcoxon_statistic": statistic,
                "wilcoxon_two_sided_p": p_value,
                "n_paired_cases": len(a),
                "seed": SEED,
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUT / "agent_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(OUT / "agent_paired_stats.csv", index=False)
    metadata = {
        "model_repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_config_sha256": sha256(MODEL_PATH / "config.json"),
        "model_weights_sha256": sha256(MODEL_PATH / "model.safetensors"),
        "device": generator.device,
        "dtype": "float16" if generator.device == "mps" else "float32",
        "generation": {"do_sample": False, "seed": SEED},
        "n_cases": N_CASES,
        "case_selection": (
            "round-robin by primary topic, within-topic shuffle seed 42; "
            f"held-out cases {DEVELOPMENT_OFFSET + 1}-"
            f"{DEVELOPMENT_OFFSET + N_CASES} after a disjoint "
            f"{DEVELOPMENT_OFFSET}-case prompt-development set"
        ),
        "prompt_development_cases": DEVELOPMENT_OFFSET,
        "prompt_development_set_disjoint_from_final": True,
        "need_classifier_threshold": need_threshold,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
    }
    (OUT / "agent_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
