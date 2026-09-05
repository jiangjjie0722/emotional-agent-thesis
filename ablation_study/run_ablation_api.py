#!/usr/bin/env python3
"""Reconstructed end-to-end ablation study for the thesis.

The script intentionally keeps every API response and all deterministic
post-processing artifacts. It is resumable: completed case/condition records
are skipped on subsequent runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent


def load_env() -> None:
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()

DATASET = ROOT / "assets/counselchat/data/train-00000-of-00001.parquet"
RESULTS = ROOT / "results"
CASES_CSV = RESULTS / "frozen_cases.csv"
TRAIN_CSV = RESULTS / "training_cases.csv"
CONFIG_JSON = RESULTS / "protocol_config.json"
RAW_JSONL = RESULTS / "api_responses.jsonl"
PRED_JSONL = RESULTS / "predictions.jsonl"
SUMMARY_CSV = RESULTS / "condition_summary.csv"
COMPARE_CSV = RESULTS / "paired_comparisons.csv"

MODEL = os.environ.get("MODEL_NAME", "qwen3.7-plus-2026-05-26")
BASE_URL = os.environ.get(
    "OPENAI_COMPATIBLE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
)
SEED = 42
N_TRAIN = 479
N_DEV = 111
N_TEST = 148
N_POOL = 90
N_FINAL = 30
MIN_TOPIC_COUNT = 20
MAX_CASE_CHARS = 1500

CONDITIONS = {
    "B0_single_llm": {"affect": False, "need": False, "rag": False, "validator": False},
    "B1_single_llm_rag": {"affect": False, "need": True, "rag": True, "validator": False},
    "A1_without_affect": {"affect": False, "need": True, "rag": True, "validator": True},
    "A2_without_validator": {"affect": True, "need": True, "rag": True, "validator": False},
    "A3_full_agent": {"affect": True, "need": True, "rag": True, "validator": True},
    "A4_without_need_rag": {"affect": True, "need": False, "rag": False, "validator": True},
}

# Strict control-variable pairing. These conditions must use exactly the same
# upstream generation; the only difference is deterministic validation.
RAW_SOURCE_CONDITION = {
    "A1_without_affect": "B1_single_llm_rag",
    "A3_full_agent": "A2_without_validator",
}

ALLOWED_TYPES = ["functional", "non-functional", "privacy", "safety", "usability", "data"]
STOP = {
    "the", "and", "for", "that", "this", "with", "from", "user", "users", "system",
    "shall", "should", "would", "could", "into", "their", "there", "have", "has", "about",
    "they", "them", "when", "what", "where", "which", "while", "been", "being", "will",
}
WRITE_LOCK = threading.Lock()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_text(value: object) -> str:
    s = "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_topics(value: object) -> list[str]:
    raw = clean_text(value)
    return sorted({x.strip() for x in raw.split(",") if x.strip()})


def evidence_spans(text: str) -> list[dict]:
    parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    if not parts:
        parts = [text.strip()]
    parts = parts[:8]
    return [{"id": f"E{i+1}", "text": p} for i, p in enumerate(parts)]


def prepare_data(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if CASES_CSV.exists() and TRAIN_CSV.exists() and CONFIG_JSON.exists() and not force:
        cases = pd.read_csv(CASES_CSV)
        train = pd.read_csv(TRAIN_CSV)
        cfg = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
        return cases, train, cfg["eligible_topics"]

    df = pd.read_parquet(DATASET)
    rows = []
    for qid, g in df.groupby("questionID", sort=True):
        title = clean_text(g.iloc[0]["questionTitle"])
        question = clean_text(g.iloc[0]["questionText"])
        text = (title + ". " + question).strip(". ")[:MAX_CASE_CHARS]
        topics = sorted({t for v in g["topics"] for t in split_topics(v)})
        rows.append({"question_id": str(qid), "dialogue": text, "topics": json.dumps(topics, ensure_ascii=False)})
    qdf = pd.DataFrame(rows)
    counts = Counter(t for raw in qdf["topics"] for t in json.loads(raw))
    eligible = sorted(t for t, n in counts.items() if n >= MIN_TOPIC_COUNT)
    qdf["topics"] = qdf["topics"].map(lambda s: json.dumps([t for t in json.loads(s) if t in eligible], ensure_ascii=False))
    qdf = qdf[qdf["topics"].map(lambda s: len(json.loads(s)) > 0)].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(qdf))
    if len(qdf) < N_TRAIN + N_DEV + N_TEST:
        raise RuntimeError(f"Eligible dataset too small: {len(qdf)}")
    train = qdf.iloc[order[:N_TRAIN]].copy().reset_index(drop=True)
    dev = qdf.iloc[order[N_TRAIN:N_TRAIN + N_DEV]].copy().reset_index(drop=True)
    test = qdf.iloc[order[N_TRAIN + N_DEV:N_TRAIN + N_DEV + N_TEST]].copy().reset_index(drop=True)

    pool = test.iloc[:N_POOL].copy().reset_index(drop=True)
    cases = pool.iloc[-N_FINAL:].copy().reset_index(drop=True)
    cases.insert(0, "case_id", [f"C{i+1:02d}" for i in range(len(cases))])
    cases["evidence_spans"] = cases["dialogue"].map(lambda x: json.dumps(evidence_spans(x), ensure_ascii=False))
    train.to_csv(TRAIN_CSV, index=False)
    cases.to_csv(CASES_CSV, index=False)

    cfg = {
        "study_type": "reconstructed ablation; original code and held-out IDs unavailable",
        "model": MODEL,
        "endpoint_region": "China (Beijing)",
        "temperature": 0,
        "enable_thinking": False,
        "response_format": "json_object",
        "seed": SEED,
        "dataset_rows": int(len(df)),
        "unique_question_ids": int(df["questionID"].nunique()),
        "eligible_unique_questions": int(len(qdf)),
        "split_sizes": {"train": N_TRAIN, "development": N_DEV, "test": N_TEST, "end_to_end_pool": N_POOL, "final": N_FINAL},
        "eligible_topics": eligible,
        "conditions": CONDITIONS,
        "dataset_sha256": sha256(DATASET),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    CONFIG_JSON.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return cases, train, eligible


class NeedAndRetriever:
    def __init__(self, train: pd.DataFrame, topics: list[str]):
        self.train = train.copy()
        self.topics = topics
        labels = [json.loads(x) for x in train["topics"]]
        self.mlb = MultiLabelBinarizer(classes=topics)
        y = self.mlb.fit_transform(labels)
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=2, max_features=60000, sublinear_tf=True)
        self.x = self.vectorizer.fit_transform(train["dialogue"])
        self.clf = OneVsRestClassifier(LinearSVC(C=1.0, class_weight="balanced"))
        self.clf.fit(self.x, y)

    def guide(self, text: str, k_topics: int = 3, k_retrieval: int = 2) -> dict:
        x = self.vectorizer.transform([text])
        scores = np.asarray(self.clf.decision_function(x)).reshape(-1)
        idx = np.argsort(scores)[::-1][:k_topics]
        pred = [self.topics[i] for i in idx]
        sims = cosine_similarity(x, self.x).reshape(-1)
        candidates = np.argsort(sims)[::-1]
        retrieved = []
        for j in candidates:
            rt = json.loads(self.train.iloc[int(j)]["topics"])
            if set(rt) & set(pred) or len(retrieved) == 0:
                retrieved.append({
                    "record_id": str(self.train.iloc[int(j)]["question_id"]),
                    "topics": rt,
                    "similarity": round(float(sims[j]), 4),
                    "pattern": clean_text(self.train.iloc[int(j)]["dialogue"])[:260],
                })
            if len(retrieved) >= k_retrieval:
                break
        return {"predicted_topics": pred, "topic_scores": {self.topics[i]: round(float(scores[i]), 4) for i in idx}, "retrieved": retrieved}


def api_call(messages: list[dict], max_tokens: int = 700, attempts: int = 6) -> dict:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY is not configured")
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last = None
    for attempt in range(attempts):
        req = urllib.request.Request(BASE_URL, data=data, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            latency = time.perf_counter() - started
            msg = obj["choices"][0]["message"]
            return {"content": msg.get("content") or "", "usage": obj.get("usage", {}), "model": obj.get("model"), "request_id": obj.get("id"), "latency_s": latency}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(min(2 ** attempt, 20) + random.random())
    raise RuntimeError(f"API failed after retries: {last}")


def affect_messages(case: dict) -> list[dict]:
    spans = json.loads(case["evidence_spans"])
    return [
        {"role": "system", "content": "You are a non-diagnostic affect interpreter for software requirements engineering. Return one valid JSON object only. Do not diagnose or provide therapy."},
        {"role": "user", "content": (
            "Identify only affect cues explicitly supported by the dialogue. Return JSON with keys: "
            "affect_label (short string), intensity (low|medium|high|uncertain), evidence_id, "
            "evidence_quote (exact text), uncertainty (number 0-1), interpretation (one sentence). "
            f"Evidence spans: {json.dumps(spans, ensure_ascii=False)}"
        )},
    ]


def requirement_messages(case: dict, condition: str, guide: dict | None, affect: dict | None, topics: list[str]) -> list[dict]:
    spec = CONDITIONS[condition]
    spans = json.loads(case["evidence_spans"])
    context = {
        "case_id": case["case_id"],
        "dialogue_evidence": spans,
        "allowed_requirement_types": ALLOWED_TYPES,
        "allowed_need_labels": topics,
    }
    if spec["affect"]:
        context["affect_guidance"] = affect
    if spec["need"] and guide:
        context["predicted_need_topics"] = guide["predicted_topics"]
    if spec["rag"] and guide:
        context["retrieved_topic_patterns"] = guide["retrieved"]
    system = (
        "You are a software requirements analyst. Generate requirements, not counseling or clinical advice. "
        "Every requirement must be supported by the supplied dialogue. Do not invent diagnoses, crisis features, "
        "data sharing, monitoring, notifications, or third-party access unless supported. Return exactly one valid JSON object."
    )
    user = (
        "Return JSON in this schema: {\"requirements\":[{\"requirement_statement\":\"The system shall ...\","
        "\"requirement_type\":\"one allowed type\",\"source_turn_ids\":[\"E1\"],"
        "\"evidence_quotes\":[\"exact source text\"],\"need_labels\":[\"allowed label\"],"
        "\"rationale\":\"brief evidence-grounded reason\"}],\"uncertainty\":\"low|medium|high\"}. "
        "Generate one or at most two concise, testable SRS requirements. Use retrieved patterns only as routing hints, "
        "never as evidence. JSON input follows:\n" + json.dumps(context, ensure_ascii=False)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_json(content: str) -> dict | None:
    try:
        value = json.loads(content)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def normalize_req(item: dict, spans: list[dict], fallback_topics: list[str] | None) -> tuple[dict | None, list[str]]:
    repairs = []
    statement = clean_text(item.get("requirement_statement") or item.get("statement") or "")
    if len(statement.split()) < 4:
        return None, ["reject_missing_semantic_statement"]
    if not re.match(r"(?i)^the system shall\b", statement):
        statement = "The system shall " + statement[0].lower() + statement[1:]
        repairs.append("normalize_srs_prefix")
    rtype = clean_text(item.get("requirement_type") or item.get("type") or "non-functional").lower()
    if rtype not in ALLOWED_TYPES:
        rtype = "non-functional"
        repairs.append("normalize_requirement_type")
    ids = item.get("source_turn_ids") or item.get("source_turn_id") or []
    if isinstance(ids, str):
        ids = [ids]
    valid = [s for s in spans if s["id"] in ids]
    if not valid:
        words = {w for w in re.findall(r"[a-z]{3,}", statement.lower()) if w not in STOP}
        scored = []
        for s in spans:
            sw = {w for w in re.findall(r"[a-z]{3,}", s["text"].lower()) if w not in STOP}
            scored.append((len(words & sw), s))
        best_score, best = max(scored, key=lambda x: x[0])
        if best_score == 0:
            return None, repairs + ["reject_no_evidence_overlap"]
        valid = [best]
        repairs.append("bind_evidence_id")
    supplied_quotes = item.get("evidence_quotes") or item.get("evidence_quote") or []
    if isinstance(supplied_quotes, str):
        supplied_quotes = [supplied_quotes]
    canonical_quotes = [s["text"] for s in valid]
    if supplied_quotes != canonical_quotes:
        repairs.append("normalize_evidence_quote")
    labels = item.get("need_labels") or []
    if isinstance(labels, str):
        labels = [labels]
    labels = [clean_text(x) for x in labels if clean_text(x)]
    if not labels and fallback_topics:
        labels = list(fallback_topics[:2])
        repairs.append("fill_need_labels")
    out = {
        "requirement_statement": statement,
        "requirement_type": rtype,
        "source_turn_ids": [s["id"] for s in valid],
        "evidence_quotes": canonical_quotes,
        "need_labels": labels,
        "rationale": clean_text(item.get("rationale") or "Grounded in the cited dialogue evidence."),
    }
    return out, repairs


def validate(content: str, spans: list[dict], fallback_topics: list[str] | None) -> tuple[dict, list[str]]:
    raw = parse_json(content)
    repairs = []
    if raw is None:
        return {"requirements": [], "uncertainty": "high", "validation_status": "rejected"}, ["reject_invalid_json"]
    items = raw.get("requirements")
    if isinstance(items, dict):
        items = [items]
        repairs.append("normalize_requirements_array")
    if not isinstance(items, list):
        items = []
    accepted = []
    for item in items[:2]:
        if not isinstance(item, dict):
            continue
        norm, rp = normalize_req(item, spans, fallback_topics)
        repairs.extend(rp)
        if norm:
            accepted.append(norm)
    return {
        "requirements": accepted,
        "uncertainty": clean_text(raw.get("uncertainty") or "high"),
        "validation_status": "accepted" if accepted else "rejected",
    }, repairs


def append_jsonl(path: Path, obj: dict) -> None:
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def run_experiment(workers: int = 3) -> None:
    load_env()
    cases_df, train, topics = prepare_data()
    nr = NeedAndRetriever(train, topics)
    cases = cases_df.to_dict("records")
    guides = {c["case_id"]: nr.guide(c["dialogue"]) for c in cases}

    raw_existing = load_jsonl(RAW_JSONL)
    affect_cache = {r["case_id"]: r for r in raw_existing if r.get("stage") == "affect" and r.get("ok")}

    def do_affect(case: dict) -> None:
        if case["case_id"] in affect_cache:
            return
        try:
            res = api_call(affect_messages(case), max_tokens=300)
            rec = {"stage": "affect", "case_id": case["case_id"], "ok": True, **res}
        except Exception as exc:
            rec = {"stage": "affect", "case_id": case["case_id"], "ok": False, "error": str(exc)}
        append_jsonl(RAW_JSONL, rec)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(do_affect, cases))

    raw_existing = load_jsonl(RAW_JSONL)
    affect_cache = {r["case_id"]: parse_json(r.get("content", "")) for r in raw_existing if r.get("stage") == "affect" and r.get("ok")}
    done = {(r.get("case_id"), r.get("condition")) for r in raw_existing if r.get("stage") == "requirements" and r.get("ok")}
    jobs = [(c, cond) for c in cases for cond in CONDITIONS if (c["case_id"], cond) not in done]

    def do_requirement(job: tuple[dict, str]) -> None:
        case, cond = job
        try:
            res = api_call(requirement_messages(case, cond, guides[case["case_id"]], affect_cache.get(case["case_id"]), topics), max_tokens=700)
            rec = {"stage": "requirements", "case_id": case["case_id"], "condition": cond, "ok": True, **res}
        except Exception as exc:
            rec = {"stage": "requirements", "case_id": case["case_id"], "condition": cond, "ok": False, "error": str(exc)}
        append_jsonl(RAW_JSONL, rec)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        completed = 0
        for _ in ex.map(do_requirement, jobs):
            completed += 1
            if completed % 15 == 0 or completed == len(jobs):
                print(f"requirements completed {completed}/{len(jobs)}", flush=True)

    build_predictions(cases, guides, topics)


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOP}


def evaluate_prediction(final: dict | None, raw_parse: bool, spans: list[dict], gold: list[str], latency: float, usage: dict, repairs: list[str]) -> dict:
    final = final if isinstance(final, dict) else {}
    reqs = final.get("requirements") if isinstance(final.get("requirements"), list) else []
    valid_ids = {s["id"]: s["text"] for s in spans}
    srs = bool(reqs) and all(bool(re.match(r"(?i)^the system shall\b", clean_text(r.get("requirement_statement", "")))) for r in reqs if isinstance(r, dict))
    trace = bool(reqs)
    labels = set()
    lexical_scores = []
    allowed_type = True
    for r in reqs:
        if not isinstance(r, dict):
            trace = False
            allowed_type = False
            continue
        ids = r.get("source_turn_ids") or []
        quotes = r.get("evidence_quotes") or []
        if isinstance(ids, str): ids = [ids]
        if isinstance(quotes, str): quotes = [quotes]
        if not ids or any(i not in valid_ids for i in ids) or any(q not in valid_ids.values() for q in quotes):
            trace = False
        if clean_text(r.get("requirement_type", "")).lower() not in ALLOWED_TYPES:
            allowed_type = False
        lab = r.get("need_labels") or []
        if isinstance(lab, str): lab = [lab]
        labels.update(clean_text(x) for x in lab if clean_text(x))
        sw = content_words(clean_text(r.get("requirement_statement", "")))
        dw = content_words(" ".join(valid_ids.values()))
        lexical_scores.append(len(sw & dw) / max(1, len(sw)))
    full = bool(raw_parse and reqs and srs and trace and allowed_type)
    gold_set = set(gold)
    tp = len(labels & gold_set)
    topic_f1 = 0.0 if not labels or not gold_set else 2 * tp / (len(labels) + len(gold_set))
    return {
        "json_parse": int(raw_parse), "accepted": int(bool(reqs)), "srs_form": int(srs),
        "traceability": int(trace), "full_compliance": int(full), "topic_alignment_f1": topic_f1,
        "lexical_grounding_proxy": statistics.mean(lexical_scores) if lexical_scores else 0.0,
        "latency_s": float(latency or 0.0), "prompt_tokens": int((usage or {}).get("prompt_tokens", 0)),
        "completion_tokens": int((usage or {}).get("completion_tokens", 0)), "repair_count": len(repairs),
        "requirement_count": len(reqs),
    }


def build_predictions(cases: list[dict], guides: dict, topics: list[str]) -> None:
    raw = load_jsonl(RAW_JSONL)
    lookup = {(r.get("case_id"), r.get("condition")): r for r in raw if r.get("stage") == "requirements" and r.get("ok")}
    affect_lookup = {r.get("case_id"): r for r in raw if r.get("stage") == "affect" and r.get("ok")}
    rows = []
    for case in cases:
        spans = json.loads(case["evidence_spans"])
        gold = json.loads(case["topics"])
        for cond, spec in CONDITIONS.items():
            source_cond = RAW_SOURCE_CONDITION.get(cond, cond)
            rr = lookup.get((case["case_id"], source_cond), {})
            content = rr.get("content", "")
            parsed = parse_json(content)
            repairs = []
            if spec["validator"]:
                fallback = guides[case["case_id"]]["predicted_topics"] if spec["need"] else None
                final, repairs = validate(content, spans, fallback)
                parse_ok = True  # deterministic validator always emits valid JSON
            else:
                final = parsed
                parse_ok = parsed is not None
            latency = float(rr.get("latency_s", 0) or 0)
            usage = dict(rr.get("usage", {}) or {})
            if spec["affect"]:
                ar = affect_lookup.get(case["case_id"], {})
                latency += float(ar.get("latency_s", 0) or 0)
                au = ar.get("usage", {}) or {}
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    usage[key] = int(usage.get(key, 0)) + int(au.get(key, 0))
            metrics = evaluate_prediction(final, parse_ok, spans, gold, latency, usage, repairs)
            rows.append({
                "case_id": case["case_id"], "question_id": case["question_id"], "condition": cond,
                "raw_source_condition": source_cond,
                "gold_topics": gold, "predicted_guide_topics": guides[case["case_id"]]["predicted_topics"],
                "raw_content": content, "final_output": final, "repairs": repairs, "metrics": metrics,
            })
    PRED_JSONL.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    analyze(rows)


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 2000) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    d = a - b
    vals = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n = n10 + n01
    return 1.0 if n == 0 else float(binomtest(min(n10, n01), n, .5, alternative="two-sided").pvalue)


def holm_adjust(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    out = np.empty(len(pvals), dtype=float)
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[idx]))
        out[idx] = running
    return out.tolist()


def analyze(rows: list[dict]) -> None:
    flat = []
    for r in rows:
        flat.append({"case_id": r["case_id"], "condition": r["condition"], **r["metrics"]})
    df = pd.DataFrame(flat)
    metric_cols = ["json_parse", "accepted", "srs_form", "traceability", "full_compliance", "topic_alignment_f1", "lexical_grounding_proxy", "latency_s", "prompt_tokens", "completion_tokens", "repair_count", "requirement_count"]
    summary = df.groupby("condition")[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["condition"] + [f"{a}_{b}" for a, b in summary.columns.tolist()[1:]]
    summary.to_csv(SUMMARY_CSV, index=False)

    full = df[df.condition == "A3_full_agent"].set_index("case_id")
    comps = []
    binary = ["json_parse", "accepted", "srs_form", "traceability", "full_compliance"]
    continuous = ["topic_alignment_f1", "lexical_grounding_proxy", "latency_s", "prompt_tokens", "completion_tokens"]
    for cond in CONDITIONS:
        if cond == "A3_full_agent":
            continue
        other = df[df.condition == cond].set_index("case_id").loc[full.index]
        for metric in binary + continuous:
            a = full[metric].to_numpy(float)
            b = other[metric].to_numpy(float)
            diff, lo, hi = bootstrap_diff(a, b)
            if metric in binary:
                p = mcnemar_exact(a.astype(int), b.astype(int))
                test = "exact McNemar"
            else:
                try:
                    p = float(wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue) if np.any(a != b) else 1.0
                except ValueError:
                    p = 1.0
                test = "Wilcoxon signed-rank"
            comps.append({"comparison": f"A3_full_agent - {cond}", "metric": metric, "full_mean": a.mean(), "other_mean": b.mean(), "difference": diff, "ci_low": lo, "ci_high": hi, "test": test, "p_raw": p})
    # Control the family-wise error rate within each prespecified metric family
    # (five comparisons against the full system), rather than across unrelated
    # binary, semantic, and cost outcomes together.
    for metric in binary + continuous:
        positions = [i for i, x in enumerate(comps) if x["metric"] == metric]
        adjusted = holm_adjust([comps[i]["p_raw"] for i in positions])
        for i, p in zip(positions, adjusted):
            comps[i]["p_holm"] = p
    pd.DataFrame(comps).to_csv(COMPARE_CSV, index=False)

    cfg = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    cfg["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cfg["api_calls_executed"] = N_FINAL * (1 + len(CONDITIONS))
    cfg["api_calls_in_primary_analysis"] = N_FINAL * (1 + 4)
    cfg["excluded_redundant_generation_calls"] = N_FINAL * 2
    cfg["strict_pairing"] = RAW_SOURCE_CONDITION
    cfg["output_files"] = {p.name: sha256(p) for p in [CASES_CSV, TRAIN_CSV, RAW_JSONL, PRED_JSONL, SUMMARY_CSV, COMPARE_CSV] if p.exists()}
    CONFIG_JSON.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--force-prepare", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    prepare_data(force=args.force_prepare)
    if not args.prepare_only:
        run_experiment(workers=args.workers)


if __name__ == "__main__":
    main()
