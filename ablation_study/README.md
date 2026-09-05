# Agent Ablation Study
Note: The ablation study is provided as supplementary experimental material. The primary thesis results are contained in original_studies/.
## Objective

This experiment evaluates which agent components improve the conversion of mental-health-related user conversations into structured, evidence-traceable software requirements. It is a reconstructed ablation study because the original implementation and held-out identifiers were unavailable when this protocol was assembled.

## Data preparation

The frozen parquet snapshot comes from [`loaiabdalslam/counselchat`](https://huggingface.co/datasets/loaiabdalslam/counselchat), whose dataset page marks it OpenRAIL. It contains 1,482 answer-level rows representing 781 unique questions in this snapshot. The pipeline groups rows by `questionID`, concatenates each question title and body, and keeps topic labels occurring in at least 20 unique-question records. With random seed 42, it creates:

- 479 training questions;
- 111 development questions;
- 148 test questions;
- a deterministic 90-question end-to-end pool; and
- the final 30 frozen cases used for the API experiment.

Evidence spans are sentence-like segments labeled `E1`, `E2`, and so on. The frozen public artifacts are `results/training_cases.csv` and `results/frozen_cases.csv`.

## Conditions

| ID | Affect extraction | Need classifier | Retrieval | Validator |
|---|---:|---:|---:|---:|
| `B0_single_llm` | No | No | No | No |
| `B1_single_llm_rag` | No | Yes | Yes | No |
| `A1_without_affect` | No | Yes | Yes | Yes |
| `A2_without_validator` | Yes | Yes | Yes | No |
| `A3_full_agent` | Yes | Yes | Yes | Yes |
| `A4_without_need_rag` | Yes | No | No | Yes |

For strict paired comparisons, `A1` reuses the upstream generation from `B1`, and `A3` reuses the upstream generation from `A2`; only deterministic validation differs within each pair. The completed run executed 210 calls, while 150 calls belong to the primary analysis and 60 redundant generation calls were excluded.

## Agent procedure

1. **Need model and retrieval.** Character 1–4 gram TF-IDF features train a one-vs-rest linear SVM on the 479 training questions. The highest-scoring topics guide the prompt, and cosine similarity retrieves two related examples.
2. **Affect extraction.** When enabled, the API receives evidence spans and returns a non-diagnostic affect label, intensity, quoted evidence and uncertainty.
3. **Requirement generation.** The API returns JSON requirements with an SRS-style statement, requirement type, evidence-span IDs, exact evidence quotes, need labels and rationale.
4. **Validation.** When enabled, deterministic post-processing repairs malformed output, unsupported evidence references and schema violations.
5. **Evaluation.** The pipeline calculates JSON parsing, acceptance, SRS form, traceability, full compliance, topic-alignment F1, a lexical-grounding proxy, latency, token usage, repair count and requirement count.
6. **Inference.** The full system is compared with every other condition using paired bootstrap confidence intervals. Binary metrics use exact McNemar tests; continuous metrics use Wilcoxon signed-rank tests. Holm correction controls the five comparisons within each metric family.

## Running the experiment

From the repository root:

```bash
python -m pip install -r requirements.txt
cp experiments/ablation_study/.env.example experiments/ablation_study/.env
# Put a valid key in .env.
python experiments/ablation_study/run_ablation_api.py --prepare-only
python experiments/ablation_study/run_ablation_api.py --workers 3
python experiments/ablation_study/make_charts.py
python experiments/ablation_study/build_report.py
```

`--prepare-only` performs no API generation. `--force-prepare` regenerates the deterministic split. The main run appends raw responses as it completes and can resume after interruption. Existing result files represent the completed published run; copy them before re-running if you want to preserve them.

## Outputs

| Path | Description |
|---|---|
| `results/protocol_config.json` | Model settings, split sizes, hashes and pairing rules |
| `results/api_responses.jsonl` | Raw API responses, usage and latency |
| `results/predictions.jsonl` | Parsed/validated requirements and per-case metrics |
| `results/condition_summary.csv` | Mean and standard deviation by condition |
| `results/paired_comparisons.csv` | Effect estimates, confidence intervals and tests |
| `results/figures/` | Outcome, effect and engineering-cost charts |
| `reports/ablation_results_qwen_api.docx` | Generated experiment report |

The downloaded local Qwen checkpoint is not used by the API experiment and has been moved to `local_assets/`, which is excluded from Git. The local dependency snapshot (`.deps/`) and report-rendering QA cache are also excluded.

## Limitations

The study uses 30 final cases and automatic metrics, so it should be interpreted as a controlled engineering evaluation rather than evidence of clinical effectiveness. Topic labels serve as a proxy for need alignment. Latency and token measurements depend on the API provider and measurement date. The conversations may contain sensitive subject matter even though they come from the source research dataset.
