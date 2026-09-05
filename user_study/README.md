# User Study Dataset

This directory contains the English-language survey export used to compare two requirement-generation outputs across three cases.

## Public release

`data/emotion_survey_public.csv` contains 43 records. Rows have fresh sequential pseudonyms (`P001`–`P043`) and retain only the survey stimulus/response columns and language. The public file excludes:

- original response IDs;
- start/end times and response duration;
- cleaning status and invalidity scores;
- country, province and city;
- IP address and browser/device user-agent;
- referrer, custom fields and empty administrative columns.

The age field is categorical rather than an exact age. Nevertheless, researchers should treat the file as human-participant research data and use it only under the consent and ethics terms applicable to the original study.

## Reproducing the public file

The original English export is intentionally stored under `private/` and ignored by Git. To rebuild the public file locally:

```bash
python experiments/user_study/scripts/sanitize_dataset.py \
  experiments/user_study/private/raw_export_english_20260815/Emotion_English.csv \
  experiments/user_study/data/emotion_survey_public.csv
```

`scripts/prepare_translation.py`, `translations.json`, and `build_english_csv.mjs` document the earlier Chinese-to-English preparation workflow. The translation script requires a DashScope-compatible API key. The JavaScript workbook preview script additionally depends on the local `@oai/artifact-tool` package; it is not required to use the released CSV.

## Data dictionary

- `participant_id`: release-specific pseudonymous row identifier.
- Questions 1–4: consent/eligibility, age band, generative-AI familiarity and requirements-engineering experience.
- Case columns and Questions 6–16: case stimuli where present and per-case comparisons of need recognition, evidence traceability and requirement quality.
- Questions 17–18: overall stability and analyst preference.
- `Language`: language recorded in the translated export.

Blank values are preserved. No attempt was made to infer missing responses.

