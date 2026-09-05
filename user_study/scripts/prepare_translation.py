#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import urllib.request
from pathlib import Path

OUTPUT = Path(__file__).with_name("translations.json")
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = os.environ.get("MODEL_NAME", "qwen3.7-plus-2026-05-26")


def load_key(env_file: Path | None = None) -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if key:
        return key
    if env_file and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("DASHSCOPE_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("DASHSCOPE_API_KEY is unavailable")


parser = argparse.ArgumentParser(description="Build a Chinese-to-English cell translation map.")
parser.add_argument("source", type=Path, help="source Chinese survey CSV")
parser.add_argument("--env-file", type=Path, help="optional file containing DASHSCOPE_API_KEY")
args = parser.parse_args()

with args.source.open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.reader(handle))

items = []
seen = set()
for row in rows:
    for value in row:
        if re.search(r"[\u3400-\u9fff]", value) and value not in seen:
            seen.add(value)
            items.append(value)

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "system",
            "content": (
                "Translate Chinese survey-dataset cell values into precise, natural English. "
                "Return a JSON object named translations mapping every supplied id to one translated string. "
                "Preserve question numbers, answer-option prefixes such as A./B./C., identifiers, punctuation, "
                "age ranges, ChatGPT, Output A/Output B, and all already-English text exactly. Translate all Chinese, "
                "including geographic names and long open-ended responses. Do not summarize, omit, normalize, "
                "clean, or infer missing content. Keep line breaks and bullet symbols when present."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({str(i): value for i, value in enumerate(items)}, ensure_ascii=False),
        },
    ],
    "temperature": 0,
    "max_completion_tokens": 12000,
    "response_format": {"type": "json_object"},
    "enable_thinking": False,
}

request = urllib.request.Request(
    ENDPOINT,
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Authorization": f"Bearer {load_key(args.env_file)}", "Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=180) as response:
    result = json.loads(response.read().decode("utf-8"))

content = json.loads(result["choices"][0]["message"]["content"])
translated = content.get("translations", content)
if set(translated) != {str(i) for i in range(len(items))}:
    raise RuntimeError("Translation response did not contain the complete id set")

mapping = {items[i]: str(translated[str(i)]) for i in range(len(items))}
leftovers = [value for value in mapping.values() if re.search(r"[\u3400-\u9fff]", value)]
if leftovers:
    raise RuntimeError(f"Chinese text remained in {len(leftovers)} translated values")

OUTPUT.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"unique_values={len(items)} translated={len(mapping)} output={OUTPUT}")
