#!/usr/bin/env python3
"""Create the de-identified public user-study CSV from a private export."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FIRST_SURVEY_COLUMN = (
    "1. Are you at least 18 years old, have read the instructions, "
    "and voluntarily participate in this survey?"
)
LAST_SURVEY_COLUMN = "Language"


def sanitize(source: Path, destination: Path) -> None:
    frame = pd.read_csv(source)
    start = frame.columns.get_loc(FIRST_SURVEY_COLUMN)
    end = frame.columns.get_loc(LAST_SURVEY_COLUMN)
    public = frame.iloc[:, start : end + 1].copy()
    public = public.dropna(axis=1, how="all")
    public.insert(0, "participant_id", [f"P{i:03d}" for i in range(1, len(public) + 1)])
    destination.parent.mkdir(parents=True, exist_ok=True)
    public.to_csv(destination, index=False, encoding="utf-8")
    print(f"wrote {len(public)} rows x {len(public.columns)} columns to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="private English survey CSV")
    parser.add_argument("destination", type=Path, help="public de-identified CSV")
    args = parser.parse_args()
    sanitize(args.source, args.destination)


if __name__ == "__main__":
    main()

