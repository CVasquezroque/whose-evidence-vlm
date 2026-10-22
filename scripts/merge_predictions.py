#!/usr/bin/env python3
"""Merge per-worker raw prediction logs (from --shard runs) into one validated file.

Refuses conflicting ok records for the same query_id. Inputs are left untouched.

Example:
  python -u scripts/merge_predictions.py --output <run-dir>/raw_predictions.jsonl <run-dir>/raw_predictions.*.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from evidence_ownership.inference.runner import merge_predictions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit(f"{a.output} exists; refusing to overwrite raw predictions")
    print(f"merged {merge_predictions(a.inputs, a.output)} records -> {a.output}")


if __name__ == "__main__":
    main()
