#!/usr/bin/env python3
"""Run staged validation (Levels 1-6) and write a machine-readable report.

  1 source assets     2 candidates/annotations/items     3 rendered interventions
  4 inference run     5 statistics                       6 repository / release hygiene

Exit code 1 if any check FAILs. SKIPPED is reported but never counted as a pass.

Examples:
  python -u scripts/validate.py                       # levels 1-3 and 6
  python -u scripts/validate.py --run <work_root>/runs/pilot_qwen35 --levels 4,5
"""
from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
from pathlib import Path

from evidence_ownership.utils.config import get_paths
from evidence_ownership.utils.io import write_json
from evidence_ownership.validation.pipeline import run_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="1,2,3,6")
    ap.add_argument("--run", type=Path, default=None, help="run directory for levels 4-5")
    ap.add_argument("--output", type=Path, default=None, help="default: <work_root>/validation/report_<utc>.json")
    ap.add_argument("--no-progress", action="store_true")
    a = ap.parse_args()
    levels = {int(x) for x in a.levels.split(",") if x}
    checks = run_all(levels, a.run, a.no_progress)
    width = max(len(c["name"]) for c in checks)
    for c in checks:
        detail = str(c.get("detail", ""))
        print(f"L{c['level']}  {c['status']:<7} {c['name']:<{width}}  {detail[:110]}")
    counts = collections.Counter(c["status"] for c in checks)
    print(f"\nSUMMARY: {dict(counts)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = a.output or get_paths().work_root / "validation" / f"report_{stamp}.json"
    write_json(out, {"created_utc": stamp, "levels": sorted(levels), "run": str(a.run) if a.run else None,
                     "summary": dict(counts), "checks": checks})
    print(f"report: {out}")
    raise SystemExit(1 if counts.get("FAIL") else 0)


if __name__ == "__main__":
    main()
