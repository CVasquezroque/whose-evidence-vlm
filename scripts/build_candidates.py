#!/usr/bin/env python3
"""Build the deterministic geometric candidate manifest for one COCO split.

Writes <work_root>/manifests/candidates_<split>.jsonl and screen_stats_<split>.json. Only
automatic criteria are applied; human eligibility review happens in the annotation tool.

Example:
  python -u scripts/build_candidates.py --coco-split val2017
  python -u scripts/build_candidates.py --coco-split train2017
"""
from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone

from evidence_ownership.data.candidates import build_candidates
from evidence_ownership.data.coco import load_coco_persons
from evidence_ownership.utils.config import experiment_config, get_paths
from evidence_ownership.utils.hashing import sha256_file
from evidence_ownership.utils.io import write_json, write_jsonl
from evidence_ownership.utils.logs import get_logger, progress


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-split", choices=["train2017", "val2017"], required=True)
    ap.add_argument("--output-dir", default=None, help="default: <work_root>/manifests")
    ap.add_argument("--dry-run", action="store_true", help="screen and report counts; write nothing")
    ap.add_argument("--hash-annotations", action="store_true",
                    help="record sha256 of the instances JSON (slow for train2017: ~470 MB)")
    ap.add_argument("--no-progress", action="store_true")
    a = ap.parse_args()

    log = get_logger("build_candidates")
    paths, cfg = get_paths(), experiment_config()
    src = paths.instances_json(a.coco_split)
    log.info("loading %s", src)
    index = load_coco_persons(src, a.coco_split)
    cands, stats = build_candidates(index, cfg, progress=lambda it, **kw: progress(it, disable=a.no_progress, **kw))
    stats["candidates"] = len(cands)
    log.info("%s: %d candidates; screen stats %s", a.coco_split, len(cands), stats)
    if a.dry_run:
        return
    out = paths.manifests if a.output_dir is None else __import__("pathlib").Path(a.output_dir)
    write_jsonl(out / f"candidates_{a.coco_split}.jsonl", cands)
    write_json(out / f"screen_stats_{a.coco_split}.json", {
        "coco_split": a.coco_split,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "instances_json": src.name,
        "instances_json_sha256": sha256_file(src) if a.hash_annotations else None,
        "hash_salt": cfg["hash_salt"],
        "ab_assignment_salt": cfg["ab_assignment_salt"],
        "iou_max_exclusive": cfg["eligibility"]["person_bbox_iou_max_exclusive"],
        "python": platform.python_version(),
        "automatic_criteria_only": True,
        "stats": stats,
    })
    log.info("wrote %s", out / f"candidates_{a.coco_split}.jsonl")


if __name__ == "__main__":
    main()
