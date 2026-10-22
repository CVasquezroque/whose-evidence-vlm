#!/usr/bin/env python3
"""Build renderer items from human annotations, with an exclusion log and selection flow.

Writes <work_root>/manifests/items_<study_split>.jsonl, exclusions_<study_split>.jsonl and
selection_flow_<study_split>.json.

Examples:
  python -u scripts/build_items.py --coco-split train2017 --mode exploratory --dry-run
  python -u scripts/build_items.py --coco-split train2017 --mode exploratory
  python -u scripts/build_items.py --coco-split val2017 --mode confirmatory --target 80
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from evidence_ownership.annotation.build_items import MODES, build_items
from evidence_ownership.data.candidates import STUDY_SPLIT
from evidence_ownership.utils.config import experiment_config, get_paths
from evidence_ownership.utils.io import read_json, read_jsonl, write_json, write_jsonl
from evidence_ownership.utils.logs import get_logger


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-split", choices=["train2017", "val2017"], required=True)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--target", type=int, default=None, help="stop after N selected clusters (default: protocol target)")
    ap.add_argument("--robustness-n", type=int, default=None, help="default: 20 held-out, 5 pilot")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    log = get_logger("build_items")
    paths, cfg = get_paths(), experiment_config()
    split = STUDY_SPLIT[a.coco_split]
    target = a.target if a.target is not None else cfg["splits"][split]["target_scenes"]
    rob_n = a.robustness_n if a.robustness_n is not None else (cfg["robustness_subset_scenes"] if split == "heldout" else 5)
    cands = read_jsonl(paths.manifests / f"candidates_{a.coco_split}.jsonl")
    images = {r["scene_id"]: r for r in read_jsonl(paths.manifests / f"images_{a.coco_split}.jsonl")}
    cl_path = paths.manifests / f"duplicate_clusters_{a.coco_split}.json"
    clusters = read_json(cl_path) if cl_path.exists() else {}
    items, excl, flow = build_items(cands, images, paths.annotations, mode=a.mode, target=target, clusters=clusters,
                                    robustness_salt=cfg["robustness_salt"], robustness_n=rob_n)
    flow.update({"mode": a.mode, "target": target, "robustness_n": rob_n,
                 "annotation_status_counts": {s: sum(i["annotation_status"] == s for i in items)
                                              for s in ("two_rater_agreed", "exploratory_single_rater")},
                 "target_reached": len(items) >= target,
                 "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    log.info("%s (%s): %d items selected; flow %s", split, a.mode, len(items), flow)
    if a.dry_run:
        return
    write_jsonl(paths.manifests / f"items_{split}.jsonl", items)
    write_jsonl(paths.manifests / f"exclusions_{split}.jsonl", excl)
    write_json(paths.manifests / f"selection_flow_{split}.json", flow)
    if split == "heldout" and not flow["target_reached"]:
        log.warning("held-out target of %d NOT reached (%d). Report the smaller n as exploratory.", target, len(items))


if __name__ == "__main__":
    main()
