#!/usr/bin/env python3
"""Make candidate source images available locally and record their SHA-256.

train2017: downloads individual JPEGs over HTTPS (certificate verified) from the COCO S3 bucket
           for the first --limit size-prescreened candidates in protocol order. Already present,
           valid files are kept (resumable).
val2017:   never downloads; verifies the local official val2017 JPEGs.

Writes <work_root>/manifests/images_<split>.jsonl  (scene_id, path, sha256, verified size).

Examples:
  python -u scripts/prepare_images.py --coco-split train2017 --limit 60
  python -u scripts/prepare_images.py --coco-split val2017
"""
from __future__ import annotations

import argparse
from pathlib import Path

from evidence_ownership.data.download import DownloadError, fetch, verify_image
from evidence_ownership.utils.config import get_paths
from evidence_ownership.utils.hashing import sha256_file
from evidence_ownership.utils.io import read_jsonl, write_jsonl
from evidence_ownership.utils.logs import get_logger, progress


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-split", choices=["train2017", "val2017"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N candidates in protocol order")
    ap.add_argument("--include-size-fail", action="store_true", help="also process size-prescreen failures")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched/verified")
    ap.add_argument("--no-progress", action="store_true")
    a = ap.parse_args()

    log = get_logger("prepare_images")
    paths = get_paths()
    cands = read_jsonl(paths.manifests / f"candidates_{a.coco_split}.jsonl")
    sel = [c for c in cands if a.include_size_fail or c["size_prescreen_pass"]]
    if a.limit is not None:
        sel = sel[: a.limit]
    img_dir = paths.images_dir(a.coco_split)
    log.info("%s: %d candidates selected (of %d) -> %s", a.coco_split, len(sel), len(cands), img_dir)
    if a.dry_run:
        for c in sel[:10]:
            log.info("  rank %d %s", c["order_rank"], c["file_name"])
        return

    out_path = paths.manifests / f"images_{a.coco_split}.jsonl"
    known = {r["scene_id"]: r for r in read_jsonl(out_path)} if out_path.exists() else {}
    failures = []
    for c in progress(sel, desc=f"images {a.coco_split}", unit="image", disable=a.no_progress):
        dest = img_dir / c["file_name"]
        try:
            if a.coco_split == "train2017":
                url = f"{paths.coco_image_base_url}/train2017/{c['file_name']}"
                digest = fetch(url, dest, c["width"], c["height"])
            else:
                if not dest.exists():
                    raise DownloadError(f"missing local val2017 file {dest}")
                verify_image(dest, c["width"], c["height"])
                digest = sha256_file(dest)
        except DownloadError as exc:
            failures.append({"scene_id": c["scene_id"], "error": str(exc)})
            log.error("%s", exc)
            continue
        prev = known.get(c["scene_id"])
        if prev and prev["sha256"] != digest:
            raise SystemExit(f"{c['scene_id']}: sha256 changed since last run ({prev['sha256']} -> {digest})")
        known[c["scene_id"]] = {"scene_id": c["scene_id"], "coco_split": a.coco_split, "file_name": c["file_name"],
                                "sha256": digest, "size": [c["width"], c["height"]], "verified": True}
    order = {c["scene_id"]: c["order_rank"] for c in cands}
    write_jsonl(out_path, sorted(known.values(), key=lambda r: order[r["scene_id"]]))
    log.info("%d verified images recorded in %s; %d failures", len(known), out_path, len(failures))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
