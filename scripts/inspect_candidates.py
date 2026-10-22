#!/usr/bin/env python3
"""Export viewable candidate images: annotated PNGs, contact sheets and an HTML gallery.

Shows COCO person boxes with canonical A/B identities (inspection colours, not the
experimental markers). Output: <work_root>/galleries/candidates_<split>/
  index.html            browsable gallery (open via SSH tunnel or after scp)
  sheet_XX.png          contact sheets with scene ids and order ranks
  img/<scene_id>.png    full-size annotated images

Examples:
  python -u scripts/inspect_candidates.py --coco-split train2017 --limit 60
  python -u scripts/inspect_candidates.py --coco-split val2017 --limit 40
"""
from __future__ import annotations

import argparse

from PIL import Image

from evidence_ownership.data.visualize import contact_sheet, draw_candidate, gallery_html, thumb_with_caption
from evidence_ownership.utils.config import get_paths
from evidence_ownership.utils.io import read_jsonl
from evidence_ownership.utils.logs import get_logger, progress


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-split", choices=["train2017", "val2017"], required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--include-size-fail", action="store_true")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--per-sheet", type=int, default=20)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--no-progress", action="store_true")
    a = ap.parse_args()
    log = get_logger("inspect_candidates")
    paths = get_paths()
    have = {r["scene_id"] for r in read_jsonl(paths.manifests / f"images_{a.coco_split}.jsonl")}
    cands = [c for c in read_jsonl(paths.manifests / f"candidates_{a.coco_split}.jsonl")
             if (a.include_size_fail or c["size_prescreen_pass"]) and c["scene_id"] in have][: a.limit]
    out = paths.galleries / f"candidates_{a.coco_split}" if a.output_dir is None else __import__("pathlib").Path(a.output_dir)
    (out / "img").mkdir(parents=True, exist_ok=True)
    tiles, entries = [], []
    for c in progress(cands, desc="render", unit="image", disable=a.no_progress):
        with Image.open(paths.images_dir(a.coco_split) / c["file_name"]) as im:
            ann = draw_candidate(im, c)
        ann.save(out / "img" / f"{c['scene_id']}.png")
        tiles.append(thumb_with_caption(ann, f"#{c['order_rank']} {c['source_image_id']}", 300))
        entries.append({"img": f"img/{c['scene_id']}.png", "caption": f"rank {c['order_rank']} · {c['scene_id']}",
                        "meta": {"A side": c["persons"]["A"]["side"], "IoU": f"{c['bbox_iou']:.3f}",
                                 "size": f"{c['width']}x{c['height']}", "license": c["license"]["name"]}})
    for i in range(0, len(tiles), a.per_sheet):
        contact_sheet(tiles[i:i + a.per_sheet], a.cols, f"{a.coco_split} candidates {i}-{i + len(tiles[i:i + a.per_sheet]) - 1} (protocol order; yellow=A, cyan=B)").save(out / f"sheet_{i // a.per_sheet:02d}.png")
    (out / "index.html").write_text(gallery_html(
        f"{a.coco_split} geometric candidates (first {len(cands)} in protocol order)", entries,
        "Automatic criteria only. Human eligibility review has NOT been applied."), encoding="utf-8")
    log.info("wrote %d images, %d sheets -> %s", len(cands), (len(tiles) + a.per_sheet - 1) // a.per_sheet, out)


if __name__ == "__main__":
    main()
