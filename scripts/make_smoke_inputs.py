#!/usr/bin/env python3
"""Engineering-only smoke-test items from pilot (train2017) candidates.

NOT HUMAN ANNOTATION. Garment "masks" are fixed rectangles (25-75% width, 22-50% height of each
COCO person box) and colours are placeholders. These items exist only to measure loading,
memory, token counts, latency and JSON validity on real photographs, as the protocol's smoke test
requires. They carry annotation_status=engineering_placeholder_not_human, go to
manifests/items_smoke.jsonl, and must never be analysed.

Example:
  python -u scripts/make_smoke_inputs.py --n 5
"""
from __future__ import annotations

import argparse

from evidence_ownership.annotation.sham import propose_shams
from evidence_ownership.utils.config import get_paths
from evidence_ownership.utils.io import read_jsonl, write_jsonl
from evidence_ownership.utils.logs import get_logger

PLACEHOLDER_COLORS = {"A": "black", "B": "white"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    log = get_logger("make_smoke_inputs")
    paths = get_paths()
    imgs = {r["scene_id"]: r for r in read_jsonl(paths.manifests / "images_train2017.jsonl")}
    out = []
    for c in read_jsonl(paths.manifests / "candidates_train2017.jsonl"):
        if len(out) >= a.n:
            break
        if c["scene_id"] not in imgs:
            continue
        # placeholder rectangles must satisfy the 48 px garment minimum, so use large people only
        if any(b[2] - b[0] < 120 or b[3] - b[1] < 180 for b in (c["persons_canvas"][p]["bbox_xyxy"] for p in ("A", "B"))):
            continue
        item = dict(c)
        item["image_sha256"] = imgs[c["scene_id"]]["sha256"]
        item["study_split"] = "pilot"
        persons = {}
        for p in ("A", "B"):
            x, y, w, h = c["persons"][p]["bbox_xywh"]
            rect = [[x + .25 * w, y + .22 * h], [x + .75 * w, y + .22 * h], [x + .75 * w, y + .5 * h], [x + .25 * w, y + .5 * h]]
            persons[p] = {**c["persons"][p], "latent_color": PLACEHOLDER_COLORS[p], "garment_polygons_orig": [rect]}
        item["persons"] = persons
        shams = propose_shams(item)
        if None in shams.values():
            log.info("skip %s: no automatic sham location", c["scene_id"])
            continue
        item["sham_translations_canvas"] = shams
        item.update(cluster_id=c["scene_id"], robustness_subset=len(out) == 0,
                    annotation_status="engineering_placeholder_not_human",
                    annotation_provenance={"note": "smoke test only; no human annotation"})
        out.append(item)
    write_jsonl(paths.manifests / "items_smoke.jsonl", out)
    log.info("wrote %d smoke items -> %s", len(out), paths.manifests / "items_smoke.jsonl")


if __name__ == "__main__":
    main()
