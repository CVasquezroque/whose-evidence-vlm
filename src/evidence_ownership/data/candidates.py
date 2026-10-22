"""Deterministic geometric candidate screen (protocol Section V.1-V.2, criteria 1-2).

A candidate passes only the *automatic* criteria: exactly two non-crowd COCO person
annotations, no crowd person annotation, and person-box IoU below the threshold. Every other
criterion (extra unannotated people, garment colors, maskability, shams) requires human review
and is NOT asserted here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..utils.hashing import order_key, sha256_text
from .coco import CocoIndex, ImageRecord

STUDY_SPLIT = {"train2017": "pilot", "val2017": "heldout"}
CANVAS = 1024


def bbox_iou_xywh(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def canvas_transform(width: int, height: int, canvas: int = CANVAS) -> dict:
    """Must agree with rendering.canvas (data_contracts.md §1); tested for equality."""
    s = min(1.0, canvas / width, canvas / height)
    w2, h2 = (round(width * s), round(height * s)) if s < 1.0 else (width, height)
    return {"scale": s, "offset": [(canvas - w2) // 2, (canvas - h2) // 2],
            "resized_size": [w2, h2], "canvas_size": [canvas, canvas], "orig_size": [width, height]}


def bbox_to_canvas(bbox_xywh, tf: dict) -> list[float]:
    x, y, w, h = bbox_xywh
    (W, H), (w2, h2), (ox, oy) = tf["orig_size"], tf["resized_size"], tf["offset"]
    sx, sy = w2 / W, h2 / H
    return [x * sx + ox, y * sy + oy, (x + w) * sx + ox, (y + h) * sy + oy]


def ab_bit(salt: str, image_id: int) -> int:
    return int(sha256_text(f"{salt}{int(image_id)}"), 16) & 1


@dataclass(frozen=True)
class ScreenResult:
    image_id: int
    passed: bool
    reason: str  # "pass" or the first failed automatic criterion


def screen_image(im: ImageRecord, iou_max_exclusive: float) -> ScreenResult:
    if not im.persons:
        return ScreenResult(im.image_id, False, "no_person")
    if any(p.iscrowd for p in im.persons):
        return ScreenResult(im.image_id, False, "crowd_person_annotation")
    if len(im.persons) != 2:
        return ScreenResult(im.image_id, False, "not_exactly_two_persons")
    if bbox_iou_xywh(im.persons[0].bbox_xywh, im.persons[1].bbox_xywh) >= iou_max_exclusive:
        return ScreenResult(im.image_id, False, "person_bbox_iou_too_high")
    return ScreenResult(im.image_id, True, "pass")


def size_prescreen(persons_canvas: dict, min_w: float, min_h: float) -> bool:
    """Strictly necessary condition for criterion 5: a garment cannot be wider/taller than its
    person's box, so a canvas person box below the garment minimum can never be eligible."""
    for p in ("A", "B"):
        x0, y0, x1, y1 = persons_canvas[p]["bbox_xyxy"]
        if x1 - x0 < min_w or y1 - y0 < min_h:
            return False
    return True


def candidate_record(im: ImageRecord, index: CocoIndex, cfg: dict) -> dict:
    p0, p1 = im.persons
    c0 = p0.bbox_xywh[0] + p0.bbox_xywh[2] / 2
    c1 = p1.bbox_xywh[0] + p1.bbox_xywh[2] / 2
    left, right = (p0, p1) if c0 <= c1 else (p1, p0)  # ties: lower ann_id is left (persons sorted)
    bit = ab_bit(cfg["ab_assignment_salt"], im.image_id)
    a, b = (left, right) if bit == 0 else (right, left)
    tf = canvas_transform(im.width, im.height)

    def person(p, side):
        return {"coco_ann_id": p.ann_id, "bbox_xywh": list(p.bbox_xywh), "side": side, "area": p.area}

    side = {left.ann_id: "left", right.ann_id: "right"}
    pc = {"A": {"bbox_xyxy": bbox_to_canvas(a.bbox_xywh, tf)}, "B": {"bbox_xyxy": bbox_to_canvas(b.bbox_xywh, tf)}}
    el = cfg["eligibility"]
    return {
        "scene_id": f"{index.split}-{im.image_id:012d}",
        "source_image_id": im.image_id,
        "coco_split": index.split,
        "study_split": STUDY_SPLIT[index.split],
        "file_name": im.file_name,
        "width": im.width,
        "height": im.height,
        "order_key": order_key(cfg["hash_salt"], im.image_id),
        "order_rank": None,
        "ab_bit": bit,
        "persons": {"A": person(a, side[a.ann_id]), "B": person(b, side[b.ann_id])},
        "bbox_iou": bbox_iou_xywh(p0.bbox_xywh, p1.bbox_xywh),
        "canvas_transform": tf,
        "persons_canvas": pc,
        # Automatic, strictly necessary pre-screen; failing scenes are queued last, never annotated as eligible.
        "size_prescreen_pass": size_prescreen(pc, el["min_garment_width_px"], el["min_garment_height_px"]),
        "license": index.licenses.get(im.license_id, {"id": im.license_id, "name": None, "url": None}),
        "coco_url": im.coco_url,
        "flickr_url": im.flickr_url,
        "image_sha256": None,
    }


def build_candidates(index: CocoIndex, cfg: dict, progress=lambda it, **kw: it) -> tuple[list[dict], dict]:
    """Return (candidates sorted by protocol order key with ranks, screen statistics)."""
    thr = cfg["eligibility"]["person_bbox_iou_max_exclusive"]
    stats: dict[str, int] = {}
    out = []
    for im in progress(index.images.values(), total=len(index.images), desc=f"screen {index.split}", unit="image"):
        r = screen_image(im, thr)
        stats[r.reason] = stats.get(r.reason, 0) + 1
        if r.passed:
            out.append(candidate_record(im, index, cfg))
    out.sort(key=lambda c: c["order_key"])
    for rank, c in enumerate(out):
        c["order_rank"] = rank
    stats["size_prescreen_pass"] = sum(c["size_prescreen_pass"] for c in out)
    stats["images_total"] = len(index.images)
    stats["images_with_person"] = len(index.images) - stats.get("no_person", 0)
    return out, stats
