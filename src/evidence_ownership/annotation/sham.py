"""Automatic sham-location proposals (an aid for the mask author; humans still confirm).

Mirrors the annotation interface: a sham is the dilated garment mask translated by an integer
offset such that its bounding rectangle lies inside the image content, away from both person
boxes (plus marker/tag zones), both garment masks and the other sham.
"""
from __future__ import annotations

import math

import numpy as np

from ..rendering.canvas import SIZE, content_box
from ..rendering.masks import dilate_mask, mask_bbox, rasterize_polygons

KEEPOUT_PAD = 12
TAG_ZONE = 34


def forbidden_map(item: dict, masks: dict[str, np.ndarray]) -> np.ndarray:
    F = np.zeros((SIZE, SIZE), bool)
    for p in ("A", "B"):
        x0, y0, x1, y1 = item["persons_canvas"][p]["bbox_xyxy"]
        if masks.get(p) is not None and masks[p].any():  # marker box also encloses the garment mask (D-017)
            mx0, my0, mx1, my1 = mask_bbox(masks[p])
            x0, y0, x1, y1 = min(x0, mx0 - 5), min(y0, my0 - 5), max(x1, mx1 + 4), max(y1, my1 + 4)
        F[max(0, math.floor(y0 - KEEPOUT_PAD - TAG_ZONE)):min(SIZE, math.ceil(y1 + KEEPOUT_PAD + TAG_ZONE) + 1),
          max(0, math.floor(x0 - KEEPOUT_PAD)):min(SIZE, math.ceil(x1 + KEEPOUT_PAD) + 1)] = True
    for m in masks.values():
        F |= m
    return F


def propose_translation(mask: np.ndarray, forbidden: np.ndarray, cbox: list[int], step: int = 4) -> tuple[int, int] | None:
    """Nearest (dx, dy) whose translated mask bounding box contains no forbidden pixel."""
    S = np.zeros((SIZE + 1, SIZE + 1), np.int64)
    S[1:, 1:] = forbidden.astype(np.int64).cumsum(0).cumsum(1)
    bx0, by0, bx1, by1 = mask_bbox(mask)  # exclusive upper bounds
    w, h = bx1 - bx0, by1 - by0
    best = None
    for ny in range(cbox[1], cbox[3] - h + 1, step):
        for nx in range(cbox[0], cbox[2] - w + 1, step):
            if S[ny + h, nx + w] - S[ny, nx + w] - S[ny + h, nx] + S[ny, nx]:
                continue
            dx, dy = nx - bx0, ny - by0
            d = dx * dx + dy * dy
            if best is None or d < best[2]:
                best = (dx, dy, d)
    return None if best is None else (int(best[0]), int(best[1]))


def propose_shams(item: dict) -> dict[str, list[int] | None]:
    tf = item["canvas_transform"]
    M = {p: dilate_mask(rasterize_polygons(item["persons"][p]["garment_polygons_orig"], tf)) for p in ("A", "B")}
    out: dict[str, list[int] | None] = {}
    placed: dict[str, np.ndarray] = dict(M)
    for p in ("A", "B"):
        t = propose_translation(M[p], forbidden_map(item, placed), content_box(tf))
        out[p] = None if t is None else list(t)
        if t is not None:
            placed["S_" + p] = np.roll(np.roll(M[p], t[1], 0), t[0], 1)
    return out
