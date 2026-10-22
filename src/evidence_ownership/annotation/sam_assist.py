"""SAM-HQ point-prompt assistance for the mask author (decision D-016).

SAM proposes a garment mask from human clicks. The human accepts, edits or rejects it, and
humans remain responsible for every mask (rater audit, render checks). SAM never sees colours,
gold labels or VLM outputs. Every accepted proposal is stored with its provenance: model,
checkpoint SHA-256, the prompt points, the chosen mask's score and the polygonization settings.

Configured by the optional ``sam:`` block in the paths YAML:
    sam: {repo: /path/to/sam-hq, checkpoint: /path/to/sam_hq_vit_l.pth, model_type: vit_l, device: cuda:0}
"""
from __future__ import annotations

import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from contourpy import FillType, contour_generator
from PIL import Image

from ..utils.config import CONFIG_DIR, load_yaml
from ..utils.hashing import sha256_file

SIMPLIFY_TOL_PX = 1.0     # Ramer-Douglas-Peucker tolerance, original-image pixels
MIN_COMPONENT_PX = 64     # ignore specks smaller than this (original-image pixels)


def sam_config() -> dict | None:
    src = Path(os.environ.get("EOB_PATHS") or CONFIG_DIR / "paths.yaml")
    cfg = load_yaml(src).get("sam") if src.exists() else None
    return cfg or None


def _rdp(points: np.ndarray, tol: float) -> np.ndarray:
    """Iterative Ramer-Douglas-Peucker on an open polyline."""
    keep = np.zeros(len(points), bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = points[b] - points[a]
        rel = points[a + 1:b] - points[a]
        norm = np.hypot(*seg)
        d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / norm if norm else np.hypot(rel[:, 0], rel[:, 1])
        i = int(np.argmax(d))
        if d[i] > tol:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return points[keep]


def mask_to_polygons(mask: np.ndarray, tol: float = SIMPLIFY_TOL_PX, min_px: int = MIN_COMPONENT_PX) -> list[list[list[float]]]:
    """Outer boundaries of a binary mask as polygons in pixel coordinates.

    Holes are filled: a hole inside a garment (e.g. between arm and torso) becomes covered.
    That errs toward covering *more* (never leaving garment pixels visible); the author can edit.
    Coordinates follow PIL's convention (integer coordinates are pixel centres), so tracing the 0.5
    iso-line through pixel centres and re-rasterising with ImageDraw.polygon recovers the mask.
    """
    m = np.pad(mask.astype(np.float64), 1)
    gen = contour_generator(z=m, fill_type=FillType.OuterOffset)
    polys = []
    for pts, offsets in zip(*gen.filled(0.5, 1.5)):
        outer = pts[offsets[0]:offsets[1]] - 1          # undo padding
        x, y = outer[:, 0], outer[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area < min_px:
            continue
        simp = _rdp(outer, tol)
        if len(simp) >= 4:  # closed ring repeats the first point
            polys.append([[round(float(px), 2), round(float(py), 2)] for px, py in simp[:-1]])
    return polys


class SamAssistant:
    """Lazy, thread-safe SAM-HQ predictor with a small per-scene embedding cache."""

    def __init__(self, cfg: dict, cache_size: int = 4):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.predictor = None
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.cache_size = cache_size
        self.info: dict = {}

    def _load(self) -> None:
        import torch
        repo = str(Path(self.cfg["repo"]).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from segment_anything import SamPredictor, sam_model_registry  # SAM-HQ fork
        device = self.cfg.get("device", "cuda:0")
        ckpt = Path(self.cfg["checkpoint"])
        model = sam_model_registry[self.cfg.get("model_type", "vit_l")](checkpoint=str(ckpt))
        model.to(device).eval()
        self.predictor = SamPredictor(model)
        self.info = {"model": f"sam-hq_{self.cfg.get('model_type', 'vit_l')}", "checkpoint": ckpt.name,
                     "checkpoint_sha256": sha256_file(ckpt), "device": device,
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "torch": torch.__version__, "simplify_tol_px": SIMPLIFY_TOL_PX, "min_component_px": MIN_COMPONENT_PX}

    def predict(self, scene_id: str, image_path: Path, points: list[list[float]], labels: list[int]) -> dict:
        """points/labels in ORIGINAL image coordinates; label 1 = garment, 0 = not garment."""
        import torch
        with self.lock, torch.inference_mode():
            if self.predictor is None:
                self._load()
            if scene_id in self.cache:
                self.cache.move_to_end(scene_id)
                state = self.cache[scene_id]
            else:
                with Image.open(image_path) as im:
                    arr = np.asarray(im.convert("RGB"))
                self.predictor.set_image(arr)
                state = {k: getattr(self.predictor, k) for k in ("features", "interm_features", "original_size",
                                                                 "input_size", "is_image_set") if hasattr(self.predictor, k)}
                self.cache[scene_id] = state
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
            for k, v in state.items():
                setattr(self.predictor, k, v)
            masks, scores, _ = self.predictor.predict(point_coords=np.asarray(points, np.float32),
                                                      point_labels=np.asarray(labels, np.int32),
                                                      multimask_output=len(points) == 1, hq_token_only=False)
            # SAM-HQ's decoder already keeps only its max-IoU multimask candidate for single clicks,
            # so one mask comes back; if it is only part of the garment, the author adds points.
            areas = masks.reshape(len(masks), -1).sum(1)
            best = int(np.argmax(scores))
            mask = masks[best]
        return {"polygons_orig": mask_to_polygons(mask), "score": float(scores[best]), "mask_area_px": int(mask.sum()),
                "candidates": [{"score": float(sc), "area_px": int(a)} for sc, a in zip(scores, areas)],
                "selection_rule": "SAM-HQ internal max-IoU multimask (1 point) or single mask (>1 point)",
                "points_orig": [[round(float(x), 2), round(float(y), 2)] for x, y in points], "labels": list(map(int, labels)),
                "multimask_index": best, **self.info}
