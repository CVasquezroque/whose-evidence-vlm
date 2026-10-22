"""Binary garment rasterization and lossless integer translations."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from .canvas import SIZE, map_point


class MaskClippingError(ValueError):
    """Translation would discard foreground pixels."""


def rasterize_polygons(polygons: list, transform: dict) -> np.ndarray:
    image = Image.new('L', (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        if len(polygon) < 3:
            raise ValueError('A polygon needs at least three vertices')
        points = [map_point(p, transform) for p in polygon]
        if not np.isfinite(points).all():
            raise ValueError('Polygon coordinates must be finite')
        draw.polygon(points, fill=255, outline=255)
    return np.asarray(image) != 0


def dilate_mask(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(mask, structure=np.ones((9, 9), dtype=bool))


def translate_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if isinstance(dx, bool) or isinstance(dy, bool) or not isinstance(dx, (int, np.integer)) or not isinstance(dy, (int, np.integer)):
        raise ValueError('Sham translations must be integers')
    y, x = np.nonzero(mask)
    h, w = mask.shape
    if len(x) and (x.min()+dx < 0 or x.max()+dx >= w or y.min()+dy < 0 or y.max()+dy >= h):
        raise MaskClippingError(f'Translation {(dx, dy)} leaves canvas')
    result = np.zeros_like(mask, dtype=bool)
    result[y+dy, x+dx] = True
    return result


def mask_bbox(mask: np.ndarray) -> list[int]:
    y, x = np.nonzero(mask)
    return [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)] if len(x) else [0, 0, 0, 0]


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    Image.fromarray(mask.astype(np.uint8)*255).save(path, format='PNG', optimize=False)


def load_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
        if image.format != 'PNG' or image.mode != 'L' or image.size != (SIZE, SIZE) or not np.isin(array, [0, 255]).all():
            raise ValueError('Expected 1024x1024 L binary PNG')
        return array == 255
