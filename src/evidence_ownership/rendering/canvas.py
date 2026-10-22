"""Deterministic original-to-canvas geometry."""
from __future__ import annotations
from typing import Any
from PIL import Image

SIZE = 1024
GRAY = (128, 128, 128)


def compute_canvas_transform(width: int, height: int) -> dict[str, Any]:
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive')
    scale = min(1.0, SIZE / width, SIZE / height)
    w, h = round(width * scale), round(height * scale)
    if not w or not h:
        raise ValueError('Resized image has a zero dimension')
    return dict(scale=scale, offset=[(SIZE-w)//2, (SIZE-h)//2],
                resized_size=[w, h], canvas_size=[SIZE, SIZE], orig_size=[width, height])


def map_point(point: tuple[float, float] | list[float], transform: dict) -> tuple[float, float]:
    w, h = transform['orig_size']
    rw, rh = transform['resized_size']
    ox, oy = transform['offset']
    return point[0]*rw/w+ox, point[1]*rh/h+oy


def content_box(transform: dict) -> list[int]:
    ox, oy = transform['offset']
    w, h = transform['resized_size']
    return [ox, oy, ox+w, oy+h]


def to_canvas(img: Image.Image) -> Image.Image:
    transform = compute_canvas_transform(*img.size)
    rgb = img.convert('RGB')
    if transform['scale'] != 1:
        rgb = rgb.resize(tuple(transform['resized_size']), Image.Resampling.LANCZOS)
    result = Image.new('RGB', (SIZE, SIZE), GRAY)
    result.paste(rgb, tuple(transform['offset']))
    return result
