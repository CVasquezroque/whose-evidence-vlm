"""Person boxes and deterministic Pillow embedded-font tags."""
from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .canvas import SIZE


def marker_geometry(item: dict, person: str) -> dict:
    pc = item['persons_canvas'][person]
    x0, y0, x1, y1 = pc['bbox_xyxy']
    box = [math.floor(x0)-3, math.floor(y0)-3, math.ceil(x1)+3, math.ceil(y1)+3]
    # D-017: the 4-px outline is drawn inward from `box`; keep it 1 px clear of the person's
    # dilated garment mask (inclusive pixel extent), which often reaches the tight COCO box edge.
    ext = pc.get('garment_extent_xyxy')
    if ext:
        box = [min(box[0], ext[0]-5), min(box[1], ext[1]-5), max(box[2], ext[2]+5), max(box[3], ext[3]+5)]
    # Fixed tag dimensions keep tag reversal geometry identical.
    font = ImageFont.load_default(size=22)
    bounds = [font.getbbox(letter, stroke_width=1) for letter in 'AB']
    w = max(b[2]-b[0] for b in bounds)+10
    h = max(b[3]-b[1] for b in bounds)+8
    x, y = box[0], box[1]-h-2
    if x < 0 or x+w > SIZE or y < 0:
        y = box[3]+2
    if x < 0 or x+w > SIZE or y+h > SIZE:
        x, y = max(0, min(SIZE-w, box[0]+3)), max(0, min(SIZE-h, box[1]+3))
    return {'box': box, 'tag': [x, y, x+w-1, y+h-1]}


def draw_markers(image: Image.Image, item: dict, persons: tuple[str, ...] = ('A', 'B'),
                 tag_reversal: bool = False) -> tuple[Image.Image, dict, np.ndarray]:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    touched = Image.new('L', result.size, 0)
    touch = ImageDraw.Draw(touched)
    geometries = {}
    font = ImageFont.load_default(size=22)
    for person in persons:
        geometry = marker_geometry(item, person)
        geometries[person] = geometry
        box, tag = geometry['box'], geometry['tag']
        # Four inward strokes: black, white, white, black.
        for k, color in enumerate(('black', 'white', 'white', 'black')):
            rect = [box[0]+k, box[1]+k, box[2]-k, box[3]-k]
            draw.rectangle(rect, outline=color, width=1)
            touch.rectangle(rect, outline=255, width=1)
        draw.rectangle(tag, fill='white', outline='black', width=1)
        touch.rectangle(tag, fill=255)
        letter = ('B' if person == 'A' else 'A') if tag_reversal else person
        b = font.getbbox(letter, stroke_width=1)
        draw.text((tag[0]+5-b[0], tag[1]+4-b[1]), letter, font=font, fill='black', stroke_width=1)
    return result, geometries, np.asarray(touched) != 0


def marker_keepout(item: dict) -> np.ndarray:
    """Return every box/tag pixel touched for both canonical people."""
    return draw_markers(Image.new('RGB', (SIZE, SIZE)), item)[2]


def tag_keepout(item: dict) -> np.ndarray:
    mask = Image.new('L', (SIZE, SIZE))
    draw = ImageDraw.Draw(mask)
    for person in ('A', 'B'):
        draw.rectangle(marker_geometry(item, person)['tag'], fill=255)
    return np.asarray(mask) != 0
