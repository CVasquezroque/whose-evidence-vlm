"""Deterministic schematic data; never a substitute for human COCO annotation."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw
from .canvas import compute_canvas_transform, map_point


def make_synthetic_scene() -> tuple[Image.Image, dict]:
    yy, xx = np.indices((600, 800))
    texture = ((xx*7+yy*11) % 19).astype(np.uint8)
    array = np.stack([170+texture, 180+texture, 160+texture], axis=-1)
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    transform = compute_canvas_transform(800, 600)
    persons, persons_canvas = {}, {}
    for p, x, color in [('A', 170, 'blue'), ('B', 530, 'red')]:
        draw.ellipse((x+25, 80, x+75, 130), fill=(215, 172, 135))
        draw.line((x+30, 250, x+15, 380), fill='black', width=16)
        draw.line((x+70, 250, x+85, 380), fill='black', width=16)
        draw.line((x+5, 150, x-15, 240), fill=(215, 172, 135), width=12)
        draw.line((x+95, 150, x+115, 240), fill=(215, 172, 135), width=12)
        draw.rectangle((x, 140, x+100, 250), fill=color)
        bbox = [x-25, 75, 150, 315]
        persons[p] = dict(coco_ann_id=1 if p == 'A' else 2, bbox_xywh=bbox,
                          side='left' if p == 'A' else 'right', area=150*315,
                          latent_color=color,
                          garment_polygons_orig=[[[x, 140], [x+100, 140], [x+100, 250], [x, 250]]])
        persons_canvas[p] = {'bbox_xyxy': [*map_point(bbox[:2], transform), *map_point([bbox[0]+bbox[2], bbox[1]+bbox[3]], transform)]}
    item = dict(scene_id='synthetic-000001', source_image_id=0, coco_split='train2017',
                study_split='pilot', file_name='synthetic.png', width=800, height=600,
                order_key='synthetic', order_rank=0, ab_bit=0, persons=persons,
                persons_canvas=persons_canvas, bbox_iou=0.0, canvas_transform=transform,
                license={'id': 0, 'name': 'SYNTHETIC', 'url': ''}, coco_url='', flickr_url='',
                image_sha256=None, sham_translations_canvas={'A': [0, 300], 'B': [0, 300]},
                cluster_id='synthetic-000001', robustness_subset=True,
                annotation_status='exploratory_single_rater',
                annotation_provenance={'author': 'synthetic', 'rater1': None, 'rater2': None})
    return image, item
