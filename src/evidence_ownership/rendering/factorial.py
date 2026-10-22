"""Render authoritative factorial scene artifacts on CPU."""
from __future__ import annotations
import copy
import math
import os
from pathlib import Path
import numpy as np
import PIL
from PIL import Image
from evidence_ownership.utils.config import experiment_config, get_paths
from evidence_ownership.utils.hashing import sha256_file, sha256_json
from evidence_ownership.utils.io import write_json
from .canvas import GRAY, compute_canvas_transform, content_box, to_canvas, map_point
from .masks import MaskClippingError, rasterize_polygons, dilate_mask, translate_mask, save_mask, mask_bbox
from .markers import draw_markers
from evidence_ownership.validation.render_checks import run_render_checks

RENDERER_VERSION = '1.0.0'


def validate_item(item: dict, source_image_path: str | Path) -> None:
    """Validate source integrity and structural geometry without writing."""
    if Path(item['scene_id']).name != item['scene_id'] or item['scene_id'] in ('.', '..'):
        raise ValueError('scene_id must be a safe directory name')
    with Image.open(source_image_path) as source:
        source.load()
        if source.size != (item['width'], item['height']):
            raise ValueError('Source dimensions differ from item')
    transform = compute_canvas_transform(item['width'], item['height'])
    if transform != item['canvas_transform']:
        raise ValueError('canvas_transform does not match source dimensions')
    if item.get('image_sha256') and sha256_file(source_image_path) != item['image_sha256']:
        raise ValueError('Source SHA256 mismatch')
    for p in 'AB':
        x, y, w, h = item['persons'][p]['bbox_xywh']
        expected = [*map_point([x, y], transform), *map_point([x+w, y+h], transform)]
        if not np.allclose(expected, item['persons_canvas'][p]['bbox_xyxy'], rtol=0, atol=1e-8):
            raise ValueError('persons_canvas differs from original bbox')
        rasterize_polygons(item['persons'][p]['garment_polygons_orig'], transform)
        shift = item['sham_translations_canvas'][p]
        if len(shift) != 2 or any(type(v) is not int for v in shift):
            raise ValueError('Sham translations must be two integers')
        item['persons'][p]['latent_color']
    if item['study_split'] not in ('pilot', 'heldout'):
        raise ValueError('Invalid study_split')


def render_scene(item: dict, source_image_path: str | Path, out_dir: str | Path,
                 tag_reversal: bool = False, *, dry_run: bool = False) -> dict:
    """Write a checked scene, or return checks without writes for a dry run."""
    validate_item(item, source_image_path)
    out_dir = Path(out_dir)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_image_path) as source:
        canvas = to_canvas(source)
    masks, undilated, errors = {}, {}, []
    for p in 'AB':
        undilated[p] = rasterize_polygons(item['persons'][p]['garment_polygons_orig'], item['canvas_transform'])
        masks['M_'+p] = dilate_mask(undilated[p])
        try:
            masks['S_'+p] = translate_mask(masks['M_'+p], *item['sham_translations_canvas'][p])
        except MaskClippingError as exc:
            # Never clip a mask silently. A failed scene retains an empty sham for QC.
            errors.append(f'{p}: {exc}')
            masks['S_'+p] = np.zeros((1024, 1024), bool)
    if not dry_run:
        for key, mask in masks.items():
            save_mask(mask, out_dir / f'{key}.png')
    # Markers and checks use a copy that also knows each person's dilated garment extent (D-017);
    # the stored input hash stays that of the original item.
    marked = copy.deepcopy(item)
    for p in 'AB':
        b = mask_bbox(masks['M_'+p])
        if b[2] > b[0]:
            marked['persons_canvas'][p]['garment_extent_xyxy'] = [b[0], b[1], b[2]-1, b[3]-1]
    clean, geometry, marker_pixels = draw_markers(canvas, marked)
    images = {'canvas': np.array(canvas), 'clean': np.array(clean)}
    gold = {'clean': {p: {'visible': True, 'task_gold': item['persons'][p]['latent_color']} for p in 'AB'}}
    gold['canvas'] = {p: entry.copy() for p, entry in gold['clean'].items()}
    applied = {}
    for key, condition in experiment_config()['conditions'].items():
        applied[key] = np.logical_or.reduce([masks[m] for m in condition['masks']])
        array = np.array(canvas)
        array[applied[key]] = GRAY
        images[key] = np.array(draw_markers(Image.fromarray(array), marked)[0])
        gold[key] = {p: {'visible': condition[p] == 'visible', 'task_gold': item['persons'][p]['latent_color'] if condition[p] == 'visible' else 'UNKNOWN'} for p in 'AB'}
        if item.get('robustness_subset') or tag_reversal:
            images['tagrev_'+key] = np.array(draw_markers(Image.fromarray(array), marked, tag_reversal=True)[0])
            gold['tagrev_'+key] = gold[key].copy()
    cb = content_box(item['canvas_transform'])
    for p in 'AB':
        b = item['persons_canvas'][p]['bbox_xyxy']
        rect = [max(cb[0], math.floor(b[0])), max(cb[1], math.floor(b[1])), min(cb[2], math.ceil(b[2])), min(cb[3], math.ceil(b[3]))]
        isolated = Image.new('RGB', canvas.size, GRAY)
        if rect[2] > rect[0] and rect[3] > rect[1]:
            isolated.paste(canvas.crop(tuple(rect)), tuple(rect[:2]))
        images['iso_'+p] = np.array(draw_markers(isolated, marked, persons=(p,))[0])
        gold['iso_'+p] = {p: {'visible': True, 'task_gold': item['persons'][p]['latent_color']}}
    if dry_run:
        checks = run_render_checks(marked, images, masks, undilated, applied, marker_pixels, None, errors)
        return {'scene_id': item['scene_id'], 'checks': checks,
                'status': 'FAIL' if any(c['status'] == 'FAIL' for c in checks) else 'PASS'}
    work_root = get_paths().work_root
    image_records = {}
    for key, array in images.items():
        path = out_dir / f'{key}.png'
        Image.fromarray(array).save(path, format='PNG', optimize=False)
        # Explicit output directories must still be addressable relative to work_root.
        image_records[key] = {'path': os.path.relpath(path.resolve(), work_root.resolve()), 'sha256': sha256_file(path), 'size': [1024, 1024]}
    checks = run_render_checks(marked, images, masks, undilated, applied, marker_pixels, out_dir, errors)
    measurements = {}
    for p in 'AB':
        bbox = mask_bbox(undilated[p])
        measurements[p] = {'width': bbox[2]-bbox[0], 'height': bbox[3]-bbox[1],
                           'area': int(undilated[p].sum())}
    record = dict(schema='eob.render.v1', scene_id=item['scene_id'], study_split=item['study_split'],
                  cluster_id=item.get('cluster_id', item['scene_id']), renderer_version=RENDERER_VERSION,
                  pillow_version=PIL.__version__, input_sha256=sha256_json(item),
                  source_image_sha256=sha256_file(source_image_path), canvas_transform=item['canvas_transform'],
                  mask_areas={k: int(v.sum()) for k, v in masks.items()},
                  applied_mask_area={k: int(v.sum()) for k, v in applied.items()},
                  masked_canvas_fraction={k: float(v.mean()) for k, v in applied.items()},
                  garment_bbox_canvas={p: mask_bbox(undilated[p]) for p in 'AB'},
                  garment_measurements_canvas=measurements,
                  marker_geometry=geometry, images=image_records, gold=gold, checks=checks,
                  status='FAIL' if any(c['status'] == 'FAIL' for c in checks) else 'PASS')
    write_json(out_dir / 'render_record.json', record)
    return record
