"""Array-level rendering invariants, independent of visual judgment."""
from __future__ import annotations
import math
from itertools import combinations
from pathlib import Path
import numpy as np
from PIL import Image
from evidence_ownership.utils.config import experiment_config
from evidence_ownership.rendering.canvas import content_box
from evidence_ownership.rendering.masks import mask_bbox, load_mask, translate_mask, MaskClippingError
from evidence_ownership.rendering.markers import tag_keepout


def check_result(name: str, passed: bool, detail: object = '') -> dict:
    return {'name': name, 'status': 'PASS' if passed else 'FAIL', 'detail': str(detail)}


def run_render_checks(item: dict, images: dict, masks: dict, undilated: dict,
                      applied: dict, marker_pixels: np.ndarray, out_dir: Path | None,
                      translation_errors: list[str] | None = None) -> list[dict]:
    checks = []
    def add(name: str, ok: bool, detail: object = '') -> None:
        checks.append(check_result(name, bool(ok), detail))
    if out_dir is None:
        checks.append({'name': 'image_roundtrip', 'status': 'SKIPPED', 'detail': 'Dry run: no files written'})
    else:
        bad = []
        for key, array in images.items():
            try:
                with Image.open(out_dir / f'{key}.png') as image:
                    if image.format != 'PNG' or image.mode != 'RGB' or image.size != (1024, 1024) or not np.array_equal(np.asarray(image), array):
                        bad.append(key)
            except (ValueError, OSError):
                bad.append(key)
        add('image_roundtrip', not bad, bad)
        bad_masks = []
        for key, mask in masks.items():
            try:
                if not np.array_equal(load_mask(out_dir / f'{key}.png'), mask):
                    bad_masks.append(key)
            except (ValueError, OSError):
                bad_masks.append(key)
        add('mask_roundtrip', not bad_masks, bad_masks)
    bad = [f'{a}/{b}' for a, b in combinations(masks, 2) if np.any(masks[a] & masks[b])]
    add('masks_disjoint', not bad, bad)
    clipping = list(translation_errors or [])
    matching_shams = True
    for p in 'AB':
        try:
            expected = translate_mask(masks['M_'+p], *item['sham_translations_canvas'][p])
            matching_shams &= np.array_equal(expected, masks['S_'+p])
        except MaskClippingError as exc:
            clipping.append(f'{p}: {exc}')
            matching_shams = False
    add('no_clipping', not clipping, clipping)
    add('sham_translation', matching_shams, 'Shams must be exact integer translations')
    box = content_box(item['canvas_transform'])
    inside = np.zeros((1024, 1024), bool)
    inside[box[1]:box[3], box[0]:box[2]] = True
    add('shams_inside_content', all(not np.any(masks['S_'+p] & ~inside) for p in 'AB'))
    outside = True
    yy, xx = np.indices((1024, 1024))
    for p in 'AB':
        x0, y0, x1, y1 = item['persons_canvas'][p]['bbox_xyxy']
        person = (xx >= math.floor(x0)) & (xx < math.ceil(x1)) & (yy >= math.floor(y0)) & (yy < math.ceil(y1))
        outside &= not np.any(person & (masks['S_A'] | masks['S_B']))
    add('shams_outside_person_bboxes', outside)
    areas = {key: int(mask.sum()) for key, mask in applied.items()}
    add('equal_applied_mask_area', len(set(areas.values())) == 1, areas)
    limit = experiment_config()['eligibility']['max_masked_canvas_fraction']
    add('masked_fraction', all(area / 1024**2 <= limit for area in areas.values()), areas)
    for p in 'AB':
        b = mask_bbox(undilated[p])
        values = dict(width=b[2]-b[0], height=b[3]-b[1], area=int(undilated[p].sum()))
        cfg = experiment_config()['eligibility']
        # A02: inclusion uses the expanded screen; every cohort flag is recorded for analysis.
        scr = cfg.get('inclusion_screen', {'min_w': cfg['min_garment_width_px'], 'min_h': cfg['min_garment_height_px'], 'min_area': cfg['min_garment_area_px']})
        values['cohorts'] = {k: values['width'] >= v['min_w'] and values['height'] >= v['min_h'] and values['area'] >= v['min_area']
                             for k, v in cfg.get('size_cohorts', {}).items()}
        add('garment_size_'+p, values['width'] >= scr['min_w'] and values['height'] >= scr['min_h'] and values['area'] >= scr['min_area'], values)
    for key, mask in applied.items():
        outside = ~mask & ~marker_pixels
        add('pixel_preservation_'+key, np.array_equal(images[key][outside], images['canvas'][outside]) and np.all(images[key][mask] == 128))
        hidden = [p for p in 'AB' if experiment_config()['conditions'][key][p] == 'hidden']
        add('complete_removal_'+key, all(np.all(images[key][masks['M_'+p]] == 128) for p in hidden))
    # D-019: catches polygons saved under the wrong person (most garment pixels outside own box).
    for p in 'AB':
        x0, y0, x1, y1 = item['persons_canvas'][p]['bbox_xyxy']
        own = (xx >= math.floor(x0)) & (xx < math.ceil(x1)) & (yy >= math.floor(y0)) & (yy < math.ceil(y1))
        n = int(undilated[p].sum())
        frac = float((undilated[p] & own).sum() / n) if n else 0.0
        add('garment_inside_own_box_'+p, frac >= 0.5, {'fraction_inside': round(frac, 3)})
    union = np.logical_or.reduce(list(masks.values()))
    add('marker_keepout', not np.any(union & marker_pixels))
    tags = tag_keepout(item)
    reversed_keys = [k for k in images if k.startswith('tagrev_')]
    if reversed_keys:
        add('tag_reversal_only_tags', all(np.array_equal(images[k][~tags], images[k.removeprefix('tagrev_')][~tags]) for k in reversed_keys))
    else:
        checks.append({'name': 'tag_reversal_only_tags', 'status': 'SKIPPED', 'detail': 'No tag reversal requested'})
    colors = [item['persons'][p]['latent_color'] for p in 'AB']
    add('distinct_valid_colors', len(set(colors)) == 2 and all(c in experiment_config()['garment_colors'] for c in colors), colors)
    return checks
