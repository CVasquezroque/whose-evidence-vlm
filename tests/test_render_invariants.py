"""Deliberate geometry and pixel corruptions must be machine-detected."""
from __future__ import annotations
import numpy as np
import pytest
from PIL import Image
from evidence_ownership.rendering.factorial import render_scene
from evidence_ownership.rendering.synthetic import make_synthetic_scene
from evidence_ownership.rendering.masks import load_mask, rasterize_polygons
from evidence_ownership.rendering.markers import marker_keepout
from evidence_ownership.validation.render_checks import run_render_checks


def failed(record):
    return {c['name'] for c in record['checks'] if c['status'] == 'FAIL'}


@pytest.mark.parametrize(('mutation', 'expected'), [
    ('overlap', 'masks_disjoint'), ('padding', 'shams_inside_content'),
    ('person', 'shams_outside_person_bboxes'), ('clipping', 'no_clipping'),
    ('colors', 'distinct_valid_colors'),
    ('small', 'garment_size_A'),
    ('invalid_color', 'distinct_valid_colors'),
])
def test_broken_items_fail(tmp_path, mutation, expected):
    image, item = make_synthetic_scene()
    source = tmp_path/'source.png'
    image.save(source)
    if mutation == 'overlap':
        item['sham_translations_canvas']['A'] = [360, 300]
    elif mutation == 'padding':
        item['sham_translations_canvas']['A'] = [0, -300]
    elif mutation == 'person':
        item['sham_translations_canvas']['A'] = [0, 120]
    elif mutation == 'clipping':
        item['sham_translations_canvas']['A'] = [2000, 0]
    elif mutation == 'colors':
        item['persons']['A']['latent_color'] = 'red'
    elif mutation == 'invalid_color':
        item['persons']['A']['latent_color'] = 'cyan'
    elif mutation == 'small':
        item['persons']['A']['garment_polygons_orig'] = [[[180, 150], [190, 150], [190, 160], [180, 160]]]
    record = render_scene(item, source, tmp_path/'scene')
    assert record['status'] == 'FAIL'
    assert expected in failed(record), record['checks']
    assert (tmp_path/'scene'/'I00.png').exists()
    if mutation in ('clipping', 'overlap'):
        assert 'equal_applied_mask_area' in failed(record)


@pytest.fixture()
def state(tmp_path):
    image, item = make_synthetic_scene()
    source = tmp_path/'source.png'
    image.save(source)
    output = tmp_path/'scene'
    record = render_scene(item, source, output)
    images = {k: np.array(Image.open(output/f'{k}.png')) for k in record['images']}
    masks = {k: load_mask(output/f'{k}.png') for k in record['mask_areas']}
    raw = {p: rasterize_polygons(item['persons'][p]['garment_polygons_orig'], item['canvas_transform']) for p in 'AB'}
    applied = {'I11': masks['S_A']|masks['S_B'], 'I01': masks['M_A']|masks['S_B'], 'I10': masks['S_A']|masks['M_B'], 'I00': masks['M_A']|masks['M_B']}
    return item, images, masks, raw, applied, marker_keepout(item), output


def test_corrupt_pixels_detected(state):
    item, images, masks, raw, applied, markers, output = state
    images['I01'][masks['M_A']] = 0
    images['I11'][0, 0] = 0
    images['tagrev_I11'][0, 0] = 255
    checks = run_render_checks(item, images, masks, raw, applied, markers, output)
    failures = failed({'checks': checks})
    assert {'image_roundtrip', 'pixel_preservation_I01', 'pixel_preservation_I11', 'complete_removal_I01', 'tag_reversal_only_tags'} <= failures


def test_impossible_edit_budget_detected(state):
    item, images, masks, raw, applied, markers, output = state
    # Simulate an illegally rescaled sham; integer translations cannot change area.
    masks['S_A'] = masks['S_A'].copy()
    y, x = np.argwhere(masks['S_A'])[0]
    masks['S_A'][y, x] = False
    applied['I11'] = masks['S_A'] | masks['S_B']
    applied['I10'] = masks['S_A'] | masks['M_B']
    checks = run_render_checks(item, images, masks, raw, applied, markers, output)
    assert 'equal_applied_mask_area' in failed({'checks': checks})


def test_excessive_mask_fraction_detected(state):
    item, images, masks, raw, applied, markers, output = state
    applied['I00'][:] = True
    checks = run_render_checks(item, images, masks, raw, applied, markers, output)
    assert 'masked_fraction' in failed({'checks': checks})


def test_bad_png_modes_detected(state):
    item, images, masks, raw, applied, markers, output = state
    Image.fromarray(images['I11']).convert('L').save(output/'I11.png')
    Image.fromarray((masks['M_A']*255).astype(np.uint8)).convert('RGB').save(output/'M_A.png')
    checks = run_render_checks(item, images, masks, raw, applied, markers, output)
    assert {'image_roundtrip', 'mask_roundtrip'} <= failed({'checks': checks})


def test_marker_over_mask_detected(state):
    # D-017 moves outlines clear of each person's own garment, so simulate a marker crossing a sham.
    item, images, masks, raw, applied, markers, output = state
    checks = run_render_checks(item, images, masks, raw, applied, markers | masks['S_A'], output)
    assert 'marker_keepout' in failed({'checks': checks})
