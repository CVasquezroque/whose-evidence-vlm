"""CPU geometry, authoritative artifacts, gold and determinism tests."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from evidence_ownership.rendering.canvas import compute_canvas_transform, to_canvas, map_point, content_box
from evidence_ownership.rendering.masks import dilate_mask, translate_mask, MaskClippingError, load_mask, save_mask, rasterize_polygons
from evidence_ownership.rendering.markers import marker_keepout, tag_keepout, marker_geometry
from evidence_ownership.rendering.factorial import render_scene
from evidence_ownership.rendering.synthetic import make_synthetic_scene


def test_transform_no_upscale_and_odd_sizes():
    t = compute_canvas_transform(801, 599)
    assert t['scale'] == 1
    assert t['offset'] == [111, 212]
    assert content_box(t) == [111, 212, 912, 811]
    image = Image.new('RGB', (801, 599), 'blue')
    canvas = np.array(to_canvas(image))
    assert np.all(canvas[212:811, 111:912] == [0, 0, 255])
    assert np.all(canvas[:212] == 128)
    t = compute_canvas_transform(2001, 1000)
    assert t['resized_size'] == [1024, round(1000*1024/2001)]
    assert map_point([2001, 1000], t) == (1024, t['offset'][1]+t['resized_size'][1])
    assert map_point([100, 100], t)[1] == 100*t['resized_size'][1]/1000+t['offset'][1]
    with pytest.raises(ValueError):
        compute_canvas_transform(0, 1)


def test_raster_dilation_translation_and_storage(tmp_path):
    t = compute_canvas_transform(1024, 1024)
    raw = rasterize_polygons([[[20, 20], [30, 20], [30, 30], [20, 30]]], t)
    assert raw.dtype == np.bool_
    assert raw.sum() == 121
    mask = dilate_mask(raw)
    assert mask.sum() == 19**2
    assert mask[16:35, 16:35].all()
    shifted = translate_mask(mask, -16, -16)
    assert shifted.sum() == mask.sum()
    with pytest.raises(MaskClippingError):
        translate_mask(mask, -17, 0)
    with pytest.raises(MaskClippingError):
        translate_mask(mask, 1000, 1000)
    with pytest.raises(ValueError):
        translate_mask(mask, 0.5, 0)
    save_mask(mask, tmp_path/'mask.png')
    assert np.array_equal(mask, load_mask(tmp_path/'mask.png'))


@pytest.fixture(scope='module')
def rendered(tmp_path_factory):
    root = tmp_path_factory.mktemp('rendered')
    image, item = make_synthetic_scene()
    source = root/'source.png'
    image.save(source)
    output = root/'scene'
    record = render_scene(item, source, output)
    return item, source, output, record


def test_synthetic_all_invariants_pass(rendered):
    _, _, _, record = rendered
    assert record['status'] == 'PASS', record['checks']
    assert all(c['status'] == 'PASS' for c in record['checks'])
    assert len(set(record['applied_mask_area'].values())) == 1


def test_gold_by_canonical_person(rendered):
    item, _, _, record = rendered
    assert set(record['gold']) == set(record['images'])
    for condition, visible in [('I11', (True, True)), ('I01', (False, True)), ('I10', (True, False)), ('I00', (False, False)), ('clean', (True, True))]:
        for p, present in zip('AB', visible):
            assert record['gold'][condition][p] == {'visible': present, 'task_gold': item['persons'][p]['latent_color'] if present else 'UNKNOWN'}
        if condition != 'clean':
            assert record['gold']['tagrev_'+condition] == record['gold'][condition]
    for p in 'AB':
        assert record['gold']['iso_'+p] == {p: {'visible': True, 'task_gold': item['persons'][p]['latent_color']}}


def test_rerender_bytes_identical(rendered):
    item, source, output, _ = rendered
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    render_scene(item, source, output)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_reversed_tags_only_and_isolation(rendered):
    item, _, output, _ = rendered
    tags = tag_keepout(item)
    for key in ('I11', 'I01', 'I10', 'I00'):
        normal = np.array(Image.open(output/f'{key}.png'))
        reverse = np.array(Image.open(output/f'tagrev_{key}.png'))
        assert np.array_equal(normal[~tags], reverse[~tags])
        assert np.any(normal[tags] != reverse[tags])
    canvas = np.array(Image.open(output/'canvas.png'))
    isolated = np.array(Image.open(output/'iso_A.png'))
    assert np.array_equal(isolated[360:450, 300:350], canvas[360:450, 300:350])
    assert np.all(isolated[360:450, 660:700] == 128)


def test_tag_placement_fallbacks():
    _, item = make_synthetic_scene()
    item['persons_canvas']['A']['bbox_xyxy'] = [20, 4, 100, 100]
    assert marker_geometry(item, 'A')['tag'][1] > 100
    item['persons_canvas']['A']['bbox_xyxy'] = [20, 4, 100, 1020]
    assert marker_geometry(item, 'A')['tag'][1] >= 4
    assert marker_keepout(item).dtype == np.bool_


def test_source_integrity_rejected(rendered, tmp_path):
    import copy
    item, source, _, _ = rendered
    broken = copy.deepcopy(item)
    broken['image_sha256'] = 'incorrect'
    with pytest.raises(ValueError, match='SHA256'):
        render_scene(broken, source, tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_no_tag_reversal_when_not_requested(tmp_path):
    image, item = make_synthetic_scene()
    item['robustness_subset'] = False
    source = tmp_path/'source.png'
    image.save(source)
    record = render_scene(item, source, tmp_path/'scene')
    assert record['status'] == 'PASS'
    assert not any(k.startswith('tagrev_') for k in record['images'])
    assert next(c for c in record['checks'] if c['name'] == 'tag_reversal_only_tags')['status'] == 'SKIPPED'


def test_cli_dry_run_qc_and_resume(tmp_path):
    import json
    import os
    import subprocess
    import sys
    image, item = make_synthetic_scene()
    images = tmp_path/'images'
    images.mkdir()
    image.save(images/item['file_name'])
    paths_file = tmp_path/'paths.yaml'
    paths_file.write_text(json.dumps({'coco': {'annotations': str(tmp_path), 'val_images': str(images), 'train_images': str(images)}, 'model': {'dir': str(tmp_path/'unused')}, 'work_root': str(tmp_path/'work')}))
    manifest = tmp_path/'items.jsonl'
    manifest.write_text(json.dumps(item)+'\n')
    output = tmp_path/'work'/'rendered'/'pilot'
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'EOB_PATHS': str(paths_file)}
    command = [sys.executable, str(repo/'scripts'/'render_factorial.py'), '--items', str(manifest), '--split', 'pilot', '--no-progress', '--qc-sheet']
    result = subprocess.run(command+['--dry-run'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not output.exists()
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (output/'_qc'/'index.html').exists()
    assert (output/'_qc'/f'{item["scene_id"]}.png').exists()
    record_path = output/item['scene_id']/'render_record.json'
    before = record_path.stat().st_mtime_ns
    result = subprocess.run(command+['--resume'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert before == record_path.stat().st_mtime_ns
    summary = json.loads((output/'render_summary.json').read_text())
    assert summary['PASS'] == 1 and summary['FAIL'] == 0 and summary['skipped'] == 1
    # Changing the manifest defeats resume and is validated in a no-write dry run.
    item['sham_translations_canvas']['A'] = [2000, 0]
    manifest.write_text(json.dumps(item)+'\n')
    result = subprocess.run(command+['--dry-run', '--resume'], env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert before == record_path.stat().st_mtime_ns
