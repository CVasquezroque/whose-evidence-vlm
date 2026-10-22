"""Synthetic renders are constructed locally; no renderer or real work root is needed."""
import json
from pathlib import Path

import pytest
import yaml

from evidence_ownership.inference.queries import build_queries
from evidence_ownership.utils.config import get_paths
from evidence_ownership.utils.hashing import canonical_json, sha256_file, sha256_text
from evidence_ownership.utils.io import read_json, write_json, write_jsonl


@pytest.fixture
def fixture_work(tmp_path, monkeypatch):
    root = tmp_path / 'work'
    config = tmp_path / 'paths.yaml'
    config.write_text(yaml.safe_dump(dict(work_root=str(root), model={'dir': str(tmp_path/'unused_model')},
        coco=dict(annotations=str(tmp_path/'annotations'), val_images=str(tmp_path/'val'), train_images=str(tmp_path/'train')))))
    monkeypatch.setenv('EOB_PATHS', str(config))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    get_paths.cache_clear()
    items = []
    for index in range(3):
        scene = f'scene-{index}'
        folder = root / 'rendered' / 'pilot' / scene
        folder.mkdir(parents=True)
        persons = {'A': dict(latent_color='blue', side='right'), 'B': dict(latent_color='red', side='left')}
        item = dict(scene_id=scene, study_split='pilot', cluster_id=scene, persons=persons,
                    robustness_subset=index == 0, annotation_status='two_rater_agreed')
        items.append(item)
        record = dict(scene_id=scene, cluster_id=scene, study_split='pilot', status='PASS' if index < 2 else 'FAIL',
                      images={}, gold={}, checks=[])
        keys = ['I11', 'I01', 'I10', 'I00', 'clean', 'iso_A', 'iso_B']
        if index == 0:
            keys += ['tagrev_' + c for c in ('I11', 'I01', 'I10', 'I00')]
        for key in keys:
            path = folder / (key + '.png')
            # Real valid PNGs support later figure/gallery and image-opening checks.
            from PIL import Image
            Image.new('RGB', (32, 32), (index, 128, 128)).save(path)
            record['images'][key] = dict(path=str(path.relative_to(root)), sha256=sha256_file(path), size=[32,32])
            cell = key.removeprefix('tagrev_')
            record['gold'][key] = {}
            for p in ('A', 'B'):
                if cell.startswith('I'):
                    visible = cell[1 if p == 'A' else 2] == '1'
                else:
                    visible = cell == 'clean' or cell == 'iso_' + p
                record['gold'][key][p] = dict(visible=visible, task_gold=persons[p]['latent_color'] if visible else 'UNKNOWN')
        write_json(folder/'render_record.json', record)
    write_jsonl(root/'manifests/items_pilot.jsonl', items)
    yield root
    get_paths.cache_clear()


def test_counts_roles_hashes_and_skips(fixture_work):
    rows = build_queries('pilot', 'unit', no_progress=True)
    assert len(rows) == 32  # 12 + 8 reversed + 12
    summary = read_json(fixture_work/'runs/unit/query_manifest_summary.json')
    assert summary['counts_by_variant'] == {'primary': 24, 'tag_reversal': 8}
    assert summary['skipped_scenes'][0]['scene_id'] == 'scene-2'
    b = next(r for r in rows if r['scene_id'] == 'scene-0' and r['image_key'] == 'I01' and r['target_person'] == 'B')
    assert (b['T'], b['D']) == (1, 0)
    a = next(r for r in rows if r['image_key'] == 'tagrev_I01' and r['target_person'] == 'A')
    assert a['target_tag'] == 'B' and a['gold']['target_latent_color'] == 'blue'
    assert 'For person B,' in a['user_prompt']
    assert a['query_id'] == sha256_text(canonical_json(a['key']))[:20]
    iso = next(r for r in rows if r['image_key'] == 'iso_B')
    assert iso['T'] is None and iso['D'] is None and iso['gold']['donor_visible'] is None
    assert build_queries('pilot', 'unit', no_progress=True) == rows


def test_limit_and_dry_build(fixture_work):
    rows = build_queries('pilot', 'dry', limit_scenes=1, dry_run=True, no_progress=True)
    assert len(rows) == 20 and all(r['evidence_level'] == 'dry_run' for r in rows)
    assert len(read_json(fixture_work/'runs/dry/skipped_scenes.json')) == 2


def test_inconsistent_gold_fails(fixture_work):
    path = fixture_work/'rendered/pilot/scene-0/render_record.json'
    rec = read_json(path)
    rec['gold']['I01']['B']['visible'] = False
    write_json(path, rec)
    with pytest.raises(ValueError, match='visibility'):
        build_queries('pilot', 'bad', no_progress=True)
