"""Build deterministic query manifests from audited PASS renders."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from evidence_ownership.inference.prompts import build_prompt, prompt_hash
from evidence_ownership.utils.config import CONFIG_DIR, experiment_config, get_paths
from evidence_ownership.utils.hashing import sha256_file, sha256_json
from evidence_ownership.utils.io import read_json, read_jsonl, write_json, write_jsonl
from evidence_ownership.utils.logs import progress

CELLS = ('I11', 'I01', 'I10', 'I00')


def build_queries(split: str, run_name: str, *, output_dir: Path | None = None,
                  work_root: Path | None = None, limit_scenes: int | None = None,
                  dry_run: bool = False, no_progress: bool = False) -> list[dict]:
    root = Path(work_root) if work_root is not None else get_paths().work_root
    out = Path(output_dir) if output_dir is not None else root / 'runs' / run_name
    items_path = root / 'manifests' / f'items_{split}.jsonl'
    items = {x['scene_id']: x for x in read_jsonl(items_path)} if items_path.exists() else {}
    queries, skipped = [], []
    passed = 0
    records = sorted((root / 'rendered' / split).glob('*/render_record.json'))
    for path in progress(records, desc='Build queries', unit='scene', disable=no_progress):
        rec = read_json(path)
        if rec['status'] != 'PASS':
            skipped.append({'scene_id': rec['scene_id'], 'status': rec['status'],
                            'checks': rec.get('checks', []), 'render_record': str(path.relative_to(root))})
            continue
        if limit_scenes is not None and passed >= limit_scenes:
            skipped.append({'scene_id': rec['scene_id'], 'status': 'SKIPPED', 'reason': 'limit_scenes'})
            continue
        passed += 1
        item = items.get(rec['scene_id'], rec)
        persons = item.get('persons', rec.get('persons', {}))
        # The render contract intentionally keeps latent colors in items, not hidden-view gold.
        colors = {}
        for person in ('A', 'B'):
            colors[person] = persons.get(person, {}).get('latent_color') or rec['gold']['clean'][person]['task_gold']
            if colors[person] not in experiment_config()['garment_colors']:
                raise ValueError(f"Invalid latent color for {rec['scene_id']}/{person}")
        if colors['A'] == colors['B']:
            raise ValueError('Distinct latent garment colors are required')
        tag_images = {f'tagrev_{c}' for c in CELLS} & rec['images'].keys()
        if tag_images and len(tag_images) != 4:
            raise ValueError('Partial tag-reversal render')
        robust = item.get('robustness_subset', rec.get('robustness_subset', len(tag_images) == 4))
        if robust and len(tag_images) != 4:
            raise ValueError('Robustness-subset scene lacks tag-reversal images')
        specs = [('primary', c, c, p) for c in CELLS for p in ('A', 'B')]
        specs += [('primary', 'clean', 'clean', p) for p in ('A', 'B')]
        specs += [('primary', 'iso', f'iso_{p}', p) for p in ('A', 'B')]
        if robust:
            specs += [('tag_reversal', c, f'tagrev_{c}', p) for c in CELLS for p in ('A', 'B')]
        for variant, condition, image_key, target in specs:
            donor = 'B' if target == 'A' else 'A'
            tag = donor if variant == 'tag_reversal' else target
            image = rec['images'][image_key]
            gold_target = rec['gold'][image_key][target]
            visible = bool(gold_target['visible'])
            expected_gold = colors[target] if visible else 'UNKNOWN'
            if gold_target['task_gold'] != expected_gold:
                raise ValueError(f'Task gold inconsistent with visibility for {image_key}/{target}')
            donor_visible = None if condition == 'iso' else bool(rec['gold'][image_key][donor]['visible'])
            factorial = condition in CELLS
            if factorial:
                av, bv = int(condition[1]), int(condition[2])
                expected = (av, bv) if target == 'A' else (bv, av)
                if (int(visible), int(donor_visible)) != expected:
                    raise ValueError(f'Gold visibility inconsistent with {image_key}/{target}')
            prompt = build_prompt(tag)
            key = dict(scene_id=rec['scene_id'], variant=variant, image_key=image_key,
                       target_person=target, target_tag=tag, image_sha256=image['sha256'],
                       prompt_sha256=prompt['prompt_sha256'])
            queries.append(dict(query_id=sha256_json(key)[:20], key=key,
                                scene_id=rec['scene_id'], cluster_id=rec['cluster_id'], study_split=split,
                                variant=variant, condition=condition, image_key=image_key,
                                target_person=target, target_tag=tag,
                                T=int(visible) if factorial else None,
                                D=int(donor_visible) if factorial else None,
                                image_path=image['path'], image_sha256=image['sha256'], **prompt,
                                gold=dict(task_gold=gold_target['task_gold'], target_latent_color=colors[target],
                                          donor_latent_color=colors[donor], target_visible=visible,
                                          donor_visible=donor_visible),
                                annotation_status=item.get('annotation_status', 'unknown'),
                                target_side=persons.get(target, {}).get('side'),
                                evidence_level='dry_run' if dry_run else split))
    summary = dict(n_scenes=passed, n_queries=len(queries), skipped_scenes=skipped,
                   counts_by_condition=dict(Counter(q['condition'] for q in queries)),
                   counts_by_variant=dict(Counter(q['variant'] for q in queries)), dry_run=dry_run)
    # Never overwrite a running manifest or its lock with a different build.
    if list(out.glob('raw_predictions*.jsonl')) or list(out.glob('dry_run_predictions*.jsonl')):
        if read_jsonl(out / 'queries.jsonl') != queries:
            raise ValueError('Cannot change queries after inference has started')
    write_jsonl(out / 'queries.jsonl', queries)
    write_json(out / 'query_manifest_summary.json', summary)
    write_json(out / 'skipped_scenes.json', skipped)
    lock = dict(protocol_version=experiment_config()['protocol_version'], prompt_sha256=prompt_hash(),
                experiment_sha256=sha256_file(CONFIG_DIR / 'experiment.yaml'))
    if (out / 'run_lock.json').exists():
        existing = read_json(out / 'run_lock.json')
        if any(existing.get(k) != v for k, v in lock.items()):
            raise ValueError('Build-time run lock mismatch')
    else:
        write_json(out / 'run_lock.json', lock)
    return queries
