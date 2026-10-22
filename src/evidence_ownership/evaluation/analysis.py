"""Analysis artifacts with explicit provenance and evidence classification."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from evidence_ownership.evaluation.bootstrap import bootstrap_contrasts
from evidence_ownership.evaluation.diagnostics import analytic_baselines, diagnostics, invalid_bounds, pilot_gates, primary_support
from evidence_ownership.utils.config import REPO_ROOT, experiment_config, get_paths
from evidence_ownership.utils.hashing import sha256_file, sha256_json
from evidence_ownership.utils.io import read_json, read_jsonl, write_json
from evidence_ownership.validation.inference_checks import inference_checks
from evidence_ownership.validation.stats_checks import stats_checks

EVIDENCE_LEVELS = ('synthetic_fixture', 'dry_run', 'pilot', 'heldout')


def code_version() -> str:
    files = [*REPO_ROOT.glob('src/evidence_ownership/inference/*.py'),
             *REPO_ROOT.glob('src/evidence_ownership/evaluation/*.py'), *REPO_ROOT.glob('scripts/*.py')]
    return sha256_json({str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in sorted(files)})


def analyze(parsed_path: Path, *, output_dir: Path | None = None, items_path: Path | None = None,
            reference_labels_path: Path | None = None, evidence_level: str | None = None,
            no_progress: bool = False, work_root: Path | None = None) -> dict:
    parsed_path = Path(parsed_path)
    rows = read_jsonl(parsed_path)
    if not rows:
        raise ValueError('No parsed predictions')
    splits = {r['study_split'] for r in rows}
    if len(splits) != 1:
        raise ValueError('Pilot and held-out records cannot be pooled')
    split = next(iter(splits))
    root = Path(work_root) if work_root is not None else get_paths().work_root
    output = Path(output_dir) if output_dir is not None else root / 'results' / split
    missing_path = parsed_path.parent / 'missing_queries.json'
    if missing_path.exists() and read_json(missing_path)['n_missing']:
        raise ValueError('Missing queries are reported; complete inference before analysis')
    inferred = {r.get('evidence_level', split) for r in rows}
    # A fake backend may never be promoted to real evidence by a command-line override.
    fake = 'dry_run' in inferred or any(r.get('model', {}).get('class') == 'FakeBackend' for r in rows)
    level = ('dry_run' if fake else 'synthetic_fixture' if 'synthetic_fixture' in inferred
             else evidence_level or split)
    if level not in EVIDENCE_LEVELS:
        raise ValueError(f'Unknown evidence level: {level}')
    if level in ('pilot', 'heldout') and level != split:
        raise ValueError('Evidence level must match the study split')
    inputs = {str(parsed_path): sha256_file(parsed_path)}
    items_path = Path(items_path) if items_path else root / 'manifests' / f'items_{split}.jsonl'
    statuses = {r.get('annotation_status', 'unknown') for r in rows}
    if items_path.exists():
        items = {r['scene_id']: r for r in read_jsonl(items_path)}
        scene_ids = {r['scene_id'] for r in rows}
        if scene_ids - items.keys():
            raise ValueError('Analyzed scenes missing from supplied item manifest')
        statuses = {items[s].get('annotation_status', 'unknown') for s in scene_ids}
        inputs[str(items_path)] = sha256_file(items_path)
    annotation = next(iter(statuses)) if len(statuses) == 1 else 'mixed:' + ','.join(sorted(statuses))
    provenance = dict(evidence_level=level, annotation_status=annotation)
    config = experiment_config()['analysis']
    contrasts = bootstrap_contrasts(rows, no_progress=no_progress)
    diag = diagnostics(rows, n_resamples=config['bootstrap_resamples'], seed=config['seed'], no_progress=no_progress)
    bounds = invalid_bounds(rows)
    gates = pilot_gates(diag['controls'], diag['valid_output_rate']['rate'])
    audit = []
    if missing_path.exists():
        missing = read_json(missing_path)
        query_path = parsed_path.parent / missing.get('queries_path', 'queries.jsonl')
        raw_path = parsed_path.parent / missing.get('raw_predictions_path', 'raw_predictions.jsonl')
        if query_path.exists() and raw_path.exists():
            manifest_queries = read_jsonl(query_path)
            if {q['query_id'] for q in manifest_queries} != {r['query_id'] for r in rows}:
                raise ValueError('Parsed query IDs do not cover the linked manifest exactly')
            for path, key in ((query_path, 'queries_sha256'), (raw_path, 'raw_predictions_sha256')):
                digest = sha256_file(path)
                if key in missing and digest != missing[key]:
                    raise ValueError(f'Stale parsed predictions: {key} changed; parse again')
                inputs[str(path)] = digest
            lock_name = 'dry_run_lock.json' if raw_path.name.startswith('dry_run') else 'run_lock.json'
            lock_path = raw_path.parent / lock_name
            audit = inference_checks(manifest_queries, read_jsonl(raw_path),
                                     run_lock=read_json(lock_path) if lock_path.exists() else None,
                                     work_root=root, missing_report=missing)
    if not audit:
        audit = [dict(name='inference_provenance', status='SKIPPED', detail='No linked raw/query artifacts')]
    split_sets = {}
    for name in ('pilot', 'heldout'):
        manifest_path = root / 'manifests' / f'items_{name}.jsonl'
        if manifest_path.exists():
            manifest_rows = read_jsonl(manifest_path)
            # Include cluster identities to catch near-duplicates split across manifests.
            split_sets[name] = {str(r['scene_id']) for r in manifest_rows} | {
                str(r.get('cluster_id', r['scene_id'])) for r in manifest_rows}
            inputs[str(manifest_path)] = sha256_file(manifest_path)
    audit += stats_checks(rows, contrasts, pilot_scene_ids=split_sets.get('pilot'),
                          heldout_scene_ids=split_sets.get('heldout'))
    diag['validation_checks'] = audit
    contrasts['primary_support'] = primary_support(contrasts, gates, diag['valid_output_rate']['rate'], bounds, annotation, level)
    failed_audits = [c['name'] for c in audit if c['status'] != 'PASS']
    if failed_audits:
        contrasts['primary_support']['status'] = 'NOT_SUPPORTED'
        contrasts['primary_support']['reasons'].append('Validation failed or remains unverified: ' + ', '.join(failed_audits))
    reference_labels = None
    if reference_labels_path:
        reference_labels_path = Path(reference_labels_path)
        records = read_jsonl(reference_labels_path)
        if any(r.get('study_split') == 'heldout' for r in records):
            raise ValueError('Constant-color baseline must use pilot/reference labels, not held-out labels')
        reference_labels = [r['persons'][p]['latent_color'] for r in records for p in ('A', 'B')]
        inputs[str(reference_labels_path)] = sha256_file(reference_labels_path)
    baselines = analytic_baselines(rows, reference_labels)
    output.mkdir(parents=True, exist_ok=True)
    for filename, content in (('primary_contrasts.json', contrasts), ('invalid_output_bounds.json', bounds),
                              ('diagnostics.json', diag), ('baselines.json', baselines), ('pilot_gates.json', gates)):
        write_json(output / filename, {**provenance, **content})
    cell_records = []
    for cell in diag['cells']:
        flat = dict(condition=cell['condition'], target_visible=cell['target_visible'], n=cell['n'], **provenance)
        for key in ('accuracy', 'donor_match', 'UNKNOWN', 'invalid', 'any_definite_color'):
            flat.update({key + '_' + k: v for k, v in cell[key].items()})
        for key, value in cell['composition'].items():
            flat.update({'composition_' + key + '_' + k: v for k, v in value.items()})
        cell_records.append(flat)
    pd.DataFrame(cell_records).to_csv(output / 'cell_counts.csv', index=False)
    control_records = []
    for name in ('clean', 'I11', 'iso'):
        control_records.append(dict(control=name, **diag['controls'][name]['accuracy'], **provenance))
    paired = diag['controls']['paired_clean_minus_sham']
    control_records.append(dict(control='paired_clean_minus_sham', rate=paired['point'],
                                numerator=paired.get('numerator'), denominator=paired.get('denominator'),
                                ci_low=paired['ci'][0] if paired['ci'] else None,
                                ci_high=paired['ci'][1] if paired['ci'] else None, **provenance))
    pd.DataFrame(control_records).to_csv(output / 'controls.csv', index=False)
    # Record other relevant run inputs as well as the parsed artifact and annotations.
    for name in ('queries.jsonl', 'run_lock.json', 'dry_run_lock.json', 'missing_queries.json'):
        path = parsed_path.parent / name
        if path.exists():
            inputs[str(path)] = sha256_file(path)
    write_json(output / 'analysis_manifest.json', dict(**provenance, input_sha256s=inputs,
               code_version=code_version(), code_version_kind='sha256_of_source_files_no_vcs',
               seed=config['seed'], bootstrap_resamples=config['bootstrap_resamples'], bootstrap_unit='cluster',
               n_clusters=contrasts['psi']['n_clusters'], n_scenes=contrasts['psi']['n_scenes']))
    return dict(output_dir=str(output), **provenance, primary_support=contrasts['primary_support'])
