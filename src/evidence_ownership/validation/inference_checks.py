"""Read-only PASS/FAIL/SKIPPED audit of query and generation provenance."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from evidence_ownership.inference.parser import latest_ok
from evidence_ownership.inference.runner import validate_ok_duplicates
from evidence_ownership.utils.hashing import sha256_file, sha256_json


def inference_checks(queries: list[dict], records: list[dict], *, run_lock: dict | None = None,
                     work_root: Path | None = None, missing_report: dict | None = None) -> list[dict]:
    checks = []

    def check(name, condition, detail, skipped=False):
        checks.append(dict(name=name, status='SKIPPED' if skipped else 'PASS' if condition else 'FAIL', detail=detail))

    ok = latest_ok(records)
    missing = {q['query_id'] for q in queries} - ok.keys()
    reported = set((missing_report or {}).get('missing_query_ids', []))
    check('query_accounting', not (missing - reported), dict(n_missing=len(missing), unreported=sorted(missing-reported)))
    check('complete_inference', not missing, sorted(missing))
    try:
        validate_ok_duplicates(records)
        conflict = None
    except ValueError as exc:
        conflict = str(exc)
    check('no_conflicting_ok_records', conflict is None, conflict)
    for field in ('model', 'generation_config_sha256', 'prompt_sha256', 'chat_template_sha256'):
        values = {sha256_json(r.get(field)) for r in records if r['status'] == 'ok'}
        check('uniform_' + field, len(values) <= 1, dict(n_values=len(values)), skipped=not values)
    lock_errors = []
    if run_lock is not None:
        for r in records:
            for field in ('model', 'generation_config_sha256', 'chat_template_sha256', 'prompt_sha256'):
                if r.get(field) != run_lock.get(field):
                    lock_errors.append((r['query_id'], field))
        if 'queries_sha256' in run_lock and run_lock['queries_sha256'] != sha256_json(queries):
            lock_errors.append(('manifest', 'queries_sha256'))
    check('run_lock_consistent', not lock_errors, lock_errors, skipped=run_lock is None)
    hash_errors = []
    for q in queries:
        rec = ok.get(q['query_id'])
        if rec and rec.get('image_sha256_checked') != q['image_sha256']:
            hash_errors.append(q['query_id'] + ':generation_hash')
        if work_root is not None:
            path = Path(work_root) / q['image_path']
            if not path.exists() or sha256_file(path) != q['image_sha256']:
                hash_errors.append(q['query_id'] + ':file_hash')
    check('image_hashes', not hash_errors, hash_errors, skipped=not ok and work_root is None)
    check('thinking_disabled', all(r.get('thinking_disabled_verified') is True for r in ok.values()),
          dict(n_verified=sum(r.get('thinking_disabled_verified') is True for r in ok.values()), n_ok=len(ok)), skipped=not ok)
    by_scene = defaultdict(list)
    for q in queries:
        if q['query_id'] in ok:
            by_scene[q['scene_id']].append(ok[q['query_id']])
    for field in ('n_input_tokens', 'image_grid_thw'):
        bad = [sid for sid, group in by_scene.items() if len({sha256_json(r.get(field)) for r in group}) != 1
               or any(r.get(field) is None for r in group)]
        check('identical_' + field + '_within_scene', not bad, bad, skipped=not by_scene)
    return checks
