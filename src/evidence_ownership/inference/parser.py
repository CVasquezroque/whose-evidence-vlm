"""Strict schema parsing; invalid responses are outcomes, never repaired."""
from __future__ import annotations

import json
import os
from pathlib import Path

from evidence_ownership.utils.config import experiment_config, inference_config
from evidence_ownership.utils.hashing import sha256_file
from evidence_ownership.utils.io import read_jsonl, write_json, write_jsonl


def parse_response(raw: str, hit_max_new_tokens: bool = False, *, allow_json_fence: bool | None = None) -> dict:
    text = raw.strip()
    allow = inference_config()['parser']['allow_json_fence'] if allow_json_fence is None else allow_json_fence
    if allow and text.startswith('```json\n') and text.endswith('\n```'):
        text = text[8:-4].strip()
    reason, label = None, None
    if not text:
        reason = 'empty'
    else:
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            reason = 'not_json'
        else:
            if not isinstance(obj, dict):
                reason = 'not_object'
            elif set(obj) != {'label'}:
                reason = 'wrong_keys'
            elif not isinstance(obj['label'], str):
                reason = 'label_not_string'
            elif obj['label'] not in experiment_config()['labels']:
                reason = 'label_not_in_vocabulary'
            else:
                label = obj['label']
    if reason is not None and hit_max_new_tokens:
        reason = 'truncated'
    return dict(label=label, valid=reason is None, invalid_reason=reason)


def latest_ok(records: list[dict]) -> dict[str, dict]:
    """File order resolves ties; attempt numbers select the latest successful attempt."""
    result: dict[str, dict] = {}
    for rec in records:
        if rec['status'] == 'ok' and rec.get('attempt', 0) >= result.get(rec['query_id'], {}).get('attempt', -1):
            result[rec['query_id']] = rec
    return result


def parse_predictions(queries_path: Path, raw_path: Path, output_path: Path | None = None) -> dict:
    queries = read_jsonl(queries_path)
    raw = read_jsonl(raw_path)
    ok = latest_ok(raw)
    output_path = output_path or raw_path.parent / 'parsed_predictions.jsonl'
    parsed, missing = [], []
    for query in queries:
        rec = ok.get(query['query_id'])
        if rec is None:
            missing.append(query['query_id'])
            continue
        row = {**rec, **query, **parse_response(rec['raw_text'], rec['hit_max_new_tokens'])}
        row['R'] = int(row['valid'] and row['label'] == row['gold']['donor_latent_color'])
        if raw_path.name.startswith(('dry_run', 'smoke_test')) or rec.get('model', {}).get('class') == 'FakeBackend':
            row['evidence_level'] = 'dry_run'
        parsed.append(row)
    write_jsonl(output_path, parsed)
    report = dict(n_queries=len(queries), n_parsed=len(parsed), n_missing=len(missing), missing_query_ids=missing,
                  orphan_ok_query_ids=sorted(set(ok) - {q['query_id'] for q in queries}),
                  queries_path=os.path.relpath(queries_path, output_path.parent),
                  raw_predictions_path=os.path.relpath(raw_path, output_path.parent),
                  queries_sha256=sha256_file(queries_path), raw_predictions_sha256=sha256_file(raw_path))
    write_json(output_path.parent / 'missing_queries.json', report)
    return report
