"""Measured GPU smoke report. This module never substitutes assumed hardware values."""
from __future__ import annotations

import os
import re
from pathlib import Path

from evidence_ownership.inference.backend import FakeBackend, QwenBackend
from evidence_ownership.inference.parser import parse_response
from evidence_ownership.inference.runner import run_inference
from evidence_ownership.utils.config import model_config
from evidence_ownership.utils.io import read_jsonl, write_json, write_jsonl


def smoke_test(queries_path: Path, run_dir: Path, *, scenes: int = 5, device: str = 'cuda:0',
               backend=None, no_progress: bool = False, work_root: Path | None = None) -> dict:
    if scenes < 1:
        raise ValueError('--scenes must be positive')
    run_dir = Path(run_dir)
    queries = read_jsonl(queries_path)
    ids = list(dict.fromkeys(q['scene_id'] for q in queries))[:scenes]
    selected = [q for q in queries if q['scene_id'] in ids]
    if not selected:
        raise ValueError('No smoke queries')
    backend = backend if backend is not None else QwenBackend(device)
    # Isolate smoke query/lock/raw artifacts from the main run.
    smoke_dir = run_dir / 'smoke_test_run'
    smoke_queries = smoke_dir / 'queries.jsonl'
    write_jsonl(smoke_queries, selected)
    captured = []
    generate = backend.generate

    def capture(*args, **kwargs):
        result = generate(*args, **kwargs)
        if not captured:
            captured.append(backend.last_rendered_prompt)
        return result

    backend.generate = capture
    try:
        summary = run_inference(smoke_queries, smoke_dir, backend=backend, work_root=work_root,
                                no_progress=no_progress, raw_name='smoke_test_raw_predictions')
    finally:
        backend.generate = generate
    raw = read_jsonl(Path(summary['raw_path'])) if Path(summary['raw_path']).exists() else []
    from evidence_ownership.inference.parser import latest_ok
    ok = latest_ok(raw)
    measurements, grids, tokens = [], {}, {}
    for q in selected:
        if q['query_id'] not in ok:
            continue
        r = ok[q['query_id']]
        grids.setdefault(q['scene_id'], set()).add(str(r['image_grid_thw']))
        tokens.setdefault(q['scene_id'], set()).add(r['n_input_tokens'])
        measurements.append(dict(query_id=q['query_id'], scene_id=q['scene_id'], condition=q['condition'],
                                 image_grid_thw=r['image_grid_thw'], n_input_tokens=r['n_input_tokens'],
                                 n_generated_tokens=r['n_generated_tokens'], latency_s=r['latency_s'], raw_text=r['raw_text'],
                                 generated_token_ids=r['generated_token_ids'], hit_max_new_tokens=r['hit_max_new_tokens'],
                                 **parse_response(r['raw_text'], r['hit_max_new_tokens'])))
    # Capture the first rendered text even on a wholly resumed smoke run.
    if not captured and ok:
        q = next(q for q in selected if q['query_id'] in ok)
        if isinstance(backend, FakeBackend):
            rendered = q['system_prompt'] + '\n<image>\n' + q['user_prompt'] + '\n<think>\n\n</think>\n\n'
        else:
            messages = [dict(role='system', content=q['system_prompt']), dict(role='user', content=[
                dict(type='image', image=q['image_path']), dict(type='text', text=q['user_prompt'])])]
            rendered = backend.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        captured.append(rendered)
    first_template = re.sub(r'(?:<\|image_pad\|>)+', '<IMAGE_TOKENS>', captured[0]) if captured else None
    metadata = backend.metadata
    flags = dict(all_queries_ok=len(ok) == len(selected),
                 image_grid_identical=bool(grids) and all(len(v) == 1 for v in grids.values()),
                 input_tokens_identical=bool(tokens) and all(len(v) == 1 for v in tokens.values()),
                 strict_parse_valid=bool(measurements) and all(r['valid'] for r in measurements),
                 no_truncation=bool(measurements) and all(not r['hit_max_new_tokens'] for r in measurements),
                 thinking_disabled=bool(ok) and all(r.get('thinking_disabled_verified') for r in ok.values()),
                 parameters_match=metadata['parameter_count_matches'] is True,
                 intended_gpu_loaded=metadata['cuda_device_name'] is not None)
    report = dict(evidence_level='dry_run' if isinstance(backend, FakeBackend) else selected[0]['study_split'],
                  **metadata, stored_tensor_elements=model_config()['stored_tensor_elements'],
                  gpu_name=metadata['cuda_device_name'], CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  max_query_peak_allocated=max((r['cuda_max_memory_allocated'] for r in ok.values()), default=None),
                  max_query_peak_reserved=max((r['cuda_max_memory_reserved'] for r in ok.values()), default=None),
                  per_query=measurements, rendered_chat_template=first_template,
                  checks={k: 'PASS' if v else 'FAIL' for k, v in flags.items()},
                  status='PASS' if all(flags.values()) else 'FAIL', n_requested_queries=len(selected))
    write_json(run_dir / 'smoke_test.json', report)
    return report
