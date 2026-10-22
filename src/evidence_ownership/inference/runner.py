"""Append-only attempts, strict resume locks, and deterministic optional sharding."""
from __future__ import annotations

import fcntl
import importlib.metadata
import platform
import signal
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from evidence_ownership.inference.backend import FakeBackend
from evidence_ownership.inference.prompts import prompt_hash
from evidence_ownership.utils.config import CONFIG_DIR, experiment_config, get_paths, inference_config
from evidence_ownership.utils.hashing import sha256_file, sha256_json
from evidence_ownership.utils.io import append_jsonl, read_json, read_jsonl, write_json
from evidence_ownership.utils.logs import get_logger, progress


def environment_versions(backend) -> dict:
    versions = {'python': platform.python_version(), 'cuda_device_name': backend.metadata['cuda_device_name']}
    for package in ('torch', 'transformers', 'pillow'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def expected_lock(backend, queries: list[dict]) -> dict:
    meta = backend.metadata
    return dict(protocol_version=experiment_config()['protocol_version'],
                experiment_sha256=sha256_file(CONFIG_DIR / 'experiment.yaml'),
                prompt_sha256=prompt_hash(), query_prompt_hashes=sorted({q['prompt_sha256'] for q in queries}),
                queries_sha256=sha256_json(queries), model=meta['model'],
                generation_config=meta['generation_config'],
                generation_config_sha256=meta['generation_config_sha256'],
                chat_template_sha256=meta['chat_template_sha256'],
                environment=environment_versions(backend))


def establish_lock(path: Path, expected: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize optional workers initializing the same lock. Never keep this held during generation.
    with open(path.with_suffix('.guard'), 'a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        old = read_json(path) if path.exists() else {}
        mismatch = [k for k, v in expected.items() if k in old and old[k] != v]
        if mismatch:
            raise ValueError(f'Run lock mismatch: {", ".join(mismatch)}')
        write_json(path, {**old, **expected})


def validate_ok_duplicates(records: list[dict]) -> None:
    seen = {}
    for rec in records:
        if rec['status'] != 'ok':
            continue
        qid = rec['query_id']
        if qid in seen and seen[qid] != rec['raw_text']:
            raise ValueError(f'Conflicting duplicate ok query_id: {qid}')
        seen[qid] = rec['raw_text']


def merge_predictions(paths: list[Path], output: Path) -> int:
    records = [rec for path in paths for rec in read_jsonl(path)]
    validate_ok_duplicates(records)
    for field in ('model', 'generation_config_sha256', 'chat_template_sha256', 'prompt_sha256'):
        if len({sha256_json(r.get(field)) for r in records if r['status'] == 'ok'}) > 1:
            raise ValueError(f'Inconsistent shard provenance: {field}')
    if output.exists():
        raise FileExistsError(output)
    for record in sorted(records, key=lambda r: (r['query_id'], r['attempt'], r['timestamp_utc'])):
        append_jsonl(output, record)
    return len(records)


def run_inference(queries_path: Path, run_dir: Path, *, backend=None, device: str = 'cuda:0',
                  resume: bool = True, dry_run: bool = False, limit: int | None = None,
                  shard: str | None = None, no_progress: bool = False, work_root: Path | None = None,
                  raw_name: str | None = None) -> dict:
    queries_path, run_dir = Path(queries_path), Path(run_dir)
    queries = read_jsonl(queries_path)
    if len({q['query_id'] for q in queries}) != len(queries):
        raise ValueError('Query manifest contains duplicate IDs')
    worker, selected = 'w0', queries
    if shard:
        i, n = map(int, shard.split('/'))
        if not 0 <= i < n:
            raise ValueError('Shard must be zero-based i/n with 0 <= i < n')
        worker = f'w{i}-of-{n}'
        selected = [q for index, q in enumerate(queries) if index % n == i]
    if limit is not None:
        selected = selected[:limit]
    name = raw_name or ('dry_run_predictions' if dry_run else 'raw_predictions')
    if shard:
        name += f'.{worker}'
    raw_path = run_dir / (name if name.endswith('.jsonl') else name + '.jsonl')
    if raw_path.exists() and not resume:
        raise FileExistsError(f'--fresh refuses to overwrite {raw_path}')
    if backend is None:
        from evidence_ownership.inference.backend import QwenBackend
        backend = FakeBackend() if dry_run else QwenBackend(device)
    is_fake = isinstance(backend, FakeBackend)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ('dry_run_lock.json' if dry_run else 'run_lock.json')
    expected = expected_lock(backend, queries)
    if raw_path.exists() and not lock_path.exists():
        raise ValueError('Existing predictions have no run lock')
    establish_lock(lock_path, expected)
    records = read_jsonl(raw_path) if raw_path.exists() else []
    validate_ok_duplicates(records)
    for rec in records:
        for field in ('model', 'generation_config_sha256', 'chat_template_sha256', 'prompt_sha256'):
            if rec.get(field) != expected[field]:
                raise ValueError(f'Existing attempt differs from lock: {field}')
    done = {r['query_id'] for r in records if r['status'] == 'ok'}
    attempts = Counter(r['query_id'] for r in records)
    max_attempts = 1 + inference_config()['max_technical_retries']
    root = Path(work_root) if work_root is not None else get_paths().work_root
    counts = dict(ok=0, error=0, skipped=sum(q['query_id'] in done for q in selected), interrupted=False)
    bar = progress(selected, desc='Inference', unit='query', disable=no_progress)
    # Defer SIGINT until a complete attempt is fsynced. A signal must never interrupt
    # an append halfway through its JSON line. Library callers in other threads
    # retain their process's signal policy.
    previous_handler = None
    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.getsignal(signal.SIGINT)

        def request_stop(signum, frame):
            counts['interrupted'] = True

        signal.signal(signal.SIGINT, request_stop)
    try:
        for query in bar:
            if counts['interrupted']:
                break
            qid = query['query_id']
            if qid in done:
                continue
            while attempts[qid] < max_attempts:
                attempts[qid] += 1
                record = dict(query_id=qid, attempt=attempts[qid], run_id=run_dir.name, worker=worker,
                              timestamp_utc=datetime.now(timezone.utc).isoformat(), status='error', error=None,
                              raw_text=None, generated_token_ids=[], n_input_tokens=None, n_generated_tokens=0,
                              hit_max_new_tokens=False, image_grid_thw=None, latency_s=None,
                              cuda_max_memory_allocated=0, cuda_max_memory_reserved=0,
                              model=backend.model_info,
                              generation_config_sha256=expected['generation_config_sha256'],
                              chat_template_sha256=expected['chat_template_sha256'],
                              prompt_sha256=expected['prompt_sha256'], query_prompt_sha256=query['prompt_sha256'],
                              rendered_prompt_sha256=None, image_sha256_checked=None,
                              evidence_level='dry_run' if is_fake else query.get('evidence_level', query['study_split']))
                interrupted = False
                try:
                    record.update(backend.generate(root / query['image_path'], query['system_prompt'],
                                                   query['user_prompt'], image_sha256=query['image_sha256']))
                    record['status'] = 'ok'
                except KeyboardInterrupt:
                    record['error'] = {'type': 'KeyboardInterrupt', 'message': 'Interrupted during query'}
                    interrupted = True
                except Exception as exc:
                    record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
                    get_logger(__name__).error('%s attempt %s: %s', qid, attempts[qid], exc)
                append_jsonl(raw_path, record)
                counts[record['status']] += 1
                if record['status'] == 'ok':
                    done.add(qid)
                if hasattr(bar, 'set_postfix'):
                    bar.set_postfix(ok=counts['ok'], error=counts['error'])
                if interrupted or counts['interrupted']:
                    raise KeyboardInterrupt
                if record['status'] == 'ok':
                    break
    except KeyboardInterrupt:
        counts['interrupted'] = True
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGINT, previous_handler)
        if hasattr(bar, 'close'):
            bar.close()
    counts.update(n_selected=len(selected), n_complete=sum(q['query_id'] in done for q in selected),
                  failed_query_ids=[q['query_id'] for q in selected if q['query_id'] not in done], raw_path=str(raw_path))
    write_json(run_dir / (raw_path.stem + '_summary.json'), counts)
    get_logger(__name__).info('Final summary: %s', counts)
    return counts
