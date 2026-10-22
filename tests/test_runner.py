import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from evidence_ownership.inference.backend import FakeBackend
from evidence_ownership.inference.queries import build_queries
from evidence_ownership.inference.runner import merge_predictions, run_inference
from evidence_ownership.utils.config import REPO_ROOT, inference_config
from evidence_ownership.utils.io import read_json, read_jsonl, write_json
from test_queries import fixture_work


def test_resume_skips_ok_and_preserves_bytes(fixture_work):
    rows = build_queries('pilot', 'runner', no_progress=True)
    run = fixture_work/'runs/runner'
    backend = FakeBackend()
    summary = run_inference(run/'queries.jsonl', run, backend=backend, limit=2, no_progress=True)
    assert backend.calls == 2 and summary['ok'] == 2
    before = (run/'raw_predictions.jsonl').read_bytes()
    summary = run_inference(run/'queries.jsonl', run, backend=backend, no_progress=True)
    assert backend.calls == len(rows) and summary['skipped'] == 2
    assert (run/'raw_predictions.jsonl').read_bytes().startswith(before)
    whole = (run/'raw_predictions.jsonl').read_bytes()
    run_inference(run/'queries.jsonl', run, backend=backend, no_progress=True)
    assert (run/'raw_predictions.jsonl').read_bytes() == whole
    with pytest.raises(FileExistsError):
        run_inference(run/'queries.jsonl', run, backend=backend, resume=False)


def test_technical_retries_and_no_content_retry(fixture_work):
    build_queries('pilot', 'retry', no_progress=True)
    run = fixture_work/'runs/retry'
    backend = FakeBackend({'I11.png': [RuntimeError('fake OOM'), 'blue explanation with red']})
    result = run_inference(run/'queries.jsonl', run, backend=backend, limit=1, no_progress=True)
    records = read_jsonl(run/'raw_predictions.jsonl')
    assert [r['status'] for r in records] == ['error', 'ok']
    assert [r['attempt'] for r in records] == [1, 2]
    assert result['ok'] == 1 and backend.calls == 2


def test_retry_limit_persists_across_resume(fixture_work):
    build_queries('pilot', 'failed', no_progress=True)
    run = fixture_work/'runs/failed'
    backend = FakeBackend({'I11.png': RuntimeError('persistent technical failure')})
    result = run_inference(run/'queries.jsonl', run, backend=backend, limit=1, no_progress=True)
    assert backend.calls == 3 and len(result['failed_query_ids']) == 1
    run_inference(run/'queries.jsonl', run, backend=backend, limit=1, no_progress=True)
    assert backend.calls == 3


def test_resume_refuses_lock_mismatch(fixture_work):
    build_queries('pilot', 'lock', no_progress=True)
    run = fixture_work/'runs/lock'
    run_inference(run/'queries.jsonl', run, backend=FakeBackend(), limit=1, no_progress=True)
    lock = read_json(run/'run_lock.json')
    lock['generation_config_sha256'] = 'changed'
    write_json(run/'run_lock.json', lock)
    with pytest.raises(ValueError, match='lock mismatch'):
        run_inference(run/'queries.jsonl', run, backend=FakeBackend(), no_progress=True)


def test_dry_run_is_separate_and_image_integrity(fixture_work):
    rows = build_queries('pilot', 'dryrun', no_progress=True)
    run = fixture_work/'runs/dryrun'
    run_inference(run/'queries.jsonl', run, dry_run=True, limit=1, no_progress=True)
    assert not (run/'raw_predictions.jsonl').exists()
    assert (run/'dry_run_predictions.jsonl').exists()
    with pytest.raises(ValueError, match='SHA256'):
        FakeBackend().generate(fixture_work/rows[0]['image_path'], '', '', image_sha256='wrong')


def test_shards_and_merge_conflicts(fixture_work):
    rows = build_queries('pilot', 'shards', no_progress=True)
    run = fixture_work/'runs/shards'
    for shard in ('0/2', '1/2'):
        run_inference(run/'queries.jsonl', run, backend=FakeBackend(), shard=shard, no_progress=True)
    paths = sorted(run.glob('raw_predictions.w*.jsonl'))
    assert merge_predictions(paths, run/'merged.jsonl') == len(rows)
    records = read_jsonl(run/'merged.jsonl')
    assert len({r['query_id'] for r in records}) == len(rows)
    bad = deepcopy(records[0])
    bad['raw_text'] = '{"label":"pink"}'
    from evidence_ownership.utils.io import write_jsonl
    write_jsonl(run/'bad.jsonl', [bad])
    with pytest.raises(ValueError, match='Conflicting'):
        merge_predictions([run/'merged.jsonl', run/'bad.jsonl'], run/'bad_merge.jsonl')


def test_interrupt_keeps_valid_jsonl(fixture_work):
    build_queries('pilot', 'interrupt', no_progress=True)
    run = fixture_work/'runs/interrupt'
    def interrupt(*args):
        raise KeyboardInterrupt
    result = run_inference(run/'queries.jsonl', run, backend=FakeBackend(interrupt), no_progress=True)
    assert result['interrupted']
    assert read_jsonl(run/'raw_predictions.jsonl')[0]['error']['type'] == 'KeyboardInterrupt'


def test_fake_end_to_end_cli_and_artifacts(fixture_work):
    run = fixture_work/'runs/e2e'
    result = fixture_work/'results/pilot'
    commands = [
        ['build_queries', '--split', 'pilot', '--run-name', 'e2e'],
        ['run_inference', '--queries', str(run/'queries.jsonl'), '--run-dir', str(run), '--backend', 'fake'],
        ['parse_predictions', '--queries', str(run/'queries.jsonl'), '--raw', str(run/'raw_predictions.jsonl')],
        ['analyze', '--parsed', str(run/'parsed_predictions.jsonl'), '--output-dir', str(result)],
        ['make_figures', '--results-dir', str(result)],
        ['prediction_gallery', '--parsed', str(run/'parsed_predictions.jsonl')],
        ['smoke_test', '--queries', str(run/'queries.jsonl'), '--run-dir', str(run), '--backend', 'fake', '--scenes', '1'],
    ]
    for stage, *args in commands:
        proc = subprocess.run([sys.executable, str(REPO_ROOT/'scripts'/f'{stage}.py'), *args, '--no-progress'],
                              capture_output=True, text=True, env=os.environ.copy())
        assert proc.returncode == 0, f'{stage}: {proc.stdout}\n{proc.stderr}'
    manifest = read_json(result/'analysis_manifest.json')
    assert manifest['evidence_level'] == 'dry_run'
    assert manifest['bootstrap_unit'] == 'cluster'
    assert manifest['n_clusters'] == 2
    assert len(list((result/'figures').glob('*.png'))) == 3
    assert not list((result/'figures').glob('*.pdf'))
    assert (fixture_work/'galleries/predictions_e2e.html').exists()
    assert read_json(run/'missing_queries.json')['n_missing'] == 0
    assert read_json(result/'primary_contrasts.json')['primary_support']['status'] == 'NOT_SUPPORTED'
    assert read_json(run/'smoke_test.json')['evidence_level'] == 'dry_run'


@pytest.mark.parametrize('name', ['build_queries','run_inference','parse_predictions','analyze','make_figures','prediction_gallery','smoke_test'])
def test_script_help(name):
    proc = subprocess.run([sys.executable, str(REPO_ROOT/'scripts'/f'{name}.py'), '--help'], capture_output=True, text=True)
    assert proc.returncode == 0 and 'usage:' in proc.stdout


def test_qwen_contract_with_mocked_local_loaders(fixture_work, monkeypatch):
    """Exercise actual QwenBackend orchestration using tiny CPU tensors, never real weights."""
    import torch
    import evidence_ownership.inference.backend as module

    class Config:
        @classmethod
        def from_dict(cls, values):
            obj = cls()
            obj.values = dict(values)
            return obj
        def update(self, **values):
            self.values.update(values)
        def to_dict(self):
            return dict(self.values)

    class Processor:
        chat_template = 'mock-chat-template'
        image_processor = SimpleNamespace()
        def __init__(self):
            self.calls = []
        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            assert kwargs['enable_thinking'] is False
            if not kwargs['tokenize']:
                return 'system/user<image>' + module.THINK_SUFFIX
            return dict(input_ids=torch.tensor([[1,2,3]]), image_grid_thw=torch.tensor([[1,64,64]]))
        def decode(self, tokens, **kwargs):
            assert tokens.tolist() == [7,8] and kwargs == {'skip_special_tokens': True}
            return '{"label":"blue"}'

    class Model:
        config = SimpleNamespace(_attn_implementation='sdpa')
        generation_config = Config.from_dict(dict(temperature=0.7, top_p=0.8, top_k=20))
        def __init__(self):
            self.p = torch.nn.Parameter(torch.zeros(2))
            self.kwargs = None
        def parameters(self):
            return iter([self.p, self.p])
        def eval(self):
            return self
        def generate(self, **kwargs):
            assert not torch.is_grad_enabled()
            self.kwargs = kwargs
            return torch.tensor([[1,2,3,7,8]])

    processor, model, loads = Processor(), Model(), []
    def load_processor(path, **kwargs):
        loads.append(('processor', path, kwargs))
        return processor
    def load_model(path, **kwargs):
        loads.append(('model', path, kwargs))
        assert kwargs['device_map'] == {'': 'cpu'}
        assert kwargs['dtype'] == torch.bfloat16
        return model
    transformers = SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
                                    AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=load_model), GenerationConfig=Config)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    cfg = module.model_config().copy()
    cfg['loaded_unique_parameters'] = 2
    monkeypatch.setattr(module, 'model_config', lambda: cfg)
    monkeypatch.setattr(module.importlib, 'import_module', lambda name: (_ for _ in ()).throw(ImportError(name)))
    backend = module.QwenBackend('cpu')
    rows = build_queries('pilot', 'mock-qwen', no_progress=True)
    q = rows[0]
    result = backend.generate(fixture_work/q['image_path'], q['system_prompt'], q['user_prompt'], q['image_sha256'])
    assert result['generated_token_ids'] == [7,8] and result['n_input_tokens'] == 3
    assert result['thinking_disabled_verified']
    assert backend.metadata['loaded_unique_parameters'] == 2
    assert all(entry[2]['local_files_only'] is True for entry in loads)
    assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'
    for key, value in inference_config()['generation'].items():
        assert key in model.kwargs and model.kwargs[key] == value
    assert [m['role'] for m in processor.calls[0][0]] == ['system', 'user']
    assert backend.effective_generation_config['temperature'] is None
    monkeypatch.setattr(processor, 'apply_chat_template', lambda *a, **kw: 'no thinking suffix')
    with pytest.raises(RuntimeError, match='thinking block'):
        backend.generate(fixture_work/q['image_path'], '', '', q['image_sha256'])


def test_validation_audits_and_stale_parse(fixture_work):
    from evidence_ownership.evaluation.analysis import analyze
    from evidence_ownership.inference.parser import parse_predictions
    from evidence_ownership.utils.io import append_jsonl
    from evidence_ownership.validation.inference_checks import inference_checks
    rows = build_queries('pilot', 'audit', no_progress=True)
    run = fixture_work/'runs/audit'
    run_inference(run/'queries.jsonl', run, backend=FakeBackend(), no_progress=True)
    raw = read_jsonl(run/'raw_predictions.jsonl')
    report = parse_predictions(run/'queries.jsonl', run/'raw_predictions.jsonl')
    checks = inference_checks(rows, raw, run_lock=read_json(run/'run_lock.json'),
                              work_root=fixture_work, missing_report=report)
    assert all(c['status'] == 'PASS' for c in checks)
    missing = inference_checks(rows, raw[:1], missing_report=dict(missing_query_ids=[r['query_id'] for r in rows[1:]]))
    assert next(c for c in missing if c['name'] == 'query_accounting')['status'] == 'PASS'
    assert next(c for c in missing if c['name'] == 'complete_inference')['status'] == 'FAIL'
    changed = deepcopy(raw)
    changed[0]['thinking_disabled_verified'] = False
    bad = inference_checks(rows, changed)
    assert next(c for c in bad if c['name'] == 'thinking_disabled')['status'] == 'FAIL'
    append_jsonl(run/'raw_predictions.jsonl', raw[0])
    with pytest.raises(ValueError, match='Stale parsed'):
        analyze(run/'parsed_predictions.jsonl', no_progress=True)


def test_sigint_finishes_current_attempt_and_restores_handler(fixture_work):
    import signal
    build_queries('pilot', 'signal', no_progress=True)
    run = fixture_work/'runs/signal'
    original = signal.getsignal(signal.SIGINT)
    def signal_during_generate(*args):
        os.kill(os.getpid(), signal.SIGINT)
        return '{"label":"UNKNOWN"}'
    result = run_inference(run/'queries.jsonl', run, backend=FakeBackend(signal_during_generate), no_progress=True)
    assert result['interrupted'] and result['ok'] == result['n_complete'] == 1
    assert len(read_jsonl(run/'raw_predictions.jsonl')) == 1
    assert signal.getsignal(signal.SIGINT) == original
