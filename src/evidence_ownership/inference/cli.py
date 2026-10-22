"""Command-line entry points shared by the thin repository scripts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(stage: str) -> None:
    parser = argparse.ArgumentParser(description={
        'build_queries': 'Build query manifests from PASS scenes.',
        'run_inference': 'Resume single-device generation. Select the physical GPU with CUDA_VISIBLE_DEVICES; --device indexes the visible set. Shards are zero-based i/n.',
        'parse_predictions': 'Strictly parse latest successful attempts and report every missing query.',
        'analyze': 'Compute paired scene contrasts and cluster-bootstrap statistics.',
        'make_figures': 'Save three 300-dpi PNG figures (no PDF).',
        'prediction_gallery': 'Build a static HTML prediction gallery.',
        'smoke_test': 'Measure the local Qwen checkpoint on a single explicitly selected GPU.'}[stage])
    if stage == 'build_queries':
        parser.add_argument('--split', choices=['pilot', 'heldout'], required=True)
        parser.add_argument('--run-name', required=True)
        parser.add_argument('--output-dir', type=Path)
        parser.add_argument('--limit-scenes', type=int)
        parser.add_argument('--dry-run', action='store_true', help='Build manifests marked dry_run; no inference')
    if stage in ('run_inference', 'parse_predictions', 'smoke_test'):
        parser.add_argument('--queries', type=Path, required=True)
    if stage in ('run_inference', 'smoke_test'):
        parser.add_argument('--run-dir', type=Path, required=True)
        parser.add_argument('--device', default='cuda:0')
        parser.add_argument('--backend', choices=['fake', 'qwen'], default='qwen')
    if stage == 'run_inference':
        parser.add_argument('--limit', type=int)
        modes = parser.add_mutually_exclusive_group()
        modes.add_argument('--resume', action='store_true', default=True)
        modes.add_argument('--fresh', action='store_true')
        parser.add_argument('--dry-run', action='store_true', help='Force FakeBackend and separate dry_run_predictions.jsonl')
        parser.add_argument('--shard', help='Zero-based i/n; each worker has its own append-only raw file')
    if stage == 'smoke_test':
        parser.add_argument('--scenes', type=int, default=5)
    if stage == 'parse_predictions':
        parser.add_argument('--raw', '--raw-predictions', type=Path, required=True)
        parser.add_argument('--output', type=Path)
    if stage in ('analyze', 'prediction_gallery'):
        parser.add_argument('--parsed', '--parsed-predictions', type=Path, required=True)
    if stage == 'analyze':
        parser.add_argument('--items', type=Path)
        parser.add_argument('--reference-labels', type=Path, help='Pilot/reference items JSONL for constant-color baseline')
        parser.add_argument('--evidence-level', choices=['synthetic_fixture', 'dry_run', 'pilot', 'heldout'])
        parser.add_argument('--output-dir', type=Path)
    if stage == 'make_figures':
        parser.add_argument('--results-dir', type=Path, required=True)
        parser.add_argument('--output-dir', type=Path)
    if stage == 'prediction_gallery':
        parser.add_argument('--output', type=Path)
    parser.add_argument('--no-progress', action='store_true')
    args = vars(parser.parse_args())
    if stage == 'build_queries':
        from evidence_ownership.inference.queries import build_queries
        rows = build_queries(**args)
        result = {'n_queries': len(rows)}
    elif stage == 'run_inference':
        from evidence_ownership.inference.backend import FakeBackend, QwenBackend
        from evidence_ownership.inference.runner import run_inference
        backend_name = args.pop('backend')
        args['resume'] = not args.pop('fresh')
        args['queries_path'] = args.pop('queries')
        # Check --fresh before constructing a potentially expensive Qwen runtime.
        prefix = 'dry_run_predictions' if args['dry_run'] else 'raw_predictions'
        if not args['resume'] and list(args['run_dir'].glob(prefix + '*.jsonl')):
            parser.error('--fresh refuses an existing raw predictions file')
        args['backend'] = FakeBackend() if args['dry_run'] or backend_name == 'fake' else QwenBackend(args['device'])
        result = run_inference(**args)
    elif stage == 'parse_predictions':
        from evidence_ownership.inference.parser import parse_predictions
        result = parse_predictions(args['queries'], args['raw'], args['output'])
    elif stage == 'analyze':
        from evidence_ownership.evaluation.analysis import analyze
        result = analyze(args['parsed'], output_dir=args['output_dir'], items_path=args['items'],
                         reference_labels_path=args['reference_labels'], evidence_level=args['evidence_level'],
                         no_progress=args['no_progress'])
    elif stage == 'make_figures':
        from evidence_ownership.evaluation.figures import make_figures
        result = [str(p) for p in make_figures(args['results_dir'], args['output_dir'])]
    elif stage == 'prediction_gallery':
        from evidence_ownership.evaluation.gallery import prediction_gallery
        result = str(prediction_gallery(args['parsed'], args['output']))
    else:
        from evidence_ownership.inference.backend import FakeBackend
        from evidence_ownership.inference.smoke import smoke_test
        result = smoke_test(args['queries'], args['run_dir'], scenes=args['scenes'], device=args['device'],
                            backend=FakeBackend() if args['backend'] == 'fake' else None, no_progress=args['no_progress'])
        result = {'status': result['status'], 'report': str(args['run_dir'] / 'smoke_test.json')}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
