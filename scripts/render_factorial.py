#!/usr/bin/env python
"""Render an annotated factorial manifest using CPU image operations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from evidence_ownership.rendering.factorial import render_scene, validate_item
from evidence_ownership.rendering.qc import contact_sheet, write_gallery
from evidence_ownership.utils.config import get_paths
from evidence_ownership.utils.hashing import sha256_json
from evidence_ownership.utils.io import read_json, read_jsonl, write_json
from evidence_ownership.utils.logs import get_logger, progress


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--items', type=Path)
    parser.add_argument('--split', choices=['pilot', 'heldout'], default='pilot')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--scene-id', action='append', default=[])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--qc-sheet', action='store_true')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error('--limit must be nonnegative')
    paths = get_paths()
    output = args.output_dir or paths.rendered / args.split
    items = read_jsonl(args.items or paths.manifests / f'items_{args.split}.jsonl')
    if args.scene_id:
        items = [i for i in items if i['scene_id'] in args.scene_id]
    if args.limit is not None:
        items = items[:args.limit]
    log = get_logger('eob.render')
    records, failures, skipped = [], {}, 0
    seen = set()
    for item in progress(items, desc='Rendering', unit='scene', disable=args.no_progress):
        scene = item['scene_id']
        try:
            if scene in seen:
                raise ValueError('Duplicate scene_id')
            seen.add(scene)
            if item['study_split'] != args.split:
                raise ValueError('Item study_split differs from --split')
            source = paths.images_dir(item['coco_split']) / item['file_name']
            validate_item(item, source)
            target = output / scene
            if args.dry_run:
                preview = render_scene(item, source, target, dry_run=True)
                if preview['status'] == 'FAIL':
                    failures[scene] = [c for c in preview['checks'] if c['status'] == 'FAIL']
                log.info(json.dumps({'scene_id': scene, 'action': 'would_render', 'status': preview['status'], 'source': str(source), 'output': str(target)}))
                continue
            record_path = target / 'render_record.json'
            if args.resume and record_path.exists() and (old := read_json(record_path)).get('input_sha256') == sha256_json(item):
                record = old
                skipped += 1
            else:
                record = render_scene(item, source, target)
            records.append(record)
            if record['status'] == 'FAIL':
                failures[scene] = [c for c in record['checks'] if c['status'] == 'FAIL']
            if args.qc_sheet:
                contact_sheet(target, output / '_qc' / f'{scene}.png')
            log.info(json.dumps({'scene_id': scene, 'status': record['status']}))
        except (ValueError, KeyError, OSError) as exc:
            failures[scene] = [str(exc)]
            log.error(json.dumps({'scene_id': scene, 'status': 'FAIL', 'error': str(exc)}))
    if not args.dry_run:
        write_json(output / 'render_summary.json', {'PASS': sum(r['status'] == 'PASS' for r in records), 'FAIL': len(failures), 'skipped': skipped, 'failures': failures})
        if args.qc_sheet:
            write_gallery(output / '_qc', records)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
