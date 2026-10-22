"""Portable HTML inspection of predictions with escaped raw output."""
from __future__ import annotations

import html
import os
from collections import defaultdict
from pathlib import Path

from evidence_ownership.evaluation.diagnostics import outcome
from evidence_ownership.utils.config import get_paths
from evidence_ownership.utils.io import read_jsonl


def prediction_gallery(parsed_path: Path, output_path: Path | None = None, *, work_root: Path | None = None) -> Path:
    root = Path(work_root) if work_root else get_paths().work_root
    parsed_path = Path(parsed_path)
    output = Path(output_path) if output_path else root / 'galleries' / f'predictions_{parsed_path.parent.name}.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(parsed_path)
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row['scene_id']][row['image_key']].append(row)
    esc = lambda value: html.escape(str(value), quote=True)
    parts = ['<!doctype html><html><head><meta charset="utf-8"><title>Prediction gallery</title>',
             '<style>body{font:15px sans-serif;margin:2rem}section{display:flex;flex-wrap:wrap;gap:1rem}article{width:300px;border:1px solid #aaa;padding:8px}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}.correct,.UNKNOWN-correct{background:#d2eee3}.donor-color{background:#ffd4b9}.invalid{background:#eed4ec}.UNKNOWN-on-visible{background:#fff0bb}.other-wrong{background:#ddd}</style></head><body>',
             '<h1>Prediction gallery</h1>',
             '<label>Outcome <select id="filter"><option value="all">All</option>']
    classes = ('correct', 'donor-color', 'other wrong', 'UNKNOWN-correct', 'UNKNOWN-on-visible', 'invalid')
    parts += [f'<option value="{esc(c)}">{esc(c)}</option>' for c in classes]
    parts.append('</select></label>')
    levels = sorted({r.get('evidence_level', 'unknown') for r in rows})
    parts.append(f'<p>Evidence: {esc(", ".join(levels))}</p>')
    for scene, images in sorted(groups.items()):
        parts.append(f'<h2>{esc(scene)}</h2><section>')
        order = ['I11', 'I01', 'I10', 'I00', 'clean', 'iso_A', 'iso_B']
        for key in sorted(images, key=lambda k: (order.index(k) if k in order else len(order), k)):
            records = images[key]
            relative = os.path.relpath(root / records[0]['image_path'], output.parent)
            outcomes = [outcome(r) for r in records]
            parts.append(f'<article data-outcomes="{esc("|".join(outcomes))}"><h3>{esc(key)}</h3><img loading="lazy" src="{esc(relative)}" alt="{esc(scene + ": " + key)}">')
            for row in records:
                cls = outcome(row)
                parts.append(f'<div class="{esc(cls.replace(" ", "-"))}"><p>Person {esc(row["target_person"])}; tag {esc(row["target_tag"])}; gold {esc(row["gold"]["task_gold"])}; parsed {esc(row["label"])}; {esc(cls)}</p><pre>{esc(row["raw_text"])}</pre></div>')
            parts.append('</article>')
        parts.append('</section>')
    parts.append('<script>document.getElementById("filter").addEventListener("change",e=>{document.querySelectorAll("article").forEach(a=>a.hidden=e.target.value!=="all"&&!a.dataset.outcomes.split("|").includes(e.target.value))});</script></body></html>')
    output.write_text('\n'.join(parts), encoding='utf-8')
    return output
