"""Static publication artifacts; no PDF or graphical display dependency."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from evidence_ownership.utils.io import read_json

COLORS = ['#0072B2', '#D55E00', '#CC79A7', '#009E73', '#999999']


def make_figures(results_dir: Path, output_dir: Path | None = None) -> list[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    contrasts = read_json(results_dir / 'primary_contrasts.json')
    diag = read_json(results_dir / 'diagnostics.json')
    level, n = contrasts['evidence_level'], contrasts['psi']['n_clusters']
    title_suffix = f'{level}; n={n} clusters'
    paths = []

    def save(fig, name: str, title: str) -> None:
        fig.suptitle(f'{title}\n{title_suffix}')
        if level in ('synthetic_fixture', 'dry_run'):
            fig.text(0.5, 0.5, 'SYNTHETIC / DRY-RUN', ha='center', va='center', fontsize=26,
                     alpha=0.16, rotation=25, zorder=20)
        fig.tight_layout()
        path = output_dir / name
        fig.savefig(path, dpi=300, metadata={'evidence_level': level, 'annotation_status': contrasts['annotation_status']})
        plt.close(fig)
        paths.append(path)

    cells = diag['cells']
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(cells))
    for category, color in zip(('target_color', 'donor_color', 'other_color', 'UNKNOWN', 'invalid'), COLORS):
        heights = [c['composition'][category]['rate'] or 0 for c in cells]
        ax.bar(range(len(cells)), heights, bottom=bottom, label=category.replace('_', ' '), color=color)
        bottom += heights
    ax.set_xticks(range(len(cells)), [f'{c["condition"]}\nT={int(c["target_visible"])}' for c in cells])
    ax.set(ylabel='Response proportion', ylim=(0, 1))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=8)
    save(fig, 'response_composition_by_cell.png', 'Response composition by cell and target visibility')

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, metric in enumerate(('delta_hidden', 'delta_visible', 'psi')):
        value = contrasts[metric]
        point, (low, high) = value['point'], value['ci']
        ax.errorbar(i, point, yerr=[[max(0, point - low)], [max(0, high - point)]], fmt='o',
                    capsize=5, color=COLORS[i])
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(0.10, color='#999999', linestyle='--', label='0.10 reference')
    ax.set_xticks(range(3), ['delta_hidden', 'delta_visible', 'psi'])
    ax.set_ylabel('Paired effect (95% cluster bootstrap CI)')
    ax.legend()
    save(fig, 'donor_effects_and_interaction.png', 'Donor effects and interaction')

    fig, ax = plt.subplots(figsize=(7, 4.5))
    entries = [diag['joint_selective_success'][c] for c in ('I01', 'I10')] + [diag['full_factorial_success']]
    for i, val in enumerate(entries):
        point, (low, high) = val['point'], val['ci']
        ax.bar(i, point, color=COLORS[i], width=0.6)
        ax.errorbar(i, point, yerr=[[max(0, point-low)], [max(0, high-point)]], color='black', capsize=5)
    ax.set_xticks(range(3), ['I01 joint', 'I10 joint', 'All eight correct'])
    ax.set(ylabel='Success proportion (95% CI)', ylim=(0, 1.05))
    save(fig, 'joint_success.png', 'Selective and full factorial success')
    return paths
