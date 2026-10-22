"""Statistical validity and sample separation checks."""
from __future__ import annotations

from evidence_ownership.evaluation.estimand import deduplicate


def stats_checks(rows: list[dict], contrasts: dict, *, pilot_scene_ids: set[str] | None = None,
                 heldout_scene_ids: set[str] | None = None) -> list[dict]:
    rows = deduplicate(rows)
    rate = sum(bool(r['valid']) for r in rows) / len(rows) if rows else None
    overlap = (pilot_scene_ids or set()) & (heldout_scene_ids or set())
    separation_known = pilot_scene_ids is not None and heldout_scene_ids is not None
    cluster_ok = all(contrasts[m].get('bootstrap_unit') == 'cluster' for m in ('psi', 'delta_hidden', 'delta_visible'))
    return [dict(name='valid_rate', status='PASS' if rate is not None and rate >= 0.99 else 'FAIL',
                 detail=dict(rate=rate, minimum=0.99)),
            dict(name='pilot_heldout_disjoint', status=('PASS' if not overlap else 'FAIL') if separation_known else 'SKIPPED',
                 detail=dict(overlap=sorted(overlap), known=separation_known)),
            dict(name='bootstrap_unit_cluster', status='PASS' if cluster_ok else 'FAIL', detail='Scenes averaged within clusters before resampling')]
