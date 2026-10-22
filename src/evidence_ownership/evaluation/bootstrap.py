"""Seeded cluster percentile bootstrap and exact zero-event occurrence bounds."""
from __future__ import annotations

import numpy as np
from scipy.stats import beta

from evidence_ownership.evaluation.estimand import METRICS, cluster_values, deduplicate, donor_match, scene_contrasts
from evidence_ownership.utils.config import experiment_config
from evidence_ownership.utils.logs import progress


def zero_event_upper_bound(n: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson bound for zero events in n independent units."""
    if n <= 0:
        raise ValueError('n must be positive')
    return float(beta.ppf(confidence, 1, n))


def bootstrap_metric(scenes: list[dict], metric: str, *, n_resamples: int = 10000,
                     seed: int = 20261022, no_progress: bool = True) -> dict:
    values = np.array(list(cluster_values(scenes, metric).values()), dtype=float)
    if not len(values) or n_resamples <= 0:
        raise ValueError('Bootstrap requires nonempty clusters and positive resamples')
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples)
    # Vectorized batches cap memory without changing the random stream or sampling unit.
    batches = range(0, n_resamples, 512)
    for start in progress(batches, desc=f'Bootstrap {metric}', unit='batch', disable=no_progress):
        stop = min(n_resamples, start + 512)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        samples[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return dict(point=float(values.mean()), ci=[float(low), float(high)], n_clusters=len(values),
                n_scenes=len(scenes), bootstrap_unit='cluster', n_resamples=n_resamples, seed=seed,
                interval='two-sided 95% percentile',
                numerator=float(values.sum()), denominator=len(values),
                numerator_definition='sum of within-cluster mean scene contrasts')


def bootstrap_contrasts(rows: list[dict], variant: str = 'primary', *, n_resamples: int | None = None,
                        seed: int | None = None, no_progress: bool = True) -> dict:
    config = experiment_config()['analysis']
    n_resamples = n_resamples if n_resamples is not None else config['bootstrap_resamples']
    seed = seed if seed is not None else config['seed']
    scenes = scene_contrasts(rows, variant)
    result = {metric: bootstrap_metric(scenes, metric, n_resamples=n_resamples, seed=seed,
                                        no_progress=no_progress) for metric in METRICS}
    counts = {}
    for event in ('any_donor_match', 'any_invalid', 'any_task_failure', 'full_factorial_success'):
        events = sum(s[event] for s in scenes)
        clusters = {}
        for s in scenes:
            clusters[s['cluster_id']] = max(clusters.get(s['cluster_id'], 0), s[event])
        n_cluster_events = sum(clusters.values())
        counts[event] = dict(scene_events=events, n_scenes=len(scenes), cluster_events=n_cluster_events,
                             n_clusters=len(clusters),
                             zero_event_scene_upper_95=zero_event_upper_bound(len(scenes)) if events == 0 else None,
                             zero_event_cluster_upper_95=zero_event_upper_bound(len(clusters)) if n_cluster_events == 0 else None,
                             scene_bound_assumes_independent_scenes=len(clusters) == len(scenes))
    raw = [r for r in deduplicate(rows) if r['variant'] == variant and r['condition'] in ('I11', 'I01', 'I10', 'I00')]
    result.update(variant=variant, scene_event_counts=counts,
                  donor_match_count=sum(donor_match(r) for r in raw), query_denominator=len(raw))
    return result
