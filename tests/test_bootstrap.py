from copy import deepcopy

import pytest

from evidence_ownership.evaluation.bootstrap import bootstrap_contrasts, zero_event_upper_bound
from test_estimand import synthetic_rows


def test_duplication_does_not_narrow_ci_or_change_point():
    rows = synthetic_rows('copy', 6)
    for r in rows:
        if r['scene_id'] in ('s0', 's1', 's2'):
            r['label'] = r['gold']['task_gold']
    first = bootstrap_contrasts(rows)
    duplicate = bootstrap_contrasts(rows + deepcopy(rows))
    assert first == duplicate
    assert first['psi']['point'] == 0.5
    assert first['psi']['ci'][0] < first['psi']['ci'][1]
    assert first['psi']['n_clusters'] == 6 and first['query_denominator'] == 48
    assert first['scene_event_counts']['any_donor_match']['scene_events'] == 3


def test_exact_zero_event_bound_and_counts():
    result = bootstrap_contrasts(synthetic_rows('correct', 80), n_resamples=200)
    assert result['psi']['ci'] == [0, 0]
    event = result['scene_event_counts']['any_task_failure']
    assert event['scene_events'] == 0 and event['n_scenes'] == 80
    assert zero_event_upper_bound(80) == pytest.approx(1 - 0.05 ** (1/80))
    assert event['zero_event_scene_upper_95'] == pytest.approx(0.0367542, abs=1e-6)
    assert event['zero_event_cluster_upper_95'] == event['zero_event_scene_upper_95']


def test_identical_seed_reproducible():
    assert bootstrap_contrasts(synthetic_rows('copy'), n_resamples=200) == bootstrap_contrasts(synthetic_rows('copy'), n_resamples=200)
