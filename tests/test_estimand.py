from copy import deepcopy

import pytest

from evidence_ownership.evaluation.diagnostics import diagnostics, invalid_bounds, pilot_gates
from evidence_ownership.evaluation.estimand import estimate, scene_contrasts


def synthetic_rows(mode='correct', n=4):
    rows = []
    for index in range(n):
        for condition in ('I11', 'I01', 'I10', 'I00', 'clean', 'iso'):
            for target in ('A', 'B'):
                factorial = condition.startswith('I')
                av, bv = (int(condition[1]), int(condition[2])) if factorial else (1, 1)
                t, d = (av, bv) if target == 'A' else (bv, av)
                color, donor = ('blue', 'red') if target == 'A' else ('red', 'blue')
                gold = color if t else 'UNKNOWN'
                label = gold
                if mode == 'unknown':
                    label = 'UNKNOWN'
                elif mode == 'copy' and not t and d:
                    label = donor
                rows.append(dict(query_id=f'{index}-{condition}-{target}', scene_id=f's{index}', cluster_id=f'c{index}',
                                 study_split='pilot', variant='primary', condition=condition, target_person=target,
                                 target_tag=target, target_side='left' if target == 'B' else 'right',
                                 T=t if factorial else None, D=d if factorial else None, valid=True, label=label,
                                 evidence_level='synthetic_fixture', annotation_status='two_rater_agreed',
                                 gold=dict(task_gold=gold, target_visible=bool(t), donor_visible=bool(d) if condition != 'iso' else None,
                                           target_latent_color=color, donor_latent_color=donor)))
    return rows


@pytest.mark.parametrize('mode,hidden,visible,psi', [('unknown',0,0,0), ('correct',0,0,0), ('copy',1,0,1)])
def test_known_contrasts(mode, hidden, visible, psi):
    assert estimate(synthetic_rows(mode)) == dict(delta_hidden=hidden, delta_visible=visible, psi=psi)


def test_target_b_reverse_role_mapping():
    rows = synthetic_rows('correct', 1)
    b_hidden_revealed = next(r for r in rows if r['condition'] == 'I10' and r['target_person'] == 'B')
    b_hidden_revealed['label'] = 'blue'
    assert estimate(rows)['delta_hidden'] == 0.5
    assert scene_contrasts(rows)[0]['targets']['B']['delta_hidden'] == 1
    # Changing B/I01 instead changes its visible contrast.
    b_visible_hidden_donor = next(r for r in rows if r['condition'] == 'I01' and r['target_person'] == 'B')
    b_visible_hidden_donor['label'] = 'blue'
    assert estimate(rows)['delta_visible'] == -0.5


def test_clusters_with_two_scenes_average_first():
    rows = synthetic_rows('copy', 3)
    for r in rows:
        if r['scene_id'] == 's0':
            r['label'] = r['gold']['task_gold']
        if r['scene_id'] == 's1':
            r['cluster_id'] = 'c0'
    assert estimate(rows)['psi'] == 0.75  # mean((0+1)/2, 1)


def test_success_and_overabstention():
    good = diagnostics(synthetic_rows('correct'), n_resamples=200)
    bad = diagnostics(synthetic_rows('unknown'), n_resamples=200)
    assert good['full_factorial_success']['point'] == 1
    assert good['joint_selective_success']['I01']['point'] == 1
    assert good['h2_any_color_change']['point'] == 0
    assert bad['controls']['clean']['accuracy']['rate'] == 0
    assert pilot_gates(bad['controls'], 1)['status'] == 'FAIL'


def test_invalid_denominator_and_adversarial_bounds():
    rows = synthetic_rows('copy', 1)
    row = next(r for r in rows if r['condition'] == 'I01' and r['target_person'] == 'A')
    row.update(valid=False, label=None)
    assert estimate(rows)['delta_hidden'] == 0.5
    bounds = invalid_bounds(rows)
    for metric in ('psi', 'delta_hidden'):
        assert bounds[metric]['lower'] <= estimate(rows)[metric] <= bounds[metric]['upper']
    assert bounds['delta_hidden'] == dict(lower=0.5, upper=1.0)
    diag = diagnostics(rows, n_resamples=100)
    assert diag['valid_output_rate']['denominator'] == 12
    assert diag['valid_output_rate']['numerator'] == 11


def test_incomplete_scenes_and_conflicts_rejected():
    rows = synthetic_rows('correct', 1)
    with pytest.raises(ValueError, match='Incomplete'):
        estimate(rows[1:])
    bad = deepcopy(rows[0])
    bad['label'] = 'red'
    with pytest.raises(ValueError, match='Conflicting'):
        estimate(rows+[bad])


def test_analytic_baselines_and_support_reasons():
    from evidence_ownership.evaluation.bootstrap import bootstrap_contrasts
    from evidence_ownership.evaluation.diagnostics import analytic_baselines, primary_support
    from evidence_ownership.validation.stats_checks import stats_checks
    rows = synthetic_rows('copy', 4)
    baselines = analytic_baselines(rows, ['red', 'red', 'blue'])
    assert baselines['always_UNKNOWN']['contrasts']['psi'] == 0
    assert baselines['constant_color']['constant_color'] == 'red'
    assert baselines['constant_color']['contrasts']['psi'] == 0
    assert baselines['scene_heuristic']['contrasts']['psi'] == 1
    assert baselines['uniform_random']['expected_accuracy'] == 1/12
    contrasts = bootstrap_contrasts(rows, n_resamples=100)
    diag = diagnostics(rows, n_resamples=100)
    gates = pilot_gates(diag['controls'], 1)
    verdict = primary_support(contrasts, gates, 1, invalid_bounds(rows), 'two_rater_agreed', 'heldout')
    assert verdict['status'] == 'SUPPORTED' and verdict['reasons']
    assert primary_support(contrasts, gates, 1, invalid_bounds(rows), 'two_rater_agreed', 'dry_run')['status'] == 'NOT_SUPPORTED'
    checks = stats_checks(rows, contrasts, pilot_scene_ids={'s0'}, heldout_scene_ids={'s0'})
    assert next(c for c in checks if c['name'] == 'pilot_heldout_disjoint')['status'] == 'FAIL'
    assert next(c for c in checks if c['name'] == 'bootstrap_unit_cluster')['status'] == 'PASS'
