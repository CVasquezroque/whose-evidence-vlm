"""Controls, response composition, sensitivity analyses and explicit support decisions."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy

from evidence_ownership.evaluation.bootstrap import bootstrap_contrasts, bootstrap_metric
from evidence_ownership.evaluation.estimand import (CELLS, cluster_values, correct, deduplicate,
                                                   definite, donor_match, scene_contrasts)
from evidence_ownership.utils.config import experiment_config


def rate(numerator: int | float, denominator: int) -> dict:
    return dict(numerator=numerator, denominator=denominator,
                rate=numerator / denominator if denominator else None)


def composition(row: dict) -> str:
    if not row['valid']:
        return 'invalid'
    if row['label'] == 'UNKNOWN':
        return 'UNKNOWN'
    if row['label'] == row['gold']['target_latent_color']:
        return 'target_color'
    if row['label'] == row['gold']['donor_latent_color']:
        return 'donor_color'
    return 'other_color'


def outcome(row: dict) -> str:
    if not row['valid']:
        return 'invalid'
    if row['label'] == 'UNKNOWN':
        return 'UNKNOWN-correct' if row['gold']['task_gold'] == 'UNKNOWN' else 'UNKNOWN-on-visible'
    if correct(row):
        return 'correct'
    return 'donor-color' if donor_match(row) else 'other wrong'


def summarize_group(rows: list[dict]) -> dict:
    n = len(rows)
    comp = Counter(composition(r) for r in rows)
    visible = [r for r in rows if r['gold']['target_visible']]
    visible_counts = Counter('invalid' if not r['valid'] else 'UNKNOWN' if r['label'] == 'UNKNOWN'
                             else 'correct' if correct(r) else 'wrong' for r in visible)
    hidden = [r for r in rows if not r['gold']['target_visible']]
    return dict(n=n, accuracy=rate(sum(correct(r) for r in rows), n),
                donor_match=rate(sum(donor_match(r) for r in rows), n),
                UNKNOWN=rate(comp['UNKNOWN'], n), invalid=rate(comp['invalid'], n),
                any_definite_color=rate(sum(definite(r) for r in rows), n),
                composition={k: rate(comp[k], n) for k in ('target_color', 'donor_color', 'other_color', 'UNKNOWN', 'invalid')},
                visible={k: rate(visible_counts[k], len(visible)) for k in ('correct', 'wrong', 'UNKNOWN', 'invalid')},
                hidden_composition={k: rate(sum(composition(r) == k for r in hidden), len(hidden))
                                    for k in ('target_color', 'donor_color', 'other_color', 'UNKNOWN', 'invalid')})


def invalid_bounds(rows: list[dict]) -> dict:
    result = {}
    for metric in ('psi', 'delta_hidden'):
        bounds = []
        for assignment in ('min', 'max'):
            scenes = scene_contrasts(rows, invalid_assignment=assignment, bound_metric=metric)
            clusters = cluster_values(scenes, metric)
            bounds.append(sum(clusters.values()) / len(clusters))
        result[metric] = dict(lower=bounds[0], upper=bounds[1])
    return result


def pilot_gates(controls: dict, valid_rate: float | None) -> dict:
    config = experiment_config()['pilot_gates']
    specs = [('clean', controls['clean']['accuracy']['rate'], config['clean_accuracy_min'], '>=', 'clean_accuracy_min'),
             ('isolated', controls['iso']['accuracy']['rate'], config['isolated_accuracy_min'], '>=', 'isolated_accuracy_min'),
             ('sham_drop', controls['paired_clean_minus_sham']['point'], config['max_sham_accuracy_drop'], '<=', 'max_sham_accuracy_drop'),
             ('valid_json', valid_rate, config['valid_json_min'], '>=', 'valid_json_min')]
    gates = {}
    for name, value, threshold, operator, source in specs:
        passed = value is not None and (value >= threshold if operator == '>=' else value <= threshold)
        if name == 'sham_drop' and controls['paired_clean_minus_sham'].get('n_missing_pairs', 0):
            passed = False
        gates[name] = dict(status='PASS' if passed else 'FAIL', measured=value, threshold=threshold,
                           operator=operator, config_key=source,
                           reason='Threshold met' if passed else 'Missing control or threshold failed')
    return dict(status='PASS' if all(g['status'] == 'PASS' for g in gates.values()) else 'FAIL', gates=gates,
                scope='Measured model-interface gates only; human audit, yield and GPU feasibility require separate review')


def primary_support(contrasts: dict, gates: dict, valid_rate: float | None,
                    bounds: dict, annotation_status: str, evidence_level: str) -> dict:
    reasons = []
    for metric in ('psi', 'delta_hidden'):
        if contrasts[metric]['ci'][0] <= 0:
            reasons.append(f'{metric} lower 95% CI is not above zero')
    if gates['status'] != 'PASS':
        reasons.append('Recognition, sham or JSON control gate failed')
    if valid_rate is None or valid_rate < 0.99:
        reasons.append('Valid-output rate is below 0.99')
    if any(bounds[m]['lower'] <= 0 for m in ('psi', 'delta_hidden')):
        reasons.append('Adversarial invalid-output bounds do not retain both positive contrasts')
    if annotation_status != 'two_rater_agreed':
        reasons.append('Independent two-rater gold is not established for every scene')
    if evidence_level in ('synthetic_fixture', 'dry_run'):
        reasons.append('Synthetic or fake-backend evidence cannot support a research claim')
    if evidence_level == 'pilot':
        reasons.append('Pilot evidence is excluded from confirmatory support')
    return dict(status='SUPPORTED' if not reasons else 'NOT_SUPPORTED',
                reasons=reasons or ['Both lower confidence bounds exceed zero and all interpretation controls pass'],
                practically_substantial=contrasts['psi']['point'] >= 0.10,
                evidence_level=evidence_level)


def analytic_baselines(rows: list[dict], reference_labels: list[str] | None = None) -> dict:
    rows = deduplicate(rows)
    labels = reference_labels or []
    constant = Counter(labels).most_common(1)[0][0] if labels else None
    results = {}
    for name in ('always_UNKNOWN', 'constant_color', 'scene_heuristic'):
        if name == 'constant_color' and constant is None:
            results[name] = dict(status='SKIPPED', reason='No supplied reference (pilot) labels')
            continue
        simulated = deepcopy(rows)
        for row in simulated:
            if name == 'always_UNKNOWN':
                label = 'UNKNOWN'
            elif name == 'constant_color':
                label = constant
            else:
                gold = row['gold']
                if gold['target_visible']:
                    label = gold['target_latent_color']
                elif gold['donor_visible']:
                    label = gold['donor_latent_color']
                else:
                    label = 'UNKNOWN'
            row.update(label=label, valid=True)
        scenes = scene_contrasts(simulated)
        results[name] = dict(status='PASS', constant_color=constant if name == 'constant_color' else None,
                             contrasts={m: sum(cluster_values(scenes, m).values()) / len(cluster_values(scenes, m))
                                        for m in ('psi', 'delta_hidden', 'delta_visible')},
                             accuracy=summarize_group([r for r in simulated if r['condition'] in CELLS and r['variant'] == 'primary'])['accuracy'],
                             joint_success=sum(s['joint_success'] for s in scenes) / len(scenes),
                             full_factorial_success=sum(s['full_factorial_success'] for s in scenes) / len(scenes))
    results['uniform_random'] = dict(status='PASS', contrasts=dict(psi=0.0, delta_hidden=0.0, delta_visible=0.0),
                                     expected_accuracy=1 / 12, expected_donor_match=1 / 12,
                                     expected_joint_success=1 / 144, expected_full_factorial_success=12 ** -8)
    results['definitions'] = dict(scene_heuristic='Use the sole visible color irrespective of target; when both visible use target color, when neither visible UNKNOWN',
                                  constant_reference='Supplied pilot/reference latent labels only; never inferred from held-out outputs',
                                  constant_tie_break='First appearance in supplied reference labels')
    return results


def diagnostics(rows: list[dict], *, n_resamples: int = 10000, seed: int = 20261022,
                no_progress: bool = True) -> dict:
    rows = deduplicate(rows)
    primary = [r for r in rows if r['variant'] == 'primary']
    factorial = [r for r in primary if r['condition'] in CELLS]
    scenes = scene_contrasts(rows)
    boot_kw = dict(n_resamples=n_resamples, seed=seed, no_progress=no_progress)
    cells = []
    for cell in CELLS:
        for visible in (False, True):
            group = [r for r in factorial if r['condition'] == cell and r['gold']['target_visible'] == visible]
            if group:
                cells.append(dict(condition=cell, target_visible=visible, **summarize_group(group)))
    controls = {c: summarize_group([r for r in primary if r['condition'] == c]) for c in ('clean', 'I11', 'iso')}
    by_scene = defaultdict(dict)
    for row in primary:
        if row['condition'] in ('clean', 'I11'):
            by_scene[row['scene_id']][(row['condition'], row['target_person'])] = row
    paired = []
    for sid, group in sorted(by_scene.items()):
        if all((c, t) in group for c in ('clean', 'I11') for t in ('A', 'B')):
            paired.append(dict(scene_id=sid, cluster_id=group[('clean', 'A')]['cluster_id'],
                               drop=sum(correct(group[('clean', p)]) - correct(group[('I11', p)]) for p in ('A', 'B')) / 2))
    controls['paired_clean_minus_sham'] = bootstrap_metric(paired, 'drop', **boot_kw) if paired else dict(point=None, ci=None, n_scenes=0)
    controls['paired_clean_minus_sham']['n_missing_pairs'] = len(scenes) - len(paired)
    subgroups = {}
    for field in ('target_tag', 'target_side'):
        groups = defaultdict(list)
        for row in factorial:
            groups[str(row.get(field, 'unknown'))].append(row)
        subgroups[field] = {key: summarize_group(group) for key, group in sorted(groups.items())}
    colors = defaultdict(list)
    for row in factorial:
        key = '/'.join(sorted([row['gold']['target_latent_color'], row['gold']['donor_latent_color']]))
        colors[key].append(row)
    excluded = {r['scene_id'] for r in primary if 'gray' in (r['gold']['target_latent_color'], r['gold']['donor_latent_color'])}
    sensitivity_rows = [r for r in rows if r['scene_id'] not in excluded]
    sensitivity = bootstrap_contrasts(sensitivity_rows, **boot_kw) if sensitivity_rows else dict(status='SKIPPED', reason='All scenes contain gray garments')
    tag_rows = [r for r in rows if r['variant'] == 'tag_reversal']
    tag = bootstrap_contrasts(tag_rows, variant='tag_reversal', **boot_kw) if tag_rows else dict(status='SKIPPED', reason='No tag-reversal scenes')
    recognition_pairs = []
    controls_by_person = defaultdict(dict)
    for row in primary:
        if row['condition'] in ('clean', 'iso'):
            controls_by_person[(row['scene_id'], row['target_person'])][row['condition']] = row
    for key, group in controls_by_person.items():
        if 'clean' in group and 'iso' in group:
            recognition_pairs.append((correct(group['clean']), correct(group['iso'])))
    return dict(cells=cells, controls=controls, subgroups=subgroups,
                color_pairs={key: summarize_group(group) for key, group in sorted(colors.items())},
                h2_any_color_change=bootstrap_metric(scenes, 'h2_any_color_change', **boot_kw),
                joint_selective_success={c: bootstrap_metric(scenes, 'joint_' + c, **boot_kw) for c in ('I01', 'I10')},
                full_factorial_success=bootstrap_metric(scenes, 'full_factorial_success', **boot_kw),
                gray_excluded=dict(n_excluded_scenes=len(excluded), results=sensitivity), tag_reversal=tag,
                valid_output_rate=rate(sum(r['valid'] for r in primary), len(primary)),
                invalid_by_condition={c: rate(sum(not r['valid'] for r in primary if r['condition'] == c),
                                                  sum(r['condition'] == c for r in primary)) for c in (*CELLS, 'clean', 'iso')},
                truncation_by_condition={c: rate(sum(r.get('hit_max_new_tokens', False) for r in primary if r['condition'] == c),
                                                      sum(r['condition'] == c for r in primary)) for c in (*CELLS, 'clean', 'iso')},
                isolation_clean_relation={f'clean_{a}_iso_{b}': sum(pair == (a, b) for pair in recognition_pairs)
                                          for a in (0, 1) for b in (0, 1)},
                operating_point=dict(visible_risk=rate(sum(definite(r) and not correct(r) for r in factorial if r['T'] == 1),
                                                        sum(definite(r) for r in factorial if r['T'] == 1)),
                                     hidden_definite_label_rate=rate(sum(definite(r) for r in factorial if r['T'] == 0),
                                                                    sum(r['T'] == 0 for r in factorial))))
