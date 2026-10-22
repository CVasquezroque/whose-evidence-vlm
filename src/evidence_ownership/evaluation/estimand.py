"""Paired target contrasts, averaged within scenes and then within clusters."""
from __future__ import annotations

from collections import defaultdict

CELLS = ('I11', 'I01', 'I10', 'I00')
METRICS = ('delta_hidden', 'delta_visible', 'psi')


def deduplicate(rows: list[dict]) -> list[dict]:
    """Repeated identical queries carry no extra weight; conflicting outcomes are errors."""
    unique = {}
    for row in rows:
        key = (row['scene_id'], row['variant'], row['condition'], row['target_person'])
        if key in unique:
            old = unique[key]
            fields = ('valid', 'label', 'gold', 'cluster_id')
            if any(old.get(f) != row.get(f) for f in fields):
                raise ValueError(f'Conflicting duplicate prediction: {key}')
        else:
            unique[key] = row
    return list(unique.values())


def donor_match(row: dict) -> int:
    return int(bool(row['valid']) and row['label'] == row['gold']['donor_latent_color'])


def correct(row: dict) -> int:
    return int(bool(row['valid']) and row['label'] == row['gold']['task_gold'])


def definite(row: dict) -> int:
    return int(bool(row['valid']) and row['label'] != 'UNKNOWN')


def scene_contrasts(rows: list[dict], variant: str = 'primary', *, invalid_assignment: str | None = None,
                    bound_metric: str = 'psi') -> list[dict]:
    groups = defaultdict(list)
    for row in deduplicate(rows):
        if row['variant'] == variant and row['condition'] in CELLS:
            groups[row['scene_id']].append(row)
    result = []
    for scene_id in sorted(groups):
        records = groups[scene_id]
        if len(records) != 8:
            raise ValueError(f'Incomplete factorial scene {scene_id}: {len(records)}/8 queries')
        cluster_ids = {r['cluster_id'] for r in records}
        if len(cluster_ids) != 1:
            raise ValueError(f'Conflicting cluster IDs in scene {scene_id}')
        values, targets = defaultdict(list), {}
        for target in ('A', 'B'):
            cells = {}
            for row in records:
                if row['target_person'] != target:
                    continue
                # Derive roles from the condition and canonical identity, independently of stored T/D.
                av, bv = int(row['condition'][1]), int(row['condition'][2])
                t, d = (av, bv) if target == 'A' else (bv, av)
                if row.get('T', t) != t or row.get('D', d) != d:
                    raise ValueError(f'Incorrect T/D roles in {scene_id}/{target}/{row["condition"]}')
                r = donor_match(row)
                if not row['valid'] and invalid_assignment:
                    coeff = {(0, 1): 1, (0, 0): -1, (1, 1): -1, (1, 0): 1}[(t, d)]
                    if bound_metric == 'delta_hidden' and t == 1:
                        coeff = 0
                    r = int(coeff > 0) if invalid_assignment == 'max' else int(coeff < 0)
                cells[(t, d)] = r
            if len(cells) != 4:
                raise ValueError(f'Incomplete target factorial {scene_id}/{target}')
            hidden = cells[(0, 1)] - cells[(0, 0)]
            visible = cells[(1, 1)] - cells[(1, 0)]
            targets[target] = dict(delta_hidden=hidden, delta_visible=visible, psi=hidden - visible)
            for metric in METRICS:
                values[metric].append(targets[target][metric])
        by_key = {(r['condition'], r['target_person']): r for r in records}
        h2 = []
        for target, revealed in (('A', 'I01'), ('B', 'I10')):
            h2.append(definite(by_key[(revealed, target)]) - definite(by_key[('I00', target)]))
        joint = {c: int(all(correct(by_key[(c, p)]) for p in ('A', 'B'))) for c in ('I01', 'I10')}
        result.append(dict(scene_id=scene_id, cluster_id=next(iter(cluster_ids)), targets=targets,
                           **{m: sum(v) / 2 for m, v in values.items()},
                           h2_any_color_change=sum(h2) / 2,
                           joint_I01=joint['I01'], joint_I10=joint['I10'],
                           joint_success=sum(joint.values()) / 2,
                           full_factorial_success=int(all(correct(r) for r in records)),
                           any_donor_match=int(any(donor_match(r) for r in records)),
                           any_invalid=int(any(not r['valid'] for r in records)),
                           any_task_failure=int(any(not correct(r) for r in records))))
    return result


def cluster_values(scenes: list[dict], metric: str) -> dict[str, float]:
    groups = defaultdict(list)
    for scene in scenes:
        groups[scene['cluster_id']].append(float(scene[metric]))
    return {cluster: sum(values) / len(values) for cluster, values in sorted(groups.items())}


def estimate(rows: list[dict], variant: str = 'primary') -> dict:
    scenes = scene_contrasts(rows, variant)
    if not scenes:
        raise ValueError(f'No complete scenes for {variant}')
    return {metric: sum(cluster_values(scenes, metric).values()) / len(cluster_values(scenes, metric))
            for metric in METRICS}
