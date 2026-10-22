import pytest

from evidence_ownership.inference.parser import parse_predictions, parse_response
from evidence_ownership.inference.queries import build_queries
from evidence_ownership.utils.io import read_jsonl, write_jsonl
from test_queries import fixture_work


@pytest.mark.parametrize('raw,reason', [
    ('', 'empty'), ('blue and red', 'not_json'), ('{"label":', 'not_json'),
    ('["blue"]', 'not_object'), ('null', 'not_object'),
    ('{"color":"blue"}', 'wrong_keys'), ('{"label":"blue","why":"red"}', 'wrong_keys'),
    ('{"label":3}', 'label_not_string'), ('{"label":null}', 'label_not_string'),
    ('{"label":"navy"}', 'label_not_in_vocabulary'), ('{"label":"Blue"}', 'label_not_in_vocabulary'),
    ('{"label":"unknown"}', 'label_not_in_vocabulary'), ('```json\n{"label":"blue"}\n```', 'not_json'),
    ('{"label":"blue"} red', 'not_json'), ('{"label":"blue or red"}', 'label_not_in_vocabulary')])
def test_strict_rejections(raw, reason):
    assert parse_response(raw, allow_json_fence=False) == dict(label=None, valid=False, invalid_reason=reason)


def test_valid_truncation_and_fences():
    assert parse_response(' \n{"label":"UNKNOWN"} \n')['valid']
    assert parse_response('{"label":"blue"}', hit_max_new_tokens=True)['valid']
    assert parse_response('{"label":', hit_max_new_tokens=True)['invalid_reason'] == 'truncated'
    assert parse_response('```json\n{"label":"red"}\n```', allow_json_fence=True)['label'] == 'red'


def test_latest_ok_join_and_missing(fixture_work):
    rows = build_queries('pilot', 'parser', no_progress=True)
    run = fixture_work/'runs/parser'
    write_jsonl(run/'raw.jsonl', [dict(query_id=rows[0]['query_id'], attempt=1, status='ok', raw_text='{"label":"red"}', hit_max_new_tokens=False),
                                dict(query_id=rows[0]['query_id'], attempt=2, status='ok', raw_text='{"label":"blue"}', hit_max_new_tokens=False),
                                dict(query_id=rows[1]['query_id'], attempt=1, status='error')])
    report = parse_predictions(run/'queries.jsonl', run/'raw.jsonl')
    assert report['n_missing'] == len(rows)-1
    assert read_jsonl(run/'parsed_predictions.jsonl')[0]['label'] == 'blue'
