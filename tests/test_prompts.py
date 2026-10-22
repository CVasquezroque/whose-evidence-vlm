import json
from pathlib import Path

from evidence_ownership.inference.prompts import build_prompt
from evidence_ownership.utils.config import REPO_ROOT
from evidence_ownership.utils.hashing import canonical_json, sha256_text


def test_exact_scientific_protocol_strings():
    protocol = REPO_ROOT / 'provenance/protocol/proposed_protocol_v1.0.json'
    spec = json.loads(protocol.read_text())['prompts']
    for tag in ('A', 'B'):
        rendered = build_prompt(tag)
        assert rendered['system_prompt'] == spec['system']
        assert rendered['user_prompt'] == spec['user'].replace('{A_or_B}', tag)
        assert rendered['prompt_sha256'] == sha256_text(canonical_json(dict(system=rendered['system_prompt'], user=rendered['user_prompt'])))
