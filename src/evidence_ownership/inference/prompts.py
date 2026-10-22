"""Frozen observation-only prompts and their content hashes."""
from __future__ import annotations

from evidence_ownership.utils.config import inference_config
from evidence_ownership.utils.hashing import canonical_json, sha256_text


def build_prompt(target: str, config: dict | None = None) -> dict:
    if target not in ('A', 'B'):
        raise ValueError('target tag must be A or B')
    prompts = (config or inference_config())['prompts']
    system = prompts['system']
    user = prompts['user'].format(target=target)
    return {'system_prompt': system, 'user_prompt': user,
            'prompt_sha256': sha256_text(canonical_json({'system': system, 'user': user}))}


def prompt_hash() -> str:
    """Lock both rendered target prompts, rather than just the template."""
    return sha256_text(canonical_json({tag: build_prompt(tag) for tag in ('A', 'B')}))
