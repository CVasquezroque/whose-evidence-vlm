#!/usr/bin/env python3
"""Freeze the protocol before held-out inference: hash every input that defines the evaluation.

Writes provenance/protocol/freeze_<version>.json and .SHA256SUMS. It covers the configs, the
rendered prompts, all package and script source, the annotation interface, the decision log,
amendments, the pre-specified protocol, and the pilot results that informed the freeze. It refuses
if the version already exists, or if any held-out raw predictions already exist.

Example:
  python -u scripts/freeze_protocol.py --version 1.0-frozen --note "no pilot revision; gates failed at n=3"
"""
from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path

from evidence_ownership.inference.prompts import build_prompt
from evidence_ownership.utils.config import REPO_ROOT, get_paths
from evidence_ownership.utils.hashing import sha256_file, sha256_json
from evidence_ownership.utils.io import write_json

TRACKED = ["configs/experiment.yaml", "configs/inference.yaml", "configs/model.yaml", "provenance/decision_log.csv",
           "provenance/protocol/proposed_protocol_v1.0.json", "pyproject.toml"]
TREES = ["src/evidence_ownership", "scripts", "annotation/interface", "provenance/amendments"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--science-dir", type=Path, default=None, help="optional folder with the original protocol report")
    a = ap.parse_args()
    paths = get_paths()
    out = REPO_ROOT / "provenance" / "protocol" / f"freeze_{a.version}"
    if Path(f"{out}.json").exists():
        raise SystemExit(f"{out} exists; a freeze is immutable")
    heldout_raw = [p for p in paths.runs.glob("*/raw_predictions*.jsonl") if "heldout" in p.parent.name]
    if heldout_raw:
        raise SystemExit(f"held-out predictions already exist ({heldout_raw[0]}); freezing now would be post hoc")
    files = {f: REPO_ROOT / f for f in TRACKED}
    for t in TREES:
        for p in sorted((REPO_ROOT / t).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                files[str(p.relative_to(REPO_ROOT))] = p
    if a.science_dir:
        for n in ("PROPOSED_PROTOCOL.json", "RESEARCH_DECISION_REPORT.md"):
            files[f"design_assessment/{n}"] = a.science_dir / n
    for n in ("pilot_gates.json", "primary_contrasts.json", "analysis_manifest.json"):
        f = paths.results / "pilot" / n
        if f.exists():
            files[f"work_root/results/pilot/{n}"] = f
    for n in ("items_pilot.jsonl", "candidates_val2017.jsonl", "images_val2017.jsonl"):
        f = paths.manifests / n
        if f.exists():
            files[f"work_root/manifests/{n}"] = f
    digests = {k: sha256_file(v) for k, v in sorted(files.items())}
    prompts = {t: build_prompt(t) for t in ("A", "B")}
    rec = {"protocol_version": a.version, "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "note": a.note, "registration": "internal hash-locked freeze; NOT an external preregistration",
           "rendered_prompts": prompts, "rendered_prompts_sha256": sha256_json(prompts),
           "python": platform.python_version(), "n_files": len(digests), "sha256": digests}
    write_json(Path(f"{out}.json"), rec)
    Path(f"{out}.SHA256SUMS").write_text("".join(f"{d}  {k}\n" for k, d in digests.items())
                                         + f"{sha256_file(Path(f'{out}.json'))}  {out.name}.json\n")
    print(f"frozen {a.version}: {len(digests)} files -> {out}")


if __name__ == "__main__":
    main()
