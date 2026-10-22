#!/usr/bin/env python3
"""A02 nested size cohorts from ONE set of parsed predictions (no re-query).

For each cohort (original 48/2304, moderate 40/1500, expanded 32/900), keep scenes whose two
garments both pass, run the standard analysis into <out>/<cohort>/, and write
cohort_summary.json: estimands, visible/isolated recognition and the hidden-target response
breakdown (donor colour, gray occluder, target's own colour, other colour, UNKNOWN, invalid).

Example:
  python -u scripts/analyze_cohorts.py --run <work_root>/runs/heldout_qwen35 --split heldout --evidence-level heldout
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections import Counter
from pathlib import Path

from evidence_ownership.utils.config import REPO_ROOT, get_paths
from evidence_ownership.utils.io import read_json, read_jsonl, write_json, write_jsonl

COHORTS = ("original", "moderate", "expanded")


def response_class(r: dict) -> str:
    if not r["valid"]:
        return "invalid"
    g, lab = r["gold"], r["label"]
    if lab == "UNKNOWN":
        return "UNKNOWN"
    if lab == g["donor_latent_color"]:
        return "donor_color"
    if lab == g["target_latent_color"]:
        return "target_own_color"
    if lab == "gray":
        return "gray_occluder"
    return "other_color"


def rate(rows, ok) -> dict:
    n = len(rows)
    k = sum(ok(r) for r in rows)
    return {"k": k, "n": n, "rate": (k / n) if n else None}


def summarize(rows: list[dict]) -> dict:
    prim = [r for r in rows if r["variant"] == "primary"]
    correct = lambda r: r["valid"] and r["label"] == r["gold"]["task_gold"]
    hidden = [r for r in prim if r["condition"] in ("I01", "I10", "I00") and r["gold"]["target_visible"] is False]
    return {
        "n_scenes": len({r["scene_id"] for r in rows}), "n_queries": len(rows),
        "clean_accuracy": rate([r for r in prim if r["condition"] == "clean"], correct),
        "both_sham_I11_accuracy": rate([r for r in prim if r["condition"] == "I11"], correct),
        "isolated_accuracy": rate([r for r in prim if r["condition"] == "iso"], correct),
        "visible_target_accuracy_single_hidden": rate([r for r in prim if r["condition"] in ("I01", "I10") and r["gold"]["target_visible"]], correct),
        "hidden_target_responses": dict(Counter(response_class(r) for r in hidden)),
        "hidden_target_UNKNOWN": rate(hidden, lambda r: r["valid"] and r["label"] == "UNKNOWN"),
        "valid_output_rate": rate(rows, lambda r: r["valid"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--evidence-level", required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--render-records", type=Path, default=None,
                    help="folder of render_record JSON files (default: <work_root>/rendered/<split>/*/)")
    a = ap.parse_args()
    P = get_paths()
    out = a.output_dir or P.results / f"{a.split}_cohorts"
    rows = read_jsonl(a.run / "parsed_predictions.jsonl")
    flags = {}
    recs = sorted(a.render_records.glob("*.json")) if a.render_records else sorted((P.rendered / a.split).glob("*/render_record.json"))
    for rec in recs:
        r = read_json(rec)
        size = {c["name"][-1]: ast.literal_eval(c["detail"]) if isinstance(c["detail"], str) else c["detail"]
                for c in r["checks"] if c["name"].startswith("garment_size_")}
        # A scene enters a cohort only if both garments pass that size policy AND every other render check passed.
        flags[r["scene_id"]] = {k: r["status"] == "PASS" and all(size[p]["cohorts"][k] for p in "AB") for k in COHORTS} | {
            "size_only": {k: all(size[p]["cohorts"][k] for p in "AB") for k in COHORTS}, "render_status": r["status"],
            "measurements": {p: {x: size[p][x] for x in ("width", "height", "area")} for p in "AB"}}
    summary = {"evidence_level": a.evidence_level, "annotation_status": "exploratory_single_rater",
               "amendment": "A02 / D-021", "note": "EXPLORATORY: single rater, pilot gates not met",
               "scene_flags": flags, "cohorts": {}}
    for k in COHORTS:
        keep = {s for s, f in flags.items() if f[k]}
        sub = [r for r in rows if r["scene_id"] in keep]
        d = out / k
        d.mkdir(parents=True, exist_ok=True)
        summary["cohorts"][k] = summarize(sub) | {"scene_ids": sorted(keep)}
        if not sub:
            continue
        write_jsonl(d / "parsed_predictions.jsonl", sub)
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "analyze.py"), "--parsed", str(d / "parsed_predictions.jsonl"),
                        "--evidence-level", a.evidence_level, "--output-dir", str(d), "--no-progress"],
                       check=True, capture_output=True)
        pc = read_json(d / "primary_contrasts.json")
        summary["cohorts"][k]["estimands"] = {m: {"point": pc[m]["point"], "ci95": pc[m]["ci"], "n_clusters": pc[m]["n_clusters"]}
                                             for m in ("delta_hidden", "delta_visible", "psi")}
        summary["cohorts"][k]["verdict"] = pc.get("verdict", {}).get("status")
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_figures.py"), "--results-dir", str(d),
                        "--output-dir", str(d / "figures"), "--no-progress"], check=True, capture_output=True)
    write_json(out / "cohort_summary.json", summary)
    print(f"wrote {out/'cohort_summary.json'}")


if __name__ == "__main__":
    main()
