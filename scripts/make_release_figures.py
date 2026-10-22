#!/usr/bin/env python3
"""Regenerate the public figures and example images from real outputs.

Inputs: the committed results/ (cohort summary, pilot predictions, failed-QC diagnostic) plus
rendered PNGs in <work_root>/rendered (rebuild them with scripts/render_factorial.py). Outputs:
  figures/factorial_example_149314.png   real pilot scene, five inputs and the model's answers
  figures/pilot_contrasts_by_cohort.png  delta_hidden, delta_visible, psi with 95% CIs per cohort
  figures/pilot_hidden_target_responses.png  what the model answered when the target was hidden
  figures/heldout_372819_failed_qc.png   failed intervention QC (excluded from analysis)
  examples/interventions/<scene>/*.png and examples/predictions/<scene>.json

Example:
  python -u scripts/make_release_figures.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from evidence_ownership.utils.config import REPO_ROOT, get_paths  # noqa: E402
from evidence_ownership.utils.io import read_json, read_jsonl  # noqa: E402

EXAMPLE = "train2017-000000149314"
EXAMPLE_CREDIT = ("Photo: COCO 2017 train 149314 (Flickr 6576248219), CC BY-SA 2.0; "
                  "modified (gray evidence masks, shams, A/B markers). Derivative shared under CC BY-SA 2.0.")
FAILED = "val2017-000000372819"
FAILED_CREDIT = "Photo: COCO 2017 val 372819 (Flickr 2516944023), CC BY 2.0; modified."
COHORTS = ("original", "moderate", "expanded")
COHORT_LABEL = {"original": "original\n48×48 / 2304", "moderate": "moderate\n40×40 / 1500", "expanded": "expanded\n32×32 / 900"}
RESP = [("UNKNOWN", "correct UNKNOWN", "#009E73"), ("gray_occluder", "gray (the mask)", "#999999"),
        ("donor_color", "other person's colour", "#D55E00"), ("target_own_color", "hidden garment's own colour", "#0072B2"),
        ("other_color", "other colour", "#CC79A7"), ("invalid", "invalid JSON", "#E69F00")]


def factorial_example(out: Path, rendered: Path, preds: list[dict]) -> None:
    keys = ["clean", "I11", "I01", "I10", "I00"]
    titles = {"clean": "clean (marked)", "I11": "I11  both visible + 2 shams", "I01": "I01  A hidden",
              "I10": "I10  B hidden", "I00": "I00  both hidden"}
    ans = {(r["image_key"], r["target_person"]): (r["label"] if r["valid"] else "invalid")
           for r in preds if r["scene_id"] == EXAMPLE and r["variant"] == "primary"}
    gold = {(r["image_key"], r["target_person"]): r["gold"]["task_gold"] for r in preds if r["scene_id"] == EXAMPLE}
    rec = read_json(rendered / "render_record.json")
    b = rec["garment_bbox_canvas"]
    pad = 70
    box = (max(0, min(b["A"][0], b["B"][0]) - pad), max(0, min(b["A"][1], b["B"][1]) - 2 * pad),
           min(1024, max(b["A"][2], b["B"][2]) + pad), min(1024, max(b["A"][3], b["B"][3]) + 2 * pad))
    fig, axes = plt.subplots(1, 5, figsize=(17, 5.2))
    for ax, k in zip(axes, keys):
        ax.imshow(Image.open(rendered / f"{k}.png").crop(box))
        ax.set_title(titles[k], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        lines = [f"{p}: {ans.get((k, p), '-')}  (gold {gold.get((k, p), '-')})" for p in "AB"]
        ax.set_xlabel("\n".join(lines), fontsize=9.5)
    fig.suptitle("Real pilot scene (exploratory, single annotator): Qwen3.5-4B answers per condition", fontsize=12)
    fig.text(0.5, 0.01, EXAMPLE_CREDIT, ha="center", fontsize=7.5, color="#444")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def contrasts(out: Path, summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    metrics = [("delta_hidden", "δ_hidden", "#D55E00"), ("delta_visible", "δ_visible", "#0072B2"), ("psi", "ψ", "#000000")]
    for i, k in enumerate(COHORTS):
        est = summary["cohorts"][k]["estimands"]
        for j, (m, lab, col) in enumerate(metrics):
            x = i + (j - 1) * 0.22
            p, (lo, hi) = est[m]["point"], est[m]["ci95"]
            ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="o", color=col, capsize=4, label=lab if i == 0 else None)
    ax.axhline(0, color="#888", lw=0.8)
    ax.axhline(0.10, color="#888", lw=0.8, ls=":")
    ax.text(2.45, 0.105, "0.10 practical threshold", fontsize=8, color="#666", ha="right")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{COHORT_LABEL[k]}\nn = {summary['cohorts'][k]['n_scenes']} scenes" for k in COHORTS], fontsize=9)
    ax.set_ylabel("contrast (donor-colour rate)")
    ax.set_title("Pilot contrasts with 95% cluster-bootstrap CIs (nested cohorts, same predictions; exploratory)", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def hidden_responses(out: Path, summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, k in enumerate(COHORTS):
        c = summary["cohorts"][k]["hidden_target_responses"]
        n = sum(c.values())
        left = 0
        for key, lab, col in RESP:
            v = c.get(key, 0)
            if v:
                ax.barh(i, v / n, left=left, color=col, edgecolor="white", label=lab if lab not in ax.get_legend_handles_labels()[1] else None)
                ax.text(left + v / n / 2, i, str(v), ha="center", va="center", fontsize=8, color="white")
            left += v / n
        ax.text(1.01, i, f"n = {n}", va="center", fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels([COHORT_LABEL[k] for k in COHORTS], fontsize=9)
    ax.set_xlim(0, 1); ax.set_xlabel("share of hidden-target queries (primary factorial)")
    ax.set_title("Answers when the target's garment was masked (pilot, exploratory)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def failed_qc(out: Path, rendered: Path) -> None:
    rec = read_json(rendered / "render_record.json")
    b = rec["garment_bbox_canvas"]
    box = (min(b["A"][0], b["B"][0]) - 60, min(b["A"][1], b["B"][1]) - 60, max(b["A"][2], b["B"][2]) + 60, max(b["A"][3], b["B"][3]) + 60)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, k, t in zip(axes, ("clean", "I00"), ("clean (marked)", "I00: both garments 'hidden'")):
        ax.imshow(Image.open(rendered / f"{k}.png").crop(box)); ax.set_title(t, fontsize=10); ax.axis("off")
    fig.suptitle("Failed intervention QC — residual garment evidence; excluded from scientific analysis",
                 fontsize=12, color="#B00020", fontweight="bold")
    fig.text(0.5, 0.02, "Visible below/beside the masks: A's jacket hem and an orange part of B's open jacket. " + FAILED_CREDIT,
             ha="center", fontsize=7.5, color="#444")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    a = ap.parse_args()
    P = get_paths()
    figs = REPO_ROOT / "figures"
    figs.mkdir(exist_ok=True)
    summary = read_json(a.results / "pilot" / "cohort_summary.json")
    preds = read_jsonl(a.results / "pilot" / "predictions" / "parsed_predictions.jsonl")
    ex = P.rendered / "pilot" / EXAMPLE
    factorial_example(figs / "factorial_example_149314.png", ex, preds)
    contrasts(figs / "pilot_contrasts_by_cohort.png", summary)
    hidden_responses(figs / "pilot_hidden_target_responses.png", summary)
    failed_qc(figs / "heldout_372819_failed_qc.png", P.rendered / "heldout" / FAILED)
    dst = REPO_ROOT / "examples" / "interventions" / EXAMPLE
    dst.mkdir(parents=True, exist_ok=True)
    for k in ("clean", "I11", "I01", "I10", "I00", "iso_A", "iso_B"):
        shutil.copy2(ex / f"{k}.png", dst / f"{k}.png")
    (dst / "ATTRIBUTION.txt").write_text(EXAMPLE_CREDIT + "\nOriginal: https://www.flickr.com/photo.gne?id=6576248219\n")
    pdir = REPO_ROOT / "examples" / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    keep = ("query_id", "variant", "image_key", "target_person", "target_tag", "gold", "raw_text", "label", "valid", "invalid_reason", "R")
    rows = [{k: r[k] for k in keep} for r in preds if r["scene_id"] == EXAMPLE]
    (pdir / f"{EXAMPLE}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    print("figures and examples written")


if __name__ == "__main__":
    main()
