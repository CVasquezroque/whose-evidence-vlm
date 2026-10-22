"""Turn human annotation records into renderer items (data_contracts.md §4).

Modes:
  confirmatory  needs an eligible author record AND two finalized independent rater reviews that
                both pass and agree on both colours. Latent colours come from the raters.
  exploratory   also accepts scenes that have only the author record (status
                ``exploratory_single_rater``). Results built on these items must be labelled exploratory.

Scenes are taken in protocol order until ``target`` eligible clusters are selected (a stopping rule
on eligible scenes, never on model outputs). Every candidate that was examined and not selected is
written to the exclusion log with its stage and reasons.
"""
from __future__ import annotations

from pathlib import Path

from ..utils.hashing import sha256_file, sha256_text
from ..utils.io import read_json
from .schema import GARMENT_COLORS, completeness_errors

MODES = ("exploratory", "confirmatory")


def _load(path: Path) -> tuple[dict | None, str | None]:
    return (read_json(path), sha256_file(path)) if path.exists() else (None, None)


def rater_verdict(rec: dict | None) -> tuple[str, list[str]]:
    """('pass'|'fail'|'missing'|'incomplete', reasons)."""
    if rec is None:
        return "missing", ["no rater record"]
    if rec.get("decision") == "draft":
        return "incomplete", ["review not finalized"]
    if rec.get("decision") != "eligible":
        return "fail", [f"rater decision {rec.get('decision')}"] + list(rec.get("reject_reasons", []))
    for p in ("A", "B"):
        if rec["persons"][p].get("color") not in GARMENT_COLORS:
            return "fail", [f"rater colour {p} missing"]
    return "pass", []


def build_items(cands: list[dict], images: dict[str, dict], ann_dir: Path, *, mode: str, target: int | None,
                clusters: dict[str, str], robustness_salt: str, robustness_n: int) -> tuple[list[dict], list[dict], dict]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    items, excl = [], []
    used_clusters: set[str] = set()
    flow = {"examined": 0, "no_author_record": 0, "author_not_eligible": 0, "rater_fail": 0,
            "rater_disagreement": 0, "awaiting_raters": 0, "near_duplicate": 0, "no_verified_image": 0,
            "selected": 0, "not_examined_after_target": 0}
    for c in cands:
        sid = c["scene_id"]
        if target is not None and len(items) >= target:
            flow["not_examined_after_target"] += 1
            continue
        author, h_author = _load(ann_dir / "author" / f"{sid}.json")
        if author is None:
            flow["no_author_record"] += 1
            continue  # not yet examined by a human: neither selected nor excluded
        flow["examined"] += 1
        base = {"scene_id": sid, "order_rank": c["order_rank"]}
        errs = completeness_errors(author)
        if errs:
            flow["author_not_eligible"] += 1
            excl.append({**base, "stage": "author_screen", "decision": author.get("decision"),
                         "reasons": list(author.get("reject_reasons", [])) or errs, "notes": author.get("notes", "")})
            continue
        if sid not in images:
            flow["no_verified_image"] += 1
            excl.append({**base, "stage": "source_image", "reasons": ["image not verified"]})
            continue
        r1, h1 = _load(ann_dir / "rater1" / f"{sid}.json")
        r2, h2 = _load(ann_dir / "rater2" / f"{sid}.json")
        v1, why1 = rater_verdict(r1)
        v2, why2 = rater_verdict(r2)
        colors = {p: author["persons"][p]["color"] for p in ("A", "B")}
        if v1 == "pass" and v2 == "pass":
            rc1 = {p: r1["persons"][p]["color"] for p in ("A", "B")}
            rc2 = {p: r2["persons"][p]["color"] for p in ("A", "B")}
            if rc1 != rc2 or rc1["A"] == rc1["B"]:
                flow["rater_disagreement"] += 1
                excl.append({**base, "stage": "rater_agreement", "reasons": ["rater colour disagreement"],
                             "rater1": rc1, "rater2": rc2, "author": colors})
                continue
            status, colors = "two_rater_agreed", rc1
        elif "fail" in (v1, v2):
            flow["rater_fail"] += 1
            excl.append({**base, "stage": "rater_review", "reasons": why1 + why2})
            continue
        elif mode == "exploratory":
            status = "exploratory_single_rater"
        else:
            flow["awaiting_raters"] += 1
            continue  # confirmatory: wait for both reviews
        cluster = clusters.get(sid, sid)
        if cluster in used_clusters:
            flow["near_duplicate"] += 1
            excl.append({**base, "stage": "cluster", "reasons": [f"near-duplicate of selected cluster {cluster}"]})
            continue
        used_clusters.add(cluster)
        item = {k: v for k, v in c.items()}
        item["image_sha256"] = images[sid]["sha256"]
        item["persons"] = {p: {**c["persons"][p], "latent_color": colors[p],
                               "garment_polygons_orig": author["persons"][p]["garment_polygons_orig"]}
                           for p in ("A", "B")}
        item["sham_translations_canvas"] = {p: list(author["sham_translations_canvas"][p]) for p in ("A", "B")}
        item["cluster_id"] = cluster
        item["robustness_subset"] = False
        item["annotation_status"] = status
        item["annotation_provenance"] = {"author": h_author, "rater1": h1, "rater2": h2,
                                         "author_id": author.get("annotator_id"),
                                         "rater_was_mask_author": [x.get("rater_was_mask_author") for x in (r1, r2) if x]}
        items.append(item)
        flow["selected"] += 1
    # Robustness subset: chosen by image-ID hash among selected items, never by model behaviour.
    ranked = sorted(items, key=lambda it: sha256_text(f"{robustness_salt}{it['source_image_id']}"))
    for it in ranked[:robustness_n]:
        it["robustness_subset"] = True
    return items, excl, flow
