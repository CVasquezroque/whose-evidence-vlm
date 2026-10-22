"""Annotation record schema (data_contracts.md §3) and structural validation.

Structural validation runs on every save: it rejects malformed data, but drafts may be
incomplete. Completeness and eligibility are checked later by ``completeness_errors``,
when items are built.
"""
from __future__ import annotations

from typing import Any

SCHEMA = "eob.annotation.v1"
ROLES = ("author", "rater1", "rater2")
DECISIONS = ("eligible", "reject", "unsure", "draft")
CONFIDENCE = ("certain", "probable", "unsure")
GARMENT_COLORS = ("black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown")

AUTHOR_CHECKS = {
    "exactly_two_people": "Exactly two actual people are visible: no extra or unannotated people, mirrors, posters or reflections",
    "boxes_correct": "Each COCO box encloses one of the two people",
    "single_outer_garment_A": "A: one unambiguous outermost upper-body garment over the torso (not bare, transparent, or an open jacket with competing layers)",
    "single_outer_garment_B": "B: one unambiguous outermost upper-body garment over the torso",
    "dominant_color_A": "A: one dominant color on about 80% of the visible fabric, with no strong gradient or ambiguous lighting",
    "dominant_color_B": "B: one dominant color on about 80% of the visible fabric",
    "colors_differ": "A and B garment colors are different vocabulary colors",
    "fully_coverable": "Masks cover ALL visible fabric of each garment (sleeves, fragments) without covering the other person's garment",
    "designation_clear": "A/B designation stays clear with both garments masked (heads or upper bodies identifiable)",
    "sham_background_ok": "Shams lie on background: not on a person, text, mirror, poster or other informative content",
}

REJECT_REASONS = (
    "extra_person", "not_two_actual_people", "box_error", "no_clear_garment", "ambiguous_color",
    "same_color", "garment_too_small", "cannot_mask_cleanly", "no_sham_location", "designation_unclear",
    "near_duplicate", "other",
)

# Staged blinded review (RESEARCH_DECISION_REPORT.md V.4). Rater 1 sees I01 in stage 2 and
# rater 2 sees I10, so the single-hidden assignment is reversed between reviewers.
REVIEW_STAGES = {
    "stage1_I00": ["fabric_visible_A", "fabric_visible_B", "designation_clear"],
    "stage2_single": ["hidden_fabric_visible", "visible_garment_preserved", "designation_clear"],
    "stage3_colors": ["color_A", "color_B", "confidence_A", "confidence_B"],
    "stage4_sham": ["sham_preserves_colors", "sham_covers_informative_content"],
}
STAGE2_IMAGE = {"rater1": "I01", "rater2": "I10"}


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def structural_errors(rec: dict, cand: dict | None = None) -> list[str]:
    err: list[str] = []
    if rec.get("schema") != SCHEMA:
        err.append(f"schema must be {SCHEMA}")
    if rec.get("role") not in ROLES:
        err.append(f"role must be one of {ROLES}")
    if cand is not None:
        for k in ("scene_id", "source_image_id", "coco_split"):
            if rec.get(k) != cand.get(k):
                err.append(f"{k} mismatch: {rec.get(k)!r} != candidate {cand.get(k)!r}")
    if rec.get("decision") not in DECISIONS:
        err.append(f"decision must be one of {DECISIONS}")
    for r in rec.get("reject_reasons", []):
        if r not in REJECT_REASONS:
            err.append(f"unknown reject reason {r!r}")
    W = cand["width"] if cand else None
    H = cand["height"] if cand else None
    for p in ("A", "B"):
        pr = rec.get("persons", {}).get(p, {})
        c = pr.get("color")
        if c not in (None, "", *GARMENT_COLORS):
            err.append(f"{p}.color {c!r} not in garment vocabulary (UNKNOWN is never a latent color)")
        if pr.get("color_confidence") not in (None, "", *CONFIDENCE):
            err.append(f"{p}.color_confidence invalid")
        for i, poly in enumerate(pr.get("garment_polygons_orig", [])):
            if not isinstance(poly, list) or len(poly) < 3:
                err.append(f"{p}.polygon[{i}] needs >= 3 vertices")
                continue
            for v in poly:
                if not (isinstance(v, list) and len(v) == 2 and all(_is_num(x) for x in v)):
                    err.append(f"{p}.polygon[{i}] has a malformed vertex {v!r}")
                    break
                if W is not None and not (-0.5 <= v[0] <= W + 0.5 and -0.5 <= v[1] <= H + 0.5):
                    err.append(f"{p}.polygon[{i}] vertex {v} outside image {W}x{H}")
                    break
    for p, t in (rec.get("sham_translations_canvas") or {}).items():
        if p not in ("A", "B") or t is None:
            continue
        if not (isinstance(t, list) and len(t) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in t)):
            err.append(f"sham translation {p} must be two integers, got {t!r}")
    return err


def completeness_errors(rec: dict) -> list[str]:
    """Why an author record cannot (yet) produce an eligible item."""
    err: list[str] = []
    if rec.get("decision") != "eligible":
        err.append(f"decision is {rec.get('decision')!r}")
    for k in AUTHOR_CHECKS:
        if rec.get("checks", {}).get(k) is not True:
            err.append(f"check not confirmed: {k}")
    for p in ("A", "B"):
        pr = rec.get("persons", {}).get(p, {})
        if pr.get("color") not in GARMENT_COLORS:
            err.append(f"{p}: no color")
        if not pr.get("garment_polygons_orig"):
            err.append(f"{p}: no garment polygon")
        if (rec.get("sham_translations_canvas") or {}).get(p) is None:
            err.append(f"{p}: no sham translation")
    cA = rec.get("persons", {}).get("A", {}).get("color")
    cB = rec.get("persons", {}).get("B", {}).get("color")
    if cA and cA == cB:
        err.append("A and B have the same color")
    return err
