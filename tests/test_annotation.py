import json

from evidence_ownership.annotation.build_items import build_items
from evidence_ownership.annotation.schema import AUTHOR_CHECKS, completeness_errors, structural_errors

CAND = {"scene_id": "val2017-000000000007", "source_image_id": 7, "coco_split": "val2017", "study_split": "heldout",
        "width": 640, "height": 480, "order_rank": 0,
        "persons": {"A": {"coco_ann_id": 1, "bbox_xywh": [0, 0, 10, 10], "side": "left", "area": 1},
                    "B": {"coco_ann_id": 2, "bbox_xywh": [50, 0, 10, 10], "side": "right", "area": 1}}}
SQ = [[10, 10], [60, 10], [60, 60], [10, 60]]


def author(**over):
    r = {"schema": "eob.annotation.v1", "role": "author", "annotator_id": "T", "scene_id": CAND["scene_id"],
         "source_image_id": 7, "coco_split": "val2017", "decision": "eligible", "reject_reasons": [],
         "checks": {k: True for k in AUTHOR_CHECKS},
         "persons": {"A": {"color": "blue", "color_confidence": "certain", "garment_polygons_orig": [SQ]},
                     "B": {"color": "red", "color_confidence": "certain", "garment_polygons_orig": [SQ]}},
         "sham_translations_canvas": {"A": [100, 0], "B": [0, 100]}}
    r.update(over)
    return r


def rater(role, a="blue", b="red", decision="eligible"):
    return {"schema": "eob.annotation.v1", "role": role, "annotator_id": role, "scene_id": CAND["scene_id"],
            "decision": decision, "reject_reasons": [],
            "persons": {"A": {"color": a}, "B": {"color": b}}, "review": {}}


def test_structural_rejects_unknown_as_latent_color_and_bad_polygons():
    bad = author()
    bad["persons"]["A"]["color"] = "UNKNOWN"
    bad["persons"]["B"]["garment_polygons_orig"] = [[[1, 2], [3, 4]]]
    bad["sham_translations_canvas"]["A"] = [1.5, 2]
    errs = structural_errors(bad, CAND)
    assert any("UNKNOWN" in e for e in errs) and any(">= 3" in e for e in errs) and any("integers" in e for e in errs)
    assert structural_errors(author(), CAND) == []


def test_structural_rejects_scene_mismatch_and_out_of_image_vertex():
    r = author(source_image_id=8)
    r["persons"]["A"]["garment_polygons_orig"] = [[[0, 0], [700, 0], [0, 10]]]
    errs = structural_errors(r, CAND)
    assert any("mismatch" in e for e in errs) and any("outside image" in e for e in errs)


def test_completeness():
    assert completeness_errors(author()) == []
    r = author(decision="unsure")
    r["checks"]["colors_differ"] = False
    r["persons"]["B"]["color"] = "blue"
    errs = completeness_errors(r)
    assert any("decision" in e for e in errs) and any("colors_differ" in e for e in errs) and any("same color" in e for e in errs)


def _write(tmp, role, rec):
    d = tmp / role
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{CAND['scene_id']}.json").write_text(json.dumps(rec))


def _build(tmp, mode):
    return build_items([CAND], {CAND["scene_id"]: {"sha256": "x"}}, tmp, mode=mode, target=80, clusters={},
                       robustness_salt="s:", robustness_n=20)


def test_exploratory_vs_confirmatory(tmp_path):
    _write(tmp_path, "author", author())
    items, excl, flow = _build(tmp_path, "exploratory")
    assert len(items) == 1 and items[0]["annotation_status"] == "exploratory_single_rater"
    assert items[0]["persons"]["A"]["latent_color"] == "blue" and items[0]["robustness_subset"]
    items, _, flow = _build(tmp_path, "confirmatory")
    assert items == [] and flow["awaiting_raters"] == 1


def test_two_raters_agree_and_disagree(tmp_path):
    _write(tmp_path, "author", author())
    _write(tmp_path, "rater1", rater("rater1"))
    _write(tmp_path, "rater2", rater("rater2"))
    items, _, _ = _build(tmp_path, "confirmatory")
    assert items[0]["annotation_status"] == "two_rater_agreed"
    _write(tmp_path, "rater2", rater("rater2", a="purple"))
    items, excl, flow = _build(tmp_path, "confirmatory")
    assert items == [] and excl[0]["stage"] == "rater_agreement" and flow["rater_disagreement"] == 1


def test_rater_rejection_excludes_even_in_exploratory(tmp_path):
    _write(tmp_path, "author", author())
    _write(tmp_path, "rater1", rater("rater1", decision="reject"))
    items, excl, _ = _build(tmp_path, "exploratory")
    assert items == [] and excl[0]["stage"] == "rater_review"


def test_author_rejection_logged(tmp_path):
    _write(tmp_path, "author", author(decision="reject", reject_reasons=["extra_person"]))
    items, excl, flow = _build(tmp_path, "exploratory")
    assert items == [] and excl[0]["reasons"] == ["extra_person"] and flow["author_not_eligible"] == 1
