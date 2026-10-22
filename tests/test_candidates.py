import hashlib

import pytest

from evidence_ownership.data.candidates import (ab_bit, bbox_iou_xywh, bbox_to_canvas, build_candidates,
                                                canvas_transform, screen_image, size_prescreen)
from evidence_ownership.data.coco import CocoIndex, ImageRecord, Person
from evidence_ownership.utils.config import experiment_config
from evidence_ownership.utils.hashing import order_key

CFG = experiment_config()


def img(iid, persons, w=640, h=480):
    return ImageRecord(iid, f"{iid:012d}.jpg", w, h, 1, "", "", tuple(persons))


def P(aid, bbox, crowd=0):
    return Person(aid, bbox, bbox[2] * bbox[3], crowd)


def test_order_key_matches_astra_definition():
    assert order_key("fatima-evidence-2026-v1:", 42) == hashlib.sha256(b"fatima-evidence-2026-v1:42").hexdigest()


def test_iou():
    assert bbox_iou_xywh([0, 0, 10, 10], [20, 20, 5, 5]) == 0
    assert bbox_iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert bbox_iou_xywh([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(50 / 150)


@pytest.mark.parametrize("persons,reason", [
    ([], "no_person"),
    ([P(1, (0, 0, 10, 10)), P(2, (50, 0, 10, 10), crowd=1)], "crowd_person_annotation"),
    ([P(1, (0, 0, 10, 10))], "not_exactly_two_persons"),
    ([P(1, (0, 0, 10, 10)), P(2, (100, 0, 10, 10)), P(3, (200, 0, 10, 10))], "not_exactly_two_persons"),
    ([P(1, (0, 0, 10, 10)), P(2, (1, 0, 10, 10))], "person_bbox_iou_too_high"),
    ([P(1, (0, 0, 10, 10)), P(2, (100, 0, 10, 10))], "pass"),
])
def test_screen(persons, reason):
    assert screen_image(img(1, persons), 0.05).reason == reason


def test_iou_threshold_is_exclusive():
    # IoU exactly 0.05 must fail (protocol: IoU < 0.05)
    a, b = (0, 0, 100, 10), (0, 0, 5, 10)  # inter 50, union 1000 -> 0.05
    assert bbox_iou_xywh(a, b) == pytest.approx(0.05)
    assert screen_image(img(1, [P(1, a), P(2, b)]), 0.05).reason == "person_bbox_iou_too_high"


@pytest.mark.parametrize("w,h", [(640, 480), (1024, 1024), (2048, 1000), (333, 1500), (1025, 1024)])
def test_canvas_transform_no_upscale_and_fits(w, h):
    t = canvas_transform(w, h)
    assert t["scale"] <= 1.0
    rw, rh = t["resized_size"]
    assert rw <= 1024 and rh <= 1024 and (rw == 1024 or rh == 1024 or (rw, rh) == (w, h))
    assert t["offset"] == [(1024 - rw) // 2, (1024 - rh) // 2]


def test_canvas_transform_agrees_with_renderer():
    from evidence_ownership.rendering import canvas as rc
    for w, h in [(640, 480), (2048, 1000), (333, 1500), (1025, 1024)]:
        ours = canvas_transform(w, h)
        theirs = rc.compute_canvas_transform(w, h)
        theirs = theirs if isinstance(theirs, dict) else theirs.to_dict()
        for k in ("scale", "offset", "resized_size", "canvas_size", "orig_size"):
            assert list(ours[k]) == list(theirs[k]) if isinstance(ours[k], list) else ours[k] == theirs[k], k


def test_ab_assignment_balanced_and_deterministic():
    bits = [ab_bit(CFG["ab_assignment_salt"], i) for i in range(2000)]
    assert bits == [ab_bit(CFG["ab_assignment_salt"], i) for i in range(2000)]
    assert 0.45 < sum(bits) / len(bits) < 0.55


def test_build_candidates_sorted_and_ab_sides():
    ims = {i: img(i, [P(10 * i, (300, 0, 50, 200)), P(10 * i + 1, (10, 0, 50, 200))]) for i in range(1, 40)}
    idx = CocoIndex("val2017", ims, {1: {"id": 1, "name": "x", "url": "u"}}, None)
    cands, stats = build_candidates(idx, CFG)
    assert stats["pass"] == 39 and [c["order_rank"] for c in cands] == list(range(39))
    assert [c["order_key"] for c in cands] == sorted(c["order_key"] for c in cands)
    for c in cands:
        a_side = "left" if c["ab_bit"] == 0 else "right"
        assert c["persons"]["A"]["side"] == a_side and c["persons"]["B"]["side"] != a_side
        left_id = 10 * c["source_image_id"] + 1  # bbox at x=10 is the left person
        assert (c["persons"]["A"]["coco_ann_id"] == left_id) == (a_side == "left")
        assert c["study_split"] == "heldout"


def test_size_prescreen_and_canvas_bbox():
    t = canvas_transform(2048, 1024)  # scale 0.5
    assert bbox_to_canvas((100, 100, 96, 96), t) == [50.0, 50.0 + t["offset"][1], 98.0, 98.0 + t["offset"][1]]
    pc = {"A": {"bbox_xyxy": [0, 0, 48, 48]}, "B": {"bbox_xyxy": [0, 0, 47.9, 100]}}
    assert not size_prescreen(pc, 48, 48)
    pc["B"]["bbox_xyxy"] = [0, 0, 48, 100]
    assert size_prescreen(pc, 48, 48)
