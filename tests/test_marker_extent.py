"""D-017: garments reaching the tight person box must not collide with the marker outline."""
import copy

from evidence_ownership.annotation.sham import propose_shams
from evidence_ownership.rendering.factorial import render_scene
from evidence_ownership.rendering.synthetic import make_synthetic_scene


def test_garment_touching_box_edge_renders_pass(tmp_path):
    image, item = make_synthetic_scene()
    item = copy.deepcopy(item)
    x, y, w, h = item["persons"]["A"]["bbox_xywh"]
    # sleeve polygon running exactly along the left and right box edges
    item["persons"]["A"]["garment_polygons_orig"] = [[[x, 140], [x + w, 140], [x + w, 250], [x, 250]]]
    item["sham_translations_canvas"] = propose_shams(item)
    src = tmp_path / "src.png"
    image.save(src)
    item["image_sha256"] = None
    rec = render_scene(item, src, tmp_path / "out")
    failed = [c["name"] for c in rec["checks"] if c["status"] == "FAIL"]
    assert rec["status"] == "PASS", failed
    gA, bA = rec["marker_geometry"]["A"]["box"], item["persons_canvas"]["A"]["bbox_xyxy"]
    assert gA[0] < bA[0] - 4 and gA[2] > bA[2] + 4  # outline moved outside the dilated garment


def test_polygons_saved_under_wrong_person_fail(tmp_path):
    image, item = make_synthetic_scene()
    item = copy.deepcopy(item)
    pa, pb = item["persons"]["A"]["garment_polygons_orig"], item["persons"]["B"]["garment_polygons_orig"]
    item["persons"]["A"]["garment_polygons_orig"], item["persons"]["B"]["garment_polygons_orig"] = pb, pa
    item["sham_translations_canvas"] = propose_shams(item)
    src = tmp_path / "src.png"
    image.save(src)
    rec = render_scene(item, src, tmp_path / "out")
    failed = {c["name"] for c in rec["checks"] if c["status"] == "FAIL"}
    assert {"garment_inside_own_box_A", "garment_inside_own_box_B"} <= failed
