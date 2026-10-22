import numpy as np
from PIL import Image, ImageDraw

from evidence_ownership.annotation.sam_assist import mask_to_polygons
from evidence_ownership.rendering.masks import rasterize_polygons

IDENTITY = {"scale": 1.0, "offset": [0, 0], "resized_size": [1024, 1024], "canvas_size": [1024, 1024], "orig_size": [1024, 1024]}


def blob():
    im = Image.new("L", (1024, 1024), 0)
    d = ImageDraw.Draw(im)
    d.ellipse([300, 200, 520, 480], fill=255)          # torso
    d.rectangle([250, 220, 300, 400], fill=255)        # sleeve touching the torso
    d.rectangle([600, 300, 640, 360], fill=255)        # separate fragment
    d.ellipse([380, 300, 420, 340], fill=0)            # hole (e.g. gap between arm and body)
    d.point((900, 900), fill=255)                      # speck, must be dropped
    return np.asarray(im) > 0


def test_polygons_reproduce_mask_and_fill_holes():
    m = blob()
    polys = mask_to_polygons(m)
    assert len(polys) == 2 and all(len(p) >= 3 for p in polys)
    r = rasterize_polygons(polys, IDENTITY)
    filled = m.copy()
    filled[300:341, 380:421] |= np.asarray(Image.new("L", (1, 1)))[0, 0] == 0  # hole region becomes covered
    missed = m & ~r
    assert missed[:800, :800].sum() / m[:800, :800].sum() < 0.005   # virtually no garment pixel left uncovered
    extra = r & ~filled
    assert extra.sum() / m.sum() < 0.03                               # tight boundary
    assert not r[890:910, 890:910].any()                              # speck ignored


def test_empty_mask():
    assert mask_to_polygons(np.zeros((64, 64), bool)) == []
