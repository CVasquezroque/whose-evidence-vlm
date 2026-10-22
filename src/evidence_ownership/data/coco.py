"""Minimal COCO 2017 instances reader (person category only).

pycocotools is used only for its index; we keep our own small dataclasses so manifests stay
independent of library internals.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PERSON_CATEGORY_ID = 1
COCO_SPLITS = ("train2017", "val2017")


@dataclass(frozen=True)
class Person:
    ann_id: int
    bbox_xywh: tuple[float, float, float, float]
    area: float
    iscrowd: int


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    license_id: int
    coco_url: str
    flickr_url: str
    persons: tuple[Person, ...]


@dataclass(frozen=True)
class CocoIndex:
    split: str
    images: dict[int, ImageRecord]
    licenses: dict[int, dict]
    source_path: Path


def load_coco_persons(instances_json: str | Path, split: str) -> CocoIndex:
    """Load images and their person annotations (crowd annotations included, flagged)."""
    if split not in COCO_SPLITS:
        raise ValueError(f"unsupported split {split!r}")
    path = Path(instances_json)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    if cats.get(PERSON_CATEGORY_ID) != "person":
        raise ValueError(f"{path}: category 1 is {cats.get(PERSON_CATEGORY_ID)!r}, expected 'person'")
    persons: dict[int, list[Person]] = {}
    for a in data["annotations"]:
        if a["category_id"] != PERSON_CATEGORY_ID:
            continue
        persons.setdefault(a["image_id"], []).append(
            Person(int(a["id"]), tuple(float(v) for v in a["bbox"]), float(a["area"]), int(a["iscrowd"]))
        )
    images = {
        int(im["id"]): ImageRecord(
            image_id=int(im["id"]), file_name=im["file_name"], width=int(im["width"]),
            height=int(im["height"]), license_id=int(im["license"]), coco_url=im.get("coco_url", ""),
            flickr_url=im.get("flickr_url", ""),
            persons=tuple(sorted(persons.get(int(im["id"]), []), key=lambda p: p.ann_id)),
        )
        for im in data["images"]
    }
    licenses = {int(l["id"]): {"id": int(l["id"]), "name": l["name"], "url": l["url"]} for l in data["licenses"]}
    return CocoIndex(split=split, images=images, licenses=licenses, source_path=path)
