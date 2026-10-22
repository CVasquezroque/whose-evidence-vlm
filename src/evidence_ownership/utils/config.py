"""Configuration loading: protocol constants, model/inference settings and machine paths.

Machine paths come from ``$EOB_PATHS`` or ``<repo>/configs/paths.yaml``. Relative paths inside
that file resolve against the file's own directory, so the repository never hardcodes a home dir.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=None)
def experiment_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "experiment.yaml")


@lru_cache(maxsize=None)
def model_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "model.yaml")


@lru_cache(maxsize=None)
def inference_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "inference.yaml")


@dataclass(frozen=True)
class Paths:
    coco_annotations: Path
    coco_val_images: Path
    coco_train_images: Path
    coco_image_base_url: str
    model_dir: Path
    work_root: Path

    # Derived locations inside work_root (all outside the Git repository).
    @property
    def manifests(self) -> Path:
        return self.work_root / "manifests"

    @property
    def annotations(self) -> Path:
        return self.work_root / "annotations"

    @property
    def rendered(self) -> Path:
        return self.work_root / "rendered"

    @property
    def runs(self) -> Path:
        return self.work_root / "runs"

    @property
    def results(self) -> Path:
        return self.work_root / "results"

    @property
    def figures(self) -> Path:
        return self.work_root / "figures"

    @property
    def galleries(self) -> Path:
        return self.work_root / "galleries"

    @property
    def release(self) -> Path:
        return self.work_root / "release"

    def images_dir(self, coco_split: str) -> Path:
        if coco_split == "val2017":
            return self.coco_val_images
        if coco_split == "train2017":
            return self.coco_train_images
        raise ValueError(f"unknown COCO split {coco_split!r}")

    def instances_json(self, coco_split: str) -> Path:
        return self.coco_annotations / f"instances_{coco_split}.json"


def _resolve(base: Path, value: str) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(value)))
    return p if p.is_absolute() else (base / p).resolve()


@lru_cache(maxsize=None)
def get_paths(path_file: str | None = None) -> Paths:
    src = Path(path_file or os.environ.get("EOB_PATHS") or CONFIG_DIR / "paths.yaml")
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Copy configs/paths.example.yaml to configs/paths.yaml "
            "or set EOB_PATHS to a YAML file with the same keys."
        )
    cfg = load_yaml(src)
    base = src.resolve().parent
    return Paths(
        coco_annotations=_resolve(base, cfg["coco"]["annotations"]),
        coco_val_images=_resolve(base, cfg["coco"]["val_images"]),
        coco_train_images=_resolve(base, cfg["coco"]["train_images"]),
        coco_image_base_url=cfg["coco"].get("image_base_url", "https://s3.amazonaws.com/images.cocodataset.org"),
        model_dir=_resolve(base, cfg["model"]["dir"]),
        work_root=_resolve(base, cfg["work_root"]),
    )
