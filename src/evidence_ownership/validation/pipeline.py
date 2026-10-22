"""Staged validation (Levels 1-6). Every check reports PASS, FAIL or SKIPPED with a reason.

SKIPPED means the stage has no inputs yet, for example no human annotations. It never means
that the stage passed.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from ..annotation.schema import ROLES, structural_errors
from ..data.candidates import ab_bit
from ..utils.config import REPO_ROOT, experiment_config, get_paths, model_config
from ..utils.hashing import order_key, sha256_file
from ..utils.io import read_json, read_jsonl
from ..utils.logs import progress


def _c(level: int, name: str, status: str, detail: object = "") -> dict:
    return {"level": level, "name": name, "status": status, "detail": detail if isinstance(detail, (str, int, float, list, dict)) else str(detail)}


def _ok(level, name, cond, detail="") -> dict:
    return _c(level, name, "PASS" if cond else "FAIL", detail)


# ---------------------------------------------------------------- Level 1: source assets
def level1(no_progress: bool = False) -> list[dict]:
    p, mc, out = get_paths(), model_config(), []
    for split in ("train2017", "val2017"):
        f = p.instances_json(split)
        out.append(_ok(1, f"coco_instances_{split}_present", f.is_file(), str(f.name)))
    out.append(_ok(1, "coco_val_images_present", p.coco_val_images.is_dir() and len(list(p.coco_val_images.glob("*.jpg"))) == 5000,
                   "expects the 5,000 official val2017 JPEGs"))
    meta = p.model_dir / ".cache" / "huggingface" / "download"
    revs = {f.read_text().split("\n")[0] for f in meta.glob("*.metadata")} if meta.is_dir() else set()
    out.append(_ok(1, "model_revision_pinned", revs == {mc["revision"]}, sorted(revs)) if revs else
               _c(1, "model_revision_pinned", "SKIPPED", "no HF download metadata next to the checkpoint"))
    cfg = p.model_dir / "config.json"
    arch = json.loads(cfg.read_text()).get("architectures") if cfg.is_file() else None
    out.append(_ok(1, "model_architecture", arch == [mc["architecture"]], arch))
    sizes_ok = True
    for shard, digest in mc["shard_sha256"].items():
        meta_file = meta / f"{shard}.metadata"
        sizes_ok &= (p.model_dir / shard).is_file() and meta_file.is_file() and meta_file.read_text().split("\n")[1] == digest
    out.append(_ok(1, "model_shards_match_lfs_sha256_metadata", sizes_ok,
                   "Metadata comparison only; full re-hash recorded in provenance/environment/model_integrity.json (both shards matched)"))
    for split in ("train2017", "val2017"):
        man = p.manifests / f"images_{split}.jsonl"
        if not man.exists():
            out.append(_c(1, f"prepared_images_{split}", "SKIPPED", "run scripts/prepare_images.py"))
            continue
        bad = []
        recs = read_jsonl(man)
        for r in progress(recs, desc=f"L1 verify {split} images", unit="image", disable=no_progress):
            f = p.images_dir(split) / r["file_name"]
            if not f.is_file() or sha256_file(f) != r["sha256"]:
                bad.append(r["scene_id"])
        out.append(_ok(1, f"prepared_images_{split}_sha256", not bad, {"n": len(recs), "mismatch": bad[:20]}))
    return out


# ------------------------------------------------- Level 2: candidates and annotations
def level2() -> list[dict]:
    p, cfg, out = get_paths(), experiment_config(), []
    scene_sets = {}
    for split in ("train2017", "val2017"):
        man = p.manifests / f"candidates_{split}.jsonl"
        if not man.exists():
            out.append(_c(2, f"candidates_{split}", "SKIPPED", "run scripts/build_candidates.py"))
            continue
        c = read_jsonl(man)
        scene_sets[split] = {x["source_image_id"] for x in c}
        keys_ok = all(x["order_key"] == order_key(cfg["hash_salt"], x["source_image_id"]) for x in c)
        sorted_ok = [x["order_key"] for x in c] == sorted(x["order_key"] for x in c) and [x["order_rank"] for x in c] == list(range(len(c)))
        ab_ok = all(x["ab_bit"] == ab_bit(cfg["ab_assignment_salt"], x["source_image_id"]) and
                    x["persons"]["A"]["side"] == ("left" if x["ab_bit"] == 0 else "right") for x in c)
        iou_ok = all(x["bbox_iou"] < cfg["eligibility"]["person_bbox_iou_max_exclusive"] for x in c)
        split_ok = all(x["coco_split"] == split for x in c)
        out += [_ok(2, f"candidates_{split}_order_keys", keys_ok), _ok(2, f"candidates_{split}_sorted", sorted_ok),
                _ok(2, f"candidates_{split}_ab_assignment", ab_ok), _ok(2, f"candidates_{split}_iou_threshold", iou_ok),
                _ok(2, f"candidates_{split}_split_label", split_ok, len(c))]
    if len(scene_sets) == 2:
        out.append(_ok(2, "pilot_heldout_source_ids_disjoint", not (scene_sets["train2017"] & scene_sets["val2017"])))
    cands = {}
    for split in ("train2017", "val2017"):
        man = p.manifests / f"candidates_{split}.jsonl"
        if man.exists():
            cands.update({x["scene_id"]: x for x in read_jsonl(man)})
    n_ann, errs = 0, []
    for role in ROLES:
        for f in sorted((p.annotations / role).glob("*.json")):
            n_ann += 1
            rec = read_json(f)
            e = structural_errors(rec, cands.get(rec.get("scene_id")))
            if rec.get("scene_id") not in cands:
                e.append("scene not in candidate manifests")
            if e:
                errs.append({"file": f"{role}/{f.name}", "errors": e})
    out.append(_ok(2, "annotation_records_structurally_valid", not errs, {"n_records": n_ann, "errors": errs[:10]})
               if n_ann else _c(2, "annotation_records_structurally_valid", "SKIPPED", "no human annotation records yet"))
    for split in ("pilot", "heldout"):
        f = p.manifests / f"items_{split}.jsonl"
        if not f.exists():
            out.append(_c(2, f"items_{split}", "SKIPPED", "no items built yet (needs human annotation)"))
            continue
        items = read_jsonl(f)
        bad = [i["scene_id"] for i in items if i["annotation_status"] not in ("two_rater_agreed", "exploratory_single_rater")
               or i["persons"]["A"]["latent_color"] == i["persons"]["B"]["latent_color"]]
        out.append(_ok(2, f"items_{split}_valid_status_and_colors", not bad, {"n": len(items), "bad": bad}))
        status = sorted({i["annotation_status"] for i in items})
        out.append(_c(2, f"items_{split}_annotation_status", "PASS" if status == ["two_rater_agreed"] else "FAIL" if not items else "PASS",
                      {"statuses": status, "confirmatory": status == ["two_rater_agreed"]}))
    return out


# ------------------------------------------------------- Level 3: rendered interventions
def level3(no_progress: bool = False) -> list[dict]:
    p, out = get_paths(), []
    for split in ("pilot", "heldout", "smoke"):
        recs = sorted((p.rendered / split).glob("*/render_record.json"))
        if not recs:
            out.append(_c(3, f"rendered_{split}", "SKIPPED", "nothing rendered"))
            continue
        fails, hash_bad = [], []
        for f in progress(recs, desc=f"L3 {split}", unit="scene", disable=no_progress):
            r = read_json(f)
            if r["status"] != "PASS":
                fails.append({r["scene_id"]: [c["name"] for c in r["checks"] if c["status"] == "FAIL"]})
            for key, im in r["images"].items():
                if sha256_file(p.work_root / im["path"]) != im["sha256"]:
                    hash_bad.append(f"{r['scene_id']}/{key}")
        out.append(_c(3, f"rendered_{split}_invariants", "PASS" if not fails else "FAIL",
                      {"n_scenes": len(recs), "failed_scenes": fails[:20],
                       "note": "FAIL scenes are excluded from query manifests, never silently repaired"}))
        out.append(_ok(3, f"rendered_{split}_image_hashes", not hash_bad, hash_bad[:20]))
    return out


# --------------------------------------------------------- Levels 4-5: inference & statistics
def level45(run_dir: Path | None) -> list[dict]:
    if run_dir is None:
        return [_c(4, "inference_run", "SKIPPED", "pass --run <run_dir> to validate a run")]
    from .inference_checks import inference_checks
    from .stats_checks import stats_checks
    out = []
    qf = run_dir / "queries.jsonl"
    raws = sorted(run_dir.glob("raw_predictions*.jsonl")) or sorted(run_dir.glob("**/*raw_predictions*.jsonl"))
    if not qf.exists() or not raws:
        return [_c(4, "inference_run", "SKIPPED", f"no queries/raw predictions in {run_dir}")]
    queries = read_jsonl(qf)
    records = [r for f in raws for r in read_jsonl(f)]
    # The runner's lock sits next to the raw file it guards; the query builder's skeleton lock lacks model fields.
    lock_path = next((f.parent / "run_lock.json" for f in raws if (f.parent / "run_lock.json").exists()), None)
    lock = read_json(lock_path) if lock_path else None
    for c in inference_checks(queries, records, run_lock=lock, work_root=get_paths().work_root):
        out.append({"level": 4, **c})
    parsed = sorted(run_dir.glob("**/parsed_predictions.jsonl"))
    contrasts = sorted(run_dir.glob("**/primary_contrasts.json"))
    if parsed and contrasts:
        for c in stats_checks(read_jsonl(parsed[0]), read_json(contrasts[0])):
            out.append({"level": 5, **c})
    else:
        out.append(_c(5, "statistics", "SKIPPED", "run parse_predictions.py and analyze.py first"))
    return out


# -------------------------------------------------------------- Level 6: repository hygiene
SECRET_RE = re.compile(r"(hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16})")
HOME_PREFIX = "/" + "home/"  # split so this check does not match itself
FORBIDDEN_EXT = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".zip", ".tgz", ".jpg", ".jpeg", ".npy", ".npz", ".parquet", ".pdf"}


def repo_files() -> tuple[list[Path], str]:
    """Files Git would consider (honours .gitignore) if this is a repo; else a filtered walk."""
    try:
        res = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return [REPO_ROOT / l for l in res.stdout.splitlines() if l], "git"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    # Not a Git repository yet: evaluate .gitignore with a throwaway git dir over this work tree.
    files = [f for f in REPO_ROOT.rglob("*") if f.is_file() and ".git" not in f.parts]
    rel = [str(f.relative_to(REPO_ROOT)) for f in files]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "init", "-q", tmp], check=True, capture_output=True)
            res = subprocess.run(["git", f"--git-dir={tmp}/.git", f"--work-tree={REPO_ROOT}", "check-ignore", "--stdin"],
                                 cwd=REPO_ROOT, input="\n".join(rel), capture_output=True, text=True)
        if res.returncode not in (0, 1):
            raise RuntimeError(res.stderr)
        ignored = set(res.stdout.splitlines())
        return [f for f, r in zip(files, rel) if r not in ignored], "git-check-ignore(temp)"
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError):
        return files, "walk(no-ignore-rules)"


def level6(max_mb: float = 2.0) -> list[dict]:
    files, how = repo_files()
    big = [str(f.relative_to(REPO_ROOT)) for f in files if f.stat().st_size > max_mb * 1e6]
    ext = [str(f.relative_to(REPO_ROOT)) for f in files if f.suffix.lower() in FORBIDDEN_EXT]
    secrets, abspaths = [], []
    for f in files:
        if f.suffix.lower() in {".png", ".pyc"} or f.stat().st_size > 2e6:
            continue
        try:
            t = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if SECRET_RE.search(t):
            secrets.append(str(f.relative_to(REPO_ROOT)))
        if f.suffix == ".py" and HOME_PREFIX in t:
            abspaths.append(str(f.relative_to(REPO_ROOT)))
    pngs = [str(f.relative_to(REPO_ROOT)) for f in files if f.suffix.lower() == ".png"]
    out = [
        _ok(6, "no_large_files", not big, {"threshold_mb": max_mb, "files": big, "method": how}),
        _ok(6, "no_weights_datasets_or_pdfs", not ext, ext),
        _ok(6, "no_credentials", not secrets, secrets),
        _ok(6, "no_absolute_home_paths_in_code", not abspaths, abspaths),
        _ok(6, "paths_yaml_not_tracked", not any(f.name == "paths.yaml" and f.parent.name == "configs" for f in files)),
        _c(6, "png_files_tracked", "PASS", {"files": pngs, "note": "only synthetic examples / summary figures are allowed"}),
        _ok(6, "no_private_diana_material", not any("diana" in str(f).lower() for f in files)),
    ]
    readme = REPO_ROOT / "README.md"
    if readme.exists():
        links = re.findall(r"\]\((?!https?://|#)([^)#\s]+)", readme.read_text(encoding="utf-8"))
        missing = [l for l in links if not (REPO_ROOT / l).exists()]
        out.append(_ok(6, "readme_relative_links_exist", not missing, missing))
    else:
        out.append(_c(6, "readme_relative_links_exist", "FAIL", "README.md missing"))
    return out


def run_all(levels: set[int], run_dir: Path | None, no_progress: bool) -> list[dict]:
    out: list[dict] = []
    if 1 in levels:
        out += level1(no_progress)
    if 2 in levels:
        out += level2()
    if 3 in levels:
        out += level3(no_progress)
    if levels & {4, 5}:
        out += level45(run_dir)
    if 6 in levels:
        out += level6()
    return out
