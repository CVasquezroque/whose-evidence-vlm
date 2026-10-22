#!/usr/bin/env python3
"""Stage a local, immutable release for the Hugging Face Storage Bucket. Nothing is uploaded.

Builds <work_root>/release/releases/<release_id>/ where release_id = eob-<UTC date>-<12 hex of the
content hash>. By default the release is the Git-eligible repository snapshot (README at the root,
code, curated data, results, figures, provenance) plus the latest validation report, SHA256SUMS and
release_manifest.json. --include-work-root adds raw work_root material under work_root/. COCO photographs and rendered images are NOT included: they are rebuilt
from the manifests and code (see README, Reproduction).

Examples:
  python -u scripts/stage_release.py --runs smoke_qwen35 --dry-run
  python -u scripts/stage_release.py --runs pilot_qwen35 --results pilot
  python -u scripts/stage_release.py --verify <work_root>/release/releases/<release_id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from evidence_ownership.utils.config import REPO_ROOT, get_paths
from evidence_ownership.utils.hashing import sha256_file
from evidence_ownership.utils.io import write_json
from evidence_ownership.utils.logs import get_logger, progress
from evidence_ownership.validation.pipeline import FORBIDDEN_EXT, SECRET_RE, repo_files

MANIFEST_GLOBS = ["candidates_*.jsonl", "screen_stats_*.json", "images_*.jsonl", "items_*.jsonl",
                  "exclusions_*.jsonl", "selection_flow_*.json", "duplicate_clusters_*.json"]
RUN_FILES = ["queries.jsonl", "query_manifest_summary.json", "run_lock.json", "*raw_predictions*.jsonl",
             "*raw_predictions*_summary.json", "smoke_test.json", "smoke_reparse_*.json",
             "**/parsed_predictions.jsonl", "**/missing_queries.json", "skipped_scenes.json"]
ALLOWED_RELEASE_EXT = {".sha256sums", ".md", ".py", ".js", ".css", ".html", ".json", ".jsonl", ".csv", ".yaml", ".yml", ".toml",
                       ".txt", ".png", ".cff", ".example", ""}
log = get_logger("stage_release")


def collect(args, paths) -> dict[str, Path]:
    """Map release-relative path -> source file."""
    files: dict[str, Path] = {}
    for f in repo_files()[0]:
        rel = str(f.relative_to(REPO_ROOT))
        if any(rel.startswith(x) for x in args.exclude):
            continue
        files[rel] = f
    files["README.md"] = REPO_ROOT / "README.md"
    sci = Path(args.science_dir) if args.science_dir else None
    if sci and (sci / "PROPOSED_PROTOCOL.json").exists():
        files["protocol/PROPOSED_PROTOCOL.json"] = sci / "PROPOSED_PROTOCOL.json"
        files["protocol/RESEARCH_DECISION_REPORT.md"] = sci / "RESEARCH_DECISION_REPORT.md"
    for f in sorted((paths.work_root / "validation").glob("report_*.json"))[-1:]:
        files[f"validation/{f.name}"] = f
    if not args.include_work_root:
        return files
    for g in MANIFEST_GLOBS:
        for f in sorted(paths.manifests.glob(g)):
            files[f"work_root/manifests/{f.name}"] = f
    for role_dir in sorted(paths.annotations.glob("*")):
        for f in sorted(role_dir.rglob("*.json")):
            files[f"work_root/annotations/{f.relative_to(paths.annotations)}"] = f
    for run in args.runs:
        rd = paths.runs / run
        if not rd.is_dir():
            raise SystemExit(f"run not found: {rd}")
        for g in RUN_FILES:
            for f in sorted(rd.glob(g)):
                files[f"runs/{run}/{f.relative_to(rd)}"] = f
    for res in args.results:
        for base in (paths.results / res, paths.runs / res):
            for f in sorted(base.rglob("*")) if base.is_dir() else []:
                if f.is_file() and f.suffix in {".json", ".csv", ".png"}:
                    files[f"results/{res}/{f.relative_to(base)}"] = f
    env = Path(args.environment_dir) if args.environment_dir else None
    if env and env.is_dir():
        for f in sorted(env.glob("*")):
            if f.is_file() and f.suffix in {".txt", ".json", ".yml", ".csv", ".md"}:
                files[f"environment/{f.name}"] = f
    if args.readiness and Path(args.readiness).exists():
        files[f"environment/{Path(args.readiness).name}"] = Path(args.readiness)
    return files


def verify(release: Path, no_progress: bool = False) -> list[str]:
    problems = []
    sums = release / "SHA256SUMS"
    if not sums.exists():
        return ["SHA256SUMS missing"]
    listed = {}
    for line in sums.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        listed[rel] = digest
    actual = {str(f.relative_to(release)) for f in release.rglob("*") if f.is_file() and f.name != "SHA256SUMS"}
    for rel in sorted(actual - set(listed)):
        problems.append(f"unlisted file {rel}")
    for rel, digest in progress(sorted(listed.items()), desc="verify release", unit="file", disable=no_progress):
        f = release / rel
        if not f.exists():
            problems.append(f"missing {rel}")
        elif sha256_file(f) != digest:
            problems.append(f"hash mismatch {rel}")
        if f.suffix.lower() in FORBIDDEN_EXT or f.suffix.lower() not in ALLOWED_RELEASE_EXT:
            problems.append(f"forbidden file type {rel}")
        if f.exists() and f.stat().st_size > 50e6:
            problems.append(f"unexpectedly large file {rel}")
        if f.suffix in {".md", ".py", ".json", ".yaml", ".txt"} and f.exists() and f.stat().st_size < 5e6:
            if SECRET_RE.search(f.read_text(encoding="utf-8", errors="ignore")):
                problems.append(f"possible credential in {rel}")
    if not (release / "README.md").exists():
        problems.append("README.md missing at release root")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="run names under <work_root>/runs to include")
    ap.add_argument("--results", nargs="*", default=[], help="result dirs under <work_root>/results (or runs) to include")
    ap.add_argument("--science-dir", default=None, help="optional folder holding the original protocol JSON and report")
    ap.add_argument("--environment-dir", default=None, help="folder of environment records (pip freeze, checks)")
    ap.add_argument("--readiness", default=None, help="optional environment audit file to include")
    ap.add_argument("--exclude", nargs="*", default=[], help="repository path prefixes left out of code/")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--include-work-root", action="store_true",
                    help="also copy work_root manifests, annotation history, runs, results and environment records")
    ap.add_argument("--verify", type=Path, default=None, help="verify an existing staged release and exit")
    a = ap.parse_args()
    if a.verify:
        problems = verify(a.verify, a.no_progress)
        print("\n".join(problems) if problems else f"PASS: {a.verify}")
        sys.exit(1 if problems else 0)
    paths = get_paths()
    files = collect(a, paths)
    bad = [r for r, f in files.items() if f.suffix.lower() in FORBIDDEN_EXT or f.suffix.lower() not in ALLOWED_RELEASE_EXT]
    if bad:
        raise SystemExit(f"refusing forbidden file types: {bad[:10]}")
    digests = {rel: sha256_file(f) for rel, f in progress(sorted(files.items()), desc="hash", unit="file", disable=a.no_progress)}
    content_hash = hashlib.sha256("".join(f"{d}  {r}\n" for r, d in sorted(digests.items())).encode()).hexdigest()
    release_id = f"eob-{datetime.now(timezone.utc):%Y%m%d}-{content_hash[:12]}"
    total = sum(f.stat().st_size for f in files.values())
    log.info("%d files, %.1f MB, release_id %s", len(files), total / 1e6, release_id)
    if a.dry_run:
        for rel in sorted(files)[:40]:
            print(rel)
        return
    dest = paths.release / "releases" / release_id
    if dest.exists():
        log.info("identical release already staged: %s", dest)
        return
    tmp = dest.with_name(dest.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    for rel, src in progress(sorted(files.items()), desc="copy", unit="file", disable=a.no_progress):
        out = tmp / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
    write_json(tmp / "release_manifest.json", {
        "release_id": release_id, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "content_sha256": content_hash, "n_files": len(files), "total_bytes": total,
        "runs": a.runs, "results": a.results,
        "excluded": "COCO photographs, rendered images, model weights, caches, credentials (reconstruct via code)",
        "files": {r: {"sha256": d, "bytes": files[r].stat().st_size} for r, d in sorted(digests.items())},
    })
    digests["release_manifest.json"] = sha256_file(tmp / "release_manifest.json")
    (tmp / "SHA256SUMS").write_text("".join(f"{d}  {r}\n" for r, d in sorted(digests.items())))
    tmp.rename(dest)
    problems = verify(dest, a.no_progress)
    print("\n".join(problems) if problems else f"PASS: staged and verified {dest}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
