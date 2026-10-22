"""Local-only annotation server (stdlib). Binds to 127.0.0.1; reach it through an SSH tunnel.

Serves the static interface in ``annotation/interface/``, source images, rendered review
images, and saves one JSON record per (role, scene). Every save also keeps a timestamped copy
in ``_history/``, so no human judgment is ever overwritten without a trace.
"""
from __future__ import annotations

import json
import mimetypes
import re
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..utils.config import REPO_ROOT, experiment_config, get_paths
from ..utils.io import read_json, read_jsonl, write_json
from ..utils.logs import get_logger
from . import schema
from .sam_assist import SamAssistant, sam_config

INTERFACE_DIR = REPO_ROOT / "annotation" / "interface"
TOOL_VERSION = "eob-annotator-0.1"
SCENE_RE = re.compile(r"^(train2017|val2017)-\d{12}$")
log = get_logger("annotation_server")


class Store:
    def __init__(self, splits: tuple[str, ...]):
        self.paths = get_paths()
        self.cands: dict[str, dict] = {}
        self.order: dict[str, list[str]] = {}
        for split in splits:
            imgs = {r["scene_id"]: r for r in read_jsonl(self.paths.manifests / f"images_{split}.jsonl")}
            ids = []
            for c in read_jsonl(self.paths.manifests / f"candidates_{split}.jsonl"):
                if c["scene_id"] in imgs:
                    c["image_sha256"] = imgs[c["scene_id"]]["sha256"]
                    self.cands[c["scene_id"]] = c
                    ids.append(c["scene_id"])
            self.order[split] = ids
        cfg = sam_config()
        self.sam = SamAssistant(cfg) if cfg else None

    def ann_path(self, role: str, scene_id: str) -> Path:
        return self.paths.annotations / role / f"{scene_id}.json"

    def status(self, role: str, scene_id: str) -> str | None:
        p = self.ann_path(role, scene_id)
        return read_json(p).get("decision") if p.exists() else None


def make_handler(store: Store):
    cfg = experiment_config()

    class Handler(BaseHTTPRequestHandler):
        server_version = TOOL_VERSION

        def log_message(self, fmt, *args):  # route through structured logger
            log.info("%s %s", self.address_string(), fmt % args)

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"))

        def _file(self, path: Path) -> None:
            if not path.is_file():
                return self._json({"error": f"not found: {path.name}"}, 404)
            self._send(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            q = parse_qs(u.query)
            if not parts:
                return self._file(INTERFACE_DIR / "index.html")
            if len(parts) == 1 and parts[0] in ("app.js", "style.css"):
                return self._file(INTERFACE_DIR / parts[0])
            if parts == ["api", "config"]:
                return self._json({
                    "garment_colors": schema.GARMENT_COLORS, "confidence": schema.CONFIDENCE,
                    "roles": schema.ROLES, "author_checks": schema.AUTHOR_CHECKS,
                    "reject_reasons": schema.REJECT_REASONS, "stage2_image": schema.STAGE2_IMAGE,
                    "eligibility": cfg["eligibility"], "rendering": cfg["rendering"],
                    "splits": list(store.order), "tool_version": TOOL_VERSION,
                    "sam_available": store.sam is not None,
                })
            if parts == ["api", "queue"]:
                split = q.get("split", [next(iter(store.order))])[0]
                return self._json([
                    {"scene_id": s, "rank": store.cands[s]["order_rank"],
                     "status": {r: store.status(r, s) for r in schema.ROLES}}
                    for s in store.order.get(split, [])
                ])
            if len(parts) == 3 and parts[:2] == ["api", "scene"] and parts[2] in store.cands:
                return self._json(store.cands[parts[2]])
            if len(parts) == 2 and parts[0] == "image" and parts[1] in store.cands:
                c = store.cands[parts[1]]
                return self._file(store.paths.images_dir(c["coco_split"]) / c["file_name"])
            if len(parts) == 3 and parts[0] == "rendered" and parts[1] in store.cands and re.fullmatch(r"[A-Za-z0-9_]+\.png", parts[2]):
                c = store.cands[parts[1]]
                return self._file(store.paths.rendered / c["study_split"] / parts[1] / parts[2])
            if len(parts) == 4 and parts[:2] == ["api", "annotation"] and parts[2] in schema.ROLES and parts[3] in store.cands:
                p = store.ann_path(parts[2], parts[3])
                return self._json(read_json(p)) if p.exists() else self._json(None)
            return self._json({"error": "not found"}, 404)

        def _sam(self, scene_id: str):
            if store.sam is None:
                return self._json({"error": "SAM assistance not configured (paths.yaml: sam)"}, 503)
            n = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(n)) if 0 < n < 100_000 else None
                pts, labels = body["points_orig"], body["labels"]
                assert 1 <= len(pts) == len(labels) <= 50 and all(l in (0, 1) for l in labels) and 1 in labels
                pts = [[float(x), float(y)] for x, y in pts]
            except Exception:
                return self._json({"error": "body must be {points_orig: [[x,y],...], labels: [1|0,...]} with >=1 positive"}, 400)
            c = store.cands[scene_id]
            try:
                out = store.sam.predict(scene_id, store.paths.images_dir(c["coco_split"]) / c["file_name"], pts, labels)
            except Exception as exc:  # surface model errors to the author; manual polygons remain available
                log.exception("SAM failure")
                return self._json({"error": f"SAM failed: {exc}"}, 500)
            return self._json(out)

        def do_POST(self):  # noqa: N802
            parts = [p for p in urlparse(self.path).path.split("/") if p]
            if len(parts) == 3 and parts[:2] == ["api", "sam"] and parts[2] in store.cands:
                return self._sam(parts[2])
            if not (len(parts) == 4 and parts[:2] == ["api", "annotation"] and parts[2] in schema.ROLES
                    and SCENE_RE.match(parts[3]) and parts[3] in store.cands):
                return self._json({"error": "bad endpoint"}, 404)
            role, scene_id = parts[2], parts[3]
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > 5_000_000:
                return self._json({"error": "bad body size"}, 400)
            try:
                rec = json.loads(self.rfile.read(n))
            except json.JSONDecodeError:
                return self._json({"error": "invalid JSON"}, 400)
            if rec.get("role") != role or rec.get("scene_id") != scene_id:
                return self._json({"error": "role/scene mismatch with URL"}, 400)
            if not str(rec.get("annotator_id", "")).strip():
                return self._json({"error": "annotator_id is required"}, 400)
            errs = schema.structural_errors(rec, store.cands[scene_id])
            if errs:
                return self._json({"error": "validation failed", "details": errs}, 422)
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            path = store.ann_path(role, scene_id)
            if path.exists():
                old = read_json(path)
                rec["created_utc"] = old.get("created_utc", now)
                hist = path.parent / "_history" / f"{scene_id}.{old.get('updated_utc', now).replace(':', '')}.json"
                if not hist.exists():
                    write_json(hist, old)
            else:
                rec["created_utc"] = now
            rec["updated_utc"] = now
            rec["tool_version"] = TOOL_VERSION
            write_json(path, rec)
            info = {"saved": str(path.relative_to(store.paths.work_root)), "updated_utc": now}
            if role == "author":
                info["completeness"] = schema.completeness_errors(rec)
            return self._json(info)

    return Handler


def serve(host: str, port: int, splits: tuple[str, ...]) -> None:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("refusing to bind a non-loopback address; use an SSH tunnel instead")
    store = Store(splits)
    httpd = ThreadingHTTPServer((host, port), make_handler(store))
    log.info("annotation server on http://%s:%d  (%d scenes; annotations -> %s)", host, port,
             len(store.cands), store.paths.annotations)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("stopped")
