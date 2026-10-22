"""Selective download of individual COCO images with decode/dimension verification.

COCO publishes no per-image checksums, so integrity means: HTTPS with certificate verification,
JPEG decodes completely, dimensions equal the annotation record, and the SHA-256 we observe is
recorded for every later stage to re-verify.
"""
from __future__ import annotations

import os
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image

from ..utils.hashing import sha256_file


class DownloadError(RuntimeError):
    pass


def verify_image(path: Path, width: int, height: int) -> None:
    with Image.open(path) as im:
        im.load()  # full decode, catches truncation
        if im.size != (width, height):
            raise DownloadError(f"{path.name}: size {im.size} != annotation {(width, height)}")


def fetch(url: str, dest: Path, width: int, height: int, timeout: float = 30.0, retries: int = 3) -> str:
    """Download to ``dest`` atomically, verify, and return the SHA-256. Existing valid files are kept."""
    if dest.exists():
        verify_image(dest, width, height)
        return sha256_file(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
        os.close(fd)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "evidence-ownership-benchmark/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as fh:
                if r.status != 200:
                    raise DownloadError(f"HTTP {r.status} for {url}")
                fh.write(r.read())
            verify_image(Path(tmp), width, height)
            os.replace(tmp, dest)
            return sha256_file(dest)
        except Exception as exc:  # network or verification failure: retry, then fail visibly
            last = exc
            Path(tmp).unlink(missing_ok=True)
            time.sleep(min(2 ** attempt, 10))
    raise DownloadError(f"failed after {retries} attempts: {url}: {last}")
