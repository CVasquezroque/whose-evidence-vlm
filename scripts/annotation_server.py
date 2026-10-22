#!/usr/bin/env python3
"""Start the local annotation interface (loopback only).

On the server:   python -u scripts/annotation_server.py --port 8765
On your laptop:  ssh -N -L 8765:127.0.0.1:8765 <user>@<server>
Then open        http://localhost:8765
"""
from __future__ import annotations

import argparse

from evidence_ownership.annotation.server import serve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--splits", default="train2017,val2017",
                    help="comma-separated COCO splits whose prepared candidates are served")
    a = ap.parse_args()
    serve(a.host, a.port, tuple(s for s in a.splits.split(",") if s))


if __name__ == "__main__":
    main()
