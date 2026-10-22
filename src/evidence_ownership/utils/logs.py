"""Structured logging and tqdm progress that coexist on the terminal."""
from __future__ import annotations

import logging
import sys
from typing import Iterable, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


class _TqdmHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:  # pragma: no cover
            self.handleError(record)


def get_logger(name: str = "eob", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = _TqdmHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def progress(it: Iterable[T], *, total: int | None = None, desc: str = "", unit: str = "it",
             disable: bool = False) -> Iterable[T]:
    """tqdm wrapper used everywhere; ``disable`` is wired to each CLI's ``--no-progress``."""
    return tqdm(it, total=total, desc=desc, unit=unit, disable=disable, dynamic_ncols=True,
                smoothing=0.1, file=sys.stderr)
