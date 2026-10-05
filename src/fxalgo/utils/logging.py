"""Logging configuration.

Wires a single root logger that writes to both stderr and `logs/fxalgo.log`
with a consistent format. Call `setup_logging()` once at process start.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Initialize root logger handlers. Safe to call multiple times."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    if root.handlers:
        return

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(stream)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "fxalgo.log", maxBytes=10_000_000, backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(file_handler)
