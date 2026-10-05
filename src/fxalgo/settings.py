"""Project paths and runtime settings.

The project root is discovered by walking up from this file until a
`pyproject.toml` is found. Override it with the `FXALGO_PROJECT_ROOT`
environment variable when running outside a source checkout. Market data,
features and outputs all live under `<project_root>/data/`.

The only runtime setting is the log level, read from `FXALGO_LOG_LEVEL`
(default `INFO`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _discover_project_root() -> Path:
    """Walk up from this file to find the directory containing pyproject.toml."""
    override = os.environ.get("FXALGO_PROJECT_ROOT")
    if override:
        return Path(override).resolve()

    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT: Path = _discover_project_root()

# The six major pairs the research covers. Any pair with bid/ask data in the
# expected layout (see README) can be used; this is only the default set.
PAIRS: tuple[str, ...] = ("EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "USDJPY")


@dataclass(frozen=True)
class Settings:
    """Runtime settings."""

    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton."""
    return Settings(log_level=os.environ.get("FXALGO_LOG_LEVEL", "INFO").upper())
