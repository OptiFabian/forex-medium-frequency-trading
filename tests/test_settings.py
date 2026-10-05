"""Smoke tests for settings."""

from __future__ import annotations

from fxalgo.settings import PAIRS, PROJECT_ROOT, get_settings


def test_settings_load() -> None:
    s = get_settings()
    assert s.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def test_project_root_has_pyproject() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_default_pairs() -> None:
    assert "EURUSD" in PAIRS and len(PAIRS) == 6
