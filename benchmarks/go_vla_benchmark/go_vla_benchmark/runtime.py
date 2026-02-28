"""Runtime helpers for Go benchmark scripts."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from typing import Dict, Optional

SUPPORTED_PYTHON_MINORS = {(3, 10), (3, 11)}


def python_version_tuple() -> tuple[int, int]:
    """Return current interpreter major / minor version."""
    return (sys.version_info.major, sys.version_info.minor)


def is_supported_python() -> bool:
    """True if current Python version is in the validated set."""
    return python_version_tuple() in SUPPORTED_PYTHON_MINORS


def python_version_warning() -> Optional[str]:
    """Return warning text when running outside the validated Python versions."""
    if is_supported_python():
        return None
    major, minor = python_version_tuple()
    return (
        f"Python {major}.{minor} detected. "
        "This benchmark is validated on Python 3.10 / 3.11."
    )


def _first_existing_path(paths: list[str]) -> Optional[str]:
    for path in paths:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    return None


def resolve_gnugo_path(preferred: Optional[str] = None) -> str:
    """Resolve a usable gnugo binary path."""
    candidates: list[str] = []

    if preferred:
        candidates.append(preferred)

    env_path = os.environ.get("GNUGO_PATH")
    if env_path:
        candidates.append(env_path)

    which_path = shutil.which("gnugo")
    if which_path:
        candidates.append(which_path)

    try:
        from physics_planning_games.board_games import go_logic

        if getattr(go_logic, "GNUGO_PATH", None):
            candidates.append(go_logic.GNUGO_PATH)
    except Exception:
        # Import can fail before PYTHONPATH bootstrap; keep fallback behavior.
        pass

    resolved = _first_existing_path(candidates)
    if resolved is None:
        raise FileNotFoundError(
            "Could not find `gnugo`. Install it and ensure it is in PATH, set "
            "GNUGO_PATH, or pass --gnugo-path."
        )
    return resolved


def configure_gnugo_path(preferred: Optional[str] = None) -> str:
    """Set physics_planning_games Go engine path to a resolved gnugo binary."""
    gnugo_path = resolve_gnugo_path(preferred=preferred)
    from physics_planning_games.board_games import go_logic

    go_logic.GNUGO_PATH = gnugo_path
    return gnugo_path


def check_import(module: str) -> Dict[str, object]:
    """Import a module and return lightweight diagnostic info."""
    try:
        importlib.import_module(module)
        return {"ok": True}
    except Exception as exc:  # pragma: no cover - diagnostics path
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
