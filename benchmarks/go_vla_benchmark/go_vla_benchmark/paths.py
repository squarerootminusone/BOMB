"""Repository path helpers for local imports."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root_from_file(file_path: str) -> Path:
    """Infer repository root from a file in benchmarks/go_vla_benchmark."""
    return Path(file_path).resolve().parents[3]


def bootstrap_pythonpath(repo_root: Path) -> None:
    """Add local dependency package roots to PYTHONPATH for scripts."""
    candidates = [
        repo_root / "deepmind-research",
        repo_root / "mimicgen",
        # NOTE: robosuite is installed as a site-package; adding the repo-level
        # directory would shadow it with a namespace package whose __file__ is
        # None and whose submodule layout differs from the installed version.
        # repo_root / "robosuite",
        repo_root / "benchmarks" / "go_vla_benchmark",
    ]
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
