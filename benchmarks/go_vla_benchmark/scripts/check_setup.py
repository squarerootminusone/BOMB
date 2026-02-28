#!/usr/bin/env python3
"""Check whether this environment is ready for Go + MimicGen data generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.paths import bootstrap_pythonpath

bootstrap_pythonpath(REPO_ROOT)

from go_vla_benchmark.robosuite_compat import ensure_robosuite_compat
from go_vla_benchmark.runtime import (
    check_import,
    is_supported_python,
    python_version_tuple,
    python_version_warning,
    resolve_gnugo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnugo-path", type=str, default=None, help="optional path to gnugo binary")
    parser.add_argument("--strict", action="store_true", help="exit non-zero if generation is not ready")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_robosuite_compat()

    imports = {
        "dm_control": check_import("dm_control"),
        "open_spiel": check_import("pyspiel"),
        "physics_planning_games": check_import("physics_planning_games"),
        "h5py": check_import("h5py"),
        "mimicgen": check_import("mimicgen"),
        "robomimic": check_import("robomimic"),
        "robosuite_or_compat": check_import("robosuite"),
    }

    python_major, python_minor = python_version_tuple()
    python_status = {
        "ok": is_supported_python(),
        "version": f"{python_major}.{python_minor}",
        "warning": python_version_warning(),
    }

    gnugo_status = {"ok": True, "path": None, "error": None}
    try:
        gnugo_status["path"] = resolve_gnugo_path(preferred=args.gnugo_path)
    except Exception as exc:
        gnugo_status["ok"] = False
        gnugo_status["error"] = f"{type(exc).__name__}: {exc}"

    ready_for_viewer = all(
        [
            python_status["ok"],
            gnugo_status["ok"],
            imports["dm_control"]["ok"],
            imports["open_spiel"]["ok"],
            imports["physics_planning_games"]["ok"],
        ]
    )
    ready_for_collect = ready_for_viewer and imports["h5py"]["ok"]
    ready_for_generate = ready_for_collect and all(
        [
            imports["mimicgen"]["ok"],
            imports["robomimic"]["ok"],
            imports["robosuite_or_compat"]["ok"],
        ]
    )

    report = {
        "python": python_status,
        "gnugo": gnugo_status,
        "imports": imports,
        "ready_for": {
            "viewer": ready_for_viewer,
            "collect_source_demos": ready_for_collect,
            "mimicgen_generate": ready_for_generate,
        },
        "next_steps": [
            "Install benchmark deps: pip install -r benchmarks/go_vla_benchmark/requirements.txt",
            "Install MimicGen: pip install -e mimicgen",
            "Install robomimic (no deps): pip install --no-deps -e git+https://github.com/ARISE-Initiative/robomimic.git@d0b37cf214bd24fb590d182edb6384333f67b661#egg=robomimic",
            "robosuite is optional for this Go benchmark (a local compatibility shim is used).",
            "Ensure gnugo is installed and available on PATH (or pass --gnugo-path).",
        ],
    }
    print(json.dumps(report, indent=2))

    if args.strict and (not ready_for_generate):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
