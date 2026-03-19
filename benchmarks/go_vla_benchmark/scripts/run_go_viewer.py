#!/usr/bin/env python3
"""Launch the DeepMind Go + Jaco simulator viewer."""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.paths import bootstrap_pythonpath

bootstrap_pythonpath(REPO_ROOT)

from dm_control import viewer
from physics_planning_games import board_games
from go_vla_benchmark.runtime import configure_gnugo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="go_7x7", type=str, help="board game env name")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gnugo-path", default=None, type=str, help="optional path to gnugo binary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_gnugo_path(preferred=args.gnugo_path)
    env_ctor = functools.partial(
        board_games.load,
        environment_name=args.env,
        seed=args.seed,
        time_limit=float("inf"),
        strip_singleton_obs_buffer_dim=True,
    )
    viewer.launch(env_ctor)


if __name__ == "__main__":
    main()
