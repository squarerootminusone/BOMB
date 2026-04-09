#!/usr/bin/env python3
"""Generate MimicGen-augmented Go demonstrations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.paths import bootstrap_pythonpath

bootstrap_pythonpath(REPO_ROOT)

from go_vla_benchmark.generate import generate_augmented_demonstrations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default=str(REPO_ROOT / "benchmarks" / "go_vla_benchmark" / "data" / "source_go.hdf5"),
        help="source hdf5 dataset path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "benchmarks" / "go_vla_benchmark" / "data" / "augmented_go.hdf5"),
        help="output hdf5 dataset path",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default=str(REPO_ROOT / "benchmarks" / "go_vla_benchmark" / "configs" / "go_single_move_task.json"),
        help="task config json",
    )
    parser.add_argument(
        "--environment-name",
        type=str,
        default="go_7x7",
        help="environment name (for example: go_7x7, go_5x5_rigid_bodies, robosuite_go_5x5_rigid_bodies)",
    )
    parser.add_argument("--num-demos", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--opening-min", type=int, default=None)
    parser.add_argument("--opening-max", type=int, default=None)
    parser.add_argument(
        "--camera-size",
        type=int,
        default=None,
        help="square camera size (pixels); overrides --camera-height / --camera-width",
    )
    # Must be 256 for readable preview videos
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--gnugo-path", type=str, default=None, help="optional path to gnugo binary")
    parser.add_argument("--robot", type=str, default="Panda", help="robot name for robosuite backend")
    parser.add_argument(
        "--gripper-types",
        type=str,
        default="default",
        help="gripper type for robosuite backend",
    )
    parser.add_argument("--action-scale", type=float, default=0.03, help="eef delta scale per step")
    parser.add_argument(
        "--success-hold-steps",
        type=int,
        default=0,
        help="extra steps after success before episode termination",
    )
    parser.add_argument(
        "--no-physical-arm",
        action="store_true",
        help="disable IK-driven physical Kinova motion and use pseudo EEF only",
    )
    parser.add_argument(
        "--enable-opponent-moves",
        action="store_true",
        help="allow opponent responses after self move placement",
    )
    parser.add_argument(
        "--no-opponent-opening",
        action="store_true",
        help="disable opponent placements during random opening seeding",
    )
    parser.add_argument(
        "--no-carried-stone",
        action="store_true",
        help="disable visual carried-stone marker that follows gripper before placement",
    )
    parser.add_argument(
        "--no-eef-overlay",
        action="store_true",
        help="disable EEF / target overlay drawn on agentview frames",
    )
    parser.add_argument(
        "--eef-overlay-trail",
        type=int,
        default=10,
        help="number of prior EEF points shown in overlay trail",
    )
    parser.add_argument("--no-image-obs", action="store_true", help="disable image observations")
    parser.add_argument("--num-workers", type=int, default=1, help="parallel worker processes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.camera_size is not None:
        if int(args.camera_size) <= 0:
            raise ValueError("--camera-size must be > 0")
        camera_height = int(args.camera_size)
        camera_width = int(args.camera_size)
    else:
        camera_height = int(args.camera_height)
        camera_width = int(args.camera_width)

    stats = generate_augmented_demonstrations(
        source_dataset_path=args.source,
        output_path=args.output,
        task_config_path=args.task_config,
        environment_name=args.environment_name,
        num_demos=args.num_demos,
        max_attempts=args.max_attempts,
        seed=args.seed,
        opening_moves_min=args.opening_min,
        opening_moves_max=args.opening_max,
        include_image_obs=not args.no_image_obs,
        camera_height=camera_height,
        camera_width=camera_width,
        gnugo_path=args.gnugo_path,
        action_scale=args.action_scale,
        success_hold_steps=args.success_hold_steps,
        drive_physical_arm=not args.no_physical_arm,
        enable_opponent_moves=args.enable_opponent_moves,
        opening_with_opponent=not args.no_opponent_opening,
        render_carried_stone=not args.no_carried_stone,
        render_eef_overlay=not args.no_eef_overlay,
        eef_overlay_trail=args.eef_overlay_trail,
        robot=args.robot,
        gripper_types=args.gripper_types,
        num_workers=args.num_workers,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
