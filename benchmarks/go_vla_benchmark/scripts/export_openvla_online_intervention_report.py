#!/usr/bin/env python3
"""Run online text-masked intervention rollouts and export a trajectory PNG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))
sys.path.insert(0, str(REPO_ROOT / "openvla"))

from go_vla_benchmark.paths import bootstrap_pythonpath  # noqa: E402

bootstrap_pythonpath(REPO_ROOT)

from go_vla_benchmark.common import GoResetOptions  # noqa: E402,F401
from go_vla_benchmark.env_factory import create_benchmark_env  # noqa: E402
from go_vla_benchmark.explainability import (  # noqa: E402
    OpenVLAInterventionAdapter,
    collect_online_text_mask_report,
    export_online_intervention_report_png,
    online_text_mask_report_manifest,
    resolve_optional_path,
    resolve_repo_relative_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an online OpenVLA task-intervention trajectory report")
    parser.add_argument("--checkpoint", required=True, type=str, help="OpenVLA checkpoint path or HF id")
    parser.add_argument("--output-png", required=True, type=str, help="destination PNG for the trajectory report")
    parser.add_argument("--summary-output", type=str, default=None, help="optional JSON sidecar path")
    parser.add_argument("--environment-name", type=str, default="robosuite_go_5x5_rigid_bodies")
    parser.add_argument("--target-row", type=int, default=3)
    parser.add_argument("--target-col", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=5, help="number of masked attempts to run")
    parser.add_argument("--max-steps", type=int, default=200, help="simulator steps per attempt")
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--opening-moves", type=int, default=0)
    parser.add_argument("--stone-color", type=str, default="black")
    parser.add_argument(
        "--instruction-template",
        type=str,
        default="Place a black stone on the Go board at row {row}, column {col}.",
        help="format string with {row} and {col}",
    )
    parser.add_argument("--mask-label", type=str, default=None, help="optional explicit text span to mask")
    parser.add_argument("--mask-index", type=int, default=None, help="optional explicit text mask candidate index")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--gripper-types", type=str, default="default")
    parser.add_argument("--prompt-style", choices=["openvla", "openvla-v01"], default=None)
    parser.add_argument("--unnorm-key", type=str, default=None)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--attention-layers", type=int, default=4)
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def _make_instruction(template: str, row: int, col: int) -> str:
    return str(template).format(row=int(row), col=int(col))


def main() -> None:
    args = parse_args()
    output_png = resolve_repo_relative_path(args.output_png, repo_root=REPO_ROOT)
    summary_path = resolve_optional_path(args.summary_output, repo_root=REPO_ROOT)
    instruction = _make_instruction(args.instruction_template, row=args.target_row, col=args.target_col)

    policy = OpenVLAInterventionAdapter(
        checkpoint=args.checkpoint,
        prompt_style=args.prompt_style,
        unnorm_key=args.unnorm_key,
        action_dim=args.action_dim,
        attention_layers=args.attention_layers,
        attn_implementation=args.attn_implementation,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    def env_factory():
        return create_benchmark_env(
            seed=args.seed,
            environment_name=args.environment_name,
            include_image_obs=True,
            camera_height=args.camera_size,
            camera_width=args.camera_size,
            max_steps=args.max_steps,
            success_hold_steps=0,
            render_eef_overlay=False,
            robot=args.robot,
            gripper_types=args.gripper_types,
        )

    report = collect_online_text_mask_report(
        policy=policy,
        env_factory=env_factory,
        instruction=instruction,
        target_row=args.target_row,
        target_col=args.target_col,
        max_steps=args.max_steps,
        masked_attempts=args.attempts,
        opening_moves=args.opening_moves,
        stone_color=args.stone_color,
        checkpoint=args.checkpoint,
        mask_index=args.mask_index,
        mask_label=args.mask_label,
    )
    export_online_intervention_report_png(report=report, output_path=output_png)

    manifest = online_text_mask_report_manifest(report)
    manifest["output_png"] = str(output_png)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
