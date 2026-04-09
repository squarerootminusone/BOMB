#!/usr/bin/env python3
"""Run online trajectory intervention rollouts and export a trajectory PNG."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import re
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))
sys.path.insert(0, str(REPO_ROOT / "openvla"))

from go_vla_benchmark.paths import bootstrap_pythonpath  # noqa: E402

bootstrap_pythonpath(REPO_ROOT)

from go_vla_benchmark.common import GoResetOptions  # noqa: E402,F401
from go_vla_benchmark.env_factory import create_benchmark_env  # noqa: E402
from go_vla_benchmark.explainability import (  # noqa: E402
    DATASET_ADAPTERS,
    OpenVLAInterventionAdapter,
    collect_online_intervention_report,
    collect_online_text_mask_report,
    export_online_intervention_report_png,
    online_text_mask_report_manifest,
    resolve_dataset_path,
    resolve_optional_path,
    resolve_repo_relative_path,
    select_online_intervention_masks_from_reference,
)
from go_vla_benchmark.rlds_preprocessing import format_instruction_template  # noqa: E402
from go_vla_benchmark.explainability.simulator import (  # noqa: E402
    DEFAULT_SIMULATOR_OPENING_MAX,
    DEFAULT_SIMULATOR_OPENING_MIN,
    collect_simulator_online_demo_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an online OpenVLA task-intervention trajectory report")
    parser.add_argument("--checkpoint", required=True, type=str, help="OpenVLA checkpoint path or HF id")
    parser.add_argument("--output-png", required=True, type=str, help="destination PNG for the trajectory report")
    parser.add_argument("--summary-output", type=str, default=None, help="optional JSON sidecar path")
    parser.add_argument(
        "--baseline-video-output",
        type=str,
        default=None,
        help="optional MP4 path for the baseline / unmasked rollout video; masked attempts are written beside it",
    )
    parser.add_argument("--video-fps", type=int, default=12, help="fps for exported rollout MP4s")
    parser.add_argument("--environment-name", type=str, default="robosuite_go_5x5_rigid_bodies")
    parser.add_argument("--dataset", type=str, default=None, help="optional Go benchmark HDF5 to drive report inputs")
    parser.add_argument(
        "--simulator-demos",
        type=int,
        default=0,
        help="number of random simulator scenes to export when not using --dataset",
    )
    parser.add_argument("--dataset-adapter", choices=sorted(DATASET_ADAPTERS.keys()), default="go-hdf5")
    parser.add_argument("--demos", type=str, default=None, help="optional comma-separated demo keys")
    parser.add_argument("--start", type=int, default=0, help="start demo index in dataset order")
    parser.add_argument("--num-demos", type=int, default=0, help="number of dataset demos to export; <= 0 means all")
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="dataset clip stride used when choosing the HDF5 reference frame for masking",
    )
    parser.add_argument(
        "--reference-step",
        type=int,
        default=0,
        help="index into the RLDS-selected demo frames used to choose the text or patch mask from HDF5",
    )
    parser.add_argument("--intervention-kind", choices=["text", "patch"], default="text")
    parser.add_argument("--target-row", type=int, default=3)
    parser.add_argument("--target-col", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=5, help="number of masked attempts to run")
    parser.add_argument("--max-steps", type=int, default=200, help="simulator steps per attempt")
    parser.add_argument(
        "--time-trajectory-color-degradation",
        dest="trajectory_alpha_mode",
        action="store_const",
        const="time",
        default="time",
        help="fade plotted trajectory opacity by time (default: start at 70% opacity, end at 100%)",
    )
    parser.add_argument(
        "--height-trajectory-color-degradation",
        dest="trajectory_alpha_mode",
        action="store_const",
        const="height",
        help="fade plotted trajectory opacity by end-effector height instead of time",
    )
    parser.add_argument(
        "--pickup-attempt-marker-threshold",
        type=float,
        default=None,
        help="deprecated compatibility flag; online intervention PNGs no longer render event markers",
    )
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--opening-moves", type=int, default=0)
    parser.add_argument("--stone-color", type=str, default="black")
    parser.add_argument("--simulator-opening-min", type=int, default=DEFAULT_SIMULATOR_OPENING_MIN)
    parser.add_argument("--simulator-opening-max", type=int, default=DEFAULT_SIMULATOR_OPENING_MAX)
    parser.add_argument(
        "--instruction-template",
        type=str,
        default="Place a black stone on the Go board at row {row}, column {col}.",
        help="format string with {color}, {r}, {c}, {row}, and {col}",
    )
    parser.add_argument("--mask-label", type=str, default=None, help="optional explicit text span to mask")
    parser.add_argument("--mask-index", type=int, default=None, help="optional explicit text mask candidate index")
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="base seed for simulator-backed demo sampling; each sampled simulator demo is exported with its own reproducible derived seed",
    )
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


def _make_instruction(template: str, row: int, col: int, color: str) -> str:
    return format_instruction_template(template, color=color, row=row, col=col)


def _default_baseline_video_path(output_png: Path) -> Path:
    return output_png.with_name(f"{output_png.stem}_baseline.mp4")


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("_") or "report"


def _with_suffix(path: Path | None, suffix: str) -> Path | None:
    if path is None:
        return None
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _video_stem_prefix(video_path: Path) -> str:
    stem = str(video_path.stem)
    if stem.endswith("_baseline"):
        return stem[: -len("_baseline")]
    return stem


def _masked_attempt_video_path(video_path: Path, attempt_index: int) -> Path:
    prefix = _video_stem_prefix(video_path)
    return video_path.with_name(f"{prefix}_masked_attempt_{int(attempt_index):02d}{video_path.suffix}")


def _append_frame_callback(writer):
    return lambda frame, _timestep: writer.append_data(frame)


def _validate_source_args(args: argparse.Namespace) -> None:
    has_dataset = args.dataset is not None
    has_simulator = int(args.simulator_demos) > 0
    if has_dataset and has_simulator:
        raise ValueError("pass at most one of --dataset or --simulator-demos")
    if has_simulator and (args.demos is not None or args.start != 0 or args.num_demos != 0):
        raise ValueError("--simulator-demos does not use --demos, --start, or --num-demos")


def _manual_reset_options(args: argparse.Namespace) -> GoResetOptions:
    return GoResetOptions(
        opening_moves=int(args.opening_moves),
        target_row=int(args.target_row),
        target_col=int(args.target_col),
        stone_color=str(args.stone_color),
    )


def _collect_manual_reference_image(*, env_factory, reset_options: GoResetOptions) -> np.ndarray:
    env = env_factory()
    try:
        obs = env.reset(options=reset_options)
        if "agentview_image" not in obs:
            raise RuntimeError("manual patch masking requires `agentview_image` observations from the simulator")
        return np.asarray(obs["agentview_image"], dtype=np.uint8)
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    _validate_source_args(args)
    output_png = resolve_repo_relative_path(args.output_png, repo_root=REPO_ROOT)
    summary_path = resolve_optional_path(args.summary_output, repo_root=REPO_ROOT)
    baseline_video_path = resolve_optional_path(args.baseline_video_output, repo_root=REPO_ROOT)
    if baseline_video_path is None:
        baseline_video_path = _default_baseline_video_path(output_png)
    instruction = _make_instruction(
        args.instruction_template,
        row=args.target_row,
        col=args.target_col,
        color=args.stone_color,
    )

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
            render_eef_overlay=True,
            robot=args.robot,
            gripper_types=args.gripper_types,
        )

    def _run_single_report(
        *,
        report_output_png: Path,
        report_summary_path: Path | None,
        report_video_path: Path,
        masked_attempt_video_paths: list[Path],
        manifest_overrides: dict[str, object] | None = None,
        report_builder,
    ) -> dict[str, object]:
        report_output_png.parent.mkdir(parents=True, exist_ok=True)
        for video_path in [report_video_path, *masked_attempt_video_paths]:
            video_path.parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            baseline_writer = stack.enter_context(
                imageio.get_writer(str(report_video_path), fps=int(args.video_fps), macro_block_size=1)
            )
            masked_attempt_callbacks = [
                _append_frame_callback(
                    stack.enter_context(imageio.get_writer(str(video_path), fps=int(args.video_fps), macro_block_size=1))
                )
                for video_path in masked_attempt_video_paths
            ]
            report = report_builder(_append_frame_callback(baseline_writer), masked_attempt_callbacks)
        trajectory_alpha_mode = str(args.trajectory_alpha_mode).strip().lower()
        export_online_intervention_report_png(
            report=report,
            output_path=report_output_png,
            trajectory_alpha_mode=trajectory_alpha_mode,
            pickup_attempt_marker_threshold=args.pickup_attempt_marker_threshold,
        )

        manifest = online_text_mask_report_manifest(report)
        manifest["output_png"] = str(report_output_png)
        manifest["output_baseline_video"] = str(report_video_path)
        manifest["output_videos"] = [
            {
                "attempt_index": int(report.baseline.attempt_index),
                "title": str(report.baseline.title),
                "masked": bool(report.baseline.masked),
                "path": str(report_video_path),
            }
        ] + [
            {
                "attempt_index": int(attempt.attempt_index),
                "title": str(attempt.title),
                "masked": bool(attempt.masked),
                "path": str(video_path),
            }
            for attempt, video_path in zip(report.masked_attempts, masked_attempt_video_paths)
        ]
        manifest["video_fps"] = int(args.video_fps)
        manifest["trajectory_alpha_mode"] = trajectory_alpha_mode
        manifest["pickup_attempt_marker_threshold"] = (
            None
            if args.pickup_attempt_marker_threshold is None
            else float(args.pickup_attempt_marker_threshold)
        )
        if manifest_overrides:
            manifest.update(manifest_overrides)
        if report_summary_path is not None:
            report_summary_path.parent.mkdir(parents=True, exist_ok=True)
            report_summary_path.write_text(json.dumps(manifest, indent=2))
        return manifest

    def _spec_reset_options(demo_spec) -> GoResetOptions:
        if hasattr(demo_spec, "reset_options"):
            return demo_spec.reset_options
        return GoResetOptions(
            opening_moves=demo_spec.opening_moves,
            opening_move_history=demo_spec.opening_move_history,
            target_row=demo_spec.target_row,
            target_col=demo_spec.target_col,
            stone_color=demo_spec.stone_color,
        )

    if args.dataset or args.simulator_demos > 0:
        if args.dataset:
            dataset_path = resolve_dataset_path(args.dataset, repo_root=REPO_ROOT)
            dataset_adapter = DATASET_ADAPTERS[args.dataset_adapter]()
            demo_specs = dataset_adapter.load_online_demo_specs(
                dataset_path=dataset_path,
                demos=args.demos,
                start=args.start,
                num_demos=args.num_demos,
                stride=args.stride,
                reference_step=args.reference_step,
            )
            aggregate_prefix: dict[str, object] = {
                "clip_source": "dataset",
                "dataset_path": str(dataset_path),
            }
            empty_error = "no dataset demos selected for online intervention export"
        else:
            demo_specs = collect_simulator_online_demo_specs(
                num_demos=args.simulator_demos,
                seed=args.seed,
                environment_name=args.environment_name,
                camera_size=args.camera_size,
                max_steps=args.max_steps,
                opening_moves_min=args.simulator_opening_min,
                opening_moves_max=args.simulator_opening_max,
                robot=args.robot,
                gripper_types=args.gripper_types,
            )
            aggregate_prefix = {
                "clip_source": "simulator",
                "simulator_demos": int(args.simulator_demos),
                "seed": int(args.seed),
            }
            empty_error = "no simulator demos selected for online intervention export"
        if not demo_specs:
            raise RuntimeError(empty_error)

        multi_output = len(demo_specs) > 1
        manifests: list[dict[str, object]] = []
        for demo_spec in demo_specs:
            selected_masks = select_online_intervention_masks_from_reference(
                policy=policy,
                reference_image=demo_spec.reference_image,
                instruction=demo_spec.instruction,
                intervention_kind=args.intervention_kind,
                max_masks=args.attempts,
            )
            suffix = _safe_name(f"{demo_spec.demo_key}_{args.intervention_kind}")
            report_output_png = _with_suffix(output_png, suffix) if multi_output else output_png
            report_summary_path = _with_suffix(summary_path, suffix) if multi_output else summary_path
            report_video_path = _with_suffix(baseline_video_path, suffix) if multi_output else baseline_video_path
            masked_attempt_video_paths = [
                _masked_attempt_video_path(report_video_path, attempt_index + 1)
                for attempt_index in range(len(selected_masks))
            ]
            assert report_output_png is not None
            assert report_video_path is not None

            manifests.append(
                _run_single_report(
                    report_output_png=report_output_png,
                    report_summary_path=report_summary_path,
                    report_video_path=report_video_path,
                    masked_attempt_video_paths=masked_attempt_video_paths,
                    manifest_overrides={"seed": int(args.seed)},
                    report_builder=lambda baseline_callback, masked_callbacks, demo_spec=demo_spec, selected_masks=selected_masks: collect_online_intervention_report(
                        policy=policy,
                        env_factory=env_factory,
                        instruction=demo_spec.instruction,
                        target_row=demo_spec.target_row,
                        target_col=demo_spec.target_col,
                        max_steps=args.max_steps,
                        masked_attempts=args.attempts,
                        checkpoint=args.checkpoint,
                        intervention_kind=args.intervention_kind,
                        mask=selected_masks[0] if selected_masks else None,
                        masked_attempt_masks=selected_masks,
                        reset_options=_spec_reset_options(demo_spec),
                        demo_key=demo_spec.demo_key,
                        reference_frame_index=demo_spec.reference_frame_index,
                        simulator_seed=None if not hasattr(demo_spec, "simulator_seed") else int(demo_spec.simulator_seed),
                        baseline_frame_callback=baseline_callback,
                        masked_attempt_frame_callbacks=masked_callbacks,
                    ),
                )
            )

        aggregate: dict[str, object]
        if multi_output:
            aggregate = dict(aggregate_prefix)
            aggregate["intervention_kind"] = args.intervention_kind
            aggregate["reports"] = manifests
            if summary_path is not None:
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps(aggregate, indent=2))
        else:
            aggregate = manifests[0]
            aggregate.update({key: value for key, value in aggregate_prefix.items() if key not in aggregate})
        print(json.dumps(aggregate, indent=2))
        return

    manual_reset_options = _manual_reset_options(args)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    baseline_video_path.parent.mkdir(parents=True, exist_ok=True)
    if args.intervention_kind == "patch":
        reference_image = _collect_manual_reference_image(env_factory=env_factory, reset_options=manual_reset_options)
        selected_masks = select_online_intervention_masks_from_reference(
            policy=policy,
            reference_image=reference_image,
            instruction=instruction,
            intervention_kind="patch",
            max_masks=args.attempts,
        )
        masked_attempt_video_paths = [
            _masked_attempt_video_path(baseline_video_path, attempt_index + 1)
            for attempt_index in range(len(selected_masks))
        ]
        manifest = _run_single_report(
            report_output_png=output_png,
            report_summary_path=summary_path,
            report_video_path=baseline_video_path,
            masked_attempt_video_paths=masked_attempt_video_paths,
            report_builder=lambda baseline_callback, masked_callbacks: collect_online_intervention_report(
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
                intervention_kind="patch",
                mask=selected_masks[0] if selected_masks else None,
                masked_attempt_masks=selected_masks,
                reset_options=manual_reset_options,
                baseline_frame_callback=baseline_callback,
                masked_attempt_frame_callbacks=masked_callbacks,
            ),
        )
    else:
        masked_attempt_video_paths = [
            _masked_attempt_video_path(baseline_video_path, attempt_index + 1)
            for attempt_index in range(max(0, int(args.attempts)))
        ]
        manifest = _run_single_report(
            report_output_png=output_png,
            report_summary_path=summary_path,
            report_video_path=baseline_video_path,
            masked_attempt_video_paths=masked_attempt_video_paths,
            report_builder=lambda baseline_callback, masked_callbacks: collect_online_text_mask_report(
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
                baseline_frame_callback=baseline_callback,
                masked_attempt_frame_callbacks=masked_callbacks,
            ),
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
