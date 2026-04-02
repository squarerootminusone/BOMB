#!/usr/bin/env python3
"""Collect activation-patching traces for OpenVLA checkpoints on Go demos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))
sys.path.insert(0, str(REPO_ROOT / "openvla"))

from go_vla_benchmark.explainability import (  # noqa: E402
    CAUSAL_MODEL_ADAPTERS,
    DATASET_ADAPTERS,
    causal_trace_manifest,
    collect_episode_causal_traces,
    load_intervention_trace_file,
    match_intervention_trace_to_clips,
    resolve_dataset_path,
    resolve_optional_path,
    resolve_repo_relative_path,
    save_causal_trace_file,
)
from go_vla_benchmark.explainability.simulator import (  # noqa: E402
    DEFAULT_SIMULATOR_CAMERA_SIZE,
    DEFAULT_SIMULATOR_ENVIRONMENT,
    DEFAULT_SIMULATOR_MAX_ATTEMPTS_PER_DEMO,
    DEFAULT_SIMULATOR_OPENING_MAX,
    DEFAULT_SIMULATOR_OPENING_MIN,
    collect_simulator_episode_clips,
)


def _print_runtime_diagnostics(args: argparse.Namespace, clips, model_adapter) -> None:
    import torch

    try:
        first_param_device = str(next(model_adapter.model.parameters()).device)
    except StopIteration:
        first_param_device = "<no-parameters>"

    total_frames = sum(int(len(clip.images)) for clip in clips)
    print(
        json.dumps(
            {
                "event": "causal_localization_start",
                "requested_device": args.device,
                "resolved_device": str(model_adapter.device),
                "model_parameter_device": first_param_device,
                "dtype": str(model_adapter.dtype),
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda_version": torch.version.cuda,
                "torch_cuda_device_count": int(torch.cuda.device_count()),
                "num_demos": int(len(clips)),
                "num_frames": int(total_frames),
                "corruption_type": args.corruption_type,
                "intervention_match": args.intervention_match is not None,
                "per_cross_attention": bool(args.per_cross_attention),
                "load_in_8bit": bool(args.load_in_8bit),
                "load_in_4bit": bool(args.load_in_4bit),
            },
            indent=2,
        ),
        file=sys.stderr,
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="optional Go benchmark HDF5 dataset; omit with --simulator-demos to sample random simulator clips",
    )
    parser.add_argument(
        "--simulator-demos",
        type=int,
        default=0,
        help="number of random successful simulator demos to collect when no --dataset is provided",
    )
    parser.add_argument("--checkpoint", required=True, type=str, help="OpenVLA checkpoint path or HF id")
    parser.add_argument("--trace-output", required=True, type=str, help="output .npz trace path")
    parser.add_argument("--summary-output", type=str, default=None, help="optional summary JSON path")
    parser.add_argument("--dataset-adapter", choices=sorted(DATASET_ADAPTERS.keys()), default="go-hdf5")
    parser.add_argument("--model-adapter", choices=sorted(CAUSAL_MODEL_ADAPTERS.keys()), default="openvla")
    parser.add_argument("--prompt-style", choices=["openvla", "openvla-v01"], default=None)
    parser.add_argument("--unnorm-key", type=str, default=None, help="dataset statistics key for de-normalizing actions")
    parser.add_argument("--action-dim", type=int, default=None, help="fallback action dimension when norm stats are absent")
    parser.add_argument("--start", type=int, default=0, help="start demo index in dataset order")
    parser.add_argument("--num-demos", type=int, default=0, help="number of dataset demos to process; <= 0 means all")
    parser.add_argument("--demos", type=str, default=None, help="optional comma-separated dataset demo keys")
    parser.add_argument("--stride", type=int, default=1, help="extra frame stride after RLDS-style subsampling/filtering")
    parser.add_argument("--max-steps", type=int, default=0, help="max timesteps per demo after applying stride")
    parser.add_argument("--seed", type=int, default=7, help="random seed for simulator-backed demo generation")
    parser.add_argument("--environment-name", type=str, default=DEFAULT_SIMULATOR_ENVIRONMENT)
    parser.add_argument("--camera-size", type=int, default=DEFAULT_SIMULATOR_CAMERA_SIZE)
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--gripper-types", type=str, default="default")
    parser.add_argument("--simulator-opening-min", type=int, default=DEFAULT_SIMULATOR_OPENING_MIN)
    parser.add_argument("--simulator-opening-max", type=int, default=DEFAULT_SIMULATOR_OPENING_MAX)
    parser.add_argument(
        "--simulator-max-attempts-per-demo",
        type=int,
        default=DEFAULT_SIMULATOR_MAX_ATTEMPTS_PER_DEMO,
    )
    parser.add_argument("--attention-layers", type=int, default=4, help="number of last decoder layers to average")
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
        help="attention backend; use eager for reliable output_attentions",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--load-in-8bit", action="store_true", help="load model with bitsandbytes 8-bit weights")
    parser.add_argument("--load-in-4bit", action="store_true", help="load model with bitsandbytes 4-bit weights")
    parser.add_argument(
        "--corruption-type",
        choices=["patch-occlusion", "text-masking"],
        default="patch-occlusion",
        help="corrupted baseline used before activation patching",
    )
    parser.add_argument(
        "--corruption-index",
        type=int,
        default=None,
        help="optional explicit patch/token index; defaults to the strongest single intervention",
    )
    parser.add_argument(
        "--intervention-match",
        nargs="?",
        const="auto",
        default=None,
        metavar="TRACE_INPUT",
        help=(
            "reuse the exact demo order, frame indices, and best per-step patch/text choice "
            "from an intervention trace; omit the value to infer a matching .npz under "
            "benchmarks/go_vla_benchmark/data/interventions/"
        ),
    )
    parser.add_argument(
        "--per-cross-attention",
        action="store_true",
        help="also patch any discovered cross-attention-style blocks if the checkpoint exposes them",
    )
    return parser.parse_args()


def _validate_source_args(args: argparse.Namespace) -> None:
    has_dataset = args.dataset is not None
    has_simulator = int(args.simulator_demos) > 0
    if args.intervention_match is None:
        if has_dataset == has_simulator:
            raise ValueError("pass exactly one of --dataset or --simulator-demos")
        if has_simulator and (args.demos is not None or args.start != 0 or args.num_demos != 0):
            raise ValueError("--simulator-demos does not use --demos, --start, or --num-demos")
        return

    if args.corruption_index is not None:
        raise ValueError("--corruption-index cannot be combined with --intervention-match")
    if has_simulator:
        raise ValueError("--intervention-match reuses demos from the intervention trace; do not pass --simulator-demos")
    if args.demos is not None or args.start != 0 or args.num_demos != 0 or args.stride != 1 or args.max_steps != 0:
        raise ValueError(
            "--intervention-match uses the demo and frame selection stored in the intervention trace; "
            "do not pass --demos, --start, --num-demos, --stride, or --max-steps"
        )


def _auto_intervention_trace_candidates(dataset_path: Path, corruption_type: str) -> list[Path]:
    intervention_dir = REPO_ROOT / "benchmarks" / "go_vla_benchmark" / "data" / "interventions"
    dataset_stem = dataset_path.stem
    suffixes = (
        ["_interventions_patches.npz", "_interventions.npz"]
        if corruption_type == "patch-occlusion"
        else ["_interventions_text.npz", "_interventions.npz"]
    )
    return [intervention_dir / f"{dataset_stem}{suffix}" for suffix in suffixes]


def _resolve_intervention_match_trace(
    requested: str,
    dataset_path: Optional[Path],
    corruption_type: str,
) -> Path:
    if requested != "auto":
        trace_path = resolve_repo_relative_path(requested, repo_root=REPO_ROOT)
        if not trace_path.is_file():
            raise FileNotFoundError(f"intervention trace not found: {trace_path}")
        return trace_path

    if dataset_path is None:
        raise ValueError(
            "--intervention-match without an explicit path requires --dataset so the matching trace stem can be inferred"
        )

    candidates = _auto_intervention_trace_candidates(dataset_path=dataset_path, corruption_type=corruption_type)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "could not infer a matching intervention trace for --intervention-match.\n"
        f"Searched:\n{searched}\n"
        "Pass an explicit path to --intervention-match to override the lookup."
    )


def _load_intervention_matched_inputs(
    dataset_path: Optional[Path],
    dataset_adapter,
    corruption_type: str,
    intervention_match: str,
):
    trace_path = _resolve_intervention_match_trace(
        requested=intervention_match,
        dataset_path=dataset_path,
        corruption_type=corruption_type,
    )
    intervention_bundle = load_intervention_trace_file(trace_path)
    if intervention_bundle.embedded_clips_by_key is not None and dataset_path is None:
        clips = [intervention_bundle.embedded_clips_by_key[demo_key] for demo_key in intervention_bundle.demo_keys]
    else:
        if dataset_path is None or dataset_adapter is None:
            raise ValueError(
                "the requested intervention trace does not embed clips; pass --dataset or use a simulator-backed trace"
            )
        clips = dataset_adapter.load_episode_clips(
            dataset_path=dataset_path,
            demos=",".join(intervention_bundle.demo_keys),
            start=0,
            num_demos=0,
            stride=1,
            max_steps=0,
        )
    adjusted_clips, corruption_indices_by_key = match_intervention_trace_to_clips(
        clips=clips,
        trace_bundle=intervention_bundle,
        corruption_type=corruption_type,
    )
    return trace_path, adjusted_clips, corruption_indices_by_key


def main() -> None:
    args = parse_args()
    _validate_source_args(args)

    trace_output_path = resolve_optional_path(args.trace_output, repo_root=REPO_ROOT)
    summary_output_path = resolve_optional_path(args.summary_output, repo_root=REPO_ROOT)
    dataset_path = resolve_dataset_path(args.dataset, repo_root=REPO_ROOT) if args.dataset else None

    dataset_adapter = DATASET_ADAPTERS[args.dataset_adapter]() if dataset_path is not None else None
    intervention_match_trace_path: Optional[Path] = None
    corruption_indices_by_key = None
    if args.intervention_match is None:
        if dataset_path is not None:
            assert dataset_adapter is not None
            clips = dataset_adapter.load_episode_clips(
                dataset_path=dataset_path,
                demos=args.demos,
                start=args.start,
                num_demos=args.num_demos,
                stride=args.stride,
                max_steps=args.max_steps,
            )
        else:
            clips = collect_simulator_episode_clips(
                num_demos=args.simulator_demos,
                seed=args.seed,
                stride=args.stride,
                max_steps=args.max_steps,
                environment_name=args.environment_name,
                camera_size=args.camera_size,
                opening_moves_min=args.simulator_opening_min,
                opening_moves_max=args.simulator_opening_max,
                max_attempts_per_demo=args.simulator_max_attempts_per_demo,
                robot=args.robot,
                gripper_types=args.gripper_types,
            )
    else:
        intervention_match_trace_path, clips, corruption_indices_by_key = _load_intervention_matched_inputs(
            dataset_path=dataset_path,
            dataset_adapter=dataset_adapter,
            corruption_type=args.corruption_type,
            intervention_match=args.intervention_match,
        )
    if not clips:
        raise RuntimeError("no demos available from the selected input source")

    model_adapter = CAUSAL_MODEL_ADAPTERS[args.model_adapter](
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
    _print_runtime_diagnostics(args=args, clips=clips, model_adapter=model_adapter)
    traces = collect_episode_causal_traces(
        clips=clips,
        model_adapter=model_adapter,
        corruption_type=args.corruption_type,
        corruption_index=args.corruption_index,
        corruption_indices_by_key=corruption_indices_by_key,
        per_cross_attention=args.per_cross_attention,
    )

    save_causal_trace_file(
        trace_path=trace_output_path,
        dataset_path=dataset_path,
        clips=clips,
        traces=traces,
        checkpoint=args.checkpoint,
        prompt_style=model_adapter.prompt_style,
        dataset_adapter=args.dataset_adapter if dataset_path is not None else None,
        model_adapter=args.model_adapter,
        embed_clips=dataset_path is None,
    )

    summary = causal_trace_manifest(
        dataset_path=dataset_path,
        checkpoint=args.checkpoint,
        prompt_style=model_adapter.prompt_style,
        dataset_adapter=args.dataset_adapter if dataset_path is not None else None,
        model_adapter=args.model_adapter,
        clips=clips,
        traces=traces,
    )
    summary["trace_path"] = str(trace_output_path)
    summary["corruption_type"] = args.corruption_type
    summary["corruption_index"] = args.corruption_index
    if intervention_match_trace_path is not None:
        summary["intervention_match_trace_path"] = str(intervention_match_trace_path)
        summary["corruption_selection"] = "intervention-match"
    summary["per_cross_attention"] = bool(args.per_cross_attention)
    summary["clip_source"] = "dataset" if dataset_path is not None else "simulator"
    summary["embedded_clips"] = bool(dataset_path is None)

    if summary_output_path is not None:
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_output_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
