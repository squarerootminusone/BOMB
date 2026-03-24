#!/usr/bin/env python3
"""Collect reusable local explanations for OpenVLA checkpoints on Go HDF5 demos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))
sys.path.insert(0, str(REPO_ROOT / "openvla"))

from go_vla_benchmark.explainability import (  # noqa: E402
    DATASET_ADAPTERS,
    MODEL_ADAPTERS,
    collect_episode_traces,
    resolve_dataset_path,
    resolve_optional_path,
    save_trace_file,
    trace_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="Go benchmark HDF5 dataset")
    parser.add_argument("--checkpoint", required=True, type=str, help="OpenVLA checkpoint path or HF id")
    parser.add_argument("--trace-output", required=True, type=str, help="output .npz trace path")
    parser.add_argument("--summary-output", type=str, default=None, help="optional summary JSON path")
    parser.add_argument("--dataset-adapter", choices=sorted(DATASET_ADAPTERS.keys()), default="go-hdf5")
    parser.add_argument("--model-adapter", choices=sorted(MODEL_ADAPTERS.keys()), default="openvla")
    parser.add_argument("--prompt-style", choices=["openvla", "openvla-v01"], default=None)
    parser.add_argument("--unnorm-key", type=str, default=None, help="dataset statistics key for de-normalizing actions")
    parser.add_argument("--action-dim", type=int, default=None, help="fallback action dimension when norm stats are absent")
    parser.add_argument("--start", type=int, default=0, help="start demo index in sorted order")
    parser.add_argument("--num-demos", type=int, default=0, help="number of demos to process; <= 0 means all")
    parser.add_argument("--demos", type=str, default=None, help="optional comma-separated demo keys")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="extra frame stride after RLDS-style subsampling/filtering",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="max timesteps per demo after applying stride")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset, repo_root=REPO_ROOT)
    trace_output_path = resolve_optional_path(args.trace_output, repo_root=REPO_ROOT)
    summary_output_path = resolve_optional_path(args.summary_output, repo_root=REPO_ROOT)

    dataset_adapter = DATASET_ADAPTERS[args.dataset_adapter]()
    clips = dataset_adapter.load_episode_clips(
        dataset_path=dataset_path,
        demos=args.demos,
        start=args.start,
        num_demos=args.num_demos,
        stride=args.stride,
        max_steps=args.max_steps,
    )
    if not clips:
        raise RuntimeError("no demos selected from dataset")

    model_adapter = MODEL_ADAPTERS[args.model_adapter](
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
    traces = collect_episode_traces(clips=clips, model_adapter=model_adapter)

    save_trace_file(
        trace_path=trace_output_path,
        dataset_path=dataset_path,
        clips=clips,
        traces=traces,
        checkpoint=args.checkpoint,
        prompt_style=model_adapter.prompt_style,
        dataset_adapter=args.dataset_adapter,
        model_adapter=args.model_adapter,
    )

    summary = trace_manifest(
        dataset_path=dataset_path,
        checkpoint=args.checkpoint,
        prompt_style=model_adapter.prompt_style,
        dataset_adapter=args.dataset_adapter,
        model_adapter=args.model_adapter,
        clips=clips,
        traces=traces,
    )
    summary["trace_path"] = str(trace_output_path)

    if summary_output_path is not None:
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_output_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
