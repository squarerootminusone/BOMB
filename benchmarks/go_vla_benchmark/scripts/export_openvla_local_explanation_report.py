#!/usr/bin/env python3
"""Export human-readable local explanation reports from a saved trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.explainability import (  # noqa: E402
    DATASET_ADAPTERS,
    EpisodeClip,
    export_local_explanation_report,
    load_trace_file,
    resolve_dataset_path,
    resolve_optional_path,
    resolve_repo_relative_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-input", required=True, type=str, help="local explanation .npz trace path")
    parser.add_argument("--output-dir", required=True, type=str, help="directory for exported reports")
    parser.add_argument("--dataset", type=str, default=None, help="optional dataset override; defaults to trace metadata")
    parser.add_argument("--dataset-adapter", choices=sorted(DATASET_ADAPTERS.keys()), default=None)
    parser.add_argument("--demos", type=str, default=None, help="optional comma-separated demo keys")
    parser.add_argument("--top-k", type=int, default=0, help="optional failure-ranked demo limit; <= 0 means all selected demos")
    parser.add_argument("--token-top-k", type=int, default=8, help="number of instruction tokens to surface per table")
    parser.add_argument("--manifest-output", type=str, default=None, help="optional JSON manifest path")
    return parser.parse_args()


def _load_adjusted_clips(args: argparse.Namespace, demo_keys: List[str]):
    trace_bundle = load_trace_file(resolve_repo_relative_path(args.trace_input, repo_root=REPO_ROOT))
    missing = [demo_key for demo_key in demo_keys if demo_key not in trace_bundle.traces]
    if missing:
        raise ValueError(f"requested demos missing from trace data: {missing}")
    dataset_path = resolve_dataset_path(args.dataset, repo_root=REPO_ROOT) if args.dataset else trace_bundle.dataset_path
    dataset_adapter_name = args.dataset_adapter or trace_bundle.dataset_adapter or "go-hdf5"
    dataset_adapter = DATASET_ADAPTERS[dataset_adapter_name]()

    requested_demos = demo_keys if demo_keys else trace_bundle.demo_keys
    clips = dataset_adapter.load_episode_clips(
        dataset_path=dataset_path,
        demos=",".join(requested_demos),
        start=0,
        num_demos=0,
        stride=1,
        max_steps=0,
    )

    adjusted_clips: List[EpisodeClip] = []
    for clip in clips:
        frame_indices = trace_bundle.frame_indices_by_key[clip.demo_key]
        if frame_indices.size == 0:
            continue
        frame_index_to_pos = {int(frame_idx): pos for pos, frame_idx in enumerate(np.asarray(clip.frame_indices, dtype=np.int32).tolist())}
        selected_positions = np.asarray([frame_index_to_pos[int(frame_idx)] for frame_idx in frame_indices], dtype=np.int32)
        adjusted_clips.append(
            EpisodeClip(
                demo_key=clip.demo_key,
                instruction=clip.instruction,
                images=clip.images[selected_positions],
                gt_actions=clip.gt_actions[selected_positions],
                frame_indices=np.asarray(frame_indices, dtype=np.int32),
            )
        )
    return trace_bundle, adjusted_clips


def main() -> None:
    args = parse_args()
    output_dir = resolve_repo_relative_path(args.output_dir, repo_root=REPO_ROOT)
    manifest_path = resolve_optional_path(args.manifest_output, repo_root=REPO_ROOT)

    requested_demo_keys = [item.strip() for item in (args.demos or "").split(",") if item.strip()]
    trace_bundle, adjusted_clips = _load_adjusted_clips(args, requested_demo_keys)
    trace_order = [clip.demo_key for clip in adjusted_clips]
    selected_demo_keys = requested_demo_keys if requested_demo_keys else trace_order
    clip_key_set = set(trace_order)
    missing_clip_keys = [demo_key for demo_key in selected_demo_keys if demo_key not in clip_key_set]
    if missing_clip_keys:
        raise RuntimeError(f"failed to load dataset frames for demos: {missing_clip_keys}")
    selected_traces = [trace_bundle.traces[demo_key] for demo_key in selected_demo_keys]

    if args.top_k > 0 and not requested_demo_keys:
        ranked = sorted(selected_traces, key=lambda item: item.score, reverse=True)
        selected_traces = ranked[: int(args.top_k)]
        selected_keys = {trace.demo_key for trace in selected_traces}
        adjusted_clips = [clip for clip in adjusted_clips if clip.demo_key in selected_keys]

    manifest = export_local_explanation_report(
        output_dir=output_dir,
        clips=adjusted_clips,
        traces=selected_traces,
        token_top_k=args.token_top_k,
        manifest_path=manifest_path,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
