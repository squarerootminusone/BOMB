#!/usr/bin/env python3
"""Inspect raw HDF5 gripper labels and their OpenVLA conversion.

This script is meant to answer:
1. What gripper values are actually stored in the source HDF5?
2. Do older demos contain legacy `0.0` open labels?
3. What will those raw values become after the Go RLDS builder standardizes
   them to OpenVLA convention (`1=open`, `0=close`)?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]


def _extract_action_4d(actions: np.ndarray) -> np.ndarray:
    """Extract `[dx, dy, dz, gripper]` from either 4D or 7D stored actions."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"expected action array with shape (T, D), got {actions.shape}")
    if actions.shape[1] >= 7:
        out = np.zeros((actions.shape[0], 4), dtype=np.float32)
        out[:, :3] = actions[:, :3]
        out[:, 3] = actions[:, 6]
        return out
    return actions[:, :4].astype(np.float32)


def _raw_to_openvla(raw_gripper: np.ndarray) -> np.ndarray:
    """Mirror the RLDS builder conversion: non-positive => open, positive => close."""
    raw_gripper = np.asarray(raw_gripper, dtype=np.float32)
    return np.where(raw_gripper <= 0.0, 1.0, 0.0).astype(np.float32)


def _sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    demos = list(data_group.keys())
    order = np.argsort([int(name.split("_")[1]) for name in demos])
    return [demos[i] for i in order]


def _counts(values: np.ndarray) -> dict[str, int]:
    uniques, counts = np.unique(values, return_counts=True)
    return {f"{float(v):.3f}": int(c) for v, c in zip(uniques, counts)}


def _transition_indices(values: np.ndarray) -> list[int]:
    values = np.asarray(values).reshape(-1)
    if values.size == 0:
        return []
    return [int(i) for i in np.nonzero(values[1:] != values[:-1])[0] + 1]


def _summarize_demo(actions_4d: np.ndarray) -> dict[str, object]:
    raw_gripper = actions_4d[:, 3]
    openvla_gripper = _raw_to_openvla(raw_gripper)
    return {
        "steps": int(actions_4d.shape[0]),
        "raw_gripper_min": float(raw_gripper.min()) if raw_gripper.size else None,
        "raw_gripper_max": float(raw_gripper.max()) if raw_gripper.size else None,
        "raw_gripper_counts": _counts(raw_gripper),
        "legacy_zero_present": bool(np.any(np.isclose(raw_gripper, 0.0))),
        "openvla_gripper_counts": _counts(openvla_gripper),
        "raw_transition_indices": _transition_indices(raw_gripper),
        "openvla_transition_indices": _transition_indices(openvla_gripper),
        "raw_head": raw_gripper[:20].astype(float).tolist(),
        "raw_tail": raw_gripper[-20:].astype(float).tolist(),
        "openvla_head": openvla_gripper[:20].astype(float).tolist(),
        "openvla_tail": openvla_gripper[-20:].astype(float).tolist(),
    }


def _iter_selected_demos(all_demos: list[str], requested: str | None, limit: int | None) -> Iterable[str]:
    if requested:
        wanted = {name.strip() for name in requested.split(",") if name.strip()}
        return [name for name in all_demos if name in wanted]
    if limit is None or limit <= 0:
        return all_demos
    return all_demos[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect gripper values in a Go benchmark HDF5 dataset.")
    parser.add_argument("--dataset", required=True, type=str, help="Path to HDF5 dataset.")
    parser.add_argument(
        "--demos",
        type=str,
        default=None,
        help="Comma-separated demo keys to inspect (e.g. demo_0,demo_7).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many demos to summarize when --demos is not provided. Use <= 0 for all.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full summary as JSON instead of human-readable text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()

    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise RuntimeError(f"{dataset_path} is missing the 'data' group")

        all_demos = _sorted_demo_keys(f["data"])
        selected = list(_iter_selected_demos(all_demos, args.demos, args.limit))

        per_demo: dict[str, dict[str, object]] = {}
        all_raw_gripper: list[np.ndarray] = []
        for demo_key in selected:
            actions_4d = _extract_action_4d(np.asarray(f[f"data/{demo_key}/actions"], dtype=np.float32))
            per_demo[demo_key] = _summarize_demo(actions_4d)
            all_raw_gripper.append(actions_4d[:, 3])

        summary: dict[str, object] = {
            "dataset": str(dataset_path),
            "selected_demos": selected,
            "num_selected_demos": len(selected),
            "conversion_rule": "raw gripper <= 0.0 -> OpenVLA 1.0 (open), raw gripper > 0.0 -> OpenVLA 0.0 (close)",
            "per_demo": per_demo,
        }

        if all_raw_gripper:
            raw_concat = np.concatenate(all_raw_gripper, axis=0)
            openvla_concat = _raw_to_openvla(raw_concat)
            summary["overall"] = {
                "total_steps": int(raw_concat.shape[0]),
                "raw_gripper_counts": _counts(raw_concat),
                "legacy_zero_present": bool(np.any(np.isclose(raw_concat, 0.0))),
                "openvla_gripper_counts": _counts(openvla_concat),
            }

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Dataset: {summary['dataset']}")
    print(f"Selected demos ({summary['num_selected_demos']}): {', '.join(selected)}")
    print(f"Conversion rule: {summary['conversion_rule']}")

    overall = summary.get("overall")
    if isinstance(overall, dict):
        print("\nOverall:")
        print(f"  total_steps          : {overall['total_steps']}")
        print(f"  raw_gripper_counts   : {overall['raw_gripper_counts']}")
        print(f"  legacy_zero_present  : {overall['legacy_zero_present']}")
        print(f"  openvla_counts       : {overall['openvla_gripper_counts']}")

    print("\nPer demo:")
    for demo_key in selected:
        demo = per_demo[demo_key]
        print(f"  {demo_key}:")
        print(f"    steps                 : {demo['steps']}")
        print(f"    raw_gripper_counts    : {demo['raw_gripper_counts']}")
        print(f"    legacy_zero_present   : {demo['legacy_zero_present']}")
        print(f"    openvla_counts        : {demo['openvla_gripper_counts']}")
        print(f"    raw_transition_idxs   : {demo['raw_transition_indices'][:20]}")
        print(f"    openvla_transition_idxs: {demo['openvla_transition_indices'][:20]}")
        print(f"    raw_head              : {demo['raw_head']}")
        print(f"    raw_tail              : {demo['raw_tail']}")
        print(f"    openvla_head          : {demo['openvla_head']}")
        print(f"    openvla_tail          : {demo['openvla_tail']}")


if __name__ == "__main__":
    main()
