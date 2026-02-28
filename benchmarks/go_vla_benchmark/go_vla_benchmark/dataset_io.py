"""Dataset IO helpers for MimicGen-compatible HDF5 files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np


@dataclass
class EpisodeRecord:
    actions: np.ndarray
    states: np.ndarray
    observations: Dict[str, np.ndarray]
    datagen_infos: List[Any]
    initial_state: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, np.ndarray] = field(default_factory=dict)


def _to_dict_of_arrays(seq: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert list[dict] recursively into dict[np.ndarray|dict]."""
    result: Dict[str, Any] = {}
    for key in seq[0].keys():
        values = [item[key] for item in seq]
        first_value = values[0]
        if isinstance(first_value, dict):
            result[key] = _to_dict_of_arrays(values)
        else:
            result[key] = np.asarray(values)
    return result


def _write_datagen_info(ep_grp: h5py.Group, datagen_infos: List[Any]) -> None:
    raw = [x.to_dict() for x in datagen_infos]
    packed = _to_dict_of_arrays(raw)

    ep_grp.create_dataset("datagen_info/eef_pose", data=np.asarray(packed["eef_pose"]))
    ep_grp.create_dataset("datagen_info/target_pose", data=np.asarray(packed["target_pose"]))
    ep_grp.create_dataset("datagen_info/gripper_action", data=np.asarray(packed["gripper_action"]))

    object_poses = packed["object_poses"]
    for obj_name, obj_pose in object_poses.items():
        ep_grp.create_dataset(f"datagen_info/object_poses/{obj_name}", data=np.asarray(obj_pose))

    term_signals = packed["subtask_term_signals"]
    for signal_name, signal_values in term_signals.items():
        ep_grp.create_dataset(
            f"datagen_info/subtask_term_signals/{signal_name}",
            data=np.asarray(signal_values, dtype=np.int32),
        )


def write_dataset(
    output_path: str | Path,
    episodes: List[EpisodeRecord],
    env_meta: Dict[str, Any],
    env_interface_name: str,
    env_interface_type: str,
) -> None:
    """Write episodes to a MimicGen-compatible HDF5 file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        total = 0

        for i, ep in enumerate(episodes):
            ep_name = f"demo_{i}"
            ep_grp = data_grp.create_group(ep_name)

            actions = np.asarray(ep.actions, dtype=np.float32)
            states = np.asarray(ep.states, dtype=np.float32)

            ep_grp.create_dataset("actions", data=actions)
            ep_grp.create_dataset("states", data=states)

            for obs_key, obs_array in ep.observations.items():
                obs_np = np.asarray(obs_array)
                if obs_np.ndim >= 3:
                    ep_grp.create_dataset(f"obs/{obs_key}", data=obs_np, compression="gzip")
                else:
                    ep_grp.create_dataset(f"obs/{obs_key}", data=obs_np)

            _write_datagen_info(ep_grp=ep_grp, datagen_infos=ep.datagen_infos)

            # MimicGen expects these attrs when inferring env interface from source data.
            ep_grp["datagen_info"].attrs["env_interface_name"] = env_interface_name
            ep_grp["datagen_info"].attrs["env_interface_type"] = env_interface_type

            for extra_key, extra_value in ep.extras.items():
                ep_grp.create_dataset(extra_key, data=np.asarray(extra_value))

            if "model" in ep.initial_state and ep.initial_state["model"] is not None:
                ep_grp.attrs["model_file"] = ep.initial_state["model"]

            ep_grp.attrs["num_samples"] = int(actions.shape[0])
            total += int(actions.shape[0])

        data_grp.attrs["total"] = total
        data_grp.attrs["env_args"] = json.dumps(env_meta, indent=4)


def read_demo_keys(dataset_path: str | Path) -> List[str]:
    """Get demo keys sorted by index from a MimicGen / robomimic hdf5."""
    with h5py.File(dataset_path, "r") as f:
        demos = list(f["data"].keys())
    order = np.argsort([int(x.split("_")[1]) for x in demos])
    return [demos[i] for i in order]


def summarize_dataset(dataset_path: str | Path) -> Dict[str, Any]:
    """Return lightweight summary stats for a generated dataset."""
    dataset_path = Path(dataset_path)
    with h5py.File(dataset_path, "r") as f:
        demos = list(f["data"].keys())
        order = np.argsort([int(x.split("_")[1]) for x in demos])
        demos = [demos[i] for i in order]
        first = f[f"data/{demos[0]}"] if demos else None

        summary: Dict[str, Any] = {
            "dataset_path": str(dataset_path),
            "num_demos": len(demos),
            "total_samples": int(f["data"].attrs.get("total", 0)),
            "demo_keys": demos,
        }

        if first is not None:
            summary["first_demo"] = {
                "num_samples": int(first.attrs.get("num_samples", 0)),
                "action_shape": tuple(first["actions"].shape),
                "state_shape": tuple(first["states"].shape),
                "obs_keys": list(first["obs"].keys()) if "obs" in first else [],
                "datagen_object_keys": list(first["datagen_info/object_poses"].keys()),
                "datagen_signal_keys": list(first["datagen_info/subtask_term_signals"].keys()),
            }

        if demos:
            lengths: list[int] = []
            action_abs_means: list[np.ndarray] = []
            eef_path_lengths: list[float] = []
            eef_start_to_end: list[float] = []

            for demo_key in demos:
                ep = f[f"data/{demo_key}"]
                actions = np.asarray(ep["actions"])
                lengths.append(int(actions.shape[0]))

                if actions.shape[0] > 0:
                    action_abs_means.append(np.mean(np.abs(actions[:, :3]), axis=0))

                if ("obs" in ep) and ("eef_pos" in ep["obs"]):
                    eef = np.asarray(ep["obs/eef_pos"])
                    if eef.shape[0] > 1:
                        eef_path_lengths.append(float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum()))
                    if eef.shape[0] > 0:
                        eef_start_to_end.append(float(np.linalg.norm(eef[-1] - eef[0])))

            motion_stats: Dict[str, Any] = {
                "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
                "episode_length_min": int(np.min(lengths)) if lengths else 0,
                "episode_length_max": int(np.max(lengths)) if lengths else 0,
            }
            if action_abs_means:
                stacked_abs = np.vstack(action_abs_means)
                motion_stats["mean_abs_action_xyz"] = stacked_abs.mean(axis=0).tolist()
            if eef_path_lengths:
                motion_stats["eef_path_length_mean"] = float(np.mean(eef_path_lengths))
                motion_stats["eef_path_length_min"] = float(np.min(eef_path_lengths))
                motion_stats["eef_path_length_max"] = float(np.max(eef_path_lengths))
            if eef_start_to_end:
                motion_stats["eef_start_to_end_mean"] = float(np.mean(eef_start_to_end))
                motion_stats["eef_start_to_end_min"] = float(np.min(eef_start_to_end))
                motion_stats["eef_start_to_end_max"] = float(np.max(eef_start_to_end))

            summary["motion_stats"] = motion_stats

        return summary
