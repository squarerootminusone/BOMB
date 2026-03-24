"""TFDS builder that converts Go VLA HDF5 demonstrations to RLDS format."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

try:
    from go_vla_benchmark.rlds_preprocessing import (
        RLDS_NOOP_THRESHOLD,
        RLDS_SUBSAMPLE_STRIDE,
        build_instruction,
        compute_rlds_keep_indices,
        extract_action_4d,
        remap_gripper_to_openvla,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from go_vla_benchmark.rlds_preprocessing import (
        RLDS_NOOP_THRESHOLD,
        RLDS_SUBSAMPLE_STRIDE,
        build_instruction,
        compute_rlds_keep_indices,
        extract_action_4d,
        remap_gripper_to_openvla,
    )

_DESCRIPTION = """\
Go VLA benchmark demonstrations for fine-tuning vision-language-action models.
Each episode is a single stone placement on a 5x5 Go board using a robot arm
with 4-DoF actions (3 position + 1 gripper).
"""

_CITATION = ""

_DEFAULT_HDF5_PATH = str(
    Path(__file__).resolve().parents[2] / "data" / "source_go.hdf5"
)

_USE_EMBED_DIM = 512

def _compute_use_embedding(text: str) -> np.ndarray:
    """Compute USE-Large/5 embedding for a string, or return zeros."""
    try:
        import tensorflow_hub as hub
        _compute_use_embedding._model = getattr(
            _compute_use_embedding, "_model", None
        ) or hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )
        embed = _compute_use_embedding._model([text]).numpy()[0]
        return embed.astype(np.float32)
    except Exception:
        return np.zeros(_USE_EMBED_DIM, dtype=np.float32)


class Builder(tfds.core.GeneratorBasedBuilder):
    """TFDS builder for Go VLA demonstrations."""

    VERSION = tfds.core.Version("3.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release (7-DoF padded actions, 20 Hz).",
        "1.1.0": "Downsample to ~5 Hz and filter no-op actions.",
        "2.0.0": "Native 4-DoF actions (dx,dy,dz,gripper), ~5 Hz, no-op filtered.",
        "3.0.0": "Gripper {-1,+1}, stone color support, dynamic image resolution, both-color board detection.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        # Read image resolution from the HDF5 to avoid hardcoding.
        hdf5_path = os.environ.get("GO_VLA_HDF5_PATH", _DEFAULT_HDF5_PATH)
        try:
            with h5py.File(hdf5_path, "r") as f:
                first_key = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[0]
                img_shape = tuple(f[f"data/{first_key}/obs/agentview_image"].shape[1:])
        except Exception:
            img_shape = (256, 256, 3)

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=img_shape,
                                        dtype=np.uint8,
                                        encoding_format="png",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(3,), dtype=np.float32
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(4,), dtype=np.float32
                            ),
                            "reward": np.float32,
                            "discount": np.float32,
                            "is_first": np.bool_,
                            "is_last": np.bool_,
                            "is_terminal": np.bool_,
                            "language_instruction": tfds.features.Text(),
                            "language_embedding": tfds.features.Tensor(
                                shape=(_USE_EMBED_DIM,), dtype=np.float32
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(),
                        }
                    ),
                }
            ),
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/anthropics/dsait4125",
        )

    def _split_generators(self, dl_manager):
        hdf5_path = os.environ.get("GO_VLA_HDF5_PATH", _DEFAULT_HDF5_PATH)
        return {
            "train": self._generate_examples(hdf5_path),
        }

    def _generate_examples(self, hdf5_path: str):
        """Yield (key, episode_dict) for each demo in the HDF5 file."""
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(
                f["data"].keys(), key=lambda k: int(k.split("_")[1])
            )
            for demo_idx, demo_key in enumerate(demo_keys):
                ep = f[f"data/{demo_key}"]
                actions_4 = np.asarray(ep["actions"], dtype=np.float32)
                images = np.asarray(ep["obs/agentview_image"], dtype=np.uint8)
                eef_pos = np.asarray(ep["obs/eef_pos"], dtype=np.float32)
                board_state = np.asarray(
                    ep["obs/board_state"], dtype=np.float32
                )

                actions_4dof = remap_gripper_to_openvla(extract_action_4d(actions_4))

                stone_color = ep["stone_color"][()].decode() if "stone_color" in ep else "black"
                instruction = build_instruction(
                    board_state=board_state,
                    demo_idx=demo_idx,
                    stone_color=stone_color,
                )
                embedding = _compute_use_embedding(instruction)

                # Downsample to ~5 Hz and filter no-op actions
                keep = compute_rlds_keep_indices(
                    actions_4dof,
                    subsample_stride=RLDS_SUBSAMPLE_STRIDE,
                    noop_threshold=RLDS_NOOP_THRESHOLD,
                ).tolist()

                num_steps = len(keep)
                steps = []
                for i, t in enumerate(keep):
                    is_last = i == num_steps - 1
                    steps.append(
                        {
                            "observation": {
                                "image": images[t],
                                "state": eef_pos[t],
                            },
                            "action": actions_4dof[t],
                            "reward": 1.0 if is_last else 0.0,
                            "discount": 1.0,
                            "is_first": i == 0,
                            "is_last": is_last,
                            "is_terminal": is_last,
                            "language_instruction": instruction,
                            "language_embedding": embedding,
                        }
                    )

                yield demo_key, {
                    "steps": steps,
                    "episode_metadata": {"file_path": hdf5_path},
                }
