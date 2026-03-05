"""TFDS builder that converts Go VLA HDF5 demonstrations to RLDS format."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """\
Go VLA benchmark demonstrations for fine-tuning vision-language-action models.
Each episode is a single stone placement on a 5x5 Go board using a robot arm
with OSC position control (4-DoF actions padded to 7-DoF).
"""

_CITATION = ""

_DEFAULT_HDF5_PATH = str(
    Path(__file__).resolve().parents[2] / "data" / "source_go.hdf5"
)

_INSTRUCTION_TEMPLATES = [
    "Place a black stone on the Go board at row {r}, column {c}.",
    "Put a black stone at position ({r}, {c}) on the Go board.",
    "Move the black stone to row {r}, column {c} on the board.",
    "Set a black stone at ({r}, {c}).",
]

_USE_EMBED_DIM = 512


def _derive_target_from_board_state(board_state: np.ndarray) -> tuple[int, int]:
    """Derive the target (row, col) from the board_state difference.

    board_state has shape (T, 5, 5, 4). Channel 1 encodes black stones.
    We compare the first and last timestep to find the newly placed stone.
    """
    ch = 1  # black stone channel
    diff = board_state[-1, :, :, ch] - board_state[0, :, :, ch]
    # Find the cell with the largest positive change
    idx = np.argmax(diff)
    row, col = divmod(int(idx), board_state.shape[2])
    return row, col


def _pad_action_4to7(actions: np.ndarray) -> np.ndarray:
    """Pad (T, 4) actions to (T, 7): [dx,dy,dz, 0,0,0, gripper]."""
    t = actions.shape[0]
    padded = np.zeros((t, 7), dtype=np.float32)
    padded[:, :3] = actions[:, :3]      # xyz deltas
    # dims 3-5 stay zero (rotation deltas)
    padded[:, 6] = actions[:, 3]        # gripper
    return padded


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

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(3,), dtype=np.float32
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,), dtype=np.float32
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

                actions_7 = _pad_action_4to7(actions_4)
                row, col = _derive_target_from_board_state(board_state)

                rng = np.random.RandomState(seed=demo_idx)
                template = _INSTRUCTION_TEMPLATES[
                    rng.randint(len(_INSTRUCTION_TEMPLATES))
                ]
                instruction = template.format(r=row, c=col)
                embedding = _compute_use_embedding(instruction)

                num_steps = actions_7.shape[0]
                steps = []
                for t in range(num_steps):
                    is_last = t == num_steps - 1
                    steps.append(
                        {
                            "observation": {
                                "image": images[t],
                                "state": eef_pos[t],
                            },
                            "action": actions_7[t],
                            "reward": 1.0 if is_last else 0.0,
                            "discount": 1.0,
                            "is_first": t == 0,
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
