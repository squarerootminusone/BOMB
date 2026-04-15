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
with 4-DoF actions (3 position + 1 gripper).
"""

_CITATION = ""

_DEFAULT_HDF5_PATH = str(
    Path(__file__).resolve().parents[2] / "data" / "datasets" / "source_go.hdf5"
)

_INSTRUCTION_TEMPLATES = [
    "Place a black stone on the Go board at row {r}, column {c}.",
    "Put a black stone at position ({r}, {c}) on the Go board.",
    "Move the black stone to row {r}, column {c} on the board.",
    "Set a black stone at ({r}, {c}).",
]

_USE_EMBED_DIM = 512

# Downsample from 20 Hz to ~5 Hz (keep every 4th step)
_SUBSAMPLE_STRIDE = 4

# No-op filter: drop steps where the action norm (xyz) is below this threshold
_NOOP_THRESHOLD = 1e-4


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

    VERSION = tfds.core.Version("2.1.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.1.0": "Downsample to ~5 Hz and filter no-op actions.",
        "2.0.0": "4-DOF actions [dx,dy,dz,gripper] instead of zero-padded 7-DOF.",
        "2.1.0": "8 Hz control, taller stones (12mm), no perturbations, 200 demos.",
    }

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

                actions = actions_4  # 4-DOF: [dx, dy, dz, gripper]
                row, col = _derive_target_from_board_state(board_state)

                rng = np.random.RandomState(seed=demo_idx)
                template = _INSTRUCTION_TEMPLATES[
                    rng.randint(len(_INSTRUCTION_TEMPLATES))
                ]
                instruction = template.format(r=row, c=col)
                embedding = _compute_use_embedding(instruction)

                # Filter no-op actions
                keep = []
                for t in range(actions.shape[0]):
                    if np.linalg.norm(actions[t, :3]) >= _NOOP_THRESHOLD:
                        keep.append(t)
                # Always keep last step for terminal signal
                last_t = actions.shape[0] - 1
                if last_t not in keep:
                    keep.append(last_t)

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
                            "action": actions[t],
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
