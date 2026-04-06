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
    Path(__file__).resolve().parents[2] / "data" / "source_go.hdf5"
)

_INSTRUCTION_TEMPLATES = [
    "Place a {color} stone on the Go board at row {r}, column {c}.",
    "Put a {color} stone at position ({r}, {c}) on the Go board.",
    "Move the {color} stone to row {r}, column {c} on the board.",
    "Set a {color} stone at ({r}, {c}).",
]

_USE_EMBED_DIM = 512


def _derive_target_from_board_state(board_state: np.ndarray) -> tuple[int, int]:
    """Derive the target (row, col) from the board_state difference.

    board_state has shape (T, 5, 5, 4). Channel 1 encodes black stones,
    channel 2 encodes white stones. We check both to find the newly placed stone.
    """
    for ch in [1, 2]:
        diff = board_state[-1, :, :, ch] - board_state[0, :, :, ch]
        if diff.max() > 0.5:
            idx = np.argmax(diff)
            row, col = divmod(int(idx), board_state.shape[2])
            return row, col
    # Fallback: channel 1
    ch = 1
    diff = board_state[-1, :, :, ch] - board_state[0, :, :, ch]
    idx = np.argmax(diff)
    row, col = divmod(int(idx), board_state.shape[2])
    return row, col


def _extract_action_4d(actions: np.ndarray) -> np.ndarray:
    """Extract 4D actions [dx, dy, dz, gripper] from any source format.

    Handles both legacy 4D (T, 4) and 7D (T, 7) HDF5 formats.
    """
    if actions.shape[1] >= 7:
        # 7D format: [dx,dy,dz, dax,day,daz, gripper] -> [dx,dy,dz, gripper]
        out = np.zeros((actions.shape[0], 4), dtype=np.float32)
        out[:, :3] = actions[:, :3]
        out[:, 3] = actions[:, 6]
        return out
    # Already 4D: [dx, dy, dz, gripper]
    return actions[:, :4].astype(np.float32)


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


def _deduplicate_indices(
    actions_4d: np.ndarray,
    eef_pos: np.ndarray,
    action_norm_thresh: float = 0.02,
    eef_disp_thresh: float = 0.001,
) -> np.ndarray:
    """Return a boolean mask selecting non-redundant frames.

    Always keeps the first and last frame.  A frame is kept if either:
    - Its action XYZ L2 norm exceeds *action_norm_thresh*, OR
    - The EEF displacement from the last kept frame exceeds *eef_disp_thresh* (1 mm).
    """
    n = actions_4d.shape[0]
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True
    last_kept_pos = eef_pos[0].copy()
    for t in range(1, n - 1):
        action_norm = float(np.linalg.norm(actions_4d[t, :3]))
        eef_disp = float(np.linalg.norm(eef_pos[t] - last_kept_pos))
        if action_norm > action_norm_thresh or eef_disp > eef_disp_thresh:
            keep[t] = True
            last_kept_pos = eef_pos[t].copy()
    return keep


class Builder(tfds.core.GeneratorBasedBuilder):
    """TFDS builder for Go VLA demonstrations."""

    VERSION = tfds.core.Version("3.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "2.0.0": "4-DOF actions [dx,dy,dz,gripper] instead of zero-padded 7-DOF.",
        "2.1.0": "8 Hz control, taller stones (12mm), no perturbations, 200 demos.",
        "3.0.0": "8 Hz control, board/lighting randomization, near-duplicate frame removal.",
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

                actions_4d = _extract_action_4d(actions_4)
                # Remap gripper from {0, 1} to {-1, +1} to match LIBERO/OpenVLA convention
                actions_4d[:, 3] = 2.0 * actions_4d[:, 3] - 1.0
                row, col = _derive_target_from_board_state(board_state)

                # Deduplicate near-identical frames (idle/settling segments)
                keep_mask = _deduplicate_indices(actions_4d, eef_pos)
                actions_4d = actions_4d[keep_mask]
                images = images[keep_mask]
                eef_pos = eef_pos[keep_mask]

                stone_color = ep["stone_color"][()].decode() if "stone_color" in ep else "black"

                rng = np.random.RandomState(seed=demo_idx)
                template = _INSTRUCTION_TEMPLATES[
                    rng.randint(len(_INSTRUCTION_TEMPLATES))
                ]
                instruction = template.format(color=stone_color, r=row, c=col)
                embedding = _compute_use_embedding(instruction)

                num_steps = actions_4d.shape[0]
                steps = []
                for t in range(num_steps):
                    is_last = t == num_steps - 1
                    steps.append(
                        {
                            "observation": {
                                "image": images[t],
                                "state": eef_pos[t],
                            },
                            "action": actions_4d[t],
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
