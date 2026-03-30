"""
libero_dataset.py

PyTorch Dataset that reads LIBERO HDF5 demonstration files directly (no RLDS conversion needed).
Follows the DummyDataset pattern from datasets.py for compatibility with OpenVLA's training pipeline.

Supports two HDF5 formats:
    1. Regenerated format (from regenerate_libero_dataset.py):
       data/demo_{i}/obs/agentview_rgb  (T, 256, 256, 3)
       data/demo_{i}/actions            (T, 7)

    2. Original LIBERO format:
       data/demo_{i}/obs/agentview_image (T, H, W, 3)
       data/demo_{i}/actions             (T, 7)

Images are rotated 180 degrees to match the RLDS conversion convention used during OpenVLA training
(see regenerate_libero_dataset.py lines 8-9 and libero_utils.py line 56).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class LIBERODataset(Dataset):
    """
    PyTorch Dataset for LIBERO HDF5 demonstrations.

    Scans a directory for HDF5 files, indexes all (file, demo, timestep) triples,
    and returns tokenized (image, action, instruction) tuples compatible with
    OpenVLA's PaddedCollatorForActionPrediction.

    Args:
        data_dir: Path to directory containing HDF5 files (e.g., LIBERO/libero/datasets/libero_spatial/)
        action_tokenizer: ActionTokenizer instance for discretizing actions
        base_tokenizer: HuggingFace tokenizer for text tokenization
        image_transform: Image transform from the VLA processor
        prompt_builder_fn: Prompt builder class (PurePromptBuilder for OpenVLA v0.2+)
        task_suite_name: Name of the LIBERO task suite (used as dataset key in statistics)
        image_aug: Whether to apply image augmentations (handled externally)
        predict_stop_token: Whether to predict stop token in labels
    """

    def __init__(
        self,
        data_dir: str,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        task_suite_name: str = "libero_spatial",
        predict_stop_token: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.task_suite_name = task_suite_name
        self.predict_stop_token = predict_stop_token

        # Scan for HDF5 files
        self.hdf5_files = sorted(self.data_dir.glob("*.hdf5"))
        assert len(self.hdf5_files) > 0, f"No HDF5 files found in {self.data_dir}"

        # Build index: list of (file_path, demo_key, timestep, task_description, num_timesteps)
        self._index: List[Tuple[Path, str, int, str, int]] = []
        self._all_actions: List[np.ndarray] = []

        for hdf5_path in self.hdf5_files:
            # Extract task description from filename: "{task_name}_demo.hdf5"
            task_name = hdf5_path.stem.replace("_demo", "")
            task_description = task_name.replace("_", " ")

            with h5py.File(hdf5_path, "r") as f:
                data_group = f["data"]
                for demo_key in sorted(data_group.keys()):
                    demo = data_group[demo_key]
                    actions = demo["actions"][()]
                    num_timesteps = len(actions)
                    self._all_actions.append(actions)
                    for t in range(num_timesteps):
                        self._index.append((hdf5_path, demo_key, t, task_description, num_timesteps))

        # Compute dataset statistics (q01/q99 per action dimension) for normalization
        all_actions = np.concatenate(self._all_actions, axis=0)  # (N, 7)
        q01 = np.percentile(all_actions, 1, axis=0).astype(np.float32)
        q99 = np.percentile(all_actions, 99, axis=0).astype(np.float32)

        self.dataset_statistics = {
            self.task_suite_name: {
                "action": {
                    "q01": q01.tolist(),
                    "q99": q99.tolist(),
                    "mean": all_actions.mean(axis=0).astype(np.float32).tolist(),
                    "std": all_actions.std(axis=0).astype(np.float32).tolist(),
                }
            }
        }

        # Free memory
        del self._all_actions

        print(f"[LIBERODataset] Loaded {len(self._index)} timesteps from {len(self.hdf5_files)} HDF5 files")
        print(f"[LIBERODataset] Action stats q01={q01}, q99={q99}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        hdf5_path, demo_key, timestep, task_description, _ = self._index[idx]

        # Read image and action from HDF5
        with h5py.File(hdf5_path, "r") as f:
            demo = f["data"][demo_key]

            # Try regenerated format first, then original format
            if "obs" in demo:
                obs = demo["obs"]
                if "agentview_rgb" in obs:
                    img_data = obs["agentview_rgb"][timestep]  # (H, W, 3)
                elif "agentview_image" in obs:
                    img_data = obs["agentview_image"][timestep]
                else:
                    raise KeyError(f"No image key found in {hdf5_path}/{demo_key}/obs/")
            else:
                raise KeyError(f"No 'obs' group found in {hdf5_path}/{demo_key}/")

            action = demo["actions"][timestep]  # (7,)

        # Rotate image 180 degrees to match RLDS conversion convention
        # (LIBERO environments return images that appear upside down)
        img_data = img_data[::-1, ::-1].copy()

        # Convert to PIL Image
        image = Image.fromarray(img_data.astype(np.uint8))

        # Normalize action to [-1, 1] using dataset q01/q99 statistics
        stats = self.dataset_statistics[self.task_suite_name]["action"]
        q01 = np.array(stats["q01"], dtype=np.float32)
        q99 = np.array(stats["q99"], dtype=np.float32)
        action = action.astype(np.float32)

        # Normalize: map [q01, q99] -> [-1, 1]
        mask = q99 - q01 > 1e-6  # Avoid division by zero
        action_normalized = np.where(
            mask,
            2.0 * (action - q01) / (q99 - q01) - 1.0,
            action,
        )
        action_normalized = np.clip(action_normalized, -1.0, 1.0)

        # Build prompt using the same pattern as DummyDataset / RLDSBatchTransform
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {task_description.lower()}?"},
            {"from": "gpt", "value": self.action_tokenizer(action_normalized)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # Only compute loss on action tokens (matching DummyDataset pattern)
        labels[: -(len(action_normalized) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
