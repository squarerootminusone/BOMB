"""
Transforms for the Go VLA dataset (4-DoF pick-and-place on a 5x5 Go board).

Why: pi0 expects inputs in a specific dict format with image slots, state,
actions, and prompt. The Go VLA dataset has a single agentview camera,
3D eef_pos state, and 4-DoF delta actions [dx, dy, dz, gripper].

How: GoVLAInputs maps dataset keys into the pi0 input schema, padding
unused image slots (wrist cameras) with zeros and masking them out.
GoVLAOutputs slices the model output back to the 4-DoF action space.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_go_vla_example() -> dict:
    """Creates a random input example for the Go VLA policy."""
    return {
        "observation/image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/state": np.random.rand(3).astype(np.float32),
        "prompt": "place a black stone on the board",
    }


def _parse_image(image) -> np.ndarray:
    """Normalize image to uint8 (H, W, C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class GoVLAInputs(transforms.DataTransformFn):
    """
    Convert Go VLA dataset inputs to the pi0 model input format.

    Why: pi0 expects a fixed set of image keys (base, left_wrist, right_wrist),
    state vector, actions, and prompt. Our dataset only has a single agentview
    camera and 3D state.

    How: Maps the agentview image to base_0_rgb, pads wrist image slots with
    zeros, and masks them appropriately (True for pi0-FAST to avoid issues,
    False for pi0 to indicate missing data).
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                # No wrist cameras in Go VLA -- pad with zeros.
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # Mask out non-existent wrist images for pi0; keep True for pi0-FAST.
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class GoVLAOutputs(transforms.DataTransformFn):
    """
    Convert pi0 model outputs back to the Go VLA 4-DoF action space.

    Why: pi0 pads actions to its internal action_dim (e.g. 32). We need to
    extract only the first 4 dimensions that correspond to [dx, dy, dz, gripper].

    How: Slices the action tensor along the last axis to keep only 4 dims.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :4])}
