"""
Convert the Go VLA RLDS dataset to LeRobot format for pi0 fine-tuning.

Why: openpi uses LeRobot format for small datasets, not RLDS directly.
The Go VLA dataset lives in ~/tensorflow_datasets/go_vla_dataset/4.0.0/
and contains 4-DoF delta actions [dx, dy, dz, gripper], 256x256 agentview
images, and 3D eef_pos state.

How: Reads the RLDS tfrecord dataset via tensorflow_datasets, iterates
over episodes and steps, and writes each frame into a LeRobot dataset
with the correct feature schema.

Usage:
    uv run examples/go_vla/convert_go_vla_to_lerobot.py \
        --data-dir ~/tensorflow_datasets

If you want to push to the Hugging Face Hub:
    uv run examples/go_vla/convert_go_vla_to_lerobot.py \
        --data-dir ~/tensorflow_datasets --push-to-hub

Note: requires tensorflow_datasets:
    uv pip install tensorflow tensorflow_datasets
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "go_vla/go_vla_dataset"
RAW_DATASET_NAME = "go_vla_dataset"


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset with features matching our Go VLA data.
    # openpi expects proprio in "state" and actions in "actions".
    # We have a single agentview camera (no wrist image).
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Load the RLDS dataset and iterate over episodes.
    # RLDS structure (from features.json):
    #   steps/observation/image: uint8 (256, 256, 3)
    #   steps/observation/state: float32 (3,)  -- eef_pos
    #   steps/action: float32 (4,)  -- [dx, dy, dz, gripper]
    #   steps/language_instruction: string
    raw_dataset = tfds.load(RAW_DATASET_NAME, data_dir=data_dir, split="train")

    episode_count = 0
    for episode in raw_dataset:
        for step in episode["steps"].as_numpy_iterator():
            dataset.add_frame(
                {
                    "image": step["observation"]["image"],
                    "state": step["observation"]["state"],
                    "actions": step["action"],
                    "task": step["language_instruction"].decode(),
                }
            )
        dataset.save_episode()
        episode_count += 1

    print(f"Converted {episode_count} episodes to LeRobot format at {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["go_vla", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
