#!/usr/bin/env python3
"""Record MimicGen augmentation attempts (success + fail) as video for debugging."""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "benchmarks" / "go_vla_benchmark"))
from go_vla_benchmark.paths import bootstrap_pythonpath
bootstrap_pythonpath(REPO)

import numpy as np
import imageio

from go_vla_benchmark.robosuite_compat import ensure_robosuite_compat
ensure_robosuite_compat()

from mimicgen.datagen.data_generator import DataGenerator
from go_vla_benchmark.config import load_task_config
from go_vla_benchmark.common import GoResetOptions
from go_vla_benchmark.env_factory import create_benchmark_env
from go_vla_benchmark.mimicgen_interface import MG_GoJacoSingleMove

SOURCE = REPO / "benchmarks" / "go_vla_benchmark" / "data" / "source_go.hdf5"
OUTPATH = REPO / "benchmarks" / "go_vla_benchmark" / "data" / "augment_debug.mp4"

NUM_ATTEMPTS = 15


def main():
    task_cfg = REPO / "benchmarks" / "go_vla_benchmark" / "configs" / "go_single_move_task.json"
    task_spec, gen_settings, raw_cfg = load_task_config(task_cfg)
    env = create_benchmark_env(
        seed=99,
        environment_name="robosuite_go_5x5_rigid_bodies",
        include_image_obs=True,
        camera_height=256,
        camera_width=256,
        action_scale=0.03,
        success_hold_steps=0,
        drive_physical_arm=True,
        enable_opponent_moves=False,
        opening_with_opponent=True,
        render_carried_stone=False,
        render_eef_overlay=True,
        eef_overlay_trail=10,
        robot="Panda",
        gripper_types="default",
    )
    env_interface = MG_GoJacoSingleMove(env=env)

    from go_vla_benchmark.dataset_io import read_demo_keys
    demo_keys = read_demo_keys(str(SOURCE))
    data_generator = DataGenerator(
        task_spec=task_spec,
        dataset_path=str(SOURCE),
        demo_keys=demo_keys,
    )

    rng = np.random.RandomState(99)
    blank = np.zeros((256, 256, 3), dtype=np.uint8)
    fps = 12

    with imageio.get_writer(str(OUTPATH), fps=fps, macro_block_size=1) as writer:
        for attempt in range(NUM_ATTEMPTS):
            opening = int(rng.randint(1, 9))
            env.queue_reset_options(GoResetOptions(opening_moves=opening))

            generated = data_generator.generate(
                env=env,
                env_interface=env_interface,
                select_src_per_subtask=gen_settings.select_src_per_subtask,
                transform_first_robot_pose=gen_settings.transform_first_robot_pose,
                interpolate_from_last_target_pose=gen_settings.interpolate_from_last_target_pose,
                render=False,
                video_writer=None,
                video_skip=5,
                camera_names=None,
                pause_subtask=False,
            )

            success = bool(generated["success"])
            tag = "OK" if success else "FAIL"

            # Grab frames from observations
            obs = generated.get("observations", [])
            if obs and "agentview_image" in obs[0]:
                frames = [o["agentview_image"] for o in obs]
            else:
                frames = []

            n_frames = len(frames)
            print(f"Attempt {attempt}: {tag}  frames={n_frames}  opening={opening}")

            for frame in frames:
                writer.append_data(frame)

            # Hold last frame
            if frames:
                for _ in range(max(1, int(0.5 * fps))):
                    writer.append_data(frames[-1])

            # Black separator
            if attempt < NUM_ATTEMPTS - 1:
                for _ in range(4):
                    writer.append_data(blank)

    env.close()
    print(f"\nSaved {OUTPATH}")


if __name__ == "__main__":
    main()
