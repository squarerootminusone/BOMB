#!/usr/bin/env python3
"""Record agentview video of N demo attempts (success or fail)."""

from __future__ import annotations

import sys
from pathlib import Path

import imageio
import numpy as np

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.paths import bootstrap_pythonpath

bootstrap_pythonpath(REPO_ROOT)

from go_vla_benchmark.collect import _collect_single_episode
from go_vla_benchmark.common import GoResetOptions
from go_vla_benchmark.env_factory import create_benchmark_env
from go_vla_benchmark.mimicgen_interface import MG_GoJacoSingleMove


def main() -> None:
    num_episodes = 5
    seed = 77
    outpath = REPO_ROOT / "benchmarks" / "go_vla_benchmark" / "data" / "demo_attempts.mp4"
    outpath.parent.mkdir(exist_ok=True)

    env = create_benchmark_env(
        seed=seed,
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
    rng = np.random.RandomState(seed)

    blank = np.zeros((256, 256, 3), dtype=np.uint8)
    fps = 12

    with imageio.get_writer(str(outpath), fps=fps, macro_block_size=1) as writer:
        for ep in range(num_episodes):
            stone_color = rng.choice(["black", "white"])
            opening_moves = int(rng.randint(0, 9))
            env.reset(options=GoResetOptions(opening_moves=opening_moves, stone_color=stone_color))

            episode, success = _collect_single_episode(
                env=env,
                env_interface=env_interface,
                controller_divisor=2.0,
                detour_steps=0,
                detour_radius=0.0,
                approach_steps=10,
                press_steps=6,
                retreat_steps=6,
                side_transfer_steps=0,
                side_margin=0.16,
                recovery_steps=3,
                hover_height_noise=0.2,
            )

            tag = "OK" if success else "FAIL"
            n_frames = episode.observations["agentview_image"].shape[0]
            print(f"Episode {ep}: {tag}  steps={n_frames}  color={stone_color}  opening={opening_moves}")

            for frame in episode.observations["agentview_image"]:
                writer.append_data(frame)

            # Hold last frame briefly
            last = episode.observations["agentview_image"][-1]
            for _ in range(max(1, int(0.5 * fps))):
                writer.append_data(last)

            # Black separator
            if ep < num_episodes - 1:
                for _ in range(4):
                    writer.append_data(blank)

    env.close()
    print(f"\nSaved {outpath}")


if __name__ == "__main__":
    main()
