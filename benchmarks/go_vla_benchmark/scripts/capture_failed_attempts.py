#!/usr/bin/env python3
"""Capture frames from robosuite Go env attempts (including failures) and export video."""

from __future__ import annotations
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.paths import bootstrap_pythonpath
bootstrap_pythonpath(REPO_ROOT)

from go_vla_benchmark.runtime import configure_gnugo_path
configure_gnugo_path(None)

import numpy as np
import imageio

from go_vla_benchmark.common import GoResetOptions
from go_vla_benchmark.env_factory import create_benchmark_env

NUM_ATTEMPTS = 2
CAMERA_SIZE = 256
MAX_STEPS = 500
OUTPUT = str(REPO_ROOT / "benchmarks" / "go_vla_benchmark" / "data" / "robosuite_failed_preview.mp4")

env = create_benchmark_env(
    environment_name="robosuite_go_5x5_rigid_bodies",
    seed=0,
    include_image_obs=True,
    camera_height=CAMERA_SIZE,
    camera_width=CAMERA_SIZE,
    action_scale=0.01,
    hover_height=0.16,
    press_height=0.03,
    reach_xy_threshold=0.03,
    max_steps=MAX_STEPS,
    success_hold_steps=20,
    drive_physical_arm=True,
    enable_opponent_moves=False,
    opening_with_opponent=True,
    render_carried_stone=True,
    render_eef_overlay=True,
    eef_overlay_trail=10,
    robot="Panda",
    gripper_types="default",
)

rng = np.random.RandomState(0)
all_frames = []

for attempt in range(NUM_ATTEMPTS):
    print(f"\n[attempt {attempt+1}/{NUM_ATTEMPTS}]", flush=True)
    opening = int(rng.randint(0, 4))  # fewer opening moves
    obs = env.reset(options=GoResetOptions(opening_moves=opening))
    target_pos = env.get_observation().get("target_pos", None)
    source_pos = env._source_xyz.copy()
    print(f"  target={target_pos}, source_stone={source_pos}", flush=True)

    frames = []
    img = obs.get("agentview_image", None)
    if img is not None:
        frames.append(img.copy())

    # Multi-phase scripted controller:
    # Phase 1: Move to source stone (where active stone spawns)
    # Phase 2: Close gripper to grasp
    # Phase 3: Lift stone
    # Phase 4: Move toward target over board
    # Phase 5: Descend and press
    phase = "approach_source"
    hover_z = env.table_top_z + 0.16

    for step_i in range(MAX_STEPS):
        obs_dict = env.get_observation()
        eef_pos = obs_dict.get("eef_pos", np.zeros(3))

        if phase == "approach_source":
            # Move to above the source stone
            target_xy = source_pos[:2]
            delta = np.zeros(3)
            delta[:2] = target_xy - eef_pos[:2]
            delta[2] = (source_pos[2] + 0.04) - eef_pos[2]  # hover above source
            xy_dist = np.linalg.norm(delta[:2])
            if xy_dist < 0.015 and abs(delta[2]) < 0.015:
                phase = "descend_source"
                print(f"  step {step_i}: -> descend_source (xy_dist={xy_dist:.4f})", flush=True)

        elif phase == "descend_source":
            delta = np.zeros(3)
            delta[:2] = source_pos[:2] - eef_pos[:2]
            delta[2] = source_pos[2] - eef_pos[2]  # go to stone height
            if eef_pos[2] - source_pos[2] < 0.008:
                phase = "grasp"
                grasp_start = step_i
                print(f"  step {step_i}: -> grasp", flush=True)

        elif phase == "grasp":
            delta = np.zeros(3)  # hold position
            if step_i - grasp_start > 20:
                grasped = env._is_active_stone_grasped()
                print(f"  step {step_i}: grasped={grasped}", flush=True)
                phase = "lift"

        elif phase == "lift":
            delta = np.array([0.0, 0.0, 1.0])  # straight up
            if eef_pos[2] > hover_z:
                phase = "approach_target"
                print(f"  step {step_i}: -> approach_target (z={eef_pos[2]:.4f})", flush=True)

        elif phase == "approach_target":
            delta = np.zeros(3)
            delta[:2] = target_pos[:2] - eef_pos[:2]
            delta[2] = hover_z - eef_pos[2]
            xy_dist = np.linalg.norm(delta[:2])
            if xy_dist < 0.015:
                phase = "descend_target"
                print(f"  step {step_i}: -> descend_target", flush=True)

        elif phase == "descend_target":
            delta = np.zeros(3)
            delta[:2] = target_pos[:2] - eef_pos[:2]
            delta[2] = (target_pos[2] - 0.01) - eef_pos[2]  # press into board

        elif phase == "release":
            # Open gripper and retreat upward after placing stone
            delta = np.array([0.0, 0.0, 1.0])

        # Normalize delta
        norm = np.linalg.norm(delta)
        if norm > 0:
            delta = delta / norm

        action = np.zeros(4)
        action[:3] = np.clip(delta, -1, 1)
        # Gripper: close during grasp/lift/carry/place, open after release
        # robosuite convention: action[3]=1.0 -> close, action[3]=-1.0 -> open
        if phase in ("grasp", "lift", "approach_target", "descend_target"):
            action[3] = 1.0  # close gripper
        else:
            action[3] = -1.0  # open gripper

        obs, reward, done, info = env.step(action)
        img = obs.get("agentview_image", None)
        if img is not None and step_i % 2 == 0:  # every other frame to keep video shorter
            frames.append(img.copy())

        # Transition to release phase once the move is committed
        if info.get("move_committed") and phase == "descend_target":
            phase = "release"
            print(f"  step {step_i}: move committed -> release", flush=True)

        if done:
            print(f"  step {step_i}: DONE reward={reward} committed={info.get('move_committed')}", flush=True)
            break

        if step_i % 50 == 0:
            print(f"  step {step_i}: phase={phase} eef={eef_pos} dist={np.linalg.norm(eef_pos[:2]-target_pos[:2]):.4f}", flush=True)

    print(f"  captured {len(frames)} frames, final phase={phase}", flush=True)

    if frames:
        black = np.zeros_like(frames[0])
        all_frames.extend(frames)
        all_frames.extend([black] * 4)

print(f"\nWriting {len(all_frames)} total frames to {OUTPUT}", flush=True)
writer = imageio.get_writer(OUTPUT, fps=16, macro_block_size=1)
for f in all_frames:
    writer.append_data(f)
writer.close()
print("Done!", flush=True)
