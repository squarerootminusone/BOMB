#!/usr/bin/env python3
"""Launch MuJoCo interactive viewer for the Go environment with stones on board.

Opens the viewer *before* settling so you can watch stones drop and stabilize.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.paths import bootstrap_pythonpath
bootstrap_pythonpath(REPO_ROOT)

import mujoco
import mujoco.viewer
from go_vla_benchmark.common import GoResetOptions
from go_vla_benchmark.env_factory import create_benchmark_env

env = create_benchmark_env(
    seed=0, environment_name="robosuite_go_5x5_rigid_bodies",
    include_image_obs=False, camera_height=84, camera_width=84,
    action_scale=0.03, success_hold_steps=0, drive_physical_arm=True,
    enable_opponent_moves=False, opening_with_opponent=True,
    render_carried_stone=True, render_eef_overlay=False, eef_overlay_trail=0,
    robot="Panda", gripper_types="default",
)

# Do a normal reset (which includes settling) to get a valid state,
# then re-place the opening stones above the board and launch the viewer
# so we can watch them drop.
env.reset(options=GoResetOptions(opening_moves=6))

rs = env._rs_env

# Re-place all assigned stones slightly above their intersections
for (player_id, row, col), stone_idx in env._stone_assignments.items():
    xyz = env._intersection_xyz[row, col].copy()
    xyz[2] += rs._board_stone_clearance
    rs.set_stone_pose(stone_idx=stone_idx, pos=xyz)

# Spread out the offboard (hidden) stones so they're not all stacked
stone_z = rs.table_top_z + rs.stone_half_height + rs._table_stone_clearance
board_half = 0.5 * (rs.board_size - 1) * rs.board_spacing

white_pool = env._available_stones[0]
for idx, stone_idx in enumerate(white_pool):
    row = idx // 4
    col = idx % 4
    x = rs.board_center_xy[0] - board_half - 0.08 - col * 0.035
    y = rs.board_center_xy[1] + board_half - row * 0.035
    rs.set_stone_pose(stone_idx, np.array([x, y, stone_z]))

black_pool = env._available_stones[1]
for idx, stone_idx in enumerate(black_pool):
    row = idx // 4
    col = idx % 4
    x = rs.board_center_xy[0] + board_half + 0.08 + col * 0.035
    y = rs.board_center_xy[1] + board_half - row * 0.035
    rs.set_stone_pose(stone_idx, np.array([x, y, stone_z]))

rs.sim.forward()

# Bake current state into model's qpos0 so viewer reset restores it
model = rs.sim.model
m = model._model
d = rs.sim.data._data
m.qpos0[:] = d.qpos[:]

import time, threading

def fps_monitor(data):
    """Print sim FPS every second by watching data.time."""
    prev_time = data.time
    while True:
        time.sleep(1.0)
        try:
            dt = data.time - prev_time
            prev_time = data.time
            if dt > 0:
                print(f"Sim speed: {dt:.2f}x realtime", flush=True)
        except Exception:
            break

t = threading.Thread(target=fps_monitor, args=(d,), daemon=True)
t.start()

mujoco.viewer.launch(m, d)
