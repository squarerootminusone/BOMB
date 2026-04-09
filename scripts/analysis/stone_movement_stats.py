#!/usr/bin/env python3
"""Report stone movement statistics with the implicit integrator."""

from __future__ import annotations
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.paths import bootstrap_pythonpath
bootstrap_pythonpath(REPO_ROOT)

from go_vla_benchmark.runtime import configure_gnugo_path
configure_gnugo_path(None)

import numpy as np
from go_vla_benchmark.common import GoResetOptions
from go_vla_benchmark.env_factory import create_benchmark_env

env = create_benchmark_env(
    environment_name="robosuite_go_5x5_rigid_bodies",
    seed=0, include_image_obs=False,
    camera_height=84, camera_width=84,
    action_scale=0.03, max_steps=500,
    success_hold_steps=0,
    robot="Panda", gripper_types="default",
)

rs = env._rs_env

# Test 1: idle drift (zero action, 4 opening stones)
print("=== IDLE DRIFT (200 zero-action steps, 4 opening stones) ===")
obs = env.reset(options=GoResetOptions(opening_moves=4))
placed = []
for key, idx in env._stone_assignments.items():
    pos = rs.get_stone_pos(idx)
    placed.append((idx, key, pos.copy()))

for _ in range(200):
    env.step(np.zeros(4))

print(f"  {'idx':>4s} | {'drift (mm)':>10s}")
print(f"  {'----':>4s} | {'----------':>10s}")
for idx, key, init in placed:
    final = rs.get_stone_pos(idx)
    drift = np.linalg.norm(final - init) * 1000
    print(f"  {idx:4d} | {drift:10.2f}")

# Test 2: arm-moving drift (move arm toward target for 300 steps)
print()
print("=== ARM-MOVING DRIFT (300 steps, arm moving toward target) ===")
obs = env.reset(options=GoResetOptions(opening_moves=4))
placed2 = []
for key, idx in env._stone_assignments.items():
    pos = rs.get_stone_pos(idx)
    placed2.append((idx, key, pos.copy()))

target = env.get_observation()["target_pos"]
for i in range(300):
    obs_dict = env.get_observation()
    eef = obs_dict["eef_pos"]
    delta = target - eef
    norm = np.linalg.norm(delta)
    if norm > 0:
        delta = delta / norm
    action = np.array([delta[0], delta[1], delta[2], 0.0])
    env.step(action)

print(f"  {'idx':>4s} | {'drift (mm)':>10s}")
print(f"  {'----':>4s} | {'----------':>10s}")
max_drift = 0
for idx, key, init in placed2:
    final = rs.get_stone_pos(idx)
    drift = np.linalg.norm(final - init) * 1000
    max_drift = max(max_drift, drift)
    print(f"  {idx:4d} | {drift:10.2f}")
print(f"  max drift: {max_drift:.2f} mm")

# Test 3: z-oscillation check
print()
print("=== Z-OSCILLATION CHECK (50 zero-action steps) ===")
obs = env.reset(options=GoResetOptions(opening_moves=4))
first_idx = list(env._stone_assignments.values())[0]
jnt_name = rs._stone_joint_names[first_idx]
sign_changes = 0
prev_vz = None
for i in range(50):
    env.step(np.zeros(4))
    vel = rs.sim.data.get_joint_qvel(jnt_name)
    vz = vel[2]
    if prev_vz is not None and np.sign(vz) != np.sign(prev_vz) and np.sign(vz) != 0:
        sign_changes += 1
    prev_vz = vz
print(f"  z-velocity sign changes in 50 steps: {sign_changes}")
print(f"  integrator: {rs.sim.model.opt.integrator} (0=Euler, 1=RK4, 2=implicit)")
