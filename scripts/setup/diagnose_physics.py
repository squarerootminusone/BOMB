#!/usr/bin/env python3
"""Diagnose physics issues: check model params, stone drift, arm movement."""

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
    seed=0,
    include_image_obs=False,
    camera_height=84,
    camera_width=84,
    action_scale=0.01,
    max_steps=1000,
    success_hold_steps=20,
    robot="Panda",
    gripper_types="default",
)

rs = env._rs_env
model = rs.sim.model

# Check solver settings
print(f"=== SOLVER ===")
print(f"timestep: {model.opt.timestep}")
print(f"noslip_iterations: {model.opt.noslip_iterations}")
print(f"noslip_tolerance: {model.opt.noslip_tolerance}")

# Check lite_physics effect
print(f"\n=== CONTROL ===")
print(f"control_freq: {rs.control_freq}")
print(f"control_timestep: {rs.control_timestep}")
print(f"model_timestep: {rs.model_timestep}")
print(f"steps per control: {int(rs.control_timestep / rs.model_timestep)}")

# Check stone geom params
print(f"\n=== STONE GEOMS ===")
for i, obj in enumerate(rs._stone_objects[:3]):
    body_id = model.body_name2id(obj.root_body)
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == body_id:
            name = model.geom_id2name(gid) or f"geom_{gid}"
            print(f"  {name}:")
            print(f"    condim={model.geom_condim[gid]}")
            print(f"    friction={model.geom_friction[gid].tolist()}")
            print(f"    solref={model.geom_solref[gid].tolist()}")
            print(f"    solimp={model.geom_solimp[gid].tolist()}")
            print(f"    margin={model.geom_margin[gid]}")
            break

# Check board collision geom
print(f"\n=== BOARD GEOM ===")
for gid in range(model.ngeom):
    name = model.geom_id2name(gid)
    if name and "board" in name:
        print(f"  {name}:")
        print(f"    condim={model.geom_condim[gid]}")
        print(f"    contype={model.geom_contype[gid]}")
        print(f"    conaffinity={model.geom_conaffinity[gid]}")
        print(f"    friction={model.geom_friction[gid].tolist()}")

# Check table geom
print(f"\n=== TABLE GEOMS ===")
for gid in range(model.ngeom):
    name = model.geom_id2name(gid)
    if name and "table" in name:
        print(f"  {name}: contype={model.geom_contype[gid]} conaffinity={model.geom_conaffinity[gid]} condim={model.geom_condim[gid]}")

# Check joint damping
print(f"\n=== JOINT DAMPING ===")
for jnt_name in rs._stone_joint_names[:3]:
    jid = model.joint_name2id(jnt_name)
    dof = model.jnt_dofadr[jid]
    damp = [model.dof_damping[dof+i] for i in range(6)]
    print(f"  {jnt_name}: damping={damp}")

# Now test: reset env, place stones, measure drift
print(f"\n=== STONE DRIFT TEST ===")
obs = env.reset(options=GoResetOptions(opening_moves=4))

# Record stone positions
placed_stones = []
for key, idx in env._stone_assignments.items():
    pos = rs.get_stone_pos(idx)
    placed_stones.append((idx, pos.copy()))
    print(f"  stone {idx} initial pos: {pos}")

# Step 100 times with zero action
for i in range(100):
    env.step(np.zeros(4))

print(f"\n  After 100 zero-action steps:")
for idx, init_pos in placed_stones:
    pos = rs.get_stone_pos(idx)
    drift = np.linalg.norm(pos - init_pos)
    print(f"  stone {idx} pos: {pos}  drift: {drift:.6f}m")

# Test arm movement
print(f"\n=== ARM CONTROLLER ===")
print(f"  _controller_xyz_max: {env._controller_xyz_max}")
print(f"  _arm_key: {env._arm_key}")
print(f"  _arm_dim: {env._arm_dim}")
print(f"  _gripper_key: {env._gripper_key}")
print(f"  action_scale: {env.action_scale}")
ctrl = env._robot.part_controllers.get(env._arm_key, None)
if ctrl:
    print(f"  controller type: {type(ctrl).__name__}")
    print(f"  input_ref_frame: {getattr(ctrl, 'input_ref_frame', '?')}")
    if hasattr(ctrl, 'output_max'):
        print(f"  output_max: {ctrl.output_max}")
    if hasattr(ctrl, 'output_min'):
        print(f"  output_min: {ctrl.output_min}")

print(f"\n=== ARM MOVEMENT TEST ===")
obs = env.reset(options=GoResetOptions(opening_moves=0))
tgt = env.get_observation()["target_pos"]
print(f"  target_pos: {tgt}")
for i in range(50):
    obs_dict = env.get_observation()
    eef = obs_dict["eef_pos"]
    # Move toward target
    delta = tgt - eef
    delta[2] = 0  # stay level first
    norm = np.linalg.norm(delta)
    if norm > 0:
        delta = delta / norm
    action = np.array([delta[0], delta[1], 0.0, 0.0])
    low_level = env._build_low_level_action(action)
    obs, r, done, info = env.step(action)
    if i % 5 == 0:
        print(f"  step {i}: eef={eef}, dist_to_target={norm:.4f}, low_level_norm={np.linalg.norm(low_level):.6f}")
