#!/usr/bin/env python3
"""Debug: trace damping on committed stone during release."""
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
from go_vla_benchmark import robosuite_go_env as rge

_orig_step = rge.GoRobosuiteBenchmarkEnv.step
_log = []

def _traced_step(self, action):
    result = _orig_step(self, action)
    eef_z = self.get_eef_pose()[2, 3]
    gripper_val = float(action[3]) if len(action) > 3 else 0.0
    committed = self._move_committed
    cidx = self._committed_stone_idx
    countdown = self._committed_stone_release_countdown
    # Get damping on committed stone
    damping = None
    stone_z = None
    if cidx is not None:
        jnt_name = self._rs_env._stone_joint_names[cidx]
        jnt_id = self._rs_env.sim.model.joint_name2id(jnt_name)
        dof = self._rs_env.sim.model.jnt_dofadr[jnt_id]
        damping = float(self._rs_env.sim.model.dof_damping[dof])
        stone_z = float(self._rs_env.get_stone_pos(cidx)[2])
    _log.append((len(_log), eef_z, stone_z, gripper_val, committed, cidx, countdown, damping))
    return result

rge.GoRobosuiteBenchmarkEnv.step = _traced_step

from go_vla_benchmark.collect import collect_source_demonstrations
collect_source_demonstrations(
    output_path="/tmp/debug_release.hdf5",
    environment_name="robosuite_go_5x5_rigid_bodies",
    num_demos=1, seed=42,
    max_attempts_per_demo=1,
    opening_moves_min=2, opening_moves_max=2,
    include_image_obs=False,
    camera_height=84, camera_width=84,
    action_scale=0.03, success_hold_steps=0,
    controller_divisor=2.0,
    detour_steps=0, detour_radius=0.0,
    approach_steps=10, press_steps=6, retreat_steps=6,
    side_transfer_steps=0, side_margin=0.16,
    recovery_steps=3,
    drive_physical_arm=True,
    enable_opponent_moves=False,
    opening_with_opponent=True,
    render_carried_stone=False,
    render_eef_overlay=False,
    eef_overlay_trail=0,
    robot="Panda", gripper_types="default",
)

commit_step = None
for entry in _log:
    if entry[4] and commit_step is None:
        commit_step = entry[0]

print(f"\ntotal steps={len(_log)}, commit_step={commit_step}")
print(f"{'step':>5s} {'eef_z':>7s} {'stn_z':>7s} {'grip':>5s} {'cidx':>4s} {'cntdn':>5s} {'damp':>6s}")
print("-" * 50)
for i, eef_z, stone_z, grip, committed, cidx, countdown, damping in _log:
    if commit_step is not None and (commit_step - 3) <= i <= (commit_step + 35):
        sz = f"{stone_z:.4f}" if stone_z is not None else "  None"
        ci = f"{cidx}" if cidx is not None else "None"
        dm = f"{damping:.1f}" if damping is not None else " None"
        print(f"{i:5d} {eef_z:7.4f} {sz:>7s} {grip:+5.1f} {ci:>4s} {countdown:5d} {dm:>6s}")
