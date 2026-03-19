"""MimicGen environment interface for the Go benchmark wrapper."""

from __future__ import annotations

import numpy as np

from mimicgen.env_interfaces.base import MG_EnvInterface


class MG_GoJacoSingleMove(MG_EnvInterface):
    """Environment interface for single-move Go trajectories."""

    INTERFACE_TYPE = "dmcontrol_go"

    def get_robot_eef_pose(self):
        return self.env.get_eef_pose()

    def target_pose_to_action(self, target_pose, relative=True):
        target_pose = np.asarray(target_pose, dtype=np.float32)
        target_xyz = target_pose[:3, 3]

        if relative:
            current_xyz = self.env.get_eef_pose()[:3, 3]
            delta_xyz = target_xyz - current_xyz
        else:
            delta_xyz = target_xyz - self.env.get_eef_pose()[:3, 3]

        action_xyz = np.clip(delta_xyz / self.env.action_scale, -1.0, 1.0)
        return action_xyz.astype(np.float32)

    def action_to_target_pose(self, action, relative=True):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action_xyz = action[:3]

        current_pose = self.env.get_eef_pose()
        target_pose = current_pose.copy()

        if relative:
            target_pose[:3, 3] = current_pose[:3, 3] + action_xyz * self.env.action_scale
        else:
            target_pose[:3, 3] = action_xyz

        return target_pose

    def action_to_gripper_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        return action[-1:]

    def get_object_poses(self):
        return {
            "target_intersection": self.env.get_target_pose(),
            "board_origin": self.env.get_board_origin_pose(),
        }

    def get_subtask_term_signals(self):
        return {
            "reach_target": int(self.env.reached_target()),
        }
