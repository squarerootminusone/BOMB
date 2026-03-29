"""DeepMind Go wrapper with a MimicGen-friendly API."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from dm_control.entities.manipulators import base as manip_base
from physics_planning_games import board_games
from physics_planning_games.board_games import go_logic
from physics_planning_games.board_games import jaco_arm_board_game
from physics_planning_games.board_games._internal import pieces as board_pieces

from .common import GoResetOptions
from .runtime import configure_gnugo_path


class GoJacoBenchmarkEnv:
    """
    Thin adapter around DeepMind's `go_7x7` composer env.

    The wrapper exposes a simple 4D action space used by this benchmark:
    - action[:3]: delta xyz (normalized to [-1, 1])
    - action[3]: gripper command (tracked for dataset consistency)

    Move execution is triggered when the pseudo end-effector reaches the chosen
    target intersection and crosses a press height threshold.
    """

    def __init__(
        self,
        seed: int = 0,
        environment_name: str = "go_7x7",
        include_image_obs: bool = True,
        camera_height: int = 84,
        camera_width: int = 84,
        action_scale: float = 0.03,
        hover_height: float = 0.16,
        press_height: float = 0.03,
        reach_xy_threshold: float = 0.03,
        max_steps: int = 200,
        success_hold_steps: int = 0,
        drive_physical_arm: bool = True,
        enable_opponent_moves: bool = False,
        opening_with_opponent: bool = True,
        render_carried_stone: bool = True,
        render_eef_overlay: bool = True,
        eef_overlay_trail: int = 10,
        gnugo_path: Optional[str] = None,
    ):
        self.seed = int(seed)
        self._rng = np.random.RandomState(self.seed)
        self.environment_name = str(environment_name)

        self.gnugo_path = configure_gnugo_path(preferred=gnugo_path)
        self._dm_env = board_games.load(
            environment_name=self.environment_name,
            seed=self.seed,
            time_limit=float("inf"),
            strip_singleton_obs_buffer_dim=True,
        )
        self._dm_action_spec = self._dm_env.action_spec()
        self._dm_zero_action = np.zeros(self._dm_action_spec.shape, dtype=self._dm_action_spec.dtype)

        self.include_image_obs = bool(include_image_obs)
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)
        self._ensure_offscreen_framebuffer_size(
            required_height=self.camera_height,
            required_width=self.camera_width,
        )

        self.action_scale = float(action_scale)
        self.hover_height = float(hover_height)
        self.press_height = float(press_height)
        self.reach_xy_threshold = float(reach_xy_threshold)
        self.max_steps = int(max_steps)
        self.success_hold_steps = max(0, int(success_hold_steps))
        self.drive_physical_arm = bool(drive_physical_arm)
        self.enable_opponent_moves = bool(enable_opponent_moves)
        self.opening_with_opponent = bool(opening_with_opponent)
        self.render_carried_stone = bool(render_carried_stone)
        self.render_eef_overlay = bool(render_eef_overlay)
        self.eef_overlay_trail = max(0, int(eef_overlay_trail))

        self.workspace_low = np.array([-0.5, -0.5, 0.0], dtype=np.float32)
        self.workspace_high = np.array([0.5, 0.5, 0.4], dtype=np.float32)

        if hasattr(self._dm_env.task._game_logic, "board_size"):
            self.board_size = int(self._dm_env.task._game_logic.board_size())
        else:
            board_state = self._dm_env.task._game_logic.get_board_state()
            self.board_size = int(board_state.shape[0])
        self._intersection_xyz = None

        self._eef_pose = np.eye(4, dtype=np.float32)
        self._eef_trail: list[np.ndarray] = []
        self._target_rc: Tuple[int, int] = (0, 0)
        self._target_pose = np.eye(4, dtype=np.float32)
        self._gripper_action = np.zeros((1,), dtype=np.float32)
        self._ik_last_success = True
        self._last_grasp_close_factor: Optional[float] = None
        self._carried_marker_site = None

        self._step_count = 0
        self._pressing_last_step = False
        self._move_committed = False
        self._success_step: Optional[int] = None
        self._queued_reset_options: Optional[GoResetOptions] = None

        self._arm = self._dm_env.task.arm
        self._hand = self._dm_env.task.hand
        self._ik_site = self._hand.pinch_site
        self._ik_target_quat = np.asarray(manip_base.DOWN_QUATERNION, dtype=np.float64)

        self.reset()

    def _ensure_offscreen_framebuffer_size(self, required_height: int, required_width: int) -> None:
        required_height = int(required_height)
        required_width = int(required_width)
        vis_global = self._dm_env.physics.model.vis.global_
        cur_h = int(vis_global.offheight)
        cur_w = int(vis_global.offwidth)
        if (required_height > cur_h) or (required_width > cur_w):
            vis_global.offheight = max(cur_h, required_height)
            vis_global.offwidth = max(cur_w, required_width)

    @property
    def base_env(self):
        """Compatibility shim with MimicGen wrappers."""
        return self

    @property
    def physics(self):
        return self._dm_env.physics

    def _pose_from_xyz(self, xyz: np.ndarray) -> np.ndarray:
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = xyz.astype(np.float32)
        return pose

    def _physical_eef_pose(self) -> np.ndarray:
        bound_site = self.physics.bind(self._ik_site)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = bound_site.xpos.copy().astype(np.float32)
        pose[:3, :3] = bound_site.xmat.reshape(3, 3).copy().astype(np.float32)
        return pose

    def _sync_physical_arm_to_command(self) -> bool:
        if not self.drive_physical_arm:
            return True

        # Solve IK so the real Kinova arm follows the scripted pseudo-EFF path.
        success = self._arm.set_site_to_xpos(
            physics=self.physics,
            random_state=self._rng,
            site=self._ik_site,
            target_pos=self._eef_pose[:3, 3].astype(np.float64),
            target_quat=self._ik_target_quat,
            max_ik_attempts=1,
        )

        close_factor = float(np.clip((float(self._gripper_action[0]) + 1.0) * 0.5, 0.0, 1.0))
        if (self._last_grasp_close_factor is None) or (abs(close_factor - self._last_grasp_close_factor) > 1e-3):
            self._hand.set_grasp(physics=self.physics, close_factors=close_factor)
            self._last_grasp_close_factor = close_factor

        self.physics.forward()
        return bool(success)

    def _reset_carry_visual(self) -> None:
        if self._carried_marker_site is None:
            return
        bound_marker = self.physics.bind(self._carried_marker_site)
        bound_marker.group = board_pieces._INVISIBLE_SITE_GROUP
        self._carried_marker_site = None

    def _update_carry_visual(self) -> None:
        if not self.render_carried_stone:
            return
        markers = self._dm_env.task._markers
        if not bool(getattr(markers, "supports_carry_preview", True)):
            return
        if self._move_committed:
            self._reset_carry_visual()
            return

        # Carry visualization is active while scripted gripper command is "closed".
        if float(self._gripper_action[0]) <= 0.0:
            self._reset_carry_visual()
            return

        move_count = int(markers._move_counts[jaco_arm_board_game.SELF])
        player_markers = markers._all_markers[jaco_arm_board_game.SELF]
        if move_count >= len(player_markers):
            self._reset_carry_visual()
            return

        if self._carried_marker_site is None:
            self._carried_marker_site = player_markers[move_count]

        pinch_xyz = self.physics.bind(self._ik_site).xpos.copy()
        bound_marker = self.physics.bind(self._carried_marker_site)
        bound_marker.pos = pinch_xyz
        bound_marker.group = board_pieces._VISIBLE_SITE_GROUP

    def _refresh_intersection_cache(self) -> None:
        xyz = np.zeros((self.board_size, self.board_size, 3), dtype=np.float32)
        for row in range(self.board_size):
            for col in range(self.board_size):
                site = self._dm_env.task._board._touch_sensors[row, col].site
                xyz[row, col] = self.physics.bind(site).xpos.copy()
        self._intersection_xyz = xyz

    def _decode_action_int(self, action_int: int) -> Tuple[int, int, bool]:
        pass_id = self.board_size * self.board_size
        if action_int == pass_id:
            return -1, -1, True
        row = int(action_int // self.board_size)
        col = int(action_int % self.board_size)
        return row, col, False

    def _choose_random_legal_action(self, exclude_pass: bool = True) -> Optional[int]:
        legal_actions = list(self._dm_env.task._game_logic.open_spiel_state.legal_actions())
        if exclude_pass:
            pass_id = self.board_size * self.board_size
            legal_actions = [a for a in legal_actions if int(a) != pass_id]
        if not legal_actions:
            return None
        return int(self._rng.choice(np.asarray(legal_actions, dtype=np.int32)))

    def _redraw_markers(self) -> None:
        task = self._dm_env.task
        task._markers.make_all_invisible(self.physics)

        board = task._game_logic.get_board_state()
        black_stones = np.transpose(np.nonzero(board[:, :, 1]))
        white_stones = np.transpose(np.nonzero(board[:, :, 2]))
        if black_stones.size > 0:
            task._markers.make_visible_by_bpos(self.physics, 0, black_stones)
        if white_stones.size > 0:
            task._markers.make_visible_by_bpos(self.physics, 1, white_stones)

    def _apply_go_action(self, action_int: int, apply_opponent: Optional[bool] = None) -> bool:
        if apply_opponent is None:
            apply_opponent = self.enable_opponent_moves
        task = self._dm_env.task
        row, col, is_pass = self._decode_action_int(action_int)

        go_action = go_logic.GoMarkerAction(row=row, col=col, pass_action=is_pass)
        valid_move = task._game_logic.apply(player=jaco_arm_board_game.SELF, action=go_action)
        if not valid_move:
            return False

        if not is_pass:
            marker_pos = task._board.sample_pos_inside_touch_sensor(
                physics=self.physics,
                random_state=self._rng,
                row=row,
                col=col,
            )
            task._markers.mark(
                physics=self.physics,
                player_id=jaco_arm_board_game.SELF,
                pos=marker_pos,
                bpos=(row, col),
            )

        if bool(apply_opponent) and (not task._game_logic.is_game_over):
            opponent_move = task._game_opponent.policy(
                game_logic=task._game_logic,
                player=jaco_arm_board_game.OPPONENT,
                random_state=self._rng,
            )
            if opponent_move is not None:
                opp_valid = task._game_logic.apply(
                    player=jaco_arm_board_game.OPPONENT,
                    action=opponent_move,
                )
                if opp_valid:
                    if not bool(opponent_move.pass_action):
                        marker_pos = task._board.sample_pos_inside_touch_sensor(
                            physics=self.physics,
                            random_state=self._rng,
                            row=opponent_move.row,
                            col=opponent_move.col,
                        )
                        task._markers.mark(
                            physics=self.physics,
                            player_id=jaco_arm_board_game.OPPONENT,
                            pos=marker_pos,
                            bpos=(opponent_move.row, opponent_move.col),
                        )

        self._redraw_markers()
        self._carried_marker_site = None
        self.physics.forward()
        return True

    def seed_random_opening(self, opening_moves: int) -> int:
        opening_moves = max(0, int(opening_moves))
        applied = 0
        for _ in range(opening_moves):
            if self._dm_env.task._game_logic.is_game_over:
                break
            action_int = self._choose_random_legal_action(exclude_pass=True)
            if action_int is None:
                break
            if self._apply_go_action(action_int, apply_opponent=self.opening_with_opponent):
                applied += 1
        return applied

    def _apply_opening_history_action(self, action_int: int) -> bool:
        task = self._dm_env.task
        current_player = int(task._game_logic.open_spiel_state.current_player())
        row, col, is_pass = self._decode_action_int(int(action_int))
        go_action = go_logic.GoMarkerAction(row=row, col=col, pass_action=is_pass)
        valid_move = task._game_logic.apply(player=current_player, action=go_action)
        if not valid_move:
            return False

        if not is_pass:
            marker_pos = task._board.sample_pos_inside_touch_sensor(
                physics=self.physics,
                random_state=self._rng,
                row=row,
                col=col,
            )
            task._markers.mark(
                physics=self.physics,
                player_id=current_player,
                pos=marker_pos,
                bpos=(row, col),
            )
        self._redraw_markers()
        self._carried_marker_site = None
        self.physics.forward()
        return True

    def replay_opening_move_history(self, move_history: np.ndarray) -> int:
        history = np.asarray(move_history, dtype=np.int32).reshape(-1)
        applied = 0
        for action_int in history.tolist():
            if int(action_int) < 0 or self._dm_env.task._game_logic.is_game_over:
                break
            if not self._apply_opening_history_action(int(action_int)):
                break
            applied += 1
        return applied

    def _set_target_pose_from_rc(self, row: int, col: int) -> None:
        target_xyz = self._intersection_xyz[row, col].copy()
        target_xyz[2] += 0.003
        self._target_pose = self._pose_from_xyz(target_xyz)
        self._target_rc = (int(row), int(col))

    def set_target_intersection(self, row: int, col: int) -> None:
        if self._intersection_xyz is None:
            self._refresh_intersection_cache()
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            raise ValueError(f"target ({row}, {col}) outside board")
        self._set_target_pose_from_rc(row=row, col=col)

    def set_target_from_random_legal_move(self) -> Tuple[int, int]:
        action_int = self._choose_random_legal_action(exclude_pass=True)
        if action_int is None:
            self.set_target_intersection(row=self.board_size // 2, col=self.board_size // 2)
            return self._target_rc
        row, col, _ = self._decode_action_int(action_int)
        self.set_target_intersection(row=row, col=col)
        return self._target_rc

    def reset(self, options: Optional[GoResetOptions] = None):
        if options is None and self._queued_reset_options is not None:
            options = self._queued_reset_options
            self._queued_reset_options = None

        if options is None:
            options = GoResetOptions()

        self._dm_env.reset()
        self._refresh_intersection_cache()

        center = self._intersection_xyz.mean(axis=(0, 1))
        center[2] = self.hover_height
        self._eef_pose = self._pose_from_xyz(center)
        self._gripper_action = np.zeros((1,), dtype=np.float32)
        self._carried_marker_site = None
        self._last_grasp_close_factor = None
        self._ik_last_success = self._sync_physical_arm_to_command()
        self._eef_trail = [self.get_eef_pose()[:3, 3].copy()]

        self._step_count = 0
        self._pressing_last_step = False
        self._move_committed = False
        self._success_step = None

        if options.opening_move_history is not None:
            self.replay_opening_move_history(options.opening_move_history)
        else:
            self.seed_random_opening(options.opening_moves)

        if options.target_row is not None and options.target_col is not None:
            self.set_target_intersection(row=options.target_row, col=options.target_col)
        else:
            self.set_target_from_random_legal_move()

        self._move_committed = False
        self._pressing_last_step = False
        return self.get_observation()

    def queue_reset_options(self, options: GoResetOptions) -> None:
        """Apply reset options to the next internal env.reset() call only."""
        self._queued_reset_options = GoResetOptions(
            opening_moves=int(options.opening_moves),
            opening_move_history=None
            if options.opening_move_history is None
            else np.asarray(options.opening_move_history, dtype=np.int32).copy(),
            target_row=options.target_row,
            target_col=options.target_col,
            stone_color=options.stone_color,
        )

    def _nearest_intersection(self, eef_xy: np.ndarray) -> Tuple[int, int, float]:
        flat_xy = self._intersection_xyz[:, :, :2].reshape(-1, 2)
        dists = np.linalg.norm(flat_xy - eef_xy.reshape(1, 2), axis=1)
        flat_idx = int(np.argmin(dists))
        row = flat_idx // self.board_size
        col = flat_idx % self.board_size
        return row, col, float(dists[flat_idx])

    def reached_target(self) -> bool:
        eef_xy = self.get_eef_pose()[:2, 3]
        tgt_xy = self._target_pose[:2, 3]
        return bool(np.linalg.norm(eef_xy - tgt_xy) <= self.reach_xy_threshold)

    def get_eef_pose(self) -> np.ndarray:
        if self.drive_physical_arm:
            return self._physical_eef_pose()
        return self._eef_pose.copy()

    def get_target_pose(self) -> np.ndarray:
        return self._target_pose.copy()

    def get_target_intersection(self) -> Tuple[int, int]:
        return self._target_rc

    def get_board_origin_pose(self) -> np.ndarray:
        center = self._intersection_xyz.mean(axis=(0, 1))
        center[2] = float(np.min(self._intersection_xyz[:, :, 2]))
        return self._pose_from_xyz(center)

    def get_board_xy_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        flat_xy = self._intersection_xyz[:, :, :2].reshape(-1, 2)
        return flat_xy.min(axis=0).astype(np.float32), flat_xy.max(axis=0).astype(np.float32)

    def get_board_state(self) -> np.ndarray:
        return self._dm_env.task._game_logic.get_board_state().astype(np.float32)

    def get_move_history(self) -> np.ndarray:
        return self._dm_env.task._game_logic.get_move_history().astype(np.float32)

    def get_state(self) -> Dict[str, np.ndarray]:
        board_flat = self.get_board_state().reshape(-1)
        eef_xyz = self.get_eef_pose()[:3, 3]
        tgt_xyz = self._target_pose[:3, 3]
        state = np.concatenate(
            [
                board_flat,
                eef_xyz,
                tgt_xyz,
                np.array([float(self._move_committed), float(self._step_count)], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        return {"states": state}

    def get_observation(self) -> Dict[str, np.ndarray]:
        eef_pose = self.get_eef_pose()
        obs: Dict[str, np.ndarray] = {
            "board_state": self.get_board_state(),
            "eef_pos": eef_pose[:3, 3].copy().astype(np.float32),
            "target_pos": self._target_pose[:3, 3].copy().astype(np.float32),
            "move_history": self.get_move_history(),
        }
        if self.include_image_obs:
            image = self.render(
                mode="rgb_array",
                height=self.camera_height,
                width=self.camera_width,
            )
            if self.render_eef_overlay:
                image = self._overlay_debug_markers(image=image)
            obs["agentview_image"] = image
        return obs

    def is_success(self) -> Dict[str, bool]:
        return {"task": bool(self._move_committed)}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 4:
            raise ValueError("GoJacoBenchmarkEnv expects action shape (4,)")

        arm_action = np.clip(action[:3], -1.0, 1.0)
        self._gripper_action = np.array([np.clip(action[3], -1.0, 1.0)], dtype=np.float32)

        new_xyz = self._eef_pose[:3, 3] + arm_action * self.action_scale
        new_xyz = np.clip(new_xyz, self.workspace_low, self.workspace_high)
        self._eef_pose[:3, 3] = new_xyz.astype(np.float32)
        self._ik_last_success = self._sync_physical_arm_to_command()
        self._update_carry_visual()
        self._eef_trail.append(self.get_eef_pose()[:3, 3].copy())
        if len(self._eef_trail) > self.eef_overlay_trail + 1:
            self._eef_trail = self._eef_trail[-(self.eef_overlay_trail + 1) :]

        press_now = bool((self.get_eef_pose()[2, 3] <= self.press_height) and self.reached_target())
        rising_edge_press = press_now and (not self._pressing_last_step)

        if rising_edge_press and (not self._move_committed):
            row, col = self._target_rc
            action_int = row * self.board_size + col
            self._move_committed = self._apply_go_action(action_int)
            if self._move_committed and self._success_step is None:
                self._success_step = int(self._step_count)

        self._pressing_last_step = press_now

        # Advance physics clock after directly setting arm / hand pose via IK.
        try:
            self.physics.step()
        except Exception:
            # Keep wrapper robust if stepping fails in terminal edge-cases.
            pass

        self._step_count += 1
        success_terminal = False
        if self._move_committed:
            if self.success_hold_steps <= 0:
                success_terminal = True
            elif self._success_step is not None:
                success_terminal = (self._step_count - self._success_step) >= self.success_hold_steps

        done = bool(
            success_terminal
            or self._dm_env.task._game_logic.is_game_over
            or (self._step_count >= self.max_steps)
        )
        reward = float(self._move_committed)
        obs = self.get_observation()
        info = {
            "target_row": int(self._target_rc[0]),
            "target_col": int(self._target_rc[1]),
            "move_committed": bool(self._move_committed),
            "step_count": int(self._step_count),
            "ik_success": bool(self._ik_last_success),
        }
        return obs, reward, done, info

    def render(
        self,
        mode: str = "rgb_array",
        height: int = 256,
        width: int = 256,
        camera_name: Optional[str] = None,
    ):
        del camera_name
        if mode == "rgb_array":
            return self.physics.render(height=height, width=width, camera_id=0)
        if mode == "human":
            return None
        raise ValueError(f"unsupported render mode: {mode}")

    def _xy_to_image_rc(self, xy: np.ndarray, height: int, width: int) -> Tuple[int, int]:
        span_xy = self.workspace_high[:2] - self.workspace_low[:2]
        norm_xy = (xy - self.workspace_low[:2]) / np.maximum(span_xy, 1e-6)
        col = int(np.clip(round(norm_xy[0] * (width - 1)), 0, width - 1))
        row = int(np.clip(round((1.0 - norm_xy[1]) * (height - 1)), 0, height - 1))
        return row, col

    @staticmethod
    def _draw_disk(image: np.ndarray, row: int, col: int, radius: int, color: np.ndarray) -> None:
        h, w = image.shape[:2]
        r0 = max(0, row - radius)
        r1 = min(h - 1, row + radius)
        c0 = max(0, col - radius)
        c1 = min(w - 1, col + radius)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if (rr - row) * (rr - row) + (cc - col) * (cc - col) <= radius * radius:
                    image[rr, cc] = color

    def _overlay_debug_markers(self, image: np.ndarray) -> np.ndarray:
        rendered = image.copy()
        h, w = rendered.shape[:2]

        target_rc = self._xy_to_image_rc(self._target_pose[:2, 3], h, w)
        self._draw_disk(
            rendered,
            row=target_rc[0],
            col=target_rc[1],
            radius=2,
            color=np.array([20, 130, 255], dtype=np.uint8),
        )

        for idx, xyz in enumerate(self._eef_trail):
            row, col = self._xy_to_image_rc(xyz[:2], h, w)
            alpha = float(idx + 1) / float(max(len(self._eef_trail), 1))
            color = np.array(
                [int(220 * alpha), int(40 + 180 * alpha), int(20 + 20 * alpha)],
                dtype=np.uint8,
            )
            self._draw_disk(rendered, row=row, col=col, radius=1, color=color)

        eef_rc = self._xy_to_image_rc(self._eef_pose[:2, 3], h, w)
        self._draw_disk(
            rendered,
            row=eef_rc[0],
            col=eef_rc[1],
            radius=2,
            color=np.array([255, 70, 40], dtype=np.uint8),
        )
        return rendered

    def serialize(self) -> Dict[str, object]:
        return {
            "env_name": self.environment_name,
            "env_type": "dmcontrol_go_benchmark",
            "board_size": self.board_size,
            "action_shape": [4],
            "action_scale": self.action_scale,
            "press_height": self.press_height,
            "reach_xy_threshold": self.reach_xy_threshold,
            "success_hold_steps": self.success_hold_steps,
            "drive_physical_arm": self.drive_physical_arm,
            "enable_opponent_moves": self.enable_opponent_moves,
            "opening_with_opponent": self.opening_with_opponent,
            "render_carried_stone": self.render_carried_stone,
            "render_eef_overlay": self.render_eef_overlay,
            "eef_overlay_trail": self.eef_overlay_trail,
            "camera_height": self.camera_height,
            "camera_width": self.camera_width,
            "gnugo_path": self.gnugo_path,
        }

    def sample_random_action(self) -> np.ndarray:
        return self._rng.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)

    def close(self) -> None:
        return
