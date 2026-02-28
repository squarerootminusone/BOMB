"""Source demonstration collection for the Go benchmark."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .common import GoResetOptions
from .dataset_io import EpisodeRecord, write_dataset
from .env_factory import create_benchmark_env
from .mimicgen_interface import MG_GoJacoSingleMove


def _stack_obs(obs_seq: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = obs_seq[0].keys()
    return {k: np.asarray([obs[k] for obs in obs_seq]) for k in keys}


def _append_transition(
    env: Any,
    env_interface: MG_GoJacoSingleMove,
    action: np.ndarray,
    states: List[np.ndarray],
    observations: List[Dict[str, np.ndarray]],
    datagen_infos: List[object],
    actions: List[np.ndarray],
) -> Tuple[bool, Dict[str, object]]:
    state = env.get_state()["states"]
    obs = env.get_observation()
    datagen_info = env_interface.get_datagen_info(action=action)

    _, _, done, info = env.step(action)

    states.append(state)
    observations.append(obs)
    datagen_infos.append(datagen_info)
    actions.append(action.astype(np.float32))

    return done, info


def _drive_to_pose(
    env: Any,
    env_interface: MG_GoJacoSingleMove,
    goal_xyz: np.ndarray,
    num_steps: int,
    gripper: float,
    controller_divisor: float,
    states: List[np.ndarray],
    observations: List[Dict[str, np.ndarray]],
    datagen_infos: List[object],
    actions: List[np.ndarray],
    stop_on_success: bool = True,
    stop_on_done: bool = True,
) -> bool:
    controller_divisor = max(float(controller_divisor), 1e-6)
    for _ in range(int(num_steps)):
        cur_xyz = env.get_eef_pose()[:3, 3]
        delta = goal_xyz - cur_xyz
        # Per-step displacement is approximately delta / controller_divisor.
        action_xyz = np.clip(delta / (env.action_scale * controller_divisor), -1.0, 1.0)
        action = np.concatenate([action_xyz, np.array([gripper], dtype=np.float32)], axis=0)

        done, _ = _append_transition(
            env=env,
            env_interface=env_interface,
            action=action,
            states=states,
            observations=observations,
            datagen_infos=datagen_infos,
            actions=actions,
        )
        if (stop_on_success and env.is_success()["task"]) or (stop_on_done and done):
            return True
    return env.is_success()["task"]


def _compute_side_hover_xyz(
    env: Any,
    target_xyz: np.ndarray,
    side_margin: float,
) -> np.ndarray:
    board_low_xy, board_high_xy = env.get_board_xy_bounds()
    board_center = env.get_board_origin_pose()[:3, 3]
    side_margin = max(0.02, float(side_margin))

    candidate_xy = np.asarray(
        [
            [board_low_xy[0] - side_margin, board_center[1]],
            [board_high_xy[0] + side_margin, board_center[1]],
            [board_center[0], board_low_xy[1] - side_margin],
            [board_center[0], board_high_xy[1] + side_margin],
        ],
        dtype=np.float32,
    )
    dists = np.linalg.norm(candidate_xy - target_xyz[:2].reshape(1, 2), axis=1)
    side_xy = candidate_xy[int(np.argmax(dists))]

    side_xyz = np.array(
        [
            float(side_xy[0]),
            float(side_xy[1]),
            max(env.hover_height, env.press_height + 0.09),
        ],
        dtype=np.float32,
    )
    pad_xy = 0.02
    low = env.workspace_low.copy()
    high = env.workspace_high.copy()
    low[:2] += pad_xy
    high[:2] -= pad_xy
    return np.clip(side_xyz, low, high).astype(np.float32)


def _collect_single_episode(
    env: Any,
    env_interface: MG_GoJacoSingleMove,
    controller_divisor: float,
    detour_steps: int,
    detour_radius: float,
    approach_steps: int,
    press_steps: int,
    retreat_steps: int,
    side_transfer_steps: int,
    side_margin: float,
    recovery_steps: int,
) -> Tuple[EpisodeRecord, bool]:
    initial_state = env.get_state()

    states: List[np.ndarray] = []
    observations: List[Dict[str, np.ndarray]] = []
    datagen_infos: List[object] = []
    actions: List[np.ndarray] = []

    target_xyz = env.get_target_pose()[:3, 3]
    source_xyz = (
        env.get_source_stone_pose()[:3, 3]
        if hasattr(env, "get_source_stone_pose")
        else target_xyz.copy()
    )
    hover_xyz = target_xyz.copy()
    hover_xyz[2] = max(env.hover_height * 0.7, env.press_height + 0.05)
    side_hover_xyz = _compute_side_hover_xyz(
        env=env,
        target_xyz=target_xyz,
        side_margin=side_margin,
    )

    press_xyz = target_xyz.copy()
    press_xyz[2] = max(0.0, env.press_height - 0.015)

    source_hover_xyz = source_xyz.copy()
    source_hover_xyz[2] = max(env.hover_height * 0.65, env.press_height + 0.05)
    source_press_xyz = source_xyz.copy()
    source_press_xyz[2] = max(env.press_height + 0.002, source_xyz[2] - 0.006)

    # Optional phase 0: move to a random detour waypoint to create longer / larger motions.
    if detour_steps > 0 and detour_radius > 0.0:
        detour_xyz = hover_xyz.copy()
        detour_xyz[:2] += env.sample_random_action()[:2] * float(detour_radius)
        detour_xyz = np.clip(detour_xyz, env.workspace_low, env.workspace_high)
        _drive_to_pose(
            env=env,
            env_interface=env_interface,
            goal_xyz=detour_xyz,
            num_steps=detour_steps,
            gripper=0.0,
            controller_divisor=controller_divisor,
            states=states,
            observations=observations,
            datagen_infos=datagen_infos,
            actions=actions,
            stop_on_success=False,
            stop_on_done=True,
        )

    # Optional phase 0b: explicitly move to a side waypoint before approaching
    # the board target to mimic a pick-then-place transfer arc.
    if side_transfer_steps > 0:
        _drive_to_pose(
            env=env,
            env_interface=env_interface,
            goal_xyz=side_hover_xyz,
            num_steps=side_transfer_steps,
            gripper=0.0,
            controller_divisor=controller_divisor,
            states=states,
            observations=observations,
            datagen_infos=datagen_infos,
            actions=actions,
            stop_on_success=False,
            stop_on_done=True,
        )

    # Physical-pick phase (robosuite backend): approach source stone and grasp it
    # before transfer to the board target.
    if hasattr(env, "get_source_stone_pose"):
        _drive_to_pose(
            env=env,
            env_interface=env_interface,
            goal_xyz=source_hover_xyz,
            num_steps=max(6, approach_steps // 2),
            gripper=0.0,
            controller_divisor=controller_divisor,
            states=states,
            observations=observations,
            datagen_infos=datagen_infos,
            actions=actions,
            stop_on_success=False,
            stop_on_done=True,
        )
        _drive_to_pose(
            env=env,
            env_interface=env_interface,
            goal_xyz=source_press_xyz,
            num_steps=max(6, press_steps),
            gripper=0.0,
            controller_divisor=controller_divisor,
            states=states,
            observations=observations,
            datagen_infos=datagen_infos,
            actions=actions,
            stop_on_success=False,
            stop_on_done=True,
        )
        _drive_to_pose(
            env=env,
            env_interface=env_interface,
            goal_xyz=source_press_xyz,
            num_steps=max(5, press_steps // 2),
            gripper=1.0,
            controller_divisor=controller_divisor,
            states=states,
            observations=observations,
            datagen_infos=datagen_infos,
            actions=actions,
            stop_on_success=False,
            stop_on_done=True,
        )
        _drive_to_pose(
            env=env,
            env_interface=env_interface,
            goal_xyz=source_hover_xyz,
            num_steps=max(6, retreat_steps),
            gripper=1.0,
            controller_divisor=controller_divisor,
            states=states,
            observations=observations,
            datagen_infos=datagen_infos,
            actions=actions,
            stop_on_success=False,
            stop_on_done=True,
        )

    # Phase 1: approach target intersection.
    success = _drive_to_pose(
        env=env,
        env_interface=env_interface,
        goal_xyz=hover_xyz,
        num_steps=approach_steps,
        gripper=1.0,
        controller_divisor=controller_divisor,
        states=states,
        observations=observations,
        datagen_infos=datagen_infos,
        actions=actions,
    )

    # Phase 2: descend and press to commit the move.
    if not success:
        success = _drive_to_pose(
            env=env,
            env_interface=env_interface,
            goal_xyz=press_xyz,
            num_steps=press_steps,
            gripper=1.0,
            controller_divisor=controller_divisor,
            states=states,
            observations=observations,
            datagen_infos=datagen_infos,
            actions=actions,
        )

    # Phase 3: retreat to hover.
    _drive_to_pose(
        env=env,
        env_interface=env_interface,
        goal_xyz=hover_xyz,
        num_steps=retreat_steps,
        gripper=0.0,
        controller_divisor=controller_divisor,
        states=states,
        observations=observations,
        datagen_infos=datagen_infos,
        actions=actions,
        stop_on_success=False,
        stop_on_done=True,
    )

    # Optional phase 4: move to side waypoint after committing the move to
    # produce large board-to-side displacement in recorded trajectories.
    if side_transfer_steps > 0:
        _drive_to_pose(
            env=env,
            env_interface=env_interface,
            goal_xyz=side_hover_xyz,
            num_steps=side_transfer_steps,
            gripper=0.0,
            controller_divisor=controller_divisor,
            states=states,
            observations=observations,
            datagen_infos=datagen_infos,
            actions=actions,
            stop_on_success=False,
            stop_on_done=False,
        )

    # Recovery: if needed, retry short press sequence with small jitter.
    if not success:
        for _ in range(recovery_steps):
            jitter = env.sample_random_action()[:3] * 0.01
            retry_xyz = press_xyz + jitter
            success = _drive_to_pose(
                env=env,
                env_interface=env_interface,
                goal_xyz=retry_xyz,
                num_steps=3,
                gripper=1.0,
                controller_divisor=controller_divisor,
                states=states,
                observations=observations,
                datagen_infos=datagen_infos,
                actions=actions,
            )
            if success:
                break

    episode = EpisodeRecord(
        actions=np.asarray(actions, dtype=np.float32),
        states=np.asarray(states, dtype=np.float32),
        observations=_stack_obs(observations),
        datagen_infos=datagen_infos,
        initial_state=initial_state,
    )
    return episode, bool(success)


def collect_source_demonstrations(
    output_path: str,
    environment_name: str = "go_7x7",
    num_demos: int = 50,
    seed: int = 0,
    max_attempts_per_demo: int = 5,
    opening_moves_min: int = 0,
    opening_moves_max: int = 8,
    include_image_obs: bool = True,
    camera_height: int = 84,
    camera_width: int = 84,
    gnugo_path: str | None = None,
    action_scale: float = 0.03,
    success_hold_steps: int = 0,
    controller_divisor: float = 2.0,
    detour_steps: int = 0,
    detour_radius: float = 0.0,
    approach_steps: int = 10,
    press_steps: int = 6,
    retreat_steps: int = 6,
    side_transfer_steps: int = 0,
    side_margin: float = 0.16,
    recovery_steps: int = 3,
    drive_physical_arm: bool = True,
    enable_opponent_moves: bool = False,
    opening_with_opponent: bool = True,
    render_carried_stone: bool = True,
    render_eef_overlay: bool = True,
    eef_overlay_trail: int = 10,
    robot: str = "Panda",
    gripper_types: str = "default",
) -> Dict[str, object]:
    """Collect source demonstrations for MimicGen using scripted control."""
    if num_demos <= 0:
        raise ValueError("num_demos must be > 0")

    rng = np.random.RandomState(seed)
    env = create_benchmark_env(
        seed=seed,
        environment_name=environment_name,
        include_image_obs=include_image_obs,
        camera_height=camera_height,
        camera_width=camera_width,
        gnugo_path=gnugo_path,
        action_scale=action_scale,
        success_hold_steps=success_hold_steps,
        drive_physical_arm=drive_physical_arm,
        enable_opponent_moves=enable_opponent_moves,
        opening_with_opponent=opening_with_opponent,
        render_carried_stone=render_carried_stone,
        render_eef_overlay=render_eef_overlay,
        eef_overlay_trail=eef_overlay_trail,
        robot=robot,
        gripper_types=gripper_types,
    )
    env_interface = MG_GoJacoSingleMove(env=env)

    episodes: List[EpisodeRecord] = []
    attempts = 0
    max_attempts = num_demos * max_attempts_per_demo

    while (len(episodes) < num_demos) and (attempts < max_attempts):
        attempts += 1
        print(
            f"[collect] attempt {attempts}/{max_attempts} demos={len(episodes)}/{num_demos}",
            flush=True,
        )

        opening_moves = int(rng.randint(opening_moves_min, opening_moves_max + 1))
        env.reset(options=GoResetOptions(opening_moves=opening_moves))

        episode, success = _collect_single_episode(
            env=env,
            env_interface=env_interface,
            controller_divisor=controller_divisor,
            detour_steps=detour_steps,
            detour_radius=detour_radius,
            approach_steps=approach_steps,
            press_steps=press_steps,
            retreat_steps=retreat_steps,
            side_transfer_steps=side_transfer_steps,
            side_margin=side_margin,
            recovery_steps=recovery_steps,
        )

        if success:
            episodes.append(episode)
            print(
                f"[collect] success demos={len(episodes)}/{num_demos} "
                f"steps={int(episode.actions.shape[0])}",
                flush=True,
            )
        else:
            print("[collect] attempt failed", flush=True)

    if len(episodes) < num_demos:
        raise RuntimeError(
            f"failed to collect requested demos: got {len(episodes)} / {num_demos} "
            f"after {attempts} attempts"
        )

    write_dataset(
        output_path=output_path,
        episodes=episodes,
        env_meta=env.serialize(),
        env_interface_name=type(env_interface).__name__,
        env_interface_type=type(env_interface).INTERFACE_TYPE,
    )

    return {
        "output_path": output_path,
        "num_demos": len(episodes),
        "attempts": attempts,
        "success_rate": float(len(episodes) / attempts),
        "env_interface": type(env_interface).__name__,
        "env_interface_type": type(env_interface).INTERFACE_TYPE,
    }
