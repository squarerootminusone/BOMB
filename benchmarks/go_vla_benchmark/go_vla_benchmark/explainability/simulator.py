"""Helpers for simulator-backed explainability inputs without an HDF5 dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..common import GoResetOptions
from ..rlds_preprocessing import (
    INSTRUCTION_TEMPLATES,
    compute_rlds_keep_indices,
    extract_action_4d,
    format_instruction_template,
    remap_gripper_to_openvla,
)
from .core import EpisodeClip


DEFAULT_SIMULATOR_OPENING_MIN = 0
DEFAULT_SIMULATOR_OPENING_MAX = 8
DEFAULT_SIMULATOR_MAX_ATTEMPTS_PER_DEMO = 6
DEFAULT_SIMULATOR_CAMERA_SIZE = 256
DEFAULT_SIMULATOR_ENVIRONMENT = "robosuite_go_5x5_rigid_bodies"
DEFAULT_SIMULATOR_MAX_STEPS = 200


@dataclass
class SimulatorOnlineDemoSpec:
    demo_key: str
    instruction: str
    target_row: int
    target_col: int
    stone_color: str
    reset_options: GoResetOptions
    reference_frame_index: int
    reference_image: np.ndarray


def _build_instruction(*, demo_index: int, target_row: int, target_col: int, stone_color: str) -> str:
    rng = np.random.RandomState(seed=int(demo_index))
    template = INSTRUCTION_TEMPLATES[int(rng.randint(len(INSTRUCTION_TEMPLATES)))]
    return format_instruction_template(template, color=stone_color, row=target_row, col=target_col)


def _make_demo_key(demo_index: int) -> str:
    return f"sim_demo_{int(demo_index):03d}"


def _sanitize_move_history(move_history: np.ndarray) -> Optional[np.ndarray]:
    history = np.asarray(move_history, dtype=np.int32).reshape(-1)
    valid = history[history >= 0]
    if valid.size == 0:
        return None
    return valid.astype(np.int32)


def _sample_reset_options(
    *,
    rng: np.random.RandomState,
    opening_moves_min: int,
    opening_moves_max: int,
) -> GoResetOptions:
    opening_low = max(0, int(opening_moves_min))
    opening_high = max(opening_low, int(opening_moves_max))
    opening_moves = int(rng.randint(opening_low, opening_high + 1))
    stone_color = str(rng.choice(np.asarray(["black", "white"], dtype=object)))
    reset_seed = int(rng.randint(0, np.iinfo(np.int32).max))
    return GoResetOptions(
        opening_moves=opening_moves,
        stone_color=stone_color,
        reset_seed=reset_seed,
    )


def _episode_to_clip(
    *,
    demo_key: str,
    instruction: str,
    episode,
    stride: int,
    max_steps: int,
) -> Optional[EpisodeClip]:
    observations = dict(episode.observations)
    if "agentview_image" not in observations:
        raise RuntimeError("simulator explainability clips require `agentview_image` observations")

    actions_4d = remap_gripper_to_openvla(extract_action_4d(np.asarray(episode.actions, dtype=np.float32)))
    frame_indices = compute_rlds_keep_indices(actions_4d)
    frame_indices = frame_indices[:: max(1, int(stride))]
    if int(max_steps) > 0:
        frame_indices = frame_indices[: int(max_steps)]
    if frame_indices.size == 0:
        return None

    images = np.asarray(observations["agentview_image"], dtype=np.uint8)
    return EpisodeClip(
        demo_key=str(demo_key),
        instruction=str(instruction),
        images=np.asarray(images[frame_indices], dtype=np.uint8),
        gt_actions=np.asarray(actions_4d[frame_indices], dtype=np.float32),
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
    )


def collect_simulator_online_demo_specs(
    *,
    num_demos: int,
    seed: int,
    environment_name: str = DEFAULT_SIMULATOR_ENVIRONMENT,
    camera_size: int = DEFAULT_SIMULATOR_CAMERA_SIZE,
    max_steps: int = DEFAULT_SIMULATOR_MAX_STEPS,
    opening_moves_min: int = DEFAULT_SIMULATOR_OPENING_MIN,
    opening_moves_max: int = DEFAULT_SIMULATOR_OPENING_MAX,
    robot: str = "Panda",
    gripper_types: str = "default",
) -> List[SimulatorOnlineDemoSpec]:
    if int(num_demos) <= 0:
        raise ValueError("num_demos must be > 0 for simulator-backed online reports")

    from ..env_factory import create_benchmark_env

    rng = np.random.RandomState(int(seed))
    env = create_benchmark_env(
        seed=int(seed),
        environment_name=environment_name,
        include_image_obs=True,
        camera_height=int(camera_size),
        camera_width=int(camera_size),
        max_steps=int(max_steps),
        success_hold_steps=0,
        render_eef_overlay=False,
        robot=robot,
        gripper_types=gripper_types,
    )
    try:
        specs: List[SimulatorOnlineDemoSpec] = []
        for demo_index in range(int(num_demos)):
            sampled_reset = _sample_reset_options(
                rng=rng,
                opening_moves_min=opening_moves_min,
                opening_moves_max=opening_moves_max,
            )
            obs = env.reset(options=sampled_reset)
            if "agentview_image" not in obs:
                raise RuntimeError("simulator-backed online reports require `agentview_image` observations")

            target_row, target_col = env.get_target_intersection()
            move_history = _sanitize_move_history(env.get_move_history())
            stone_color = str(sampled_reset.stone_color or getattr(env, "_stone_color", "black"))
            resolved_reset = GoResetOptions(
                opening_moves=int(sampled_reset.opening_moves),
                opening_move_history=None if move_history is None else move_history.copy(),
                target_row=int(target_row),
                target_col=int(target_col),
                stone_color=stone_color,
                reset_seed=sampled_reset.reset_seed,
            )
            specs.append(
                SimulatorOnlineDemoSpec(
                    demo_key=_make_demo_key(demo_index),
                    instruction=_build_instruction(
                        demo_index=demo_index,
                        target_row=int(target_row),
                        target_col=int(target_col),
                        stone_color=stone_color,
                    ),
                    target_row=int(target_row),
                    target_col=int(target_col),
                    stone_color=stone_color,
                    reset_options=resolved_reset,
                    reference_frame_index=0,
                    reference_image=np.asarray(obs["agentview_image"], dtype=np.uint8),
                )
            )
        return specs
    finally:
        env.close()


def collect_simulator_episode_clips(
    *,
    num_demos: int,
    seed: int,
    stride: int,
    max_steps: int,
    environment_name: str = DEFAULT_SIMULATOR_ENVIRONMENT,
    camera_size: int = DEFAULT_SIMULATOR_CAMERA_SIZE,
    env_max_steps: int = DEFAULT_SIMULATOR_MAX_STEPS,
    opening_moves_min: int = DEFAULT_SIMULATOR_OPENING_MIN,
    opening_moves_max: int = DEFAULT_SIMULATOR_OPENING_MAX,
    max_attempts_per_demo: int = DEFAULT_SIMULATOR_MAX_ATTEMPTS_PER_DEMO,
    robot: str = "Panda",
    gripper_types: str = "default",
) -> List[EpisodeClip]:
    if int(num_demos) <= 0:
        raise ValueError("num_demos must be > 0 for simulator-backed clip collection")

    from ..collect import _collect_single_episode
    from ..env_factory import create_benchmark_env
    from ..mimicgen_interface import MG_GoJacoSingleMove

    rng = np.random.RandomState(int(seed))
    env = create_benchmark_env(
        seed=int(seed),
        environment_name=environment_name,
        include_image_obs=True,
        camera_height=int(camera_size),
        camera_width=int(camera_size),
        max_steps=int(env_max_steps),
        success_hold_steps=0,
        robot=robot,
        gripper_types=gripper_types,
    )
    env_interface = MG_GoJacoSingleMove(env=env)
    required_env_attrs = ("table_top_z", "stone_height", "board_thickness")
    missing_env_attrs = [name for name in required_env_attrs if not hasattr(env, name)]
    if missing_env_attrs:
        raise RuntimeError(
            "simulator-backed clip collection currently requires the robosuite Go backend; "
            f"missing env attributes: {missing_env_attrs}"
        )

    max_attempts = int(num_demos) * max(1, int(max_attempts_per_demo))
    clips: List[EpisodeClip] = []
    attempts = 0
    try:
        while len(clips) < int(num_demos) and attempts < max_attempts:
            attempts += 1
            sampled_reset = _sample_reset_options(
                rng=rng,
                opening_moves_min=opening_moves_min,
                opening_moves_max=opening_moves_max,
            )
            env.reset(options=sampled_reset)
            target_row, target_col = env.get_target_intersection()
            stone_color = str(sampled_reset.stone_color or getattr(env, "_stone_color", "black"))
            instruction = _build_instruction(
                demo_index=len(clips),
                target_row=int(target_row),
                target_col=int(target_col),
                stone_color=stone_color,
            )

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
                hover_height_noise=0.0,
            )
            if not success:
                continue

            clip = _episode_to_clip(
                demo_key=_make_demo_key(len(clips)),
                instruction=instruction,
                episode=episode,
                stride=int(stride),
                max_steps=int(max_steps),
            )
            if clip is not None:
                clips.append(clip)

        if len(clips) < int(num_demos):
            raise RuntimeError(
                f"failed to collect requested simulator demos: got {len(clips)} / {int(num_demos)} "
                f"after {attempts} attempts"
            )
        return clips
    finally:
        env.close()
