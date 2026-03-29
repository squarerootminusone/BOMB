"""Online task-level text-intervention rollouts for the Go benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np

from ..common import GoResetOptions
from .openvla_adapter import TextMaskCandidateSpec


ONLINE_TASK_INTERVENTION_FORMAT = "openvla_online_task_intervention_v1"
ONLINE_TASK_INTERVENTION_SCHEMA_VERSION = 1

TASK_PHASE_ORDER = (
    "move_to_puck",
    "pick_up_puck",
    "move_puck",
    "drop_puck",
)

TASK_PHASE_LABELS = {
    "move_to_puck": "Move To Puck",
    "pick_up_puck": "Pick Up Puck",
    "move_puck": "Move Puck",
    "drop_puck": "Drop Puck",
}

TASK_PHASE_COLORS = {
    "move_to_puck": (45, 91, 188),
    "pick_up_puck": (227, 146, 58),
    "move_puck": (67, 154, 97),
    "drop_puck": (196, 67, 64),
}

_PHASE_INDEX = {phase: idx for idx, phase in enumerate(TASK_PHASE_ORDER)}


@dataclass
class OnlineInterventionStep:
    timestep: int
    eef_xyz: np.ndarray
    stone_xyz: Optional[np.ndarray]
    action_xyzg: np.ndarray
    gripper_action: float
    phase: str
    move_committed: bool
    stone_grasped: bool
    dist_to_source_xy: float
    dist_to_target_xy: float


@dataclass
class OnlineInterventionAttempt:
    attempt_index: int
    title: str
    instruction: str
    masked: bool
    mask: Optional[TextMaskCandidateSpec]
    success: bool
    timed_out: bool
    steps_taken: int
    phase_counts: Dict[str, int]
    trajectory: List[OnlineInterventionStep]
    target_row: int
    target_col: int
    target_xyz: np.ndarray
    source_xyz: np.ndarray
    final_stone_xyz: Optional[np.ndarray]
    phase_transition_steps: Dict[str, Optional[int]]
    event_steps: Dict[str, Optional[int]]
    ever_grasped: bool
    ever_moved_puck: bool
    ever_released: bool


@dataclass
class OnlineInterventionReport:
    checkpoint: Optional[str]
    prompt_style: Optional[str]
    environment_name: str
    instruction: str
    target_row: int
    target_col: int
    max_steps: int
    text_mask: TextMaskCandidateSpec
    baseline: OnlineInterventionAttempt
    masked_attempts: List[OnlineInterventionAttempt]


class _PredictedStep(Protocol):
    pred_action_xyzg: np.ndarray


class OnlineInterventionPolicy(Protocol):
    prompt_style: Optional[str]

    def predict_step(
        self,
        image: np.ndarray,
        instruction: str,
        masked_instruction_positions: Optional[np.ndarray] = None,
    ) -> _PredictedStep:
        """Predict one normalized benchmark action."""

    def resolve_text_mask_candidate(
        self,
        instruction: str,
        *,
        index: Optional[int] = None,
        label: Optional[str] = None,
        target_row: Optional[int] = None,
        target_col: Optional[int] = None,
    ) -> TextMaskCandidateSpec:
        """Resolve the instruction span to mask for rollout interventions."""


class OnlineInterventionEnv(Protocol):
    environment_name: str

    def reset(self, options: Optional[GoResetOptions] = None):
        """Reset environment state."""

    def step(self, action: np.ndarray):
        """Advance the simulation by one step."""

    def get_eef_pose(self) -> np.ndarray:
        """Return current end-effector pose."""

    def get_target_pose(self) -> np.ndarray:
        """Return target placement pose."""

    def get_source_stone_pose(self) -> np.ndarray:
        """Return source puck / stone pose."""

    def get_task_stone_pose(self) -> Optional[np.ndarray]:
        """Return the currently manipulated or recently placed puck pose."""

    def is_active_stone_grasped(self) -> bool:
        """True while the gripper is holding the active puck."""

    def close(self) -> None:
        """Release simulator resources."""


def _xyz_from_pose(pose_or_xyz: np.ndarray) -> np.ndarray:
    array = np.asarray(pose_or_xyz, dtype=np.float32)
    if array.shape == (4, 4):
        return array[:3, 3].copy().astype(np.float32)
    flat = array.reshape(-1)
    if flat.size < 3:
        raise ValueError(f"expected pose or xyz with at least 3 values, got shape {array.shape}")
    return flat[:3].copy().astype(np.float32)


def _optional_xyz_from_pose(pose_or_xyz: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if pose_or_xyz is None:
        return None
    return _xyz_from_pose(pose_or_xyz)


@dataclass
class _TaskPhaseTracker:
    source_xyz: np.ndarray
    target_xyz: np.ndarray
    move_from_source_xy_threshold: float = 0.03
    lift_from_source_z_threshold: float = 0.012
    ever_grasped: bool = False
    ever_moved_puck: bool = False
    ever_released: bool = False
    first_grasp_step: Optional[int] = None
    first_move_puck_step: Optional[int] = None
    first_drop_step: Optional[int] = None
    phase_transition_steps: Dict[str, Optional[int]] | None = None
    current_phase: str = "move_to_puck"

    def __post_init__(self) -> None:
        self.source_xyz = np.asarray(self.source_xyz, dtype=np.float32).reshape(-1)[:3]
        self.target_xyz = np.asarray(self.target_xyz, dtype=np.float32).reshape(-1)[:3]
        if self.phase_transition_steps is None:
            self.phase_transition_steps = {phase: None for phase in TASK_PHASE_ORDER}
        self.phase_transition_steps["move_to_puck"] = 0

    def update(
        self,
        *,
        timestep: int,
        stone_xyz: Optional[np.ndarray],
        stone_grasped: bool,
        move_committed: bool,
    ) -> str:
        reference_stone_xyz = self.source_xyz if stone_xyz is None else np.asarray(stone_xyz, dtype=np.float32).reshape(-1)[:3]

        if stone_grasped and not self.ever_grasped:
            self.ever_grasped = True
            self.first_grasp_step = int(timestep)

        stone_source_xy_dist = float(np.linalg.norm(reference_stone_xyz[:2] - self.source_xyz[:2]))
        stone_lift = float(reference_stone_xyz[2] - self.source_xyz[2])
        if (
            self.ever_grasped
            and stone_grasped
            and not self.ever_moved_puck
            and self.first_grasp_step is not None
            and int(timestep) > int(self.first_grasp_step)
        ):
            if (
                stone_source_xy_dist >= float(self.move_from_source_xy_threshold)
                or stone_lift >= float(self.lift_from_source_z_threshold)
            ):
                self.ever_moved_puck = True
                self.first_move_puck_step = int(timestep)

        if self.ever_grasped and (not stone_grasped) and not self.ever_released:
            self.ever_released = True
            self.first_drop_step = int(timestep)

        if move_committed and self.first_drop_step is None:
            self.first_drop_step = int(timestep)
        if move_committed:
            self.ever_released = True

        phase = infer_online_task_phase(
            eef_xyz=np.zeros((3,), dtype=np.float32),
            source_xyz=self.source_xyz,
            target_xyz=self.target_xyz,
            stone_xyz=reference_stone_xyz,
            stone_grasped=stone_grasped,
            move_committed=move_committed,
            gripper_action=0.0,
            previous_phase=self.current_phase,
            ever_grasped=self.ever_grasped,
            ever_moved_puck=self.ever_moved_puck,
            ever_released=self.ever_released,
            move_from_source_xy_threshold=self.move_from_source_xy_threshold,
            lift_from_source_z_threshold=self.lift_from_source_z_threshold,
        )
        if self.phase_transition_steps.get(phase) is None:
            self.phase_transition_steps[phase] = int(timestep)
        self.current_phase = str(phase)
        return self.current_phase

    def event_steps(self) -> Dict[str, Optional[int]]:
        return {
            "first_grasp": None if self.first_grasp_step is None else int(self.first_grasp_step),
            "first_move_puck": None if self.first_move_puck_step is None else int(self.first_move_puck_step),
            "first_drop": None if self.first_drop_step is None else int(self.first_drop_step),
        }


def infer_online_task_phase(
    *,
    eef_xyz: np.ndarray,
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    stone_xyz: Optional[np.ndarray],
    stone_grasped: bool,
    move_committed: bool,
    gripper_action: float,
    previous_phase: Optional[str] = None,
    source_xy_threshold: float = 0.05,
    target_xy_threshold: float = 0.045,
    ever_grasped: bool = False,
    ever_moved_puck: bool = False,
    ever_released: bool = False,
    move_from_source_xy_threshold: float = 0.03,
    lift_from_source_z_threshold: float = 0.012,
) -> str:
    source_xyz = np.asarray(source_xyz, dtype=np.float32).reshape(-1)[:3]
    target_xyz = np.asarray(target_xyz, dtype=np.float32).reshape(-1)[:3]
    reference_stone_xyz = source_xyz if stone_xyz is None else np.asarray(stone_xyz, dtype=np.float32).reshape(-1)[:3]
    del eef_xyz
    del gripper_action
    del previous_phase
    del source_xy_threshold
    del target_xy_threshold

    if bool(move_committed) or bool(ever_released):
        return "drop_puck"
    if bool(ever_moved_puck):
        return "move_puck"
    if bool(ever_grasped) or bool(stone_grasped):
        return "pick_up_puck"
    return "move_to_puck"


def _step_from_env(
    *,
    timestep: int,
    env: OnlineInterventionEnv,
    action_xyzg: np.ndarray,
    gripper_action: float,
    phase: str,
    move_committed: bool,
) -> OnlineInterventionStep:
    eef_xyz = _xyz_from_pose(env.get_eef_pose())
    source_xyz = _xyz_from_pose(env.get_source_stone_pose())
    target_xyz = _xyz_from_pose(env.get_target_pose())
    stone_xyz = _optional_xyz_from_pose(env.get_task_stone_pose())
    return OnlineInterventionStep(
        timestep=int(timestep),
        eef_xyz=eef_xyz,
        stone_xyz=None if stone_xyz is None else np.asarray(stone_xyz, dtype=np.float32),
        action_xyzg=np.asarray(action_xyzg, dtype=np.float32).reshape(-1)[:4],
        gripper_action=float(gripper_action),
        phase=str(phase),
        move_committed=bool(move_committed),
        stone_grasped=bool(env.is_active_stone_grasped()),
        dist_to_source_xy=float(np.linalg.norm(eef_xyz[:2] - source_xyz[:2])),
        dist_to_target_xy=float(np.linalg.norm(eef_xyz[:2] - target_xyz[:2])),
    )


def _rollout_attempt(
    *,
    env: OnlineInterventionEnv,
    policy: OnlineInterventionPolicy,
    instruction: str,
    target_row: int,
    target_col: int,
    max_steps: int,
    mask: Optional[TextMaskCandidateSpec],
    attempt_index: int,
    title: str,
    reset_options: GoResetOptions,
    frame_callback: Optional[Callable[[np.ndarray, int], None]] = None,
) -> OnlineInterventionAttempt:
    obs = env.reset(options=reset_options)
    if "agentview_image" not in obs:
        raise RuntimeError("online intervention rollouts require env observations with `agentview_image`")
    if frame_callback is not None:
        frame_callback(np.asarray(obs["agentview_image"], dtype=np.uint8).copy(), 0)

    target_xyz = _xyz_from_pose(env.get_target_pose())
    source_xyz = _xyz_from_pose(env.get_source_stone_pose())
    initial_stone_xyz = _optional_xyz_from_pose(env.get_task_stone_pose())
    phase_tracker = _TaskPhaseTracker(source_xyz=source_xyz, target_xyz=target_xyz)
    initial_phase = phase_tracker.update(
        timestep=0,
        stone_xyz=initial_stone_xyz,
        stone_grasped=bool(env.is_active_stone_grasped()),
        move_committed=False,
    )

    trajectory = [
        _step_from_env(
            timestep=0,
            env=env,
            action_xyzg=np.zeros((4,), dtype=np.float32),
            gripper_action=0.0,
            phase=initial_phase,
            move_committed=False,
        )
    ]

    masked_positions = None
    if mask is not None:
        masked_positions = np.asarray(mask.prompt_token_positions, dtype=np.int64).reshape(-1)

    last_info: Dict[str, object] = {"move_committed": False}
    for timestep in range(max(0, int(max_steps))):
        image = np.asarray(obs["agentview_image"], dtype=np.uint8)
        predicted = policy.predict_step(
            image=image,
            instruction=instruction,
            masked_instruction_positions=masked_positions,
        )
        action_xyzg = np.asarray(predicted.pred_action_xyzg, dtype=np.float32).reshape(-1)
        if action_xyzg.size < 4:
            raise ValueError(f"expected 4D action prediction, got shape {action_xyzg.shape}")

        obs, _reward, done, info = env.step(action_xyzg[:4].astype(np.float32))
        if frame_callback is not None:
            frame_callback(np.asarray(obs["agentview_image"], dtype=np.uint8).copy(), timestep + 1)
        last_info = dict(info)
        current_stone_xyz = _optional_xyz_from_pose(env.get_task_stone_pose())
        phase = phase_tracker.update(
            timestep=timestep + 1,
            stone_xyz=current_stone_xyz,
            stone_grasped=bool(env.is_active_stone_grasped()),
            move_committed=bool(info.get("move_committed", False)),
        )
        trajectory.append(
            _step_from_env(
                timestep=timestep + 1,
                env=env,
                action_xyzg=action_xyzg[:4],
                gripper_action=float(action_xyzg[3]),
                phase=phase,
                move_committed=bool(info.get("move_committed", False)),
            )
        )
        if done:
            break

    success = bool(last_info.get("move_committed", trajectory[-1].move_committed))
    steps_taken = max(0, len(trajectory) - 1)
    timed_out = (not success) and (steps_taken >= int(max_steps))
    phase_counts = {
        phase: int(sum(1 for step in trajectory if step.phase == phase))
        for phase in TASK_PHASE_ORDER
    }
    final_stone_xyz = trajectory[-1].stone_xyz.copy() if trajectory[-1].stone_xyz is not None else None

    return OnlineInterventionAttempt(
        attempt_index=int(attempt_index),
        title=str(title),
        instruction=str(instruction),
        masked=bool(mask is not None),
        mask=mask,
        success=success,
        timed_out=timed_out,
        steps_taken=int(steps_taken),
        phase_counts=phase_counts,
        trajectory=trajectory,
        target_row=int(target_row),
        target_col=int(target_col),
        target_xyz=target_xyz,
        source_xyz=source_xyz,
        final_stone_xyz=final_stone_xyz,
        phase_transition_steps={
            phase: None if step is None else int(step)
            for phase, step in phase_tracker.phase_transition_steps.items()
        },
        event_steps=phase_tracker.event_steps(),
        ever_grasped=bool(phase_tracker.ever_grasped),
        ever_moved_puck=bool(phase_tracker.ever_moved_puck),
        ever_released=bool(phase_tracker.ever_released),
    )


def collect_online_text_mask_report(
    *,
    policy: OnlineInterventionPolicy,
    env_factory: Callable[[], OnlineInterventionEnv],
    instruction: str,
    target_row: int,
    target_col: int,
    max_steps: int = 200,
    masked_attempts: int = 5,
    opening_moves: int = 0,
    stone_color: Optional[str] = "black",
    checkpoint: Optional[str] = None,
    mask_index: Optional[int] = None,
    mask_label: Optional[str] = None,
    baseline_frame_callback: Optional[Callable[[np.ndarray, int], None]] = None,
) -> OnlineInterventionReport:
    env = env_factory()
    try:
        resolved_mask = policy.resolve_text_mask_candidate(
            instruction,
            index=mask_index,
            label=mask_label,
            target_row=target_row,
            target_col=target_col,
        )
        reset_options = GoResetOptions(
            opening_moves=int(opening_moves),
            target_row=int(target_row),
            target_col=int(target_col),
            stone_color=stone_color,
        )
        baseline = _rollout_attempt(
            env=env,
            policy=policy,
            instruction=instruction,
            target_row=target_row,
            target_col=target_col,
            max_steps=max_steps,
            mask=None,
            attempt_index=0,
            title="No Mask",
            reset_options=reset_options,
            frame_callback=baseline_frame_callback,
        )
        masked_runs = [
            _rollout_attempt(
                env=env,
                policy=policy,
                instruction=instruction,
                target_row=target_row,
                target_col=target_col,
                max_steps=max_steps,
                mask=resolved_mask,
                attempt_index=attempt_idx + 1,
                title=f"Mask Attempt {attempt_idx + 1}",
                reset_options=reset_options,
                frame_callback=None,
            )
            for attempt_idx in range(max(0, int(masked_attempts)))
        ]
        return OnlineInterventionReport(
            checkpoint=checkpoint,
            prompt_style=getattr(policy, "prompt_style", None),
            environment_name=str(getattr(env, "environment_name", type(env).__name__)),
            instruction=str(instruction),
            target_row=int(target_row),
            target_col=int(target_col),
            max_steps=int(max_steps),
            text_mask=resolved_mask,
            baseline=baseline,
            masked_attempts=masked_runs,
        )
    finally:
        env.close()


def _mask_to_dict(mask: TextMaskCandidateSpec) -> Dict[str, object]:
    return {
        "index": int(mask.index),
        "label": str(mask.label),
        "prompt_token_positions": np.asarray(mask.prompt_token_positions, dtype=np.int64).tolist(),
        "task_char_start": int(mask.task_char_start),
        "task_char_end": int(mask.task_char_end),
    }


def _step_to_dict(step: OnlineInterventionStep) -> Dict[str, object]:
    return {
        "timestep": int(step.timestep),
        "eef_xyz": np.asarray(step.eef_xyz, dtype=np.float32).tolist(),
        "stone_xyz": None if step.stone_xyz is None else np.asarray(step.stone_xyz, dtype=np.float32).tolist(),
        "action_xyzg": np.asarray(step.action_xyzg, dtype=np.float32).tolist(),
        "gripper_action": float(step.gripper_action),
        "phase": str(step.phase),
        "move_committed": bool(step.move_committed),
        "stone_grasped": bool(step.stone_grasped),
        "dist_to_source_xy": float(step.dist_to_source_xy),
        "dist_to_target_xy": float(step.dist_to_target_xy),
    }


def _attempt_to_dict(attempt: OnlineInterventionAttempt) -> Dict[str, object]:
    return {
        "attempt_index": int(attempt.attempt_index),
        "title": str(attempt.title),
        "instruction": str(attempt.instruction),
        "masked": bool(attempt.masked),
        "mask": None if attempt.mask is None else _mask_to_dict(attempt.mask),
        "success": bool(attempt.success),
        "timed_out": bool(attempt.timed_out),
        "steps_taken": int(attempt.steps_taken),
        "phase_counts": {phase: int(count) for phase, count in attempt.phase_counts.items()},
        "target_row": int(attempt.target_row),
        "target_col": int(attempt.target_col),
        "target_xyz": np.asarray(attempt.target_xyz, dtype=np.float32).tolist(),
        "source_xyz": np.asarray(attempt.source_xyz, dtype=np.float32).tolist(),
        "final_stone_xyz": None
        if attempt.final_stone_xyz is None
        else np.asarray(attempt.final_stone_xyz, dtype=np.float32).tolist(),
        "phase_transition_steps": {
            phase: None if step is None else int(step)
            for phase, step in attempt.phase_transition_steps.items()
        },
        "event_steps": {
            key: None if step is None else int(step)
            for key, step in attempt.event_steps.items()
        },
        "ever_grasped": bool(attempt.ever_grasped),
        "ever_moved_puck": bool(attempt.ever_moved_puck),
        "ever_released": bool(attempt.ever_released),
        "trajectory": [_step_to_dict(step) for step in attempt.trajectory],
    }


def online_text_mask_report_manifest(report: OnlineInterventionReport) -> Dict[str, object]:
    return {
        "trace_format": ONLINE_TASK_INTERVENTION_FORMAT,
        "schema_version": ONLINE_TASK_INTERVENTION_SCHEMA_VERSION,
        "checkpoint": report.checkpoint,
        "prompt_style": report.prompt_style,
        "environment_name": report.environment_name,
        "instruction": report.instruction,
        "target_row": int(report.target_row),
        "target_col": int(report.target_col),
        "max_steps": int(report.max_steps),
        "text_mask": _mask_to_dict(report.text_mask),
        "baseline": _attempt_to_dict(report.baseline),
        "masked_attempts": [_attempt_to_dict(attempt) for attempt in report.masked_attempts],
    }
