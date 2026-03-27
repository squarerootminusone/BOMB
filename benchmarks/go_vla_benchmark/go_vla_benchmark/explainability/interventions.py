"""Collection and IO helpers for intervention-test explainability stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .core import EpisodeClip, LocalExplanationStep, _to_object_array, compute_episode_scores


INTERVENTION_TRACE_FORMAT = "openvla_intervention_tests_v1"
INTERVENTION_SCHEMA_VERSION = 2


@dataclass
class InterventionCandidateEffect:
    index: int
    label: str
    score: float
    pred_action_xyzg: Optional[np.ndarray] = None
    raw_pred_action: Optional[np.ndarray] = None
    target_token_ids: Optional[np.ndarray] = None
    target_token_probs: Optional[np.ndarray] = None
    target_token_bin_indices: Optional[np.ndarray] = None
    target_token_bin_centers: Optional[np.ndarray] = None
    task_char_start: Optional[int] = None
    task_char_end: Optional[int] = None


@dataclass
class InterventionScan:
    effect_map: np.ndarray
    top_candidates: List[InterventionCandidateEffect]


@dataclass
class CounterfactualEdit:
    step_rank: int
    edit_type: str
    index: int
    label: str
    single_effect_score: float
    cumulative_sequence_logprob: float
    cumulative_sequence_logprob_drop: float
    changed_prediction: bool
    pred_action_xyzg: np.ndarray
    raw_pred_action: np.ndarray
    target_token_ids: np.ndarray
    target_token_probs: np.ndarray
    target_token_bin_indices: Optional[np.ndarray] = None
    target_token_bin_centers: Optional[np.ndarray] = None


@dataclass
class InterventionStepTrace:
    baseline_pred_action_xyzg: np.ndarray
    baseline_raw_pred_action: np.ndarray
    baseline_target_token_ids: np.ndarray
    baseline_target_token_probs: np.ndarray
    patch_occlusion: Optional[InterventionScan]
    text_masking: Optional[InterventionScan]
    counterfactual_edits: List[CounterfactualEdit]
    counterfactual_success: bool
    prompt: str
    task_text: str
    text_token_ids: np.ndarray
    text_tokens: List[str]
    baseline_target_token_bin_indices: Optional[np.ndarray] = None
    baseline_target_token_bin_centers: Optional[np.ndarray] = None


@dataclass
class EpisodeInterventionTrace:
    demo_key: str
    pred_actions: np.ndarray
    step_errors: np.ndarray
    mean_l1: float
    max_l1: float
    score: float
    steps: List[InterventionStepTrace]


@dataclass
class InterventionTraceBundle:
    dataset_path: Path
    checkpoint: Optional[str]
    prompt_style: Optional[str]
    dataset_adapter: Optional[str]
    model_adapter: Optional[str]
    trace_format: str
    schema_version: int
    traces: Dict[str, EpisodeInterventionTrace]
    frame_indices_by_key: Dict[str, np.ndarray]
    demo_keys: List[str]


class InterventionModelAdapter(Protocol):
    """Model-side interface for intervention tests."""

    def explain_step(self, image: np.ndarray, instruction: str) -> LocalExplanationStep:
        """Run the clean baseline step."""

    def patch_occlusion(
        self,
        image: np.ndarray,
        instruction: str,
        baseline_step: LocalExplanationStep,
        top_k: int,
    ) -> InterventionScan:
        """Evaluate image-patch occlusion interventions."""

    def text_masking(
        self,
        image: np.ndarray,
        instruction: str,
        baseline_step: LocalExplanationStep,
        top_k: int,
    ) -> InterventionScan:
        """Evaluate instruction word / phrase masking interventions."""

    def minimal_counterfactual_edits(
        self,
        image: np.ndarray,
        instruction: str,
        baseline_step: LocalExplanationStep,
        patch_occlusion: Optional[InterventionScan],
        text_masking: Optional[InterventionScan],
        max_edits: int,
    ) -> tuple[List[CounterfactualEdit], bool]:
        """Search for a small intervention set that changes the predicted output."""


def collect_episode_intervention_traces(
    clips: Sequence[EpisodeClip],
    model_adapter: InterventionModelAdapter,
    run_patch_occlusion: bool = True,
    run_text_masking: bool = True,
    run_counterfactual: bool = True,
    top_k: int = 8,
    max_counterfactual_edits: int = 4,
) -> List[EpisodeInterventionTrace]:
    traces: List[EpisodeInterventionTrace] = []
    for clip in clips:
        pred_actions: List[np.ndarray] = []
        step_traces: List[InterventionStepTrace] = []

        for frame in clip.images:
            baseline_step = model_adapter.explain_step(frame, clip.instruction)
            pred_actions.append(np.asarray(baseline_step.pred_action_xyzg, dtype=np.float32))

            patch_scan = None
            if run_patch_occlusion:
                patch_scan = model_adapter.patch_occlusion(
                    image=frame,
                    instruction=clip.instruction,
                    baseline_step=baseline_step,
                    top_k=top_k,
                )

            text_scan = None
            if run_text_masking:
                text_scan = model_adapter.text_masking(
                    image=frame,
                    instruction=clip.instruction,
                    baseline_step=baseline_step,
                    top_k=top_k,
                )

            counterfactual_edits: List[CounterfactualEdit] = []
            counterfactual_success = False
            if run_counterfactual:
                counterfactual_edits, counterfactual_success = model_adapter.minimal_counterfactual_edits(
                    image=frame,
                    instruction=clip.instruction,
                    baseline_step=baseline_step,
                    patch_occlusion=patch_scan,
                    text_masking=text_scan,
                    max_edits=max_counterfactual_edits,
                )

            step_traces.append(
                InterventionStepTrace(
                    baseline_pred_action_xyzg=np.asarray(baseline_step.pred_action_xyzg, dtype=np.float32),
                    baseline_raw_pred_action=np.asarray(baseline_step.raw_pred_action, dtype=np.float32),
                    baseline_target_token_ids=np.asarray(baseline_step.target_token_ids, dtype=np.int64),
                    baseline_target_token_probs=np.asarray(baseline_step.target_token_probs, dtype=np.float32),
                    patch_occlusion=patch_scan,
                    text_masking=text_scan,
                    counterfactual_edits=counterfactual_edits,
                    counterfactual_success=bool(counterfactual_success),
                    prompt=baseline_step.prompt,
                    task_text=baseline_step.task_text,
                    text_token_ids=np.asarray(baseline_step.text_token_ids, dtype=np.int64),
                    text_tokens=list(baseline_step.text_tokens),
                    baseline_target_token_bin_indices=None
                    if baseline_step.target_token_bin_indices is None
                    else np.asarray(baseline_step.target_token_bin_indices, dtype=np.int64),
                    baseline_target_token_bin_centers=None
                    if baseline_step.target_token_bin_centers is None
                    else np.asarray(baseline_step.target_token_bin_centers, dtype=np.float32),
                )
            )

        pred_np = np.stack(pred_actions, axis=0)
        step_errors, mean_l1, max_l1, score = compute_episode_scores(clip.gt_actions, pred_np)
        traces.append(
            EpisodeInterventionTrace(
                demo_key=clip.demo_key,
                pred_actions=pred_np,
                step_errors=step_errors,
                mean_l1=mean_l1,
                max_l1=max_l1,
                score=score,
                steps=step_traces,
            )
        )

    return traces


def _candidate_to_dict(candidate: InterventionCandidateEffect) -> Dict[str, object]:
    return {
        "index": int(candidate.index),
        "label": str(candidate.label),
        "score": float(candidate.score),
        "pred_action_xyzg": None if candidate.pred_action_xyzg is None else np.asarray(candidate.pred_action_xyzg).tolist(),
        "raw_pred_action": None if candidate.raw_pred_action is None else np.asarray(candidate.raw_pred_action).tolist(),
        "target_token_ids": None if candidate.target_token_ids is None else np.asarray(candidate.target_token_ids).tolist(),
        "target_token_probs": None if candidate.target_token_probs is None else np.asarray(candidate.target_token_probs).tolist(),
        "target_token_bin_indices": None
        if candidate.target_token_bin_indices is None
        else np.asarray(candidate.target_token_bin_indices).tolist(),
        "target_token_bin_centers": None
        if candidate.target_token_bin_centers is None
        else np.asarray(candidate.target_token_bin_centers).tolist(),
        "task_char_start": None if candidate.task_char_start is None else int(candidate.task_char_start),
        "task_char_end": None if candidate.task_char_end is None else int(candidate.task_char_end),
    }


def _candidate_from_dict(payload: Dict[str, object]) -> InterventionCandidateEffect:
    return InterventionCandidateEffect(
        index=int(payload["index"]),
        label=str(payload["label"]),
        score=float(payload["score"]),
        pred_action_xyzg=None if payload.get("pred_action_xyzg") is None else np.asarray(payload["pred_action_xyzg"], dtype=np.float32),
        raw_pred_action=None if payload.get("raw_pred_action") is None else np.asarray(payload["raw_pred_action"], dtype=np.float32),
        target_token_ids=None if payload.get("target_token_ids") is None else np.asarray(payload["target_token_ids"], dtype=np.int64),
        target_token_probs=None if payload.get("target_token_probs") is None else np.asarray(payload["target_token_probs"], dtype=np.float32),
        target_token_bin_indices=None
        if payload.get("target_token_bin_indices") is None
        else np.asarray(payload["target_token_bin_indices"], dtype=np.int64),
        target_token_bin_centers=None
        if payload.get("target_token_bin_centers") is None
        else np.asarray(payload["target_token_bin_centers"], dtype=np.float32),
        task_char_start=None if payload.get("task_char_start") is None else int(payload["task_char_start"]),
        task_char_end=None if payload.get("task_char_end") is None else int(payload["task_char_end"]),
    )


def _scan_to_dict(scan: Optional[InterventionScan]) -> Optional[Dict[str, object]]:
    if scan is None:
        return None
    return {
        "effect_map": np.asarray(scan.effect_map, dtype=np.float32).tolist(),
        "top_candidates": [_candidate_to_dict(candidate) for candidate in scan.top_candidates],
    }


def _scan_from_dict(payload: Optional[Dict[str, object]]) -> Optional[InterventionScan]:
    if payload is None:
        return None
    return InterventionScan(
        effect_map=np.asarray(payload["effect_map"], dtype=np.float32),
        top_candidates=[_candidate_from_dict(item) for item in payload.get("top_candidates", [])],
    )


def _counterfactual_to_dict(edit: CounterfactualEdit) -> Dict[str, object]:
    return {
        "step_rank": int(edit.step_rank),
        "edit_type": str(edit.edit_type),
        "index": int(edit.index),
        "label": str(edit.label),
        "single_effect_score": float(edit.single_effect_score),
        "cumulative_sequence_logprob": float(edit.cumulative_sequence_logprob),
        "cumulative_sequence_logprob_drop": float(edit.cumulative_sequence_logprob_drop),
        "changed_prediction": bool(edit.changed_prediction),
        "pred_action_xyzg": np.asarray(edit.pred_action_xyzg, dtype=np.float32).tolist(),
        "raw_pred_action": np.asarray(edit.raw_pred_action, dtype=np.float32).tolist(),
        "target_token_ids": np.asarray(edit.target_token_ids, dtype=np.int64).tolist(),
        "target_token_probs": np.asarray(edit.target_token_probs, dtype=np.float32).tolist(),
        "target_token_bin_indices": None
        if edit.target_token_bin_indices is None
        else np.asarray(edit.target_token_bin_indices).tolist(),
        "target_token_bin_centers": None
        if edit.target_token_bin_centers is None
        else np.asarray(edit.target_token_bin_centers).tolist(),
    }


def _counterfactual_from_dict(payload: Dict[str, object]) -> CounterfactualEdit:
    return CounterfactualEdit(
        step_rank=int(payload["step_rank"]),
        edit_type=str(payload["edit_type"]),
        index=int(payload["index"]),
        label=str(payload["label"]),
        single_effect_score=float(payload["single_effect_score"]),
        cumulative_sequence_logprob=float(payload["cumulative_sequence_logprob"]),
        cumulative_sequence_logprob_drop=float(payload["cumulative_sequence_logprob_drop"]),
        changed_prediction=bool(payload["changed_prediction"]),
        pred_action_xyzg=np.asarray(payload["pred_action_xyzg"], dtype=np.float32),
        raw_pred_action=np.asarray(payload["raw_pred_action"], dtype=np.float32),
        target_token_ids=np.asarray(payload["target_token_ids"], dtype=np.int64),
        target_token_probs=np.asarray(payload["target_token_probs"], dtype=np.float32),
        target_token_bin_indices=None
        if payload.get("target_token_bin_indices") is None
        else np.asarray(payload["target_token_bin_indices"], dtype=np.int64),
        target_token_bin_centers=None
        if payload.get("target_token_bin_centers") is None
        else np.asarray(payload["target_token_bin_centers"], dtype=np.float32),
    )


def _step_to_dict(step: InterventionStepTrace) -> Dict[str, object]:
    return {
        "baseline_pred_action_xyzg": np.asarray(step.baseline_pred_action_xyzg, dtype=np.float32).tolist(),
        "baseline_raw_pred_action": np.asarray(step.baseline_raw_pred_action, dtype=np.float32).tolist(),
        "baseline_target_token_ids": np.asarray(step.baseline_target_token_ids, dtype=np.int64).tolist(),
        "baseline_target_token_probs": np.asarray(step.baseline_target_token_probs, dtype=np.float32).tolist(),
        "baseline_target_token_bin_indices": None
        if step.baseline_target_token_bin_indices is None
        else np.asarray(step.baseline_target_token_bin_indices, dtype=np.int64).tolist(),
        "baseline_target_token_bin_centers": None
        if step.baseline_target_token_bin_centers is None
        else np.asarray(step.baseline_target_token_bin_centers, dtype=np.float32).tolist(),
        "patch_occlusion": _scan_to_dict(step.patch_occlusion),
        "text_masking": _scan_to_dict(step.text_masking),
        "counterfactual_edits": [_counterfactual_to_dict(item) for item in step.counterfactual_edits],
        "counterfactual_success": bool(step.counterfactual_success),
        "prompt": step.prompt,
        "task_text": step.task_text,
        "text_token_ids": np.asarray(step.text_token_ids, dtype=np.int64).tolist(),
        "text_tokens": list(step.text_tokens),
    }


def _step_from_dict(payload: Dict[str, object]) -> InterventionStepTrace:
    return InterventionStepTrace(
        baseline_pred_action_xyzg=np.asarray(payload["baseline_pred_action_xyzg"], dtype=np.float32),
        baseline_raw_pred_action=np.asarray(payload["baseline_raw_pred_action"], dtype=np.float32),
        baseline_target_token_ids=np.asarray(payload["baseline_target_token_ids"], dtype=np.int64),
        baseline_target_token_probs=np.asarray(payload["baseline_target_token_probs"], dtype=np.float32),
        patch_occlusion=_scan_from_dict(payload.get("patch_occlusion")),
        text_masking=_scan_from_dict(payload.get("text_masking")),
        counterfactual_edits=[_counterfactual_from_dict(item) for item in payload.get("counterfactual_edits", [])],
        counterfactual_success=bool(payload.get("counterfactual_success", False)),
        prompt=str(payload.get("prompt", "")),
        task_text=str(payload.get("task_text", "")),
        text_token_ids=np.asarray(payload.get("text_token_ids", []), dtype=np.int64),
        text_tokens=[str(item) for item in payload.get("text_tokens", [])],
        baseline_target_token_bin_indices=None
        if payload.get("baseline_target_token_bin_indices") is None
        else np.asarray(payload["baseline_target_token_bin_indices"], dtype=np.int64),
        baseline_target_token_bin_centers=None
        if payload.get("baseline_target_token_bin_centers") is None
        else np.asarray(payload["baseline_target_token_bin_centers"], dtype=np.float32),
    )


def save_intervention_trace_file(
    trace_path: Path,
    dataset_path: Path,
    clips: Sequence[EpisodeClip],
    traces: Sequence[EpisodeInterventionTrace],
    checkpoint: Optional[str],
    prompt_style: Optional[str],
    dataset_adapter: Optional[str],
    model_adapter: Optional[str],
) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    clip_by_key = {clip.demo_key: clip for clip in clips}
    np.savez_compressed(
        trace_path,
        trace_format=INTERVENTION_TRACE_FORMAT,
        schema_version=np.int32(INTERVENTION_SCHEMA_VERSION),
        dataset_path=str(dataset_path),
        checkpoint="" if checkpoint is None else checkpoint,
        prompt_style="" if prompt_style is None else prompt_style,
        dataset_adapter="" if dataset_adapter is None else dataset_adapter,
        model_adapter="" if model_adapter is None else model_adapter,
        demo_keys=_to_object_array(trace.demo_key for trace in traces),
        frame_indices=_to_object_array(clip_by_key[trace.demo_key].frame_indices for trace in traces),
        pred_actions=_to_object_array(trace.pred_actions for trace in traces),
        step_errors=_to_object_array(trace.step_errors for trace in traces),
        mean_l1=np.asarray([trace.mean_l1 for trace in traces], dtype=np.float32),
        max_l1=np.asarray([trace.max_l1 for trace in traces], dtype=np.float32),
        score=np.asarray([trace.score for trace in traces], dtype=np.float32),
        steps=_to_object_array([_step_to_dict(step) for step in trace.steps] for trace in traces),
    )


def load_intervention_trace_file(trace_path: Path) -> InterventionTraceBundle:
    from .core import _optional_scalar

    with np.load(trace_path, allow_pickle=True) as data:
        demo_keys = [str(item) for item in data["demo_keys"].tolist()]
        pred_actions = data["pred_actions"].tolist()
        step_errors = data["step_errors"].tolist()
        mean_l1 = np.asarray(data["mean_l1"], dtype=np.float32)
        max_l1 = np.asarray(data["max_l1"], dtype=np.float32)
        score = np.asarray(data["score"], dtype=np.float32)
        frame_indices = data["frame_indices"].tolist()
        steps = data["steps"].tolist()

        traces: Dict[str, EpisodeInterventionTrace] = {}
        frame_indices_by_key: Dict[str, np.ndarray] = {}
        for idx, demo_key in enumerate(demo_keys):
            traces[demo_key] = EpisodeInterventionTrace(
                demo_key=demo_key,
                pred_actions=np.asarray(pred_actions[idx], dtype=np.float32),
                step_errors=np.asarray(step_errors[idx], dtype=np.float32),
                mean_l1=float(mean_l1[idx]),
                max_l1=float(max_l1[idx]),
                score=float(score[idx]),
                steps=[_step_from_dict(item) for item in steps[idx]],
            )
            frame_indices_by_key[demo_key] = np.asarray(frame_indices[idx], dtype=np.int32)

        trace_format = _optional_scalar(data, "trace_format") or INTERVENTION_TRACE_FORMAT
        schema_version = int(np.asarray(data["schema_version"]).item()) if "schema_version" in data else 1
        return InterventionTraceBundle(
            dataset_path=Path(str(data["dataset_path"].tolist())).expanduser(),
            checkpoint=_optional_scalar(data, "checkpoint"),
            prompt_style=_optional_scalar(data, "prompt_style"),
            dataset_adapter=_optional_scalar(data, "dataset_adapter"),
            model_adapter=_optional_scalar(data, "model_adapter"),
            trace_format=trace_format,
            schema_version=schema_version,
            traces=traces,
            frame_indices_by_key=frame_indices_by_key,
            demo_keys=demo_keys,
        )


def _best_intervention_match_index(step: InterventionStepTrace, corruption_type: str) -> int:
    if corruption_type == "patch-occlusion":
        scan = step.patch_occlusion
        label = "patch occlusion"
    elif corruption_type == "text-masking":
        scan = step.text_masking
        label = "text masking"
    else:
        raise ValueError(f"unsupported corruption_type: {corruption_type}")

    if scan is None:
        raise RuntimeError(f"intervention trace is missing {label} data for requested causal matching")

    if scan.top_candidates:
        return int(scan.top_candidates[0].index)

    effect_map = np.asarray(scan.effect_map, dtype=np.float32).reshape(-1)
    if effect_map.size == 0:
        raise RuntimeError(f"intervention trace has empty {label} scores for requested causal matching")
    return int(np.argmax(effect_map))


def match_intervention_trace_to_clips(
    clips: Sequence[EpisodeClip],
    trace_bundle: InterventionTraceBundle,
    corruption_type: str,
) -> Tuple[List[EpisodeClip], Dict[str, np.ndarray]]:
    clip_by_key = {clip.demo_key: clip for clip in clips}
    adjusted_clips: List[EpisodeClip] = []
    corruption_indices_by_key: Dict[str, np.ndarray] = {}

    missing_demo_keys = [demo_key for demo_key in trace_bundle.demo_keys if demo_key not in clip_by_key]
    if missing_demo_keys:
        raise ValueError(f"requested demos missing from dataset clips: {missing_demo_keys}")

    for demo_key in trace_bundle.demo_keys:
        clip = clip_by_key[demo_key]
        trace = trace_bundle.traces.get(demo_key)
        if trace is None:
            raise ValueError(f"demo missing from intervention trace bundle: {demo_key}")

        frame_indices = np.asarray(trace_bundle.frame_indices_by_key[demo_key], dtype=np.int32)
        if len(trace.steps) != int(frame_indices.shape[0]):
            raise RuntimeError(
                f"intervention trace length mismatch for {demo_key}: "
                f"{len(trace.steps)} steps vs {int(frame_indices.shape[0])} stored frame indices"
            )

        frame_index_to_pos = {
            int(frame_idx): pos
            for pos, frame_idx in enumerate(np.asarray(clip.frame_indices, dtype=np.int32).tolist())
        }
        try:
            selected_positions = np.asarray(
                [frame_index_to_pos[int(frame_idx)] for frame_idx in frame_indices],
                dtype=np.int32,
            )
        except KeyError as exc:
            missing_frame_idx = int(exc.args[0])
            raise RuntimeError(
                f"frame index {missing_frame_idx} from intervention trace for {demo_key} "
                "was not found in the dataset clip selection"
            ) from exc

        adjusted_clips.append(
            EpisodeClip(
                demo_key=clip.demo_key,
                instruction=clip.instruction,
                images=clip.images[selected_positions],
                gt_actions=clip.gt_actions[selected_positions],
                frame_indices=frame_indices,
            )
        )
        corruption_indices_by_key[demo_key] = np.asarray(
            [_best_intervention_match_index(step, corruption_type=corruption_type) for step in trace.steps],
            dtype=np.int32,
        )

    return adjusted_clips, corruption_indices_by_key


def intervention_trace_manifest(
    dataset_path: Path,
    checkpoint: Optional[str],
    prompt_style: Optional[str],
    dataset_adapter: Optional[str],
    model_adapter: Optional[str],
    clips: Sequence[EpisodeClip],
    traces: Sequence[EpisodeInterventionTrace],
) -> Dict[str, object]:
    clip_by_key = {clip.demo_key: clip for clip in clips}
    items = []
    for trace in traces:
        clip = clip_by_key[trace.demo_key]
        patch_grid_shape: List[int] = []
        text_token_count = 0
        if trace.steps:
            first_step = trace.steps[0]
            if first_step.patch_occlusion is not None:
                patch_grid_shape = list(np.asarray(first_step.patch_occlusion.effect_map).shape)
            text_token_count = int(len(first_step.text_token_ids))
        items.append(
            {
                "demo_key": trace.demo_key,
                "instruction": clip.instruction,
                "num_steps": int(len(clip.images)),
                "frame_indices": clip.frame_indices.astype(int).tolist(),
                "mean_l1": trace.mean_l1,
                "max_l1": trace.max_l1,
                "score": trace.score,
                "patch_occlusion_shape": patch_grid_shape,
                "text_token_count": text_token_count,
                "has_patch_occlusion": any(step.patch_occlusion is not None for step in trace.steps),
                "has_text_masking": any(step.text_masking is not None for step in trace.steps),
                "has_counterfactual": any(step.counterfactual_edits for step in trace.steps),
            }
        )

    return {
        "trace_format": INTERVENTION_TRACE_FORMAT,
        "schema_version": INTERVENTION_SCHEMA_VERSION,
        "dataset_path": str(dataset_path),
        "checkpoint": checkpoint,
        "prompt_style": prompt_style,
        "dataset_adapter": dataset_adapter,
        "model_adapter": model_adapter,
        "num_episodes": len(items),
        "episodes": items,
    }
