"""Collection and IO helpers for internal causal-localization runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from .core import EpisodeClip, LocalExplanationStep, _to_object_array, compute_episode_scores


CAUSAL_TRACE_FORMAT = "openvla_causal_localization_v1"
CAUSAL_SCHEMA_VERSION = 1


@dataclass
class CausalLocalizationStep:
    baseline_pred_action_xyzg: np.ndarray
    baseline_raw_pred_action: np.ndarray
    baseline_target_token_ids: np.ndarray
    baseline_target_token_probs: np.ndarray
    clean_sequence_logprob: float
    corruption_type: str
    corruption_index: int
    corruption_label: str
    corrupted_pred_action_xyzg: np.ndarray
    corrupted_raw_pred_action: np.ndarray
    corrupted_target_token_ids: np.ndarray
    corrupted_target_token_probs: np.ndarray
    corrupted_sequence_logprob: float
    layer_restoration_scores: np.ndarray
    head_restoration_scores: np.ndarray
    layer_labels: List[str]
    cross_attention_restoration_scores: Optional[np.ndarray] = None
    cross_attention_labels: Optional[List[str]] = None
    prompt: str = ""
    task_text: str = ""
    text_token_ids: Optional[np.ndarray] = None
    text_tokens: Optional[List[str]] = None


@dataclass
class EpisodeCausalTrace:
    demo_key: str
    pred_actions: np.ndarray
    step_errors: np.ndarray
    mean_l1: float
    max_l1: float
    score: float
    steps: List[CausalLocalizationStep]


@dataclass
class CausalTraceBundle:
    dataset_path: Path
    checkpoint: Optional[str]
    prompt_style: Optional[str]
    dataset_adapter: Optional[str]
    model_adapter: Optional[str]
    trace_format: str
    schema_version: int
    traces: Dict[str, EpisodeCausalTrace]
    frame_indices_by_key: Dict[str, np.ndarray]
    demo_keys: List[str]


class CausalLocalizationModelAdapter(Protocol):
    """Model-side interface for causal localization."""

    def explain_step(self, image: np.ndarray, instruction: str) -> LocalExplanationStep:
        """Run the clean baseline step."""

    def causal_localization(
        self,
        image: np.ndarray,
        instruction: str,
        baseline_step: LocalExplanationStep,
        corruption_type: str,
        corruption_index: Optional[int],
        per_cross_attention: bool,
    ) -> CausalLocalizationStep:
        """Run activation patching for one frame."""


def collect_episode_causal_traces(
    clips: Sequence[EpisodeClip],
    model_adapter: CausalLocalizationModelAdapter,
    corruption_type: str = "patch-occlusion",
    corruption_index: Optional[int] = None,
    per_cross_attention: bool = False,
) -> List[EpisodeCausalTrace]:
    traces: List[EpisodeCausalTrace] = []
    for clip in clips:
        pred_actions: List[np.ndarray] = []
        step_traces: List[CausalLocalizationStep] = []
        for frame in clip.images:
            baseline_step = model_adapter.explain_step(frame, clip.instruction)
            pred_actions.append(np.asarray(baseline_step.pred_action_xyzg, dtype=np.float32))
            step_traces.append(
                model_adapter.causal_localization(
                    image=frame,
                    instruction=clip.instruction,
                    baseline_step=baseline_step,
                    corruption_type=corruption_type,
                    corruption_index=corruption_index,
                    per_cross_attention=per_cross_attention,
                )
            )

        pred_np = np.stack(pred_actions, axis=0)
        step_errors, mean_l1, max_l1, score = compute_episode_scores(clip.gt_actions, pred_np)
        traces.append(
            EpisodeCausalTrace(
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


def _step_to_dict(step: CausalLocalizationStep) -> Dict[str, object]:
    return {
        "baseline_pred_action_xyzg": np.asarray(step.baseline_pred_action_xyzg, dtype=np.float32).tolist(),
        "baseline_raw_pred_action": np.asarray(step.baseline_raw_pred_action, dtype=np.float32).tolist(),
        "baseline_target_token_ids": np.asarray(step.baseline_target_token_ids, dtype=np.int64).tolist(),
        "baseline_target_token_probs": np.asarray(step.baseline_target_token_probs, dtype=np.float32).tolist(),
        "clean_sequence_logprob": float(step.clean_sequence_logprob),
        "corruption_type": step.corruption_type,
        "corruption_index": int(step.corruption_index),
        "corruption_label": step.corruption_label,
        "corrupted_pred_action_xyzg": np.asarray(step.corrupted_pred_action_xyzg, dtype=np.float32).tolist(),
        "corrupted_raw_pred_action": np.asarray(step.corrupted_raw_pred_action, dtype=np.float32).tolist(),
        "corrupted_target_token_ids": np.asarray(step.corrupted_target_token_ids, dtype=np.int64).tolist(),
        "corrupted_target_token_probs": np.asarray(step.corrupted_target_token_probs, dtype=np.float32).tolist(),
        "corrupted_sequence_logprob": float(step.corrupted_sequence_logprob),
        "layer_restoration_scores": np.asarray(step.layer_restoration_scores, dtype=np.float32).tolist(),
        "head_restoration_scores": np.asarray(step.head_restoration_scores, dtype=np.float32).tolist(),
        "layer_labels": list(step.layer_labels),
        "cross_attention_restoration_scores": None
        if step.cross_attention_restoration_scores is None
        else np.asarray(step.cross_attention_restoration_scores, dtype=np.float32).tolist(),
        "cross_attention_labels": None if step.cross_attention_labels is None else list(step.cross_attention_labels),
        "prompt": step.prompt,
        "task_text": step.task_text,
        "text_token_ids": None if step.text_token_ids is None else np.asarray(step.text_token_ids, dtype=np.int64).tolist(),
        "text_tokens": None if step.text_tokens is None else list(step.text_tokens),
    }


def _step_from_dict(payload: Dict[str, object]) -> CausalLocalizationStep:
    return CausalLocalizationStep(
        baseline_pred_action_xyzg=np.asarray(payload["baseline_pred_action_xyzg"], dtype=np.float32),
        baseline_raw_pred_action=np.asarray(payload["baseline_raw_pred_action"], dtype=np.float32),
        baseline_target_token_ids=np.asarray(payload["baseline_target_token_ids"], dtype=np.int64),
        baseline_target_token_probs=np.asarray(payload["baseline_target_token_probs"], dtype=np.float32),
        clean_sequence_logprob=float(payload["clean_sequence_logprob"]),
        corruption_type=str(payload["corruption_type"]),
        corruption_index=int(payload["corruption_index"]),
        corruption_label=str(payload["corruption_label"]),
        corrupted_pred_action_xyzg=np.asarray(payload["corrupted_pred_action_xyzg"], dtype=np.float32),
        corrupted_raw_pred_action=np.asarray(payload["corrupted_raw_pred_action"], dtype=np.float32),
        corrupted_target_token_ids=np.asarray(payload["corrupted_target_token_ids"], dtype=np.int64),
        corrupted_target_token_probs=np.asarray(payload["corrupted_target_token_probs"], dtype=np.float32),
        corrupted_sequence_logprob=float(payload["corrupted_sequence_logprob"]),
        layer_restoration_scores=np.asarray(payload["layer_restoration_scores"], dtype=np.float32),
        head_restoration_scores=np.asarray(payload["head_restoration_scores"], dtype=np.float32),
        layer_labels=[str(item) for item in payload.get("layer_labels", [])],
        cross_attention_restoration_scores=None
        if payload.get("cross_attention_restoration_scores") is None
        else np.asarray(payload["cross_attention_restoration_scores"], dtype=np.float32),
        cross_attention_labels=None
        if payload.get("cross_attention_labels") is None
        else [str(item) for item in payload.get("cross_attention_labels", [])],
        prompt=str(payload.get("prompt", "")),
        task_text=str(payload.get("task_text", "")),
        text_token_ids=None if payload.get("text_token_ids") is None else np.asarray(payload["text_token_ids"], dtype=np.int64),
        text_tokens=None if payload.get("text_tokens") is None else [str(item) for item in payload.get("text_tokens", [])],
    )


def save_causal_trace_file(
    trace_path: Path,
    dataset_path: Path,
    clips: Sequence[EpisodeClip],
    traces: Sequence[EpisodeCausalTrace],
    checkpoint: Optional[str],
    prompt_style: Optional[str],
    dataset_adapter: Optional[str],
    model_adapter: Optional[str],
) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    clip_by_key = {clip.demo_key: clip for clip in clips}
    np.savez_compressed(
        trace_path,
        trace_format=CAUSAL_TRACE_FORMAT,
        schema_version=np.int32(CAUSAL_SCHEMA_VERSION),
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


def load_causal_trace_file(trace_path: Path) -> CausalTraceBundle:
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

        traces: Dict[str, EpisodeCausalTrace] = {}
        frame_indices_by_key: Dict[str, np.ndarray] = {}
        for idx, demo_key in enumerate(demo_keys):
            traces[demo_key] = EpisodeCausalTrace(
                demo_key=demo_key,
                pred_actions=np.asarray(pred_actions[idx], dtype=np.float32),
                step_errors=np.asarray(step_errors[idx], dtype=np.float32),
                mean_l1=float(mean_l1[idx]),
                max_l1=float(max_l1[idx]),
                score=float(score[idx]),
                steps=[_step_from_dict(item) for item in steps[idx]],
            )
            frame_indices_by_key[demo_key] = np.asarray(frame_indices[idx], dtype=np.int32)

        trace_format = _optional_scalar(data, "trace_format") or CAUSAL_TRACE_FORMAT
        schema_version = int(np.asarray(data["schema_version"]).item()) if "schema_version" in data else 1
        return CausalTraceBundle(
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


def causal_trace_manifest(
    dataset_path: Path,
    checkpoint: Optional[str],
    prompt_style: Optional[str],
    dataset_adapter: Optional[str],
    model_adapter: Optional[str],
    clips: Sequence[EpisodeClip],
    traces: Sequence[EpisodeCausalTrace],
) -> Dict[str, object]:
    clip_by_key = {clip.demo_key: clip for clip in clips}
    items = []
    for trace in traces:
        clip = clip_by_key[trace.demo_key]
        layer_count = 0
        head_shape: List[int] = []
        cross_attention_count = 0
        if trace.steps:
            first_step = trace.steps[0]
            layer_count = int(np.asarray(first_step.layer_restoration_scores).shape[0])
            head_shape = list(np.asarray(first_step.head_restoration_scores).shape)
            cross_attention_count = 0 if first_step.cross_attention_labels is None else len(first_step.cross_attention_labels)
        items.append(
            {
                "demo_key": trace.demo_key,
                "instruction": clip.instruction,
                "num_steps": int(len(clip.images)),
                "frame_indices": clip.frame_indices.astype(int).tolist(),
                "mean_l1": trace.mean_l1,
                "max_l1": trace.max_l1,
                "score": trace.score,
                "layer_count": layer_count,
                "head_score_shape": head_shape,
                "cross_attention_block_count": cross_attention_count,
            }
        )

    return {
        "trace_format": CAUSAL_TRACE_FORMAT,
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "dataset_path": str(dataset_path),
        "checkpoint": checkpoint,
        "prompt_style": prompt_style,
        "dataset_adapter": dataset_adapter,
        "model_adapter": model_adapter,
        "num_episodes": len(items),
        "episodes": items,
    }

