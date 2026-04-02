"""Core data structures and IO for explainability traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, TypeVar

import numpy as np

from ..openvla_action_utils import canonicalize_action_to_benchmark


TRACE_FORMAT = "openvla_local_explanations_v1"
TRACE_SCHEMA_VERSION = 2


@dataclass
class EpisodeClip:
    demo_key: str
    instruction: str
    images: np.ndarray
    gt_actions: np.ndarray
    frame_indices: np.ndarray


@dataclass
class LocalExplanationStep:
    pred_action_xyzg: np.ndarray
    raw_pred_action: np.ndarray
    target_token_ids: np.ndarray
    target_token_probs: np.ndarray
    image_patch_attributions: np.ndarray
    text_token_attributions: np.ndarray
    mean_image_patch_attribution: np.ndarray
    mean_text_token_attribution: np.ndarray
    text_token_ids: np.ndarray
    text_tokens: List[str]
    prompt: str
    task_text: str
    target_token_bin_indices: Optional[np.ndarray] = None
    target_token_bin_centers: Optional[np.ndarray] = None


@dataclass
class EpisodeTrace:
    demo_key: str
    pred_actions: np.ndarray
    attention_grids: np.ndarray
    step_errors: np.ndarray
    mean_l1: float
    max_l1: float
    score: float
    raw_pred_actions: Optional[np.ndarray] = None
    target_token_ids: Optional[np.ndarray] = None
    target_token_probs: Optional[np.ndarray] = None
    image_patch_attributions: Optional[np.ndarray] = None
    text_token_attributions: Optional[np.ndarray] = None
    text_token_ids: Optional[np.ndarray] = None
    text_tokens: Optional[List[str]] = None
    mean_text_token_attributions: Optional[np.ndarray] = None
    prompt: Optional[str] = None
    task_text: Optional[str] = None


@dataclass
class TraceBundle:
    dataset_path: Path
    checkpoint: Optional[str]
    prompt_style: Optional[str]
    dataset_adapter: Optional[str]
    model_adapter: Optional[str]
    trace_format: str
    schema_version: int
    traces: Dict[str, EpisodeTrace]
    frame_indices_by_key: Dict[str, np.ndarray]
    demo_keys: List[str]


class LocalExplanationModelAdapter(Protocol):
    """Model-side interface for explainability collectors."""

    def explain_step(self, image: np.ndarray, instruction: str) -> LocalExplanationStep:
        """Run one local explanation step."""


class ScoredTrace(Protocol):
    demo_key: str
    score: float


ScoredTraceT = TypeVar("ScoredTraceT", bound=ScoredTrace)


def _dedupe_paths(paths: Sequence[Path]) -> List[Path]:
    deduped: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def resolve_dataset_path(dataset: str, repo_root: Path) -> Path:
    requested = Path(dataset).expanduser()
    if requested.is_absolute():
        if requested.is_file():
            return requested.resolve()
        raise FileNotFoundError(f"dataset not found at absolute path: {requested}")

    direct_candidates = _dedupe_paths([Path.cwd() / requested, repo_root / requested])
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    search_roots = [
        repo_root / "benchmarks" / "go_vla_benchmark" / "data",
        repo_root / "benchmarks",
    ]
    discovered: List[Path] = []
    for root in search_roots:
        if root.exists():
            discovered.extend(root.rglob(requested.name))
    discovered = _dedupe_paths([path for path in discovered if path.is_file()])

    if len(discovered) == 1:
        return discovered[0]

    tried_lines = "\n".join(f"  - {path}" for path in direct_candidates)
    if discovered:
        found_lines = "\n".join(f"  - {path}" for path in discovered)
        raise FileNotFoundError(
            f"dataset '{dataset}' was not found at expected locations.\n"
            f"Tried:\n{tried_lines}\n"
            f"Found multiple files with the same name:\n{found_lines}\n"
            "Pass an absolute --dataset path to disambiguate."
        )

    raise FileNotFoundError(
        f"dataset '{dataset}' was not found.\n"
        f"Tried:\n{tried_lines}\n"
        "Tip: pass an absolute --dataset path from the collector output."
    )


def resolve_repo_relative_path(path: str, repo_root: Path) -> Path:
    requested = Path(path).expanduser()
    if requested.is_absolute():
        return requested.resolve()
    return (repo_root / requested).resolve()


def resolve_optional_path(path: Optional[str], repo_root: Path) -> Optional[Path]:
    if path is None:
        return None
    return resolve_repo_relative_path(path, repo_root=repo_root)


def extract_xyzg(actions: np.ndarray) -> np.ndarray:
    """Extract `[dx, dy, dz, gripper]` and canonicalize to benchmark semantics."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.shape[1] >= 7:
        out = np.zeros((actions.shape[0], 4), dtype=np.float32)
        out[:, :3] = actions[:, :3]
        out[:, 3] = actions[:, 6]
    elif actions.shape[1] >= 4:
        out = actions[:, :4].astype(np.float32)
    else:
        raise ValueError(f"expected action dimension >= 4, got {actions.shape}")

    return canonicalize_action_to_benchmark(out, binarize=False)


def compute_episode_scores(gt_actions: np.ndarray, pred_actions: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    if gt_actions.shape != pred_actions.shape:
        raise ValueError(f"shape mismatch for GT vs pred actions: {gt_actions.shape} vs {pred_actions.shape}")

    step_errors = np.mean(np.abs(gt_actions - pred_actions), axis=1).astype(np.float32)
    mean_l1 = float(step_errors.mean()) if step_errors.size > 0 else 0.0
    max_l1 = float(step_errors.max()) if step_errors.size > 0 else 0.0
    score = (0.65 * mean_l1) + (0.35 * max_l1)
    return step_errors, mean_l1, max_l1, float(score)


def collect_episode_traces(
    clips: Sequence[EpisodeClip],
    model_adapter: LocalExplanationModelAdapter,
) -> List[EpisodeTrace]:
    traces: List[EpisodeTrace] = []
    for clip in clips:
        pred_actions: List[np.ndarray] = []
        raw_pred_actions: List[np.ndarray] = []
        attention_grids: List[np.ndarray] = []
        target_token_ids: List[np.ndarray] = []
        target_token_probs: List[np.ndarray] = []
        image_patch_attributions: List[np.ndarray] = []
        text_token_attributions: List[np.ndarray] = []
        mean_text_token_attributions: List[np.ndarray] = []
        text_token_ids: Optional[np.ndarray] = None
        text_tokens: Optional[List[str]] = None
        prompt: Optional[str] = None
        task_text: Optional[str] = None

        for frame in clip.images:
            explanation = model_adapter.explain_step(frame, clip.instruction)
            pred_actions.append(np.asarray(explanation.pred_action_xyzg, dtype=np.float32))
            raw_pred_actions.append(np.asarray(explanation.raw_pred_action, dtype=np.float32))
            attention_grids.append(np.asarray(explanation.mean_image_patch_attribution, dtype=np.float32))
            target_token_ids.append(np.asarray(explanation.target_token_ids, dtype=np.int64))
            target_token_probs.append(np.asarray(explanation.target_token_probs, dtype=np.float32))
            image_patch_attributions.append(np.asarray(explanation.image_patch_attributions, dtype=np.float32))
            text_token_attributions.append(np.asarray(explanation.text_token_attributions, dtype=np.float32))
            mean_text_token_attributions.append(np.asarray(explanation.mean_text_token_attribution, dtype=np.float32))

            if text_token_ids is None:
                text_token_ids = np.asarray(explanation.text_token_ids, dtype=np.int64)
                text_tokens = list(explanation.text_tokens)
                prompt = explanation.prompt
                task_text = explanation.task_text

        pred_np = np.stack(pred_actions, axis=0)
        step_errors, mean_l1, max_l1, score = compute_episode_scores(clip.gt_actions, pred_np)
        traces.append(
            EpisodeTrace(
                demo_key=clip.demo_key,
                pred_actions=pred_np,
                attention_grids=np.stack(attention_grids, axis=0),
                step_errors=step_errors,
                mean_l1=mean_l1,
                max_l1=max_l1,
                score=score,
                raw_pred_actions=np.stack(raw_pred_actions, axis=0),
                target_token_ids=np.stack(target_token_ids, axis=0),
                target_token_probs=np.stack(target_token_probs, axis=0),
                image_patch_attributions=np.stack(image_patch_attributions, axis=0),
                text_token_attributions=np.stack(text_token_attributions, axis=0),
                text_token_ids=text_token_ids if text_token_ids is not None else np.zeros((0,), dtype=np.int64),
                text_tokens=text_tokens if text_tokens is not None else [],
                mean_text_token_attributions=np.stack(mean_text_token_attributions, axis=0),
                prompt=prompt,
                task_text=task_text,
            )
        )

    return traces


def _to_object_array(values: Sequence[object]) -> np.ndarray:
    return np.asarray(list(values), dtype=object)


def serialize_embedded_episode_clips(clips: Sequence[EpisodeClip]) -> Dict[str, np.ndarray]:
    return {
        "embedded_clip_demo_keys": _to_object_array(clip.demo_key for clip in clips),
        "embedded_clip_instructions": _to_object_array(clip.instruction for clip in clips),
        "embedded_clip_images": _to_object_array(np.asarray(clip.images, dtype=np.uint8) for clip in clips),
        "embedded_clip_gt_actions": _to_object_array(np.asarray(clip.gt_actions, dtype=np.float32) for clip in clips),
        "embedded_clip_frame_indices": _to_object_array(
            np.asarray(clip.frame_indices, dtype=np.int32) for clip in clips
        ),
    }


def load_embedded_episode_clips(data: np.lib.npyio.NpzFile) -> Optional[Dict[str, EpisodeClip]]:
    if "embedded_clip_demo_keys" not in data:
        return None

    demo_keys = [str(item) for item in data["embedded_clip_demo_keys"].tolist()]
    instructions = data["embedded_clip_instructions"].tolist()
    images = data["embedded_clip_images"].tolist()
    gt_actions = data["embedded_clip_gt_actions"].tolist()
    frame_indices = data["embedded_clip_frame_indices"].tolist()

    clips: Dict[str, EpisodeClip] = {}
    for idx, demo_key in enumerate(demo_keys):
        clips[demo_key] = EpisodeClip(
            demo_key=demo_key,
            instruction=str(instructions[idx]),
            images=np.asarray(images[idx], dtype=np.uint8),
            gt_actions=np.asarray(gt_actions[idx], dtype=np.float32),
            frame_indices=np.asarray(frame_indices[idx], dtype=np.int32),
        )
    return clips


def save_trace_file(
    trace_path: Path,
    dataset_path: Path,
    clips: Sequence[EpisodeClip],
    traces: Sequence[EpisodeTrace],
    checkpoint: Optional[str],
    prompt_style: Optional[str],
    dataset_adapter: Optional[str],
    model_adapter: Optional[str],
) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    clip_by_key = {clip.demo_key: clip for clip in clips}

    np.savez_compressed(
        trace_path,
        trace_format=TRACE_FORMAT,
        schema_version=np.int32(TRACE_SCHEMA_VERSION),
        dataset_path=str(dataset_path),
        checkpoint="" if checkpoint is None else checkpoint,
        prompt_style="" if prompt_style is None else prompt_style,
        dataset_adapter="" if dataset_adapter is None else dataset_adapter,
        model_adapter="" if model_adapter is None else model_adapter,
        demo_keys=_to_object_array(trace.demo_key for trace in traces),
        frame_indices=_to_object_array(clip_by_key[trace.demo_key].frame_indices for trace in traces),
        pred_actions=_to_object_array(trace.pred_actions for trace in traces),
        attention_grids=_to_object_array(trace.attention_grids for trace in traces),
        step_errors=_to_object_array(trace.step_errors for trace in traces),
        mean_l1=np.asarray([trace.mean_l1 for trace in traces], dtype=np.float32),
        max_l1=np.asarray([trace.max_l1 for trace in traces], dtype=np.float32),
        score=np.asarray([trace.score for trace in traces], dtype=np.float32),
        raw_pred_actions=_to_object_array(trace.raw_pred_actions for trace in traces),
        target_token_ids=_to_object_array(trace.target_token_ids for trace in traces),
        target_token_probs=_to_object_array(trace.target_token_probs for trace in traces),
        image_patch_attributions=_to_object_array(trace.image_patch_attributions for trace in traces),
        text_token_attributions=_to_object_array(trace.text_token_attributions for trace in traces),
        text_token_ids=_to_object_array(trace.text_token_ids for trace in traces),
        text_tokens=_to_object_array(trace.text_tokens for trace in traces),
        mean_text_token_attributions=_to_object_array(trace.mean_text_token_attributions for trace in traces),
        prompt=_to_object_array(trace.prompt for trace in traces),
        task_text=_to_object_array(trace.task_text for trace in traces),
    )


def _optional_scalar(data: np.lib.npyio.NpzFile, key: str) -> Optional[str]:
    if key not in data:
        return None
    value = str(data[key].tolist())
    return value or None


def _optional_object_list(data: np.lib.npyio.NpzFile, key: str, count: int) -> Optional[list[object]]:
    if key not in data:
        return None
    values = data[key].tolist()
    if isinstance(values, list):
        return values
    return [values] * count


def _optional_object_array(data: np.lib.npyio.NpzFile, key: str, idx: int) -> Optional[np.ndarray]:
    values = _optional_object_list(data, key, count=idx + 1)
    if values is None:
        return None
    value = values[idx]
    if value is None:
        return None
    return np.asarray(value)


def _optional_str_list(data: np.lib.npyio.NpzFile, key: str, idx: int) -> Optional[List[str]]:
    values = _optional_object_list(data, key, count=idx + 1)
    if values is None:
        return None
    value = values[idx]
    if value is None:
        return None
    return [str(item) for item in list(value)]


def load_trace_file(trace_path: Path) -> TraceBundle:
    with np.load(trace_path, allow_pickle=True) as data:
        demo_keys = [str(item) for item in data["demo_keys"].tolist()]
        traces: Dict[str, EpisodeTrace] = {}
        frame_indices_by_key: Dict[str, np.ndarray] = {}

        pred_actions = data["pred_actions"].tolist()
        attention_grids = data["attention_grids"].tolist()
        step_errors = data["step_errors"].tolist()
        mean_l1 = np.asarray(data["mean_l1"], dtype=np.float32)
        max_l1 = np.asarray(data["max_l1"], dtype=np.float32)
        score = np.asarray(data["score"], dtype=np.float32)
        frame_indices = data["frame_indices"].tolist()

        for idx, demo_key in enumerate(demo_keys):
            traces[demo_key] = EpisodeTrace(
                demo_key=demo_key,
                pred_actions=np.asarray(pred_actions[idx], dtype=np.float32),
                attention_grids=np.asarray(attention_grids[idx], dtype=np.float32),
                step_errors=np.asarray(step_errors[idx], dtype=np.float32),
                mean_l1=float(mean_l1[idx]),
                max_l1=float(max_l1[idx]),
                score=float(score[idx]),
                raw_pred_actions=_optional_object_array(data, "raw_pred_actions", idx),
                target_token_ids=_optional_object_array(data, "target_token_ids", idx),
                target_token_probs=_optional_object_array(data, "target_token_probs", idx),
                image_patch_attributions=_optional_object_array(data, "image_patch_attributions", idx),
                text_token_attributions=_optional_object_array(data, "text_token_attributions", idx),
                text_token_ids=_optional_object_array(data, "text_token_ids", idx),
                text_tokens=_optional_str_list(data, "text_tokens", idx),
                mean_text_token_attributions=_optional_object_array(data, "mean_text_token_attributions", idx),
                prompt=_optional_scalar_from_list(data, "prompt", idx),
                task_text=_optional_scalar_from_list(data, "task_text", idx),
            )
            frame_indices_by_key[demo_key] = np.asarray(frame_indices[idx], dtype=np.int32)

        trace_format = _optional_scalar(data, "trace_format") or "attention_video_trace_v1"
        schema_version = int(np.asarray(data["schema_version"]).item()) if "schema_version" in data else 1

        return TraceBundle(
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


def _optional_scalar_from_list(data: np.lib.npyio.NpzFile, key: str, idx: int) -> Optional[str]:
    values = _optional_object_list(data, key, count=idx + 1)
    if values is None:
        return None
    value = values[idx]
    if value in (None, ""):
        return None
    return str(value)


def select_ranked_traces(
    traces: Sequence[EpisodeTrace],
    demos: Optional[str],
    top_k: int,
) -> List[EpisodeTrace]:
    if demos:
        requested = {item.strip() for item in demos.split(",") if item.strip()}
        ordered = [trace for trace in traces if trace.demo_key in requested]
        if len(ordered) != len(requested):
            present = {trace.demo_key for trace in traces}
            missing = sorted(requested - present)
            raise ValueError(f"requested demos missing from trace data: {missing}")
        return ordered

    ranked = sorted(traces, key=lambda item: item.score, reverse=True)
    if top_k <= 0:
        return ranked
    return ranked[:top_k]


def select_top_k_traces_preserving_order(
    traces: Sequence[ScoredTraceT],
    top_k: int,
) -> List[ScoredTraceT]:
    if top_k <= 0 or top_k >= len(traces):
        return list(traces)

    ranked_indices = sorted(range(len(traces)), key=lambda idx: traces[idx].score, reverse=True)[: int(top_k)]
    selected_indices = set(ranked_indices)
    return [trace for idx, trace in enumerate(traces) if idx in selected_indices]


def trace_manifest(
    dataset_path: Path,
    checkpoint: Optional[str],
    prompt_style: Optional[str],
    dataset_adapter: Optional[str],
    model_adapter: Optional[str],
    clips: Sequence[EpisodeClip],
    traces: Sequence[EpisodeTrace],
) -> Dict[str, object]:
    clip_by_key = {clip.demo_key: clip for clip in clips}
    items = []
    for trace in traces:
        clip = clip_by_key[trace.demo_key]
        patch_shape = list(trace.attention_grids.shape[1:]) if trace.attention_grids.ndim >= 3 else []
        items.append(
            {
                "demo_key": trace.demo_key,
                "instruction": clip.instruction,
                "num_steps": int(len(clip.images)),
                "frame_indices": clip.frame_indices.astype(int).tolist(),
                "mean_l1": trace.mean_l1,
                "max_l1": trace.max_l1,
                "score": trace.score,
                "patch_grid_shape": patch_shape,
                "text_token_count": int(0 if trace.text_token_ids is None else len(trace.text_token_ids)),
                "text_tokens": [] if trace.text_tokens is None else list(trace.text_tokens),
            }
        )

    return {
        "trace_format": TRACE_FORMAT,
        "schema_version": TRACE_SCHEMA_VERSION,
        "dataset_path": str(dataset_path),
        "checkpoint": checkpoint,
        "prompt_style": prompt_style,
        "dataset_adapter": dataset_adapter,
        "model_adapter": model_adapter,
        "num_episodes": len(items),
        "episodes": items,
    }
