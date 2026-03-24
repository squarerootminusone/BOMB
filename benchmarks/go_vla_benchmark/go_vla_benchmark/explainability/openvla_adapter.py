"""OpenVLA-backed explainability adapters for local, intervention, and causal stages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .causal_localization import CausalLocalizationStep
from .core import LocalExplanationStep, extract_xyzg
from .interventions import CounterfactualEdit, InterventionCandidateEffect, InterventionScan


BLANK_ACTION_TOKEN_ID = 29871
_WORD_SPAN_PATTERN = re.compile(r"\b[a-z0-9]+\b", flags=re.IGNORECASE)
_TEXT_MASK_PATTERNS = (
    re.compile(r"position\s*\(\s*\d+\s*,\s*\d+\s*\)", flags=re.IGNORECASE),
    re.compile(r"\(\s*\d+\s*,\s*\d+\s*\)", flags=re.IGNORECASE),
    re.compile(r"row\s+\d+", flags=re.IGNORECASE),
    re.compile(r"column\s+\d+", flags=re.IGNORECASE),
    re.compile(r"go\s+board", flags=re.IGNORECASE),
)


@dataclass(frozen=True)
class _PromptContext:
    prompt: str
    task_text: str
    instruction_token_positions: np.ndarray
    instruction_token_ids: np.ndarray
    instruction_tokens: List[str]
    text_mask_candidates: List["_TextMaskCandidate"]


@dataclass(frozen=True)
class _TextMaskCandidate:
    label: str
    prompt_token_positions: np.ndarray
    task_char_start: int
    task_char_end: int


@dataclass
class _DecodedSequence:
    pred_action_xyzg: np.ndarray
    raw_pred_action: np.ndarray
    target_token_ids: np.ndarray
    target_token_probs: np.ndarray
    prompt_context: _PromptContext
    num_patches: int
    patch_token_attributions: Optional[np.ndarray] = None
    text_token_attributions: Optional[np.ndarray] = None


@dataclass
class _TargetSequenceScore:
    target_token_probs: np.ndarray
    sequence_logprob: float
    num_patches: int


def _safe_register(auto_class, key, value) -> None:
    try:
        auto_class.register(key, value)
    except ValueError:
        pass


def _ensure_openvla_imports():
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    _safe_register(AutoConfig, "openvla", OpenVLAConfig)
    _safe_register(AutoImageProcessor, OpenVLAConfig, PrismaticImageProcessor)
    _safe_register(AutoProcessor, OpenVLAConfig, PrismaticProcessor)
    _safe_register(AutoModelForVision2Seq, OpenVLAConfig, OpenVLAForActionPrediction)

    return AutoModelForVision2Seq, AutoProcessor


def _build_prompt_parts(instruction: str, prompt_style: str) -> Tuple[str, str, str, str]:
    task_text = instruction.strip().rstrip(".").lower()
    if prompt_style == "openvla-v01":
        system_prompt = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        prefix = f"{system_prompt} USER: What action should the robot take to "
        suffix = "? ASSISTANT:"
        return prefix + task_text + suffix, prefix, task_text, suffix
    prefix = "In: What action should the robot take to "
    suffix = "?\nOut:"
    return prefix + task_text + suffix, prefix, task_text, suffix


def _infer_prompt_style(explicit: Optional[str], checkpoint: str) -> str:
    if explicit:
        return explicit
    return "openvla-v01" if "openvla-v01" in checkpoint.lower() else "openvla"


def _decode_action_tokens(model, token_ids: np.ndarray, action_dim: int, unnorm_key: Optional[str]) -> np.ndarray:
    predicted_action_token_ids = np.asarray(token_ids[-action_dim:], dtype=np.int64)
    discretized_actions = model.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    normalized_actions = model.bin_centers[discretized_actions]

    try:
        action_norm_stats = model.get_action_stats(unnorm_key)
    except Exception:
        return normalized_actions.astype(np.float32)

    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high = np.asarray(action_norm_stats["q99"], dtype=np.float32)
    action_low = np.asarray(action_norm_stats["q01"], dtype=np.float32)
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1.0) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return np.asarray(actions, dtype=np.float32)


def _fit_attention_grid(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    side = int(round(np.sqrt(vector.size)))
    if side * side != vector.size:
        raise ValueError(f"expected square number of image patches, got {vector.size}")
    return vector.reshape(side, side)


def _normalize_vector(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= eps:
        return np.zeros_like(values)
    return values / total


def _find_instruction_tokens(
    tokenizer,
    prompt: str,
    prefix: str,
    task_text: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    prompt_inputs = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = np.asarray(prompt_inputs["input_ids"], dtype=np.int64)
    offset_mapping = prompt_inputs.get("offset_mapping")

    if offset_mapping is not None:
        span_start = len(prefix)
        span_end = span_start + len(task_text)
        positions = [
            idx
            for idx, (token_start, token_end) in enumerate(offset_mapping)
            if int(token_end) > span_start and int(token_start) < span_end
        ]
        if positions:
            instruction_token_positions = np.asarray(positions, dtype=np.int64)
            instruction_token_ids = prompt_ids[instruction_token_positions]
            instruction_offsets = np.asarray([offset_mapping[idx] for idx in instruction_token_positions.tolist()], dtype=np.int64)
            if hasattr(tokenizer, "convert_ids_to_tokens"):
                instruction_tokens = [str(token) for token in tokenizer.convert_ids_to_tokens(instruction_token_ids.tolist())]
            else:
                instruction_tokens = [tokenizer.decode([int(token_id)]) for token_id in instruction_token_ids]
            return instruction_token_positions, instruction_token_ids, instruction_tokens, instruction_offsets

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    prefix_task_ids = tokenizer(prefix + task_text, add_special_tokens=False)["input_ids"]
    start = len(prefix_ids)
    end = len(prefix_task_ids)
    instruction_token_positions = np.arange(start, end, dtype=np.int64)
    instruction_token_ids = prompt_ids[start:end]
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        instruction_tokens = [str(token) for token in tokenizer.convert_ids_to_tokens(instruction_token_ids.tolist())]
    else:
        instruction_tokens = [tokenizer.decode([int(token_id)]) for token_id in instruction_token_ids]
    return instruction_token_positions, instruction_token_ids, instruction_tokens, np.zeros((instruction_token_positions.shape[0], 2), dtype=np.int64)


def _collect_text_mask_spans(task_text: str) -> List[Tuple[str, int, int]]:
    spans: List[Tuple[str, int, int]] = []
    seen: set[Tuple[int, int]] = set()

    def add_span(start: int, end: int) -> None:
        start = int(start)
        end = int(end)
        if start < 0 or end <= start:
            return
        key = (start, end)
        if key in seen:
            return
        label = task_text[start:end].strip()
        if not label:
            return
        seen.add(key)
        spans.append((label, start, end))

    for pattern in _TEXT_MASK_PATTERNS:
        for match in pattern.finditer(task_text):
            add_span(match.start(), match.end())
    for match in _WORD_SPAN_PATTERN.finditer(task_text):
        add_span(match.start(), match.end())

    spans.sort(key=lambda item: (item[1], -(item[2] - item[1]), item[0]))
    return spans


def _build_text_mask_candidates(
    task_text: str,
    prefix_len: int,
    instruction_token_positions: np.ndarray,
    instruction_token_offsets: np.ndarray,
    instruction_tokens: Sequence[str],
) -> List[_TextMaskCandidate]:
    valid_offsets = (
        instruction_token_offsets.ndim == 2
        and instruction_token_offsets.shape[0] == instruction_token_positions.shape[0]
        and instruction_token_offsets.shape[1] == 2
        and bool(np.any(instruction_token_offsets[:, 1] > instruction_token_offsets[:, 0]))
    )

    candidates: List[_TextMaskCandidate] = []
    if valid_offsets:
        for label, start, end in _collect_text_mask_spans(task_text):
            prompt_start = prefix_len + int(start)
            prompt_end = prefix_len + int(end)
            overlap = (instruction_token_offsets[:, 1] > prompt_start) & (instruction_token_offsets[:, 0] < prompt_end)
            prompt_positions = instruction_token_positions[overlap]
            if prompt_positions.size == 0:
                continue
            candidates.append(
                _TextMaskCandidate(
                    label=label,
                    prompt_token_positions=np.asarray(prompt_positions, dtype=np.int64),
                    task_char_start=int(start),
                    task_char_end=int(end),
                )
            )
        if candidates:
            return candidates

    for token_position, token in zip(instruction_token_positions.tolist(), instruction_tokens):
        candidates.append(
            _TextMaskCandidate(
                label=str(token),
                prompt_token_positions=np.asarray([int(token_position)], dtype=np.int64),
                task_char_start=-1,
                task_char_end=-1,
            )
        )
    return candidates


def _extract_hook_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_hook_tensor(output, replacement):
    if isinstance(output, tuple):
        return (replacement,) + tuple(output[1:])
    return replacement


class OpenVLAExplainabilityAdapter:
    """Shared adapter used by all explainability stages."""

    def __init__(
        self,
        checkpoint: str,
        prompt_style: Optional[str] = None,
        unnorm_key: Optional[str] = None,
        action_dim: Optional[int] = None,
        attention_layers: int = 4,
        attn_implementation: str = "eager",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        import torch

        AutoModelForVision2Seq, AutoProcessor = _ensure_openvla_imports()

        self.prompt_style = _infer_prompt_style(prompt_style, checkpoint)
        self.unnorm_key = unnorm_key
        self.attention_layers = max(1, int(attention_layers))
        self._prompt_cache: Dict[str, _PromptContext] = {}

        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": attn_implementation,
            "torch_dtype": self.dtype,
            "load_in_8bit": bool(load_in_8bit),
            "load_in_4bit": bool(load_in_4bit),
        }
        self.model = AutoModelForVision2Seq.from_pretrained(checkpoint, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        if not load_in_8bit and not load_in_4bit:
            self.model = self.model.to(self.device)
        self.model.eval()

        if action_dim is not None:
            self.action_dim = int(action_dim)
        else:
            try:
                self.action_dim = int(self.model.get_action_dim(self.unnorm_key))
            except Exception:
                self.action_dim = 7

    def _prompt_context(self, instruction: str) -> _PromptContext:
        prompt, prefix, task_text, _suffix = _build_prompt_parts(instruction, prompt_style=self.prompt_style)
        cache_key = f"{self.prompt_style}::{prompt}"
        cached = self._prompt_cache.get(cache_key)
        if cached is not None:
            return cached

        tokenizer = self.processor.tokenizer
        instruction_token_positions, instruction_token_ids, instruction_tokens, instruction_token_offsets = _find_instruction_tokens(
            tokenizer=tokenizer,
            prompt=prompt,
            prefix=prefix,
            task_text=task_text,
        )
        text_mask_candidates = _build_text_mask_candidates(
            task_text=task_text,
            prefix_len=len(prefix),
            instruction_token_positions=instruction_token_positions,
            instruction_token_offsets=instruction_token_offsets,
            instruction_tokens=instruction_tokens,
        )

        context = _PromptContext(
            prompt=prompt,
            task_text=task_text,
            instruction_token_positions=instruction_token_positions,
            instruction_token_ids=instruction_token_ids,
            instruction_tokens=instruction_tokens,
            text_mask_candidates=text_mask_candidates,
        )
        self._prompt_cache[cache_key] = context
        return context

    def _aggregate_attention(
        self,
        attentions,
        num_patches: int,
        instruction_token_positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        if attentions is None:
            raise RuntimeError("model returned no attentions; use --attn-implementation eager")

        layers = list(attentions)[-self.attention_layers :]
        patch_vectors = []
        text_vectors = []
        for layer_attn in layers:
            attn = layer_attn.detach().float()
            query_attention = attn[0, :, -1, :].mean(dim=0)
            patch_slice = query_attention[1 : 1 + num_patches]
            patch_vectors.append(patch_slice.cpu().numpy())

            if instruction_token_positions.size == 0:
                text_vectors.append(np.zeros((0,), dtype=np.float32))
                continue

            text_start = 1 + num_patches
            text_indices = torch.as_tensor(text_start + instruction_token_positions, device=query_attention.device)
            text_slice = torch.index_select(query_attention, dim=0, index=text_indices)
            text_vectors.append(text_slice.cpu().numpy())

        patch_vector = _normalize_vector(np.mean(np.stack(patch_vectors, axis=0), axis=0))
        if text_vectors and text_vectors[0].size > 0:
            text_vector = _normalize_vector(np.mean(np.stack(text_vectors, axis=0), axis=0))
        else:
            text_vector = np.zeros((0,), dtype=np.float32)
        return patch_vector, text_vector

    def _prepare_prompt_batch(
        self,
        image: np.ndarray,
        prompt_context: _PromptContext,
        masked_instruction_positions: Optional[np.ndarray] = None,
    ):
        batch = self.processor(prompt_context.prompt, Image.fromarray(image).convert("RGB"), return_tensors="pt")
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values = batch["pixel_values"].to(device=self.device, dtype=self.dtype)

        if masked_instruction_positions is not None and len(masked_instruction_positions) > 0:
            attention_mask = attention_mask.clone()
            mask_positions = np.asarray(masked_instruction_positions, dtype=np.int64).reshape(-1)
            valid = mask_positions[(mask_positions >= 0) & (mask_positions < attention_mask.shape[1])]
            if valid.size > 0:
                attention_mask[:, valid] = 0
        return input_ids, attention_mask, pixel_values

    def _ensure_blank_action_token(self, input_ids, attention_mask):
        import torch

        if not torch.all(input_ids[:, -1] == BLANK_ACTION_TOKEN_ID):
            blank = torch.full((input_ids.shape[0], 1), BLANK_ACTION_TOKEN_ID, dtype=input_ids.dtype, device=self.device)
            input_ids = torch.cat([input_ids, blank], dim=1)
            blank_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=self.device)
            attention_mask = torch.cat([attention_mask, blank_mask], dim=1)
        return input_ids, attention_mask

    def _decode_sequence(
        self,
        image: np.ndarray,
        instruction: str,
        masked_instruction_positions: Optional[np.ndarray] = None,
        collect_attributions: bool = False,
    ) -> _DecodedSequence:
        import torch

        prompt_context = self._prompt_context(instruction)
        input_ids, attention_mask, pixel_values = self._prepare_prompt_batch(
            image=image,
            prompt_context=prompt_context,
            masked_instruction_positions=masked_instruction_positions,
        )
        input_ids, attention_mask = self._ensure_blank_action_token(input_ids, attention_mask)

        generated_token_ids: List[int] = []
        generated_token_probs: List[float] = []
        patch_token_attributions = []
        text_token_attributions = []
        past_key_values = None
        next_token = None
        num_patches: Optional[int] = None

        for token_idx in range(self.action_dim):
            with torch.no_grad():
                if token_idx == 0:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        use_cache=True,
                        output_attentions=collect_attributions,
                        output_projector_features=True,
                        return_dict=True,
                    )
                    if outputs.projector_features is None:
                        raise RuntimeError("projector features are required to locate image patch tokens")
                    num_patches = int(outputs.projector_features.shape[1])
                else:
                    outputs = self.model(
                        input_ids=next_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=collect_attributions,
                        return_dict=True,
                    )

            logits = outputs.logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated_token_ids.append(int(next_token.item()))
            generated_token_probs.append(float(torch.gather(probs, dim=-1, index=next_token).item()))

            if collect_attributions:
                patch_vector, text_vector = self._aggregate_attention(
                    outputs.attentions,
                    num_patches=int(num_patches),
                    instruction_token_positions=prompt_context.instruction_token_positions,
                )
                patch_token_attributions.append(_fit_attention_grid(patch_vector))
                text_token_attributions.append(text_vector)

            past_key_values = outputs.past_key_values

        raw_pred_action = _decode_action_tokens(
            self.model,
            token_ids=np.asarray(generated_token_ids, dtype=np.int64),
            action_dim=self.action_dim,
            unnorm_key=self.unnorm_key,
        )
        pred_action_xyzg = extract_xyzg(raw_pred_action)[0]

        patch_array = None
        if collect_attributions:
            patch_array = np.stack(patch_token_attributions, axis=0).astype(np.float32)
        text_array = None
        if collect_attributions:
            if text_token_attributions and text_token_attributions[0].size > 0:
                text_array = np.stack(text_token_attributions, axis=0).astype(np.float32)
            else:
                text_array = np.zeros((self.action_dim, 0), dtype=np.float32)

        return _DecodedSequence(
            pred_action_xyzg=np.asarray(pred_action_xyzg, dtype=np.float32),
            raw_pred_action=np.asarray(raw_pred_action, dtype=np.float32),
            target_token_ids=np.asarray(generated_token_ids, dtype=np.int64),
            target_token_probs=np.asarray(generated_token_probs, dtype=np.float32),
            prompt_context=prompt_context,
            num_patches=int(num_patches if num_patches is not None else 0),
            patch_token_attributions=patch_array,
            text_token_attributions=text_array,
        )

    def _build_teacher_forced_inputs(
        self,
        image: np.ndarray,
        instruction: str,
        target_token_ids: np.ndarray,
        masked_instruction_positions: Optional[np.ndarray] = None,
    ):
        import torch

        prompt_context = self._prompt_context(instruction)
        input_ids, attention_mask, pixel_values = self._prepare_prompt_batch(
            image=image,
            prompt_context=prompt_context,
            masked_instruction_positions=masked_instruction_positions,
        )
        input_ids, attention_mask = self._ensure_blank_action_token(input_ids, attention_mask)
        base_length = int(input_ids.shape[1])

        target_token_ids = np.asarray(target_token_ids, dtype=np.int64).reshape(-1)
        if target_token_ids.size > 1:
            teacher_prefix = torch.as_tensor(
                target_token_ids[:-1][None],
                dtype=input_ids.dtype,
                device=self.device,
            )
            prefix_mask = torch.ones((1, target_token_ids.size - 1), dtype=attention_mask.dtype, device=self.device)
            input_ids = torch.cat([input_ids, teacher_prefix], dim=1)
            attention_mask = torch.cat([attention_mask, prefix_mask], dim=1)

        query_text_positions = np.arange(base_length - 1, (base_length - 1) + target_token_ids.size, dtype=np.int64)
        return input_ids, attention_mask, pixel_values, prompt_context, query_text_positions

    def _forward_teacher_forced(self, input_ids, attention_mask, pixel_values):
        import torch

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                use_cache=False,
                output_attentions=False,
                output_projector_features=True,
                return_dict=True,
            )
        if outputs.projector_features is None:
            raise RuntimeError("projector features are required for teacher-forced scoring")
        return outputs, int(outputs.projector_features.shape[1])

    def _score_from_outputs(
        self,
        outputs,
        query_text_positions: np.ndarray,
        target_token_ids: np.ndarray,
        num_patches: int,
    ) -> _TargetSequenceScore:
        import torch

        query_positions = torch.as_tensor(query_text_positions + num_patches, dtype=torch.long, device=outputs.logits.device)
        logits = outputs.logits[:, query_positions, :].float()[0]
        probs = torch.softmax(logits, dim=-1)
        targets = torch.as_tensor(np.asarray(target_token_ids, dtype=np.int64), dtype=torch.long, device=outputs.logits.device)
        target_probs = (
            torch.gather(probs, dim=-1, index=targets[:, None]).squeeze(-1).detach().cpu().numpy().astype(np.float32)
        )
        sequence_logprob = float(np.log(np.clip(target_probs, 1e-12, 1.0)).sum())
        return _TargetSequenceScore(
            target_token_probs=target_probs,
            sequence_logprob=sequence_logprob,
            num_patches=num_patches,
        )

    def predict_step(
        self,
        image: np.ndarray,
        instruction: str,
        masked_instruction_positions: Optional[np.ndarray] = None,
    ) -> _DecodedSequence:
        return self._decode_sequence(
            image=image,
            instruction=instruction,
            masked_instruction_positions=masked_instruction_positions,
            collect_attributions=False,
        )

    def score_target_tokens(
        self,
        image: np.ndarray,
        instruction: str,
        target_token_ids: np.ndarray,
        masked_instruction_positions: Optional[np.ndarray] = None,
    ) -> _TargetSequenceScore:
        input_ids, attention_mask, pixel_values, _prompt_context, query_text_positions = self._build_teacher_forced_inputs(
            image=image,
            instruction=instruction,
            target_token_ids=target_token_ids,
            masked_instruction_positions=masked_instruction_positions,
        )
        outputs, num_patches = self._forward_teacher_forced(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        return self._score_from_outputs(
            outputs=outputs,
            query_text_positions=query_text_positions,
            target_token_ids=target_token_ids,
            num_patches=num_patches,
        )

    def explain_step(self, image: np.ndarray, instruction: str) -> LocalExplanationStep:
        decoded = self._decode_sequence(
            image=image,
            instruction=instruction,
            collect_attributions=True,
        )
        patch_array = np.asarray(decoded.patch_token_attributions, dtype=np.float32)
        text_array = np.asarray(decoded.text_token_attributions, dtype=np.float32)
        return LocalExplanationStep(
            pred_action_xyzg=decoded.pred_action_xyzg.astype(np.float32),
            raw_pred_action=decoded.raw_pred_action.astype(np.float32),
            target_token_ids=decoded.target_token_ids.astype(np.int64),
            target_token_probs=decoded.target_token_probs.astype(np.float32),
            image_patch_attributions=patch_array,
            text_token_attributions=text_array,
            mean_image_patch_attribution=patch_array.mean(axis=0).astype(np.float32),
            mean_text_token_attribution=text_array.mean(axis=0).astype(np.float32),
            text_token_ids=decoded.prompt_context.instruction_token_ids.astype(np.int64),
            text_tokens=list(decoded.prompt_context.instruction_tokens),
            prompt=decoded.prompt_context.prompt,
            task_text=decoded.prompt_context.task_text,
        )

    def _patch_grid_side(self, baseline_step: LocalExplanationStep) -> int:
        mean_patch = np.asarray(baseline_step.mean_image_patch_attribution, dtype=np.float32)
        if mean_patch.ndim != 2 or mean_patch.shape[0] != mean_patch.shape[1]:
            raise ValueError(f"expected square patch grid, got {mean_patch.shape}")
        return int(mean_patch.shape[0])

    def _apply_patch_occlusions(self, image: np.ndarray, patch_indices: Sequence[int], grid_side: int) -> np.ndarray:
        occluded = np.asarray(image, dtype=np.uint8).copy()
        if not patch_indices:
            return occluded
        fill_color = np.asarray(np.mean(occluded, axis=(0, 1)), dtype=np.uint8)
        height, width = occluded.shape[:2]
        for patch_index in sorted(set(int(item) for item in patch_indices)):
            row, col = divmod(patch_index, grid_side)
            y0 = int(round((row / grid_side) * height))
            y1 = int(round(((row + 1) / grid_side) * height))
            x0 = int(round((col / grid_side) * width))
            x1 = int(round(((col + 1) / grid_side) * width))
            occluded[y0:y1, x0:x1] = fill_color
        return occluded

    def patch_occlusion(
        self,
        image: np.ndarray,
        instruction: str,
        baseline_step: LocalExplanationStep,
        top_k: int,
    ) -> InterventionScan:
        baseline_logprob = float(np.log(np.clip(np.asarray(baseline_step.target_token_probs, dtype=np.float32), 1e-12, 1.0)).sum())
        grid_side = self._patch_grid_side(baseline_step)
        num_patches = grid_side * grid_side

        effects = np.zeros((num_patches,), dtype=np.float32)
        for patch_index in range(num_patches):
            occluded = self._apply_patch_occlusions(image=image, patch_indices=[patch_index], grid_side=grid_side)
            score = self.score_target_tokens(
                image=occluded,
                instruction=instruction,
                target_token_ids=baseline_step.target_token_ids,
            )
            effects[patch_index] = np.float32(baseline_logprob - score.sequence_logprob)

        order = np.argsort(-effects, kind="stable")
        limit = 0 if top_k <= 0 else min(int(top_k), int(order.shape[0]))
        candidates: List[InterventionCandidateEffect] = []
        for patch_index in order[:limit]:
            occluded = self._apply_patch_occlusions(image=image, patch_indices=[int(patch_index)], grid_side=grid_side)
            decoded = self.predict_step(image=occluded, instruction=instruction)
            row, col = divmod(int(patch_index), grid_side)
            candidates.append(
                InterventionCandidateEffect(
                    index=int(patch_index),
                    label=f"patch ({row}, {col})",
                    score=float(effects[patch_index]),
                    pred_action_xyzg=np.asarray(decoded.pred_action_xyzg, dtype=np.float32),
                    raw_pred_action=np.asarray(decoded.raw_pred_action, dtype=np.float32),
                    target_token_ids=np.asarray(decoded.target_token_ids, dtype=np.int64),
                    target_token_probs=np.asarray(decoded.target_token_probs, dtype=np.float32),
                )
            )
        return InterventionScan(effect_map=effects.reshape(grid_side, grid_side), top_candidates=candidates)

    def text_masking(
        self,
        image: np.ndarray,
        instruction: str,
        baseline_step: LocalExplanationStep,
        top_k: int,
    ) -> InterventionScan:
        prompt_context = self._prompt_context(instruction)
        baseline_logprob = float(np.log(np.clip(np.asarray(baseline_step.target_token_probs, dtype=np.float32), 1e-12, 1.0)).sum())

        count = int(len(prompt_context.text_mask_candidates))
        effects = np.zeros((count,), dtype=np.float32)
        for candidate_index, candidate in enumerate(prompt_context.text_mask_candidates):
            masked_positions = np.asarray(candidate.prompt_token_positions, dtype=np.int64)
            score = self.score_target_tokens(
                image=image,
                instruction=instruction,
                target_token_ids=baseline_step.target_token_ids,
                masked_instruction_positions=masked_positions,
            )
            effects[candidate_index] = np.float32(baseline_logprob - score.sequence_logprob)

        order = np.argsort(-effects, kind="stable")
        limit = 0 if top_k <= 0 else min(int(top_k), int(order.shape[0]))
        candidates: List[InterventionCandidateEffect] = []
        for candidate_index in order[:limit]:
            candidate = prompt_context.text_mask_candidates[int(candidate_index)]
            masked_positions = np.asarray(candidate.prompt_token_positions, dtype=np.int64)
            decoded = self.predict_step(
                image=image,
                instruction=instruction,
                masked_instruction_positions=masked_positions,
            )
            candidates.append(
                InterventionCandidateEffect(
                    index=int(candidate_index),
                    label=str(candidate.label),
                    score=float(effects[candidate_index]),
                    pred_action_xyzg=np.asarray(decoded.pred_action_xyzg, dtype=np.float32),
                    raw_pred_action=np.asarray(decoded.raw_pred_action, dtype=np.float32),
                    target_token_ids=np.asarray(decoded.target_token_ids, dtype=np.int64),
                    target_token_probs=np.asarray(decoded.target_token_probs, dtype=np.float32),
                    task_char_start=int(candidate.task_char_start),
                    task_char_end=int(candidate.task_char_end),
                )
            )
        return InterventionScan(effect_map=effects, top_candidates=candidates)

    def minimal_counterfactual_edits(
        self,
        image: np.ndarray,
        instruction: str,
        baseline_step: LocalExplanationStep,
        patch_occlusion: Optional[InterventionScan],
        text_masking: Optional[InterventionScan],
        max_edits: int,
    ) -> Tuple[List[CounterfactualEdit], bool]:
        if patch_occlusion is None:
            patch_occlusion = self.patch_occlusion(image=image, instruction=instruction, baseline_step=baseline_step, top_k=0)
        if text_masking is None:
            text_masking = self.text_masking(image=image, instruction=instruction, baseline_step=baseline_step, top_k=0)

        baseline_logprob = float(np.log(np.clip(np.asarray(baseline_step.target_token_probs, dtype=np.float32), 1e-12, 1.0)).sum())
        grid_side = self._patch_grid_side(baseline_step)
        prompt_context = self._prompt_context(instruction)

        ranked_candidates: List[Tuple[str, int, str, float]] = []
        for patch_index, score in enumerate(np.asarray(patch_occlusion.effect_map, dtype=np.float32).reshape(-1)):
            row, col = divmod(int(patch_index), grid_side)
            ranked_candidates.append(("patch", int(patch_index), f"patch ({row}, {col})", float(score)))
        for candidate_index, score in enumerate(np.asarray(text_masking.effect_map, dtype=np.float32).reshape(-1)):
            label = (
                prompt_context.text_mask_candidates[candidate_index].label
                if candidate_index < len(prompt_context.text_mask_candidates)
                else f"text span {candidate_index}"
            )
            ranked_candidates.append(("text", int(candidate_index), str(label), float(score)))
        ranked_candidates.sort(key=lambda item: item[3], reverse=True)

        selected_patches: List[int] = []
        selected_mask_positions: set[int] = set()
        edits: List[CounterfactualEdit] = []
        success = False

        max_edits = max(0, int(max_edits))
        for edit_type, index, label, single_effect_score in ranked_candidates:
            if len(edits) >= max_edits:
                break
            if edit_type == "patch":
                selected_patches.append(index)
            else:
                if index < 0 or index >= len(prompt_context.text_mask_candidates):
                    continue
                candidate_positions = np.asarray(prompt_context.text_mask_candidates[index].prompt_token_positions, dtype=np.int64).reshape(-1)
                if candidate_positions.size == 0:
                    continue
                candidate_position_set = set(int(item) for item in candidate_positions.tolist())
                if candidate_position_set.issubset(selected_mask_positions):
                    continue
                selected_mask_positions.update(candidate_position_set)

            modified_image = self._apply_patch_occlusions(image=image, patch_indices=selected_patches, grid_side=grid_side)
            masked_positions = None if not selected_mask_positions else np.asarray(sorted(selected_mask_positions), dtype=np.int64)
            decoded = self.predict_step(
                image=modified_image,
                instruction=instruction,
                masked_instruction_positions=masked_positions,
            )
            score = self.score_target_tokens(
                image=modified_image,
                instruction=instruction,
                target_token_ids=baseline_step.target_token_ids,
                masked_instruction_positions=masked_positions,
            )
            changed = not np.array_equal(
                np.asarray(decoded.target_token_ids, dtype=np.int64),
                np.asarray(baseline_step.target_token_ids, dtype=np.int64),
            )
            edits.append(
                CounterfactualEdit(
                    step_rank=len(edits) + 1,
                    edit_type=edit_type,
                    index=index,
                    label=label,
                    single_effect_score=float(single_effect_score),
                    cumulative_sequence_logprob=float(score.sequence_logprob),
                    cumulative_sequence_logprob_drop=float(baseline_logprob - score.sequence_logprob),
                    changed_prediction=bool(changed),
                    pred_action_xyzg=np.asarray(decoded.pred_action_xyzg, dtype=np.float32),
                    raw_pred_action=np.asarray(decoded.raw_pred_action, dtype=np.float32),
                    target_token_ids=np.asarray(decoded.target_token_ids, dtype=np.int64),
                    target_token_probs=np.asarray(decoded.target_token_probs, dtype=np.float32),
                )
            )
            if changed:
                success = True
                break
        return edits, success

    def _decoder_layers(self) -> List[object]:
        language_model = self.model.language_model
        candidates = [
            ("model", "layers"),
            ("model", "decoder", "layers"),
            ("base_model", "model", "layers"),
        ]
        for path in candidates:
            node = language_model
            found = True
            for attr in path:
                if not hasattr(node, attr):
                    found = False
                    break
                node = getattr(node, attr)
            if found:
                return list(node)
        raise RuntimeError("could not locate decoder layers inside the OpenVLA language model")

    def _cross_attention_modules(self) -> List[Tuple[str, object]]:
        modules: List[Tuple[str, object]] = []
        for name, module in self.model.language_model.named_modules():
            lowered = name.lower()
            if "cross_attn" in lowered or "encoder_attn" in lowered:
                modules.append((name, module))
        return modules

    def _prepare_corruption(
        self,
        image: np.ndarray,
        instruction: str,
        baseline_step: LocalExplanationStep,
        corruption_type: str,
        corruption_index: Optional[int],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], int, str]:
        prompt_context = self._prompt_context(instruction)
        if corruption_type == "patch-occlusion":
            grid_side = self._patch_grid_side(baseline_step)
            if corruption_index is None:
                patch_scan = self.patch_occlusion(image=image, instruction=instruction, baseline_step=baseline_step, top_k=0)
                corruption_index = int(np.argmax(np.asarray(patch_scan.effect_map, dtype=np.float32).reshape(-1)))
            row, col = divmod(int(corruption_index), grid_side)
            corrupted_image = self._apply_patch_occlusions(image=image, patch_indices=[int(corruption_index)], grid_side=grid_side)
            return corrupted_image, None, int(corruption_index), f"patch ({row}, {col})"

        if corruption_type == "text-masking":
            if corruption_index is None:
                text_scan = self.text_masking(image=image, instruction=instruction, baseline_step=baseline_step, top_k=0)
                if np.asarray(text_scan.effect_map).size == 0:
                    raise RuntimeError("no instruction spans available for text masking")
                corruption_index = int(np.argmax(np.asarray(text_scan.effect_map, dtype=np.float32)))
            if int(corruption_index) < 0 or int(corruption_index) >= len(prompt_context.text_mask_candidates):
                raise IndexError(f"text corruption index out of range: {corruption_index}")
            candidate = prompt_context.text_mask_candidates[int(corruption_index)]
            masked_positions = np.asarray(candidate.prompt_token_positions, dtype=np.int64)
            return np.asarray(image, dtype=np.uint8).copy(), masked_positions, int(corruption_index), str(candidate.label)

        raise ValueError(f"unsupported corruption_type: {corruption_type}")

    def _normalized_restoration(self, clean_logprob: float, corrupted_logprob: float, patched_logprob: float) -> float:
        gap = clean_logprob - corrupted_logprob
        if abs(gap) <= 1e-8:
            return 0.0
        return float((patched_logprob - corrupted_logprob) / gap)

    def causal_localization(
        self,
        image: np.ndarray,
        instruction: str,
        baseline_step: LocalExplanationStep,
        corruption_type: str,
        corruption_index: Optional[int],
        per_cross_attention: bool,
    ) -> CausalLocalizationStep:
        import torch

        decoder_layers = self._decoder_layers()
        self_attn_layers = [layer.self_attn for layer in decoder_layers if hasattr(layer, "self_attn")]
        if not self_attn_layers:
            raise RuntimeError("decoder layers do not expose self_attn modules for causal localization")

        head_count = int(getattr(self_attn_layers[0], "num_heads"))
        default_head_dim = int(getattr(self_attn_layers[0].o_proj, "in_features", 0) // max(1, head_count))
        head_dim = int(getattr(self_attn_layers[0], "head_dim", default_head_dim))

        clean_input_ids, clean_attention_mask, clean_pixel_values, prompt_context, query_text_positions = self._build_teacher_forced_inputs(
            image=image,
            instruction=instruction,
            target_token_ids=baseline_step.target_token_ids,
            masked_instruction_positions=None,
        )

        clean_layer_cache: Dict[int, torch.Tensor] = {}
        clean_head_cache: Dict[int, torch.Tensor] = {}
        clean_cross_cache: Dict[str, torch.Tensor] = {}
        capture_handles = []
        try:
            for layer_idx, layer in enumerate(decoder_layers):
                def layer_capture(_module, _inputs, output, layer_idx=layer_idx):
                    clean_layer_cache[layer_idx] = _extract_hook_tensor(output).detach()

                capture_handles.append(layer.register_forward_hook(layer_capture))

                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                    def head_capture(_module, inputs, layer_idx=layer_idx):
                        clean_head_cache[layer_idx] = inputs[0].detach()

                    capture_handles.append(layer.self_attn.o_proj.register_forward_pre_hook(head_capture))

            cross_attention_modules = self._cross_attention_modules() if per_cross_attention else []
            for name, module in cross_attention_modules:
                def cross_capture(_module, _inputs, output, name=name):
                    tensor = _extract_hook_tensor(output)
                    if getattr(tensor, "ndim", 0) == 3:
                        clean_cross_cache[name] = tensor.detach()

                capture_handles.append(module.register_forward_hook(cross_capture))

            clean_outputs, num_patches = self._forward_teacher_forced(
                input_ids=clean_input_ids,
                attention_mask=clean_attention_mask,
                pixel_values=clean_pixel_values,
            )
        finally:
            for handle in capture_handles:
                handle.remove()

        clean_score = self._score_from_outputs(
            outputs=clean_outputs,
            query_text_positions=query_text_positions,
            target_token_ids=baseline_step.target_token_ids,
            num_patches=num_patches,
        )
        query_positions = query_text_positions + num_patches
        query_index_tensor = torch.as_tensor(query_positions, dtype=torch.long, device=self.device)

        corrupted_image, masked_positions, resolved_index, corruption_label = self._prepare_corruption(
            image=image,
            instruction=instruction,
            baseline_step=baseline_step,
            corruption_type=corruption_type,
            corruption_index=corruption_index,
        )
        corrupted_decoded = self.predict_step(
            image=corrupted_image,
            instruction=instruction,
            masked_instruction_positions=masked_positions,
        )
        corrupted_input_ids, corrupted_attention_mask, corrupted_pixel_values, _prompt_context, corrupted_query_text_positions = self._build_teacher_forced_inputs(
            image=corrupted_image,
            instruction=instruction,
            target_token_ids=baseline_step.target_token_ids,
            masked_instruction_positions=masked_positions,
        )
        corrupted_outputs, corrupted_num_patches = self._forward_teacher_forced(
            input_ids=corrupted_input_ids,
            attention_mask=corrupted_attention_mask,
            pixel_values=corrupted_pixel_values,
        )
        corrupted_score = self._score_from_outputs(
            outputs=corrupted_outputs,
            query_text_positions=corrupted_query_text_positions,
            target_token_ids=baseline_step.target_token_ids,
            num_patches=corrupted_num_patches,
        )
        if corrupted_num_patches != num_patches:
            raise RuntimeError("clean and corrupted runs produced different patch counts; causal localization requires a fixed patch layout")

        layer_scores = np.zeros((len(decoder_layers),), dtype=np.float32)
        head_scores = np.zeros((len(decoder_layers), head_count), dtype=np.float32)

        for layer_idx, layer in enumerate(decoder_layers):
            clean_tensor = clean_layer_cache.get(layer_idx)
            if clean_tensor is None:
                continue

            def patch_layer(_module, _inputs, output, clean_tensor=clean_tensor):
                tensor = _extract_hook_tensor(output)
                patched = tensor.clone()
                patched[:, query_index_tensor, :] = clean_tensor[:, query_index_tensor, :]
                return _replace_hook_tensor(output, patched)

            handle = layer.register_forward_hook(patch_layer)
            try:
                patched_outputs, patched_num_patches = self._forward_teacher_forced(
                    input_ids=corrupted_input_ids,
                    attention_mask=corrupted_attention_mask,
                    pixel_values=corrupted_pixel_values,
                )
            finally:
                handle.remove()
            patched_score = self._score_from_outputs(
                outputs=patched_outputs,
                query_text_positions=corrupted_query_text_positions,
                target_token_ids=baseline_step.target_token_ids,
                num_patches=patched_num_patches,
            )
            layer_scores[layer_idx] = np.float32(
                self._normalized_restoration(clean_score.sequence_logprob, corrupted_score.sequence_logprob, patched_score.sequence_logprob)
            )

            if layer_idx not in clean_head_cache or not hasattr(layer, "self_attn") or not hasattr(layer.self_attn, "o_proj"):
                continue
            clean_head_tensor = clean_head_cache[layer_idx]
            for head_idx in range(head_count):
                start = head_idx * head_dim
                end = start + head_dim

                def patch_head(_module, inputs, clean_head_tensor=clean_head_tensor, start=start, end=end):
                    tensor = inputs[0]
                    patched = tensor.clone()
                    patched[:, query_index_tensor, start:end] = clean_head_tensor[:, query_index_tensor, start:end]
                    return (patched,)

                head_handle = layer.self_attn.o_proj.register_forward_pre_hook(patch_head)
                try:
                    patched_outputs, patched_num_patches = self._forward_teacher_forced(
                        input_ids=corrupted_input_ids,
                        attention_mask=corrupted_attention_mask,
                        pixel_values=corrupted_pixel_values,
                    )
                finally:
                    head_handle.remove()
                patched_score = self._score_from_outputs(
                    outputs=patched_outputs,
                    query_text_positions=corrupted_query_text_positions,
                    target_token_ids=baseline_step.target_token_ids,
                    num_patches=patched_num_patches,
                )
                head_scores[layer_idx, head_idx] = np.float32(
                    self._normalized_restoration(clean_score.sequence_logprob, corrupted_score.sequence_logprob, patched_score.sequence_logprob)
                )

        cross_scores = None
        cross_labels = None
        cross_attention_modules = self._cross_attention_modules() if per_cross_attention else []
        if cross_attention_modules:
            cross_labels = []
            collected_scores = []
            for name, module in cross_attention_modules:
                clean_tensor = clean_cross_cache.get(name)
                if clean_tensor is None:
                    continue

                def patch_cross(_module, _inputs, output, clean_tensor=clean_tensor):
                    tensor = _extract_hook_tensor(output)
                    if getattr(tensor, "ndim", 0) != 3:
                        return output
                    patched = tensor.clone()
                    patched[:, query_index_tensor, :] = clean_tensor[:, query_index_tensor, :]
                    return _replace_hook_tensor(output, patched)

                handle = module.register_forward_hook(patch_cross)
                try:
                    patched_outputs, patched_num_patches = self._forward_teacher_forced(
                        input_ids=corrupted_input_ids,
                        attention_mask=corrupted_attention_mask,
                        pixel_values=corrupted_pixel_values,
                    )
                finally:
                    handle.remove()
                patched_score = self._score_from_outputs(
                    outputs=patched_outputs,
                    query_text_positions=corrupted_query_text_positions,
                    target_token_ids=baseline_step.target_token_ids,
                    num_patches=patched_num_patches,
                )
                cross_labels.append(name)
                collected_scores.append(
                    self._normalized_restoration(clean_score.sequence_logprob, corrupted_score.sequence_logprob, patched_score.sequence_logprob)
                )
            if collected_scores:
                cross_scores = np.asarray(collected_scores, dtype=np.float32)

        return CausalLocalizationStep(
            baseline_pred_action_xyzg=np.asarray(baseline_step.pred_action_xyzg, dtype=np.float32),
            baseline_raw_pred_action=np.asarray(baseline_step.raw_pred_action, dtype=np.float32),
            baseline_target_token_ids=np.asarray(baseline_step.target_token_ids, dtype=np.int64),
            baseline_target_token_probs=np.asarray(baseline_step.target_token_probs, dtype=np.float32),
            clean_sequence_logprob=float(clean_score.sequence_logprob),
            corruption_type=corruption_type,
            corruption_index=int(resolved_index),
            corruption_label=corruption_label,
            corrupted_pred_action_xyzg=np.asarray(corrupted_decoded.pred_action_xyzg, dtype=np.float32),
            corrupted_raw_pred_action=np.asarray(corrupted_decoded.raw_pred_action, dtype=np.float32),
            corrupted_target_token_ids=np.asarray(corrupted_decoded.target_token_ids, dtype=np.int64),
            corrupted_target_token_probs=np.asarray(corrupted_score.target_token_probs, dtype=np.float32),
            corrupted_sequence_logprob=float(corrupted_score.sequence_logprob),
            layer_restoration_scores=layer_scores,
            head_restoration_scores=head_scores,
            layer_labels=[f"layer_{idx}" for idx in range(len(decoder_layers))],
            cross_attention_restoration_scores=cross_scores,
            cross_attention_labels=cross_labels,
            prompt=baseline_step.prompt,
            task_text=baseline_step.task_text,
            text_token_ids=np.asarray(baseline_step.text_token_ids, dtype=np.int64),
            text_tokens=list(baseline_step.text_tokens),
        )


class OpenVLALocalExplanationAdapter(OpenVLAExplainabilityAdapter):
    """Backwards-compatible alias for the stage-1 local explainability adapter."""


class OpenVLAInterventionAdapter(OpenVLAExplainabilityAdapter):
    """Stage-2 intervention-test adapter."""


class OpenVLACausalLocalizationAdapter(OpenVLAExplainabilityAdapter):
    """Stage-3 causal-localization adapter."""
