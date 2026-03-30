"""
explainability.py

Comprehensive explainability module for OpenVLA action predictions.
Provides 5 features:
  1. Action Token Probabilities — distribution over 256 bins per action dim
  2. Attention Heatmaps — cross-modal attention from action tokens to vision patches
  3. Logit Lens — project hidden states at each LLM layer through LM head
  4. GradCAM Saliency — gradient-based importance map over image regions
  5. Projector Features — raw projected vision embeddings
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class ExplainabilityConfig:
    """Configuration for explainability data collection."""

    enabled: bool = False
    save_dir: str = "./experiments/logs/explainability"
    save_every_n_steps: int = 1

    # Feature toggles
    save_action_probs: bool = True
    save_attention_maps: bool = True
    save_logit_lens: bool = True
    save_gradcam: bool = True
    save_projector_features: bool = True

    # Attention settings — which layers to capture
    attention_layers: List[int] = field(default_factory=lambda: [0, 7, 15, 23, 31])
    top_k: int = 10


# Action dimension labels for visualization
ACTION_DIM_LABELS = [
    "x (forward/back)",
    "y (left/right)",
    "z (up/down)",
    "roll",
    "pitch",
    "yaw",
    "gripper",
]


class ExplainabilityCollector:
    """Collects and saves explainability data for OpenVLA action predictions."""

    def __init__(self, config: ExplainabilityConfig, model, processor):
        self.config = config
        self.model = model
        self.processor = processor

        # Cache model properties
        self.vocab_size = model.vocab_size
        self.bin_centers = model.bin_centers
        self.n_action_bins = len(model.bin_centers)

        # Action token ID range: action tokens sit BELOW vocab_size in the logits.
        # Mapping: bin b -> token_id = vocab_size - b - 1
        # So token IDs range from (vocab_size - n_action_bins) to (vocab_size - 1).
        # In the logits tensor, we slice [vocab_size - n_bins : vocab_size] then
        # flip so index 0 = bin 0.
        self.action_token_start = self.vocab_size - self.n_action_bins
        self.action_token_end = self.vocab_size

        # Create run directory
        self.run_id = None

    def set_run_id(self, run_id: str):
        """Set the run ID and create the output directory."""
        self.run_id = run_id
        os.makedirs(os.path.join(self.config.save_dir, run_id), exist_ok=True)

    def _get_step_dir(self, task_id: int, episode_idx: int, step_idx: int) -> str:
        """Get the directory for a specific step's explainability data."""
        step_dir = os.path.join(
            self.config.save_dir,
            self.run_id,
            f"task_{task_id}",
            f"episode_{episode_idx}",
            f"step_{step_idx}",
        )
        os.makedirs(step_dir, exist_ok=True)
        return step_dir

    def predict_action_with_explanations(
        self,
        inputs: Dict[str, torch.Tensor],
        unnorm_key: str,
        image_pil: Image.Image,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run action prediction with all enabled explainability features.

        Returns:
            (action, data_dict) where action is the 7D unnormalized action
            and data_dict contains all collected explainability data.
        """
        data_dict = {}
        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        attention_mask = inputs.get("attention_mask")

        # Ensure empty token at end of prompt
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (
                    input_ids,
                    torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(
                        input_ids.device
                    ),
                ),
                dim=1,
            )

        action_dim = self.model.get_action_dim(unnorm_key)

        # =====================================================================
        # Phase 1: Action Token Probabilities
        # =====================================================================
        action = None
        if self.config.save_action_probs:
            try:
                action, probs_data = self._collect_action_probs(
                    input_ids, pixel_values, attention_mask, unnorm_key, action_dim
                )
                data_dict["action_probs"] = probs_data
            except Exception as e:
                print(f"[Explainability] Action probs failed: {e}")

        # Fallback: standard prediction if action probs failed or disabled
        if action is None:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_new_tokens=action_dim,
                    do_sample=False,
                )
            action = self._decode_action(generated_ids, unnorm_key, action_dim)

        data_dict["action"] = action.tolist()

        # =====================================================================
        # Phase 2: Forward pass features (attention, hidden states, projector)
        # =====================================================================
        need_forward = (
            self.config.save_attention_maps
            or self.config.save_logit_lens
            or self.config.save_projector_features
        )

        if need_forward:
            try:
                forward_data = self._collect_forward_features(
                    input_ids, pixel_values, attention_mask
                )
            except Exception as e:
                print(f"[Explainability] Forward features failed: {e}")
                forward_data = {}

            if self.config.save_attention_maps and "attentions" in forward_data:
                try:
                    attn_data = self._process_attention_maps(
                        forward_data["attentions"], image_pil
                    )
                    data_dict["attention_maps"] = attn_data
                except Exception as e:
                    print(f"[Explainability] Attention maps failed: {e}")

            if self.config.save_logit_lens and "hidden_states" in forward_data:
                try:
                    lens_data = self._process_logit_lens(forward_data["hidden_states"])
                    data_dict["logit_lens"] = lens_data
                except Exception as e:
                    print(f"[Explainability] Logit lens failed: {e}")

            if (
                self.config.save_projector_features
                and "projector_features" in forward_data
            ):
                data_dict["projector_features"] = forward_data[
                    "projector_features"
                ]

        # =====================================================================
        # Phase 3: GradCAM
        # =====================================================================
        if self.config.save_gradcam:
            try:
                gradcam_data = self._collect_gradcam(
                    input_ids, pixel_values, attention_mask, unnorm_key, action_dim, image_pil
                )
                data_dict["gradcam"] = gradcam_data
            except Exception as e:
                print(f"[Explainability] GradCAM failed: {e}")

        # Save input image
        data_dict["input_image"] = image_pil

        return action, data_dict

    def _collect_action_probs(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        unnorm_key: str,
        action_dim: int,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Collect action token probability distributions using generate(output_scores=True)."""
        with torch.inference_mode():
            gen_output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=action_dim,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_ids = gen_output.sequences
        scores = gen_output.scores  # tuple of (action_dim,) tensors, each [1, full_vocab_size]

        action = self._decode_action(generated_ids, unnorm_key, action_dim)

        # Process scores for each action dimension
        probs_data = {"per_dim": [], "summary": {}}
        entropies = []
        max_probs = []

        for dim_idx, score_tensor in enumerate(scores):
            # Get logits for action token range only
            # Action tokens: IDs from vocab_size to vocab_size + n_action_bins
            full_logits = score_tensor[0]  # [full_vocab_size]

            # Slice action bin range and flip so index 0 = bin 0
            # (higher token IDs map to lower bin indices)
            action_logits = full_logits[
                self.action_token_start : self.action_token_end
            ].flip(0)
            action_probs = F.softmax(action_logits.float(), dim=-1).cpu().numpy()

            # Top-k alternatives
            top_k_indices = np.argsort(action_probs)[-self.config.top_k :][::-1]
            top_k_probs = action_probs[top_k_indices]
            top_k_bin_values = self.bin_centers[top_k_indices]

            # Confidence metrics
            entropy = -np.sum(
                action_probs * np.log(action_probs + 1e-10)
            )
            max_prob = float(np.max(action_probs))
            entropies.append(float(entropy))
            max_probs.append(max_prob)

            dim_data = {
                "distribution": action_probs.tolist(),
                "top_k_indices": top_k_indices.tolist(),
                "top_k_probs": top_k_probs.tolist(),
                "top_k_bin_values": top_k_bin_values.tolist(),
                "entropy": float(entropy),
                "max_prob": max_prob,
                "predicted_bin": int(top_k_indices[0]),
                "predicted_value": float(top_k_bin_values[0]),
                "label": ACTION_DIM_LABELS[dim_idx]
                if dim_idx < len(ACTION_DIM_LABELS)
                else f"dim_{dim_idx}",
            }
            probs_data["per_dim"].append(dim_data)

        probs_data["summary"] = {
            "mean_entropy": float(np.mean(entropies)),
            "mean_max_prob": float(np.mean(max_probs)),
            "min_max_prob": float(np.min(max_probs)),
            "entropies": entropies,
            "max_probs": max_probs,
        }

        # Clean up GPU memory
        del scores, gen_output
        torch.cuda.empty_cache()

        return action, probs_data

    def _collect_forward_features(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Run a forward pass collecting attention maps, hidden states, and projector features.

        Note: output_attentions=True forces SDPA -> eager fallback, which may not
        support BFloat16 with quantized models. If that fails, we retry without
        attention collection so hidden_states and projector_features still work.
        """
        result = {}
        request_attentions = self.config.save_attention_maps

        with torch.inference_mode():
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    output_attentions=request_attentions,
                    output_hidden_states=self.config.save_logit_lens,
                    output_projector_features=self.config.save_projector_features,
                    return_dict=True,
                )
            except Exception as e:
                if request_attentions:
                    # Attention collection failed (likely BFloat16 + eager fallback).
                    # Retry without attentions so we still get hidden_states + projector.
                    print(f"[Explainability] Attention collection failed ({e}), retrying without attentions")
                    request_attentions = False
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        output_attentions=False,
                        output_hidden_states=self.config.save_logit_lens,
                        output_projector_features=self.config.save_projector_features,
                        return_dict=True,
                    )
                else:
                    raise

        # Collect attentions (move to CPU immediately)
        if request_attentions and outputs.attentions is not None:
            # Only keep selected layers to save memory
            selected_attentions = []
            for layer_idx in self.config.attention_layers:
                if layer_idx < len(outputs.attentions):
                    selected_attentions.append(
                        outputs.attentions[layer_idx].cpu()
                    )

            result["attentions"] = selected_attentions

        # Collect hidden states (move to CPU immediately)
        if outputs.hidden_states is not None:
            result["hidden_states"] = [
                hs.cpu() for hs in outputs.hidden_states
            ]

        # Collect projector features
        if outputs.projector_features is not None:
            result["projector_features"] = (
                outputs.projector_features[0].cpu().float().numpy()
            )  # [num_patches, llm_dim]

        # Clean up GPU memory
        del outputs
        torch.cuda.empty_cache()

        return result

    def _process_attention_maps(
        self,
        attentions: List[torch.Tensor],
        image_pil: Image.Image,
    ) -> Dict[str, Any]:
        """Process attention maps into heatmaps overlaid on input image.

        Vision patches are at positions 1:258 in the attention matrix
        (position 0 is BOS, 1-257 are 256 spatial + 1 CLS vision patches).
        """
        result = {"layers": {}}

        # The last token position attends to vision patches
        # Vision patch positions: 1 to 257 (256 spatial + 1 CLS)
        vision_start = 1
        vision_end = 257  # exclusive, 256 spatial patches
        n_spatial_patches = 256
        grid_size = 16  # sqrt(256) = 16

        for layer_offset, attn_tensor in enumerate(attentions):
            layer_idx = self.config.attention_layers[layer_offset] if layer_offset < len(self.config.attention_layers) else layer_offset
            # attn_tensor: [batch, num_heads, seq_len, seq_len]
            # Take last token's attention over all positions, average across heads
            attn = attn_tensor[0]  # [num_heads, seq_len, seq_len]
            last_token_attn = attn[:, -1, :]  # [num_heads, seq_len]

            # Average across heads
            avg_attn = last_token_attn.mean(dim=0).float()  # [seq_len]

            # Extract attention to spatial vision patches
            if avg_attn.shape[0] > vision_end:
                vision_attn = avg_attn[vision_start:vision_end].numpy()
            else:
                # Fallback: take what we can
                vision_attn = avg_attn[vision_start : min(vision_end, avg_attn.shape[0])].numpy()
                if len(vision_attn) < n_spatial_patches:
                    vision_attn = np.pad(
                        vision_attn, (0, n_spatial_patches - len(vision_attn))
                    )

            # Reshape to 16x16 spatial grid
            attn_map = vision_attn[:n_spatial_patches].reshape(
                grid_size, grid_size
            )

            # Normalize to [0, 1]
            attn_min, attn_max = attn_map.min(), attn_map.max()
            if attn_max > attn_min:
                attn_map = (attn_map - attn_min) / (attn_max - attn_min)

            result["layers"][f"layer_{layer_idx}"] = attn_map

        # Clean up
        del attentions

        return result

    def _process_logit_lens(
        self, hidden_states: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Project hidden states at each layer through LM head to see action prediction evolution.

        For the last position (which predicts the first action token), project
        each layer's hidden state through the LM head, slice action bins, softmax.
        """
        result = {"layers": [], "n_layers": len(hidden_states)}

        lm_head = self.model.language_model.lm_head

        for layer_idx, hs in enumerate(hidden_states):
            # hs: [batch, seq_len, hidden_dim]
            # Take the last position
            last_pos_hs = hs[0, -1, :]  # [hidden_dim]

            # Project through LM head
            with torch.inference_mode():
                logits = lm_head(last_pos_hs.to(lm_head.weight.device).to(lm_head.weight.dtype))  # [vocab_size_full]

            # Slice action token range and flip so index 0 = bin 0
            action_logits = logits[
                self.action_token_start : self.action_token_end
            ].flip(0)
            action_probs = F.softmax(action_logits.float(), dim=-1).cpu().numpy()

            # Get predicted bin and value
            predicted_bin = int(np.argmax(action_probs))
            predicted_value = float(self.bin_centers[predicted_bin])

            layer_data = {
                "layer": layer_idx,
                "predicted_bin": predicted_bin,
                "predicted_value": predicted_value,
                "max_prob": float(np.max(action_probs)),
                "entropy": float(
                    -np.sum(action_probs * np.log(action_probs + 1e-10))
                ),
                "distribution": action_probs.tolist(),
            }
            result["layers"].append(layer_data)

        # Clean up
        del hidden_states
        torch.cuda.empty_cache()

        return result

    def _collect_gradcam(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        unnorm_key: str,
        action_dim: int,
        image_pil: Image.Image,
    ) -> Dict[str, Any]:
        """Compute GradCAM saliency map.

        Run a separate forward pass with gradients enabled.
        Register a hook on the projector output.
        Compute pseudo-loss as sum of log-probs of predicted action tokens.
        Backward to get gradients at projector output.
        GradCAM = ReLU(sum(grad * features)).
        Works with 4-bit quantization because the projector is in bf16.
        """
        result = {}

        # Storage for hook
        projector_output = {}
        projector_grad = {}

        def forward_hook(module, input, output):
            projector_output["features"] = output

        def backward_hook(module, grad_input, grad_output):
            projector_grad["grad"] = grad_output[0]

        # Register hooks on the projector
        handle_fwd = self.model.projector.register_forward_hook(forward_hook)
        handle_bwd = self.model.projector.register_full_backward_hook(backward_hook)

        try:
            # Run forward pass WITH gradients (not under inference_mode)
            with torch.enable_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

                # Get logits at the last position for the first action token
                logits = outputs.logits[0, -1, :]  # [vocab_size_full]

                # Compute pseudo-loss: sum of log-probs for action bin tokens
                action_logits = logits[
                    self.action_token_start : self.action_token_end
                ].flip(0)
                action_log_probs = F.log_softmax(action_logits.float(), dim=-1)

                # Use the argmax token as the predicted action
                predicted_bin = action_logits.argmax()
                pseudo_loss = action_log_probs[predicted_bin]

                # Backward
                pseudo_loss.backward()

            # Compute GradCAM from projector hook outputs
            if "features" in projector_output and "grad" in projector_grad:
                features = projector_output["features"][0].detach()  # [num_patches, llm_dim]
                grads = projector_grad["grad"][0].detach()  # [num_patches, llm_dim]

                # GradCAM: weight each feature channel by its gradient, then sum
                weights = grads.mean(dim=0)  # [llm_dim] — global average pooling of gradients
                cam = (features * weights).sum(dim=-1)  # [num_patches]
                cam = F.relu(cam)  # ReLU

                cam = cam.cpu().float().numpy()

                # Handle spatial patches (skip CLS if present)
                n_spatial = 256
                if len(cam) == 257:
                    spatial_cam = cam[1:]  # skip CLS token
                elif len(cam) >= 256:
                    spatial_cam = cam[:n_spatial]
                else:
                    spatial_cam = np.pad(cam, (0, n_spatial - len(cam)))

                # Reshape to 16x16
                cam_2d = spatial_cam.reshape(16, 16)

                # Normalize
                cam_min, cam_max = cam_2d.min(), cam_2d.max()
                if cam_max > cam_min:
                    cam_2d = (cam_2d - cam_min) / (cam_max - cam_min)

                result["cam_2d"] = cam_2d
                result["raw_cam"] = spatial_cam

        finally:
            # Remove hooks and clean up
            handle_fwd.remove()
            handle_bwd.remove()

            # Zero out gradients
            self.model.zero_grad(set_to_none=True)

            # Clean up
            del projector_output, projector_grad
            torch.cuda.empty_cache()

        return result

    def _decode_action(
        self,
        generated_ids: torch.Tensor,
        unnorm_key: str,
        action_dim: int,
    ) -> np.ndarray:
        """Decode generated token IDs into an unnormalized action vector."""
        predicted_action_token_ids = (
            generated_ids[0, -action_dim:].cpu().numpy()
        )
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize
        action_norm_stats = self.model.get_action_stats(unnorm_key)
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high = np.array(action_norm_stats["q99"])
        action_low = np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low)
            + action_low,
            normalized_actions,
        )
        return actions

    def save_step(
        self,
        data_dict: Dict[str, Any],
        task_id: int,
        episode_idx: int,
        step_idx: int,
        task_description: str = "",
        prompt: str = "",
    ):
        """Save all explainability data for a single step."""
        step_dir = self._get_step_dir(task_id, episode_idx, step_idx)

        # Save input image
        if "input_image" in data_dict:
            data_dict["input_image"].save(os.path.join(step_dir, "input_image.png"))

        # Save metadata
        metadata = {
            "task_id": task_id,
            "episode_idx": episode_idx,
            "step_idx": step_idx,
            "task_description": task_description,
            "prompt": prompt,
            "action": data_dict.get("action", []),
        }
        with open(os.path.join(step_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Save action probabilities
        if "action_probs" in data_dict:
            self._save_action_probs(data_dict["action_probs"], step_dir)

        # Save attention heatmaps
        if "attention_maps" in data_dict:
            self._save_attention_maps(
                data_dict["attention_maps"], data_dict.get("input_image"), step_dir
            )

        # Save logit lens
        if "logit_lens" in data_dict:
            self._save_logit_lens(data_dict["logit_lens"], step_dir)

        # Save GradCAM
        if "gradcam" in data_dict:
            self._save_gradcam(
                data_dict["gradcam"], data_dict.get("input_image"), step_dir
            )

        # Save projector features
        if "projector_features" in data_dict:
            self._save_projector_features(
                data_dict["projector_features"], step_dir
            )

    def _save_action_probs(self, probs_data: Dict[str, Any], step_dir: str):
        """Save action probability data as JSON and plot."""
        # Save JSON
        with open(os.path.join(step_dir, "action_probs.json"), "w") as f:
            json.dump(probs_data, f, indent=2)

        # Create visualization
        n_dims = len(probs_data["per_dim"])
        fig, axes = plt.subplots(n_dims, 1, figsize=(12, 3 * n_dims))
        if n_dims == 1:
            axes = [axes]

        for dim_idx, dim_data in enumerate(probs_data["per_dim"]):
            ax = axes[dim_idx]
            distribution = np.array(dim_data["distribution"])
            bin_values = self.bin_centers

            ax.bar(
                bin_values,
                distribution,
                width=(bin_values[1] - bin_values[0]) * 0.8,
                alpha=0.7,
                color="steelblue",
            )
            ax.axvline(
                x=dim_data["predicted_value"],
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Predicted: {dim_data['predicted_value']:.3f}",
            )
            ax.set_title(
                f"{dim_data['label']} | "
                f"entropy={dim_data['entropy']:.2f}, "
                f"max_prob={dim_data['max_prob']:.3f}"
            )
            ax.set_xlabel("Bin value")
            ax.set_ylabel("Probability")
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(
            os.path.join(step_dir, "action_distribution.png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _save_attention_maps(
        self,
        attn_data: Dict[str, Any],
        image_pil: Optional[Image.Image],
        step_dir: str,
    ):
        """Save attention heatmaps overlaid on input image."""
        layers = attn_data.get("layers", {})
        if not layers:
            return

        # Save raw attention maps as npz
        npz_data = {}
        for layer_name, attn_map in layers.items():
            npz_data[layer_name] = attn_map
        np.savez_compressed(
            os.path.join(step_dir, "attention_maps.npz"), **npz_data
        )

        # Create overlay visualization
        n_layers = len(layers)
        fig, axes = plt.subplots(1, n_layers + 1, figsize=(4 * (n_layers + 1), 4))
        if n_layers + 1 == 1:
            axes = [axes]

        # Show original image
        if image_pil is not None:
            axes[0].imshow(image_pil)
            axes[0].set_title("Input Image")
            axes[0].axis("off")

        # Show attention heatmaps overlaid
        for idx, (layer_name, attn_map) in enumerate(layers.items()):
            ax = axes[idx + 1]
            if image_pil is not None:
                # Resize attention map to image size
                attn_resized = np.array(
                    Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
                        image_pil.size, Image.BILINEAR
                    )
                ) / 255.0
                ax.imshow(image_pil)
                ax.imshow(attn_resized, cmap="jet", alpha=0.5)
            else:
                ax.imshow(attn_map, cmap="jet")
            ax.set_title(layer_name)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(step_dir, "attention_heatmap.png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _save_logit_lens(self, lens_data: Dict[str, Any], step_dir: str):
        """Save logit lens data as JSON and plot."""
        # Save JSON (without full distributions to keep file size small)
        lens_json = {
            "n_layers": lens_data["n_layers"],
            "layers": [
                {
                    "layer": ld["layer"],
                    "predicted_bin": ld["predicted_bin"],
                    "predicted_value": ld["predicted_value"],
                    "max_prob": ld["max_prob"],
                    "entropy": ld["entropy"],
                }
                for ld in lens_data["layers"]
            ],
        }
        with open(os.path.join(step_dir, "logit_lens.json"), "w") as f:
            json.dump(lens_json, f, indent=2)

        # Create heatmap of predicted values across layers
        n_layers = len(lens_data["layers"])
        predicted_values = [ld["predicted_value"] for ld in lens_data["layers"]]
        max_probs = [ld["max_prob"] for ld in lens_data["layers"]]
        entropies = [ld["entropy"] for ld in lens_data["layers"]]

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Plot predicted value evolution
        axes[0].plot(range(n_layers), predicted_values, "b-o", markersize=3)
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("Predicted Value")
        axes[0].set_title("Logit Lens: Predicted Action Value per Layer (1st action dim)")
        axes[0].grid(True, alpha=0.3)

        # Plot max probability evolution
        axes[1].plot(range(n_layers), max_probs, "r-o", markersize=3)
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Max Probability")
        axes[1].set_title("Confidence (Max Prob) per Layer")
        axes[1].grid(True, alpha=0.3)

        # Plot entropy evolution
        axes[2].plot(range(n_layers), entropies, "g-o", markersize=3)
        axes[2].set_xlabel("Layer")
        axes[2].set_ylabel("Entropy")
        axes[2].set_title("Entropy per Layer")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(step_dir, "logit_lens_plot.png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _save_gradcam(
        self,
        gradcam_data: Dict[str, Any],
        image_pil: Optional[Image.Image],
        step_dir: str,
    ):
        """Save GradCAM saliency map overlaid on input image."""
        if "cam_2d" not in gradcam_data:
            return

        cam_2d = gradcam_data["cam_2d"]

        # Save raw data
        np.savez_compressed(
            os.path.join(step_dir, "gradcam.npz"),
            cam_2d=cam_2d,
            raw_cam=gradcam_data.get("raw_cam", cam_2d.flatten()),
        )

        # Create overlay visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        if image_pil is not None:
            axes[0].imshow(image_pil)
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            # GradCAM heatmap
            axes[1].imshow(cam_2d, cmap="jet")
            axes[1].set_title("GradCAM Heatmap")
            axes[1].axis("off")

            # Overlay
            cam_resized = np.array(
                Image.fromarray((cam_2d * 255).astype(np.uint8)).resize(
                    image_pil.size, Image.BILINEAR
                )
            ) / 255.0
            axes[2].imshow(image_pil)
            axes[2].imshow(cam_resized, cmap="jet", alpha=0.5)
            axes[2].set_title("GradCAM Overlay")
            axes[2].axis("off")
        else:
            axes[0].imshow(cam_2d, cmap="jet")
            axes[0].set_title("GradCAM Heatmap")
            axes[0].axis("off")
            axes[1].axis("off")
            axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(step_dir, "gradcam_overlay.png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _save_projector_features(
        self, features: np.ndarray, step_dir: str
    ):
        """Save raw projected vision embeddings."""
        np.savez_compressed(
            os.path.join(step_dir, "projector_features.npz"),
            features=features,
        )
