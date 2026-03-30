"""
vpt.py

Visual Prompt Tuning (VPT) module for OpenVLA's vision backbone.

Injects learnable prompt tokens into the frozen ViT featurizer within PrismaticVisionBackbone,
enabling parameter-efficient fine-tuning with 30-100x fewer parameters than LoRA.

Handles both single and fused vision backbones:
    - Fused (OpenVLA default): DINOv2 ViT-L (24 blocks, 1024-dim, 5 prefix tokens) + SigLIP
    - Single: SigLIP ViT-SO400M (27 blocks, 1152-dim, 0 prefix tokens)

References:
    - Jia et al., "Visual Prompt Tuning", ECCV 2022
"""

import json
import os
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn


@dataclass
class VPTConfig:
    """Configuration for Visual Prompt Tuning."""

    num_prompts: int = 50  # Number of learnable prompt tokens per layer
    mode: str = "shallow"  # "shallow" (single set of prompts) or "deep" (per-block prompts)
    init_std: float = 0.02  # Std for prompt initialization
    dropout: float = 0.0  # Dropout on prompt tokens

    def __post_init__(self):
        assert self.mode in ("shallow", "deep"), f"Invalid VPT mode: {self.mode}"
        assert self.num_prompts > 0, "num_prompts must be positive"

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "VPTConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            return cls(**json.load(f))


class VPTWrapper(nn.Module):
    """
    Wraps a TIMM ViT featurizer with Visual Prompt Tuning.

    Key design:
    - Uses object.__setattr__ to hold a reference to the featurizer WITHOUT registering
      it as a submodule, so state_dict() only contains VPT prompt parameters.
    - Monkey-patches featurizer.forward to inject VPT tokens into the ViT forward pass.
    - Replicates TIMM's _intermediate_layers + get_intermediate_layers logic with VPT injection.
    - Properly handles prefix tokens (CLS + registers for DINOv2, none for SigLIP).
    - Output shape is always [bsz, num_patches, embed_dim] with VPT and prefix tokens stripped.

    Args:
        vision_backbone: PrismaticVisionBackbone instance (we wrap its .featurizer)
        config: VPTConfig with prompt tuning hyperparameters
    """

    def __init__(self, vision_backbone: nn.Module, config: VPTConfig) -> None:
        super().__init__()

        # Store reference to featurizer WITHOUT registering as submodule
        # This ensures state_dict() only contains VPT parameters
        featurizer = vision_backbone.featurizer
        object.__setattr__(self, "_featurizer", featurizer)

        self.config = config
        self.embed_dim = featurizer.embed_dim
        self.num_blocks = len(featurizer.blocks)
        self.num_prompts = config.num_prompts
        # Number of prefix tokens (CLS + registers) to strip from output
        # DINOv2 ViT-L: 5 (1 CLS + 4 registers), SigLIP: 0
        self.num_prefix_tokens = getattr(featurizer, "num_prefix_tokens", 0)

        # Determine which block to collect output from (second-to-last, matching original forward)
        self._take_indices = {self.num_blocks - 2}

        # Create learnable VPT prompt tokens
        if config.mode == "shallow":
            # Single set of prompts prepended before block 0, flows through all blocks
            self.prompt_tokens = nn.Parameter(
                torch.zeros(1, config.num_prompts, self.embed_dim)
            )
            nn.init.normal_(self.prompt_tokens, std=config.init_std)
        else:  # deep
            # Per-block prompts: at each block, discard previous VPT outputs and inject fresh tokens
            self.prompt_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, config.num_prompts, self.embed_dim))
                for _ in range(self.num_blocks)
            ])
            for p in self.prompt_tokens:
                nn.init.normal_(p, std=config.init_std)

        # Optional dropout on prompt tokens
        self.prompt_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Monkey-patch the featurizer's forward to use our VPT forward
        # Store original forward for potential unwrapping
        object.__setattr__(self, "_original_forward", featurizer.forward)
        featurizer.forward = self._make_vpt_forward()

    def _make_vpt_forward(self):
        """Create a VPT-augmented forward function that replaces featurizer.forward."""
        wrapper = self

        def _vpt_forward(x: torch.Tensor) -> torch.Tensor:
            """VPT-augmented forward replicating TIMM _intermediate_layers with prompt injection."""
            featurizer = wrapper._featurizer
            outputs = []

            # Standard ViT preprocessing (matching TIMM _intermediate_layers lines 621-624)
            x = featurizer.patch_embed(x)
            x = featurizer._pos_embed(x)
            x = featurizer.patch_drop(x)
            x = featurizer.norm_pre(x)

            num_patches = x.shape[1]  # Should be 256 for SigLIP

            if wrapper.config.mode == "shallow":
                # Prepend prompt tokens before first block
                bsz = x.shape[0]
                prompts = wrapper.prompt_dropout(
                    wrapper.prompt_tokens.expand(bsz, -1, -1)
                )
                x = torch.cat([prompts, x], dim=1)  # [bsz, P + 256, 1152]

                # Run through all blocks
                for i, blk in enumerate(featurizer.blocks):
                    x = blk(x)
                    if i in wrapper._take_indices:
                        outputs.append(x)

            else:  # deep
                bsz = x.shape[0]

                for i, blk in enumerate(featurizer.blocks):
                    # Inject fresh prompts at each block
                    prompts = wrapper.prompt_dropout(
                        wrapper.prompt_tokens[i].expand(bsz, -1, -1)
                    )

                    if i == 0:
                        # First block: prepend prompts
                        x = torch.cat([prompts, x], dim=1)
                    else:
                        # Subsequent blocks: strip old prompts, prepend fresh ones
                        x = torch.cat([prompts, x[:, wrapper.num_prompts:, :]], dim=1)

                    x = blk(x)
                    if i in wrapper._take_indices:
                        outputs.append(x)

            # Strip VPT tokens from output
            result = outputs[0][:, wrapper.num_prompts:, :]
            # Strip prefix tokens (CLS + registers) to match get_intermediate_layers behavior
            # DINOv2: num_prefix_tokens=5, SigLIP: num_prefix_tokens=0
            result = result[:, wrapper.num_prefix_tokens:, :]  # [bsz, num_patches, embed_dim]
            return result

        return _vpt_forward

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass through VPT-augmented vision backbone.

        This is called when using the wrapper directly. In practice, the monkey-patched
        featurizer.forward handles the VPT logic transparently within the existing pipeline.
        """
        return self._featurizer(pixel_values)

    def unwrap(self) -> None:
        """Remove VPT wrapper, restoring original featurizer forward."""
        self._featurizer.forward = self._original_forward

    def save_weights(self, save_dir: str) -> None:
        """Save VPT weights and config to directory."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "vpt_weights.pt"))
        self.config.save(os.path.join(save_dir, "vpt_config.json"))

    @classmethod
    def load_weights(cls, wrapper: "VPTWrapper", load_dir: str, map_location: str = "cpu") -> "VPTWrapper":
        """Load VPT weights into an existing wrapper."""
        weights_path = os.path.join(load_dir, "vpt_weights.pt")
        state_dict = torch.load(weights_path, map_location=map_location)
        wrapper.load_state_dict(state_dict)
        return wrapper

    @classmethod
    def from_pretrained(cls, vision_backbone: nn.Module, load_dir: str, map_location: str = "cpu") -> "VPTWrapper":
        """Create VPTWrapper from saved config and weights."""
        config = VPTConfig.load(os.path.join(load_dir, "vpt_config.json"))
        wrapper = cls(vision_backbone, config)
        weights_path = os.path.join(load_dir, "vpt_weights.pt")
        state_dict = torch.load(weights_path, map_location=map_location)
        wrapper.load_state_dict(state_dict)
        return wrapper

    def get_num_trainable_params(self) -> int:
        """Return total number of trainable VPT parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        mode = self.config.mode
        n = self.config.num_prompts
        params = self.get_num_trainable_params()
        return (
            f"VPTWrapper(mode={mode}, num_prompts={n}, embed_dim={self.embed_dim}, "
            f"num_blocks={self.num_blocks}, trainable_params={params:,})"
        )
