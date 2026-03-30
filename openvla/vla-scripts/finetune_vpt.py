"""
finetune_vpt.py

Visual Prompt Tuning (VPT) fine-tuning script for OpenVLA on LIBERO.

Instead of LoRA (which adapts the LLM backbone with ~100M params), VPT injects learnable tokens
into the frozen ViT, keeping the entire 7B model frozen. This tests whether visual adaptation
alone can drive policy improvement with 30-100x fewer parameters.

Run with:
    python vla-scripts/finetune_vpt.py \
        --libero_data_dir LIBERO/libero/datasets/libero_object \
        --libero_task_suite libero_object \
        --vpt_mode deep --vpt_num_prompts 50 \
        --batch_size 1 --grad_accumulation_steps 16 \
        --load_in_8bit True
"""

import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import numpy as np
import torch
import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.libero_dataset import LIBERODataset
from prismatic.vla.vpt import VPTConfig, VPTWrapper

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class VPTFinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    libero_data_dir: str = "LIBERO/libero/datasets/libero_spatial"  # Path to LIBERO HDF5 demos
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    libero_task_suite: str = "libero_spatial"                        # LIBERO task suite name

    # VPT Parameters
    vpt_mode: str = "deep"                                          # VPT mode: "shallow" or "deep"
    vpt_num_prompts: int = 50                                       # Number of VPT prompt tokens
    vpt_init_std: float = 0.02                                      # Std for VPT prompt initialization
    vpt_dropout: float = 0.0                                        # Dropout on VPT prompt tokens

    # Fine-tuning Parameters
    batch_size: int = 1                                             # Fine-tuning batch size
    max_steps: int = -1                                             # Max gradient steps (-1 = use num_epochs instead)
    num_epochs: int = 1                                             # Number of epochs (ignored if max_steps > 0)
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 1e-3                                     # VPT learning rate (higher than LoRA's 5e-4)
    grad_accumulation_steps: int = 16                               # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    save_latest_checkpoint_only: bool = True                        # Whether to save only the latest checkpoint
    load_in_8bit: bool = True                                       # Load frozen model in 8-bit (saves ~7GB VRAM)
    load_in_4bit: bool = False                                      # Load frozen model in 4-bit (saves ~10GB VRAM)

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging
    use_wandb: bool = False                                         # Whether to log to W&B

    # fmt: on


@draccus.wrap()
def finetune_vpt(cfg: VPTFinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA `{cfg.vla_path}` with VPT on `{cfg.libero_task_suite}`")
    print(f"  VPT mode: {cfg.vpt_mode}, num_prompts: {cfg.vpt_num_prompts}")

    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()

    # CUDA performance settings for RTX 5080 / Ampere+ GPUs
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Configure Unique Experiment ID & Log Directory
    quant_tag = "8bit" if cfg.load_in_8bit else ("4bit" if cfg.load_in_4bit else "bf16")
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.libero_task_suite}"
        f"+vpt-{cfg.vpt_mode}-p{cfg.vpt_num_prompts}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}+{quant_tag}"
    )
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    run_dir = cfg.run_root_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Detect ECoT by pre-loading config (ECoT uses single SigLIP, not fused DINOv2+SigLIP)
    model_config = AutoConfig.from_pretrained(cfg.vla_path, trust_remote_code=True)
    is_ecot = getattr(model_config, "use_fused_vision_backbone", True) is False

    if not is_ecot:
        # Base OpenVLA: register local classes
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    # ECoT: trust_remote_code loads ECoT's own classes

    # Load OpenVLA Processor and Model
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
    )
    # Only call .to() if not using quantization (bitsandbytes handles placement)
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(device)

    # Freeze ALL model parameters
    vla.requires_grad_(False)

    # Create VPT wrapper -- injects learnable prompts into the ViT
    vpt_config = VPTConfig(
        num_prompts=cfg.vpt_num_prompts,
        mode=cfg.vpt_mode,
        init_std=cfg.vpt_init_std,
        dropout=cfg.vpt_dropout,
    )
    vpt_wrapper = VPTWrapper(vla.vision_backbone, vpt_config)
    vpt_wrapper = vpt_wrapper.to(device)

    # torch.compile: only compile when not using quantization (bnb 8/4-bit ops are
    # not supported by torch.compile's inductor/triton backend)
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        print("  Compiling vision_backbone and projector with mode='reduce-overhead'")
        vla.vision_backbone = torch.compile(vla.vision_backbone, mode="reduce-overhead")
        vla.projector = torch.compile(vla.projector, mode="reduce-overhead")
    else:
        print("  Skipping torch.compile (incompatible with bitsandbytes quantization)")

    # Print parameter counts
    total_params = sum(p.numel() for p in vla.parameters())
    trainable_params = vpt_wrapper.get_num_trainable_params()
    print(f"  Total model parameters: {total_params:,}")
    print(f"  VPT trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.4f}%)")
    print(f"  VPT wrapper: {vpt_wrapper}")

    # Create Optimizer -- only optimize VPT parameters
    trainable_params_list = [param for param in vpt_wrapper.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params_list, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load LIBERO Dataset from HDF5 files
    vla_dataset = LIBERODataset(
        data_dir=cfg.libero_data_dir,
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        task_suite_name=cfg.libero_task_suite,
    )

    # Save Dataset Statistics (used to de-normalize actions for inference)
    stats_path = run_dir / "dataset_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(vla_dataset.dataset_statistics, f, indent=2)
    print(f"  Saved dataset statistics to {stats_path}")

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Initialize Logging =>> W&B
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"vpt+{exp_id}")

    # Deque to store recent train metrics (for smoothened logging with gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Compute effective max_steps and num_epochs
    steps_per_epoch = len(dataloader) // cfg.grad_accumulation_steps
    if cfg.max_steps > 0:
        max_steps = cfg.max_steps
        num_epochs = (max_steps // steps_per_epoch) + 1  # enough epochs to reach max_steps
    else:
        num_epochs = cfg.num_epochs
        max_steps = steps_per_epoch * num_epochs
    print(f"  Dataset size: {len(vla_dataset)}, steps/epoch: {steps_per_epoch}, max_steps: {max_steps}, num_epochs: {num_epochs}")

    # Train!
    global_step = 0
    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches

    with tqdm.tqdm(total=max_steps, leave=False) as progress:
        vla.train()
        vla.requires_grad_(False)

        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(dataloader):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device, non_blocking=True),
                        attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                        pixel_values=batch["pixel_values"].to(device=device, dtype=torch.bfloat16, non_blocking=True),
                        labels=batch["labels"].to(device, non_blocking=True),
                    )
                    loss = output.loss

                # Normalize loss for gradient accumulation
                normalized_loss = loss / cfg.grad_accumulation_steps
                normalized_loss.backward()

                # Compute Accuracy (lightweight GPU ops, no sync)
                action_logits = output.logits[:, num_patches : -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)

                recent_losses.append(loss.item())
                recent_action_accuracies.append(action_accuracy.item())

                # Compute L1 Loss only at logging boundaries (expensive: GPU->CPU sync + numpy decode)
                is_logging_step = (global_step + 1) % cfg.grad_accumulation_steps == 0
                if is_logging_step:
                    if mask.sum() > 0:
                        continuous_actions_pred = torch.tensor(
                            action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                        )
                        continuous_actions_gt = torch.tensor(
                            action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                        )
                        action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                    else:
                        action_l1_loss = torch.tensor(0.0)
                    recent_l1_losses.append(action_l1_loss.item())

                # Compute gradient step index
                gradient_step_idx = global_step // cfg.grad_accumulation_steps

                # Optimizer Step
                if (global_step + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    progress.update()

                    # Compute smoothened train metrics
                    smoothened_loss = sum(recent_losses) / len(recent_losses)
                    smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
                    smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

                    # Push Metrics to W&B (every 10 gradient steps)
                    if cfg.use_wandb and gradient_step_idx % 10 == 0:
                        wandb.log(
                            {
                                "train_loss": smoothened_loss,
                                "action_accuracy": smoothened_action_accuracy,
                                "l1_loss": smoothened_l1_loss,
                                "epoch": epoch,
                            },
                            step=gradient_step_idx,
                        )

                    progress.set_postfix(
                        loss=f"{smoothened_loss:.4f}",
                        acc=f"{smoothened_action_accuracy:.3f}",
                        l1=f"{smoothened_l1_loss:.4f}",
                    )

                # Save VPT Checkpoint
                if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0 and (global_step + 1) % cfg.grad_accumulation_steps == 0:
                    print(f"\nSaving VPT Checkpoint for Step {gradient_step_idx}")

                    if cfg.save_latest_checkpoint_only:
                        save_dir = run_dir / "vpt_checkpoint"
                    else:
                        save_dir = run_dir / f"vpt_checkpoint-{gradient_step_idx}"

                    vpt_wrapper.save_weights(str(save_dir))
                    processor.save_pretrained(run_dir)
                    print(f"  Saved VPT weights to {save_dir}")

                global_step += 1

                # Stop training when max_steps is reached
                if gradient_step_idx >= max_steps:
                    print(f"Max step {max_steps} reached! Stopping training...")
                    save_dir = run_dir / "vpt_checkpoint"
                    vpt_wrapper.save_weights(str(save_dir))
                    processor.save_pretrained(run_dir)
                    print(f"  Saved final VPT weights to {save_dir}")
                    return

            print(f"Epoch {epoch + 1}/{cfg.num_epochs} complete")

    # Final save
    save_dir = run_dir / "vpt_checkpoint"
    vpt_wrapper.save_weights(str(save_dir))
    processor.save_pretrained(run_dir)
    print(f"Training complete! Saved final VPT weights to {save_dir}")


if __name__ == "__main__":
    finetune_vpt()
