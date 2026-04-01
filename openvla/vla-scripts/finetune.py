"""
finetune.py

Parameter-efficient fine-tuning of OpenVLA models using Hydra for configuration and W&B for logging.

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    training.batch_size=8 \
                                    training.learning_rate=2e-5 \
                                    lora.enabled=false \
                                    wandb.project=my_project
"""

import os
from collections import deque
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Optional engine imports for rollout evaluation (not required for training)
try:
    import sys as _sys
    from pathlib import Path as _Path

    _REPO_ROOT = _Path(__file__).resolve().parents[2]
    _sys.path.insert(0, str(_REPO_ROOT / "benchmarks" / "go_vla_benchmark"))
    from go_vla_benchmark.paths import bootstrap_pythonpath

    bootstrap_pythonpath(_REPO_ROOT)
    from go_vla_benchmark.openvla_action_utils import standardized_action_to_benchmark
    from go_vla_benchmark.robosuite_compat import ensure_robosuite_compat

    ensure_robosuite_compat()
    from go_vla_benchmark.common import GoResetOptions
    from go_vla_benchmark.env_factory import create_benchmark_env

    _HAS_ENGINE = True
except Exception:
    _HAS_ENGINE = False

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@torch.no_grad()
def run_validation(vla, val_dataloader, action_tokenizer, unwrapped_vla, device_id, max_batches=20):
    """Run validation on held-out split and return metrics."""
    vla.eval()
    val_losses, val_accs, val_l1s = [], [], []
    for i, batch in enumerate(val_dataloader):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
            )
        val_losses.append(output.loss.item())

        action_logits = output.logits[:, unwrapped_vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
        action_preds = action_logits.argmax(dim=2)
        action_gt = batch["labels"][:, 1:].to(action_preds.device)
        mask = action_gt > action_tokenizer.action_token_begin_idx

        if mask.sum() > 0:
            correct_preds = (action_preds == action_gt) & mask
            val_accs.append((correct_preds.sum().float() / mask.sum().float()).item())
            continuous_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            val_l1s.append(torch.nn.functional.l1_loss(continuous_pred, continuous_gt).item())

    vla.train()
    n = max(len(val_losses), 1)
    return {
        "val_loss": sum(val_losses) / n,
        "val_accuracy": sum(val_accs) / max(len(val_accs), 1),
        "val_l1_loss": sum(val_l1s) / max(len(val_l1s), 1),
    }


def _save_video(frames, path, fps=30):
    """Save frames as MP4 video using imageio."""
    try:
        import imageio
    except ImportError:
        print(f"  [skip video] pip install imageio[ffmpeg] to save videos")
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", pixelformat="yuv420p")
    for frame in frames:
        writer.append_data(frame)
    writer.close()


# Cached engine env (created once, reused across validation steps)
_engine_env = None


@torch.no_grad()
def run_engine_rollout(vla, processor, device_id, output_dir, step, use_wandb, dataset_statistics=None, unnorm_key="go_vla_dataset", seed=None, tag="rollout"):
    """Run 1 episode in the actual physics engine and save video."""
    if not _HAS_ENGINE:
        return None

    import numpy as np
    from PIL import Image

    global _engine_env
    if _engine_env is None:
        _engine_env = create_benchmark_env(
            environment_name="robosuite_go_5x5_rigid_bodies",
            seed=42,
            include_image_obs=True,
            camera_height=256,
            camera_width=256,
            action_scale=0.03,
            drive_physical_arm=True,
            render_carried_stone=True,
            render_eef_overlay=True,
        )

    env = _engine_env

    _INSTRUCTION_TEMPLATES = [
        "Place a {color} stone on the Go board at row {r}, column {c}.",
        "Put a {color} stone at position ({r}, {c}) on the Go board.",
        "Move the {color} stone to row {r}, column {c} on the board.",
        "Set a {color} stone at ({r}, {c}).",
    ]

    rng = np.random.RandomState(seed if seed is not None else step)
    opening_moves = int(rng.randint(0, 5))
    env.queue_reset_options(GoResetOptions(opening_moves=opening_moves))
    obs = env.reset()

    r, c = env._target_rc
    color = env._stone_color
    template = _INSTRUCTION_TEMPLATES[rng.randint(len(_INSTRUCTION_TEMPLATES))]
    instruction = template.format(color=color, r=r, c=c)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    # Ensure norm_stats are set so predict_action can denormalize.
    # With PEFT wrappers, predict_action uses self.norm_stats on the inner
    # OpenVLAForActionPrediction model. Walk the wrapper chain to find it.
    if dataset_statistics is not None:
        target = vla
        # PeftModel.get_base_model() -> LoraModel; LoraModel.model -> OpenVLA
        if hasattr(target, "get_base_model"):
            target = target.get_base_model()
        if hasattr(target, "model"):
            target = target.model
        target.norm_stats = {**getattr(target, "norm_stats", {}), **dataset_statistics}

    vla.eval()

    # Disable gradient checkpointing for generate() compatibility, restore after
    had_grad_ckpt = getattr(vla.config, "use_cache", None) is not None
    if hasattr(vla, "gradient_checkpointing_disable"):
        vla.gradient_checkpointing_disable()
        vla.config.use_cache = True

    frames = []
    max_steps = 200
    success = False

    for step_idx in range(max_steps):
        image = obs.get("agentview_image", obs.get("image"))
        if image is None:
            break
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        frames.append(image.copy())

        pil_image = Image.fromarray(image).convert("RGB")
        w, h = pil_image.size
        crop_size = int(min(w, h) * 0.9)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        pil_image = pil_image.crop((left, top, left + crop_size, top + crop_size))
        inputs = processor(prompt, pil_image).to(device_id, dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

        action = standardized_action_to_benchmark(action, binarize=True)
        obs, reward, done, info = env.step(action)
        success_info = env.is_success()
        if success_info.get("task", False):
            success = True
            break

    # Restore gradient checkpointing for training
    if hasattr(vla, "gradient_checkpointing_enable"):
        vla.config.use_cache = False
        vla.gradient_checkpointing_enable()

    vla.train()

    # Save video
    video_path = Path(output_dir) / f"{tag}_step_{step}.mp4"
    if frames:
        _save_video(frames, video_path)

    result = {f"{tag}_success": int(success), f"{tag}_steps": step_idx + 1}

    if use_wandb:
        log_data = dict(result)
        if frames:
            try:
                log_data[f"{tag}_video"] = wandb.Video(str(video_path), format="mp4")
            except Exception:
                pass
        wandb.log(log_data, step=step)

    tqdm.tqdm.write(
        f"  >> {tag}: success={success}, steps={step_idx + 1}"
    )
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="finetune")
def finetune(cfg: DictConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.data.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Output dir from Hydra
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir = output_dir / "checkpoints"
    os.makedirs(run_dir, exist_ok=True)

    # Configure Unique Experiment ID
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.data.dataset_name}"
        f"+b{cfg.training.batch_size * cfg.training.grad_accumulation_steps}"
        f"+lr-{cfg.training.learning_rate}"
    )
    if cfg.lora.enabled:
        exp_id += f"+lora-r{cfg.lora.rank}+dropout-{cfg.lora.dropout}"
    if cfg.lora.quantization:
        exp_id += "+q-4bit"
    if cfg.data.image_aug:
        exp_id += "--image_aug"

    # Resolve optional checkpoint init paths
    resume_path = Path(cfg.resume_from).expanduser() if cfg.get("resume_from") else None
    warm_start_path = Path(cfg.warm_start_from).expanduser() if cfg.get("warm_start_from") else None
    if resume_path is not None and warm_start_path is not None:
        raise ValueError(
            "Set only one of `resume_from` or `warm_start_from`. "
            "`resume_from` restores optimizer state; `warm_start_from` starts a fresh run from saved weights."
        )
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint directory does not exist: {resume_path}")
    if warm_start_path is not None and not warm_start_path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint directory does not exist: {warm_start_path}")

    model_path = cfg.vla_path
    processor_path = cfg.vla_path
    adapter_init_path = None
    if resume_path is not None:
        if cfg.lora.enabled:
            adapter_init_path = resume_path / "adapter"
            if not adapter_init_path.exists():
                raise FileNotFoundError(
                    f"Expected LoRA adapter directory at `{adapter_init_path}` for `resume_from`."
                )
        else:
            model_path = str(resume_path / "model")
            processor_path = model_path
    elif warm_start_path is not None:
        if cfg.lora.enabled:
            adapter_init_path = warm_start_path
            adapter_weights_exist = any((warm_start_path / name).exists() for name in ("adapter_model.safetensors", "adapter_model.bin"))
            if not (warm_start_path / "adapter_config.json").exists() or not adapter_weights_exist:
                raise FileNotFoundError(
                    f"`warm_start_from` must point to a saved LoRA adapter directory such as "
                    f"`checkpoints/latest` or `checkpoints/best`: {warm_start_path}"
                )
        else:
            model_path = str(warm_start_path)
        if (warm_start_path / "preprocessor_config.json").exists():
            processor_path = str(warm_start_path)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.lora.quantization:
        assert cfg.lora.enabled, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.lora.quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # Enable gradient checkpointing to reduce activation memory
    if cfg.training.get("gradient_checkpointing", True):
        vla.config.use_cache = False
        vla.gradient_checkpointing_enable()

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.lora.enabled:
        if adapter_init_path is not None:
            vla = PeftModel.from_pretrained(vla, str(adapter_init_path), is_trainable=True)
            print(f"Loaded trainable LoRA weights from `{adapter_init_path}`")
        else:
            lora_config = LoraConfig(
                r=cfg.lora.rank,
                lora_alpha=min(cfg.lora.rank, 16),
                lora_dropout=cfg.lora.dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training (skip for single-GPU)
    use_ddp = dist.is_initialized() and dist.get_world_size() > 1
    if use_ddp:
        vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Unwrapped model reference (works with or without DDP)
    unwrapped_vla = vla.module if use_ddp else vla

    # Create Optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.training.learning_rate)

    # Resume from checkpoint if specified
    resume_step = 0
    best_val_loss = float("inf")
    if resume_path is not None:
        ckpt = torch.load(resume_path / "training_state.pt", map_location=f"cuda:{device_id}")
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        resume_step = ckpt.get("completed_steps")
        if resume_step is None:
            # Backward-compatibility for older checkpoints that stored the
            # zero-based step label plus the final micro-batch index.
            resume_batch_idx = ckpt.get("batch_idx")
            if resume_batch_idx is not None:
                resume_step = (resume_batch_idx + 1) // cfg.training.grad_accumulation_steps
            else:
                resume_step = ckpt.get("gradient_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from completed step {resume_step} (best_val_loss={best_val_loss:.4f})")

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset (train split)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        Path(cfg.data.root_dir).expanduser(),
        cfg.data.dataset_name,
        batch_transform,
        resize_resolution=tuple(unwrapped_vla.config.image_sizes),
        shuffle_buffer_size=cfg.data.shuffle_buffer_size,
        image_aug=cfg.data.image_aug,
    )

    # Load Validation Dataset (explicit `val` split when present; otherwise a held-out fallback split)
    val_batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    val_dataset = RLDSDataset(
        Path(cfg.data.root_dir).expanduser(),
        cfg.data.dataset_name,
        val_batch_transform,
        resize_resolution=tuple(unwrapped_vla.config.image_sizes),
        shuffle_buffer_size=1000,
        image_aug=False,
        train=False,
    )

    # Compute max_steps from epochs and flattened train transitions
    steps_per_epoch = len(vla_dataset) // (cfg.training.batch_size * cfg.training.grad_accumulation_steps)
    max_steps = cfg.training.epochs * steps_per_epoch
    print(
        f"  {cfg.training.epochs} epochs x {steps_per_epoch} optimizer steps/epoch "
        f"({len(vla_dataset)} train transitions) = {max_steps} total optimizer steps"
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoaders
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.training.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    # Validation frequency (in completed optimizer steps)
    val_steps = cfg.training.get("val_steps", 250)
    resume_checkpoint_steps = cfg.training.get("resume_checkpoint_steps", 1000)
    resume_dir = output_dir / "resume"
    os.makedirs(resume_dir, exist_ok=True)

    # Best model tracking
    latest_checkpoint_dir = run_dir / "latest"
    os.makedirs(latest_checkpoint_dir, exist_ok=True)
    best_checkpoint_dir = run_dir / "best"
    os.makedirs(best_checkpoint_dir, exist_ok=True)

    def _save_model_checkpoint(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        unwrapped_vla.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        save_dataset_statistics(vla_dataset.dataset_statistics, save_dir)

    def _save_resume_checkpoint(step, batch_idx):
        tqdm.tqdm.write(f"[Step {step}] Saving resumable checkpoint to {resume_dir}")
        if cfg.lora.enabled:
            unwrapped_vla.save_pretrained(resume_dir / "adapter")
        else:
            unwrapped_vla.save_pretrained(resume_dir / "model")
        torch.save(
            {
                "completed_steps": step,
                "gradient_step": step,
                "batch_idx": batch_idx,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            resume_dir / "training_state.pt",
        )

    # Initialize Logging =>> W&B
    use_wandb = False
    if distributed_state.is_main_process:
        try:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.run_name or f"ft+{exp_id}",
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=str(output_dir),
            )
            use_wandb = True
        except Exception as e:
            print(f"[W&B] Failed to initialize: {e}. Continuing without W&B.")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.training.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.training.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.training.grad_accumulation_steps)

    # Initial validation before training starts for fresh runs and warm starts
    if distributed_state.is_main_process and resume_step == 0:
        val_metrics = run_validation(vla, val_dataloader, action_tokenizer, unwrapped_vla, device_id)
        print(
            f"[Step 0] val_loss={val_metrics['val_loss']:.4f}  "
            f"val_acc={val_metrics['val_accuracy']:.3f}  "
            f"val_l1={val_metrics['val_l1_loss']:.4f}"
        )
        if use_wandb:
            wandb.log(val_metrics, step=0)
        # Save initial checkpoint as both best and latest
        best_val_loss = val_metrics["val_loss"]
        print(f"  >> Initial val_loss={best_val_loss:.4f}, saving best & latest")
        for save_dir in [best_checkpoint_dir, latest_checkpoint_dir]:
            _save_model_checkpoint(save_dir)

    if resume_step >= max_steps:
        print(
            f"Resume step {resume_step} already reaches max_steps={max_steps}; "
            "saving resumable/latest checkpoints and exiting."
        )
        if distributed_state.is_main_process:
            _save_model_checkpoint(latest_checkpoint_dir)
            last_batch_idx = (resume_step * cfg.training.grad_accumulation_steps) - 1
            _save_resume_checkpoint(resume_step, last_batch_idx)
        return

    # Train!
    completed_steps = resume_step
    last_batch_idx = None
    reached_max_steps = False
    with tqdm.tqdm(total=max_steps, initial=resume_step, leave=False, desc="Training") as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            # Skip batches if resuming
            step_group_idx = batch_idx // cfg.training.grad_accumulation_steps
            if step_group_idx < resume_step:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.training.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, unwrapped_vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute smoothened train metrics
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # Update progress bar with live metrics
            progress.set_postfix(
                loss=f"{smoothened_loss:.4f}",
                acc=f"{smoothened_action_accuracy:.3f}",
                l1=f"{smoothened_l1_loss:.4f}",
            )

            # Optimizer Step
            if (batch_idx + 1) % cfg.training.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                completed_steps = step_group_idx + 1
                last_batch_idx = batch_idx
                progress.update()

                # Push Metrics to W&B (every 10 completed optimizer steps)
                if distributed_state.is_main_process and use_wandb and completed_steps % 10 == 0:
                    wandb.log(
                        {
                            "train_loss": smoothened_loss,
                            "action_accuracy": smoothened_action_accuracy,
                            "l1_loss": smoothened_l1_loss,
                        },
                        step=completed_steps,
                    )

                # Run Validation periodically
                if (
                    completed_steps > 0
                    and completed_steps % val_steps == 0
                    and distributed_state.is_main_process
                ):
                    val_metrics = run_validation(
                        vla, val_dataloader, action_tokenizer, unwrapped_vla, device_id
                    )
                    tqdm.tqdm.write(
                        f"[Step {completed_steps}] "
                        f"val_loss={val_metrics['val_loss']:.4f}  "
                        f"val_acc={val_metrics['val_accuracy']:.3f}  "
                        f"val_l1={val_metrics['val_l1_loss']:.4f}"
                    )
                    if use_wandb:
                        wandb.log(val_metrics, step=completed_steps)

                    # Always save latest checkpoint at validation time
                    tqdm.tqdm.write(
                        f"  >> Saving latest checkpoint (step {completed_steps}) to {latest_checkpoint_dir}"
                    )
                    _save_model_checkpoint(latest_checkpoint_dir)

                    # Save best model if validation loss improved
                    if val_metrics["val_loss"] < best_val_loss:
                        best_val_loss = val_metrics["val_loss"]
                        tqdm.tqdm.write(
                            f"  >> New best val_loss={best_val_loss:.4f}, saving to {best_checkpoint_dir}"
                        )
                        _save_model_checkpoint(best_checkpoint_dir)

                    # Engine rollout evaluation
                    if _HAS_ENGINE:
                        # Fixed-seed rollout (same board state every time = train sample)
                        run_engine_rollout(
                            unwrapped_vla, processor, device_id, output_dir,
                            completed_steps, use_wandb,
                            dataset_statistics=vla_dataset.dataset_statistics,
                            seed=0, tag="train_rollout",
                        )
                        # Random rollout (different board state each val step)
                        run_engine_rollout(
                            unwrapped_vla, processor, device_id, output_dir,
                            completed_steps, use_wandb,
                            dataset_statistics=vla_dataset.dataset_statistics,
                            tag="rollout",
                        )

                # Save resumable training checkpoint every N completed optimizer steps
                if (
                    completed_steps > 0
                    and completed_steps % resume_checkpoint_steps == 0
                    and distributed_state.is_main_process
                ):
                    _save_resume_checkpoint(completed_steps, batch_idx)

                # Stop training when max_steps is reached
                if completed_steps >= max_steps:
                    print(f"Max step {max_steps} reached! Stopping training...")
                    reached_max_steps = True
                    break

    if reached_max_steps and distributed_state.is_main_process:
        tqdm.tqdm.write(f"[Step {completed_steps}] Saving final latest checkpoint to {latest_checkpoint_dir}")
        _save_model_checkpoint(latest_checkpoint_dir)
        _save_resume_checkpoint(completed_steps, last_batch_idx)


if __name__ == "__main__":
    finetune()
