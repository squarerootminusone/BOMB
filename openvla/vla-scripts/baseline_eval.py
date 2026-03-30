"""
baseline_eval.py

Evaluate the frozen base VLA (no VPT prompts) on a LIBERO dataset.
Reports loss, action accuracy, and L1 over one full epoch.

Usage:
    python vla-scripts/baseline_eval.py \
        --vla_path openvla/openvla-7b \
        --libero_data_dir ../LIBERO/libero/datasets/libero_spatial \
        --libero_task_suite libero_spatial \
        --load_in_8bit True
"""

import os
from dataclasses import dataclass
from typing import Optional

import draccus
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.libero_dataset import LIBERODataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class BaselineEvalConfig:
    vla_path: str = "openvla/openvla-7b"
    libero_data_dir: str = "LIBERO/libero/datasets/libero_spatial"
    libero_task_suite: str = "libero_spatial"
    batch_size: int = 1
    load_in_8bit: bool = True
    load_in_4bit: bool = False
    num_batches: Optional[int] = None  # None = full epoch


@draccus.wrap()
def baseline_eval(cfg: BaselineEvalConfig) -> None:
    print(f"Evaluating frozen base VLA `{cfg.vla_path}` on `{cfg.libero_task_suite}`")

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()

    # Register & load model
    model_config = AutoConfig.from_pretrained(cfg.vla_path, trust_remote_code=True)
    is_ecot = getattr(model_config, "use_fused_vision_backbone", True) is False
    if not is_ecot:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
    )
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(device)

    vla.eval()

    # Dataset
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    vla_dataset = LIBERODataset(
        data_dir=cfg.libero_data_dir,
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        task_suite_name=cfg.libero_task_suite,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collator, num_workers=4, pin_memory=True,
    )

    num_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
    total_batches = cfg.num_batches or len(dataloader)

    all_losses, all_accs, all_l1s = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, total=total_batches)):
            if cfg.num_batches and batch_idx >= cfg.num_batches:
                break

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                    labels=batch["labels"],
                )

            action_logits = output.logits[:, num_patches:-1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            acc = (((action_preds == action_gt) & mask).sum().float() / mask.sum().float()).item() if mask.sum() > 0 else 0.0

            if mask.sum() > 0:
                pred_actions = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()))
                gt_actions = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()))
                l1 = torch.nn.functional.l1_loss(pred_actions, gt_actions).item()
            else:
                l1 = 0.0

            all_losses.append(output.loss.item())
            all_accs.append(acc)
            all_l1s.append(l1)

    print(f"\n{'='*50}")
    print(f"Baseline (no VPT) results over {len(all_losses)} batches:")
    print(f"  loss = {np.mean(all_losses):.4f}")
    print(f"  acc  = {np.mean(all_accs):.4f}")
    print(f"  l1   = {np.mean(all_l1s):.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    baseline_eval()
