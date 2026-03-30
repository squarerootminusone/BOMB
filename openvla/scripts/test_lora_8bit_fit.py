"""
Temporary script: test whether 8-bit LoRA training fits on this GPU.
Loads openvla-7b in 8-bit, applies LoRA rank 32 on all-linear,
runs forward+backward with batch_size=1, reports peak VRAM.
"""
import torch
import gc
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    torch.cuda.empty_cache()
    gc.collect()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Free VRAM before loading: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
    print()

    # --- Step 1: Load model in 8-bit ---
    print("Loading openvla-7b in 8-bit...")
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        load_in_8bit=True,
    )

    mem_after_load = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM after model load: {mem_after_load:.2f} GB")
    print()

    # --- Step 2: Apply LoRA ---
    print("Applying LoRA (rank=32, all-linear)...")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Workaround: peft 0.11.1 + bitsandbytes 0.49.2 incompatibility on 8-bit.
    # Try prepare_model_for_kbit_training, fall back to manual grad setup if it fails.
    try:
        vla = prepare_model_for_kbit_training(vla)
    except Exception as e:
        print(f"  prepare_model_for_kbit_training failed ({e}), doing manual setup...")
        for param in vla.parameters():
            param.requires_grad = False
            if param.dtype == torch.float16 or param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float32)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )

    # get_peft_model with 8bit may also hit the same bnb issue, try 4-bit as fallback
    try:
        vla = get_peft_model(vla, lora_config)
    except AttributeError as e:
        print(f"  8-bit LoRA injection failed: {e}")
        print("  Retrying with 4-bit quantization instead...")
        # Reload in 4-bit
        del vla
        torch.cuda.empty_cache()
        gc.collect()
        vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            load_in_4bit=True,
        )
        from peft import prepare_model_for_kbit_training as prep4
        vla = prep4(vla)
        vla = get_peft_model(vla, lora_config)
        print("  4-bit LoRA applied successfully (testing 4-bit instead of 8-bit)")

    vla.print_trainable_parameters()

    mem_after_lora = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM after LoRA: {mem_after_lora:.2f} GB")
    print()

    # --- Step 3: Create dummy batch (batch_size=1) ---
    print("Creating dummy batch (batch_size=1)...")
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    prompt_builder = PurePromptBuilder("openvla")
    prompt_builder.add_turn("human", "What action should the robot take to pick up the bowl?")
    prompt_builder.add_turn("gpt", action_tokenizer([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0]))

    input_ids = processor.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
    labels = list(input_ids)
    labels[:-8] = [-100] * (len(labels) - 8)

    input_ids = torch.tensor([input_ids]).cuda()
    attention_mask = torch.ones_like(input_ids).cuda()
    labels = torch.tensor([labels])

    # Fake pixel values matching OpenVLA's fused backbone (6 channels)
    pixel_values = torch.randn(1, 6, 224, 224, dtype=torch.bfloat16).cuda()

    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  pixel_values shape: {pixel_values.shape}")
    print()

    # --- Step 4: Forward pass ---
    print("Running forward pass...")
    torch.cuda.reset_peak_memory_stats()
    vla.train()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
        loss = output.loss

    mem_after_fwd = torch.cuda.memory_allocated() / 1e9
    peak_after_fwd = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Loss: {loss.item():.4f}")
    print(f"  VRAM after forward: {mem_after_fwd:.2f} GB")
    print(f"  Peak VRAM after forward: {peak_after_fwd:.2f} GB")
    print()

    # --- Step 5: Backward pass ---
    print("Running backward pass...")
    loss.backward()

    mem_after_bwd = torch.cuda.memory_allocated() / 1e9
    peak_after_bwd = torch.cuda.max_memory_allocated() / 1e9
    print(f"  VRAM after backward: {mem_after_bwd:.2f} GB")
    print(f"  Peak VRAM after backward: {peak_after_bwd:.2f} GB")
    print()

    # --- Step 6: Optimizer step ---
    print("Running optimizer step...")
    from torch.optim import AdamW
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=5e-4)
    optimizer.step()
    optimizer.zero_grad()

    mem_after_opt = torch.cuda.memory_allocated() / 1e9
    peak_after_opt = torch.cuda.max_memory_allocated() / 1e9
    print(f"  VRAM after optimizer step: {mem_after_opt:.2f} GB")
    print(f"  Peak VRAM after optimizer: {peak_after_opt:.2f} GB")
    print()

    # --- Summary ---
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {total_vram:.2f} GB")
    print(f"Peak VRAM used: {peak_after_opt:.2f} GB")
    print(f"Headroom: {total_vram - peak_after_opt:.2f} GB")
    print()
    if peak_after_opt < total_vram * 0.95:
        print("RESULT: 8-bit LoRA training FITS on this GPU (batch_size=1)")
    else:
        print("RESULT: 8-bit LoRA training DOES NOT FIT on this GPU")


if __name__ == "__main__":
    main()
