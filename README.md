# Go VLA Benchmark

Benchmark for fine-tuning and evaluating Vision-Language-Action (VLA) models on a simulated 5x5 Go stone placement task using a Panda arm in robosuite/MuJoCo.

## Models & Results

Pre-trained checkpoints and evaluation videos are available on HuggingFace: [MJ22x/go-vla-benchmark](https://huggingface.co/datasets/MJ22x/go-vla-benchmark)

| Model | Base | Params | Fine-tuning | Train L1 | Engine Success |
|-------|------|--------|-------------|----------|---------------|
| **OpenVLA** | `openvla/openvla-7b` | 7B | QLoRA r=32 | 0.039 | 0% |
| **SpatialVLA** | `IPEC-COMMUNITY/spatialvla-4b-224-pt` | 4B | LoRA r=32 | — | 0% |
| **pi0** | `pi0_base` (PaliGemma 3B) | 3.3B | Full fine-tune | 0.083 | 0% |

## Loading Checkpoints

### OpenVLA (LoRA adapter)

```python
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

adapter_path = "path/to/openvla/best"  # from HF dataset
processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True)
base = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b",
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16))
vla = PeftModel.from_pretrained(base, adapter_path)

# Set norm stats on inner model
vla.base_model.model.norm_stats = json.load(open(f"{adapter_path}/dataset_statistics.json"))
vla.eval()

# Inference
inputs = processor(prompt, image)
inputs["pixel_values"] = inputs["pixel_values"].to("cuda", dtype=torch.bfloat16)
inputs["input_ids"] = inputs["input_ids"].to("cuda")
inputs["attention_mask"] = inputs["attention_mask"].to("cuda")
action = vla.predict_action(**inputs, unnorm_key="go_vla_dataset", do_sample=False)
# action is [dx, dy, dz, gripper] in raw action space
```

### SpatialVLA (LoRA adapter)

```python
from model import SpatialVLAConfig, SpatialVLAForConditionalGeneration, SpatialVLAProcessor
from peft import PeftModel

adapter_path = "path/to/spatialvla/best"
processor = SpatialVLAProcessor.from_pretrained(adapter_path, trust_remote_code=True)
base = SpatialVLAForConditionalGeneration.from_pretrained(
    "IPEC-COMMUNITY/spatialvla-4b-224-pt", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(base, adapter_path)
model = model.merge_and_unload().eval().cuda()

# Inference
inputs = processor(images=[image], text=prompt, return_tensors="pt")
generation_outputs = model.predict_action(inputs)
result = processor.decode_actions(generation_outputs, unnorm_key="go_vla_dataset/5.0.0")
action_7d = result["actions"][0]  # 7-DoF
action_4d = np.concatenate([action_7d[:3], action_7d[6:7]])  # [dx, dy, dz, gripper]
```

### pi0 (full checkpoint, JAX)

```python
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config

config = _config.get_config("pi0_go_vla")
policy = _policy_config.create_trained_policy(config, "path/to/pizero/best")

# Inference
obs = {
    "image": image,  # uint8 (256, 256, 3)
    "state": np.zeros(4, dtype=np.float32),
    "prompt": "What action should the robot take to place a black stone at row 2, column 3?",
}
result = policy.infer(obs)
action = result["actions"][0, :4]  # [dx, dy, dz, gripper]
```

## Environment Setup

```bash
conda create -n mujogo python=3.11 -y
conda activate mujogo
pip install torch torchvision torchaudio
pip install -e openvla --no-deps
pip install -r openvla/requirements-finetune.txt
pip install -e mimicgen --no-deps
```

## Data Generation

### Collect source demonstrations

```bash
MUJOCO_GL=glfw python benchmarks/go_vla_benchmark/scripts/collect_source_demos.py \
    --num-demos 100 --camera-size 256 \
    --environment-name robosuite_go_5x5_rigid_bodies
```

### Convert to RLDS

```bash
python benchmarks/go_vla_benchmark/scripts/convert_to_rlds.py \
    --input benchmarks/go_vla_benchmark/data/source_go.hdf5
```

## Training

### OpenVLA (QLoRA)

```bash
MUJOCO_GL=egl PYTHONPATH=openvla python openvla/vla-scripts/finetune.py \
    training.batch_size=4 training.grad_accumulation_steps=32 \
    lora.enabled=true lora.quantization=true lora.rank=32 \
    training.epochs=20 training.val_steps=25
```

### SpatialVLA (LoRA)

```bash
PYTHONPATH=spatialvla LAUNCHER=pytorch torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    spatialvla/train/spatialvla_finetune.py \
    --model_name_or_path IPEC-COMMUNITY/spatialvla-4b-224-pt \
    --data_root_dir ~/tensorflow_datasets --data_mix go_vla \
    --lora 32 --lora_alpha 32 --lora_target linear \
    --mask_rotation_loss True --action_forward_steps 3 \
    --flash_attn True --grad_checkpoint True \
    --output_dir outputs/spatialvla_lora \
    --bf16 True --tf32 True --num_train_epochs 50 \
    --per_device_train_batch_size 32 --gradient_accumulation_steps 8 \
    --save_strategy steps --save_steps 500 --save_total_limit 2 \
    --learning_rate 5e-4 --deepspeed spatialvla/scripts/zero1.json \
    --report_to wandb
```

### pi0 (full fine-tune, requires openpi)

```bash
cd openpi
uv run examples/go_vla/convert_go_vla_to_lerobot.py --data-dir ~/tensorflow_datasets
uv run scripts/compute_norm_stats.py pi0_go_vla
uv run scripts/train.py pi0_go_vla --exp-name go_vla
```

## Evaluation

### Engine rollout (compounding error)

```bash
MUJOCO_GL=egl python openvla/vla-scripts/eval_engine.py \
    --adapter-path outputs/<path>/checkpoints/latest \
    --num-episodes 20 --save-videos --load-4bit
```

### Train data eval (isolated frames, no compounding)

```bash
python openvla/vla-scripts/eval_traindata.py \
    --adapter-path outputs/<path>/checkpoints/latest \
    --split all --save-videos --load-4bit
```

### Self-play

```bash
MUJOCO_GL=egl python benchmarks/go_vla_benchmark/scripts/self_play.py \
    --model-path outputs/<path>/checkpoints/latest \
    --max-moves 5 --max-steps-per-move 200 --load-4bit \
    --player1 random --player2 random
```

### Scripted controller baseline

```bash
MUJOCO_GL=egl python benchmarks/go_vla_benchmark/scripts/self_play.py \
    --model-path outputs/<path>/checkpoints/latest \
    --max-moves 5 --force-trajectory --force-on-failure
```
