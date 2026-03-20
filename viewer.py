"""napari RLDS dataset viewer for Go VLA episodes with model inference.

Usage:
    MUJOCO_GL=egl conda run --no-capture-output -n mujogo python viewer.py

Controls:
    N/P     = next/prev episode
    L       = load LoRA checkpoint (file dialog)
    R       = run inference on current episode (side-by-side)
    Escape  = return to GT-only view
"""

import json
import sys
import threading
import numpy as np
from pathlib import Path
from PIL import Image

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_PROJ_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJ_ROOT / "benchmarks" / "go_vla_benchmark" / "rlds_builder"))
sys.path.insert(0, str(_PROJ_ROOT / "openvla"))

from go_vla_dataset.go_vla_dataset_dataset_builder import Builder

_UNNORM_KEY = "go_vla_dataset"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_episodes(data_dir=os.path.expanduser("~/tensorflow_datasets"), split="train"):
    """Load episodes from the RLDS dataset.

    Args:
        data_dir: TFDS data directory.
        split: TFDS split name ("train", "val", or "all" for both).
    """
    builder = Builder(data_dir=data_dir)
    if split == "all":
        splits_to_load = ["train", "val"] if "val" in builder.info.splits else ["train"]
    else:
        splits_to_load = [split]
    episodes = []
    for s in splits_to_load:
        ds = builder.as_dataset(split=s)
        for ep in ds:
            images, actions, instruction = [], [], None
            for step in ep["steps"]:
                images.append(step["observation"]["image"].numpy())
                actions.append(step["action"].numpy())
                if instruction is None:
                    instruction = step["language_instruction"].numpy().decode("utf-8")
            episodes.append({
                "images": np.stack(images),
                "actions": np.stack(actions),
                "instruction": instruction or "",
                "split": s,
            })
    return episodes


# ---------------------------------------------------------------------------
# HUD rendering (baked into pixel arrays)
# ---------------------------------------------------------------------------

AXES = [
    # (label, action_idx, pos_rgb, neg_rgb)
    ("x depth",  0, (255, 77, 77),  (77, 230, 230)),
    ("y horiz",  1, (255, 77, 255), (255, 255, 51)),
    ("z height", 2, (51, 128, 255), (255, 128, 26)),
]
GRIP_OPEN_RGB = (0, 255, 0)
GRIP_CLOSED_RGB = (255, 0, 0)

BAR_Y_START = -44   # offset from bottom
BAR_SPACING = 12
BAR_H = 6
BAR_CENTER_X = 90
BAR_MAX_W = 55
LABEL_X = 4


def _draw_rect(img, r0, c0, r1, c1, color):
    """Draw filled rectangle on image (row, col coords). Clips to bounds."""
    h, w = img.shape[:2]
    r0, r1 = max(0, int(r0)), min(h, int(r1))
    c0, c1 = max(0, int(c0)), min(w, int(c1))
    if r0 < r1 and c0 < c1:
        img[r0:r1, c0:c1] = color


def _draw_text_simple(img, row, col, text, color=(255, 255, 255)):
    """Draw text using PIL (small font). Mutates img in-place."""
    from PIL import ImageDraw, ImageFont
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 9)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((int(col), int(row)), text, fill=color, font=font)
    img[:] = np.array(pil)


def render_hud(img, actions_t, label_prefix=""):
    """Render action HUD onto a copy of img. Returns new image."""
    out = img.copy()
    h = out.shape[0]
    gripper = actions_t[3]

    for i, (label, aidx, pos_rgb, neg_rgb) in enumerate(AXES):
        val = actions_t[aidx]
        row_center = h + BAR_Y_START + i * BAR_SPACING
        color = pos_rgb if val >= 0 else neg_rgb

        # Bar
        bar_w = abs(val) * BAR_MAX_W
        if val >= 0:
            _draw_rect(out, row_center - BAR_H // 2, BAR_CENTER_X,
                        row_center + BAR_H // 2, BAR_CENTER_X + bar_w, color)
        else:
            _draw_rect(out, row_center - BAR_H // 2, BAR_CENTER_X - bar_w,
                        row_center + BAR_H // 2, BAR_CENTER_X, color)

        # Zero tick (white)
        _draw_rect(out, row_center - BAR_H // 2 - 1, BAR_CENTER_X - 1,
                    row_center + BAR_H // 2 + 1, BAR_CENTER_X + 1, (255, 255, 255))

        # Label + value
        prefix = f"{label_prefix}{label}" if label_prefix else label
        _draw_text_simple(out, row_center - 5, LABEL_X, f"{prefix} {val:+.2f}", color)

    # Gripper dot
    grip_row = h + BAR_Y_START + 1 * BAR_SPACING  # middle bar
    grip_col = BAR_CENTER_X + BAR_MAX_W + 20
    grip_color = GRIP_OPEN_RGB if gripper < 0 else GRIP_CLOSED_RGB
    grip_label = "open" if gripper < 0 else "closed"
    _draw_rect(out, grip_row - 4, grip_col - 4, grip_row + 4, grip_col + 4, grip_color)
    _draw_text_simple(out, grip_row - 5, grip_col + 8, grip_label, grip_color)

    return out


def build_gt_frames(images, actions):
    """Bake GT HUD into all frames."""
    return np.stack([render_hud(images[t], actions[t]) for t in range(len(images))])


def build_sidebyside_frames(images, gt_actions, pred_actions):
    """Build side-by-side frames: GT left, predicted right."""
    T = len(images)
    frames = []
    for t in range(T):
        left = render_hud(images[t], gt_actions[t])
        right = render_hud(images[t], pred_actions[t])
        # Add labels
        _draw_text_simple(left, 2, 2, "GT", (255, 255, 255))
        _draw_text_simple(right, 2, 2, "PRED", (255, 255, 255))
        # Add thin white separator
        sep = np.full((images[t].shape[0], 2, 3), 255, dtype=np.uint8)
        frames.append(np.concatenate([left, sep, right], axis=1))
    return np.stack(frames)


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

class ModelRunner:
    """Manages OpenVLA base model + LoRA adapter loading and inference."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.adapter_path = None
        self._base_loaded = False

    def _ensure_base(self):
        if self._base_loaded:
            return
        import torch
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

        print("Loading base OpenVLA model (this takes a minute)...")
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            "openvla/openvla-7b", trust_remote_code=True
        )
        self._base_loaded = True
        print("Base model loaded.")

    def load_adapter(self, adapter_dir: str):
        """Load or swap LoRA adapter."""
        import torch
        from peft import PeftModel

        self._ensure_base()
        adapter_dir = str(adapter_dir)

        # If already a PeftModel, unload old adapter first
        if isinstance(self.model, PeftModel):
            self.model = self.model.unload()

        self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.eval()

        # Load dataset statistics for unnormalization — merge into existing norm_stats
        stats_path = Path(adapter_dir) / "dataset_statistics.json"
        if stats_path.exists():
            with open(stats_path) as f:
                new_stats = json.load(f)
            if not hasattr(self.model, "norm_stats") or self.model.norm_stats is None:
                self.model.norm_stats = {}
            self.model.norm_stats.update(new_stats)
            print(f"Loaded dataset stats from {stats_path} (keys: {list(new_stats.keys())})")

        self.adapter_path = adapter_dir
        print(f"LoRA adapter loaded from {adapter_dir}")

    def predict_episode(self, images, instruction, progress_cb=None):
        """Run inference on all frames. Returns (T, 4) predicted actions."""
        import torch

        self._ensure_base()
        T = len(images)
        pred_actions = np.zeros((T, 4), dtype=np.float32)

        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

        with torch.inference_mode():
            for t in range(T):
                pil_img = Image.fromarray(images[t]).convert("RGB")
                inputs = self.processor(prompt, pil_img).to("cuda", dtype=torch.bfloat16)
                action = self.model.predict_action(
                    **inputs, unnorm_key=_UNNORM_KEY, do_sample=False
                )
                pred_actions[t] = action[:4]
                if progress_cb:
                    progress_cb(t + 1, T)

        return pred_actions


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------

def main():
    import napari
    from qtpy.QtWidgets import QFileDialog

    print("Loading RLDS dataset...")
    episodes = load_episodes(split="all")
    print(f"Loaded {len(episodes)} episodes.")

    if not episodes:
        print("No episodes found.")
        return

    viewer = napari.Viewer(title="Go VLA RLDS Viewer")

    current_ep = [0]
    runner = ModelRunner()
    side_by_side = [False]  # whether we're currently showing side-by-side
    pred_actions_cache = {}  # (ep_idx, adapter_path) -> pred_actions

    def _show_gt(idx):
        """Show GT-only view for episode idx."""
        idx = idx % len(episodes)
        current_ep[0] = idx
        ep = episodes[idx]
        side_by_side[0] = False
        gt_frames = build_gt_frames(ep["images"], ep["actions"])
        img_layer.data = gt_frames
        viewer.title = f"Ep {idx}/{len(episodes)-1} [{ep['split']}]: {ep['instruction']}"

    def _show_sidebyside(idx, pred_actions):
        """Show side-by-side view for episode idx."""
        ep = episodes[idx]
        side_by_side[0] = True
        frames = build_sidebyside_frames(ep["images"], ep["actions"], pred_actions)
        img_layer.data = frames
        adapter_name = Path(runner.adapter_path).parent.name if runner.adapter_path else "?"
        viewer.title = f"Ep {idx}/{len(episodes)-1} [{ep['split']} | {adapter_name}]: {ep['instruction']}"

    # Initial display
    ep = episodes[0]
    gt_frames = build_gt_frames(ep["images"], ep["actions"])
    img_layer = viewer.add_image(gt_frames, name="frames", rgb=True)
    viewer.title = f"Ep 0/{len(episodes)-1} [{ep['split']}]: {ep['instruction']}"

    @viewer.bind_key("n")
    def next_episode(viewer):
        _show_gt(current_ep[0] + 1)

    @viewer.bind_key("p")
    def prev_episode(viewer):
        _show_gt(current_ep[0] - 1)

    @viewer.bind_key("Escape")
    def back_to_gt(viewer):
        _show_gt(current_ep[0])

    @viewer.bind_key("l")
    def load_checkpoint(viewer):
        """Open file dialog to select checkpoint directory."""
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setWindowTitle("Select checkpoint directory (with adapter_config.json)")
        if dialog.exec_():
            selected = dialog.selectedFiles()[0]
            if not (Path(selected) / "adapter_config.json").exists():
                print(f"No adapter_config.json found in {selected}")
                return
            try:
                runner.load_adapter(selected)
            except Exception as e:
                print(f"Failed to load adapter: {e}")

    @viewer.bind_key("r")
    def run_inference(viewer):
        """Run model on current episode in background thread."""
        if not runner._base_loaded and runner.adapter_path is None:
            print("No checkpoint loaded. Press L first.")
            return

        idx = current_ep[0]
        cache_key = (idx, runner.adapter_path)
        if cache_key in pred_actions_cache:
            _show_sidebyside(idx, pred_actions_cache[cache_key])
            return

        ep = episodes[idx]
        viewer.title = f"Running inference... 0/{len(ep['images'])}"

        def _progress(done, total):
            viewer.title = f"Running inference... {done}/{total}"

        def _run():
            try:
                pred = runner.predict_episode(ep["images"], ep["instruction"], _progress)
                pred_actions_cache[cache_key] = pred
                _show_sidebyside(idx, pred)
            except Exception as e:
                print(f"Inference failed: {e}")
                import traceback
                traceback.print_exc()
                _show_gt(idx)

        threading.Thread(target=_run, daemon=True).start()

    print("Controls:")
    print("  N/P    = next/prev episode")
    print("  L      = load LoRA checkpoint")
    print("  R      = run inference (side-by-side)")
    print("  Escape = back to GT-only view")
    napari.run()


if __name__ == "__main__":
    main()
