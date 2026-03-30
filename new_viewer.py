"""napari RLDS dataset viewer for Go VLA episodes.

Usage:
    MUJOCO_GL=egl conda run --no-capture-output -n mujogo python viewer.py
"""

import json
from pathlib import Path
import sys
import os

import numpy as np
from PIL import Image

# Must set before importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


REPO_ROOT = Path(__file__).resolve().parent


def _safe_register(auto_class, key, value):
    try:
        auto_class.register(key, value)
    except ValueError:
        pass


def _ensure_openvla_imports():
    openvla_root = REPO_ROOT / "openvla"
    if str(openvla_root) not in sys.path:
        sys.path.insert(0, str(openvla_root))

    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    _safe_register(AutoConfig, "openvla", OpenVLAConfig)
    _safe_register(AutoImageProcessor, OpenVLAConfig, PrismaticImageProcessor)
    _safe_register(AutoProcessor, OpenVLAConfig, PrismaticProcessor)
    _safe_register(AutoModelForVision2Seq, OpenVLAConfig, OpenVLAForActionPrediction)

    return AutoModelForVision2Seq, AutoProcessor


class ModelRunner:
    """Manages OpenVLA base model + LoRA adapter loading and inference."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.adapter_path = None
        self._base_loaded = False

    def _base_model_id_or_path(self):
        return os.environ.get("OPENVLA_BASE_MODEL", "openvla/openvla-7b")

    def _ensure_base(self):
        if self._base_loaded:
            return

        import torch

        AutoModelForVision2Seq, AutoProcessor = _ensure_openvla_imports()

        print("Loading base OpenVLA model (this takes a minute)...")
        base_model = self._base_model_id_or_path()
        load_kwargs = {
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["torch_dtype"] = torch.float32
        self.model = AutoModelForVision2Seq.from_pretrained(base_model, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        self._base_loaded = True
        print("Base model loaded.")

    def load_adapter(self, adapter_dir: str):
        """Load or swap LoRA adapter."""
        from peft import PeftModel

        self._ensure_base()
        adapter_dir = str(adapter_dir)

        if isinstance(self.model, PeftModel):
            self.model = self.model.unload()

        self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.eval()

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

    def export_merged_checkpoint(self, output_dir: str, adapter_dir: str = None, base_model: str = None):
        """Export a benchmark-ready merged checkpoint using the viewer loader wiring."""
        import shutil
        import torch
        from peft import PeftModel

        AutoModelForVision2Seq, AutoProcessor = _ensure_openvla_imports()

        adapter_source = adapter_dir or self.adapter_path
        if not adapter_source:
            raise ValueError("adapter_dir is required when no adapter has been loaded yet")
        adapter_path = Path(adapter_source)

        base_model = str(base_model or self._base_model_id_or_path())
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading full-precision base model from {base_model} for checkpoint export...")
        base = AutoModelForVision2Seq.from_pretrained(
            base_model,
            attn_implementation="sdpa",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        merged = PeftModel.from_pretrained(base, str(adapter_path))
        merged = merged.merge_and_unload()

        stats_path = adapter_path / "dataset_statistics.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            merged.config.norm_stats = stats
            merged.norm_stats = stats
            shutil.copy2(stats_path, out_dir / "dataset_statistics.json")
            print(f"Copied dataset stats from {stats_path} (keys: {list(stats.keys())})")

        merged.save_pretrained(out_dir)

        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        processor.save_pretrained(out_dir)

        for name in [
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ]:
            src = adapter_path / name
            if src.exists():
                shutil.copy2(src, out_dir / name)

        print(f"Merged checkpoint written to {out_dir}")
        return out_dir

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
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.bfloat16 if device == "cuda" else torch.float32
                inputs = self.processor(prompt, pil_img).to(device, dtype=dtype)
                action = self.model.predict_action(
                    **inputs, unnorm_key="go_vla_dataset", do_sample=False
                )
                pred_actions[t] = action[:4]
                if progress_cb:
                    progress_cb(t + 1, T)

        return pred_actions


def load_episodes(data_dir=os.path.expanduser("~/tensorflow_datasets")):
    """Load all episodes into lists of dicts."""
    rlds_builder_root = REPO_ROOT / "benchmarks" / "go_vla_benchmark" / "rlds_builder"
    if str(rlds_builder_root) not in sys.path:
        sys.path.insert(0, str(rlds_builder_root))

    import tensorflow_datasets as tfds  # noqa: F401
    from go_vla_dataset.go_vla_dataset_dataset_builder import Builder

    builder = Builder(data_dir=data_dir)
    ds = builder.as_dataset(split="train")

    episodes = []
    for ep in ds:
        images = []
        actions = []
        instruction = None
        for step in ep["steps"]:
            images.append(step["observation"]["image"].numpy())
            actions.append(step["action"].numpy())
            if instruction is None:
                instruction = step["language_instruction"].numpy().decode("utf-8")
        episodes.append(
            {
                "images": np.stack(images),
                "actions": np.stack(actions),
                "instruction": instruction or "",
            }
        )
    return episodes


# -- HUD layout --
HUD_LEFT = 6.0
BAR_CENTER = 90.0
BAR_MAX_W = 55.0
BAR_H = 6.0
BAR_SPACING = 12.0
GRIP_RADIUS = 5.0
VALUE_COL = BAR_CENTER + BAR_MAX_W + 5.0  # where numeric value text goes

# Axis definitions: (name, description, action_index, pos_color, neg_color)
AXES = [
    ("x", "depth",  0, [1.0, 0.3, 0.3, 0.85], [0.3, 0.9, 0.9, 0.85]),
    ("y", "horiz",  1, [1.0, 0.3, 1.0, 0.85], [1.0, 1.0, 0.2, 0.85]),
    ("z", "height", 2, [0.2, 0.5, 1.0, 0.85], [1.0, 0.5, 0.1, 0.85]),
]

GRIP_OPEN_COLOR = [0, 1, 0, 1]
GRIP_CLOSED_COLOR = [1, 0, 0, 1]


def _hbar(value, center_row, center_col, max_w, h):
    """Horizontal bar rectangle for a signed value in [-1,1]."""
    w = abs(value) * max_w
    r0 = center_row - h / 2.0
    r1 = center_row + h / 2.0
    if value >= 0:
        c0, c1 = center_col, center_col + w
    else:
        c0, c1 = center_col - w, center_col
    return np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]])


def build_overlays(actions, img_h, img_w):
    """Pre-compute per-frame HUD overlays."""
    T = len(actions)
    gripper = actions[:, 3]

    base_row = img_h - 12.0

    # Compute bar row positions (bottom axis = z, then y, then x on top)
    axis_rows = {}
    for i, (name, desc, aidx, pc, nc) in enumerate(AXES):
        axis_rows[name] = base_row - (len(AXES) - 1 - i) * BAR_SPACING

    # Bars and colors per axis per frame
    bars = {a[0]: [] for a in AXES}
    bar_colors = {a[0]: np.zeros((T, 4)) for a in AXES}
    # Per-frame value strings for each axis
    value_strs = {a[0]: [] for a in AXES}

    for name, desc, aidx, pos_color, neg_color in AXES:
        vals = actions[:, aidx]
        row = axis_rows[name]
        for i in range(T):
            bars[name].append(_hbar(vals[i], row, BAR_CENTER, BAR_MAX_W, BAR_H))
            bar_colors[name][i] = pos_color if vals[i] >= 0 else neg_color
            value_strs[name].append(f"{vals[i]:+.2f}")

    # Gripper
    grip_col = BAR_CENTER + BAR_MAX_W + 45.0
    grip_row = axis_rows["y"]  # middle bar height
    grip_colors = np.zeros((T, 4))
    uses_openvla_gripper = (gripper >= 0.0) & (gripper <= 1.0)
    is_open = np.where(uses_openvla_gripper, gripper > 0.5, gripper < 0.0)
    grip_colors[is_open] = GRIP_OPEN_COLOR
    grip_colors[~is_open] = GRIP_CLOSED_COLOR
    grip_strs = ["open" if bool(is_open[i]) else "closed" for i in range(T)]

    # Zero-line ticks
    zero_lines = []
    for name in axis_rows:
        row = axis_rows[name]
        zero_lines.append(np.array([
            [row - BAR_H / 2.0 - 1, BAR_CENTER],
            [row + BAR_H / 2.0 + 1, BAR_CENTER],
        ]))

    return {
        "bars": bars,
        "bar_colors": bar_colors,
        "value_strs": value_strs,
        "axis_rows": axis_rows,
        "grip_row": grip_row,
        "grip_col": grip_col,
        "grip_colors": grip_colors,
        "grip_strs": grip_strs,
        "zero_lines": zero_lines,
        "T": T,
    }


def main():
    import napari

    print("Loading RLDS dataset...")
    episodes = load_episodes()
    print(f"Loaded {len(episodes)} episodes.")

    if not episodes:
        print("No episodes found.")
        return

    viewer = napari.Viewer(title="Go VLA RLDS Viewer")

    current_ep = [0]
    ep = episodes[0]
    img_h, img_w = ep["images"].shape[1], ep["images"].shape[2]

    overlays = [None]
    overlays[0] = build_overlays(ep["actions"], img_h, img_w)
    ov = overlays[0]

    # Image layer
    img_layer = viewer.add_image(ep["images"], name="frames", rgb=True)

    # Bar layers
    bar_layers = {}
    for name, desc, aidx, pc, nc in AXES:
        bar_layers[name] = viewer.add_shapes(
            [ov["bars"][name][0]], shape_type="polygon",
            face_color=ov["bar_colors"][name][0:1],
            edge_color="transparent", name=f"{name} ({desc})",
        )

    # Zero-line markers
    zero_layer = viewer.add_shapes(
        ov["zero_lines"], shape_type="line",
        edge_color=[1, 1, 1, 0.5], edge_width=1, name="zeros",
    )

    # Static axis labels: "x (depth)", "y (horiz)", "z (height)"
    label_pts = []
    label_strs = []
    for name, desc, aidx, pc, nc in AXES:
        label_pts.append([ov["axis_rows"][name], HUD_LEFT])
        label_strs.append(f"{name} ({desc})")
    viewer.add_points(
        np.array(label_pts), size=1, face_color="transparent",
        text={"string": label_strs, "color": "white", "size": 8, "anchor": "lower_left"},
        name="axis labels",
    )

    # Dynamic value labels (updated per frame)
    val_pts = []
    for name, desc, aidx, pc, nc in AXES:
        val_pts.append([ov["axis_rows"][name], VALUE_COL])
    val_pts = np.array(val_pts)
    init_val_strs = [ov["value_strs"][a[0]][0] for a in AXES]
    val_layer = viewer.add_points(
        val_pts, size=1, face_color="transparent",
        text={"string": init_val_strs, "color": "white", "size": 8, "anchor": "lower_left"},
        name="values",
    )

    # Gripper dot + label
    grip_layer = viewer.add_points(
        np.array([[ov["grip_row"], ov["grip_col"]]]),
        size=GRIP_RADIUS * 2, face_color=ov["grip_colors"][0:1],
        name="gripper",
    )
    grip_label_layer = viewer.add_points(
        np.array([[ov["grip_row"], ov["grip_col"] + 10]]),
        size=1, face_color="transparent",
        text={"string": [ov["grip_strs"][0]], "color": "white", "size": 8, "anchor": "lower_left"},
        name="grip label",
    )

    def _on_dims_change(event):
        t = viewer.dims.current_step[0]
        ov = overlays[0]
        t = min(t, ov["T"] - 1)

        for name, desc, aidx, pc, nc in AXES:
            bar_layers[name].data = [ov["bars"][name][t]]
            bar_layers[name].face_color = ov["bar_colors"][name][t:t+1]

        val_layer.text = {
            "string": [ov["value_strs"][a[0]][t] for a in AXES],
            "color": "white", "size": 8, "anchor": "lower_left",
        }

        grip_layer.face_color = ov["grip_colors"][t:t+1]
        grip_label_layer.text = {
            "string": [ov["grip_strs"][t]],
            "color": "white", "size": 8, "anchor": "lower_left",
        }

    viewer.dims.events.current_step.connect(_on_dims_change)
    _on_dims_change(None)

    viewer.title = f"Ep {current_ep[0]}/{len(episodes)-1}: {ep['instruction']}"

    def _load_episode(idx):
        idx = idx % len(episodes)
        current_ep[0] = idx
        ep = episodes[idx]
        img_layer.data = ep["images"]
        overlays[0] = build_overlays(ep["actions"], ep["images"].shape[1], ep["images"].shape[2])
        zero_layer.data = overlays[0]["zero_lines"]
        _on_dims_change(None)
        viewer.title = f"Ep {current_ep[0]}/{len(episodes)-1}: {ep['instruction']}"

    @viewer.bind_key("n")
    def next_episode(viewer):
        """Next episode."""
        _load_episode(current_ep[0] + 1)

    @viewer.bind_key("p")
    def prev_episode(viewer):
        """Previous episode."""
        _load_episode(current_ep[0] - 1)

    print("Controls: N/P = next/prev episode, slider = scrub frames")
    napari.run()


if __name__ == "__main__":
    main()
