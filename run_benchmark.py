"""Evaluate a finetuned OpenVLA model on the full RLDS dataset.

For each episode, runs inference frame-by-frame, computes action prediction
error + action token accuracy, saves side-by-side GT vs predicted videos,
and logs metrics to wandb.

Usage:
    conda run --no-capture-output -n mujogo python run_benchmark.py \
        adapter_path=outputs/2026-03-11/16-18-14/checkpoints/best
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import hydra
import imageio
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from viewer import load_episodes, build_sidebyside_frames, ModelRunner, _UNNORM_KEY


def _get_action_norm_stats(runner):
    """Extract q01/q99 norm stats from the loaded model."""
    stats = runner.model.norm_stats[_UNNORM_KEY]["action"]
    q01 = np.array(stats["q01"][:4], dtype=np.float64)
    q99 = np.array(stats["q99"][:4], dtype=np.float64)
    mask = np.array(stats.get("mask", [True] * 4)[:4])
    return q01, q99, mask


def _normalize_actions(actions, q01, q99, mask):
    """Unnormalized continuous actions -> normalized [-1, 1]."""
    normed = np.where(
        mask,
        2.0 * (actions - q01) / (q99 - q01) - 1.0,
        actions,
    )
    return np.clip(normed, -1.0, 1.0)


def _discretize(normed_actions, n_bins=256):
    """Normalized actions [-1,1] -> bin indices (matching ActionTokenizer)."""
    bins = np.linspace(-1, 1, n_bins)
    return np.digitize(normed_actions, bins)  # values in [1, n_bins]


def compute_token_accuracy(gt_actions, pred_actions, q01, q99, mask):
    """Compute per-axis and total action token accuracy.

    Returns dict with token_acc_x, token_acc_y, token_acc_z, token_acc_gripper, token_acc_total.
    """
    gt_norm = _normalize_actions(gt_actions, q01, q99, mask)
    pred_norm = _normalize_actions(pred_actions, q01, q99, mask)

    gt_tokens = _discretize(gt_norm)    # (T, 4)
    pred_tokens = _discretize(pred_norm)  # (T, 4)

    match = (gt_tokens == pred_tokens)  # (T, 4)
    axis_names = ["x", "y", "z", "gripper"]
    metrics = {}
    for j, name in enumerate(axis_names):
        metrics[f"token_acc_{name}"] = float(match[:, j].mean())
    metrics["token_acc_total"] = float(match.mean())
    return metrics


def check_task_success(gt_actions, pred_actions, l1_threshold=0.05):
    """Proxy for task success on offline data.

    An episode is considered successful if the mean L1 error across all
    axes and timesteps is below the threshold.
    """
    total_l1 = np.abs(pred_actions - gt_actions).mean()
    return total_l1 < l1_threshold


@hydra.main(version_base=None, config_path="configs", config_name="benchmark")
def main(cfg: DictConfig):
    output_dir = Path(cfg.output_dir)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # --- wandb init ---
    wandb_cfg = cfg.wandb
    wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        name=wandb_cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # --- Load dataset ---
    print("Loading RLDS episodes...")
    episodes = load_episodes(data_dir=os.path.expanduser(cfg.data_dir))
    print(f"Loaded {len(episodes)} episodes.")

    # --- Load model ---
    runner = ModelRunner()
    if cfg.adapter_path is not None:
        adapter_path = Path(cfg.adapter_path)
        if not adapter_path.is_absolute():
            adapter_path = Path(hydra.utils.get_original_cwd()) / adapter_path
        runner.load_adapter(str(adapter_path))
    else:
        runner._ensure_base()
        print("Warning: no adapter_path specified, using base model only.")

    # --- Get normalization stats for token accuracy ---
    q01, q99, mask = _get_action_norm_stats(runner)

    # --- Per-episode evaluation ---
    axis_names = ["x", "y", "z", "gripper"]
    all_metrics = []
    num_success = 0

    for i, ep in enumerate(episodes):
        images = ep["images"]
        gt_actions = ep["actions"][:, :4]  # (T, 4)
        instruction = ep["instruction"]

        print(f"Episode {i}/{len(episodes)-1}: {instruction}")

        # Run inference
        def _progress(done, total):
            print(f"  frame {done}/{total}", end="\r")

        pred_actions = runner.predict_episode(images, instruction, progress_cb=_progress)
        print()

        # L1 metrics
        err = np.abs(pred_actions - gt_actions)  # (T, 4)
        per_axis = err.mean(axis=0)  # (4,)
        total_l1 = per_axis.mean()

        metrics = {f"l1_{axis_names[j]}": float(per_axis[j]) for j in range(4)}
        metrics["l1_total"] = float(total_l1)

        # Token accuracy
        token_metrics = compute_token_accuracy(gt_actions, pred_actions, q01, q99, mask)
        metrics.update(token_metrics)

        # Task success (proxy)
        success = check_task_success(gt_actions, pred_actions, l1_threshold=cfg.success_threshold)
        metrics["success"] = int(success)
        num_success += int(success)

        all_metrics.append(metrics)

        print(f"  L1 total={total_l1:.4f}  x={per_axis[0]:.4f} y={per_axis[1]:.4f} "
              f"z={per_axis[2]:.4f} grip={per_axis[3]:.4f}")
        print(f"  Token acc={token_metrics['token_acc_total']:.3f}  success={success}")

        # Build side-by-side video
        frames = build_sidebyside_frames(images, ep["actions"], pred_actions)
        video_path = str(video_dir / f"ep_{i}.mp4")
        writer = imageio.get_writer(video_path, fps=cfg.video_fps,
                                    codec="libx264", pixelformat="yuv420p",
                                    macro_block_size=1)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        # Log to wandb
        ep_log = {f"ep_{i}/{k}": v for k, v in metrics.items()}
        ep_log[f"ep_{i}/video"] = wandb.Video(video_path, format="mp4")
        wandb.log(ep_log)

    # --- Aggregate metrics ---
    agg = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        if key == "success":
            agg["success_rate"] = float(np.mean(vals))
            agg["num_success"] = int(np.sum(vals))
        else:
            agg[f"mean_{key}"] = float(np.mean(vals))
            agg[f"std_{key}"] = float(np.std(vals))

    print(f"\n=== Aggregate metrics ({len(episodes)} episodes) ===")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}")

    wandb.log(agg)

    # --- Summary table ---
    columns = ["episode", "instruction"] + list(all_metrics[0].keys())
    table = wandb.Table(columns=columns)
    for i, ep in enumerate(episodes):
        row = [i, ep["instruction"]] + [all_metrics[i][k] for k in all_metrics[0]]
        table.add_data(*row)
    wandb.log({"summary_table": table})

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
