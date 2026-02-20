"""
Download all VLA model checkpoints referenced in the LIBERO-Plus paper.

Models evaluated in LIBERO-Plus (Table 6, arxiv.org/abs/2510.13626):
  1. OpenVLA          — per-suite fine-tuned (LoRA)
  2. OpenVLA-OFT      — per-suite fine-tuned (OFT)
  3. OpenVLA-OFT_m    — multi-suite fine-tuned (OFT on all 4 suites)
  4. NORA             — 3B VLA (Qwen-2.5-VL backbone)
  5. WorldVLA         — autoregressive action world model (Alibaba DAMO)
  6. UniVLA           — latent-action VLA
  7. pi0              — Physical Intelligence (GCS bucket, not HF)
  8. pi0-fast         — Physical Intelligence (GCS bucket, not HF)
  9. RIPT-VLA         — RL post-trained VLA

Usage:
    python download_checkpoints.py                # download all
    python download_checkpoints.py --only openvla # download only OpenVLA checkpoints
    python download_checkpoints.py --list         # just print what would be downloaded
"""

import argparse
import subprocess
import sys


# ============================================================================
# All checkpoints from the LIBERO-Plus paper
# ============================================================================
CHECKPOINTS = {
    # --- OpenVLA (original, LoRA fine-tuned per suite) ---
    "openvla": [
        "openvla/openvla-7b-finetuned-libero-spatial",
        "openvla/openvla-7b-finetuned-libero-object",
        "openvla/openvla-7b-finetuned-libero-goal",
        "openvla/openvla-7b-finetuned-libero-10",
    ],
    # --- OpenVLA-OFT (per-suite) ---
    "openvla-oft": [
        "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "moojink/openvla-7b-oft-finetuned-libero-object",
        "moojink/openvla-7b-oft-finetuned-libero-goal",
        "moojink/openvla-7b-oft-finetuned-libero-10",
    ],
    # --- OpenVLA-OFT_m (multi-suite, single checkpoint for all 4) ---
    "openvla-oft-multi": [
        "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
    ],
    # --- OpenVLA-OFT+ (LIBERO-Plus mix-SFT from Sylvest) ---
    "openvla-oft-plus": [
        "Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata",
        "Sylvest/openvla-7b-oft-finetuned-libero-without-wrist",
    ],
    # --- NORA (3B, declare-lab) ---
    "nora": [
        "declare-lab/nora",
        "declare-lab/nora-finetuned-libero-object",
    ],
    # --- WorldVLA (Alibaba DAMO) ---
    "worldvla": [
        "Alibaba-DAMO-Academy/WorldVLA",
    ],
    # --- UniVLA ---
    "univla": [
        "qwbu/univla-7b-224-sft-libero",
    ],
    # --- RIPT-VLA (RL post-trained) ---
    "ript-vla": [
        "tanshh97/RIPT_VLA",
    ],
}

# pi0 and pi0-fast are on Google Cloud Storage, not HuggingFace
GCS_CHECKPOINTS = {
    "pi0": "https://storage.googleapis.com/openpi-assets/checkpoints/pi0_libero",
    "pi0-fast": "https://storage.googleapis.com/openpi-assets/checkpoints/pi0_fast_libero",
}


def download_hf_model(model_id: str, dry_run: bool = False) -> bool:
    """Download a HuggingFace model using huggingface-cli."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_id}")
    print(f"{'='*60}")
    if dry_run:
        print(f"  [DRY RUN] Would download {model_id}")
        return True
    try:
        subprocess.run(
            ["huggingface-cli", "download", model_id],
            check=True,
        )
        print(f"  Done: {model_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {model_id} (exit code {e.returncode})")
        return False
    except FileNotFoundError:
        print("  ERROR: huggingface-cli not found. Install with: pip install huggingface-hub[cli]")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download LIBERO-Plus model checkpoints")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(CHECKPOINTS.keys()) + list(GCS_CHECKPOINTS.keys()),
        help="Only download specific model groups",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all checkpoints without downloading",
    )
    parser.add_argument(
        "--skip-gcs",
        action="store_true",
        help="Skip Google Cloud Storage checkpoints (pi0, pi0-fast)",
    )
    args = parser.parse_args()

    # Determine which groups to download
    hf_groups = list(CHECKPOINTS.keys())
    gcs_groups = list(GCS_CHECKPOINTS.keys())
    if args.only:
        hf_groups = [g for g in args.only if g in CHECKPOINTS]
        gcs_groups = [g for g in args.only if g in GCS_CHECKPOINTS]

    # List mode
    if args.list:
        print("HuggingFace checkpoints:")
        for group in hf_groups:
            print(f"\n  [{group}]")
            for model_id in CHECKPOINTS[group]:
                print(f"    - {model_id}")
        if not args.skip_gcs:
            print("\nGoogle Cloud Storage checkpoints:")
            for group in gcs_groups:
                print(f"    - {GCS_CHECKPOINTS[group]}")
        total = sum(len(CHECKPOINTS[g]) for g in hf_groups) + (0 if args.skip_gcs else len(gcs_groups))
        print(f"\nTotal: {total} checkpoints")
        return

    # Download HuggingFace models
    succeeded, failed = 0, 0
    for group in hf_groups:
        for model_id in CHECKPOINTS[group]:
            if download_hf_model(model_id):
                succeeded += 1
            else:
                failed += 1

    # Download GCS checkpoints
    if not args.skip_gcs and gcs_groups:
        print("\n" + "=" * 60)
        print("Google Cloud Storage checkpoints (pi0 / pi0-fast)")
        print("These require gsutil or manual download:")
        for group in gcs_groups:
            url = GCS_CHECKPOINTS[group]
            print(f"  gsutil -m cp -r {url} ./checkpoints/{group}/")
        print("=" * 60)

    print(f"\nDone. {succeeded} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()
