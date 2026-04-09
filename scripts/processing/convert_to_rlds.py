#!/usr/bin/env python3
"""Convert Go VLA HDF5 demonstrations to RLDS (TensorFlow Datasets) format."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
BUILDER_DIR = THIS_FILE.parents[1] / "rlds_builder"

sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Go VLA HDF5 to RLDS format."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to the source HDF5 file (e.g. data/source_go.hdf5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/tensorflow_datasets"),
        help="TFDS data directory (default: ~/tensorflow_datasets)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hdf5_path = str(Path(args.input).resolve())

    # Point the builder at the HDF5 file via env var.
    os.environ["GO_VLA_HDF5_PATH"] = hdf5_path

    sys.path.insert(0, str(BUILDER_DIR))
    from go_vla_dataset.go_vla_dataset_dataset_builder import Builder

    builder = Builder(data_dir=args.output_dir)
    builder.download_and_prepare()

    # Print summary stats.
    info = builder.info
    print("\n=== RLDS Conversion Complete ===")
    print(f"  Dataset name : {info.name}")
    print(f"  Version      : {info.version}")
    print(f"  Data dir     : {builder.data_dir}")
    print(f"  Description  : {info.description.strip()[:120]}...")

    for split_name in ["train", "val"]:
        ds = builder.as_dataset(split=split_name)
        num_episodes = 0
        total_steps = 0
        for episode in ds:
            num_episodes += 1
            total_steps += sum(1 for _ in episode["steps"])
        print(f"  [{split_name:5s}] Episodes: {num_episodes}  Steps: {total_steps}")
    print(f"  Source HDF5  : {hdf5_path}")


if __name__ == "__main__":
    main()
