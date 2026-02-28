#!/usr/bin/env python3
"""Print a concise summary of a MimicGen-compatible dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.paths import bootstrap_pythonpath

bootstrap_pythonpath(REPO_ROOT)

from go_vla_benchmark.dataset_io import summarize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="hdf5 dataset to inspect")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_dataset(args.dataset)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
