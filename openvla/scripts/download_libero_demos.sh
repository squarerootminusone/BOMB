#!/bin/bash
#
# download_libero_demos.sh
#
# Downloads LIBERO training demonstrations (HDF5 files) from Hugging Face.
# Wrapper around LIBERO/benchmark_scripts/download_libero_datasets.py.
#
# Usage:
#   bash scripts/download_libero_demos.sh [--datasets DATASET_NAME]
#
# Options:
#   --datasets    Which datasets to download. Options: all, libero_goal, libero_spatial, libero_object, libero_100
#                 Default: all
#
# The downloaded HDF5 files will be placed in LIBERO/libero/datasets/ by default
# (configurable via ~/.libero/config.yaml).

set -euo pipefail

DATASETS="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LIBERO_ROOT="${PROJECT_ROOT}/../LIBERO"

echo "======================================"
echo "LIBERO Demo Dataset Downloader"
echo "======================================"
echo "Datasets: ${DATASETS}"
echo "LIBERO root: ${LIBERO_ROOT}"
echo "======================================"

# Verify LIBERO directory exists
if [ ! -d "$LIBERO_ROOT" ]; then
    echo "ERROR: LIBERO directory not found at ${LIBERO_ROOT}"
    echo "Please clone the LIBERO repository first:"
    echo "  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git"
    exit 1
fi

# Verify the download script exists
DOWNLOAD_SCRIPT="${LIBERO_ROOT}/benchmark_scripts/download_libero_datasets.py"
if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
    echo "ERROR: Download script not found at ${DOWNLOAD_SCRIPT}"
    exit 1
fi

# Add LIBERO benchmark_scripts to Python path (for init_path import)
cd "$LIBERO_ROOT/benchmark_scripts"

echo ""
echo "Starting download from Hugging Face..."
python download_libero_datasets.py --datasets "$DATASETS" --use-huggingface

echo ""
echo "Download complete!"
echo ""

# Verify downloaded files
DATASETS_DIR=$(python -c "from libero.libero import get_libero_path; print(get_libero_path('datasets'))" 2>/dev/null || echo "${LIBERO_ROOT}/libero/datasets")
echo "Checking downloaded datasets in: ${DATASETS_DIR}"

for suite in libero_spatial libero_object libero_goal libero_10 libero_90; do
    suite_dir="${DATASETS_DIR}/${suite}"
    if [ -d "$suite_dir" ]; then
        num_files=$(find "$suite_dir" -name "*.hdf5" | wc -l)
        echo "  ${suite}: ${num_files} HDF5 files"
    fi
done

echo ""
echo "Done! HDF5 files are ready for VPT fine-tuning."
