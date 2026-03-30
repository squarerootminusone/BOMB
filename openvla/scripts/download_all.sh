#!/bin/bash
set -uo pipefail

DATA_DIR="/home/marcin/Projects/dsait4125/LIBERO/libero/datasets"
HF_BASE="https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main"
MAX_RETRIES=5

# REAL filenames verified from HuggingFace API on 2026-02-26
LIBERO_GOAL_FILES=(
    "open_the_middle_drawer_of_the_cabinet_demo.hdf5"
    "open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5"
    "push_the_plate_to_the_front_of_the_stove_demo.hdf5"
    "put_the_bowl_on_the_plate_demo.hdf5"
    "put_the_bowl_on_the_stove_demo.hdf5"
    "put_the_bowl_on_top_of_the_cabinet_demo.hdf5"
    "put_the_cream_cheese_in_the_bowl_demo.hdf5"
    "put_the_wine_bottle_on_the_rack_demo.hdf5"
    "put_the_wine_bottle_on_top_of_the_cabinet_demo.hdf5"
    "turn_on_the_stove_demo.hdf5"
)

download_file() {
    local url="$1"
    local dest="$2"

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "SKIP (exists): $(basename "$dest") ($(du -h "$dest" | cut -f1))"
        return 0
    fi

    mkdir -p "$(dirname "$dest")"
    for attempt in $(seq 1 $MAX_RETRIES); do
        echo "Downloading (attempt ${attempt}/${MAX_RETRIES}): $(basename "$dest")"
        curl -L -C - --retry 3 --retry-delay 5 -o "$dest" "$url" 2>&1
        if [ -f "$dest" ] && [ -s "$dest" ]; then
            echo "OK: $(basename "$dest") ($(du -h "$dest" | cut -f1))"
            return 0
        fi
        echo "WARN: Failed attempt ${attempt}, retrying in $((2**attempt))s..."
        sleep $((2**attempt))
    done
    echo "FAILED: $(basename "$dest")"
    return 1
}

echo "=== Downloading libero_goal ==="
FAIL=0
for fname in "${LIBERO_GOAL_FILES[@]}"; do
    download_file "${HF_BASE}/libero_goal/${fname}" "${DATA_DIR}/libero_goal/${fname}" || FAIL=$((FAIL+1))
done

echo ""
echo "=== Downloading openvla-7b model ==="
OPENVLA_BASE="https://huggingface.co/openvla/openvla-7b/resolve/main"
OPENVLA_DIR="/home/marcin/.cache/huggingface/hub/models--openvla--openvla-7b/manual"
mkdir -p "$OPENVLA_DIR"

# Just use huggingface-cli for the model since it handles shards + config properly
for attempt in $(seq 1 $MAX_RETRIES); do
    echo "openvla-7b attempt ${attempt}/${MAX_RETRIES}..."
    huggingface-cli download openvla/openvla-7b 2>&1 && break
    echo "Retrying in $((2**attempt))s..."
    sleep $((2**attempt))
done

echo ""
echo "=== FINAL VERIFICATION ==="
for suite in libero_object libero_spatial libero_goal; do
    count=$(find "${DATA_DIR}/${suite}" -maxdepth 1 -name "*.hdf5" -size +0c 2>/dev/null | wc -l)
    echo "${suite}: ${count}/10 files"
done

SNAP=$(find ~/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
if [ -n "$SNAP" ]; then
    for s in 1 2 3; do
        f="$SNAP/model-0000${s}-of-00003.safetensors"
        [ -f "$f" ] && [ -s "$f" ] && echo "openvla-7b shard ${s}: OK ($(du -h "$f" | cut -f1))" || echo "openvla-7b shard ${s}: MISSING"
    done
fi

echo ""
[ $FAIL -eq 0 ] && echo "ALL DONE" || echo "${FAIL} files failed"
echo "Finished: $(date)"
