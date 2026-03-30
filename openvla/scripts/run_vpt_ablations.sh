#!/bin/bash
#
# run_vpt_ablations.sh
#
# Full VPT ablation matrix for OpenVLA on LIBERO.
# Trains {shallow, deep} × {10, 50, 100} prompts, evaluates on 3 LIBERO suites.
#
# Usage:
#   bash scripts/run_vpt_ablations.sh [--train-only] [--eval-only]
#
# Requirements:
#   - LIBERO HDF5 demos downloaded (see scripts/download_libero_demos.sh)
#   - OpenVLA checkpoint available (default: openvla/openvla-7b)

set -euo pipefail

# ==== Configuration ====
VLA_PATHS=("openvla/openvla-7b" "Embodied-CoT/ecot-openvla-7b-bridge")
VLA_NAMES=("openvla" "ecot")
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-../LIBERO/libero/datasets}"
RUN_ROOT="${RUN_ROOT:-runs/vpt_ablations}"
PYTHON="${PYTHON:-/home/marcin/miniforge3/envs/openvla/bin/python}"
WANDB_PROJECT="${WANDB_PROJECT:-openvla}"
WANDB_ENTITY="${WANDB_ENTITY:-stanford-voltron}"

# Training hyperparameters (tuned for single 16GB GPU with 8-bit quantization)
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_STEPS="${MAX_STEPS:-200000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-True}"

# Evaluation parameters
NUM_TRIALS="${NUM_TRIALS:-50}"

# Ablation matrix
VPT_MODES=("shallow" "deep")
VPT_PROMPTS=(10 50 100)
EVAL_SUITES=("libero_spatial" "libero_object" "libero_goal")

# Parse arguments
TRAIN_ONLY=false
EVAL_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --train-only) TRAIN_ONLY=true; shift ;;
        --eval-only) EVAL_ONLY=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "======================================"
echo "VPT Ablation Experiments for OpenVLA"
echo "======================================"
echo "VLA Models:      ${VLA_NAMES[*]}"
echo "VLA Paths:       ${VLA_PATHS[*]}"
echo "Data Root:       ${LIBERO_DATA_ROOT}"
echo "Run Root:        ${RUN_ROOT}"
echo "Modes:           ${VPT_MODES[*]}"
echo "Prompt counts:   ${VPT_PROMPTS[*]}"
echo "Eval suites:     ${EVAL_SUITES[*]}"
echo "8-bit:           ${LOAD_IN_8BIT}"
echo "Batch (effective): $((BATCH_SIZE * GRAD_ACCUM))"
echo "======================================"

# ==== Training Phase ====
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "=== TRAINING PHASE ==="

    for model_idx in "${!VLA_PATHS[@]}"; do
        VLA_PATH="${VLA_PATHS[$model_idx]}"
        VLA_NAME="${VLA_NAMES[$model_idx]}"

        for mode in "${VPT_MODES[@]}"; do
            for num_prompts in "${VPT_PROMPTS[@]}"; do
                for suite in "${EVAL_SUITES[@]}"; do
                    EXP_NAME="${VLA_NAME}-vpt-${mode}-p${num_prompts}-${suite}"
                    echo ""
                    echo "--- Training: ${EXP_NAME} (${VLA_PATH}) ---"

                    DATA_DIR="${LIBERO_DATA_ROOT}/${suite}"
                    if [ ! -d "$DATA_DIR" ]; then
                        echo "WARNING: Data directory not found: ${DATA_DIR}, skipping..."
                        continue
                    fi

                    $PYTHON vla-scripts/finetune_vpt.py \
                        --vla_path "$VLA_PATH" \
                        --libero_data_dir "$DATA_DIR" \
                        --libero_task_suite "$suite" \
                        --run_root_dir "$RUN_ROOT" \
                        --vpt_mode "$mode" \
                        --vpt_num_prompts "$num_prompts" \
                        --batch_size "$BATCH_SIZE" \
                        --grad_accumulation_steps "$GRAD_ACCUM" \
                        --max_steps "$MAX_STEPS" \
                        --save_steps "$SAVE_STEPS" \
                        --learning_rate "$LEARNING_RATE" \
                        --load_in_8bit "$LOAD_IN_8BIT" \
                        --wandb_project "$WANDB_PROJECT" \
                        --wandb_entity "$WANDB_ENTITY" \
                        --run_id_note "$EXP_NAME"

                    echo "--- Completed training: ${EXP_NAME} ---"
                done
            done
        done
    done
fi

# ==== Evaluation Phase ====
if [ "$TRAIN_ONLY" = false ]; then
    echo ""
    echo "=== EVALUATION PHASE ==="

    RESULTS_FILE="${RUN_ROOT}/ablation_results.txt"
    mkdir -p "$RUN_ROOT"
    echo "VPT Ablation Results" > "$RESULTS_FILE"
    echo "===================" >> "$RESULTS_FILE"
    printf "%-10s %-10s %-10s %-20s %-15s\n" "Model" "Mode" "Prompts" "Suite" "Success Rate" >> "$RESULTS_FILE"
    echo "----------------------------------------------------------------------" >> "$RESULTS_FILE"

    for model_idx in "${!VLA_PATHS[@]}"; do
        VLA_PATH="${VLA_PATHS[$model_idx]}"
        VLA_NAME="${VLA_NAMES[$model_idx]}"

        for mode in "${VPT_MODES[@]}"; do
            for num_prompts in "${VPT_PROMPTS[@]}"; do
                for suite in "${EVAL_SUITES[@]}"; do
                    EXP_NAME="${VLA_NAME}-vpt-${mode}-p${num_prompts}-${suite}"

                    VPT_DIR=$(find "$RUN_ROOT" -name "vpt_checkpoint" -path "*${EXP_NAME}*" 2>/dev/null | head -1)

                    if [ -z "$VPT_DIR" ]; then
                        echo "WARNING: No VPT checkpoint found for ${EXP_NAME}, skipping eval..."
                        printf "%-10s %-10s %-10s %-20s %-15s\n" "$VLA_NAME" "$mode" "$num_prompts" "$suite" "N/A" >> "$RESULTS_FILE"
                        continue
                    fi

                    echo ""
                    echo "--- Evaluating: ${EXP_NAME} (${VLA_PATH}) ---"
                    echo "  VPT weights: ${VPT_DIR}"

                    $PYTHON experiments/robot/libero/run_libero_eval_vpt.py \
                        --model_family openvla \
                        --pretrained_checkpoint "$VLA_PATH" \
                        --vpt_weights_dir "$VPT_DIR" \
                        --task_suite_name "$suite" \
                        --center_crop True \
                        --num_trials_per_task "$NUM_TRIALS" \
                        --load_in_8bit "$LOAD_IN_8BIT" \
                        --run_id_note "eval-${EXP_NAME}"

                    echo "--- Completed evaluation: ${EXP_NAME} ---"
                done
            done
        done
    done

    echo ""
    echo "=== ABLATION RESULTS ==="
    cat "$RESULTS_FILE"
fi

echo ""
echo "All experiments complete!"
