#!/bin/bash
set -euo pipefail

# Learning-rate sweep for GRPO (reinforce_with_baseline, GRPO length-norm).
# Six log-spaced LRs covering the typical RL fine-tuning range for 1-3B models.
PROJECT="grpo-lr-sweep"
LRS=(1e-6 3e-6 1e-5 3e-5 1e-4 3e-4)

for lr in "${LRS[@]}"; do
    model_name="grpo_reinforce_lr${lr}"
    echo "========================================"
    echo "  Project : ${PROJECT}"
    echo "  Run     : ${model_name}"
    echo "  LR      : ${lr}"
    echo "========================================"

    uv run python cs336_alignment/grpo.py \
        --lr "${lr}" \
        --project-name "${PROJECT}" \
        --model-name "${model_name}"

    echo "  Finished: ${model_name}"
    echo ""
done

echo "All LR sweep runs completed."
