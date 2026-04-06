#!/bin/bash
set -euo pipefail

# Ablation: compare REINFORCE with vs. without group baseline.
# Both runs use the same LR and settings for a fair comparison.
PROJECT="grpo-baseline-ablation"

LOSS_TYPES=(reinforce_with_baseline no_baseline)

for loss_type in "${LOSS_TYPES[@]}"; do
    model_name="grpo_${loss_type}"
    echo "========================================"
    echo "  Project   : ${PROJECT}"
    echo "  Run       : ${model_name}"
    echo "  Loss type : ${loss_type}"
    echo "========================================"

    uv run python cs336_alignment/grpo.py \
        --loss-type "${loss_type}" \
        --project-name "${PROJECT}" \
        --model-name "${model_name}"

    echo "  Finished: ${model_name}"
    echo ""
done

echo "All baseline ablation runs completed."
