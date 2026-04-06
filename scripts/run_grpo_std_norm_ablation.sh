#!/bin/bash
set -euo pipefail

# Ablation: compare advantage normalization with vs. without std normalization.
# All other settings are identical for a fair comparison.
PROJECT="grpo-std-norm-ablation"
LR="1e-5"

for use_std in true false; do
    if [ "${use_std}" = "true" ]; then
        std_flag="--use-std-normalization"
        model_name="grpo_std_norm_lr${LR}"
        std_label="with_std"
    else
        std_flag=""
        model_name="grpo_no_std_norm_lr${LR}"
        std_label="no_std"
    fi

    echo "========================================"
    echo "  Project          : ${PROJECT}"
    echo "  Run              : ${model_name}"
    echo "  Std normalization: ${std_label}"
    echo "  LR               : ${LR}"
    echo "========================================"

    uv run python cs336_alignment/grpo.py \
        ${std_flag} \
        --project-name "${PROJECT}" \
        --model-name "${model_name}"

    echo "  Finished: ${model_name}"
    echo ""
done

echo "All std normalization ablation runs completed."
