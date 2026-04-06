#!/bin/bash
set -euo pipefail

# Ablation: compare GRPO (per-sequence mean) vs. Dr.GRPO (batch token mean)
# length normalization. All other settings are identical for a fair comparison.
PROJECT="grpo-length-norm-ablation"

LENGTH_NORMS=(grpo dr.grpo)

for length_norm in "${LENGTH_NORMS[@]}"; do
    # Replace "." with "-" for a cleaner filename-safe model name
    norm_label="${length_norm//./-}"
    model_name="grpo_${norm_label}"
    echo "========================================"
    echo "  Project     : ${PROJECT}"
    echo "  Run         : ${model_name}"
    echo "  Length norm : ${length_norm}"
    echo "========================================"

    uv run python cs336_alignment/grpo.py \
        --length-normalization "${length_norm}" \
        --project-name "${PROJECT}" \
        --model-name "${model_name}"

    echo "  Finished: ${model_name}"
    echo ""
done

echo "All length normalization ablation runs completed."
