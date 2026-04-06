#!/bin/bash
set -euo pipefail

# Prompt ablation: compare R1-Zero prompt vs. question-only prompt.
# Each prompt type uses its paired reward function (set automatically in grpo.py).
# All other settings are identical for a fair comparison.
#
# Deliverable metrics to observe:
#   - validation answer reward curve
#   - entropy, response length, gradient norm trends
PROJECT="grpo-prompt-ablation"

PROMPT_TYPES=(r1-zero question-only)

for prompt_type in "${PROMPT_TYPES[@]}"; do
    model_name="grpo_${prompt_type}"
    echo "========================================"
    echo "  Project     : ${PROJECT}"
    echo "  Run         : ${model_name}"
    echo "  Prompt type : ${prompt_type}"
    echo "========================================"

    uv run python cs336_alignment/grpo.py \
        --prompt-type "${prompt_type}" \
        --project-name "${PROJECT}" \
        --model-name "${model_name}"

    echo "  Finished: ${model_name}"
    echo ""
done

echo "All prompt ablation runs completed."
