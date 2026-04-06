#!/bin/bash
set -euo pipefail

# Master script: run all GRPO ablation experiments in sequence.
#
# Order:
#   1. baseline ablation       (reinforce_with_baseline vs no_baseline)
#   2. length norm ablation    (GRPO vs Dr.GRPO)
#   3. std norm ablation       (with vs without std normalization)
#   4. off-policy sweep        (epochs x train_batch_size grid)
#   5. prompt ablation         (r1-zero vs question-only)
#
# All child scripts use set -euo pipefail, so any failure will propagate
# and stop this master script immediately.

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_TIME=$(date +%s)

run_script() {
    local name="$1"
    local path="${SCRIPTS_DIR}/$1"
    echo ""
    echo "████████████████████████████████████████████████████"
    echo "  Starting : ${name}"
    echo "  Time     : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "████████████████████████████████████████████████████"
    bash "${path}"
    echo "  Finished : ${name}  ($(date '+%Y-%m-%d %H:%M:%S'))"
}

run_script "run_grpo_baseline_ablation.sh"
run_script "run_grpo_length_norm_ablation.sh"
run_script "run_grpo_std_norm_ablation.sh"
run_script "run_grpo_off_policy_sweep.sh"
run_script "run_grpo_prompt_ablation.sh"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo ""
echo "████████████████████████████████████████████████████"
echo "  All ablation runs completed."
echo "  Total elapsed: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "████████████████████████████████████████████████████"
