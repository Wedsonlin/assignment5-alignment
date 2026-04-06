#!/bin/bash
set -euo pipefail

# Off-policy GRPO broad sweep (n_grpo_steps=50).
#
# Full grid: epochs_per_rollout_batch x train_batch_size
#   epochs  : 2  4  6  8          (all off-policy → grpo_clip required)
#   train_bs: 32 64 128 256 512 1024
#
# grad_accum = train_bs / MICRO_BS  (MICRO_BS=2, same as default)
# → micro batch size fixed at 2, GPU memory usage stays constant.
#
# Total: 4 x 6 = 24 runs.
# On-policy baseline (epochs=1, train_bs=256) is run separately.

PROJECT="grpo-off-policy-sweep"
N_STEPS=50
MICRO_BS=2

EPOCHS_LIST=(2 4 6 8)
TRAIN_BS_LIST=(32 64 128 256 512)

# On-policy baseline: epochs=1, train_bs=256
echo "========================================"
echo "  Project    : ${PROJECT}"
echo "  Run        : grpo_ep1_bs256  [on-policy baseline]"
echo "  epochs     : 1"
echo "  train_bs   : 256"
echo "  grad_accum : 128  (micro_bs=${MICRO_BS})"
echo "========================================"
uv run python cs336_alignment/grpo.py \
    --n-grpo-steps "${N_STEPS}" \
    --epochs-per-rollout 1 \
    --train-batch-size 256 \
    --gradient-accumulation-steps 128 \
    --loss-type reinforce_with_baseline \
    --project-name "${PROJECT}" \
    --model-name "grpo_ep1_bs256"
echo "  Finished: grpo_ep1_bs256"
echo ""

for EPOCHS in "${EPOCHS_LIST[@]}"; do
    for TRAIN_BS in "${TRAIN_BS_LIST[@]}"; do
        GRAD_ACCUM=$(( TRAIN_BS / MICRO_BS ))
        MODEL="grpo_ep${EPOCHS}_bs${TRAIN_BS}"
        echo "========================================"
        echo "  Project    : ${PROJECT}"
        echo "  Run        : ${MODEL}"
        echo "  epochs     : ${EPOCHS}"
        echo "  train_bs   : ${TRAIN_BS}"
        echo "  grad_accum : ${GRAD_ACCUM}  (micro_bs=${MICRO_BS})"
        echo "========================================"

        uv run python cs336_alignment/grpo.py \
            --n-grpo-steps "${N_STEPS}" \
            --epochs-per-rollout "${EPOCHS}" \
            --train-batch-size "${TRAIN_BS}" \
            --gradient-accumulation-steps "${GRAD_ACCUM}" \
            --loss-type grpo_clip \
            --project-name "${PROJECT}" \
            --model-name "${MODEL}"

        echo "  Finished: ${MODEL}"
        echo ""
    done
done

echo "All off-policy sweep runs completed (${#EPOCHS_LIST[@]} x ${#TRAIN_BS_LIST[@]} = $(( ${#EPOCHS_LIST[@]} * ${#TRAIN_BS_LIST[@]} )) runs)."
