#!/bin/bash
set -euo pipefail

SAMPLE_NUMS=(128 256 512 1024 0)

for n in "${SAMPLE_NUMS[@]}"; do
    if [ "$n" -eq 0 ]; then
        num_label="all"
    else
        num_label="$n"
    fi
    echo "============================================"
    echo "  Running SFT: filtered, sample_num=${num_label}"
    echo "============================================"
    uv run python cs336_alignment/sft.py --sample_num "$n" --filtered
    echo "  Finished: filtered, sample_num=${num_label}"
    echo ""
done

echo "All runs completed."
