#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../cs336_alignment"

for N in 128 256 512 1024; do
    echo "=========================================="
    echo "Running SFT with SAMPLE_NUM=${N}"
    echo "=========================================="
    uv run python sft.py --sample-num "$N"
    echo ""
done

echo "All runs completed."
