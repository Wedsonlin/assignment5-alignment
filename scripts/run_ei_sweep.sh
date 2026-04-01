#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../cs336_alignment"

for G in 2 3 4 5; do
    for EPOCH in 1 2 3; do
        echo "=========================================="
        echo "Running EI with G=${G}, epoch_size=${EPOCH}"
        echo "=========================================="
        uv run python ei.py --G "$G" --epoch-size "$EPOCH"
        echo ""
    done
done

echo "All EI sweep runs completed."
