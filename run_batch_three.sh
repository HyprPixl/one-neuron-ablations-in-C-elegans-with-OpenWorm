#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <batch_number>" >&2
  exit 1
fi

BATCH_NUM=$1
if ! [[ $BATCH_NUM =~ ^[0-9]+$ && $BATCH_NUM -ge 1 ]]; then
  echo "Batch number must be a positive integer (1-indexed)." >&2
  exit 1
fi

BATCH=$(sed -n "${BATCH_NUM}p" scripts/neuron_batches_local_canonical_three_batches.txt)
if [[ -z $BATCH ]]; then
  echo "No batch at index $BATCH_NUM" >&2
  exit 1
fi

PREFIX=$(printf 'abl_local_three_b%02d' "$BATCH_NUM")

python3 scripts/single_ablation.py \
  --duration 12 \
  --neurons "$BATCH" \
  --max-neurons 300 \
  --prefix "$PREFIX" \
  --output-root output/single_ablation
