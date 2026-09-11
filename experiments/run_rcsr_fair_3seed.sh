#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PROCESSED_DIR:?set PROCESSED_DIR to processed dataset directory}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to output directory}"
DEVICE="${DEVICE:-cuda:0}"
PYTHONPATH="$ROOT/src:$ROOT" python "$ROOT/scripts/run_pipeline.py" \
  --config "$ROOT/configs/rcsr_fair_3seed.yaml" \
  --processed-dir "$PROCESSED_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --device "$DEVICE"
