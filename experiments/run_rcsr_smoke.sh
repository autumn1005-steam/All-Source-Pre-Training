#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROCESSED_DIR:-$ROOT/.smoke_data_rcsr}"
OUT_DIR="${OUTPUT_ROOT:-$ROOT/.smoke_runs_rcsr}"
DEVICE="${DEVICE:-cpu}"
PYTHONPATH="$ROOT/src:$ROOT" python "$ROOT/scripts/generate_smoke_data.py" --output "$DATA_DIR"
PYTHONPATH="$ROOT/src:$ROOT" python "$ROOT/scripts/run_pipeline.py" \
  --config "$ROOT/configs/rcsr_smoke.yaml" \
  --processed-dir "$DATA_DIR" --output-root "$OUT_DIR" --device "$DEVICE"
