#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${REAL_DATA_ROOT:?Set REAL_DATA_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
mkdir -p "$OUTPUT_ROOT"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUTPUT_ROOT/exit_code.txt"' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u "$REPO_ROOT/scripts/prepare_jodie.py" --root "$REAL_DATA_ROOT"
python -u "$REPO_ROOT/scripts/run_pipeline.py" \
  --config "$REPO_ROOT/configs/rasp_real_initial.yaml" \
  --processed-dir "$REAL_DATA_ROOT/processed" \
  --output-root "$OUTPUT_ROOT" --device cuda:0
