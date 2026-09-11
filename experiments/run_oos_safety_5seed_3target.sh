#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROCESSED_DIR="${PROCESSED_DIR:?Set PROCESSED_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
TARGETS=(wikipedia reddit mooc)
VARIANTS=(calib_strict_ranked calib_suite calib_route calib_identity)
SEEDS=(0 1 2 3 42)
CONFIG="${CONFIG:-$ROOT/configs/rasp_real_initial.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
for variant in "${VARIANTS[@]}"; do
  for target in "${TARGETS[@]}"; do
    PYTHONPATH="$ROOT/src:$ROOT" "$PYTHON_BIN" "$ROOT/scripts/run_calibrated_safety_check.py" \
      --config "$CONFIG" \
      --processed-dir "$PROCESSED_DIR" \
      --output-root "$OUTPUT_ROOT" \
      --target "$target" \
      --variant "$variant" \
      --seeds "${SEEDS[@]}"
  done
done
PYTHONPATH="$ROOT/src:$ROOT" "$PYTHON_BIN" "$ROOT/scripts/analyze_oos_safety.py" \
  --output-root "$OUTPUT_ROOT" \
  --targets "${TARGETS[@]}" \
  --write "$OUTPUT_ROOT/oos_safety_decision_report.json"
