#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROCESSED_DIR="${PROCESSED_DIR:?Set PROCESSED_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
TARGETS=(wikipedia reddit)
# calib_suite evaluates route / identity / hard / soft from the SAME fitted
# routed model. calib_strict_ranked is the matched no-source-expert control.
VARIANTS=(calib_strict_ranked calib_suite)

for variant in "${VARIANTS[@]}"; do
  for target in "${TARGETS[@]}"; do
    PYTHONPATH="$ROOT/src:$ROOT" python "$ROOT/scripts/run_calibrated_safety_check.py" \
      --processed-dir "$PROCESSED_DIR" \
      --output-root "$OUTPUT_ROOT" \
      --target "$target" \
      --variant "$variant"
  done
done
