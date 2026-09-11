#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROCESSED_DIR="${PROCESSED_DIR:?Set PROCESSED_DIR to the corrected processed-data directory}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/hrcsr_strict_ablation}"
TARGETS=(wikipedia reddit)
VARIANTS=(strict_ranked strict_source_ranked rcsr_no_regret rcsr rcsr_event hrcsr)
for target in "${TARGETS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    echo "=== target=$target variant=$variant ==="
    PYTHONPATH="$ROOT/src:$ROOT" python "$ROOT/scripts/run_three_seed_check.py" \
      --processed-dir "$PROCESSED_DIR" \
      --output-root "$OUTPUT_ROOT/$variant" \
      --target "$target" \
      --variant "$variant"
  done
done
