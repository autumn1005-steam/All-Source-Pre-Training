#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${PROCESSED_DIR:?Set PROCESSED_DIR to the directory containing ml_<dataset> files}"
python "${REPO_ROOT}/scripts/run_pipeline.py" \
  --config "${REPO_ROOT}/configs/graphmixer_control.yaml" \
  --processed-dir "${PROCESSED_DIR}" \
  --output-root "${OUTPUT_ROOT:-${REPO_ROOT}/server_runs/graphmixer_control}" \
  --device "${DEVICE:-cuda:0}"
