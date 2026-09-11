#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PROCESSED_DIR:?Set PROCESSED_DIR to the directory containing ml_<dataset> files}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/server_runs/rasp_k4}"
DEVICE="${DEVICE:-cuda:0}"
python "${REPO_ROOT}/scripts/run_pipeline.py" \
  --config "${REPO_ROOT}/configs/rasp_k4.yaml" \
  --processed-dir "${PROCESSED_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --device "${DEVICE}"
