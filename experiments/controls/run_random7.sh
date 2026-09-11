#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SOURCES:?Set SOURCES to the space-separated source pool}"
read -r -a SOURCE_ARRAY <<< "${SOURCES}"
python "${REPO_ROOT}/scripts/control_candidates.py" \
  --method random7 --sources "${SOURCE_ARRAY[@]}" \
  --seed "${SEED:-42}" --output "${OUTPUT:-random7_candidates.json}"
