#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SOURCES:?Set SOURCES to the space-separated source pool}"
: "${SCORES_JSON:?Set SCORES_JSON to a JSON map such as a+b -> proxy score}"
read -r -a SOURCE_ARRAY <<< "${SOURCES}"
python "${REPO_ROOT}/scripts/control_candidates.py" \
  --method standard-greedy --sources "${SOURCE_ARRAY[@]}" \
  --scores-json "${SCORES_JSON}" --output "${OUTPUT:-standard_greedy_selection.json}"
