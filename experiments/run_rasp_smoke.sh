#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_WORKDIR="${SMOKE_WORKDIR:-$(mktemp -d)}"
PROCESSED_DIR="${SMOKE_WORKDIR}/processed"
OUTPUT_ROOT="${SMOKE_WORKDIR}/rasp-output"
python "${REPO_ROOT}/scripts/generate_smoke_data.py" --output "${PROCESSED_DIR}"
python "${REPO_ROOT}/scripts/run_pipeline.py" \
  --config "${REPO_ROOT}/configs/rasp_smoke.yaml" \
  --processed-dir "${PROCESSED_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --device "${DEVICE:-cpu}"
python - "${OUTPUT_ROOT}/target/seed42/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert summary["method"] == "rasp", summary
assert summary["evaluation_protocol"] == "ranking", summary
assert 3 <= summary["candidate_count"] <= 6, summary
assert 1 <= len(summary["selected_sources"]) <= 2, summary
print(f"RASP synthetic acceptance passed (NON-PAPER): {sys.argv[1]}")
PY
