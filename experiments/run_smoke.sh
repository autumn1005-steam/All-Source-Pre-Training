#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_WORKDIR="${SMOKE_WORKDIR:-$(mktemp -d)}"
PROCESSED_DIR="${SMOKE_WORKDIR}/processed"
OUTPUT_ROOT="${SMOKE_WORKDIR}/output"
python "${REPO_ROOT}/scripts/generate_smoke_data.py" --output "${PROCESSED_DIR}"
python "${REPO_ROOT}/scripts/run_pipeline.py" \
  --config "${REPO_ROOT}/configs/smoke.yaml" \
  --processed-dir "${PROCESSED_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --device "${DEVICE:-cpu}"
python - "${OUTPUT_ROOT}/target/seed42/summary.json" <<'PY'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert summary["candidate_count"] == 7, summary
assert len(summary["selected_sources"]) in {1, 2}, summary
print(f"Smoke test passed: {sys.argv[1]}")
PY
