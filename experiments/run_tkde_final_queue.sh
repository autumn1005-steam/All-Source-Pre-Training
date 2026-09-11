#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${WORK_ROOT:-/root/autodl-tmp/tkde-final-work-20260911}
PYTHON_BIN=${PYTHON_BIN:-/root/autodl-tmp/TempoFound-TKDE-runs/rasp-core-78b1779/.venv/bin/python}
PROCESSED_DIR=${PROCESSED_DIR:-/root/autodl-tmp/processed-westb-verified/processed}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/TempoFound-TKDE-runs}
QUEUE_LOG=${QUEUE_LOG:-${RUN_ROOT}/tkde-final-queue.log}

log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$QUEUE_LOG"; }

log "waiting for the corrected three-target matched-width checkpoint audit"
while pgrep -f 'run_checkpoint_boundary_audit.py.*checkpoint-boundary-audit-matched-v2-main' >/dev/null; do
  sleep 60
done

cd "$WORK_ROOT"
if pgrep -f 'run_checkpoint_boundary_audit.py.*checkpoint-boundary-audit-matched-v2-extra' >/dev/null; then
  log "waiting for already-running matched-width UCI/Enron checkpoint audit"
  while pgrep -f 'run_checkpoint_boundary_audit.py.*checkpoint-boundary-audit-matched-v2-extra' >/dev/null; do
    sleep 60
  done
else
  log "starting matched-width UCI/Enron checkpoint audit"
  mkdir -p "${RUN_ROOT}/checkpoint-boundary-audit-matched-v2-extra"
  PYTHONPATH=src "$PYTHON_BIN" scripts/run_checkpoint_boundary_audit.py \
    --config configs/checkpoint_boundary_audit_matched.yaml \
    --checkpoint-template "${RUN_ROOT}/hrcsr-crossdomain-5seed-v1/calib_suite/{target}/seed{seed}/final/rasp/pretrained.pth" \
    --processed-dir "$PROCESSED_DIR" \
    --output-root "${RUN_ROOT}/checkpoint-boundary-audit-matched-v2-extra" \
    --targets uci enron \
    --seeds 0 1 2 3 42 \
    --device cuda \
    --transfer-mode frozen \
    > "${RUN_ROOT}/checkpoint-boundary-audit-matched-v2-extra/runner.log" 2>&1
fi
extra_count=$(find "${RUN_ROOT}/checkpoint-boundary-audit-matched-v2-extra" -type f -name 'seed*.json' | wc -l)
if [[ "$extra_count" -ne 30 ]]; then
  log "ERROR matched-width UCI/Enron audit incomplete: ${extra_count}/30 results"
  exit 1
fi
log "finished matched-width UCI/Enron checkpoint audit"

log "waiting for concurrent SocialEvo profiling/rerun jobs before common-hardware timing"
while pgrep -f 'run_target_only_dataset.py.*socialevo|transfer.py.*sparse-protocol-rerun/socialevo' >/dev/null; do
  sleep 60
done

log "waiting for matched component-ablation jobs before common-hardware timing"
while pgrep -f 'run_component_ablation.py.*component-ablation-matched-v1' >/dev/null; do
  sleep 60
done

log "starting common-hardware efficiency benchmark"
mkdir -p "${RUN_ROOT}/efficiency-common-hardware-v1"
PYTHONPATH=src "$PYTHON_BIN" scripts/run_efficiency_benchmark.py \
  --config configs/hrcsr_matched_control.yaml \
  --processed-dir "$PROCESSED_DIR" \
  --output-root "${RUN_ROOT}/efficiency-common-hardware-v1" \
  --target wikipedia \
  --seed 0 \
  --methods target_only_tgat all_source forward_pair strict_rasp hrcsr \
  --device cuda \
  --timing-repetitions 5 \
  --warmup-repetitions 2 \
  --json-output "${RUN_ROOT}/efficiency-common-hardware-v1/efficiency_results.json" \
  --csv-output "${RUN_ROOT}/efficiency-common-hardware-v1/EFFICIENCY_RESULTS.csv" \
  > "${RUN_ROOT}/efficiency-common-hardware-v1/runner.log" 2>&1
log "finished common-hardware efficiency benchmark"

log "starting synthetic expert-count computational scaling profile"
PYTHONPATH=src "$PYTHON_BIN" scripts/profile_expert_scaling.py \
  --source-counts 1 2 4 6 \
  --dim 64 \
  --batch-size 128 \
  --device cuda \
  --warmups 20 \
  --repetitions 100 \
  --json-output "${RUN_ROOT}/efficiency-common-hardware-v1/expert_scaling.json" \
  --csv-output "${RUN_ROOT}/efficiency-common-hardware-v1/EXPERT_SCALING.csv" \
  >> "${RUN_ROOT}/efficiency-common-hardware-v1/runner.log" 2>&1
log "finished synthetic expert-count computational scaling profile"
