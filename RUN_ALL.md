# Reproduction workflow

The commands below are ordered so that every derived artifact can be traced to
raw outputs.  Paths are placeholders and must point to the released data and
checkpoint locations.

```bash
python -m pip install -e .
python scripts/prepare_jodie.py --help
python scripts/pretrain.py --config configs/hrcsr_matched_control.yaml \
  --processed-dir /path/to/processed --output-root results/raw/pretraining
python scripts/run_pipeline.py --config configs/hrcsr_matched_control.yaml \
  --processed-dir /path/to/processed --output-root results/raw/hrcsr-main
python scripts/run_checkpoint_boundary_audit.py \
  --config configs/checkpoint_boundary_audit.yaml \
  --checkpoint-template '/path/to/{target}/seed{seed}/pretrained.pth' \
  --processed-dir /path/to/processed \
  --output-root results/raw/checkpoint-boundary-audit
python scripts/aggregate_checkpoint_boundary.py \
  --root results/raw/checkpoint-boundary-audit \
  --output results/aggregated/checkpoint_boundary_audit.json
python scripts/audit_sparse_protocol.py \
  --root results/raw/sparse-rerun --output results/aggregated/sparse_audit.json
python scripts/aggregate_gate_sensitivity.py \
  --root results/raw/remote/hrcsr-gate-sensitivity-v1 \
  --output results/aggregated/gate_sensitivity_3seed.json
python scripts/aggregate_mrr.py \
  --strict-root results/raw/hrcsr-main \
  --hrcsr-root results/raw/hrcsr-main \
  --output results/aggregated/hrcsr_mrr.json
python scripts/profile_efficiency.py \
  --root results/raw/efficiency --output results/aggregated/efficiency.json
python scripts/run_efficiency_benchmark.py --help
python scripts/profile_expert_scaling.py --help
python scripts/aggregate_component_ablation.py \
  --ablation-root 'results/raw/remote/component-ablation-matched-v1/{target}' \
  --output results/aggregated/component_ablation_mrr.json
python scripts/aggregate_checkpoint_boundary.py \
  --root results/raw/remote/checkpoint-boundary-audit-matched-v2-main \
  --extra-root results/raw/remote/checkpoint-boundary-audit-matched-v2-extra \
  --output results/aggregated/checkpoint_boundary_matched_v2.json
python scripts/audit_socialevo_5seed.py \
  --raw-root results/raw/remote \
  --audit-csv results/socialevo_5seed_audit.csv \
  --summary-json results/aggregated/socialevo_5seed_summary.json \
  --summary-csv results/aggregated/socialevo_5seed_summary.csv \
  --gate-panels-json results/aggregated/gate_summary_dense_sparse.json
python -m pytest -q
```

The package contains five direct SocialEvo seed artifacts for HRCSR, strict
RASP, all-source, T-GAT, and GraphMixer. The SocialEvo auditor intentionally
ignores a stale resumed-run aggregate containing only seeds 3 and 42.
SocialEvo is reported in a separate sparse-domain table because its
skip-unsampleable policy differs from the dense logged-fallback policy.
MiNT, BRIDGE, GraphBridge, and CRAFT are documented in the compatibility audit
because their released tasks or metrics do not support a direct matched
comparison.
