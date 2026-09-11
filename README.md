# HRCSR: Hierarchical Risk-Controlled Source Routing

This repository is the unified implementation for the manuscript on
hierarchical risk-controlled source routing in cross-dataset temporal graph
pre-training. The RASP path adds uncertainty-calibrated portfolio search,
target-affinity compute allocation, conflict-aware optimization, and
fixed-negative ranking evaluation.

## What is unified

- one TempoFound model for K=4 and K=6 experiments;
- dataset-specific node/edge adapters and a shared causal TGAT backbone;
- MTLP, TNP, EDC, NPP, TIP, and domain-adversarial learning;
- risk-adjusted multi-window source portfolio search beyond anchor-only pairs;
- target-affinity weighted batches and projected conflicting source gradients;
- collision-free temporal negatives and fixed-negative MRR evaluation;
- exact global per-epoch update budgets across source sets;
- deterministic target-conditioned source screening with a fixed candidate budget;
- frozen, full, and scratch transfer protocols;
- a GraphMixer-style backbone and Random-7/exhaustive/greedy controls;
- paper `paper_sincos` and released-checkpoint `legacy_cos` time modes.

New training defaults to the normalized paired sine/cosine encoding described
in the manuscript. Released TempoFound checkpoint payloads without format
metadata are loaded as `legacy_cos`; the TGAT parameter names and shapes are
kept compatible.
Legacy payloads with four adaptive task weights are migrated deterministically:
the four stored values are preserved and the added NPP weight starts at zero.
Version-2 five-weight checkpoints are remapped explicitly: the former domain
classification weight is not reinterpreted as EDC, and the new TIP weight starts
at zero. New checkpoints record the objective names and format version 3.

## Install

```bash
python -m pip install -e '.[test]'
```

Processed data use the original layout:

```text
processed/
├── ml_mooc.csv
├── ml_mooc.npy
└── ml_mooc_node.npy
```

The CSV requires `u`, `i`, `idx`, and `ts` columns. Row zero in feature arrays
is reserved for padding.

## Verify locally

```bash
python -m pytest -q
bash experiments/run_smoke.sh
bash experiments/run_rasp_smoke.sh
python scripts/check_package.py
```

The smoke commands create deterministic temporary graphs. Their metrics are
engineering checks and must never be reported as manuscript results.

## Run manuscript protocols

```bash
PROCESSED_DIR=/path/to/processed \
OUTPUT_ROOT=/path/to/runs/k4 \
DEVICE=cuda:0 \
bash experiments/run_k4_primary.sh

PROCESSED_DIR=/path/to/processed \
OUTPUT_ROOT=/path/to/runs/rasp-k4 \
DEVICE=cuda:0 \
bash experiments/run_rasp_k4.sh

PROCESSED_DIR=/path/to/processed \
OUTPUT_ROOT=/path/to/runs/k6 \
DEVICE=cuda:0 \
bash experiments/run_k6_scalability.sh
```

The pipeline resumes only artifacts whose target, seed, source set, and full
configuration digest match. Stale outputs are recomputed. Selection artifacts
record all proxy scores, the anchor, pair gain, margin, source commits, and the
target-training-only information boundary.

See `README_CN.md`, `docs/PAPER_CODE_MAPPING.md`, and
`docs/SERVER_ACCEPTANCE_CN.md` for the full handoff.

## TKDE hardening artifacts

`main.tex`, `supplement.tex`, and `figures/` contain the manuscript package.
`RESULT_PROVENANCE.md`, `RESULT_CONSISTENCY_REPORT.md`,
`TKDE_REJECT_RISK_AUDIT.md`, and `EXPERIMENT_MANIFEST.csv` define the evidence
boundary.  `RUN_ALL.md` gives the end-to-end commands.  The checkpoint-boundary
and sparse-protocol scripts explicitly record missing inputs as pending and do
not invent results.

## v3 out-of-sample safety calibration

See `docs/OOS_SAFETY_CALIBRATION_V3_CN.md`. v3 adds a disjoint chronological target-train calibration suffix, reciprocal-rank-aligned gate calibration, path-neutral target fitting, and same-model route/identity/hard/soft gate sweeps.
