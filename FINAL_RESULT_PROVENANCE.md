# Final Result Provenance

Updated: 2026-09-11. This ledger is authoritative for the TKDE hardening
cycle. A manuscript number is admissible only when it follows:

```text
immutable raw JSON/log -> seed-level record -> versioned aggregation script
-> generated table data -> manuscript table or figure
```

## Frozen evidence

### Dense five-target main results and unified Table VII

- Raw roots: `results/raw/remote/target-only-matched-v1`,
  `graphmixer-target-only-matched-v1`, `hrcsr-ablation-matched-v1`, and
  `hrcsr-crossdomain-5seed-v1`.
- Seed grid: targets Wikipedia, Reddit, MOOC, UCI and Enron; seeds
  `0,1,2,3,42`.
- Aggregator: `scripts/audit_unified_table.py`.
- Seed ledger: `TABLE_VII_PROVENANCE.csv` (125 records).
- Generated aggregate: `results/aggregated/unified_table_from_raw.json`.
- Protocol assertion: `TABLE_VII_CONSISTENCY_CHECK.txt` is `PASS`.
- Split evidence: `results/raw/remote/dataset_split_ids.json`, containing
  server-side dataset CSV hashes and the split-code hash.

No Table VII mean or standard deviation is copied manually. Target-only TGAT
and GraphMixer historical summaries do not embed a config digest, so their
`CodeVersion` cells remain `UNRECORDED`; their recorded metric, split,
negative sampler, checkpoint scope and seeds are complete.

### Authoritative confidence sensitivity

- Raw evidence: the 25 authoritative dense HRCSR `transfer/rasp.json` files,
  including stored chronological window means.
- Recalculation: `scripts/recompute_matched_gate_sensitivity.py` changes only
  the bootstrap confidence quantile; it does not retrain or predict.
- Generated results: `results/aggregated/matched_gate_sensitivity.{json,csv}`
  (75 rows).
- Assertion: `GATE_CONSISTENCY_CHECK.txt` is `PASS` for 25/25 c=0.95 cells.
- Forensic explanation: `REDDIT_GATE_FORENSIC_AUDIT.md`.

The archived `hrcsr-gate-sensitivity-v1` runs changed model width, source
portfolio, parameters and predictions. They are retained as forensic evidence
but excluded from the operating-point sensitivity table.

### Contemporary methods

- Machine-readable evidence: `results/raw/contemporary_baselines/compatibility_evidence.json`.
- Decision audit: `CONTEMPORARY_BASELINE_AUDIT.md`.
- MiNT, BRIDGE and GraphBridge are not numerically imported because their
  prediction unit, temporal setting, target-use boundary, and/or metric do not
  match continuous-time historical-negative MRR.
- CRAFT is a compatible target-only problem family but not a multi-source
  transfer method; its released DyGLib/TGB-Seq protocol is not relabelled as a
  matched HRCSR baseline.

## Completed evidence branches

### Matched-width checkpoint boundary

- Correct config: `configs/checkpoint_boundary_audit_matched.yaml`, including
  `transfer_regret_mode: event_pairwise`.
- Reproduction check: Wikipedia seed 0 validation mode exactly reproduced the
  authoritative identity, route, hard MRR, LCB and gate (zero numerical
  difference).
- Server roots:
  `/root/autodl-tmp/TempoFound-TKDE-runs/checkpoint-boundary-audit-matched-v2-main`
  and `checkpoint-boundary-audit-matched-v2-extra`.
- Aggregator: `scripts/aggregate_checkpoint_boundary.py`.
- Final outputs: `CHECKPOINT_BOUNDARY_RESULTS.csv` and
  `results/aggregated/checkpoint_boundary_matched_v2.json` (75 rows; all
  protocols completed).

The prior run under
`INVALID-checkpoint-boundary-audit-matched-batch-bce-20260911` used
`batch_bce` rather than the main `event_pairwise` objective. It is never used
as matched-width evidence.

### Common-hardware efficiency

- Runner: `scripts/run_efficiency_benchmark.py`.
- Required methods: target-only TGAT, all-source, strict RASP and HRCSR.
- Measurements include parameters, stage wall times, peak allocated/reserved
  GPU memory, GPU-hours, synchronized repeated inference latency and
  throughput.
- Expert-count computational profile: `scripts/profile_expert_scaling.py`,
  `S in {1,2,4,6}`. Synthetic inputs are labelled computational profiling and
  never used for a performance claim.
- Server output root:
  `/root/autodl-tmp/TempoFound-TKDE-runs/efficiency-common-hardware-v1`.

### Component ablation

- Raw roots: `results/raw/remote/component-ablation-matched-v1/{mooc,reddit,uci}`.
- Aggregator: `scripts/aggregate_component_ablation.py`.
- Final outputs: `COMPONENT_ABLATION_RESULTS.csv` and
  `results/aggregated/component_ablation_mrr.json` (90 seed-level rows,
  18 summaries; all 60 newly run cells completed).

## SocialEvo five-seed sparse-domain evidence

- Authoritative HRCSR raw artifacts: the five direct per-seed
  `results/raw/remote/hrcsr-lastfm-socialevo-5seed-v1/calib_suite/socialevo/seed*/transfer/rasp.json`
  files for seeds `0,1,2,3,42`. The stale resumed-run `paired_results.json`,
  which contains only seeds 3 and 42, is intentionally ignored.
- Matched transfer controls: direct per-seed strict-RASP and all-source JSONs
  under the same run root. Target-only T-GAT and GraphMixer each have five
  seed-level summaries.
- Auditor and seed ledger: `scripts/audit_socialevo_5seed.py` and
  `results/socialevo_5seed_audit.csv`.
- Generated aggregates: `results/aggregated/socialevo_5seed_summary.{json,csv}`,
  `results/aggregated/gate_summary_dense_sparse.json`, and the manuscript's
  separate sparse-domain table.
- Artifact and dataset hashes: `results/raw/remote/socialevo_remote_artifact_manifest.json`.
- Protocol evidence: `docs/SOCIALEVO_5SEED_AUDIT.md` and
  `docs/SOCIALEVO_SPARSE_PROTOCOL_AUDIT.md`.

SocialEvo is not pooled into the dense unified table. It uses the separately
audited skip-unsampleable policy. Route, identity, hard, and soft share one
query population within the HRCSR run, and the transfer-family comparison is
matched. Target-only summaries do not preserve exact per-query identifiers,
so T-GAT and GraphMixer are labelled contextual controls rather than asserted
to have auditable cross-process query-ID equality.

## Statistical rule

For every primary HRCSR-Hard minus Strict RASP comparison,
`scripts/audit_unified_table.py` reports mean and median paired difference,
sample standard deviation, 95% paired percentile-bootstrap interval, wins out
of five, and the exact two-sided sign-flip value. With `n=5`, all inferential
quantities are explicitly descriptive.

## Release rule

Any incomplete branch remains absent from the manuscript table or is labelled
`NA`/unrecorded. Negative results and protocol changes are retained. The
matched-width audit, common-hardware efficiency table, component ablation,
consistency scan, LaTeX build, and reviewer attack are now completed in this
release. Submission readiness still depends on the authors' final venue and
anonymization decisions; no unsupported universal-SOTA claim is made.
