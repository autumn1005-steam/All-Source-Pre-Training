# Fixed-epoch / prefix-only checkpoint audit

## Purpose

The primary HRCSR runs select a target checkpoint by validation MRR.  This is
not part of the gate statistic, but it is a separate information path that a
reviewer may question.  This audit therefore exposes three modes:

* `validation`: existing protocol; validation MRR selects the checkpoint.
* `fixed_epoch`: the predeclared final target-training epoch is used and no
  validation score is used for checkpoint selection.
* `prefix_only`: the checkpoint is selected using only the chronological target
  optimization prefix; the calibration suffix, validation split, and test split
  are not used for selection.

## Reproduction command

```bash
python scripts/run_checkpoint_boundary_audit.py \
  --config configs/checkpoint_boundary_audit.yaml \
  --checkpoint-template /path/to/checkpoints/{target}/seed{seed}/checkpoint.pt \
  --processed-dir /path/to/processed \
  --output-root results/raw/checkpoint-boundary-audit
```

The template must resolve to an existing checkpoint for every target/seed.  A
missing checkpoint is recorded as `PENDING_MISSING_CHECKPOINT`; it is never
replaced by another seed or another protocol.

## Status

The audit harness and metadata fields are implemented and unit-tested.  The
first completed branch is a reduced-width diagnostic under
`checkpoint_boundary_audit.yaml`; its raw JSON files and aggregate are kept
for traceability but are not used as a main performance claim because the
audit width differs from the main HRCSR configuration.  A matched-width branch
using the main 64/32 representation, source budget, negative sampler, and
target budget is running under `checkpoint-boundary-audit-matched-v1/`.
