# SocialEvo Sparse-Negative Protocol Audit

Audit date: 2026-09-11.

## Admissible historical negatives

For a query `(u, v, t)`, an admissible historical negative destination must
have appeared strictly before `t` and must not have interacted with source `u`
at or before `t`. Sampling is deterministic in `(seed, source, timestamp,
query_id, policy)`. SocialEvo uses skip-unsampleable evaluation: if no such
destination exists, the query is removed rather than assigned an invalid or
colliding negative.

Training is different and is reported separately. It uses one historical
negative with a collision-free random cold-start fallback. The recorded
SocialEvo training fallback rate is 0.0002862558 (0.028626%) for all five HRCSR
seeds. This training fallback is not the sparse test policy.

## Chronological calibration suffix

All five HRCSR seeds record the same counts:

| Quantity | Count |
|---|---:|
| Requested reserved calibration events | 293,933 |
| Retained calibration events | 286,990 |
| Skipped calibration events | 6,943 |
| Calibration skip rate | 2.362103% |

The calibration suffix receives no gradient updates. Route and identity gains
are evaluated on the same retained calibration queries and deterministic
historical negatives.

## Reconstructed validation/test query population

The executed SocialEvo JSON predates the result-schema fields that persist
test query counters. Therefore the following population was reconstructed from
the hashed processed CSV and the final deterministic sampler semantics; it is
not mislabeled as directly recorded output.

| Split | Requested | Retained | Skipped | Skip rate | Fallbacks | Collision rejections |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 314,930 | 307,953 | 6,977 | 2.215413% | 0 | 20,684,310 |
| Test | 314,924 | 310,119 | 4,805 | 1.525765% | 0 | 20,815,985 |

Collision rejections count source-destination candidates removed because they
had already interacted by the query time; it is not a count of skipped
queries.

## Cross-method population check

- HRCSR route, identity, hard, and soft paths are computed from one fitted run.
  They share the same test split, deterministic query IDs, and sampled
  historical negatives.
- HRCSR and its all-source path use the same transfer/evaluation implementation
  and per-seed sampler construction.
- Strict RASP records the same chronological split, ranking protocol,
  historical negative policy, 20 negatives per positive, and source-training
  budget. It is a separately trained control and may select a different source
  portfolio.
- Target-only T-GAT and GraphMixer also record ranking, historical negatives,
  20 negatives per positive, and the same 70/15/15 timestamp split. However,
  their summaries do not retain per-query counters or adapted checkpoints.
  Exact post-hoc query-ID equality across those separate processes is therefore
  **UNRECORDED** and cannot be repaired without retraining those controls.

## Manuscript consequence

SocialEvo is not merged into the five-dense-target unified main table. It is
reported as a separate complete five-seed sparse-domain validation table.
Transfer-family comparisons (all-source, strict RASP, HRCSR) are treated as the
primary sparse evidence. T-GAT and GraphMixer are labeled contextual
target-only controls with the common recorded high-level protocol but without
a claim of persisted cross-process query-ID equality.

This separation preserves the scientific distinction between protocol
difference and method performance.
