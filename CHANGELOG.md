# Changelog

## 2026-09-11 — TKDE hardening

- reconciled the five dense-target HRCSR table against copied remote paired
  JSONs; the initial two-seed SocialEvo label was superseded after all five
  direct seed artifacts were audited;
- added validation, fixed-epoch, and optimization-prefix-only checkpoint
  selection modes plus an auditable boundary runner;
- added ranking query counters for valid/skipped/fallback/collision events;
- added sparse-protocol and efficiency audit scripts and machine-readable
  provenance/manifest ledgers;
- added a unified baseline provenance note and a standalone supplement file;
- retained incompatible contemporary baselines as an explicit compatibility
  boundary and measured the common hardware profile rather than imputing values.
- fixed-epoch runs now emit null checkpoint-selection scores instead of a
  validation sentinel, and ranking outputs expose the evaluator fallback
  policy for sparse-domain auditing.
- package-boundary checks now allow curated JSON/CSV result ledgers and figure
  assets while continuing to reject checkpoints, tensor dumps, caches, logs,
  and nested archives.
- copied and aggregated the matched five-target/five-seed gate-confidence
  sweep (`c=0.90, 0.95, 0.99`) with a 25/25 c=.95 consistency check;
- completed the matched-width checkpoint-boundary audit (75 rows), common-
  hardware efficiency profile, expert-count scaling profile, and 60-cell
  MOOC/Reddit/UCI component ablation;
- updated the manuscript, supplement, provenance ledgers, consistency report,
  and reviewer attack to retain negative and protocol-boundary results.
- rebuilt SocialEvo from five direct seed-level HRCSR, strict-RASP, all-source,
  T-GAT, and GraphMixer artifacts and excluded the stale two-row resumed-run
  aggregate;
- added the SocialEvo raw-artifact/hash ledger, sparse-query reconstruction,
  separate five-seed sparse-domain table, and dense/sparse gate panels;
- revised abstract, setup, results, discussion, limitations, conclusion, and
  supplement so SocialEvo strengthens external validity without being pooled
  into the dense unified protocol.
- removed the method associated with a separate concurrent submission from
  this manuscript, supplement, unified tables, provenance ledgers, and final
  release evidence.
