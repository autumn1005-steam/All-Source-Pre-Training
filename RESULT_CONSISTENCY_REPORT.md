# Result consistency report (final audit, 2026-09-11)

This final machine-assisted pass checks the manuscript, supplement, generated
tables, manifests, and provenance ledgers. It does not silently rewrite any
result.

## Checks completed

| Check | Outcome |
|---|---|
| Main TeX builds with `latexmk -pdf -interaction=nonstopmode` | PASS (13 pages) |
| Automated test suite | PASS: 178 passed, 1 skipped (bash unavailable on Windows) |
| Undefined references/TODO markers in latest log | PASS (none observed) |
| Primary metric and historical-negative protocol are consistent | PASS: MRR with 20 historical negatives |
| Seed count stated for completed dense HRCSR table | PASS: five seeds (0,1,2,3,42) |
| SocialEvo five-seed raw coverage | PASS: HRCSR, strict RASP, all-source, T-GAT, and GraphMixer are 5/5 |
| SocialEvo sparse patch disclosed | PASS; separate five-seed sparse table, not pooled with dense results |
| SocialEvo direct-seed aggregation | PASS; stale two-row `paired_results.json` is ignored |
| Dense/sparse gate panels | PASS: 25 dense + 5 sparse decisions, reported separately and descriptively combined |
| Dense main raw JSON reproduces tables | PASS; Table VII has 125 seed-level provenance rows |
| OOS safety and gate sensitivity | PASS; c=.95 projection matches 25/25 main decisions |
| Unified baseline raw JSONs are bundled | PASS for the five dense targets and five in-scope methods |
| Fixed-epoch and prefix-only checkpoint audit exists | PASS; matched-width 75 rows complete |
| Efficiency values have a unified provenance manifest | PASS; common RTX 5090 profile and expert scaling CSV bundled |
| Component ablation | PASS; 90 seed-level rows and 18 summaries bundled |
| Out-of-scope concurrent-submission method absent | PASS: no mention in main TeX, supplement TeX, or rendered PDFs |

## Historical branches retained but excluded

1. Older three-seed and reduced-width branches remain under `results/raw` and
   are not mixed into the dense five-seed main tables.
2. The invalid batch-BCE checkpoint audit is retained under its `INVALID-`
   directory and excluded because it does not match the event-pairwise loss.
3. The archived confidence-sensitivity rerun is retained as forensic evidence;
   only the matched no-retraining sweep enters the manuscript.
4. The stale SocialEvo two-row resumed-run aggregate is excluded. The five
   direct seed artifacts are authoritative; target-only per-query identifiers
   are unrecorded.

## Interpretation

These are traceability findings, not conclusions about model quality.  The
main dense HRCSR values now have a copied raw-to-aggregation chain.  Any result
without a raw-to-table chain remains `UNVERIFIED` in the final manifest.

## Final disposition

No unresolved numerical conflict was found among the authoritative dense
tables, SocialEvo sparse panel, sensitivity sweep, checkpoint audit,
efficiency profile, and component ablation. All historical or incompatible
branches are explicitly labelled and excluded from unsupported claims.
Remaining limitations are scientific scope (five seeds, empirical rather than
formal risk control, a distinct sparse-query policy, and hardware-specific
efficiency), not untracked result conflicts.
