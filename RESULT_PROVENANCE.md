# Result Provenance Compatibility Index

Updated: 2026-09-11.

The authoritative ledger is FINAL_RESULT_PROVENANCE.md. A manuscript value is
admissible only through:

    raw result -> seed-level record -> aggregation script -> manuscript table

## Current evidence map

| Evidence | Raw source | Aggregator | Status |
|---|---|---|---|
| Five dense targets | results/raw/remote/{hrcsr-ablation-matched-v1,hrcsr-crossdomain-5seed-v1} | scripts/audit_unified_table.py | complete, five seeds |
| SocialEvo sparse HRCSR/Strict/All-source | direct seed*/transfer/*.json under hrcsr-lastfm-socialevo-5seed-v1 | scripts/audit_socialevo_5seed.py | complete, five seeds |
| SocialEvo T-GAT/GraphMixer | direct seed*/summary.json | scripts/audit_socialevo_5seed.py | complete, five seeds; query IDs unrecorded |
| Dense gate sensitivity | authoritative direct HRCSR JSONs | scripts/recompute_matched_gate_sensitivity.py | complete; c=.95 checks 25/25 |
| Dense + sparse gate panels | direct HRCSR JSONs | scripts/audit_socialevo_5seed.py | complete; panels remain separate |
| Matched checkpoint boundary | checkpoint-boundary-audit-matched-v2-{main,extra} | scripts/aggregate_checkpoint_boundary.py | complete, 75 rows |
| Component ablation | component-ablation-matched-v1 | scripts/aggregate_component_ablation.py | complete, 90 rows |
| Efficiency | efficiency-common-hardware-v1 | efficiency/profile scripts | complete on recorded RTX 5090 |

The stale SocialEvo paired_results.json containing only seeds 3 and 42 is not
authoritative. All five direct per-seed files are used. SocialEvo remains
separate from the dense unified table because it uses the audited
skip-unsampleable policy. Missing fields are labelled UNRECORDED or MISSING;
no value is inferred.
