# Table VII Provenance Audit

Date: 2026-09-11

## Outcome

The unified dense-target table was rebuilt from 125 seed-level records:
five targets, five in-scope methods, and seeds `0,1,2,3,42`. The regenerated means and
sample standard deviations match the values currently printed in Table VII.
`TABLE_VII_CONSISTENCY_CHECK.txt` is `PASS`.

The authoritative chain is:

```text
results/raw/remote/**/seed*/transfer/*.json or summary.json
  -> scripts/audit_unified_table.py
  -> TABLE_VII_PROVENANCE.csv
  -> results/aggregated/unified_table_from_raw.json
  -> Table VII
```

## Matched protocol fields

Every one of the 125 rows explicitly records or is joined to evidence for:

- metric: MRR;
- chronological split: the target CSV SHA-256, the SHA-256 of the split
  implementation, and quantile boundaries 0.70/0.85;
- evaluation sampler: historical negatives;
- negatives per positive: 20;
- checkpoint policy: target-validation selection after source selection;
- seed: one of `0,1,2,3,42`.

The split join is based on `results/raw/remote/dataset_split_ids.json`. The
dataset hashes were computed on the server-side processed files, not copied
from manuscript prose.

## Method-specific raw roots

| Method | Authoritative raw root | Seed-level metric field |
|---|---|---|
| Target-only T-GAT | `results/raw/remote/target-only-matched-v1` | `test_mrr` |
| Target-only GraphMixer | `results/raw/remote/graphmixer-target-only-matched-v1` | `test_mrr` |
| All-source | dense `calib_suite/*/seed*/transfer/all_sources.json` | `test_mrr` |
| Strict RASP | dense `calib_strict_ranked/*/seed*/transfer/rasp.json` | `test_mrr` |
| HRCSR-Hard | dense `calib_suite/*/seed*/transfer/rasp.json` | hard-gated `test_mrr` |

Wikipedia, Reddit, and MOOC use the `hrcsr-ablation-matched-v1` dense HRCSR
root. UCI and Enron use `hrcsr-crossdomain-5seed-v1`.

## Statistical recomputation

The aggregation also computes HRCSR-Hard minus Strict RASP per target using
the paired seed values. It reports mean, median, sample standard deviation,
95% paired percentile-bootstrap interval, positive-seed count, and the exact
two-sided sign-flip value. With only five pairs, these tests are descriptive.

## Remaining metadata boundary

The target-only T-GAT and GraphMixer summaries do not contain a config digest
or Git commit. Their `CodeVersion` cells therefore remain `UNRECORDED`; the
audit does not back-fill this field from a later code copy. The exact runner
and GraphMixer configuration recovered from the server are preserved under
`scripts/remote_reproductions/` and `configs/remote_reproductions/`, but that
recovery is reproducibility support rather than proof of an embedded code
version in the historical result.

This metadata gap does not change the recorded MRR, split, sampler, seed, or
checkpoint policy. It must remain disclosed in the final provenance ledger.
