# SocialEvo Five-Seed Raw-Result Audit

Audit date: 2026-09-11. Authoritative seeds: 0, 1, 2, 3, and 42.

## Decision

The five direct HRCSR, strict-RASP, all-source, target-only T-GAT, and
target-only GraphMixer result artifacts are complete. These are the in-scope
methods used by the SocialEvo sparse-domain table.

The file
`calib_suite/socialevo/paired_results.json` contains only seeds 3 and 42 because
it was overwritten by the resume command. It is retained as forensic evidence
but excluded from every new aggregate. The authoritative aggregation reads the
five direct `seed*/transfer/rasp.json` files.

## Coverage

| Method / path | Seeds | Coverage | Checkpoint hash |
|---|---:|---|---|
| HRCSR-Hard, `hrcsr-lastfm-socialevo-5seed-v1/calib_suite/socialevo/seed*/transfer/rasp.json` | 0,1,2,3,42 | complete | recorded per seed |
| Strict RASP, `hrcsr-lastfm-socialevo-5seed-v1/calib_strict_ranked/socialevo/seed*/transfer/rasp.json` | 0,1,2,3,42 | complete | recorded per seed |
| All-source, `hrcsr-lastfm-socialevo-5seed-v1/calib_suite/socialevo/seed*/transfer/all_sources.json` | 0,1,2,3,42 | complete | recorded per seed |
| Target-only T-GAT, `socialevo-target-only-tgat/socialevo/seed*/summary.json` | 0,1,2,3,42 | complete | **UNRECORDED**; runner retained summaries only |
| Target-only GraphMixer, `socialevo-target-only-graphmixer/socialevo/seed*/summary.json` | 0,1,2,3,42 | complete | **UNRECORDED**; runner retained summaries only |

Exact local/remote result paths, UTC modification times, result hashes,
checkpoint hashes, configuration digests, selected sources, protocol fields,
and seed-level metrics are in `results/socialevo_5seed_audit.csv`. Full remote
hash metadata are in
`results/raw/remote/socialevo_remote_artifact_manifest.json`.

The composite processed-dataset hash over `ml_socialevo.csv`,
`ml_socialevo.npy`, and `ml_socialevo_node.npy` is:

`ec07657ea3e7d25c008f65a03b76d808c2f71ef366527d3e1f2a387943980b93`.

## Configuration-digest forensic result

HRCSR seeds 0/1/2 record digest
`9c5fb89135999b90f33dcdc78ad15d56eba7904d01c43201573f5981da91032f`,
whereas resumed seeds 3/42 record
`5923860d83e56111ce93d6be9e80e843282e53669598a8ebd105ee7708aa9d46`.
Remote recomputation with the executed code reproduced the first digest from
the original five-seed run list and the second from the resumed `(3, 42)` run
list. `ExperimentConfig` hashes the whole run-level seed tuple; the model,
training, selection, evaluation, and gate hyperparameters are otherwise the
same. Every candidate referenced by the final seed-3 and seed-42
`selection.json` has the resumed digest. Two unused seed-3 candidate
directories (`portfolio_enron_uci` and `portfolio_lastfm_uci`) retain the old
digest and are ignored.

## Seed-level results

| Seed | Sources | Strict RASP | Identity | Route | HRCSR-Hard | Hard - Strict | Gate / LCB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | MOOC | 0.796791 | 0.788352 | 0.797116 | 0.797116 | +0.000325 | 1 / 0.003143 |
| 1 | Reddit | 0.746220 | 0.786798 | 0.812593 | 0.812593 | +0.066373 | 1 / 0.023135 |
| 2 | Reddit | 0.802271 | 0.790305 | 0.797327 | 0.797327 | -0.004944 | 1 / 0.004119 |
| 3 | Reddit | 0.803268 | 0.767886 | 0.782264 | 0.782264 | -0.021004 | 1 / 0.011387 |
| 42 | LastFM + Reddit | 0.792892 | 0.819999 | 0.819999 | 0.819999 | +0.027106 | 0 / 0.000000 |

## Five-seed aggregates

| Method | Mean MRR | Sample std |
|---|---:|---:|
| Target-only T-GAT | 0.505213 | 0.070599 |
| Target-only GraphMixer | 0.586334 | 0.048796 |
| All-source | 0.782312 | 0.012549 |
| Strict RASP | 0.788288 | 0.023891 |
| HRCSR-Hard | **0.801860** | 0.014758 |

For HRCSR-Hard minus strict RASP, the mean is +0.013571, median +0.000325,
sample standard deviation 0.034223, paired-bootstrap 95% interval
[-0.010314, 0.042813], wins 3/5, and exact two-sided sign-flip p=0.4375.
These n=5 statistics are descriptive and do not establish statistical
significance.

Against all-source, HRCSR-Hard has a mean difference of +0.019548 and wins in
5/5 seeds. The target-only controls are reported only in the separately audited
sparse panel because their executed summaries do not retain per-query IDs or
post-adaptation checkpoints for matched re-evaluation.

## Gate diagnostics

- Gate open: 4/5.
- Sign-decision accuracy: 5/5.
- Observed safety violations (false opens): 0/5.
- Observed missed opportunities (false closes): 0/5.
- Mean and maximum selector regret: 0.
- Spearman correlation between calibration LCB and future route-minus-identity
  gain: 0.9.

These are observed diagnostics, not a distribution-free safety guarantee.
