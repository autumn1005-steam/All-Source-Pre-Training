# TKDE Reject-Risk Audit (2026-09-11)

## Scope and evidence

This audit covers the current IEEEtran manuscript `TGFM_paper.tex`, the v3.1
HRCSR validation package, the available v3 result JSON files, the package
metadata and tests, and the rendered 16-page PDF. No unobserved server result
is treated as completed evidence. The audit is deliberately conservative:
missing evidence is recorded as a gap rather than inferred from trends.

## Critical

### C1. Information-boundary robustness (resolved with scoped caveat)

The current transfer implementation records
`checkpoint_selection_scope=target_validation_only_after_source_selection`
and selects the best checkpoint by validation MRR (`src/tempofound/transfer.py`).
The OOS gate itself uses a disjoint chronological calibration suffix, but no
fixed-epoch or prefix-only checkpoint experiment is present. A reviewer can
therefore ask whether the reported gate behavior depends on indirect validation
selection. The manuscript acknowledges this but only as a limitation.

**Evidence:** matched-width validation, fixed-epoch, and prefix-only runs are
complete for Wikipedia, Reddit, MOOC, UCI, and Enron (75 seed/protocol rows).
The aggregate is `CHECKPOINT_BOUNDARY_RESULTS.csv`; MOOC, UCI, and Reddit
retain their qualitative regimes. Enron is sensitive under fixed epochs, so
the claim is qualitative rather than checkpoint-invariant.

### C2. SocialEvo is evaluated under a different admissible-query rule

The manuscript and protocol audit state that dense targets use collision-free
historical negatives with random cold-start fallback, while SocialEvo skips
queries with no valid negative. The v3.1 code exposes both behaviors through
`sample_with_fallback` and `sampleable_mask`; the current manuscript reports
6,943/293,933 reserved calibration queries skipped for SocialEvo. This is a
protocol difference, not automatically a method gain.

**Disposition:** five direct HRCSR, strict-RASP, all-source, target-only T-GAT,
and target-only GraphMixer seed artifacts are now audited for SocialEvo. The
skip-unsampleable rule remains a real protocol difference, so SocialEvo is
reported in a separate five-seed sparse panel.
Route, identity, hard, and soft share one query set; target-only query IDs are
unrecorded and those rows are labelled contextual controls.

### C3. Reproducibility package (resolved)

The current paper archive contains the manuscript and figures, while the code
and results live in a separate v3.1 directory. There is no single package with
`README`, environment specification, configs, scripts, raw results,
aggregated results, `RUN_ALL.md`, and `EXPERIMENT_MANIFEST.csv`. There is also
no `supplement.tex`; the appendix is embedded in the monolithic main TeX file.

**Evidence:** the release tree contains code, configs, scripts, raw and
aggregated results, `RUN_ALL.md`, `EXPERIMENT_MANIFEST.csv`, the final
provenance ledger, and a compileable manuscript/supplement. Every new table
has a machine-readable aggregation path.

## High

### H1. Contemporary direct baseline is absent (compatibility-resolved)

Related Work discusses MiNT, BRIDGE, and GraphBridge, but the primary table
contains target-only T-GAT/GraphMixer, all-source, strict RASP, and HRCSR only.
No official contemporary temporal-transfer implementation is
run in the available results. Direct comparison is appropriate only after
checking task, temporal, inductive, feature, negative-sampling, budget, and
metric compatibility.

**Disposition:** `CONTEMPORARY_BASELINE_AUDIT.md` documents task, temporal,
inductive, metric, negative-sampling, and target-information mismatches for
MiNT, BRIDGE, GraphBridge, and CRAFT. No cross-task score is imported. This
leaves a comparison boundary, not a fabricated baseline.

### H2. Efficiency evidence (resolved for the recorded hardware)

The manuscript gives asymptotic expert-bank cost (`O(Sd^2)`) and a legacy
runtime discussion, but the requested comparable table of parameters,
trainable parameters, source-search runs, pretraining/adaptation/calibration
time, inference latency, peak memory, and GPU-hours is not backed by a unified
measurement manifest in the package.

**Evidence:** `EFFICIENCY_RESULTS.csv` and `EXPERT_SCALING.csv` report
parameters, search runs, stage times, peak GPU memory, GPU-hours, and warmed
inference timing on one RTX 5090 environment. The profile is hardware-
specific and does not claim universal runtime scaling.

### H3. Main claims rely on five seeds and several small per-target decisions

Five seeds support descriptive mean/std and paired seed differences, but not
strong asymptotic significance. Gate diagnostics have only five target-seed
decisions per target. The manuscript mostly uses cautious wording, but any
future use of “significant”, “safe”, or “prevents negative transfer” would be
reject-prone.

**Status:** addressed in the main table and provenance ledger. Exact paired
sign-flip values and bootstrap intervals are descriptive at n=5; zero observed
violations are reported with denominators and missed-opportunity rates.

### H4. Negative-sampling protocols are not identical across every stage

Primary ranking evaluation uses historical negatives; target training uses
historical negatives with a collision-free random fallback; and SocialEvo uses
a separately audited skip-unsampleable evaluation rule. These differences are
documented in executable audit outputs.

**Action:** make policy, candidate pool, collision boundary, fallback/skip
rule, and negative count explicit in every raw result and aggregation row.

### H5. Legacy stitching risk (resolved)

The current draft uses one primary MRR protocol and excludes unrelated legacy
diagnostic material. This removes the strongest two-paper-stitching risk.

## Medium

### M1. Baseline strength is uneven across targets

The unified table is honest and shows target-dependent winners: strict RASP on
Wikipedia, target-only T-GAT on Reddit/Enron, and HRCSR on MOOC/UCI. This
supports the paper's premise but rules out a universal-superiority claim.
The contemporary baseline gap and lack of TGN/DyGFormer-style controls remain
open questions.

### M2. Gate sensitivity is local, not a global threshold study

The matched recomputation covers five dense targets, five seeds, and
confidence values 0.90/0.95/0.99 without retraining. The c=0.95 projection
matches the authoritative main decisions 25/25. This is operating-point
robustness evidence, not evidence that 0.95 is universally optimal. Window-
count sensitivity remains outside the frozen main claim.

### M3. Local gate diagnostics do not establish calibrated probabilities

The event acceptance value is a learned sigmoid modulated by routing entropy;
the target LCB is a block-bootstrap statistic under temporal dependence. The
current manuscript correctly avoids a formal coverage claim. Future text must
not call either quantity a calibrated probability without a calibration
analysis.

### M4. Package test invocation is fragile outside an editable install

Running `python -m pytest -q` from the validation package without installing
the project fails collection with `ModuleNotFoundError` for `tempofound`; the
README expects `PYTHONPATH=src:.` or an editable install.
This is a packaging/reproducibility issue, not a model failure.

**Action:** make the documented command self-contained (editable install or a
checked-in pytest configuration) and rerun the complete suite from a clean
environment.

### M5. Checkpoint selection dependence is disclosed and quantified

The manuscript states conventional validation MRR selection and that
validation/test outcomes do not enter the gate statistic. The matched-width
audit now quantifies the separation in three checkpoint protocols and records
the Enron sensitivity rather than hiding it.

## Low

### L1. LaTeX layout warnings remain

The latest 13-page IEEEtran build succeeds and has no undefined references or
TODO markers. The log contains ordinary underfull boxes and one 1.96pt
overfull vertical box. Full-page rendering found no clipping or overlap.

### L2. No permanent artifact URL is included

The code-availability section intentionally avoids an unverifiable URL. This
is acceptable for anonymous review, but the final camera-ready package needs
a DOI/repository/archive identifier.

## Novelty and claim consistency assessment

The strongest defensible novelty statement is not “a new MoE” or “guaranteed
safe transfer.” It is a hierarchical selective-transfer protocol that combines
target-conditioned source portfolio selection, source-provenance residual
experts, event-level response routing, and chronological target-training-only
abstention. The paper presents TempoFound as the representation substrate,
RASP as the current target-level selector, and HRCSR as the main proposed
framework.

The current evidence supports target-dependent gains and conservative gate
behavior, but not universal dominance, distribution-free safety, or a claim
that every source expert transfers beneficially.

## Audit verdict

**Current status: Major Revision required before TKDE submission.** The method
story is coherent and the existing results are valuable, but C1--C3 and H1--H4
must be closed or explicitly bounded in the final artifact. No manuscript
number should be changed until those outputs are traceable to raw results.
