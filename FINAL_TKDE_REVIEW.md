# TKDE Final Reviewer Simulation

Date: 2026-09-11

This simulation evaluates the frozen dense results and the separately audited
five-seed SocialEvo sparse-domain extension. Scores are deliberately
conservative and do not predict an editorial decision.

## Reviewer 1: Temporal Graph Learning

**Score:** 7/10  
**Confidence:** 4/5

**Top strengths**

1. HRCSR separates source portfolio selection, provenance retention,
   event-response routing, and target-level abstention instead of presenting a
   generic temporal MoE.
2. The primary evaluation uses chronological historical-negative ranking and
   reports route, identity, hard, and strict controls.
3. Complete SocialEvo five-seed evidence adds a sparse, high-interaction regime
   without being pooled into the dense protocol.

**Top weaknesses**

1. The temporal encoder is TGAT-style and the strongest gains are concentrated
   on MOOC and UCI rather than every target.
2. Only five random seeds are available per target.
3. SocialEvo uses a different no-valid-negative policy and the target-only
   summaries do not persist query identifiers.

**Likely major-revision request**

Clarify whether the method's conclusions persist with larger source pools and
newer temporal encoders.

**Likely rejection reason**

If the paper were framed as universal temporal-graph SOTA, the unified table
would contradict that claim. The current conditional-transfer framing avoids
this mismatch.

**Does complete SocialEvo evidence materially strengthen external validity?**

Yes, moderately. It replaces an incomplete two-seed boundary with five direct
seed-level HRCSR, strict-RASP, all-source, T-GAT, and GraphMixer artifacts and
adds a sparse regime with distinct gate behavior. It does not erase the
protocol difference or substitute for a larger number of independent domains.

## Reviewer 2: Transfer Learning and Multi-Source Adaptation

**Score:** 7/10  
**Confidence:** 4/5

**Top strengths**

1. The novelty is a hierarchical decision problem under a chronological
   information boundary, not a list of loosely connected components.
2. Reddit provides an informative negative case: all gates close while the
   expert-identity representation improves over strict RASP, separating
   representation transfer from routed residual acceptance.
3. SocialEvo provides a second useful regime: four gates open and the observed
   gate decisions match the future route-versus-identity signs in all five
   seeds.

**Top weaknesses**

1. The LCB is an empirical operating rule, not a finite-sample risk guarantee.
2. The calibration-to-future relationship is target dependent; Reddit shows
   missed opportunities.
3. MiNT, BRIDGE, and GraphBridge are not direct numerical baselines.

**Likely major-revision request**

Request either a matched contemporary temporal-transfer adaptation or a more
formal account of why the released tasks, supervision boundaries, and metrics
make direct reproduction scientifically invalid. The compatibility matrix
addresses the latter but does not create a new numerical competitor.

**Likely rejection reason**

Overstating “risk-controlled” as guaranteed safety would be fatal. The
manuscript now defines it as an operational accept/reject mechanism and reports
zero observed violations only with finite denominators.

**Does complete SocialEvo evidence materially strengthen external validity?**

Yes, because the central claim is conditional transfer. A sparse target where
HRCSR has a positive but uncertain mean gain and a different gate pattern is
more informative than another selected positive result. The effect remains
descriptive because the paired interval crosses zero.

## Reviewer 3: TKDE Experimental Methodology

**Score:** 7/10  
**Confidence:** 5/5

**Top strengths**

1. SocialEvo table values are regenerated from direct per-seed artifacts; a
   stale two-row aggregate is explicitly excluded.
2. The release records result hashes, checkpoint hashes where available, a
   processed-dataset composite hash, query counters, sparse skip rates, and
   missing fields as UNRECORDED.
3. The matched-width checkpoint audit, gate consistency assertion, component
   ablation, and common-hardware efficiency profile address major information-
   boundary and budget objections.

**Top weaknesses**

1. SocialEvo T-GAT and GraphMixer lack saved per-query identifiers and adapted
   checkpoints, so exact cross-process evaluation-population equality cannot
   be independently reconstructed.
2. Five-seed sign-flip and bootstrap quantities have low resolution and must
   remain descriptive.

**Likely major-revision request**

Persist query IDs and adapted checkpoints in a future artifact revision, then
re-evaluate every SocialEvo baseline on a byte-identical candidate ledger.

**Likely rejection reason**

Pooling SocialEvo into the dense unified table would undermine fairness. The
separate sparse table avoids that error.

**Does complete SocialEvo evidence materially strengthen external validity?**

Yes, subject to the disclosed comparison boundary. The matched transfer-family
panel is valid; the target-only rows are useful contextual controls but not
evidence of exact cross-process query-ID equality.

## Area Editor Synthesis

**Recommendation:** Major Revision / borderline accept after venue-specific
formatting and anonymity checks.

The work now has a coherent scientific position: source utility is
target-conditioned, event routing can fail out of sample, and chronological
abstention provides an auditable empirical control. The complete SocialEvo
branch meaningfully reduces the external-validity concern, but it cannot solve
the remaining contemporary-baseline and formal-risk-guarantee objections.

The largest remaining reject risks are:

1. reviewers may still prefer a directly adapted MiNT-like temporal-transfer
   baseline despite the documented protocol mismatch;
2. the six-domain scale is much smaller than multi-network foundation-model
   benchmarks;
3. the term “risk-controlled” may be misread unless the operational, empirical
   definition remains prominent;
4. SocialEvo target-only query identity remains unrecorded.

These are disclosed scope limitations, not hidden missing results. The paper
should not claim universal SOTA, formal safety, or identical dense/sparse
protocols.
