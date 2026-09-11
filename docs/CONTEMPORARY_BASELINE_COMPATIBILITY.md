# Contemporary Baseline Compatibility Matrix

Audit date: 2026-09-11. No cross-task performance number is imported.

| Method | Temporal? | Multi-network? | Target supervision? | Historical 20-negative ranking? | Source selection? | Event routing? | Abstention? | Directly reproducible in the matched table? | Reason if not |
|---|---|---|---|---|---|---|---|---|---|
| MiNT | Discrete-time weekly snapshots | Yes | Zero-shot state updates on held-out networks | No; graph-level ROC-AUC | No explicit target-conditioned portfolio | No continuous-time source-response router | No | No | Prediction unit, temporal formalism, target use, decoder, and metric differ. A TGAT/MRR rewrite would be a new adaptation. |
| BRIDGE | Static graphs | Yes | Few-shot prompted downstream classification | No | Bounded multi-domain assembly | MoE-style assembly, not temporal-event routing | No chronological transfer gate | No | No continuous-time event encoder, chronological link-ranking task, or historical-negative evaluator. |
| GraphBridge | Static graphs | Transfer across graph tasks | Target-task fine-tuning | No | No multi-source temporal portfolio | Target-side fusion/side-tuning | No | No | Static adjacency and classification/ROC-AUC protocols are not a drop-in temporal transfer baseline. |
| Target-only T-GAT | Continuous time | No | Target train/validation | Yes in this codebase | No | No | No | Yes, as an architecture/control baseline | It does not test multi-source transfer. |
| Target-only GraphMixer | Continuous time | No | Target train/validation | Yes in this codebase | No | No | No | Yes, as an architecture/control baseline | It does not test multi-source transfer. |
| All-source | Continuous time | Yes | Target adaptation | Yes | No | No | No | Yes | Matched in the common codebase. |
| Strict RASP | Continuous time | Yes | Target-training-only portfolio plus adaptation | Yes | Yes | No | No | Yes | Separately trained source-selected control. |
| HRCSR | Continuous time | Yes | Optimization prefix plus disjoint calibration suffix | Yes | Yes | Yes | Target-level chronological abstention | Yes | “Risk-controlled” is operational and empirical, not a formal guarantee. |

MiNT is closest in motivation but not in estimand. BRIDGE and GraphBridge are
useful conceptual comparators for multi-domain assembly and transfer bias, but
their reported numbers cannot enter an event-level continuous-time MRR table.
The strongest direct controls remain the methods actually executed under the
common chronological ranking pipeline. This compatibility finding supports
careful positioning; it does not imply that HRCSR outperforms the incompatible
methods.
