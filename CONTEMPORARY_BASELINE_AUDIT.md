# Contemporary Baseline Compatibility Audit

Audit date: 2026-09-11. This audit determines whether an official result can enter the matched continuous-time temporal link-ranking table. It does not infer or copy performance numbers across tasks.

## Eligibility rule

A direct main-table baseline must support the same scientific estimand: cross-domain transfer to a held-out continuous-time interaction network, chronological target split, event-level future-link ranking, 20 historical collision-filtered negatives per positive, MRR, and the same target-information boundary. A method that changes the prediction unit, temporal representation, target supervision, or metric is not a numerical control for HRCSR.

## Candidate assessment

| Candidate | Official setting | Transfer and target use | Metric / negatives | Official code / license | Directly eligible? |
|---|---|---|---|---|---|
| MiNT (NeurIPS 2025) | Discrete-time dynamic graphs formed as weekly snapshots; graph-level binary prediction of future network-property growth/shrinkage; HTGN and GC-LSTM backbones | Multi-network pretraining followed by zero-shot network-transferred inference; train/validation histories update recurrent state but the target model is not adapted under HRCSR's optimization-prefix/calibration-suffix protocol | ROC-AUC; no event-level 20-historical-negative ranking interface | [official paper](https://papers.nips.cc/paper_files/paper/2025/hash/c5548168cb7324f714365a971dfe76d1-Abstract-Datasets_and_Benchmarks_Track.html), [official code](https://github.com/ntgbaoo/ScalingTGNs), MIT; inspected HEAD `b60a300fc930b6d13e3d4543fa945c04800b24ea` | **No direct numerical comparison.** Prediction unit, temporal formalism, target use, decoder, and metric differ. |
| BRIDGE (ICML 2025) | Static multi-domain graph foundation model for node and graph classification with prompting and domain-invariant alignment | Multi-domain pretrain-then-prompt; downstream few-shot classification | Classification accuracy / ROC-AUC depending on dataset; no chronological temporal sampler | [official paper](https://proceedings.mlr.press/v267/yuan25h.html), [official code](https://github.com/RingBDStack/BRIDGE), MIT | **No.** It has no temporal event encoder, chronological link-ranking task, or historical-negative evaluator. |
| GraphBridge (ICLR 2025) | Static graph transfer across Graph2Graph, Node2Node, Graph2Node and graph-to-point-cloud tasks; supplementary static Graph2Edge | Source GNN plus target-side/backup side-tuning; requires target-task fine-tuning | Accuracy or ROC-AUC; supplementary edge prediction samples binary positives/negatives rather than historical temporal candidates | [official paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/62a04ae910be5679887f81417b765203-Abstract-Conference.html), [official code](https://github.com/jujulili888/GraphBridge); no license file was visible in the inspected repository | **No.** Static adjacency and side-tuning cannot be evaluated as a drop-in continuous-time transfer baseline without designing a new method. |
| CRAFT (NeurIPS 2025) | Continuous-time future-link prediction, including Wikipedia, Reddit, MOOC, LastFM and UCI | Single-network future-link predictor, not multi-source transfer | Future-link ranking, but the release uses its own DyGLib/TGB-Seq data path and BPR/BCE training protocol rather than HRCSR's target optimization/calibration boundary | [official paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/036912a83bdbb1fd792baf6532f102d8-Abstract-Conference.html), [official code](https://github.com/luyi256/CRAFT), MIT | **Related but not a transfer baseline.** It is eligible only as a separately labelled target-only architecture control after a matched sampler/data adaptation; it cannot answer the source-transfer question. |

## MiNT forensic compatibility decision

MiNT is the closest contemporary work by scientific motivation, but its released task predicts a property of an entire weekly graph snapshot. The paper explicitly limits the method to discrete-time dynamic graphs, reports graph-level ROC-AUC, and performs zero-shot inference on held-out networks. HRCSR instead predicts each future continuous-time interaction, adapts on a target optimization prefix, calibrates on a disjoint target-training suffix, and ranks the positive edge against 20 admissible historical destinations. Consequently:

1. an official MiNT number cannot be placed in the HRCSR MRR table;
2. replacing MiNT's snapshot target, HTGN/GC-LSTM encoder, graph pooling and AUC decoder with TGAT event ranking would be a new adaptation, not an official reproduction;
3. calling the existing all-source TGAT run “MiNT” would be mislabeling and is prohibited;
4. an original-setting MiNT reproduction would validate its own benchmark but would not supply a matched HRCSR comparison.

## Decision for the manuscript

No MiNT, BRIDGE or GraphBridge performance number is admitted to the main matched table. The manuscript will state the incompatibility precisely and use these methods for positioning, not as absent-but-promised experiments. This satisfies the contemporary-method audit through explicit task-incompatibility evidence; it does **not** support a claim that HRCSR outperforms those methods.

The strongest admissible controls remain methods actually evaluated under the identical continuous-time ranking pipeline: target-only TGAT, target-only GraphMixer, all-source pretraining, strict RASP, and HRCSR. CRAFT is tracked as a target-only architectural control rather than a multi-source transfer comparator. Its released MOOC command uses 30 neighbors, batch size 200, BPR loss and method-specific architecture settings; copying its published number into the matched table, or relabelling the existing target-only control, would be invalid. A CRAFT adaptation would therefore need its own clearly labelled protocol branch and is not used to claim superiority in the present transfer comparison.
