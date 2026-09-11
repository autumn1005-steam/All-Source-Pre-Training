# Manuscript-to-code mapping

| Manuscript component | Implementation |
|---|---|
| Dataset-Specific Representation Adapters | `tempofound.model.DatasetAdapter` |
| Causal Shared Temporal Encoder | `tempofound.model.SharedTemporalEncoder` and `tempofound.graph.NeighborFinder` |
| Normalized sine/cosine time encoding | `tempofound.layers.HarmonicTimeEncoder(mode="paper_sincos")` |
| Legacy cosine encoding | `tempofound.layers.HarmonicTimeEncoder(mode="legacy_cos")` |
| MTLP, TNP, EDC, NPP, TIP and domain-adversarial loss | `tempofound.model.TemporalGraphFoundationModel.pretrain_forward` |
| Gradient reversal for domain invariance | `tempofound.layers.gradient_reverse` |
| Adaptive uncertainty weighting | `tempofound.model.TemporalGraphFoundationModel.weighted_pretrain_loss` |
| Equal global update budget | `tempofound.budget.allocate_batch_budget` |
| Target-affinity weighted fixed budget | `tempofound.optimization.source_affinities` and `allocate_weighted_batch_budget` |
| Conflict-aware multi-source gradients | `tempofound.optimization.project_conflicting_gradients` |
| Target-training-only proxy | `forward_pair.proxy.score_target_train_proxy` |
| Forward-Pair, exactly 2K-1 candidates | `forward_pair.selector.select_forward_pair` |
| RASP uncertainty-aware portfolio search | `forward_pair.uncertainty` and `forward_pair.portfolio` |
| Collision-free fixed negatives and MRR | `tempofound.sampling.TemporalNegativeSampler` and `tempofound.evaluation.evaluate_link_ranking` |
| Frozen/full/scratch transfer | `tempofound.transfer.run_transfer` |
| K=4 and K=6 protocols | `configs/k4_primary.yaml`, `configs/k6_scalability.yaml` |
| GraphMixer control | `configs/graphmixer_control.yaml` and `GraphMixerTemporalEncoder` |
| Random-7, exhaustive, greedy controls | `forward_pair.controls` and `experiments/controls/` |

The manuscript's node-classification paragraph is a supplementary limitation
statement rather than a primary result table. The unified default pipeline
therefore reproduces the paper's link-prediction protocol; it does not relabel
events or silently substitute a different node-classification protocol.

The JSON version of this mapping is tested by importing every referenced
symbol. This prevents the public documentation from drifting away from the
actual package structure.

## RCSR-TKDE extension

| RCSR component | Implementation |
|---|---|
| Source-specialized residual knowledge bank | `tempofound.temporal_experts.SourceExpertBank` |
| Source-response-aware causal event router | `tempofound.temporal_experts.RiskControlledSourceRouter` |
| RASP affinity as global routing prior | `tempofound.model.TemporalGraphFoundationModel.set_source_routing_prior` + `tempofound.pretrain.run_pretraining` |
| Unseen-target no-new-expert transfer | `forward_pair.proxy.extend_model_for_target` + `TemporalGraphFoundationModel.encode_nodes` |
| Identity abstention under uncertainty | `RiskControlledSourceRouter.forward` |
| Counterfactual routed-vs-identity regret | `tempofound.transfer.run_transfer` and `forward_pair.proxy._calibrate_proxy` |
| Historical proxy negatives with collision-free cold-start fallback | `tempofound.sampling.TemporalNegativeSampler.sample_with_fallback` + `forward_pair.proxy` |
