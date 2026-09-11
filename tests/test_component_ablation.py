from dataclasses import replace

from scripts.run_component_ablation import VARIANTS, variant_config
from tempofound.config import ExperimentConfig


def _base() -> ExperimentConfig:
    return ExperimentConfig(
        name="test",
        datasets=("a", "b", "target"),
        targets=("target",),
        source_overrides={"target": ("a", "b")},
        selection_method="rasp",
        proxy_metric="mrr",
        proxy_negative_policy="historical",
        max_sources=2,
        candidate_budget=3,
        beam_width=2,
        proxy_windows=3,
    )


def test_component_ablation_variants_change_only_target_side_controls():
    base = _base()
    configs = {name: variant_config(base, name) for name in VARIANTS}

    assert not configs["a1_expert_identity"].target_router_training_enabled
    assert configs["a1_expert_identity"].target_gate_mode == "force_identity"
    assert not configs["a2_response_router"].router_local_acceptance
    assert configs["a2_response_router"].target_gate_mode == "force_route"
    assert configs["a3_local_acceptance"].router_local_acceptance
    assert configs["a3_local_acceptance"].transfer_regret_weight == 0.0
    assert configs["a4_pairwise_regret"].transfer_regret_weight == 0.25
    assert configs["a4_pairwise_regret"].transfer_regret_mode == "event_pairwise"
    for config in configs.values():
        assert config.shared_dim == base.shared_dim
        assert config.sources_for("target") == base.sources_for("target")
