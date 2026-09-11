import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_every_manuscript_component_resolves_to_a_real_symbol() -> None:
    mapping = json.loads(
        (ROOT / "docs/PAPER_CODE_MAPPING.json").read_text(encoding="utf-8")
    )
    expected = {
        "dataset_adapters",
        "causal_temporal_encoder",
        "paper_time_encoding",
        "multi_objective_pretraining",
        "adaptive_weighting",
        "target_training_proxy",
        "forward_pair",
        "global_batch_budget",
        "frozen_transfer",
        "graphmixer_control",
        "risk_adjusted_portfolio",
        "collision_free_negatives",
        "fixed_negative_ranking",
        "target_affinity_budget",
        "conflict_aware_gradients",
        "domain_adversarial_learning",
    }

    assert set(mapping) == expected
    for component, target in mapping.items():
        module = importlib.import_module(target["module"])
        assert hasattr(module, target["symbol"]), component
        assert target["manuscript_section"]
