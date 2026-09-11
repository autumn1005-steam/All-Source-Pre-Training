import pytest
import torch

from tempofound.checkpoint import (
    build_checkpoint,
    load_foundation_checkpoint_payload,
    resolve_checkpoint_time_mode,
)
from tempofound.model import DatasetSpec, TemporalGraphFoundationModel


def test_checkpoint_without_format_metadata_is_legacy() -> None:
    payload = {"model_state_dict": {}}

    assert resolve_checkpoint_time_mode(payload) == "legacy_cos"


def test_new_checkpoint_records_paper_mode_and_format_version() -> None:
    model = torch.nn.Linear(2, 1)

    payload = build_checkpoint(
        model,
        time_encoding_mode="paper_sincos",
        model_config={"shared_dim": 128, "time_dim": 64},
    )

    assert payload["format_version"] == 3
    assert payload["time_encoding_mode"] == "paper_sincos"
    assert payload["model_config"]["time_dim"] == 64
    assert set(payload["model_state_dict"]) == {"weight", "bias"}
    assert payload["objective_names"] == [
        "mtlp",
        "tnp",
        "edc",
        "npp",
        "tip",
        "domain",
    ]


def test_v2_five_task_weights_are_remapped_without_domain_reinterpretation() -> None:
    model = TemporalGraphFoundationModel(
        (DatasetSpec("source", 3, 2, 4, 4),),
        shared_dim=6,
        time_dim=4,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_neighbors=2,
        time_encoding_mode="paper_sincos",
    )
    payload = build_checkpoint(
        model,
        time_encoding_mode="paper_sincos",
        model_config=model.model_config(),
    )
    payload["format_version"] = 2
    payload.pop("objective_names")
    payload["model_state_dict"]["task_log_vars"] = torch.tensor(
        [0.1, 0.2, 9.9, 0.3, 0.4]
    )

    loaded = load_foundation_checkpoint_payload(payload)

    assert torch.equal(
        loaded.task_log_vars.detach(), torch.tensor([0.1, 0.2, 0.3, 0.4, 0.0])
    )


def test_requested_mode_cannot_silently_override_checkpoint() -> None:
    payload = {"format_version": 2, "time_encoding_mode": "paper_sincos"}

    with pytest.raises(ValueError, match="conflicts"):
        resolve_checkpoint_time_mode(payload, requested="legacy_cos")


def test_legacy_tempofound_payload_loads_without_parameter_loss() -> None:
    model = TemporalGraphFoundationModel(
        (DatasetSpec("source", 3, 2, 4, 4),),
        shared_dim=6,
        time_dim=4,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_neighbors=2,
        time_encoding_mode="legacy_cos",
    )
    state = model.state_dict()
    assert "encoder.attn_layers.0.multi_head_target.w_qs.weight" in state
    assert "adapters.source.dataset_embed.weight" in state
    legacy_payload = {
        "model_state_dict": state,
        "dataset_configs": [
            {
                "name": "source",
                "raw_node_dim": 3,
                "raw_edge_dim": 2,
                "n_nodes": 4,
                "n_edges": 4,
            }
        ],
        "shared_dim": 6,
        "time_dim": 4,
        "num_layers": 1,
        "num_neighbors": 2,
    }

    loaded = load_foundation_checkpoint_payload(legacy_payload)

    assert loaded.time_encoding_mode == "legacy_cos"
    for name, value in state.items():
        assert torch.equal(loaded.state_dict()[name], value), name


def test_legacy_four_task_weights_are_migrated_to_five_tasks() -> None:
    model = TemporalGraphFoundationModel(
        (DatasetSpec("source", 3, 2, 4, 4),),
        shared_dim=6,
        time_dim=4,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_neighbors=2,
        time_encoding_mode="legacy_cos",
    )
    state = model.state_dict()
    legacy_weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
    state["task_log_vars"] = legacy_weights
    payload = {
        "model_state_dict": state,
        "dataset_configs": [
            {
                "name": "source",
                "raw_node_dim": 3,
                "raw_edge_dim": 2,
                "n_nodes": 4,
                "n_edges": 4,
            }
        ],
        "shared_dim": 6,
        "time_dim": 4,
        "num_layers": 1,
        "num_neighbors": 2,
    }

    loaded = load_foundation_checkpoint_payload(payload)

    migrated = loaded.state_dict()["task_log_vars"]
    assert torch.equal(migrated[:4], legacy_weights)
    assert migrated[4].item() == 0.0
