"""Versioned checkpoint helpers and time-mode compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn


FORMAT_VERSION = 3
OBJECTIVE_NAMES = ("mtlp", "tnp", "edc", "npp", "tip", "domain")
SUPPORTED_TIME_MODES = frozenset({"paper_sincos", "legacy_cos"})


def resolve_checkpoint_time_mode(
    payload: Mapping[str, Any], requested: str | None = None
) -> str:
    """Resolve a checkpoint's behavior without silently changing old models."""
    stored = payload.get("time_encoding_mode")
    if stored is None:
        stored = "legacy_cos"
    if stored not in SUPPORTED_TIME_MODES:
        raise ValueError(f"checkpoint has unsupported time encoding mode: {stored}")
    if requested is not None and requested not in SUPPORTED_TIME_MODES:
        raise ValueError(f"unsupported requested time encoding mode: {requested}")
    if requested is not None and requested != stored:
        raise ValueError(
            f"requested time encoding mode {requested!r} conflicts with "
            f"checkpoint mode {stored!r}"
        )
    return str(stored)


def build_checkpoint(
    model: nn.Module,
    *,
    time_encoding_mode: str,
    model_config: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if time_encoding_mode not in SUPPORTED_TIME_MODES:
        raise ValueError(f"unsupported time encoding mode: {time_encoding_mode}")
    return {
        "format_version": FORMAT_VERSION,
        "time_encoding_mode": time_encoding_mode,
        "model_config": dict(model_config),
        "model_state_dict": model.state_dict(),
        "objective_names": list(OBJECTIVE_NAMES),
        "metadata": dict(metadata or {}),
    }


def load_foundation_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    requested_time_encoding_mode: str | None = None,
) -> nn.Module:
    """Construct a foundation model from new or released legacy metadata."""
    from .model import DatasetSpec, TemporalGraphFoundationModel

    mode = resolve_checkpoint_time_mode(payload, requested_time_encoding_mode)
    if "model_config" in payload:
        config = dict(payload["model_config"])
        raw_specs = config.pop("dataset_specs")
        config["time_encoding_mode"] = mode
        specs = tuple(DatasetSpec.from_mapping(value) for value in raw_specs)
    else:
        specs = tuple(
            DatasetSpec.from_mapping(value) for value in payload["dataset_configs"]
        )
        config = {
            "shared_dim": int(payload["shared_dim"]),
            "time_dim": int(payload["time_dim"]),
            "num_layers": int(payload["num_layers"]),
            "num_heads": int(payload.get("num_heads", payload.get("n_head", 2))),
            "dropout": float(payload.get("dropout", 0.1)),
            "num_neighbors": int(payload["num_neighbors"]),
            "time_encoding_mode": mode,
        }
    model = TemporalGraphFoundationModel(specs, **config)
    state = payload["model_state_dict"]
    current_state = model.state_dict()
    format_version = int(payload.get("format_version", 1))
    if (
        format_version == 2
        and "task_log_vars" in state
        and tuple(state["task_log_vars"].shape) == (5,)
        and tuple(current_state["task_log_vars"].shape) == (5,)
    ):
        old_weights = state["task_log_vars"]
        migrated_state = dict(state)
        migrated_weights = current_state["task_log_vars"].clone()
        # v2 order: MTLP, TNP, domain-classification, EDC, NPP.
        # v3 order: MTLP, TNP, EDC, NPP, TIP; domain is explicitly weighted.
        migrated_weights[:4].copy_(old_weights[[0, 1, 3, 4]])
        migrated_state["task_log_vars"] = migrated_weights
        state = migrated_state
    # v4 HRCSR adds a non-trainable target-level routing gate buffer.  Older
    # RCSR checkpoints do not contain it; initialize to identity-neutral 1.0.
    if "source_router.target_gate" in current_state and "source_router.target_gate" not in state:
        migrated_state = dict(state)
        migrated_state["source_router.target_gate"] = current_state["source_router.target_gate"].clone()
        state = migrated_state
    if (
        set(state) == set(current_state)
        and "task_log_vars" in state
        and tuple(state["task_log_vars"].shape) == (4,)
        and tuple(current_state["task_log_vars"].shape) == (5,)
        and all(
            tuple(state[name].shape) == tuple(value.shape)
            for name, value in current_state.items()
            if name != "task_log_vars"
        )
    ):
        migrated_state = dict(state)
        migrated_task_weights = current_state["task_log_vars"].clone()
        migrated_task_weights[:4].copy_(state["task_log_vars"])
        migrated_state["task_log_vars"] = migrated_task_weights
        state = migrated_state
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "checkpoint parameters do not match the unified TempoFound architecture"
        ) from error
    return model
