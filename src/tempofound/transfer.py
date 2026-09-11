"""Frozen-encoder, full-fine-tuning, and scratch target transfer."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from forward_pair.proxy import extend_model_for_target

from .checkpoint import load_foundation_checkpoint_payload
from .config import ExperimentConfig
from .data import load_processed_dataset
from .evaluation import evaluate_link_ranking, reciprocal_ranks
from .model import DatasetSpec, LinkPredictionHead, TemporalGraphFoundationModel
from .sampling import TemporalNegativeSampler


def _target_spec(name: str, data: dict[str, Any]) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        raw_node_dim=int(data["node_features"].shape[1]),
        raw_edge_dim=int(data["edge_features"].shape[1]),
        n_nodes=int(data["node_features"].shape[0] - 1),
        n_edges=int(data["edge_features"].shape[0] - 1),
    )


def _logits(
    model: TemporalGraphFoundationModel,
    head: LinkPredictionHead,
    target: str,
    data: dict[str, Any],
    source: np.ndarray,
    destination: np.ndarray,
    timestamps: np.ndarray,
    *,
    use_source_router: bool = True,
) -> torch.Tensor:
    source_embedding = model.encode_nodes(
        target, data["neighbor_finder"], source, timestamps,
        use_source_router=use_source_router,
    )
    destination_embedding = model.encode_nodes(
        target, data["neighbor_finder"], destination, timestamps,
        use_source_router=use_source_router,
    )
    return head(source_embedding, destination_embedding)


@torch.no_grad()
def _evaluate(
    model: TemporalGraphFoundationModel,
    head: LinkPredictionHead,
    target: str,
    data: dict[str, Any],
    split: str,
    destination_pool: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    model.eval()
    head.eval()
    source = data[f"{split}_src"]
    destination = data[f"{split}_dst"]
    timestamps = data[f"{split}_ts"]
    rng = np.random.default_rng(seed)
    negative = rng.choice(destination_pool, size=len(source), replace=True)
    positive_logits = _logits(
        model, head, target, data, source, destination, timestamps
    )
    negative_logits = _logits(
        model, head, target, data, source, negative, timestamps
    )
    labels = np.concatenate((np.ones(len(source)), np.zeros(len(source))))
    scores = torch.sigmoid(torch.cat((positive_logits, negative_logits))).cpu().numpy()
    return (
        float(average_precision_score(labels, scores)),
        float(roc_auc_score(labels, scores)),
    )


def _build_negative_sampler(
    target_data: dict[str, Any], seed: int
) -> TemporalNegativeSampler:
    sources = np.concatenate(
        [target_data[f"{split}_src"] for split in ("train", "val", "test")]
    )
    destinations = np.concatenate(
        [target_data[f"{split}_dst"] for split in ("train", "val", "test")]
    )
    timestamps = np.concatenate(
        [target_data[f"{split}_ts"] for split in ("train", "val", "test")]
    )
    return TemporalNegativeSampler.from_events(
        sources,
        destinations,
        timestamps,
        seed=seed,
        training_end=float(target_data["validation_time"]),
    )




def _pairwise_ranking_loss(positive_logits: torch.Tensor, negative_logits: torch.Tensor) -> torch.Tensor:
    """Per-event logistic ranking loss; smaller is better and aligned with ranking."""
    if positive_logits.shape != negative_logits.shape:
        raise ValueError("positive and negative logits must have matching shapes")
    return F.softplus(-(positive_logits - negative_logits))


def _bootstrap_gate_from_gains(
    gains: np.ndarray,
    *,
    windows: int,
    replicates: int,
    confidence: float,
    min_gain: float,
    temperature: float,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    """Calibrate a target gate from chronological out-of-sample gains.

    ``gains`` are per-event reciprocal-rank improvements (routed - reference).
    Bootstrap is performed over chronological window means to avoid treating
    temporally correlated events as i.i.d. observations.
    """
    values = np.asarray(gains, dtype=np.float64)
    if values.ndim != 1 or len(values) < windows:
        raise ValueError("target-gate calibration needs at least one event per window")
    chunks = [chunk for chunk in np.array_split(values, windows) if len(chunk)]
    window_means = np.asarray([float(chunk.mean()) for chunk in chunks], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(window_means, size=(replicates, len(window_means)), replace=True)
    bootstrap_means = samples.mean(axis=1)
    alpha = 1.0 - confidence
    lcb = float(np.quantile(bootstrap_means, alpha))
    ucb = float(np.quantile(bootstrap_means, confidence))
    mean_gain = float(values.mean())
    if mode in {"lcb_hard", "holdout_lcb_hard"}:
        gate = 1.0 if lcb > min_gain else 0.0
    elif mode == "lcb_soft":
        # Legacy v2 logistic rule retained only for reproducibility.
        scaled = float(np.clip((lcb - min_gain) / temperature, -30.0, 30.0))
        gate = float(1.0 / (1.0 + np.exp(-scaled)))
    elif mode == "holdout_lcb_soft":
        # Zero-at-threshold soft abstention: unsupported transfer maps to zero,
        # rather than the ~0.5 gate induced by a centered logistic transform.
        excess = max(0.0, lcb - min_gain)
        gate = float(1.0 - np.exp(-excess / temperature)) if excess > 0.0 else 0.0
    elif mode == "force_identity":
        gate = 0.0
    elif mode == "force_route":
        gate = 1.0
    elif mode == "none":
        gate = 1.0
    else:
        raise ValueError(f"unsupported target gate mode: {mode}")
    return {
        "target_gate": gate,
        "target_gate_mean_gain": mean_gain,
        "target_gate_lcb": lcb,
        "target_gate_ucb": ucb,
        "target_gate_window_means": window_means.tolist(),
        "target_gate_calibration_events": int(len(values)),
        "target_gate_mode": mode,
    }


def _chronological_target_partition(
    target_data: dict[str, Any], config: ExperimentConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Split target-train into optimization prefix and calibration suffix."""
    total = len(target_data["train_src"])
    order = np.argsort(np.asarray(target_data["train_ts"], dtype=np.float64), kind="stable")
    if config.target_gate_mode not in {"holdout_lcb_soft", "holdout_lcb_hard", "force_identity", "force_route"}:
        return order, np.empty(0, dtype=np.int64)
    fractional_size = int(round(total * config.target_calibration_fraction))
    effective_min = min(
        config.target_calibration_min_events,
        max(config.target_gate_windows, total // 3),
    )
    calibration_size = max(config.target_gate_windows, fractional_size, effective_min)
    # Keep at least one optimization event; for normal JODIE-scale targets the
    # configured fraction dominates and this remains the intended 80/20 split.
    calibration_size = min(calibration_size, total - 1)
    if calibration_size < config.target_gate_windows:
        raise ValueError("target calibration suffix is too small for target_gate_windows")
    split = total - calibration_size
    return order[:split], order[split:]


@torch.no_grad()
def _calibrate_target_gate(
    model: TemporalGraphFoundationModel,
    head: LinkPredictionHead,
    *,
    target: str,
    target_data: dict[str, Any],
    config: ExperimentConfig,
    seed: int,
    calibration_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    if model.source_router is None or target in model.source_expert_names:
        return {
            "target_gate": 1.0,
            "target_gate_mode": "not_applicable",
            "target_gate_information_boundary": "not_applicable",
        }
    model.eval()
    head.eval()
    model.source_router.set_target_gate(1.0)

    if calibration_indices is None or len(calibration_indices) == 0:
        indices = np.arange(len(target_data["train_src"]), dtype=np.int64)
        boundary = "target_training_period_in_sample_legacy"
    else:
        indices = np.asarray(calibration_indices, dtype=np.int64)
        boundary = "held_out_chronological_target_train_suffix"

    # Build the sampler from the full target-training history. At each query
    # timestamp the historical policy only exposes destinations first seen in
    # the past and removes observed source-destination links up to query time.
    sampler = TemporalNegativeSampler.from_events(
        target_data["train_src"],
        target_data["train_dst"],
        target_data["train_ts"],
        seed=seed + 51017,
    )
    rr_gains: list[float] = []
    routed_rr_values: list[float] = []
    reference_rr_values: list[float] = []
    batch_size = max(1, config.evaluation_batch_size)
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        source = target_data["train_src"][selected]
        destination = target_data["train_dst"][selected]
        timestamps = target_data["train_ts"][selected]
        query_ids = np.asarray(target_data.get("train_eidx", np.arange(len(target_data["train_src"]))))[selected]
        sampleable = sampler.sampleable_mask(
            source, timestamps, policy="historical", fallback_policy="random"
        )
        if not sampleable.any():
            continue
        source = source[sampleable]
        destination = destination[sampleable]
        timestamps = timestamps[sampleable]
        query_ids = query_ids[sampleable]
        negative = sampler.sample_with_fallback(
            source,
            timestamps,
            count=config.target_gate_negatives,
            policy="historical",
            fallback_policy="random",
            query_ids=query_ids,
        )[0]

        routed_source = model.encode_nodes(
            target, target_data["neighbor_finder"], source, timestamps, use_source_router=True
        )
        routed_destination = model.encode_nodes(
            target, target_data["neighbor_finder"], destination, timestamps, use_source_router=True
        )
        routed_positive = head(routed_source, routed_destination).reshape(-1)
        flat_negative = negative.reshape(-1)
        repeated_timestamps = np.repeat(timestamps, config.target_gate_negatives)
        routed_negative_embeddings = model.encode_nodes(
            target, target_data["neighbor_finder"], flat_negative, repeated_timestamps,
            use_source_router=True,
        )
        routed_repeated_source = routed_source[:, None, :].expand(
            -1, config.target_gate_negatives, -1
        ).reshape(-1, routed_source.shape[-1])
        routed_negative = head(routed_repeated_source, routed_negative_embeddings).reshape(
            len(source), config.target_gate_negatives
        )

        reference_source = model.encode_nodes(
            target, target_data["neighbor_finder"], source, timestamps, use_source_router=False
        )
        reference_destination = model.encode_nodes(
            target, target_data["neighbor_finder"], destination, timestamps, use_source_router=False
        )
        reference_positive = head(reference_source, reference_destination).reshape(-1)
        reference_negative_embeddings = model.encode_nodes(
            target, target_data["neighbor_finder"], flat_negative, repeated_timestamps,
            use_source_router=False,
        )
        reference_repeated_source = reference_source[:, None, :].expand(
            -1, config.target_gate_negatives, -1
        ).reshape(-1, reference_source.shape[-1])
        reference_negative = head(reference_repeated_source, reference_negative_embeddings).reshape(
            len(source), config.target_gate_negatives
        )

        routed_rr = reciprocal_ranks(routed_positive, routed_negative)
        reference_rr = reciprocal_ranks(reference_positive, reference_negative)
        routed_rr_values.extend(routed_rr.cpu().tolist())
        reference_rr_values.extend(reference_rr.cpu().tolist())
        rr_gains.extend((routed_rr - reference_rr).cpu().tolist())

    diagnostics = _bootstrap_gate_from_gains(
        np.asarray(rr_gains, dtype=np.float64),
        windows=config.target_gate_windows,
        replicates=config.target_gate_bootstrap_replicates,
        confidence=config.target_gate_confidence,
        min_gain=config.target_gate_min_gain,
        temperature=config.target_gate_temperature,
        mode=config.target_gate_mode,
        seed=seed + 52009,
    )
    diagnostics.update({
        "target_gate_information_boundary": boundary,
        "target_gate_metric": "per_event_reciprocal_rank_gain",
        "target_gate_negatives": int(config.target_gate_negatives),
        "target_gate_routed_calibration_mrr": float(np.mean(routed_rr_values)),
        "target_gate_reference_calibration_mrr": float(np.mean(reference_rr_values)),
        "target_gate_calibration_mrr_delta": float(np.mean(rr_gains)),
    })
    model.source_router.set_target_gate(diagnostics["target_gate"])
    return diagnostics


def run_transfer(
    config: ExperimentConfig,
    *,
    checkpoint_path: str | Path | None,
    target: str,
    seed: int,
    processed_dir: str | Path,
    mode: str = "frozen",
    device: str = "cpu",
    evaluation_protocol: str = "legacy_binary",
    negative_policy: str = "historical",
    negatives_per_positive: int = 100,
    evaluation_batch_size: int = 256,
    profile_timing_repetitions: int = 1,
    profile_warmup_repetitions: int = 0,
) -> dict[str, Any]:
    if mode not in {"frozen", "full", "scratch"}:
        raise ValueError("mode must be frozen, full, or scratch")
    if evaluation_protocol not in {"legacy_binary", "ranking"}:
        raise ValueError("evaluation_protocol must be legacy_binary or ranking")
    if negatives_per_positive <= 0 or evaluation_batch_size <= 0:
        raise ValueError("ranking negative count and batch size must be positive")
    if profile_timing_repetitions < 1 or profile_warmup_repetitions < 0:
        raise ValueError("profiling repetitions must be positive and warmups non-negative")
    transfer_started = time.perf_counter()

    def synchronize() -> None:
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    target_data = load_processed_dataset(target, processed_dir)
    torch_device = torch.device(device)
    if mode == "scratch":
        model = TemporalGraphFoundationModel(
            (_target_spec(target, target_data),),
            shared_dim=config.shared_dim,
            time_dim=config.time_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            num_neighbors=config.num_neighbors,
            time_encoding_mode=config.time_encoding_mode,
            backbone=config.backbone,
            temporal_experts=config.temporal_experts,
            expert_identity_init=config.expert_identity_init,
            source_experts=(config.source_experts and not config.risk_controlled_routing),
            risk_controlled_routing=False,
        ).to(torch_device)
        model.load_dataset_features(
            target, target_data["node_features"], target_data["edge_features"]
        )
    else:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for frozen or full transfer")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = load_foundation_checkpoint_payload(payload).to(torch_device)
        model = extend_model_for_target(model, target, target_data)
        if model.source_router is not None:
            # These switches alter only the target-side ablation behavior.  They
            # do not add parameters and therefore remain compatible with the
            # authoritative HRCSR source checkpoint.
            model.source_router.local_acceptance = bool(config.router_local_acceptance)
            if not config.target_router_training_enabled:
                model.source_router.set_target_gate(0.0)
    head = LinkPredictionHead(model.shared_dim, config.dropout).to(torch_device)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if mode == "frozen":
        for parameter in model.adapters[target].parameters():
            parameter.requires_grad_(True)
        if model.source_router is not None:
            for parameter in model.source_router.parameters():
                parameter.requires_grad_(True)
    else:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(True)
        if model.temporal_mixture is not None:
            for parameter in model.temporal_mixture.parameters():
                parameter.requires_grad_(True)
        if model.source_expert_bank is not None:
            for parameter in model.source_expert_bank.parameters():
                parameter.requires_grad_(True)
        if model.source_router is not None:
            for parameter in model.source_router.parameters():
                parameter.requires_grad_(True)
        for adapter in model.adapters.values():
            for parameter in adapter.parameters():
                parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable.extend(head.parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters()) + sum(
        parameter.numel() for parameter in head.parameters()
    )
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    source_expert_parameters = (
        sum(parameter.numel() for parameter in model.source_expert_bank.parameters())
        if model.source_expert_bank is not None else 0
    )
    source_router_parameters = (
        sum(parameter.numel() for parameter in model.source_router.parameters())
        if model.source_router is not None else 0
    )
    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    destination_pool = np.unique(target_data["train_dst"])
    negative_sampler = (
        _build_negative_sampler(target_data, seed + 29009)
        if evaluation_protocol == "ranking"
        else None
    )
    rng = np.random.default_rng(seed + 20011)
    training_negative_sampler = TemporalNegativeSampler.from_events(
        target_data["train_src"],
        target_data["train_dst"],
        target_data["train_ts"],
        seed=seed + 20017,
    )
    optimization_indices, calibration_indices = _chronological_target_partition(
        target_data, config
    )
    prefix_data = dict(target_data)
    prefix_data["prefix_src"] = target_data["train_src"][optimization_indices]
    prefix_data["prefix_dst"] = target_data["train_dst"][optimization_indices]
    prefix_data["prefix_ts"] = target_data["train_ts"][optimization_indices]
    if "train_eidx" in target_data:
        prefix_data["prefix_eidx"] = target_data["train_eidx"][optimization_indices]
    holdout_safety = config.target_gate_mode in {
        "holdout_lcb_soft", "holdout_lcb_hard", "force_identity", "force_route"
    }
    regret_values: list[float] = []
    regret_violations = 0
    training_negative_fallbacks = 0
    training_negative_queries = 0
    # ``None`` is intentional for fixed-epoch runs: no validation/prefix
    # score is computed or used for checkpoint selection in that protocol.
    best_validation_score: float | None = None
    selected_epoch: int | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    best_head_state: dict[str, torch.Tensor] | None = None

    synchronize()
    adaptation_started = time.perf_counter()
    for epoch in range(config.transfer_epochs):
        if mode == "frozen":
            model.eval()
            model.adapters[target].train()
        else:
            model.train()
        head.train()
        indices = rng.permutation(optimization_indices)
        for start in range(0, len(indices), config.batch_size):
            selected = indices[start : start + config.batch_size]
            source = target_data["train_src"][selected]
            destination = target_data["train_dst"][selected]
            timestamps = target_data["train_ts"][selected]
            sampleable = training_negative_sampler.sampleable_mask(
                source, timestamps, policy="historical", fallback_policy="random"
            )
            if not sampleable.any():
                continue
            source = source[sampleable]
            destination = destination[sampleable]
            timestamps = timestamps[sampleable]
            sampled_negative, fallback_count = training_negative_sampler.sample_with_fallback(
                source, timestamps, count=1, policy="historical", fallback_policy="random"
            )
            training_negative_fallbacks += fallback_count
            training_negative_queries += len(source)
            negative = sampled_negative.reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            positive_logits = _logits(
                model, head, target, target_data, source, destination, timestamps
            )
            negative_logits = _logits(
                model, head, target, target_data, source, negative, timestamps
            )
            positive_bce = F.binary_cross_entropy_with_logits(
                positive_logits, torch.ones_like(positive_logits), reduction="none"
            )
            negative_bce = F.binary_cross_entropy_with_logits(
                negative_logits, torch.zeros_like(negative_logits), reduction="none"
            )
            event_bce = positive_bce + negative_bce
            loss = event_bce.mean()
            total_loss = loss
            identity_positive = None
            identity_negative = None

            # In held-out safety-calibration variants, make the target head and
            # adapter path-neutral during optimization: both router-on and
            # router-off representations must solve the target task. This avoids
            # calibrating against an identity path whose head was only fitted to
            # routed features.
            if (
                holdout_safety
                and model.source_router is not None
                and target not in model.source_expert_names
                and config.target_reference_weight > 0.0
            ):
                identity_positive = _logits(
                    model, head, target, target_data, source, destination, timestamps,
                    use_source_router=False,
                )
                identity_negative = _logits(
                    model, head, target, target_data, source, negative, timestamps,
                    use_source_router=False,
                )
                identity_event_bce = (
                    F.binary_cross_entropy_with_logits(
                        identity_positive, torch.ones_like(identity_positive), reduction="none"
                    )
                    + F.binary_cross_entropy_with_logits(
                        identity_negative, torch.zeros_like(identity_negative), reduction="none"
                    )
                )
                total_loss = (
                    loss + config.target_reference_weight * identity_event_bce.mean()
                ) / (1.0 + config.target_reference_weight)

            if (
                model.source_router is not None
                and target not in model.source_expert_names
                and config.transfer_regret_weight > 0.0
            ):
                if identity_positive is None or identity_negative is None:
                    with torch.no_grad():
                        identity_positive = _logits(
                            model, head, target, target_data, source, destination, timestamps,
                            use_source_router=False,
                        )
                        identity_negative = _logits(
                            model, head, target, target_data, source, negative, timestamps,
                            use_source_router=False,
                        )
                # The reference path is a fixed comparator for regret. Detach it
                # so the model cannot satisfy the constraint by making the
                # reference worse.
                reference_positive = identity_positive.detach()
                reference_negative = identity_negative.detach()
                if config.transfer_regret_mode == "event_pairwise":
                    routed_event_loss = _pairwise_ranking_loss(positive_logits, negative_logits)
                    identity_event_loss = _pairwise_ranking_loss(
                        reference_positive, reference_negative
                    )
                    event_regret = F.relu(
                        routed_event_loss - identity_event_loss + config.transfer_regret_margin
                    )
                    total_loss = total_loss + config.transfer_regret_weight * event_regret.mean()
                    signed_regret = (routed_event_loss.detach() - identity_event_loss).cpu().numpy()
                    regret_values.extend(float(value) for value in signed_regret)
                    regret_violations += int(
                        np.count_nonzero(signed_regret > config.transfer_regret_margin)
                    )
                else:
                    identity_bce = F.binary_cross_entropy_with_logits(
                        reference_positive, torch.ones_like(reference_positive)
                    ) + F.binary_cross_entropy_with_logits(
                        reference_negative, torch.zeros_like(reference_negative)
                    )
                    regret = F.relu(loss - identity_bce + config.transfer_regret_margin)
                    total_loss = total_loss + config.transfer_regret_weight * regret
                    regret_value = float((loss.detach() - identity_bce).cpu())
                    regret_values.append(regret_value)
                    regret_violations += int(regret_value > config.transfer_regret_margin)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
        if config.checkpoint_selection_mode == "fixed_epoch":
            if epoch == config.transfer_epochs - 1:
                best_model_state = deepcopy(model.state_dict())
                best_head_state = deepcopy(head.state_dict())
                selected_epoch = epoch + 1
            continue

        selection_split = (
            "val" if config.checkpoint_selection_mode == "validation" else "prefix"
        )
        selection_data = (
            target_data if config.checkpoint_selection_mode == "validation" else prefix_data
        )
        if evaluation_protocol == "legacy_binary":
            selection_score, _ = _evaluate(
                model,
                head,
                target,
                selection_data,
                selection_split,
                destination_pool,
                seed + 30011 + epoch,
            )
        else:
            assert negative_sampler is not None
            selection_score = evaluate_link_ranking(
                model,
                head,
                target,
                selection_data,
                selection_split,
                negative_sampler,
                negatives_per_positive=negatives_per_positive,
                batch_size=evaluation_batch_size,
                policy=negative_policy,
            )["mrr"]
        if best_validation_score is None or selection_score > best_validation_score:
            best_validation_score = selection_score
            best_model_state = deepcopy(model.state_dict())
            best_head_state = deepcopy(head.state_dict())
            selected_epoch = epoch + 1

    if best_model_state is None or best_head_state is None:
        raise RuntimeError("transfer produced no validation checkpoint")
    model.load_state_dict(best_model_state)
    head.load_state_dict(best_head_state)
    synchronize()
    target_adaptation_seconds = time.perf_counter() - adaptation_started
    synchronize()
    calibration_started = time.perf_counter()
    target_gate_diagnostics = _calibrate_target_gate(
        model,
        head,
        target=target,
        target_data=target_data,
        config=config,
        seed=seed,
        calibration_indices=calibration_indices,
    )
    synchronize()
    calibration_seconds = time.perf_counter() - calibration_started
    common = {
        "target": target,
        "seed": seed,
        "mode": mode,
        "evaluation_protocol": evaluation_protocol,
        "checkpoint_selection_mode": config.checkpoint_selection_mode,
        "checkpoint_selected_epoch": selected_epoch,
        "checkpoint_selection_scope": {
            "validation": "target_validation_only_after_source_selection",
            "fixed_epoch": "predeclared_final_epoch_without_validation_selection",
            "prefix_only": "target_optimization_prefix_only",
        }[config.checkpoint_selection_mode],
        "training_negative_policy": "historical_with_collision_free_random_cold_start_fallback",
        "training_negative_fallback_policy": "random",
        "target_optimization_events": int(len(optimization_indices)),
        "target_calibration_events_reserved": int(len(calibration_indices)),
        "target_calibration_fraction": (
            float(len(calibration_indices) / len(target_data["train_src"]))
            if len(target_data["train_src"]) else 0.0
        ),
        "target_reference_weight": float(config.target_reference_weight if holdout_safety else 0.0),
        "training_negative_fallback_rate": (
            training_negative_fallbacks / training_negative_queries
            if training_negative_queries else 0.0
        ),
        "risk_controlled_routing": model.source_router is not None,
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "source_expert_parameters": int(source_expert_parameters),
        "source_router_parameters": int(source_router_parameters),
        "target_adaptation_seconds": float(target_adaptation_seconds),
        "calibration_seconds": float(calibration_seconds),
        "transfer_regret_weight": config.transfer_regret_weight if model.source_router is not None else 0.0,
        "transfer_regret_mode": config.transfer_regret_mode if model.source_router is not None else "not_applicable",
        **target_gate_diagnostics,
        "mean_train_routing_regret": (float(np.mean(regret_values)) if regret_values else 0.0),
        "routing_regret_violation_rate": (regret_violations / len(regret_values) if regret_values else 0.0),
    }
    if evaluation_protocol == "legacy_binary":
        synchronize()
        inference_started = time.perf_counter()
        test_ap, test_auc = _evaluate(
            model,
            head,
            target,
            target_data,
            "test",
            destination_pool,
            seed + 40009,
        )
        synchronize()
        inference_seconds = time.perf_counter() - inference_started
        return {
            **common,
            "validation_ap": best_validation_score,
            "checkpoint_selection_score": best_validation_score,
            "test_ap": test_ap,
            "test_auc": test_auc,
            "inference_timing_repetitions": 1,
            "inference_warmup_repetitions": 0,
            "inference_seconds_mean": float(inference_seconds),
            "inference_seconds_std": 0.0,
            "transfer_total_seconds": float(time.perf_counter() - transfer_started),
        }
    assert negative_sampler is not None
    path_diagnostics: dict[str, Any] = {}
    if model.source_router is not None and target not in model.source_expert_names:
        calibrated_gate = float(model.source_router.target_gate.item())
        model.source_router.set_target_gate(1.0)
        routed_test = evaluate_link_ranking(
            model,
            head,
            target,
            target_data,
            "test",
            negative_sampler,
            negatives_per_positive=negatives_per_positive,
            batch_size=evaluation_batch_size,
            policy=negative_policy,
            use_source_router=True,
        )
        reference_test = evaluate_link_ranking(
            model,
            head,
            target,
            target_data,
            "test",
            negative_sampler,
            negatives_per_positive=negatives_per_positive,
            batch_size=evaluation_batch_size,
            policy=negative_policy,
            use_source_router=False,
        )
        lcb = float(target_gate_diagnostics.get("target_gate_lcb", 0.0))
        excess = max(0.0, lcb - config.target_gate_min_gain)
        hard_gate = 1.0 if lcb > config.target_gate_min_gain else 0.0
        soft_gate = (
            float(1.0 - np.exp(-excess / config.target_gate_temperature))
            if excess > 0.0 else 0.0
        )
        gate_sweep = {
            "identity": {"gate": 0.0, "test_mrr": reference_test["mrr"]},
            "route": {"gate": 1.0, "test_mrr": routed_test["mrr"]},
        }
        for gate_name, gate_value in (("hard", hard_gate), ("soft", soft_gate)):
            model.source_router.set_target_gate(gate_value)
            metrics = evaluate_link_ranking(
                model,
                head,
                target,
                target_data,
                "test",
                negative_sampler,
                negatives_per_positive=negatives_per_positive,
                batch_size=evaluation_batch_size,
                policy=negative_policy,
                use_source_router=True,
            )
            gate_sweep[gate_name] = {"gate": gate_value, "test_mrr": metrics["mrr"]}
        model.source_router.set_target_gate(calibrated_gate)
        path_diagnostics = {
            "raw_routed_test_mrr": routed_test["mrr"],
            "router_off_reference_test_mrr": reference_test["mrr"],
            "raw_route_minus_reference_test_mrr": routed_test["mrr"] - reference_test["mrr"],
            "routed_test_total_queries": routed_test["total_queries"],
            "routed_test_valid_queries": routed_test["valid_queries"],
            "routed_test_skipped_queries": routed_test["skipped_queries"],
            "routed_test_fallback_count": routed_test["fallback_count"],
            "routed_test_collision_rejection_count": routed_test["collision_rejection_count"],
            "reference_test_total_queries": reference_test["total_queries"],
            "reference_test_valid_queries": reference_test["valid_queries"],
            "reference_test_skipped_queries": reference_test["skipped_queries"],
            "reference_test_fallback_count": reference_test["fallback_count"],
            "reference_test_collision_rejection_count": reference_test["collision_rejection_count"],
            "safety_gate_sweep": gate_sweep,
        }
    evaluation_kwargs = {
        "negatives_per_positive": negatives_per_positive,
        "batch_size": evaluation_batch_size,
        "policy": negative_policy,
    }
    for _ in range(profile_warmup_repetitions):
        evaluate_link_ranking(
            model, head, target, target_data, "test", negative_sampler, **evaluation_kwargs
        )
    inference_durations: list[float] = []
    timed_metrics: list[dict[str, Any]] = []
    for _ in range(profile_timing_repetitions):
        synchronize()
        inference_started = time.perf_counter()
        metrics = evaluate_link_ranking(
            model, head, target, target_data, "test", negative_sampler, **evaluation_kwargs
        )
        synchronize()
        inference_durations.append(time.perf_counter() - inference_started)
        timed_metrics.append(metrics)
    test_metrics = timed_metrics[0]
    inference_mean = float(np.mean(inference_durations))
    inference_std = float(np.std(inference_durations, ddof=1)) if len(inference_durations) > 1 else 0.0
    valid_queries = int(test_metrics["valid_queries"])
    return {
        **common,
        **path_diagnostics,
        "validation_mrr": best_validation_score,
        "checkpoint_selection_score": best_validation_score,
        "test_mrr": test_metrics["mrr"],
        "test_num_positives": test_metrics["num_positives"],
        "test_total_queries": test_metrics["total_queries"],
        "test_valid_queries": test_metrics["valid_queries"],
        "test_skipped_queries": test_metrics["skipped_queries"],
        "test_fallback_count": test_metrics["fallback_count"],
        "test_collision_rejection_count": test_metrics["collision_rejection_count"],
        "negative_policy": test_metrics["negative_policy"],
        "fallback_policy": test_metrics["fallback_policy"],
        "negatives_per_positive": test_metrics["negatives_per_positive"],
        "evaluation_batch_size": evaluation_batch_size,
        "inference_timing_repetitions": int(profile_timing_repetitions),
        "inference_warmup_repetitions": int(profile_warmup_repetitions),
        "inference_seconds_mean": inference_mean,
        "inference_seconds_std": inference_std,
        "inference_ms_per_query": 1000.0 * inference_mean / valid_queries,
        "inference_queries_per_second": valid_queries / inference_mean,
        "transfer_total_seconds": float(time.perf_counter() - transfer_started),
    }
