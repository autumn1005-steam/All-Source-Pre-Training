"""Target-training-only calibration and proxy scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from tempofound.checkpoint import load_foundation_checkpoint_payload
from tempofound.evaluation import evaluate_link_ranking
from tempofound.sampling import TemporalNegativeSampler
from tempofound.data import load_processed_dataset, split_target_train_period
from tempofound.model import (
    DatasetSpec,
    LinkPredictionHead,
    TemporalGraphFoundationModel,
)

from .uncertainty import bootstrap_mean, risk_adjusted_utility


PROXY_FIELDS = (
    "train_src",
    "train_dst",
    "train_ts",
    "neighbor_finder",
    "node_features",
    "edge_features",
)


def _restricted_proxy(partition: dict[str, Any]) -> dict[str, Any]:
    return {field: partition[field] for field in PROXY_FIELDS}


def build_proxy_partitions(
    target_data: dict[str, Any], calibration_fraction: float = 0.8
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration, probe = split_target_train_period(
        target_data, calibration_fraction
    )
    return _restricted_proxy(calibration), _restricted_proxy(probe)


def build_proxy_windows(
    target_data: dict[str, Any],
    *,
    calibration_fraction: float,
    window_count: int,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if window_count < 1:
        raise ValueError("proxy window_count must be positive")
    calibration, probe = split_target_train_period(
        target_data, calibration_fraction
    )
    event_count = len(probe["train_ts"])
    if window_count > event_count:
        raise ValueError(
            f"proxy window_count={window_count} exceeds probe events={event_count}"
        )
    windows: list[dict[str, Any]] = []
    for indices in np.array_split(np.arange(event_count), window_count):
        window: dict[str, Any] = {}
        for field in PROXY_FIELDS:
            value = probe[field]
            if field.startswith("train_"):
                window[field] = np.asarray(value)[indices]
            else:
                window[field] = value
        windows.append(window)
    return _restricted_proxy(calibration), tuple(windows)


def summarize_proxy_windows(
    window_scores: list[float] | tuple[float, ...],
    *,
    seed: int,
    bootstrap_replicates: int,
    confidence: float,
    risk_beta: float,
    cardinality: int,
    size_penalty: float,
) -> dict[str, Any]:
    summary = bootstrap_mean(
        window_scores,
        seed=seed,
        replicates=bootstrap_replicates,
        confidence=confidence,
    )
    return {
        **summary.to_dict(),
        "risk_adjusted_utility": risk_adjusted_utility(
            summary,
            beta=risk_beta,
            cardinality=cardinality,
            size_penalty=size_penalty,
        ),
    }


def extend_model_for_target(
    model: TemporalGraphFoundationModel,
    target: str,
    target_data: dict[str, Any],
) -> TemporalGraphFoundationModel:
    if target in model.name_to_id:
        model.load_dataset_features(
            target, target_data["node_features"], target_data["edge_features"]
        )
        return model
    target_spec = DatasetSpec(
        name=target,
        raw_node_dim=int(target_data["node_features"].shape[1]),
        raw_edge_dim=int(target_data["edge_features"].shape[1]),
        n_nodes=int(target_data["node_features"].shape[0] - 1),
        n_edges=int(target_data["edge_features"].shape[0] - 1),
    )
    extended = TemporalGraphFoundationModel(
        (*model.dataset_specs, target_spec),
        shared_dim=model.shared_dim,
        time_dim=model.time_dim,
        num_layers=model.num_layers,
        num_heads=model.num_heads,
        dropout=model.dropout,
        num_neighbors=model.num_neighbors,
        time_encoding_mode=model.time_encoding_mode,
        backbone=model.backbone,
        temporal_experts=model.temporal_experts,
        expert_identity_init=model.expert_identity_init,
        source_experts=model.source_experts,
        source_expert_names=(model.source_expert_names if model.risk_controlled_routing else None),
        risk_controlled_routing=model.risk_controlled_routing,
        router_temperature=model.router_temperature,
        router_prior_strength=model.router_prior_strength,
        router_entropy_penalty=model.router_entropy_penalty,
        router_abstain_bias=model.router_abstain_bias,
    ).to(model.device)
    old_state = model.state_dict()
    new_state = extended.state_dict()
    with torch.no_grad():
        for name, old_value in old_state.items():
            if name not in new_state:
                continue
            destination = new_state[name]
            if destination.shape == old_value.shape:
                destination.copy_(old_value)
            elif (
                destination.ndim == old_value.ndim
                and destination.ndim > 0
                and destination.shape[1:] == old_value.shape[1:]
                and destination.shape[0] >= old_value.shape[0]
            ):
                destination[: old_value.shape[0]].copy_(old_value)
    extended.load_state_dict(new_state)
    extended.load_dataset_features(
        target, target_data["node_features"], target_data["edge_features"]
    )
    return extended


def _link_logits(
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


def _pairwise_ranking_loss(positive_logits: torch.Tensor, negative_logits: torch.Tensor) -> torch.Tensor:
    if positive_logits.shape != negative_logits.shape:
        raise ValueError("positive and negative logits must have matching shapes")
    return F.softplus(-(positive_logits - negative_logits))


def _calibrate_proxy(
    model: TemporalGraphFoundationModel,
    *,
    target: str,
    target_data: dict[str, Any],
    calibration: dict[str, Any],
    seed: int,
    batch_size: int,
    calibration_steps: int,
    learning_rate: float,
    negative_policy: str = "random",
    regret_weight: float = 0.0,
    regret_margin: float = 0.0,
    regret_mode: str = "batch_bce",
) -> tuple[
    TemporalGraphFoundationModel,
    LinkPredictionHead,
    np.random.Generator,
    np.ndarray,
    list[float],
]:
    torch.manual_seed(seed + 11003)
    np.random.seed(seed + 11003)
    if regret_weight < 0.0 or regret_margin < 0.0:
        raise ValueError("proxy regret weight and margin must be non-negative")
    if regret_mode not in {"batch_bce", "event_pairwise"}:
        raise ValueError("proxy regret_mode must be batch_bce or event_pairwise")
    model = extend_model_for_target(model, target, target_data)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.adapters[target].parameters():
        parameter.requires_grad_(True)
    router_parameters: list[torch.nn.Parameter] = []
    if model.source_router is not None:
        for parameter in model.source_router.parameters():
            parameter.requires_grad_(True)
        router_parameters = list(model.source_router.parameters())
    head = LinkPredictionHead(model.shared_dim, model.dropout).to(model.device)
    parameters = list(model.adapters[target].parameters()) + router_parameters + list(head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    model.eval()
    model.adapters[target].train()
    head.train()
    rng = np.random.default_rng(seed + 12007)
    destination_pool = np.unique(calibration["train_dst"])
    if negative_policy not in {"random", "historical"}:
        raise ValueError("proxy calibration negative_policy must be random or historical")
    calibration_negative_sampler = (
        TemporalNegativeSampler.from_events(
            target_data["train_src"],
            target_data["train_dst"],
            target_data["train_ts"],
            seed=seed + 11117,
        )
        if negative_policy == "historical"
        else None
    )
    calibration_losses: list[float] = []
    for _ in range(calibration_steps):
        indices = rng.choice(
            len(calibration["train_src"]),
            size=batch_size,
            replace=len(calibration["train_src"]) < batch_size,
        )
        source = calibration["train_src"][indices]
        destination = calibration["train_dst"][indices]
        timestamps = calibration["train_ts"][indices]
        if calibration_negative_sampler is None:
            negative = rng.choice(destination_pool, size=batch_size, replace=True)
        else:
            sampleable = calibration_negative_sampler.sampleable_mask(
                source, timestamps, policy=negative_policy, fallback_policy="random"
            )
            if not sampleable.any():
                continue
            source = source[sampleable]
            destination = destination[sampleable]
            timestamps = timestamps[sampleable]
            negative = calibration_negative_sampler.sample_with_fallback(
                source, timestamps, count=1, policy=negative_policy, fallback_policy="random"
            )[0].reshape(-1)
        optimizer.zero_grad(set_to_none=True)
        positive_logits = _link_logits(
            model, head, target, calibration, source, destination, timestamps
        )
        negative_logits = _link_logits(
            model, head, target, calibration, source, negative, timestamps
        )
        positive_bce = F.binary_cross_entropy_with_logits(
            positive_logits, torch.ones_like(positive_logits), reduction="none"
        )
        negative_bce = F.binary_cross_entropy_with_logits(
            negative_logits, torch.zeros_like(negative_logits), reduction="none"
        )
        loss = (positive_bce + negative_bce).mean()
        total_loss = loss
        if (
            model.source_router is not None
            and target not in model.source_expert_names
            and regret_weight > 0.0
        ):
            with torch.no_grad():
                identity_positive = _link_logits(
                    model, head, target, calibration, source, destination, timestamps,
                    use_source_router=False,
                )
                identity_negative = _link_logits(
                    model, head, target, calibration, source, negative, timestamps,
                    use_source_router=False,
                )
            if regret_mode == "event_pairwise":
                routed_event = _pairwise_ranking_loss(positive_logits, negative_logits)
                identity_event = _pairwise_ranking_loss(identity_positive, identity_negative)
                regret = F.relu(routed_event - identity_event + regret_margin).mean()
            else:
                identity_loss = F.binary_cross_entropy_with_logits(
                    identity_positive, torch.ones_like(identity_positive)
                ) + F.binary_cross_entropy_with_logits(
                    identity_negative, torch.zeros_like(identity_negative)
                )
                regret = F.relu(loss - identity_loss + regret_margin)
            total_loss = loss + regret_weight * regret
        total_loss.backward()
        optimizer.step()
        calibration_losses.append(float(total_loss.detach().cpu()))

    model.eval()
    head.eval()
    return model, head, rng, destination_pool, calibration_losses


def _score_probe_partition(
    model: TemporalGraphFoundationModel,
    head: LinkPredictionHead,
    *,
    target: str,
    probe: dict[str, Any],
    rng: np.random.Generator,
    destination_pool: np.ndarray,
    batch_size: int,
    probe_batches: int,
) -> tuple[float, float]:
    if not len(probe["train_src"]):
        raise ValueError("proxy probe partition cannot be empty")

    sample_count = batch_size * probe_batches
    indices = rng.choice(
        len(probe["train_src"]),
        size=sample_count,
        replace=len(probe["train_src"]) < sample_count,
    )
    labels: list[int] = []
    scores: list[float] = []
    with torch.no_grad():
        for batch_index in range(probe_batches):
            selected = indices[
                batch_index * batch_size : (batch_index + 1) * batch_size
            ]
            source = probe["train_src"][selected]
            destination = probe["train_dst"][selected]
            timestamps = probe["train_ts"][selected]
            negative = rng.choice(destination_pool, size=len(selected), replace=True)
            positive_logits = _link_logits(
                model, head, target, probe, source, destination, timestamps
            )
            negative_logits = _link_logits(
                model, head, target, probe, source, negative, timestamps
            )
            batch_scores = torch.sigmoid(
                torch.cat((positive_logits, negative_logits))
            ).cpu()
            labels.extend([1] * len(selected) + [0] * len(selected))
            scores.extend(batch_scores.tolist())
    return (
        float(average_precision_score(labels, scores)),
        float(roc_auc_score(labels, scores)),
    )


def score_target_train_proxy(
    model: TemporalGraphFoundationModel,
    *,
    target: str,
    target_data: dict[str, Any],
    seed: int,
    batch_size: int,
    calibration_steps: int,
    probe_batches: int,
    calibration_fraction: float,
    learning_rate: float = 1e-3,
) -> dict[str, Any]:
    if calibration_steps <= 0 or probe_batches <= 0 or batch_size <= 0:
        raise ValueError("proxy steps, batches, and batch_size must be positive")
    calibration, probe = build_proxy_partitions(
        target_data, calibration_fraction
    )
    model, head, rng, destination_pool, calibration_losses = _calibrate_proxy(
        model,
        target=target,
        target_data=target_data,
        calibration=calibration,
        seed=seed,
        batch_size=batch_size,
        calibration_steps=calibration_steps,
        learning_rate=learning_rate,
    )
    proxy_ap, proxy_auc = _score_probe_partition(
        model,
        head,
        target=target,
        probe=probe,
        rng=rng,
        destination_pool=destination_pool,
        batch_size=batch_size,
        probe_batches=probe_batches,
    )
    return {
        "proxy_ap": proxy_ap,
        "proxy_auc": proxy_auc,
        "calibration_final_loss": calibration_losses[-1],
        "calibration_steps": calibration_steps,
        "probe_batches": probe_batches,
        "information_boundary": "target_training_period_only",
    }


def score_target_train_proxy_windows(
    model: TemporalGraphFoundationModel,
    *,
    target: str,
    target_data: dict[str, Any],
    seed: int,
    batch_size: int,
    calibration_steps: int,
    probe_batches: int,
    calibration_fraction: float,
    proxy_windows: int,
    bootstrap_replicates: int,
    bootstrap_confidence: float,
    risk_beta: float,
    size_penalty: float,
    cardinality: int,
    learning_rate: float = 1e-3,
    proxy_metric: str = 'ap',
    negatives_per_positive: int = 20,
    proxy_negative_policy: str = 'random',
    regret_weight: float = 0.0,
    regret_margin: float = 0.0,
    regret_mode: str = "batch_bce",
) -> dict[str, Any]:
    if calibration_steps <= 0 or probe_batches <= 0 or batch_size <= 0:
        raise ValueError("proxy steps, batches, and batch_size must be positive")
    calibration, windows = build_proxy_windows(
        target_data,
        calibration_fraction=calibration_fraction,
        window_count=proxy_windows,
    )
    model, head, rng, destination_pool, calibration_losses = _calibrate_proxy(
        model,
        target=target,
        target_data=target_data,
        calibration=calibration,
        seed=seed,
        batch_size=batch_size,
        calibration_steps=calibration_steps,
        learning_rate=learning_rate,
        negative_policy=proxy_negative_policy,
        regret_weight=regret_weight,
        regret_margin=regret_margin,
        regret_mode=regret_mode,
    )
    window_ap: list[float] = []
    window_auc: list[float] = []
    window_mrr: list[float] = []
    if proxy_metric not in ('ap', 'mrr'):
        raise ValueError('proxy_metric must be ap or mrr')
    sampler = (TemporalNegativeSampler.from_events(
        target_data['train_src'], target_data['train_dst'], target_data['train_ts'],
        seed=seed + 29009) if proxy_metric == 'mrr' else None)
    for window in windows:
        proxy_ap, proxy_auc = _score_probe_partition(
            model,
            head,
            target=target,
            probe=window,
            rng=rng,
            destination_pool=destination_pool,
            batch_size=batch_size,
            probe_batches=probe_batches,
        )
        window_ap.append(proxy_ap)
        window_auc.append(proxy_auc)
        if sampler is not None:
            # Restrict queries, destination universe, and exclusion history to training.
            # Deterministic evenly spaced probes avoid replacement duplicates.
            count = min(len(window['train_src']), batch_size * probe_batches)
            indices = np.linspace(0, len(window['train_src']) - 1, count, dtype=int)
            probe = {key: (value[indices] if key.startswith('train_') else value)
                     for key, value in window.items()}
            window_mrr.append(evaluate_link_ranking(model, head, target, probe, 'train',
                sampler, negatives_per_positive=negatives_per_positive,
                batch_size=batch_size, policy=proxy_negative_policy)['mrr'])
    window_scores = window_mrr if proxy_metric == 'mrr' else window_ap
    summary = summarize_proxy_windows(
        window_scores,
        seed=seed + 13007,
        bootstrap_replicates=bootstrap_replicates,
        confidence=bootstrap_confidence,
        risk_beta=risk_beta,
        cardinality=cardinality,
        size_penalty=size_penalty,
    )
    utility = summary.pop("risk_adjusted_utility")
    return {
        "window_proxy_ap": window_ap,
        "window_proxy_mrr": window_mrr,
        "window_proxy_scores": window_scores,
        "proxy_metric": proxy_metric,
        "proxy_negatives_per_positive": negatives_per_positive if sampler is not None else 1,
        "proxy_negative_policy": proxy_negative_policy if sampler is not None else "binary_random",
        "window_proxy_auc": window_auc,
        "bootstrap": summary,
        "risk_adjusted_utility": utility,
        "calibration_final_loss": calibration_losses[-1],
        "calibration_steps": calibration_steps,
        "probe_batches_per_window": probe_batches,
        "proxy_windows": proxy_windows,
        "information_boundary": "target_training_period_only",
        "proxy_regret_weight": regret_weight,
        "proxy_regret_margin": regret_margin,
    }


def score_checkpoint_proxy(
    checkpoint_path: str | Path,
    *,
    target: str,
    processed_dir: str | Path,
    seed: int,
    batch_size: int,
    calibration_steps: int,
    probe_batches: int,
    calibration_fraction: float,
    device: str = "cpu",
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = load_foundation_checkpoint_payload(payload).to(torch.device(device))
    target_data = load_processed_dataset(target, processed_dir)
    return score_target_train_proxy(
        model,
        target=target,
        target_data=target_data,
        seed=seed,
        batch_size=batch_size,
        calibration_steps=calibration_steps,
        probe_batches=probe_batches,
        calibration_fraction=calibration_fraction,
    )


def score_checkpoint_proxy_windows(
    checkpoint_path: str | Path,
    *,
    target: str,
    processed_dir: str | Path,
    seed: int,
    batch_size: int,
    calibration_steps: int,
    probe_batches: int,
    calibration_fraction: float,
    proxy_windows: int,
    bootstrap_replicates: int,
    bootstrap_confidence: float,
    risk_beta: float,
    size_penalty: float,
    cardinality: int,
    device: str = "cpu",
    proxy_metric: str = 'ap',
    negatives_per_positive: int = 20,
    proxy_negative_policy: str = 'random',
    regret_weight: float = 0.0,
    regret_margin: float = 0.0,
    regret_mode: str = "batch_bce",
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = load_foundation_checkpoint_payload(payload).to(torch.device(device))
    target_data = load_processed_dataset(target, processed_dir)
    return score_target_train_proxy_windows(
        model,
        target=target,
        target_data=target_data,
        seed=seed,
        batch_size=batch_size,
        calibration_steps=calibration_steps,
        probe_batches=probe_batches,
        calibration_fraction=calibration_fraction,
        proxy_windows=proxy_windows,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_confidence=bootstrap_confidence,
        risk_beta=risk_beta,
        size_penalty=size_penalty,
        cardinality=cardinality,
        proxy_metric=proxy_metric,
        negatives_per_positive=negatives_per_positive,
        proxy_negative_policy=proxy_negative_policy,
        regret_weight=regret_weight,
        regret_margin=regret_margin,
        regret_mode=regret_mode,
    )
