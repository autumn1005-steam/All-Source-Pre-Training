"""Compute-matched multi-source TempoFound pre-training."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .budget import allocate_batch_budget
from .checkpoint import build_checkpoint
from .config import ExperimentConfig
from .data import load_processed_dataset
from .model import DatasetSpec, TemporalGraphFoundationModel
from .optimization import allocate_weighted_batch_budget, project_conflicting_gradients
from .sampling import TemporalNegativeSampler


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _resolve_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    return resolved


def _dataset_spec(name: str, data: dict[str, Any]) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        raw_node_dim=int(data["node_features"].shape[1]),
        raw_edge_dim=int(data["edge_features"].shape[1]),
        n_nodes=int(data["node_features"].shape[0] - 1),
        n_edges=int(data["edge_features"].shape[0] - 1),
    )


def _normalized_source_weights(
    sources: tuple[str, ...], weights: Mapping[str, float] | None
) -> dict[str, float]:
    if weights is None:
        return {name: 1.0 / len(sources) for name in sources}
    if set(weights) != set(sources):
        raise ValueError("source_weights must contain exactly the selected sources")
    normalized = {name: float(weights[name]) for name in sources}
    if any(not math.isfinite(value) or value < 0.0 for value in normalized.values()):
        raise ValueError("source_weights must be finite and non-negative")
    total = sum(normalized.values())
    if total <= 0.0:
        raise ValueError("at least one source weight must be positive")
    return {name: value / total for name, value in normalized.items()}


def _make_pretraining_batch(
    model: TemporalGraphFoundationModel,
    data: dict[str, Any],
    selected: np.ndarray,
    rng: np.random.Generator,
    negative_sampler: TemporalNegativeSampler | None = None,
) -> dict[str, Any] | None:
    source = data["train_src"][selected]
    destination = data["train_dst"][selected]
    timestamps = data["train_ts"][selected]
    # Construct negatives from the same training history and exclude every
    # destination already observed for the source up to the query time.
    # This prevents positive-label collisions in the multi-task objective.
    sampler = negative_sampler or TemporalNegativeSampler.from_events(
        data["train_src"], data["train_dst"], data["train_ts"], seed=0
    )
    sampleable = sampler.sampleable_mask(
        source, timestamps, policy="random", fallback_policy="random"
    )
    if not sampleable.any():
        return None
    source = source[sampleable]
    destination = destination[sampleable]
    timestamps = timestamps[sampleable]
    selected = selected[sampleable]
    negative = sampler.sample(
        source, timestamps, count=1, policy="random", query_ids=selected
    ).reshape(-1)
    statistics = model.compute_neighborhood_stats(
        data["neighbor_finder"], source, timestamps
    )
    return {
        "src_idx": source,
        "dst_idx": destination,
        "neg_dst_idx": negative,
        "cut_time": timestamps,
        "time_gaps": np.expm1(statistics[:, 1]),
        "ngh_stats": statistics,
        "node_props": statistics,
    }


def _conflict_aware_step(
    model: TemporalGraphFoundationModel,
    optimizer: torch.optim.Optimizer,
    source_losses: list[tuple[str, torch.Tensor]],
) -> dict[str, float]:
    """Merge one group of source losses, projecting shared encoder conflicts."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    shared_ids = {id(parameter) for parameter in model.encoder.parameters()}
    if model.temporal_mixture is not None:
        shared_ids.update(id(parameter) for parameter in model.temporal_mixture.parameters())
    if model.source_expert_bank is not None:
        shared_ids.update(id(parameter) for parameter in model.source_expert_bank.parameters())
    shared_indices = [
        index for index, parameter in enumerate(parameters) if id(parameter) in shared_ids
    ]
    if not shared_indices:
        raise RuntimeError("shared encoder has no trainable parameters")

    gradients: list[tuple[torch.Tensor | None, ...]] = []
    for _, loss in source_losses:
        gradients.append(
            torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=False)
        )
    shared_vectors = [
        torch.cat(
            [
                (
                    source_gradient[index]
                    if source_gradient[index] is not None
                    else torch.zeros_like(parameters[index])
                ).reshape(-1)
                for index in shared_indices
            ]
        )
        for source_gradient in gradients
    ]
    projected, stats = project_conflicting_gradients(shared_vectors)

    optimizer.zero_grad(set_to_none=True)
    mean_projected = torch.stack(projected).mean(dim=0)
    offset = 0
    for index in shared_indices:
        parameter = parameters[index]
        length = parameter.numel()
        parameter.grad = mean_projected[offset : offset + length].reshape_as(parameter).clone()
        offset += length

    shared_index_set = set(shared_indices)
    group_size = len(source_losses)
    for index, parameter in enumerate(parameters):
        if index in shared_index_set:
            continue
        contributions = [
            source_gradient[index]
            for source_gradient in gradients
            if source_gradient[index] is not None
        ]
        if contributions:
            parameter.grad = torch.stack(contributions).sum(dim=0) / group_size
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
    optimizer.step()
    return stats


def run_pretraining(
    config: ExperimentConfig,
    sources: tuple[str, ...],
    seed: int,
    processed_dir: str | Path,
    output_path: str | Path,
    device: str = "cpu",
    source_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if len(sources) != len(set(sources)) or not sources:
        raise ValueError("sources must contain unique dataset names")
    unknown = set(sources).difference(config.datasets)
    if unknown:
        raise ValueError(f"sources are not configured: {sorted(unknown)}")
    _set_seed(seed)
    torch_device = _resolve_device(device)
    data_by_source = {
        name: load_processed_dataset(name, processed_dir) for name in sources
    }
    model = TemporalGraphFoundationModel(
        tuple(_dataset_spec(name, data_by_source[name]) for name in sources),
        shared_dim=config.shared_dim,
        time_dim=config.time_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        num_neighbors=config.num_neighbors,
        temporal_experts=config.temporal_experts,
        expert_identity_init=config.expert_identity_init,
        source_experts=config.source_experts,
        risk_controlled_routing=config.risk_controlled_routing,
        router_temperature=config.router_temperature,
        router_prior_strength=config.router_prior_strength,
        router_entropy_penalty=config.router_entropy_penalty,
        router_abstain_bias=config.router_abstain_bias,
        router_local_acceptance=config.router_local_acceptance,
        time_encoding_mode=config.time_encoding_mode,
        backbone=config.backbone,
    ).to(torch_device)
    for name, data in data_by_source.items():
        model.load_dataset_features(
            name, data["node_features"], data["edge_features"]
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.pretrain_epochs, 1)
    )
    epoch_batch_counts: list[int] = []
    source_batch_counts = {name: 0 for name in sources}
    epoch_mean_losses: list[float] = []
    rng = np.random.default_rng(seed)
    normalized_weights = _normalized_source_weights(sources, source_weights)
    model.set_source_routing_prior(normalized_weights)
    negative_samplers = {
        name: TemporalNegativeSampler.from_events(
            data_by_source[name]["train_src"], data_by_source[name]["train_dst"],
            data_by_source[name]["train_ts"], seed=seed + 41003
        ) for name in sources
    }
    total_conflict_pairs = 0.0
    total_gradient_pairs = 0.0
    cosine_before_sum = 0.0
    cosine_after_sum = 0.0
    optimizer_step_counts: list[int] = []

    for epoch in range(config.pretrain_epochs):
        model.train()
        order = list(sources)
        random.Random(seed + epoch).shuffle(order)
        capacities = {
            name: math.ceil(len(data_by_source[name]["train_src"]) / config.batch_size)
            for name in order
        }
        if config.pretraining_strategy == "conflict_aware":
            budgets = allocate_weighted_batch_budget(
                capacities,
                config.total_batches_per_epoch,
                normalized_weights,
                minimum=config.minimum_source_batches,
            )
        else:
            budgets = allocate_batch_budget(
                capacities, config.total_batches_per_epoch
            )
        losses: list[float] = []
        source_batches: dict[str, list[np.ndarray]] = {}
        for name in order:
            data = data_by_source[name]
            indices = rng.permutation(len(data["train_src"]))
            source_batches[name] = []
            for batch_index in range(budgets[name]):
                start = batch_index * config.batch_size
                stop = min(start + config.batch_size, len(indices))
                selected = indices[start:stop]
                if len(selected):
                    source_batches[name].append(selected)

        epoch_steps = 0
        if config.pretraining_strategy == "legacy_sequential":
            for name in order:
                data = data_by_source[name]
                for selected in source_batches[name]:
                    batch = _make_pretraining_batch(model, data, selected, rng, negative_samplers[name])
                    if batch is None:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    task_losses = model.pretrain_forward(
                        name,
                        data["neighbor_finder"],
                        batch,
                        domain_adversarial_scale=config.domain_adversarial_scale,
                    )
                    total = model.weighted_pretrain_loss(
                        task_losses,
                        domain_coefficient=config.domain_loss_coefficient,
                    )
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    losses.append(float(total.detach().cpu()))
                    source_batch_counts[name] += 1
                    epoch_steps += 1
        else:
            max_groups = max((len(batches) for batches in source_batches.values()), default=0)
            for group_index in range(max_groups):
                grouped_losses: list[tuple[str, torch.Tensor]] = []
                for name in order:
                    if group_index >= len(source_batches[name]):
                        continue
                    data = data_by_source[name]
                    batch = _make_pretraining_batch(
                        model,
                        data,
                        source_batches[name][group_index],
                        rng,
                        negative_samplers[name],
                    )
                    if batch is None:
                        continue
                    task_losses = model.pretrain_forward(
                        name,
                        data["neighbor_finder"],
                        batch,
                        domain_adversarial_scale=config.domain_adversarial_scale,
                    )
                    total = model.weighted_pretrain_loss(
                        task_losses,
                        domain_coefficient=config.domain_loss_coefficient,
                    )
                    grouped_losses.append((name, total))
                    losses.append(float(total.detach().cpu()))
                    source_batch_counts[name] += 1
                stats = _conflict_aware_step(model, optimizer, grouped_losses)
                pairs = stats["total_pairs"]
                total_conflict_pairs += stats["conflict_pairs"]
                total_gradient_pairs += pairs
                cosine_before_sum += stats["mean_cosine_before"] * pairs
                cosine_after_sum += stats["mean_cosine_after"] * pairs
                epoch_steps += 1
        epoch_count = len(losses)
        if epoch_count > config.total_batches_per_epoch:
            raise RuntimeError("global batch budget was exceeded")
        epoch_batch_counts.append(epoch_count)
        epoch_mean_losses.append(float(np.mean(losses)) if losses else float("nan"))
        optimizer_step_counts.append(epoch_steps)
        scheduler.step()

    summary = {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "temporal_expert_parameters": (sum(p.numel() for p in model.temporal_mixture.parameters())
                                        if model.temporal_mixture is not None else 0),
        "source_expert_parameters": (sum(p.numel() for p in model.source_expert_bank.parameters())
                                     if model.source_expert_bank is not None else 0),
        "source_router_parameters": (sum(p.numel() for p in model.source_router.parameters())
                                     if model.source_router is not None else 0),
        "risk_controlled_routing": config.risk_controlled_routing,
        "sources": list(sources),
        "seed": seed,
        "time_encoding_mode": config.time_encoding_mode,
        "epoch_batch_counts": epoch_batch_counts,
        "source_batch_counts": source_batch_counts,
        "epoch_mean_losses": epoch_mean_losses,
        "global_batch_cap": config.total_batches_per_epoch,
        "pretraining_strategy": config.pretraining_strategy,
        "source_weights": normalized_weights,
        "optimizer_step_counts": optimizer_step_counts,
        "conflict_pairs": int(total_conflict_pairs),
        "gradient_pairs": int(total_gradient_pairs),
        "conflict_rate": (
            total_conflict_pairs / total_gradient_pairs
            if total_gradient_pairs
            else 0.0
        ),
        "mean_cosine_before": (
            cosine_before_sum / total_gradient_pairs if total_gradient_pairs else 0.0
        ),
        "mean_cosine_after": (
            cosine_after_sum / total_gradient_pairs if total_gradient_pairs else 0.0
        ),
    }
    payload = build_checkpoint(
        model,
        time_encoding_mode=config.time_encoding_mode,
        model_config=model.model_config(),
        metadata={"pretraining": summary, "experiment": config.name},
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return summary
