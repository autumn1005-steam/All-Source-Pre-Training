"""Unified TempoFound model with released-checkpoint-compatible parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .graph import NeighborFinder
from .layers import AttnModel, HarmonicTimeEncoder, MergeLayer, gradient_reverse


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    raw_node_dim: int
    raw_edge_dim: int
    n_nodes: int
    n_edges: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "DatasetSpec":
        return cls(
            name=str(value["name"]),
            raw_node_dim=int(value["raw_node_dim"]),
            raw_edge_dim=int(value["raw_edge_dim"]),
            n_nodes=int(value["n_nodes"]),
            n_edges=int(value.get("n_edges", 1)),
        )


class DatasetAdapter(nn.Module):
    def __init__(
        self,
        raw_node_dim: int,
        raw_edge_dim: int,
        shared_dim: int,
        dataset_id: int,
        num_datasets: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.raw_node_dim = raw_node_dim
        self.raw_edge_dim = raw_edge_dim
        self.shared_dim = shared_dim
        self.dataset_id = dataset_id
        self.node_project = nn.Sequential(
            nn.Linear(raw_node_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, shared_dim),
        )
        self.edge_project = nn.Sequential(
            nn.Linear(raw_edge_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, shared_dim),
        )
        self.dataset_embed = nn.Embedding(num_datasets, shared_dim)

    def forward(
        self, node_features: torch.Tensor, edge_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.node_project(node_features), self.edge_project(edge_features)


class SharedTemporalEncoder(nn.Module):
    """Recursive TGAT-style encoder shared across dataset-specific adapters."""

    def __init__(
        self,
        shared_dim: int,
        time_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        time_encoding_mode: str,
    ) -> None:
        super().__init__()
        self.shared_dim = shared_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.time_encoder = HarmonicTimeEncoder(time_dim, time_encoding_mode)
        self.time_project = nn.Linear(time_dim, shared_dim)
        self.attn_layers = nn.ModuleList(
            AttnModel(
                shared_dim,
                shared_dim,
                shared_dim,
                n_head=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.merge_layer = MergeLayer(shared_dim, shared_dim, shared_dim, shared_dim)
        self._feature_fn: Callable[[np.ndarray], torch.Tensor] | None = None
        self._edge_feature_fn: Callable[[np.ndarray], torch.Tensor] | None = None

    def set_feature_fn(self, function: Callable[[np.ndarray], torch.Tensor]) -> None:
        self._feature_fn = function

    def set_edge_feature_fn(
        self, function: Callable[[np.ndarray], torch.Tensor]
    ) -> None:
        self._edge_feature_fn = function

    def forward(
        self,
        node_ids: np.ndarray,
        query_times: np.ndarray,
        *,
        num_neighbors: int,
        neighbor_finder: NeighborFinder,
    ) -> torch.Tensor:
        if self._feature_fn is None or self._edge_feature_fn is None:
            raise RuntimeError("dataset feature callbacks are not configured")
        return self._temporal_convolution(
            np.asarray(node_ids, dtype=np.int64),
            np.asarray(query_times, dtype=np.float64),
            neighbor_finder,
            self.num_layers,
            num_neighbors,
        )

    def _temporal_convolution(
        self,
        node_ids: np.ndarray,
        query_times: np.ndarray,
        finder: NeighborFinder,
        layer: int,
        num_neighbors: int,
    ) -> torch.Tensor:
        if self._feature_fn is None or self._edge_feature_fn is None:
            raise RuntimeError("dataset feature callbacks are not configured")
        base = self._feature_fn(node_ids)
        if layer == 0:
            return base
        previous = self._temporal_convolution(
            node_ids, query_times, finder, layer - 1, num_neighbors
        )
        neighbor_nodes, neighbor_edges, neighbor_times = finder.get_temporal_neighbor(
            node_ids, query_times, num_neighbors
        )
        neighbor_previous = self._temporal_convolution(
            neighbor_nodes.reshape(-1),
            neighbor_times.reshape(-1),
            finder,
            layer - 1,
            num_neighbors,
        ).reshape(len(node_ids), num_neighbors, self.shared_dim)
        edge_features = self._edge_feature_fn(neighbor_edges.reshape(-1)).reshape(
            len(node_ids), num_neighbors, self.shared_dim
        )
        device = base.device
        query_tensor = torch.as_tensor(
            query_times, dtype=torch.float32, device=device
        )
        delta = query_tensor[:, None] - torch.as_tensor(
            neighbor_times, dtype=torch.float32, device=device
        )
        source_time = self.time_project(
            self.time_encoder(torch.zeros(len(node_ids), 1, device=device))
        )
        neighbor_time = self.time_project(self.time_encoder(delta))
        mask = torch.as_tensor(neighbor_nodes == 0, dtype=torch.bool, device=device)
        local, _ = self.attn_layers[layer - 1](
            previous,
            source_time,
            neighbor_previous,
            neighbor_time,
            edge_features,
            mask,
        )
        return self.merge_layer(local, previous)


class GraphMixerTemporalEncoder(nn.Module):
    """Non-recursive sequence-mixing control with the same feature callbacks."""

    def __init__(
        self,
        shared_dim: int,
        time_dim: int,
        num_layers: int,
        dropout: float,
        time_encoding_mode: str,
    ) -> None:
        super().__init__()
        self.shared_dim = shared_dim
        self.time_encoder = HarmonicTimeEncoder(time_dim, time_encoding_mode)
        self.time_project = nn.Linear(time_dim, shared_dim)
        self.mixers = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(shared_dim),
                nn.Linear(shared_dim, shared_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(shared_dim * 2, shared_dim),
            )
            for _ in range(num_layers)
        )
        self.merge_layer = MergeLayer(shared_dim, shared_dim, shared_dim, shared_dim)
        self._feature_fn: Callable[[np.ndarray], torch.Tensor] | None = None
        self._edge_feature_fn: Callable[[np.ndarray], torch.Tensor] | None = None

    def set_feature_fn(self, function: Callable[[np.ndarray], torch.Tensor]) -> None:
        self._feature_fn = function

    def set_edge_feature_fn(
        self, function: Callable[[np.ndarray], torch.Tensor]
    ) -> None:
        self._edge_feature_fn = function

    def forward(
        self,
        node_ids: np.ndarray,
        query_times: np.ndarray,
        *,
        num_neighbors: int,
        neighbor_finder: NeighborFinder,
    ) -> torch.Tensor:
        if self._feature_fn is None or self._edge_feature_fn is None:
            raise RuntimeError("dataset feature callbacks are not configured")
        base = self._feature_fn(node_ids)
        neighbor_nodes, neighbor_edges, neighbor_times = (
            neighbor_finder.get_temporal_neighbor(
                node_ids, query_times, num_neighbors
            )
        )
        neighbors = self._feature_fn(neighbor_nodes.reshape(-1)).reshape(
            len(node_ids), num_neighbors, self.shared_dim
        )
        edges = self._edge_feature_fn(neighbor_edges.reshape(-1)).reshape(
            len(node_ids), num_neighbors, self.shared_dim
        )
        delta = torch.as_tensor(
            np.asarray(query_times)[:, None] - neighbor_times,
            dtype=torch.float32,
            device=base.device,
        )
        tokens = neighbors + edges + self.time_project(self.time_encoder(delta))
        for mixer in self.mixers:
            tokens = tokens + mixer(tokens)
        valid = torch.as_tensor(
            neighbor_nodes != 0, dtype=tokens.dtype, device=base.device
        ).unsqueeze(-1)
        aggregate = (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.merge_layer(aggregate, base)


class LinkPredictionHead(nn.Module):
    def __init__(self, shared_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, 1),
        )

    def forward(self, source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
        return self.scorer(torch.cat((source, destination), dim=-1)).squeeze(-1)


class NeighborhoodPredictionHead(nn.Module):
    def __init__(self, shared_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, 5),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.predictor(embeddings)


class DomainDiscriminatorHead(nn.Module):
    def __init__(self, shared_dim: int, num_datasets: int, dropout: float) -> None:
        super().__init__()
        hidden = max(shared_dim // 2, 1)
        self.discriminator = nn.Sequential(
            nn.Linear(shared_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_datasets),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.discriminator(embeddings)


class TimeIntervalPredictionHead(nn.Module):
    def __init__(self, shared_dim: int, dropout: float) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, 1),
        )

    def forward(self, source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat((source, destination), dim=-1)).squeeze(-1)


class NodePropertyHead(nn.Module):
    def __init__(self, shared_dim: int, dropout: float) -> None:
        super().__init__()
        hidden = max(shared_dim // 2, 1)
        self.predictor = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 5),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.predictor(embeddings)


def edge_direction_consistency_loss(
    forward_logits: torch.Tensor, reverse_logits: torch.Tensor
) -> torch.Tensor:
    return torch.mean(
        (torch.sigmoid(forward_logits) - torch.sigmoid(reverse_logits)) ** 2
    )


class TemporalGraphFoundationModel(nn.Module):
    def __init__(
        self,
        dataset_specs: tuple[DatasetSpec, ...],
        *,
        shared_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        num_neighbors: int = 10,
        time_encoding_mode: str = "paper_sincos",
        backbone: str = "tgat",
        temporal_experts: int = 0,
        expert_identity_init: bool = False,
        source_experts: bool = False,
        source_expert_names: tuple[str, ...] | None = None,
        risk_controlled_routing: bool = False,
        router_temperature: float = 1.0,
        router_prior_strength: float = 1.0,
        router_entropy_penalty: float = 1.0,
        router_abstain_bias: float = -1.0,
        router_local_acceptance: bool = True,
    ) -> None:
        super().__init__()
        if not dataset_specs:
            raise ValueError("at least one dataset specification is required")
        names = [spec.name for spec in dataset_specs]
        if len(names) != len(set(names)):
            raise ValueError("dataset names must be unique")
        self.dataset_specs = tuple(dataset_specs)
        self.dataset_configs = [spec.to_dict() for spec in dataset_specs]
        self.dataset_names = names
        self.name_to_id = {name: index for index, name in enumerate(names)}
        self.shared_dim = shared_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_neighbors = num_neighbors
        self.time_encoding_mode = time_encoding_mode
        if backbone not in {"tgat", "graphmixer"}:
            raise ValueError("backbone must be tgat or graphmixer")
        self.backbone = backbone
        self.temporal_experts = temporal_experts
        self.expert_identity_init = expert_identity_init
        self.source_experts = source_experts
        self.risk_controlled_routing = bool(risk_controlled_routing)
        self.router_temperature = float(router_temperature)
        self.router_prior_strength = float(router_prior_strength)
        self.router_entropy_penalty = float(router_entropy_penalty)
        self.router_abstain_bias = float(router_abstain_bias)
        self.router_local_acceptance = bool(router_local_acceptance)
        if source_expert_names is None:
            source_expert_names = tuple(names)
        else:
            source_expert_names = tuple(str(name) for name in source_expert_names)
        if source_experts:
            if not source_expert_names or len(source_expert_names) != len(set(source_expert_names)):
                raise ValueError("source_expert_names must be non-empty and unique")
            unknown_expert_names = set(source_expert_names).difference(names)
            if unknown_expert_names:
                raise ValueError(
                    f"source_expert_names missing from datasets: {sorted(unknown_expert_names)}"
                )
        self.source_expert_names = source_expert_names if source_experts else ()
        if self.risk_controlled_routing and not source_experts:
            raise ValueError("risk_controlled_routing requires source_experts=True")
        num_datasets = len(dataset_specs)

        self.adapters = nn.ModuleDict(
            {
                spec.name: DatasetAdapter(
                    spec.raw_node_dim,
                    spec.raw_edge_dim,
                    shared_dim,
                    self.name_to_id[spec.name],
                    num_datasets,
                    dropout,
                )
                for spec in dataset_specs
            }
        )
        if backbone == "tgat":
            self.encoder = SharedTemporalEncoder(
                shared_dim,
                time_dim,
                num_layers,
                num_heads,
                dropout,
                time_encoding_mode,
            )
        else:
            self.encoder = GraphMixerTemporalEncoder(
                shared_dim,
                time_dim,
                num_layers,
                dropout,
                time_encoding_mode,
            )
        self.dataset_node_embed = nn.ModuleDict(
            {
                spec.name: nn.Embedding(
                    spec.n_nodes + 1, spec.raw_node_dim, padding_idx=0
                )
                for spec in dataset_specs
            }
        )
        self.dataset_edge_embed = nn.ModuleDict(
            {
                spec.name: nn.Embedding(
                    spec.n_edges + 1, spec.raw_edge_dim, padding_idx=0
                )
                for spec in dataset_specs
            }
        )
        for table in (*self.dataset_node_embed.values(), *self.dataset_edge_embed.values()):
            table.weight.requires_grad_(False)
        self.link_head = LinkPredictionHead(shared_dim, dropout)
        self.neighborhood_head = NeighborhoodPredictionHead(shared_dim, dropout)
        self.domain_head = DomainDiscriminatorHead(shared_dim, num_datasets, dropout)
        self.time_shift_head = TimeIntervalPredictionHead(shared_dim, dropout)
        self.node_property_head = NodePropertyHead(shared_dim, dropout)
        self.task_log_vars = nn.Parameter(torch.zeros(5))
        from .temporal_experts import TemporalResidualExperts
        self.temporal_mixture = None
        if temporal_experts:
            if expert_identity_init:
                # Construct on CPU without perturbing subsequent head/adapter RNG.
                with torch.random.fork_rng(devices=[]):
                    self.temporal_mixture = TemporalResidualExperts(shared_dim, temporal_experts, dropout)
                    for expert in self.temporal_mixture.experts:
                        nn.init.zeros_(expert[-1].weight)
                        nn.init.zeros_(expert[-1].bias)
            else:
                self.temporal_mixture = TemporalResidualExperts(shared_dim, temporal_experts, dropout)
        from .temporal_experts import RiskControlledSourceRouter, SourceExpertBank
        self.source_expert_bank = (
            SourceExpertBank(self.source_expert_names, shared_dim, dropout)
            if source_experts
            else None
        )
        if self.source_expert_bank is not None and expert_identity_init:
            for expert in self.source_expert_bank.experts.values():
                nn.init.zeros_(expert[-1].weight)
                nn.init.zeros_(expert[-1].bias)
        self.source_router = None
        if self.source_expert_bank is not None and self.risk_controlled_routing:
            self.source_router = RiskControlledSourceRouter(
                self.source_expert_names,
                shared_dim,
                dropout,
                temperature=self.router_temperature,
                prior_strength=self.router_prior_strength,
                entropy_penalty=self.router_entropy_penalty,
                abstain_bias=self.router_abstain_bias,
                local_acceptance=self.router_local_acceptance,
            )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def model_config(self) -> dict[str, Any]:
        return {
            "dataset_specs": [spec.to_dict() for spec in self.dataset_specs],
            "shared_dim": self.shared_dim,
            "time_dim": self.time_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "num_neighbors": self.num_neighbors,
            "time_encoding_mode": self.time_encoding_mode,
            "backbone": self.backbone,
            "temporal_experts": self.temporal_experts,
            "expert_identity_init": self.expert_identity_init,
            "source_experts": self.source_experts,
            "source_expert_names": list(self.source_expert_names),
            "risk_controlled_routing": self.risk_controlled_routing,
            "router_temperature": self.router_temperature,
            "router_prior_strength": self.router_prior_strength,
            "router_entropy_penalty": self.router_entropy_penalty,
            "router_abstain_bias": self.router_abstain_bias,
            "router_local_acceptance": self.router_local_acceptance,
        }

    def load_dataset_features(
        self, name: str, node_features: np.ndarray, edge_features: np.ndarray
    ) -> None:
        if name not in self.adapters:
            raise ValueError(f"unknown dataset: {name}")
        node_tensor = torch.as_tensor(node_features, dtype=torch.float32)
        edge_tensor = torch.as_tensor(edge_features, dtype=torch.float32)
        if node_tensor.shape != self.dataset_node_embed[name].weight.shape:
            raise ValueError(
                f"node feature shape for {name} is {tuple(node_tensor.shape)}, "
                f"expected {tuple(self.dataset_node_embed[name].weight.shape)}"
            )
        if edge_tensor.shape != self.dataset_edge_embed[name].weight.shape:
            raise ValueError(
                f"edge feature shape for {name} is {tuple(edge_tensor.shape)}, "
                f"expected {tuple(self.dataset_edge_embed[name].weight.shape)}"
            )
        with torch.no_grad():
            self.dataset_node_embed[name].weight.copy_(node_tensor.to(self.device))
            self.dataset_edge_embed[name].weight.copy_(edge_tensor.to(self.device))

    load_dataset_embeddings = load_dataset_features

    def _raw_features(
        self, name: str, node_ids: np.ndarray, edge_ids: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_tensor = torch.as_tensor(node_ids, dtype=torch.long, device=self.device)
        edge_tensor = torch.as_tensor(edge_ids, dtype=torch.long, device=self.device)
        return self.dataset_node_embed[name](node_tensor), self.dataset_edge_embed[name](
            edge_tensor
        )

    def encode_nodes(
        self,
        name: str,
        finder: NeighborFinder,
        node_ids: np.ndarray,
        query_times: np.ndarray,
        *,
        use_source_router: bool = True,
    ) -> torch.Tensor:
        if name not in self.adapters:
            raise ValueError(f"unknown dataset: {name}")
        adapter = self.adapters[name]
        dataset_id = torch.tensor(self.name_to_id[name], device=self.device)
        domain_embedding = adapter.dataset_embed(dataset_id)

        def node_features(ids: np.ndarray) -> torch.Tensor:
            raw_nodes, _ = self._raw_features(
                name, ids, np.zeros(len(ids), dtype=np.int64)
            )
            projected = adapter.node_project(raw_nodes.float())
            return projected + domain_embedding.unsqueeze(0).expand(len(ids), -1)

        def edge_features(ids: np.ndarray) -> torch.Tensor:
            _, raw_edges = self._raw_features(
                name, np.zeros(len(ids), dtype=np.int64), ids
            )
            return adapter.edge_project(raw_edges.float())

        self.encoder.set_feature_fn(node_features)
        self.encoder.set_edge_feature_fn(edge_features)
        embeddings = self.encoder(
            np.asarray(node_ids, dtype=np.int64),
            np.asarray(query_times, dtype=np.float64),
            num_neighbors=self.num_neighbors,
            neighbor_finder=finder,
        )
        if self.temporal_mixture is not None:
            stats = torch.as_tensor(self.compute_neighborhood_stats(finder, node_ids, query_times),
                                    dtype=embeddings.dtype, device=embeddings.device)
            embeddings = self.temporal_mixture(embeddings, stats)
        if self.source_expert_bank is not None:
            stats = torch.as_tensor(
                self.compute_neighborhood_stats(finder, node_ids, query_times),
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            if name in self.source_expert_bank.experts:
                embeddings = self.source_expert_bank.forward_source(name, embeddings, stats)
            elif self.source_router is not None and use_source_router:
                source_deltas = self.source_expert_bank.stack_target_deltas(embeddings, stats)
                embeddings = self.source_router(embeddings, stats, source_deltas)
            elif self.source_router is None:
                raise ValueError(
                    f"dataset {name!r} has no source expert; use risk-controlled routing "
                    "for unseen targets or extend the legacy expert bank"
                )
        return embeddings

    def set_source_routing_prior(self, weights: dict[str, float]) -> None:
        if self.source_router is None:
            return
        self.source_router.set_prior(weights)

    def compute_neighborhood_stats(
        self, finder: NeighborFinder, node_ids: np.ndarray, query_times: np.ndarray
    ) -> np.ndarray:
        neighbor_nodes, _, neighbor_times = finder.get_temporal_neighbor(
            np.asarray(node_ids), np.asarray(query_times), self.num_neighbors
        )
        stats = np.zeros((len(node_ids), 5), dtype=np.float32)
        for row, query_time in enumerate(query_times):
            valid = (neighbor_nodes[row] != 0) & (neighbor_times[row] < query_time)
            count = int(valid.sum())
            stats[row, 0] = np.log1p(count)
            if count:
                delta = float(query_time) - neighbor_times[row][valid]
                stats[row, 1] = np.log1p(delta.min())
                stats[row, 2] = np.log1p(delta.mean())
                stats[row, 3] = np.log1p(delta.max())
                stats[row, 4] = min(count / self.num_neighbors, 1.0)
        return stats

    def pretrain_forward(
        self,
        name: str,
        finder: NeighborFinder,
        batch: dict[str, Any],
        *,
        domain_adversarial_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        source = self.encode_nodes(name, finder, batch["src_idx"], batch["cut_time"])
        destination = self.encode_nodes(
            name, finder, batch["dst_idx"], batch["cut_time"]
        )
        negative = self.encode_nodes(
            name, finder, batch["neg_dst_idx"], batch["cut_time"]
        )
        positive_logits = self.link_head(source, destination)
        negative_logits = self.link_head(source, negative)
        mtlp = F.binary_cross_entropy_with_logits(
            positive_logits, torch.ones_like(positive_logits)
        ) + F.binary_cross_entropy_with_logits(
            negative_logits, torch.zeros_like(negative_logits)
        )
        neighborhood_target = torch.as_tensor(
            batch["ngh_stats"], dtype=torch.float32, device=self.device
        )
        property_target = torch.as_tensor(
            batch["node_props"], dtype=torch.float32, device=self.device
        )
        tnp = F.mse_loss(self.neighborhood_head(source), neighborhood_target)
        domain_target = torch.full(
            (len(source),), self.name_to_id[name], dtype=torch.long, device=self.device
        )
        domain = F.cross_entropy(
            self.domain_head(
                gradient_reverse(source, scale=domain_adversarial_scale)
            ),
            domain_target,
        )
        edc = edge_direction_consistency_loss(
            positive_logits, self.link_head(destination, source)
        )
        npp = F.mse_loss(self.node_property_head(source), property_target)
        time_gaps = torch.as_tensor(
            batch["time_gaps"], dtype=torch.float32, device=self.device
        )
        if time_gaps.shape != positive_logits.shape:
            raise ValueError("time_gaps must contain one value per positive edge")
        if not bool(torch.isfinite(time_gaps).all()):
            raise ValueError("time_gaps must be finite")
        log_gaps = torch.log1p(time_gaps.clamp_min(0.0))
        # A fixed log transform avoids batch-dependent labels and division by
        # an almost-zero median for nodes with no interaction history.
        tip_target = log_gaps
        tip = F.mse_loss(self.time_shift_head(source, destination), tip_target)
        return {
            "mtlp": mtlp,
            "tnp": tnp,
            "edc": edc,
            "npp": npp,
            "tip": tip,
            "domain": domain,
        }

    def weighted_pretrain_loss(
        self,
        losses: dict[str, torch.Tensor],
        *,
        domain_coefficient: float = 1.0,
    ) -> torch.Tensor:
        names = ("mtlp", "tnp", "edc", "npp", "tip")
        expected = (*names, "domain")
        if set(losses) != set(expected):
            raise ValueError(f"losses must contain exactly: {', '.join(expected)}")
        coefficient = float(domain_coefficient)
        if not np.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("domain_coefficient must be finite and non-negative")
        pretext = torch.stack(
            [
                torch.exp(-alpha) * losses[name] + alpha
                for alpha, name in zip(self.task_log_vars, names, strict=True)
            ]
        ).sum()
        return pretext + coefficient * losses["domain"]
