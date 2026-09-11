"""History-conditioned expert modules for temporal graph transfer.

The legacy dense temporal mixture and per-source experts are preserved for
reproducibility.  The new :class:`RiskControlledSourceRouter` is only enabled
by an explicit configuration flag so old checkpoints remain loadable.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn


class TemporalResidualExperts(nn.Module):
    """Dense expert mixture routed using representation and past-only statistics.

    Statistics are log count, min/mean/max log age, and observed-neighbor
    occupancy. No domain ID or absolute timestamp is supplied to the router.
    This component is retained as an ablation/control rather than the main
    transfer mechanism.
    """

    def __init__(self, dim: int, experts: int, dropout: float):
        super().__init__()
        if experts < 2:
            raise ValueError("temporal experts requires at least two experts")
        self.norm = nn.LayerNorm(dim)
        self.history = nn.Sequential(nn.Linear(5, dim), nn.GELU(), nn.Linear(dim, dim))
        self.router = nn.Sequential(
            nn.Linear(dim + 5, dim), nn.GELU(), nn.Linear(dim, experts)
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, 2 * dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(2 * dim, dim),
                )
                for _ in range(experts)
            ]
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, embeddings, history_stats, *, return_weights=False):
        normalized = self.norm(embeddings)
        stats = torch.tanh(history_stats / 5.0)
        weights = self.router(torch.cat((normalized, stats), dim=-1)).softmax(dim=-1)
        inputs = normalized + self.history(stats)
        outputs = torch.stack([expert(inputs) for expert in self.experts], dim=1)
        residual = (weights.unsqueeze(-1) * outputs).sum(dim=1)
        observed = (history_stats[:, 4:5] > 0).to(embeddings.dtype)
        result = embeddings + observed * self.residual_scale.tanh() * residual
        return (result, weights) if return_weights else result


class SourceExpertBank(nn.Module):
    """Per-source residual experts specialized during source pre-training.

    ``forward_source`` is used while processing events from a known source.
    ``stack_target_deltas`` exposes every source expert to an unseen target;
    the target itself never receives a newly initialized source expert in the
    risk-controlled path.
    """

    def __init__(self, names: Sequence[str], dim: int, dropout: float):
        super().__init__()
        names = tuple(str(name) for name in names)
        if not names or len(names) != len(set(names)):
            raise ValueError("source expert names must be non-empty and unique")
        self.names = names
        self.experts = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, 2 * dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(2 * dim, dim),
                )
                for name in names
            }
        )
        hidden = max(dim // 2, 1)
        self.gates = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, 1)
                )
                for name in names
            }
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.05))

    def _bounded_stats(self, history_stats: torch.Tensor) -> torch.Tensor:
        return torch.tanh(history_stats / 5.0)

    def source_delta(
        self, name: str, embeddings: torch.Tensor, history_stats: torch.Tensor
    ) -> torch.Tensor:
        if name not in self.experts:
            raise ValueError(f"unknown source expert: {name}")
        stats = self._bounded_stats(history_stats)
        gate = torch.sigmoid(self.gates[name](stats))
        return gate * self.experts[name](embeddings)

    def forward_source(
        self, name: str, embeddings: torch.Tensor, history_stats: torch.Tensor
    ) -> torch.Tensor:
        delta = self.source_delta(name, embeddings, history_stats)
        observed = (history_stats[:, 4:5] > 0).to(embeddings.dtype)
        return embeddings + observed * self.residual_scale.tanh() * delta

    # Backward-compatible alias used by legacy checkpoints/code paths.
    def forward(self, name, embeddings, history_stats):
        return self.forward_source(name, embeddings, history_stats)

    def stack_target_deltas(
        self, embeddings: torch.Tensor, history_stats: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack(
            [self.source_delta(name, embeddings, history_stats) for name in self.names],
            dim=1,
        )


class RiskControlledSourceRouter(nn.Module):
    """Event-level source assembly with an explicit identity abstention path.

    The router uses only the current representation and strictly past-only
    neighborhood statistics.  Global source priors come from the same
    target-training-only source utility used by RASP.  High routing entropy
    reduces the transfer gate, so uncertain events fall back toward identity.
    """

    def __init__(
        self,
        names: Sequence[str],
        dim: int,
        dropout: float,
        *,
        temperature: float = 1.0,
        prior_strength: float = 1.0,
        entropy_penalty: float = 1.0,
        abstain_bias: float = -1.0,
        local_acceptance: bool = True,
    ) -> None:
        super().__init__()
        names = tuple(str(name) for name in names)
        if not names or len(names) != len(set(names)):
            raise ValueError("router source names must be non-empty and unique")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("router temperature must be finite and positive")
        if not math.isfinite(prior_strength) or prior_strength < 0.0:
            raise ValueError("router prior strength must be finite and non-negative")
        if not math.isfinite(entropy_penalty) or entropy_penalty < 0.0:
            raise ValueError("router entropy penalty must be finite and non-negative")
        if not math.isfinite(abstain_bias):
            raise ValueError("router abstain bias must be finite")
        self.names = names
        self.temperature = float(temperature)
        self.prior_strength = float(prior_strength)
        self.entropy_penalty = float(entropy_penalty)
        self.local_acceptance = bool(local_acceptance)
        self.norm = nn.LayerNorm(dim)
        self.delta_norm = nn.LayerNorm(dim)
        hidden = max(dim, 16)
        # One shared source scorer is applied to every expert response.  This
        # makes event routing source-permutation equivariant: source identity
        # enters through its learned expert response and its global RASP prior,
        # rather than a source-specific output neuron.
        self.router = nn.Sequential(
            nn.Linear(2 * dim + 6, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # The safety critic sees the routed residual itself in addition to the
        # causal history state, so it can reject locally implausible transfer.
        self.safety = nn.Sequential(
            nn.Linear(2 * dim + 7, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Start from a deterministic global-prior router rather than random
        # event-level preferences; target calibration then learns deviations.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        nn.init.zeros_(self.safety[-1].weight)
        nn.init.zeros_(self.safety[-1].bias)
        self.abstain_bias = nn.Parameter(torch.tensor(float(abstain_bias)))
        self.residual_scale = nn.Parameter(torch.tensor(0.05))
        self.register_buffer("prior_logits", torch.zeros(len(names)))
        # Global target-level safety multiplier. It is calibrated from target-train
        # paired routed-vs-identity evidence and never uses validation/test labels.
        self.register_buffer("target_gate", torch.ones(1))

    def set_prior(self, weights: Mapping[str, float]) -> None:
        if set(weights) != set(self.names):
            raise ValueError("routing prior must contain exactly the source experts")
        values = torch.tensor(
            [float(weights[name]) for name in self.names],
            dtype=self.prior_logits.dtype,
            device=self.prior_logits.device,
        )
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("routing prior weights must be finite and non-negative")
        total = values.sum()
        if float(total) <= 0.0:
            raise ValueError("at least one routing prior weight must be positive")
        probabilities = values / total
        logits = probabilities.clamp_min(1e-8).log()
        logits = logits - logits.mean()
        with torch.no_grad():
            self.prior_logits.copy_(logits)


    def set_target_gate(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("target gate must be finite and in [0, 1]")
        with torch.no_grad():
            self.target_gate.fill_(value)

    def forward(
        self,
        embeddings: torch.Tensor,
        history_stats: torch.Tensor,
        source_deltas: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ):
        if source_deltas.ndim != 3:
            raise ValueError("source_deltas must have shape [batch, sources, dim]")
        if source_deltas.shape[0] != embeddings.shape[0]:
            raise ValueError("source_deltas batch dimension mismatch")
        if source_deltas.shape[1] != len(self.names):
            raise ValueError("source_deltas source dimension mismatch")
        if source_deltas.shape[2] != embeddings.shape[1]:
            raise ValueError("source_deltas embedding dimension mismatch")

        stats = torch.tanh(history_stats / 5.0)
        normalized = self.norm(embeddings)
        normalized_deltas = self.delta_norm(source_deltas)
        batch, sources, _ = source_deltas.shape
        expanded_embedding = normalized.unsqueeze(1).expand(-1, sources, -1)
        expanded_stats = stats.unsqueeze(1).expand(-1, sources, -1)
        expanded_prior = self.prior_logits.view(1, sources, 1).expand(batch, -1, -1)
        source_features = torch.cat(
            (expanded_embedding, normalized_deltas, expanded_stats, expanded_prior), dim=-1
        )
        dynamic_logits = self.router(source_features).squeeze(-1)
        logits = dynamic_logits + self.prior_strength * self.prior_logits.unsqueeze(0)
        weights = torch.softmax(logits / self.temperature, dim=-1)

        if len(self.names) == 1:
            normalized_entropy = torch.zeros(
                (len(embeddings), 1), dtype=embeddings.dtype, device=embeddings.device
            )
        else:
            entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(
                dim=-1, keepdim=True
            )
            normalized_entropy = entropy / math.log(len(self.names))

        residual = (weights.unsqueeze(-1) * source_deltas).sum(dim=1)
        residual_view = self.delta_norm(residual)
        residual_size = torch.log1p(residual.norm(dim=-1, keepdim=True))
        max_weight = weights.max(dim=-1, keepdim=True).values
        safety_input = torch.cat(
            (normalized, residual_view, stats, residual_size, max_weight), dim=-1
        )
        safety_logits = (
            self.safety(safety_input)
            + self.abstain_bias
            - self.entropy_penalty * normalized_entropy
        )
        local_accept = (
            torch.sigmoid(safety_logits)
            if self.local_acceptance
            else torch.ones_like(safety_logits)
        )
        accept = self.target_gate.to(local_accept.dtype) * local_accept
        observed = (history_stats[:, 4:5] > 0).to(embeddings.dtype)
        result = embeddings + observed * accept * self.residual_scale.tanh() * residual

        if not return_diagnostics:
            return result
        diagnostics = {
            "source_weights": weights,
            "normalized_entropy": normalized_entropy,
            "accept_probability": accept,
            "local_accept_probability": local_accept,
            "target_gate": self.target_gate.expand(len(embeddings), 1),
            "observed": observed,
        }
        return result, diagnostics
