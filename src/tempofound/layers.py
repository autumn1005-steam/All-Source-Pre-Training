"""Neural layers shared by TempoFound encoders."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale  # type: ignore[attr-defined]
        return value.view_as(value)

    @staticmethod
    def backward(
        ctx: object, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None  # type: ignore[attr-defined]


def gradient_reverse(value: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Keep the forward value and reverse its representation gradient."""

    scale_value = float(scale)
    if not math.isfinite(scale_value) or scale_value < 0.0:
        raise ValueError("gradient reversal scale must be finite and non-negative")
    return _GradientReversal.apply(value, scale_value)


class HarmonicTimeEncoder(nn.Module):
    """Harmonic time features with paper and legacy-compatible modes."""

    SUPPORTED_MODES = frozenset({"paper_sincos", "legacy_cos"})

    def __init__(self, time_dim: int, mode: str = "paper_sincos") -> None:
        super().__init__()
        if time_dim <= 0:
            raise ValueError("time_dim must be positive")
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"unsupported time encoding mode: {mode}")
        if mode == "paper_sincos" and time_dim % 2:
            raise ValueError("paper_sincos requires an even time_dim")
        self.time_dim = int(time_dim)
        self.mode = mode
        initial_frequencies = 1.0 / 10 ** np.linspace(0, 9, self.time_dim)
        self.basis_freq = nn.Parameter(
            torch.from_numpy(initial_frequencies).to(dtype=torch.float32)
        )
        self.phase = nn.Parameter(torch.zeros(self.time_dim, dtype=torch.float32))

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        values = timestamps.unsqueeze(-1)
        if self.mode == "legacy_cos":
            return torch.cos(values * self.basis_freq + self.phase)
        half = self.time_dim // 2
        angles = values * self.basis_freq[:half] + self.phase[:half]
        harmonic = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        return harmonic / math.sqrt(self.time_dim)


class MergeLayer(nn.Module):
    """Two-layer merge used by the original TempoFound TGAT backbone."""

    def __init__(self, dim1: int, dim2: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim1 + dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(torch.cat((first, second), dim=1))))


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float, dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention = torch.bmm(query, key.transpose(1, 2)) / self.temperature
        if mask is not None:
            attention = attention.masked_fill(mask, -1e10)
        attention = self.dropout(torch.softmax(attention, dim=2))
        return torch.bmm(attention, value), attention


class MultiHeadAttention(nn.Module):
    """Parameter-compatible attention used by released TempoFound checkpoints."""

    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(
            temperature=float(np.power(d_k, 0.5)), dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, query_length, _ = q.size()
        _, key_length, _ = k.size()
        _, value_length, _ = v.size()
        residual = q
        q = self.w_qs(q).view(batch, query_length, self.n_head, self.d_k)
        k = self.w_ks(k).view(batch, key_length, self.n_head, self.d_k)
        v = self.w_vs(v).view(batch, value_length, self.n_head, self.d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, query_length, self.d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, key_length, self.d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, value_length, self.d_v)
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1)
        output, attention = self.attention(q, k, v, mask=mask)
        output = output.view(self.n_head, batch, query_length, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            batch, query_length, -1
        )
        output = self.layer_norm(self.dropout(self.fc(output)) + residual)
        return output, attention


class AttnModel(nn.Module):
    """TGAT attention block with released-checkpoint parameter names."""

    def __init__(
        self,
        feature_dim: int,
        edge_dim: int,
        time_dim: int,
        n_head: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.time_dim = time_dim
        self.edge_in_dim = feature_dim + edge_dim + time_dim
        self.model_dim = self.edge_in_dim
        if self.model_dim % n_head:
            raise ValueError("attention model dimension must be divisible by n_head")
        self.merger = MergeLayer(self.model_dim, feature_dim, feature_dim, feature_dim)
        self.multi_head_target = MultiHeadAttention(
            n_head,
            d_model=self.model_dim,
            d_k=self.model_dim // n_head,
            d_v=self.model_dim // n_head,
            dropout=dropout,
        )

    def forward(
        self,
        source: torch.Tensor,
        source_time: torch.Tensor,
        sequence: torch.Tensor,
        sequence_time: torch.Tensor,
        sequence_edge: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_extended = source.unsqueeze(1)
        query = torch.cat((source_extended, torch.zeros_like(source_extended), source_time), dim=2)
        key = torch.cat((sequence, sequence_edge, sequence_time), dim=2)
        attention_mask = mask.unsqueeze(2).permute(0, 2, 1)
        output, attention = self.multi_head_target(
            q=query, k=key, v=key, mask=attention_mask
        )
        return self.merger(output.squeeze(1), source), attention.squeeze(1)
