"""Validated experiment configuration shared by K=4 and K=6 protocols."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


def _tuple_map(values: Mapping[str, Any] | None) -> dict[str, tuple[Any, ...]]:
    return {str(key): tuple(value) for key, value in (values or {}).items()}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    datasets: tuple[str, ...]
    targets: tuple[str, ...]
    seeds: tuple[int, ...] = (0, 1, 2, 3, 42)
    source_overrides: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    seed_overrides: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    shared_dim: int = 128
    time_dim: int = 64
    num_layers: int = 2
    num_heads: int = 2
    num_neighbors: int = 10
    dropout: float = 0.1
    pretrain_epochs: int = 25
    transfer_epochs: int = 20
    checkpoint_selection_mode: str = "validation"
    batch_size: int = 512
    total_batches_per_epoch: int = 160
    pretraining_strategy: str = "legacy_sequential"
    affinity_temperature: float = 0.1
    minimum_source_batches: int = 1
    domain_adversarial_scale: float = 1.0
    domain_loss_coefficient: float = 0.1
    calibration_fraction: float = 0.8
    calibration_steps: int = 100
    proxy_batch_size: int = 256
    proxy_batches: int = 12
    margin: float = 0.002
    time_encoding_mode: str = "paper_sincos"
    backbone: str = "tgat"
    temporal_experts: int = 0
    expert_identity_init: bool = False
    source_experts: bool = False
    risk_controlled_routing: bool = False
    router_temperature: float = 1.0
    router_prior_strength: float = 1.0
    router_entropy_penalty: float = 1.0
    router_abstain_bias: float = -1.0
    router_local_acceptance: bool = True
    target_router_training_enabled: bool = True
    transfer_regret_weight: float = 0.25
    transfer_regret_margin: float = 0.0
    transfer_regret_mode: str = "batch_bce"
    target_gate_mode: str = "none"
    target_gate_windows: int = 5
    target_gate_bootstrap_replicates: int = 1000
    target_gate_confidence: float = 0.95
    target_gate_min_gain: float = 0.0
    target_gate_temperature: float = 0.01
    target_calibration_fraction: float = 0.2
    target_calibration_min_events: int = 128
    target_reference_weight: float = 1.0
    target_gate_negatives: int = 20
    proxy_metric: str = "ap"
    proxy_negative_policy: str = "random"
    selection_method: str = "forward_pair"
    proxy_windows: int = 3
    bootstrap_replicates: int = 1000
    bootstrap_confidence: float = 0.95
    risk_beta: float = 1.0
    size_penalty: float = 0.0
    beam_width: int = 2
    max_sources: int = 3
    candidate_budget: int = 16
    min_risk_adjusted_gain: float = 0.002
    consistent_candidate_weights: bool = False
    temporal_paired_gate: bool = False
    evaluation_protocol: str = "ranking"
    negative_policy: str = "historical"
    negatives_per_positive: int = 100
    evaluation_batch_size: int = 256

    def __post_init__(self) -> None:
        if self.checkpoint_selection_mode not in {"validation", "fixed_epoch", "prefix_only"}:
            raise ValueError(
                "checkpoint_selection_mode must be validation, fixed_epoch, or prefix_only"
            )
        if self.proxy_metric not in ('ap', 'mrr'):
            raise ValueError('proxy_metric must be ap or mrr')
        if self.proxy_negative_policy not in {'random', 'historical'}:
            raise ValueError('proxy_negative_policy must be random or historical')
        if self.risk_controlled_routing and not self.source_experts:
            raise ValueError('risk_controlled_routing requires source_experts=True')
        if not math.isfinite(self.router_temperature) or self.router_temperature <= 0.0:
            raise ValueError('router_temperature must be finite and positive')
        if not math.isfinite(self.router_prior_strength) or self.router_prior_strength < 0.0:
            raise ValueError('router_prior_strength must be finite and non-negative')
        if not math.isfinite(self.router_entropy_penalty) or self.router_entropy_penalty < 0.0:
            raise ValueError('router_entropy_penalty must be finite and non-negative')
        if not math.isfinite(self.router_abstain_bias):
            raise ValueError('router_abstain_bias must be finite')
        if not math.isfinite(self.transfer_regret_weight) or self.transfer_regret_weight < 0.0:
            raise ValueError('transfer_regret_weight must be finite and non-negative')
        if not math.isfinite(self.transfer_regret_margin) or self.transfer_regret_margin < 0.0:
            raise ValueError('transfer_regret_margin must be finite and non-negative')
        if self.transfer_regret_mode not in {'batch_bce', 'event_pairwise'}:
            raise ValueError('transfer_regret_mode must be batch_bce or event_pairwise')
        if self.target_gate_mode not in {'none', 'lcb_soft', 'lcb_hard', 'holdout_lcb_soft', 'holdout_lcb_hard', 'force_identity', 'force_route'}:
            raise ValueError('unsupported target_gate_mode')
        if self.target_gate_windows < 2:
            raise ValueError('target_gate_windows must be at least two')
        if self.target_gate_bootstrap_replicates < 1:
            raise ValueError('target_gate_bootstrap_replicates must be positive')
        if not 0.0 < self.target_gate_confidence < 1.0:
            raise ValueError('target_gate_confidence must be between zero and one')
        if not math.isfinite(self.target_gate_min_gain):
            raise ValueError('target_gate_min_gain must be finite')
        if not math.isfinite(self.target_gate_temperature) or self.target_gate_temperature <= 0.0:
            raise ValueError('target_gate_temperature must be finite and positive')
        if not 0.0 < self.target_calibration_fraction < 0.5:
            raise ValueError('target_calibration_fraction must be between zero and 0.5')
        if self.target_calibration_min_events < 1:
            raise ValueError('target_calibration_min_events must be positive')
        if not math.isfinite(self.target_reference_weight) or self.target_reference_weight < 0.0:
            raise ValueError('target_reference_weight must be finite and non-negative')
        if self.target_gate_negatives < 1:
            raise ValueError('target_gate_negatives must be positive')
        if self.temporal_experts != 0 and self.temporal_experts < 2:
            raise ValueError('temporal_experts must be zero or at least two')
        if len(self.datasets) != len(set(self.datasets)):
            raise ValueError("datasets must be unique")
        if len(self.targets) != len(set(self.targets)):
            raise ValueError("targets must be unique")
        unknown_targets = set(self.targets).difference(self.datasets)
        if unknown_targets:
            raise ValueError(f"targets missing from datasets: {sorted(unknown_targets)}")
        if self.time_encoding_mode not in {"paper_sincos", "legacy_cos"}:
            raise ValueError("time_encoding_mode must be paper_sincos or legacy_cos")
        if self.backbone not in {"tgat", "graphmixer"}:
            raise ValueError("backbone must be tgat or graphmixer")
        if self.selection_method not in {"forward_pair", "rasp"}:
            raise ValueError("selection_method must be forward_pair or rasp")
        if self.time_encoding_mode == "paper_sincos" and self.time_dim % 2:
            raise ValueError("paper_sincos requires an even time_dim")
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must be between zero and one")
        if self.margin < 0.0:
            raise ValueError("margin must be non-negative")
        if self.total_batches_per_epoch <= 0:
            raise ValueError("total_batches_per_epoch must be positive")
        if self.pretraining_strategy not in {"legacy_sequential", "conflict_aware"}:
            raise ValueError(
                "pretraining_strategy must be legacy_sequential or conflict_aware"
            )
        if not math.isfinite(self.affinity_temperature) or self.affinity_temperature <= 0.0:
            raise ValueError("affinity_temperature must be finite and positive")
        if self.minimum_source_batches < 0:
            raise ValueError("minimum_source_batches must be non-negative")
        if (
            not math.isfinite(self.domain_adversarial_scale)
            or self.domain_adversarial_scale < 0.0
        ):
            raise ValueError("domain_adversarial_scale must be finite and non-negative")
        if (
            not math.isfinite(self.domain_loss_coefficient)
            or self.domain_loss_coefficient < 0.0
        ):
            raise ValueError("domain_loss_coefficient must be finite and non-negative")
        if self.proxy_windows < 1:
            raise ValueError("proxy_windows must be positive")
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise ValueError("bootstrap_confidence must be between zero and one")
        if self.risk_beta < 0.0 or self.size_penalty < 0.0:
            raise ValueError("risk_beta and size_penalty must be non-negative")
        if self.beam_width < 1:
            raise ValueError("beam_width must be positive")
        if self.max_sources < 1:
            raise ValueError("max_sources must be positive")
        if self.candidate_budget < 1:
            raise ValueError("candidate_budget must be positive")
        if self.min_risk_adjusted_gain < 0.0:
            raise ValueError("min_risk_adjusted_gain must be non-negative")
        if self.evaluation_protocol not in {"legacy_binary", "ranking"}:
            raise ValueError("evaluation_protocol must be legacy_binary or ranking")
        if self.negative_policy not in {"random", "historical", "inductive"}:
            raise ValueError("negative_policy must be random, historical, or inductive")
        if self.negatives_per_positive <= 0 or self.evaluation_batch_size <= 0:
            raise ValueError("evaluation negative count and batch size must be positive")
        for target in self.targets:
            sources = self.sources_for(target)
            if target in sources:
                raise ValueError("target cannot appear in source pool")
            if len(sources) < 2 or len(sources) != len(set(sources)):
                raise ValueError("each target requires at least two unique sources")
            unknown_sources = set(sources).difference(self.datasets)
            if unknown_sources:
                raise ValueError(
                    f"source pool contains unknown datasets: {sorted(unknown_sources)}"
                )
            if self.selection_method == "rasp":
                if self.temporal_paired_gate and (
                    len(sources) != 2 or self.max_sources != len(sources)
                    or self.candidate_budget < 2 ** len(sources) - 1
                    or self.beam_width < len(sources)
                ):
                    raise ValueError("pilot temporal paired gate requires exactly two sources and exhaustive portfolios")
                if self.proxy_windows < 2:
                    raise ValueError("rasp proxy_windows must be at least two")
                if self.max_sources > len(sources):
                    raise ValueError(
                        "rasp max_sources cannot exceed the target source count"
                    )
                if self.candidate_budget < len(sources):
                    raise ValueError(
                        "rasp candidate_budget must cover all singleton sources"
                    )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        model = dict(raw.get("model", {}))
        training = dict(raw.get("training", {}))
        proxy = dict(raw.get("proxy", {}))
        selection = dict(raw.get("selection", {}))
        evaluation = dict(raw.get("evaluation", {}))
        merged = {
            "name": raw["name"],
            "datasets": tuple(raw["datasets"]),
            "targets": tuple(raw["targets"]),
            "seeds": tuple(int(seed) for seed in raw.get("seeds", (0, 1, 2, 3, 42))),
            "source_overrides": _tuple_map(raw.get("source_overrides")),
            "seed_overrides": {
                key: tuple(int(seed) for seed in value)
                for key, value in _tuple_map(raw.get("seed_overrides")).items()
            },
            **model,
            **training,
            **proxy,
            **selection,
            **evaluation,
        }
        if "time_encoding_mode" in raw:
            merged["time_encoding_mode"] = raw["time_encoding_mode"]
        return cls(**merged)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError("configuration root must be a mapping")
        return cls.from_mapping(raw)

    def sources_for(self, target: str) -> tuple[str, ...]:
        if target not in self.targets:
            raise ValueError(f"target is not configured: {target}")
        override = self.source_overrides.get(target)
        if override is not None:
            return tuple(override)
        return tuple(dataset for dataset in self.datasets if dataset != target)

    def seeds_for(self, target: str) -> tuple[int, ...]:
        if target not in self.targets:
            raise ValueError(f"target is not configured: {target}")
        return tuple(self.seed_overrides.get(target, self.seeds))
