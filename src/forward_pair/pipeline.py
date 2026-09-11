"""Resumable end-to-end Forward-Pair experiment orchestration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tempofound.config import ExperimentConfig
from tempofound.optimization import source_affinities
from tempofound.pretrain import run_pretraining
from tempofound.transfer import run_transfer

from .artifacts import write_selection_artifacts
from .portfolio import advance_search, initialize_search, select_portfolio
from .proxy import score_checkpoint_proxy, score_checkpoint_proxy_windows
from .selector import CandidateScore, select_forward_pair
from .uncertainty import paired_temporal_gate


@dataclass(frozen=True)
class PipelineJob:
    name: str
    kind: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class PipelinePlan:
    target: str
    seed: int
    singleton_jobs: tuple[PipelineJob, ...]
    dynamic_anchor_pair_count: int
    expected_candidate_count: int
    final_stages: tuple[str, ...]


@dataclass(frozen=True)
class RaspPipelinePlan:
    target: str
    seed: int
    initial_candidates: tuple[tuple[str, ...], ...]
    beam_width: int
    max_sources: int
    candidate_budget: int
    final_stages: tuple[str, ...]


def config_digest(config: ExperimentConfig) -> str:
    encoded = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_forward_pair_plan(
    config: ExperimentConfig, *, target: str, seed: int
) -> PipelinePlan:
    sources = config.sources_for(target)
    if seed not in config.seeds_for(target):
        raise ValueError(f"seed {seed} is not configured for target {target}")
    singleton_jobs = tuple(
        PipelineJob(name=f"singleton_{source}", kind="singleton", sources=(source,))
        for source in sources
    )
    return PipelinePlan(
        target=target,
        seed=seed,
        singleton_jobs=singleton_jobs,
        dynamic_anchor_pair_count=len(sources) - 1,
        expected_candidate_count=2 * len(sources) - 1,
        final_stages=("selected_pretrain", "all_source_pretrain", "transfer"),
    )


def build_rasp_plan(
    config: ExperimentConfig, *, target: str, seed: int
) -> RaspPipelinePlan:
    sources = tuple(sorted(config.sources_for(target)))
    if seed not in config.seeds_for(target):
        raise ValueError(f"seed {seed} is not configured for target {target}")
    if config.selection_method != "rasp":
        raise ValueError("build_rasp_plan requires selection_method=rasp")
    return RaspPipelinePlan(
        target=target,
        seed=seed,
        initial_candidates=tuple((source,) for source in sources),
        beam_width=config.beam_width,
        max_sources=config.max_sources,
        candidate_budget=config.candidate_budget,
        final_stages=("selected_pretrain", "all_source_pretrain", "transfer"),
    )


def score_artifact_matches(
    path: str | Path,
    *,
    target: str,
    seed: int,
    sources: tuple[str, ...],
    digest: str,
) -> bool:
    artifact = Path(path)
    if not artifact.is_file():
        return False
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    scores = (payload.get("proxy_ap"), payload.get("proxy_auc"))
    return (
        payload.get("target") == target
        and payload.get("seed") == seed
        and tuple(payload.get("sources", ())) == sources
        and payload.get("config_digest") == digest
        and all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in scores
        )
    )


def rasp_score_artifact_matches(
    path: str | Path,
    *,
    target: str,
    seed: int,
    sources: tuple[str, ...],
    digest: str,
    proxy_windows: int,
    proxy_metric: str = 'ap',
) -> bool:
    artifact = Path(path)
    if not artifact.is_file():
        return False
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    windows = payload.get('window_proxy_mrr' if proxy_metric == 'mrr' else 'window_proxy_ap')
    if proxy_metric == 'mrr' and payload.get('proxy_metric') != 'mrr':
        return False
    bootstrap = payload.get("bootstrap")
    if not isinstance(windows, list) or len(windows) != proxy_windows:
        return False
    if not isinstance(bootstrap, dict):
        return False
    scalar_values = [
        payload.get("risk_adjusted_utility"),
        *windows,
        *(bootstrap.get(field) for field in ("mean", "standard_error", "lower", "upper")),
    ]
    replicates = bootstrap.get("replicates")
    return (
        payload.get("target") == target
        and payload.get("seed") == seed
        and tuple(payload.get("sources", ())) == sources
        and payload.get("config_digest") == digest
        and isinstance(replicates, list)
        and bool(replicates)
        and all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in (*scalar_values, *replicates)
        )
    )


def result_artifact_matches(
    path: str | Path,
    *,
    target: str,
    seed: int,
    sources: tuple[str, ...],
    digest: str,
    mode: str,
) -> bool:
    artifact = Path(path)
    if not artifact.is_file():
        return False
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metrics = (payload.get("test_ap"), payload.get("test_auc"))
    return (
        payload.get("target") == target
        and payload.get("seed") == seed
        and tuple(payload.get("sources", ())) == sources
        and payload.get("config_digest") == digest
        and payload.get("mode") == mode
        and all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in metrics
        )
    )


def _candidate_name(sources: tuple[str, ...]) -> str:
    return sources[0] if len(sources) == 1 else "pair_" + "_".join(sorted(sources))


def _read_candidate(path: Path) -> CandidateScore:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CandidateScore(
        config=payload["config"],
        sources=tuple(payload["sources"]),
        proxy_ap=float(payload["proxy_ap"]),
        proxy_auc=float(payload["proxy_auc"]),
        seed=int(payload["seed"]),
        target=payload["target"],
        config_digest=payload["config_digest"],
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_manifest_matches(
    path: Path,
    *,
    checkpoint: Path,
    target: str,
    seed: int,
    sources: tuple[str, ...],
    digest: str,
) -> bool:
    if not path.is_file() or not checkpoint.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload == {
        "config_digest": digest,
        "seed": seed,
        "sources": list(sources),
        "target": target,
    }


def run_forward_pair_pipeline(
    config: ExperimentConfig,
    *,
    target: str,
    seed: int,
    processed_dir: str | Path,
    output_root: str | Path,
    device: str,
) -> dict[str, Any]:
    plan = build_forward_pair_plan(config, target=target, seed=seed)
    source_pool = config.sources_for(target)
    digest = config_digest(config)
    run_root = Path(output_root) / target / f"seed{seed}"
    candidate_root = run_root / "candidates"
    scores: list[CandidateScore] = []

    def evaluate_candidate(sources: tuple[str, ...]) -> CandidateScore:
        name = _candidate_name(sources)
        checkpoint = candidate_root / name / "pretrained.pth"
        score_path = candidate_root / name / "score.json"
        if not score_artifact_matches(
            score_path,
            target=target,
            seed=seed,
            sources=sources,
            digest=digest,
        ):
            run_pretraining(
                config,
                sources,
                seed,
                processed_dir,
                checkpoint,
                device,
            )
            metrics = score_checkpoint_proxy(
                checkpoint,
                target=target,
                processed_dir=processed_dir,
                seed=seed,
                batch_size=config.proxy_batch_size,
                calibration_steps=config.calibration_steps,
                probe_batches=config.proxy_batches,
                calibration_fraction=config.calibration_fraction,
                device=device,
            )
            _write_json(
                score_path,
                {
                    "config": name,
                    "sources": list(sources),
                    "proxy_ap": metrics["proxy_ap"],
                    "proxy_auc": metrics["proxy_auc"],
                    "seed": seed,
                    "target": target,
                    "config_digest": digest,
                    "information_boundary": metrics["information_boundary"],
                },
            )
        return _read_candidate(score_path)

    for job in plan.singleton_jobs:
        scores.append(evaluate_candidate(job.sources))
    singleton_scores = {score.sources[0]: score.proxy_ap for score in scores}
    anchor = min(
        source_pool, key=lambda source: (-singleton_scores[source], source)
    )
    for partner in source_pool:
        if partner != anchor:
            scores.append(evaluate_candidate((anchor, partner)))
    selection = select_forward_pair(scores, source_pool, config.margin)
    provenance = json.loads(
        (Path(__file__).resolve().parents[2] / "PROVENANCE.json").read_text(
            encoding="utf-8"
        )
    )
    selection_paths = write_selection_artifacts(
        run_root / "selection",
        scores,
        selection,
        source_commits={
            "TempoFound": provenance["tempofound_commit"],
            "Forward-Pair": provenance["forward_pair_commit"],
        },
    )

    def final_pretrain(label: str, sources: tuple[str, ...]) -> Path:
        checkpoint = run_root / "final" / label / "pretrained.pth"
        manifest = checkpoint.with_suffix(".manifest.json")
        if not _checkpoint_manifest_matches(
            manifest,
            checkpoint=checkpoint,
            target=target,
            seed=seed,
            sources=sources,
            digest=digest,
        ):
            run_pretraining(
                config, sources, seed, processed_dir, checkpoint, device
            )
            _write_json(
                manifest,
                {
                    "config_digest": digest,
                    "seed": seed,
                    "sources": list(sources),
                    "target": target,
                },
            )
        return checkpoint

    selected_checkpoint = final_pretrain("forward_pair", selection.selected_sources)
    all_checkpoint = final_pretrain("all_sources", source_pool)
    metrics: dict[str, Any] = {}
    for label, checkpoint in (
        ("forward_pair", selected_checkpoint),
        ("all_sources", all_checkpoint),
    ):
        result_path = run_root / "transfer" / f"{label}.json"
        result_sources = (
            selection.selected_sources if label == "forward_pair" else source_pool
        )
        if result_artifact_matches(
            result_path,
            target=target,
            seed=seed,
            sources=result_sources,
            digest=digest,
            mode="frozen",
        ):
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = run_transfer(
                config,
                checkpoint_path=checkpoint,
                target=target,
                seed=seed,
                processed_dir=processed_dir,
                mode="frozen",
                evaluation_protocol=config.evaluation_protocol,
                negative_policy=config.negative_policy,
                negatives_per_positive=config.negatives_per_positive,
                evaluation_batch_size=config.evaluation_batch_size,
                device=device,
            )
            result["config_digest"] = digest
            result["sources"] = list(result_sources)
            _write_json(result_path, result)
        metrics[label] = result
    metric_name = "test_mrr" if config.evaluation_protocol == "ranking" else "test_ap"
    delta_name = "mrr_delta" if metric_name == "test_mrr" else "ap_delta"
    summary = {
        "target": target,
        "seed": seed,
        "candidate_count": selection.candidate_count,
        "anchor_source": selection.anchor_source,
        "selected_sources": list(selection.selected_sources),
        "selection_json": str(selection_paths.selection_json),
        f"forward_pair_{metric_name}": metrics["forward_pair"][metric_name],
        f"all_sources_{metric_name}": metrics["all_sources"][metric_name],
        delta_name: metrics["forward_pair"][metric_name]
        - metrics["all_sources"][metric_name],
        "evaluation_protocol": config.evaluation_protocol,
        "config_digest": digest,
    }
    _write_json(run_root / "summary.json", summary)
    return summary


def _ranking_result_artifact_matches(
    path: Path,
    *,
    target: str,
    seed: int,
    sources: tuple[str, ...],
    digest: str,
    protocol: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metric_name = "test_mrr" if protocol == "ranking" else "test_ap"
    metric = payload.get(metric_name)
    return (
        payload.get("target") == target
        and payload.get("seed") == seed
        and tuple(payload.get("sources", ())) == sources
        and payload.get("config_digest") == digest
        and payload.get("mode") == "frozen"
        and payload.get("evaluation_protocol") == protocol
        and isinstance(metric, (int, float))
        and math.isfinite(metric)
    )


def _portfolio_name(sources: tuple[str, ...]) -> str:
    if len(sources) == 1:
        return sources[0]
    return "portfolio_" + "_".join(sources)


def run_rasp_pipeline(
    config: ExperimentConfig,
    *,
    target: str,
    seed: int,
    processed_dir: str | Path,
    output_root: str | Path,
    device: str,
) -> dict[str, Any]:
    """Run uncertainty-aware portfolio search and fixed-budget transfer."""

    plan = build_rasp_plan(config, target=target, seed=seed)
    source_pool = tuple(source for (source,) in plan.initial_candidates)
    digest = config_digest(config)
    run_root = Path(output_root) / target / f"seed{seed}"
    candidate_root = run_root / "candidates"
    state = initialize_search(
        source_pool,
        beam_width=plan.beam_width,
        max_sources=plan.max_sources,
        candidate_budget=plan.candidate_budget,
    )

    while state.pending:
        utilities: dict[tuple[str, ...], float] = {}
        for sources in state.pending:
            name = _portfolio_name(sources)
            checkpoint = candidate_root / name / "pretrained.pth"
            score_path = candidate_root / name / "score.json"
            if not rasp_score_artifact_matches(
                score_path,
                target=target,
                seed=seed,
                sources=sources,
                digest=digest,
                proxy_windows=config.proxy_windows,
                proxy_metric=config.proxy_metric,
            ):
                candidate_weights = None
                if config.consistent_candidate_weights:
                    candidate_weights = (
                        {sources[0]: 1.0} if len(sources) == 1 else
                        source_affinities(sources,
                            {source: state.utilities[(source,)] for source in source_pool},
                            {}, temperature=config.affinity_temperature)
                    )
                run_pretraining(
                    config,
                    sources,
                    seed,
                    processed_dir,
                    checkpoint,
                    device,
                    source_weights=candidate_weights,
                )
                metrics = score_checkpoint_proxy_windows(
                    checkpoint,
                    target=target,
                    processed_dir=processed_dir,
                    seed=seed,
                    batch_size=config.proxy_batch_size,
                    calibration_steps=config.calibration_steps,
                    probe_batches=config.proxy_batches,
                    calibration_fraction=config.calibration_fraction,
                    proxy_windows=config.proxy_windows,
                    bootstrap_replicates=config.bootstrap_replicates,
                    bootstrap_confidence=config.bootstrap_confidence,
                    risk_beta=config.risk_beta,
                    size_penalty=config.size_penalty,
                    cardinality=len(sources),
                    proxy_metric=config.proxy_metric,
                    negatives_per_positive=config.negatives_per_positive,
                    proxy_negative_policy=config.proxy_negative_policy,
                    regret_weight=config.transfer_regret_weight,
                    regret_margin=config.transfer_regret_margin,
                    regret_mode=config.transfer_regret_mode,
                    device=device,
                )
                _write_json(
                    score_path,
                    {
                        "config": name,
                        "sources": list(sources),
                        "seed": seed,
                        "target": target,
                        "config_digest": digest,
                        "source_weights": candidate_weights,
                        **metrics,
                    },
                )
            payload = json.loads(score_path.read_text(encoding="utf-8"))
            utilities[sources] = float(payload["risk_adjusted_utility"])
        state = advance_search(
            state,
            utilities,
            min_gain=config.min_risk_adjusted_gain,
        )
        state_path = (
            run_root
            / "selection"
            / f"search_state_after_{len(state.explored):03d}_candidates.json"
        )
        _write_json(state_path, state.to_dict())

    selected_sources = select_portfolio(state)
    gate_diagnostics = {}
    if config.temporal_paired_gate:
        reference = json.loads((candidate_root / _portfolio_name(source_pool) / "score.json").read_text())
        eligible = []
        for sources in state.explored:
            if sources == source_pool:
                continue
            candidate = json.loads((candidate_root / _portfolio_name(sources) / "score.json").read_text())
            diagnostic = paired_temporal_gate(
                candidate['window_proxy_scores'], reference['window_proxy_scores'],
                seed=seed + 17011, replicates=config.bootstrap_replicates,
                confidence=config.bootstrap_confidence,
                comparisons=len(state.explored) - 1,
                min_gain=config.min_risk_adjusted_gain,
            )
            gate_diagnostics["+".join(sources)] = diagnostic
            if diagnostic["accepted"]:
                eligible.append(sources)
        selected_sources = (min(eligible, key=lambda sources:
            (-gate_diagnostics["+".join(sources)]["lower_delta"], sources))
            if eligible else source_pool)
    singleton_utilities = {
        source: state.utilities[(source,)] for source in source_pool
    }
    selected_weights = source_affinities(
        selected_sources,
        singleton_utilities,
        {} if config.consistent_candidate_weights else state.utilities,
        temperature=config.affinity_temperature,
    )
    all_source_weights = source_affinities(
        source_pool,
        singleton_utilities,
        {} if config.consistent_candidate_weights else state.utilities,
        temperature=config.affinity_temperature,
    )
    selection_path = run_root / "selection" / "selection.json"
    _write_json(
        selection_path,
        {
            "method": "rasp",
            "temporal_paired_gate": gate_diagnostics,
            "consistent_candidate_weights": config.consistent_candidate_weights,
            "target": target,
            "seed": seed,
            "config_digest": digest,
            "candidate_sources": list(source_pool),
            "selected_sources": list(selected_sources),
            "candidate_count": len(state.explored),
            "risk_adjusted_utility": state.utilities[selected_sources],
            "candidate_utilities": {
                "+".join(sources): utility
                for sources, utility in sorted(state.utilities.items())
            },
            "stop_reason": state.stop_reason,
            "information_boundary": "target_training_period_only",
        },
    )
    _write_json(
        run_root / "selection" / "domain_weights.json",
        {
            "method": "risk_adjusted_target_affinity",
            "temperature": config.affinity_temperature,
            "selected_weights": selected_weights,
            "all_source_weights": all_source_weights,
        },
    )

    def final_pretrain(
        label: str, sources: tuple[str, ...], weights: dict[str, float]
    ) -> Path:
        checkpoint = run_root / "final" / label / "pretrained.pth"
        manifest = checkpoint.with_suffix(".manifest.json")
        if not _checkpoint_manifest_matches(
            manifest,
            checkpoint=checkpoint,
            target=target,
            seed=seed,
            sources=sources,
            digest=digest,
        ):
            run_pretraining(
                config,
                sources,
                seed,
                processed_dir,
                checkpoint,
                device,
                source_weights=weights,
            )
            _write_json(
                manifest,
                {
                    "config_digest": digest,
                    "seed": seed,
                    "sources": list(sources),
                    "target": target,
                    "source_weights": weights,
                },
            )
        return checkpoint

    selected_checkpoint = final_pretrain(
        "rasp", selected_sources, selected_weights
    )
    all_checkpoint = final_pretrain(
        "all_sources", source_pool, all_source_weights
    )
    results: dict[str, dict[str, Any]] = {}
    for label, checkpoint, sources in (
        ("rasp", selected_checkpoint, selected_sources),
        ("all_sources", all_checkpoint, source_pool),
    ):
        result_path = run_root / "transfer" / f"{label}.json"
        if _ranking_result_artifact_matches(
            result_path,
            target=target,
            seed=seed,
            sources=sources,
            digest=digest,
            protocol=config.evaluation_protocol,
        ):
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = run_transfer(
                config,
                checkpoint_path=checkpoint,
                target=target,
                seed=seed,
                processed_dir=processed_dir,
                mode="frozen",
                device=device,
                evaluation_protocol=config.evaluation_protocol,
                negative_policy=config.negative_policy,
                negatives_per_positive=config.negatives_per_positive,
                evaluation_batch_size=config.evaluation_batch_size,
            )
            result["config_digest"] = digest
            result["sources"] = list(sources)
            _write_json(result_path, result)
        results[label] = result

    metric_name = (
        "test_mrr" if config.evaluation_protocol == "ranking" else "test_ap"
    )
    summary = {
        "method": "rasp",
        "target": target,
        "seed": seed,
        "candidate_count": len(state.explored),
        "selected_sources": list(selected_sources),
        "selection_json": str(selection_path),
        "evaluation_protocol": config.evaluation_protocol,
        f"rasp_{metric_name}": results["rasp"][metric_name],
        f"all_sources_{metric_name}": results["all_sources"][metric_name],
        f"{metric_name}_delta": (
            results["rasp"][metric_name] - results["all_sources"][metric_name]
        ),
        "config_digest": digest,
    }
    _write_json(run_root / "summary.json", summary)
    return summary


def run_pipeline(
    config: ExperimentConfig,
    *,
    target: str,
    seed: int,
    processed_dir: str | Path,
    output_root: str | Path,
    device: str,
) -> dict[str, Any]:
    if config.selection_method == "rasp":
        return run_rasp_pipeline(
            config,
            target=target,
            seed=seed,
            processed_dir=processed_dir,
            output_root=output_root,
            device=device,
        )
    return run_forward_pair_pipeline(
        config,
        target=target,
        seed=seed,
        processed_dir=processed_dir,
        output_root=output_root,
        device=device,
    )
