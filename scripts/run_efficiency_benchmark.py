#!/usr/bin/env python3
"""Profile matched methods on one common host without imputing missing costs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from forward_pair import pipeline
from tempofound.config import ExperimentConfig
from tempofound.pretrain import run_pretraining
from tempofound.transfer import run_transfer

from scripts.efficiency_timing import classify_pipeline_stage


METHODS = ("target_only_tgat", "all_source", "forward_pair", "strict_rasp", "hrcsr")


def _sync(device: str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


class StageRecorder:
    def __init__(self, device: str) -> None:
        self.device = device
        self.records: list[dict[str, Any]] = []

    def measure(self, stage: str, label: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        resolved = torch.device(self.device)
        if resolved.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved)
        _sync(self.device)
        started = time.perf_counter()
        result = function(*args, **kwargs)
        _sync(self.device)
        elapsed = time.perf_counter() - started
        allocated = reserved = 0.0
        if resolved.type == "cuda":
            allocated = torch.cuda.max_memory_allocated(resolved) / 2**20
            reserved = torch.cuda.max_memory_reserved(resolved) / 2**20
        self.records.append(
            {
                "stage": stage,
                "label": label,
                "elapsed_seconds": elapsed,
                "peak_gpu_allocated_mb": allocated,
                "peak_gpu_reserved_mb": reserved,
                "result": result if isinstance(result, dict) else None,
            }
        )
        return result


def _hardware(device: str) -> dict[str, Any]:
    ram_bytes = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                ram_bytes = int(line.split()[1]) * 1024
                break
    driver = None
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    resolved = torch.device(device)
    gpu = None
    gpu_memory_mb = None
    if resolved.type == "cuda":
        properties = torch.cuda.get_device_properties(resolved)
        gpu = properties.name
        gpu_memory_mb = properties.total_memory / 2**20
    return {
        "gpu": gpu,
        "gpu_memory_mb": gpu_memory_mb,
        "cpu": platform.processor() or platform.machine(),
        "cpu_logical_count": os.cpu_count(),
        "ram_bytes": ram_bytes,
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
        "python": platform.python_version(),
        "driver": driver,
        "platform": platform.platform(),
    }


def _main_hrcsr_config(base: ExperimentConfig, target: str, seed: int) -> ExperimentConfig:
    return replace(
        base,
        name=f"efficiency_hrcsr_{target}",
        targets=(target,),
        seeds=(seed,),
        source_experts=True,
        expert_identity_init=True,
        risk_controlled_routing=True,
        consistent_candidate_weights=True,
        temporal_paired_gate=False,
        proxy_metric="mrr",
        proxy_negative_policy="historical",
        negative_policy="historical",
        evaluation_protocol="ranking",
        transfer_regret_weight=0.25,
        transfer_regret_mode="event_pairwise",
        target_gate_mode="holdout_lcb_hard",
        target_gate_windows=5,
        target_gate_bootstrap_replicates=2000,
        target_gate_confidence=0.95,
        target_gate_min_gain=0.0,
        target_gate_temperature=0.01,
        target_calibration_fraction=0.20,
        target_calibration_min_events=128,
        target_reference_weight=1.0,
        target_gate_negatives=20,
        total_batches_per_epoch=32,
    )


def _variant(base: ExperimentConfig, method: str, target: str, seed: int) -> ExperimentConfig:
    common = replace(
        base,
        name=f"efficiency_{method}_{target}",
        targets=(target,),
        seeds=(seed,),
        evaluation_protocol="ranking",
        negative_policy="historical",
        total_batches_per_epoch=32,
    )
    if method == "hrcsr":
        return _main_hrcsr_config(common, target, seed)
    if method == "strict_rasp":
        return replace(
            common,
            selection_method="rasp",
            source_experts=False,
            risk_controlled_routing=False,
            expert_identity_init=False,
            proxy_metric="mrr",
            proxy_negative_policy="historical",
            target_gate_mode="force_identity",
            transfer_regret_weight=0.0,
        )
    if method == "forward_pair":
        return replace(
            common,
            selection_method="forward_pair",
            source_experts=False,
            risk_controlled_routing=False,
            expert_identity_init=False,
            target_gate_mode="none",
            transfer_regret_weight=0.0,
        )
    return replace(
        common,
        source_experts=False,
        risk_controlled_routing=False,
        expert_identity_init=False,
        target_gate_mode="none",
        transfer_regret_weight=0.0,
    )


def _selected_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record["stage"] != "control_branch"]


def _row(method: str, target: str, seed: int, recorder: StageRecorder, transfer_result: dict[str, Any], search_runs: int) -> dict[str, Any]:
    records = _selected_records(recorder.records)
    selection = sum(record["elapsed_seconds"] for record in records if record["stage"] == "source_selection")
    pretrain = sum(record["elapsed_seconds"] for record in records if record["stage"] == "source_pretraining")
    transfer_records = [record for record in records if record["stage"] == "target_transfer"]
    transfer_wall = sum(record["elapsed_seconds"] for record in transfer_records)
    total = selection + pretrain + transfer_wall
    return {
        "method": method,
        "target": target,
        "seed": seed,
        "total_parameters": transfer_result.get("total_parameters"),
        "trainable_parameters": transfer_result.get("trainable_parameters"),
        "source_expert_parameters": transfer_result.get("source_expert_parameters", 0),
        "source_router_parameters": transfer_result.get("source_router_parameters", 0),
        "source_search_runs": int(search_runs),
        "source_selection_wall_seconds": selection,
        "source_pretraining_wall_seconds": pretrain,
        "target_adaptation_wall_seconds": transfer_result.get("target_adaptation_seconds"),
        "calibration_wall_seconds": transfer_result.get("calibration_seconds"),
        "transfer_total_wall_seconds": transfer_wall,
        "total_wall_seconds": total,
        "peak_gpu_allocated_mb": max((record["peak_gpu_allocated_mb"] for record in records), default=0.0),
        "peak_gpu_reserved_mb": max((record["peak_gpu_reserved_mb"] for record in records), default=0.0),
        "inference_seconds_mean": transfer_result.get("inference_seconds_mean"),
        "inference_seconds_std": transfer_result.get("inference_seconds_std"),
        "inference_timing_repetitions": transfer_result.get("inference_timing_repetitions"),
        "inference_warmup_repetitions": transfer_result.get("inference_warmup_repetitions"),
        "inference_ms_per_query": transfer_result.get("inference_ms_per_query"),
        "throughput_queries_per_second": transfer_result.get("inference_queries_per_second"),
        "gpu_hours": total / 3600.0 if torch.device(recorder.device).type == "cuda" else 0.0,
        "test_mrr": transfer_result.get("test_mrr"),
    }


def _profile_pipeline(config: ExperimentConfig, *, method: str, target: str, seed: int, processed_dir: Path, output_root: Path, device: str, repetitions: int, warmups: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    recorder = StageRecorder(device)
    originals = {
        "pretrain": pipeline.run_pretraining,
        "proxy": pipeline.score_checkpoint_proxy,
        "proxy_windows": pipeline.score_checkpoint_proxy_windows,
        "transfer": pipeline.run_transfer,
    }

    def wrapped_pretrain(*args: Any, **kwargs: Any) -> Any:
        checkpoint = kwargs.get("checkpoint_path", args[4] if len(args) > 4 else "")
        stage = classify_pipeline_stage(str(checkpoint))
        return recorder.measure(stage, str(checkpoint), originals["pretrain"], *args, **kwargs)

    def wrapped_proxy(*args: Any, **kwargs: Any) -> Any:
        return recorder.measure("source_selection", "proxy", originals["proxy"], *args, **kwargs)

    def wrapped_proxy_windows(*args: Any, **kwargs: Any) -> Any:
        return recorder.measure("source_selection", "proxy_windows", originals["proxy_windows"], *args, **kwargs)

    transfer_payloads: list[tuple[str, dict[str, Any]]] = []

    def wrapped_transfer(*args: Any, **kwargs: Any) -> Any:
        checkpoint = kwargs.get("checkpoint_path", "")
        stage = "control_branch" if "all_sources" in str(checkpoint) else "target_transfer"
        kwargs["profile_timing_repetitions"] = repetitions
        kwargs["profile_warmup_repetitions"] = warmups
        result = recorder.measure(stage, str(checkpoint), originals["transfer"], *args, **kwargs)
        transfer_payloads.append((stage, result))
        return result

    pipeline.run_pretraining = wrapped_pretrain
    pipeline.score_checkpoint_proxy = wrapped_proxy
    pipeline.score_checkpoint_proxy_windows = wrapped_proxy_windows
    pipeline.run_transfer = wrapped_transfer
    try:
        summary = pipeline.run_pipeline(
            config,
            target=target,
            seed=seed,
            processed_dir=processed_dir,
            output_root=output_root,
            device=device,
        )
    finally:
        pipeline.run_pretraining = originals["pretrain"]
        pipeline.score_checkpoint_proxy = originals["proxy"]
        pipeline.score_checkpoint_proxy_windows = originals["proxy_windows"]
        pipeline.run_transfer = originals["transfer"]
    selected = [payload for stage, payload in transfer_payloads if stage == "target_transfer"]
    if len(selected) != 1:
        raise RuntimeError(f"expected one selected transfer payload, found {len(selected)}")
    return _row(method, target, seed, recorder, selected[0], int(summary.get("candidate_count", 0))), recorder.records


def _profile_direct(config: ExperimentConfig, *, method: str, target: str, seed: int, processed_dir: Path, output_root: Path, device: str, repetitions: int, warmups: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    recorder = StageRecorder(device)
    if method == "target_only_tgat":
        transfer_result = recorder.measure(
            "target_transfer",
            "scratch",
            run_transfer,
            config,
            checkpoint_path=None,
            target=target,
            seed=seed,
            processed_dir=processed_dir,
            mode="scratch",
            device=device,
            evaluation_protocol="ranking",
            negative_policy="historical",
            negatives_per_positive=config.negatives_per_positive,
            evaluation_batch_size=config.evaluation_batch_size,
            profile_timing_repetitions=repetitions,
            profile_warmup_repetitions=warmups,
        )
        return _row(method, target, seed, recorder, transfer_result, 0), recorder.records
    sources = config.sources_for(target)
    checkpoint = output_root / target / f"seed{seed}" / "final" / "all_sources" / "pretrained.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    recorder.measure(
        "source_pretraining",
        str(checkpoint),
        run_pretraining,
        config,
        sources,
        seed,
        processed_dir,
        checkpoint,
        device,
    )
    transfer_result = recorder.measure(
        "target_transfer",
        str(checkpoint),
        run_transfer,
        config,
        checkpoint_path=checkpoint,
        target=target,
        seed=seed,
        processed_dir=processed_dir,
        mode="frozen",
        device=device,
        evaluation_protocol="ranking",
        negative_policy="historical",
        negatives_per_positive=config.negatives_per_positive,
        evaluation_batch_size=config.evaluation_batch_size,
        profile_timing_repetitions=repetitions,
        profile_warmup_repetitions=warmups,
    )
    return _row(method, target, seed, recorder, transfer_result, 0), recorder.records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target", default="wikipedia")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timing-repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    base = ExperimentConfig.from_yaml(args.config)
    hardware = _hardware(args.device)
    rows: list[dict[str, Any]] = []
    raw_root = args.output_root / "raw"
    for method in args.methods:
        method_root = raw_root / method
        if method_root.exists():
            if not args.overwrite:
                raise SystemExit(f"refusing to reuse timed output: {method_root}")
            shutil.rmtree(method_root)
        config = _variant(base, method, args.target, args.seed)
        print(f"START efficiency method={method}", flush=True)
        if method in {"target_only_tgat", "all_source"}:
            row, stage_records = _profile_direct(
                config,
                method=method,
                target=args.target,
                seed=args.seed,
                processed_dir=args.processed_dir,
                output_root=method_root,
                device=args.device,
                repetitions=args.timing_repetitions,
                warmups=args.warmup_repetitions,
            )
        else:
            row, stage_records = _profile_pipeline(
                config,
                method=method,
                target=args.target,
                seed=args.seed,
                processed_dir=args.processed_dir,
                output_root=method_root,
                device=args.device,
                repetitions=args.timing_repetitions,
                warmups=args.warmup_repetitions,
            )
        row["hardware_id"] = "common_host_0"
        rows.append(row)
        method_root.mkdir(parents=True, exist_ok=True)
        (method_root / "efficiency_raw.json").write_text(
            json.dumps({"hardware": hardware, "row": row, "stage_records": stage_records}, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"FINISH efficiency method={method} total={row['total_wall_seconds']:.3f}s", flush=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps({"hardware": hardware, "rows": rows}, indent=2), encoding="utf-8")
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "hardware": hardware}, indent=2), flush=True)


if __name__ == "__main__":
    main()
