#!/usr/bin/env python3
"""Measure source-expert/routing computational scaling on synthetic inputs.

This is a computational profile only. It deliberately does not report MRR or
represent synthetic expert replication as a benchmark dataset experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from tempofound.temporal_experts import RiskControlledSourceRouter, SourceExpertBank


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_expert_counts(
    *,
    source_counts: Iterable[int],
    dim: int,
    batch_size: int,
    device: str,
    warmups: int,
    repetitions: int,
) -> dict:
    resolved = torch.device(device)
    rows = []
    for count in source_counts:
        if count <= 0:
            raise ValueError("source counts must be positive")
        names = tuple(f"synthetic_source_{index}" for index in range(count))
        bank = SourceExpertBank(names, dim, dropout=0.0).to(resolved).eval()
        router = RiskControlledSourceRouter(names, dim, dropout=0.0).to(resolved).eval()
        router.set_prior({name: 1.0 for name in names})
        embeddings = torch.randn(batch_size, dim, device=resolved)
        history = torch.randn(batch_size, 5, device=resolved)
        history[:, 4] = 1.0

        def forward() -> None:
            with torch.inference_mode():
                deltas = bank.stack_target_deltas(embeddings, history)
                router(embeddings, history, deltas)

        for _ in range(warmups):
            forward()
        _sync(resolved)
        if resolved.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved)
        durations = []
        for _ in range(repetitions):
            _sync(resolved)
            started = time.perf_counter()
            forward()
            _sync(resolved)
            durations.append(time.perf_counter() - started)
        peak_allocated = peak_reserved = 0.0
        if resolved.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(resolved) / 2**20
            peak_reserved = torch.cuda.max_memory_reserved(resolved) / 2**20
        bank_parameters = sum(parameter.numel() for parameter in bank.parameters())
        router_parameters = sum(parameter.numel() for parameter in router.parameters())
        mean_seconds = statistics.mean(durations)
        rows.append(
            {
                "sources": count,
                "shared_dim": dim,
                "batch_size": batch_size,
                "expert_bank_parameters": bank_parameters,
                "router_parameters": router_parameters,
                "total_module_parameters": bank_parameters + router_parameters,
                "peak_gpu_allocated_mb": peak_allocated,
                "peak_gpu_reserved_mb": peak_reserved,
                "latency_ms_per_batch_mean": 1000.0 * mean_seconds,
                "latency_ms_per_batch_std": 1000.0 * (statistics.stdev(durations) if len(durations) > 1 else 0.0),
                "latency_ms_per_query": 1000.0 * mean_seconds / batch_size,
                "throughput_queries_per_second": batch_size / mean_seconds,
                "timing_repetitions": repetitions,
                "warmup_repetitions": warmups,
                "profile_scope": "synthetic_expert_routing_only",
                "performance_metric": "not_applicable",
            }
        )
    return {
        "profile_scope": "computational scaling profile with synthetic inputs; no benchmark performance claim",
        "hardware": {
            "device": str(resolved),
            "gpu": torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else None,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-counts", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()
    payload = profile_expert_counts(
        source_counts=args.source_counts,
        dim=args.dim,
        batch_size=args.batch_size,
        device=args.device,
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["rows"][0]))
        writer.writeheader()
        writer.writerows(payload["rows"])
    print(json.dumps({"rows": len(payload["rows"]), "scope": payload["profile_scope"]}))


if __name__ == "__main__":
    main()
