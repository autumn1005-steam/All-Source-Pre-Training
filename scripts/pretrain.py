#!/usr/bin/env python3
"""Train a TempoFound checkpoint on an explicit source set."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tempofound.config import ExperimentConfig
from tempofound.pretrain import run_pretraining


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--target")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed-dir")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    if args.print_config:
        print(json.dumps(asdict(config), sort_keys=True))
        return
    sources = tuple(args.sources or ())
    if not sources:
        if not args.target:
            raise ValueError("--sources or --target is required")
        sources = config.sources_for(args.target)
    if not args.processed_dir or not args.output:
        raise ValueError("--processed-dir and --output are required for training")
    summary = run_pretraining(
        config,
        sources,
        args.seed,
        args.processed_dir,
        args.output,
        args.device,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
