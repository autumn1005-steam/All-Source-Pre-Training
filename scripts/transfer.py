#!/usr/bin/env python3
"""Run target transfer from a selected-source TempoFound checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tempofound.config import ExperimentConfig
from tempofound.transfer import run_transfer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--target", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--mode", choices=("frozen", "full", "scratch"), default="frozen")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_transfer(
        ExperimentConfig.from_yaml(args.config),
        checkpoint_path=args.checkpoint,
        target=args.target,
        seed=args.seed,
        processed_dir=args.processed_dir,
        mode=args.mode,
        device=args.device,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
