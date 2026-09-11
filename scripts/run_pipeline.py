#!/usr/bin/env python3
"""Run the unified Forward-Pair and TempoFound experiment pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forward_pair.pipeline import build_forward_pair_plan, build_rasp_plan, run_pipeline
from tempofound.config import ExperimentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--processed-dir")
    parser.add_argument("--output-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    targets = tuple(args.targets or config.targets)
    if args.plan_only:
        payload = []
        for target in targets:
            for seed in tuple(args.seeds or config.seeds_for(target)):
                if config.selection_method == "rasp":
                    plan = build_rasp_plan(config, target=target, seed=seed)
                    payload.append(
                        {
                            "method": "rasp",
                            "target": target,
                            "seed": seed,
                            "singleton_count": len(plan.initial_candidates),
                            "beam_width": plan.beam_width,
                            "max_sources": plan.max_sources,
                            "candidate_budget": plan.candidate_budget,
                        }
                    )
                else:
                    plan = build_forward_pair_plan(config, target=target, seed=seed)
                    payload.append(
                        {
                            "method": "forward_pair",
                            "target": target,
                            "seed": seed,
                            "singleton_count": len(plan.singleton_jobs),
                            "anchor_pair_count": plan.dynamic_anchor_pair_count,
                            "candidate_count": plan.expected_candidate_count,
                        }
                    )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if not args.processed_dir or not args.output_root:
        raise ValueError("--processed-dir and --output-root are required")
    summaries = []
    for target in targets:
        for seed in tuple(args.seeds or config.seeds_for(target)):
            summaries.append(
                run_pipeline(
                    config,
                    target=target,
                    seed=seed,
                    processed_dir=args.processed_dir,
                    output_root=args.output_root,
                    device=args.device,
                )
            )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
