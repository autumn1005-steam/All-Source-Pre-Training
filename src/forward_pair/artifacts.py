"""Validation and serialization for Forward-Pair screening artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .selector import CandidateScore, SelectionResult


@dataclass(frozen=True)
class ArtifactPaths:
    candidate_scores_csv: Path
    selection_json: Path
    domain_weights_json: Path


def validate_candidate_scores(
    scores: Iterable[CandidateScore],
) -> tuple[CandidateScore, ...]:
    records = tuple(scores)
    if not records:
        raise ValueError("candidate scores cannot be empty")
    for field in ("seed", "target", "config_digest"):
        if len({getattr(score, field) for score in records}) != 1:
            raise ValueError(f"candidate scores have inconsistent {field}")
    for score in records:
        if not all(
            isinstance(value, (int, float)) and value == value and abs(value) != float("inf")
            for value in (score.proxy_ap, score.proxy_auc)
        ):
            raise ValueError("candidate proxy scores must be finite")
    configs = [score.config for score in records]
    if len(configs) != len(set(configs)):
        raise ValueError("candidate configs must be unique")
    return records


def write_selection_artifacts(
    output_dir: str | Path,
    scores: Iterable[CandidateScore],
    result: SelectionResult,
    *,
    source_commits: Mapping[str, str],
) -> ArtifactPaths:
    records = validate_candidate_scores(scores)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    score_path = root / "candidate_scores.csv"
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "config",
                "sources",
                "proxy_ap",
                "proxy_auc",
                "seed",
                "target",
                "config_digest",
            ),
        )
        writer.writeheader()
        for score in records:
            row = score.to_dict()
            row["sources"] = "+".join(score.sources)
            writer.writerow(row)

    selection_path = root / "selection.json"
    payload = result.to_dict()
    payload["source_commits"] = dict(source_commits)
    selection_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    weights_path = root / "domain_weights.json"
    weights_path.write_text(
        json.dumps(
            {"weights": {source: 1.0 for source in result.selected_sources}},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return ArtifactPaths(score_path, selection_path, weights_path)
