"""Processed temporal-graph loading and chronological partitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .graph import NeighborFinder


def split_target_train_period(
    target_data: dict[str, Any], calibration_fraction: float = 0.8
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")
    timestamps = np.asarray(target_data["train_ts"])
    order = np.argsort(timestamps, kind="stable")
    split = int(len(order) * calibration_fraction)
    if split == 0 or split == len(order):
        raise ValueError("target training period is too small for calibration/probe split")

    def partition(indices: np.ndarray) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in target_data.items():
            if key.startswith("train_") and hasattr(value, "__len__") and len(value) == len(order):
                output[key] = np.asarray(value)[indices]
            else:
                output[key] = value
        return output

    return partition(order[:split]), partition(order[split:])


def load_processed_dataset(name: str, processed_dir: str | Path) -> dict[str, Any]:
    root = Path(processed_dir)
    csv_path = root / f"ml_{name}.csv"
    edge_path = root / f"ml_{name}.npy"
    node_path = root / f"ml_{name}_node.npy"
    missing = [str(path) for path in (csv_path, edge_path, node_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing processed dataset files: " + ", ".join(missing))

    frame = pd.read_csv(csv_path)
    required = {"u", "i", "idx", "ts"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{csv_path} requires columns: {sorted(required)}")
    sources = frame["u"].to_numpy(dtype=np.int64)
    destinations = frame["i"].to_numpy(dtype=np.int64)
    edge_indices = frame["idx"].to_numpy(dtype=np.int64)
    timestamps = frame["ts"].to_numpy(dtype=np.float64)
    validation_time, test_time = np.quantile(timestamps, [0.70, 0.85])
    train = timestamps <= validation_time
    validation = (timestamps > validation_time) & (timestamps <= test_time)
    test = timestamps > test_time
    finder = NeighborFinder.from_events(
        sources, destinations, edge_indices, timestamps
    )
    return {
        "name": name,
        "node_features": np.load(node_path),
        "edge_features": np.load(edge_path),
        "neighbor_finder": finder,
        "train_src": sources[train],
        "train_dst": destinations[train],
        "train_eidx": edge_indices[train],
        "train_ts": timestamps[train],
        "val_src": sources[validation],
        "val_dst": destinations[validation],
        "val_eidx": edge_indices[validation],
        "val_ts": timestamps[validation],
        "test_src": sources[test],
        "test_dst": destinations[test],
        "test_eidx": edge_indices[test],
        "test_ts": timestamps[test],
        "validation_time": float(validation_time),
        "test_time": float(test_time),
    }
