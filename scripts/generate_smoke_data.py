#!/usr/bin/env python3
"""Generate deterministic tiny processed graphs for CPU smoke testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for phase, name in enumerate(
        ("source_a", "source_b", "source_c", "source_d", "target")
    ):
        count = 24
        order = np.arange(count)
        frame = pd.DataFrame(
            {
                "u": ((order + phase) % 5) + 1,
                "i": ((order + phase + 1) % 5) + 1,
                "idx": order + 1,
                "ts": order.astype(float) + 1.0,
                "label": np.zeros(count, dtype=int),
            }
        )
        frame.to_csv(root / f"ml_{name}.csv", index=False)
        np.save(
            root / f"ml_{name}_node.npy",
            np.arange(18, dtype=np.float32).reshape(6, 3) / (20.0 + phase),
        )
        np.save(
            root / f"ml_{name}.npy",
            np.arange(50, dtype=np.float32).reshape(25, 2) / (30.0 + phase),
        )


if __name__ == "__main__":
    main()
