import numpy as np

from forward_pair.proxy import build_proxy_partitions


def test_proxy_partitions_exclude_validation_and_test_arrays() -> None:
    target_data = {
        "train_src": np.array([4, 1, 3, 2, 5]),
        "train_dst": np.array([5, 2, 4, 3, 1]),
        "train_ts": np.array([4.0, 1.0, 3.0, 2.0, 5.0]),
        "val_src": np.array([9]),
        "val_dst": np.array([10]),
        "val_ts": np.array([6.0]),
        "test_src": np.array([11]),
        "test_dst": np.array([12]),
        "test_ts": np.array([7.0]),
        "neighbor_finder": object(),
        "node_features": np.zeros((13, 2), dtype=np.float32),
        "edge_features": np.zeros((8, 2), dtype=np.float32),
    }

    calibration, probe = build_proxy_partitions(target_data, 0.8)

    allowed = {
        "train_src",
        "train_dst",
        "train_ts",
        "neighbor_finder",
        "node_features",
        "edge_features",
    }
    assert set(calibration) == allowed
    assert set(probe) == allowed
    assert calibration["train_ts"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert probe["train_ts"].tolist() == [5.0]
    assert calibration["train_ts"].max() <= probe["train_ts"].min()
