import numpy as np

from tempofound.graph import NeighborFinder


def test_equal_timestamp_is_not_historical() -> None:
    finder = NeighborFinder.from_events(
        sources=np.array([1, 1]),
        destinations=np.array([2, 3]),
        edge_indices=np.array([1, 2]),
        timestamps=np.array([5.0, 6.0]),
    )

    nodes, edges, times = finder.get_temporal_neighbor(
        np.array([1]), np.array([6.0]), num_neighbors=3
    )

    assert nodes.tolist() == [[0, 0, 2]]
    assert edges.tolist() == [[0, 0, 1]]
    assert times.tolist() == [[0.0, 0.0, 5.0]]


def test_neighbor_results_keep_only_most_recent_history() -> None:
    finder = NeighborFinder.from_events(
        sources=np.array([1, 1, 1]),
        destinations=np.array([2, 3, 4]),
        edge_indices=np.array([10, 11, 12]),
        timestamps=np.array([1.0, 2.0, 3.0]),
    )

    nodes, _, times = finder.get_temporal_neighbor(
        np.array([1]), np.array([4.0]), num_neighbors=2
    )

    assert nodes.tolist() == [[3, 4]]
    assert times.tolist() == [[2.0, 3.0]]
