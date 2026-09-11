"""Causal temporal-neighbor indexing."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class NeighborFinder:
    """Return only neighbors whose event time is strictly before a query."""

    def __init__(self, adjacency: Sequence[Sequence[tuple[int, int, float]]]):
        nodes: list[int] = []
        edges: list[int] = []
        times: list[float] = []
        offsets = [0]
        for entries in adjacency:
            ordered = sorted(entries, key=lambda item: item[2])
            nodes.extend(int(item[0]) for item in ordered)
            edges.extend(int(item[1]) for item in ordered)
            times.extend(float(item[2]) for item in ordered)
            offsets.append(len(nodes))
        self.node_indices = np.asarray(nodes, dtype=np.int64)
        self.edge_indices = np.asarray(edges, dtype=np.int64)
        self.timestamps = np.asarray(times, dtype=np.float64)
        self.offsets = np.asarray(offsets, dtype=np.int64)

    @classmethod
    def from_events(
        cls,
        sources: np.ndarray,
        destinations: np.ndarray,
        edge_indices: np.ndarray,
        timestamps: np.ndarray,
    ) -> "NeighborFinder":
        if not (
            len(sources) == len(destinations) == len(edge_indices) == len(timestamps)
        ):
            raise ValueError("event arrays must have equal length")
        max_node = int(max(np.max(sources), np.max(destinations))) if len(sources) else 0
        adjacency: list[list[tuple[int, int, float]]] = [
            [] for _ in range(max_node + 1)
        ]
        for source, destination, edge_index, timestamp in zip(
            sources, destinations, edge_indices, timestamps, strict=True
        ):
            adjacency[int(source)].append(
                (int(destination), int(edge_index), float(timestamp))
            )
            adjacency[int(destination)].append(
                (int(source), int(edge_index), float(timestamp))
            )
        return cls(adjacency)

    def find_before(
        self, node: int, query_time: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if node < 0 or node + 1 >= len(self.offsets):
            raise IndexError(f"node index out of range: {node}")
        start, stop = self.offsets[node : node + 2]
        times = self.timestamps[start:stop]
        before = np.searchsorted(times, query_time, side="left")
        return (
            self.node_indices[start : start + before],
            self.edge_indices[start : start + before],
            times[:before],
        )

    def get_temporal_neighbor(
        self,
        nodes: np.ndarray,
        query_times: np.ndarray,
        num_neighbors: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(nodes) != len(query_times):
            raise ValueError("nodes and query_times must have equal length")
        if num_neighbors <= 0:
            raise ValueError("num_neighbors must be positive")
        result_nodes = np.zeros((len(nodes), num_neighbors), dtype=np.int64)
        result_edges = np.zeros((len(nodes), num_neighbors), dtype=np.int64)
        result_times = np.zeros((len(nodes), num_neighbors), dtype=np.float64)
        for row, (node, query_time) in enumerate(zip(nodes, query_times, strict=True)):
            history_nodes, history_edges, history_times = self.find_before(
                int(node), float(query_time)
            )
            keep = min(len(history_nodes), num_neighbors)
            if keep:
                result_nodes[row, -keep:] = history_nodes[-keep:]
                result_edges[row, -keep:] = history_edges[-keep:]
                result_times[row, -keep:] = history_times[-keep:]
        return result_nodes, result_edges, result_times
