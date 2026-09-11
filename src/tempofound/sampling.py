"""Deterministic collision-free negative sampling for temporal links."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


NEGATIVE_POLICIES = ("random", "historical", "inductive")


@dataclass(frozen=True)
class TemporalNegativeSampler:
    """Sample destination nodes without colliding with observed source links.

    The sampler is immutable. Each query gets its own deterministic random
    stream, so fixed query identifiers make results independent of evaluation
    batch size and process restarts.
    """

    destination_nodes: np.ndarray
    destination_first_seen: dict[int, float]
    source_histories: dict[int, tuple[np.ndarray, np.ndarray]]
    seed: int
    training_end: float | None = None

    @classmethod
    def from_events(
        cls,
        sources: np.ndarray,
        destinations: np.ndarray,
        timestamps: np.ndarray,
        seed: int,
        training_end: float | None = None,
    ) -> "TemporalNegativeSampler":
        source_array = np.asarray(sources, dtype=np.int64)
        destination_array = np.asarray(destinations, dtype=np.int64)
        time_array = np.asarray(timestamps, dtype=np.float64)
        if not (
            source_array.ndim
            == destination_array.ndim
            == time_array.ndim
            == 1
        ):
            raise ValueError("sources, destinations, and timestamps must be 1-D")
        if not (
            len(source_array) == len(destination_array) == len(time_array)
        ):
            raise ValueError("sources, destinations, and timestamps must align")
        if not len(source_array):
            raise ValueError("at least one temporal event is required")
        if np.any(source_array < 0) or np.any(destination_array <= 0):
            raise ValueError("source IDs must be non-negative and destinations positive")
        if not np.isfinite(time_array).all():
            raise ValueError("timestamps must be finite")
        if training_end is not None and not np.isfinite(training_end):
            raise ValueError("training_end must be finite")

        order = np.argsort(time_array, kind="stable")
        ordered_sources = source_array[order]
        ordered_destinations = destination_array[order]
        ordered_times = time_array[order]
        first_seen: dict[int, float] = {}
        history_lists: dict[int, tuple[list[int], list[float]]] = {}
        for source, destination, timestamp in zip(
            ordered_sources,
            ordered_destinations,
            ordered_times,
            strict=True,
        ):
            destination_id = int(destination)
            first_seen.setdefault(destination_id, float(timestamp))
            destinations_for_source, times_for_source = history_lists.setdefault(
                int(source), ([], [])
            )
            destinations_for_source.append(destination_id)
            times_for_source.append(float(timestamp))
        histories = {
            source: (
                np.asarray(destinations_for_source, dtype=np.int64),
                np.asarray(times_for_source, dtype=np.float64),
            )
            for source, (destinations_for_source, times_for_source) in history_lists.items()
        }
        return cls(
            destination_nodes=np.unique(destination_array),
            destination_first_seen=first_seen,
            source_histories=histories,
            seed=int(seed),
            training_end=None if training_end is None else float(training_end),
        )

    def _policy_pool(self, timestamp: float, policy: str) -> np.ndarray:
        if policy not in NEGATIVE_POLICIES:
            raise ValueError(
                f"negative policy must be one of {', '.join(NEGATIVE_POLICIES)}"
            )
        if policy == "random":
            return self.destination_nodes
        if policy == "inductive" and self.training_end is None:
            raise ValueError("inductive negative sampling requires training_end")
        if policy == "historical":
            return np.asarray(
                sorted(
                    node
                    for node, first_time in self.destination_first_seen.items()
                    if first_time < timestamp
                ),
                dtype=np.int64,
            )
        assert self.training_end is not None
        return np.asarray(
            sorted(
                node
                for node, first_time in self.destination_first_seen.items()
                if self.training_end < first_time <= timestamp
            ),
            dtype=np.int64,
        )

    def valid_candidates(
        self, source: int, timestamp: float, *, policy: str
    ) -> np.ndarray:
        if not np.isfinite(timestamp):
            raise ValueError("query timestamps must be finite")
        pool = self._policy_pool(float(timestamp), policy)
        history = self.source_histories.get(int(source))
        if history is None or not len(pool):
            return pool.copy()
        destinations, times = history
        boundary = int(np.searchsorted(times, timestamp, side="right"))
        observed = np.unique(destinations[:boundary])
        return np.setdiff1d(pool, observed, assume_unique=True)

    def _rng(
        self,
        *,
        source: int,
        timestamp: float,
        query_id: object | None,
        policy: str,
    ) -> np.random.Generator:
        identity = "" if query_id is None else str(query_id)
        payload = f"{self.seed}|{source}|{timestamp:.17g}|{identity}|{policy}".encode()
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return np.random.default_rng(int.from_bytes(digest, byteorder="little"))

    def sample(
        self,
        sources: np.ndarray,
        timestamps: np.ndarray,
        count: int,
        policy: str,
        query_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        source_array = np.asarray(sources, dtype=np.int64)
        time_array = np.asarray(timestamps, dtype=np.float64)
        if source_array.ndim != 1 or time_array.ndim != 1:
            raise ValueError("query sources and timestamps must be 1-D")
        if len(source_array) != len(time_array):
            raise ValueError("query sources and timestamps must align")
        if count <= 0:
            raise ValueError("negative count must be positive")
        id_array: np.ndarray | None = None
        if query_ids is not None:
            id_array = np.asarray(query_ids)
            if id_array.ndim != 1 or len(id_array) != len(source_array):
                raise ValueError("query_ids must be a 1-D array aligned with queries")
        output = np.empty((len(source_array), count), dtype=np.int64)
        for index, (source, timestamp) in enumerate(
            zip(source_array, time_array, strict=True)
        ):
            candidates = self.valid_candidates(
                int(source), float(timestamp), policy=policy
            )
            if not len(candidates):
                raise ValueError(
                    "no valid negative candidates for "
                    f"source={int(source)}, time={float(timestamp):.17g}, "
                    f"policy={policy}, requested={count}, available=0"
                )
            query_id = None if id_array is None else id_array[index]
            rng = self._rng(
                source=int(source),
                timestamp=float(timestamp),
                query_id=query_id,
                policy=policy,
            )
            output[index] = rng.choice(
                candidates,
                size=count,
                replace=len(candidates) < count,
            )
        return output
    def sample_with_fallback(
        self,
        sources: np.ndarray,
        timestamps: np.ndarray,
        count: int,
        *,
        policy: str,
        fallback_policy: str = "random",
        query_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        """Sample per query, falling back only when the primary pool is empty.

        The fallback still uses ``valid_candidates`` and therefore never
        collides with an observed source-destination edge up to the query time.
        This is useful for cold-start events where a strictly historical
        destination universe can be empty.
        """
        source_array = np.asarray(sources, dtype=np.int64)
        time_array = np.asarray(timestamps, dtype=np.float64)
        if source_array.ndim != 1 or time_array.ndim != 1 or len(source_array) != len(time_array):
            raise ValueError("query sources and timestamps must be aligned 1-D arrays")
        ids = None if query_ids is None else np.asarray(query_ids)
        if ids is not None and (ids.ndim != 1 or len(ids) != len(source_array)):
            raise ValueError("query_ids must align with queries")
        output = np.empty((len(source_array), count), dtype=np.int64)
        fallback_count = 0
        for index, (source, timestamp) in enumerate(zip(source_array, time_array, strict=True)):
            active_policy = policy
            if not len(self.valid_candidates(int(source), float(timestamp), policy=active_policy)):
                active_policy = fallback_policy
                fallback_count += 1
            query_id = None if ids is None else np.asarray([ids[index]])
            output[index] = self.sample(
                np.asarray([source]),
                np.asarray([timestamp]),
                count=count,
                policy=active_policy,
                query_ids=query_id,
            )[0]
        return output, fallback_count

    def sampleable_mask(
        self,
        sources: np.ndarray,
        timestamps: np.ndarray,
        *,
        policy: str,
        fallback_policy: str = "random",
    ) -> np.ndarray:
        """Return queries that admit a collision-free negative candidate.

        Some very small or saturated temporal graphs contain source nodes that
        have already interacted with every destination in the dataset.  Such
        events have no mathematically valid negative under either the primary
        or fallback policy and must be excluded from ranking loss/evaluation.
        """
        source_array = np.asarray(sources, dtype=np.int64)
        time_array = np.asarray(timestamps, dtype=np.float64)
        if source_array.ndim != 1 or time_array.ndim != 1 or len(source_array) != len(time_array):
            raise ValueError("query sources and timestamps must be aligned 1-D arrays")
        return np.asarray(
            [
                bool(len(self.valid_candidates(int(source), float(timestamp), policy=policy)))
                or bool(len(self.valid_candidates(int(source), float(timestamp), policy=fallback_policy)))
                for source, timestamp in zip(source_array, time_array, strict=True)
            ],
            dtype=bool,
        )
