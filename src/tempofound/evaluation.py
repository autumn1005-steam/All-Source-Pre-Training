"""Streaming ranking evaluation for temporal link prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .sampling import TemporalNegativeSampler


def reciprocal_ranks(
    positive_logits: torch.Tensor, negative_logits: torch.Tensor
) -> torch.Tensor:
    """Return reciprocal ranks with average ranks for score ties."""

    if positive_logits.ndim != 1 or negative_logits.ndim != 2:
        raise ValueError("positive shape must be [batch] and negative shape [batch, n]")
    if len(positive_logits) != len(negative_logits) or negative_logits.shape[1] < 1:
        raise ValueError("positive and negative shapes must have the same non-empty batch")
    if not torch.isfinite(positive_logits).all() or not torch.isfinite(
        negative_logits
    ).all():
        raise ValueError("ranking logits must be finite")
    positive = positive_logits.unsqueeze(1)
    optimistic = 1 + (negative_logits > positive).sum(dim=1)
    pessimistic = 1 + (negative_logits >= positive).sum(dim=1)
    average_rank = (optimistic + pessimistic).to(torch.float32) / 2.0
    return average_rank.reciprocal()


@torch.no_grad()
def evaluate_link_ranking(
    model: Any,
    head: Any,
    target: str,
    data: dict[str, Any],
    split: str,
    sampler: TemporalNegativeSampler,
    *,
    negatives_per_positive: int,
    batch_size: int,
    policy: str,
    fallback_policy: str | None = None,
    use_source_router: bool = True,
) -> dict[str, Any]:
    """Evaluate fixed-negative MRR without materializing the full split."""

    if negatives_per_positive <= 0 or batch_size <= 0:
        raise ValueError("negatives_per_positive and batch_size must be positive")
    source = np.asarray(data[f"{split}_src"], dtype=np.int64)
    destination = np.asarray(data[f"{split}_dst"], dtype=np.int64)
    timestamps = np.asarray(data[f"{split}_ts"], dtype=np.float64)
    if not (len(source) == len(destination) == len(timestamps)):
        raise ValueError(f"{split} source, destination, and timestamp arrays must align")
    if not len(source):
        raise ValueError(f"{split} split is empty")
    edge_key = f"{split}_eidx"
    query_ids = np.asarray(
        data.get(edge_key, np.arange(len(source), dtype=np.int64))
    )
    if query_ids.ndim != 1 or len(query_ids) != len(source):
        raise ValueError(f"{edge_key} must align with the split")

    model.eval()
    head.eval()

    def encode(nodes: np.ndarray, times: np.ndarray):
        # Preserve compatibility with lightweight test/backbone mocks that do
        # not expose the optional routing keyword on the default routed path.
        if use_source_router:
            return model.encode_nodes(target, data["neighbor_finder"], nodes, times)
        return model.encode_nodes(
            target, data["neighbor_finder"], nodes, times, use_source_router=False
        )

    reciprocal_rank_chunks: list[torch.Tensor] = []
    total_queries = int(len(source))
    valid_queries = 0
    fallback_count = 0
    collision_rejection_count = 0
    active_fallback_policy = policy if fallback_policy is None else fallback_policy
    for start in range(0, len(source), batch_size):
        stop = min(start + batch_size, len(source))
        batch_source = source[start:stop]
        batch_destination = destination[start:stop]
        batch_timestamps = timestamps[start:stop]
        batch_query_ids = query_ids[start:stop]
        sampleable = sampler.sampleable_mask(
            batch_source,
            batch_timestamps,
            policy=policy,
            fallback_policy=active_fallback_policy,
        )
        for query_source, query_time in zip(batch_source, batch_timestamps, strict=True):
            pool = sampler._policy_pool(float(query_time), policy)
            valid = sampler.valid_candidates(int(query_source), float(query_time), policy=policy)
            collision_rejection_count += max(0, int(len(pool) - len(valid)))
        if not sampleable.any():
            continue
        batch_source = batch_source[sampleable]
        batch_destination = batch_destination[sampleable]
        batch_timestamps = batch_timestamps[sampleable]
        batch_query_ids = batch_query_ids[sampleable]
        valid_queries += int(len(batch_source))
        negative_destinations, batch_fallbacks = sampler.sample_with_fallback(
            batch_source,
            batch_timestamps,
            count=negatives_per_positive,
            policy=policy,
            fallback_policy=active_fallback_policy,
            query_ids=batch_query_ids,
        )
        fallback_count += int(batch_fallbacks)
        source_embeddings = encode(batch_source, batch_timestamps)
        destination_embeddings = encode(batch_destination, batch_timestamps)
        positive_logits = head(
            source_embeddings, destination_embeddings
        ).reshape(-1)

        flat_negative = negative_destinations.reshape(-1)
        repeated_timestamps = np.repeat(batch_timestamps, negatives_per_positive)
        negative_embeddings = encode(flat_negative, repeated_timestamps)
        repeated_source = (
            source_embeddings[:, None, :]
            .expand(-1, negatives_per_positive, -1)
            .reshape(-1, source_embeddings.shape[-1])
        )
        negative_logits = head(repeated_source, negative_embeddings).reshape(
            len(batch_source), negatives_per_positive
        )
        reciprocal_rank_chunks.append(
            reciprocal_ranks(positive_logits, negative_logits).cpu()
        )

    if not reciprocal_rank_chunks:
        raise ValueError(f"{split} split has no events with valid negative candidates")
    reciprocal_rank = torch.cat(reciprocal_rank_chunks)
    return {
        "mrr": float(reciprocal_rank.mean()),
        "num_positives": int(len(reciprocal_rank)),
        "total_queries": total_queries,
        "valid_queries": valid_queries,
        "skipped_queries": total_queries - valid_queries,
        "fallback_count": fallback_count,
        "collision_rejection_count": int(collision_rejection_count),
        "negatives_per_positive": int(negatives_per_positive),
        "negative_policy": policy,
        # Explicitly expose the no-valid-negative behavior so sparse-domain
        # audits can distinguish skipped queries from fallback sampling.
        "fallback_policy": active_fallback_policy,
    }
