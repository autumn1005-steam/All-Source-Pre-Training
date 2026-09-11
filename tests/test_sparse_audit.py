from scripts.audit_sparse_protocol import summarize_result_payload


def test_sparse_audit_preserves_explicit_query_counters() -> None:
    payload = {
        "target": "socialevo",
        "seed": 42,
        "test_mrr": 0.2,
        "total_queries": 100,
        "valid_queries": 93,
        "skipped_queries": 7,
        "fallback_count": 4,
        "collision_rejection_count": 12,
    }

    row = summarize_result_payload(payload, source="demo.json")

    assert row["total_queries"] == 100
    assert row["valid_queries"] == 93
    assert row["skipped_queries"] == 7
    assert row["fallback_count"] == 4
    assert row["collision_rejection_count"] == 12


def test_sparse_audit_marks_unavailable_counters_as_na() -> None:
    row = summarize_result_payload({"target": "socialevo"}, source="legacy.json")

    assert row["total_queries"] == "NA"
    assert row["valid_queries"] == "NA"
    assert row["skipped_queries"] == "NA"
