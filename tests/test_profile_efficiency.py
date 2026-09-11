import json

from scripts.profile_efficiency import profile_root


def test_profile_efficiency_preserves_missing_costs_as_na(tmp_path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps({
        "target": "wikipedia", "seed": 0, "elapsed_seconds": 12.5,
        "training_diagnostics": {"rasp": {"trainable_parameters": 1234}}
    }), encoding="utf-8")
    rows = profile_root(tmp_path)
    assert rows[0]["elapsed_seconds"] == 12.5
    assert rows[0]["trainable_parameters"] == 1234
    assert rows[0]["peak_gpu_memory_mb"] == "NA"
