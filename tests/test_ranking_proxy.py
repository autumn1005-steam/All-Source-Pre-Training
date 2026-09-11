import numpy as np
import pytest
import forward_pair.proxy as proxy


def test_ranking_proxy_uses_mrr_and_training_events_only(monkeypatch):
    train = np.arange(1, 21)
    data = dict(train_src=np.ones(20, dtype=int), train_dst=train + 1,
                train_ts=train.astype(float), neighbor_finder=object(),
                node_features=np.zeros((22, 2)), edge_features=np.zeros((21, 2)),
                val_dst=np.array([999]), test_dst=np.array([1000]))
    monkeypatch.setattr(proxy, '_calibrate_proxy', lambda *a, **k:
        (object(), object(), np.random.default_rng(1), np.array([2]), [0.5]))
    monkeypatch.setattr(proxy, '_score_probe_partition', lambda *a, **k: (0.95, 0.95))
    def evaluate(model, head, target, window, split, sampler, **kwargs):
        assert split == 'train'
        assert not any(key.startswith(('val_', 'test_')) for key in window)
        assert sampler.destination_nodes.max() == 21
        assert kwargs['negatives_per_positive'] == 20
        return {'mrr': 0.25}
    monkeypatch.setattr(proxy, 'evaluate_link_ranking', evaluate)
    result = proxy.score_target_train_proxy_windows(object(), target='x', target_data=data,
        seed=42, batch_size=2, calibration_steps=1, probe_batches=1, calibration_fraction=.5,
        proxy_windows=2, bootstrap_replicates=20, bootstrap_confidence=.95,
        risk_beta=1, size_penalty=0, cardinality=1, proxy_metric='mrr')
    assert result['window_proxy_ap'] == [0.95, 0.95]
    assert result['window_proxy_scores'] == [0.25, 0.25]
    assert result['risk_adjusted_utility'] == pytest.approx(.25)
