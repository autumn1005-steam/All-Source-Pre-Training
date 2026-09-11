import pytest
from forward_pair.uncertainty import paired_temporal_gate


def gate(candidate, reference):
    return paired_temporal_gate(candidate, reference, seed=42, replicates=1000,
        confidence=0.95, comparisons=2, min_gain=0.002)


def test_common_temporal_difficulty_cancels_in_paired_gain():
    result = gate([0.4, 0.9, 0.6], [0.3, 0.8, 0.5])
    assert result['accepted']
    assert result['lower_delta'] == pytest.approx(0.1)


def test_positive_average_does_not_hide_recent_negative_transfer():
    result = gate([0.9, 0.9, 0.49], [0.5, 0.5, 0.5])
    assert result['mean_delta'] > 0
    assert not result['accepted']


def test_tie_and_harm_do_not_pass_gate():
    assert not gate([0.5] * 3, [0.5] * 3)['accepted']
    assert not gate([0.4] * 3, [0.5] * 3)['accepted']


def test_unpaired_or_nonfinite_data_rejected():
    with pytest.raises(ValueError):
        gate([0.5, 0.6], [0.5, 0.6, 0.7])
    with pytest.raises(ValueError):
        gate([0.5, float('nan')], [0.5, 0.6])
