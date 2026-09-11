import pytest
from scripts.analyze_oos_safety import _spearman

def test_spearman_monotone():
    assert _spearman([1.0,2.0,3.0],[2.0,4.0,8.0]) == pytest.approx(1.0)
    assert _spearman([1.0,2.0,3.0],[8.0,4.0,2.0]) == pytest.approx(-1.0)
