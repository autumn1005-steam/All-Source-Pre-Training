import numpy as np

from tempofound.pretrain import _make_pretraining_batch


class _StatsModel:
    def compute_neighborhood_stats(self, finder, source, timestamps):
        return np.zeros((len(source), 5), dtype=np.float32)


def test_pretraining_negatives_exclude_source_history_at_query_time():
    data = {
        'train_src': np.array([1, 1, 1, 2]),
        'train_dst': np.array([2, 3, 4, 5]),
        'train_ts': np.array([1., 2., 3., 1.]),
        'neighbor_finder': object(),
    }
    batch = _make_pretraining_batch(_StatsModel(), data, np.array([1, 2]), np.random.default_rng(9))
    # row 0 is source 1 at t=2: destinations 2 and 3 are positives/history
    assert batch['neg_dst_idx'][0] not in {2, 3}
    # row 1 is source 1 at t=3: destinations 2,3,4 are positives/history
    assert batch['neg_dst_idx'][1] not in {2, 3, 4}
