"""Download official JODIE data and record deterministic bipartite conversion."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    raw, processed = root / 'raw', root / 'processed'
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ('wikipedia', 'mooc', 'reddit'):
        url = f'https://snap.stanford.edu/jodie/{name}.csv'
        path = raw / f'{name}.csv'
        if not path.exists():
            partial = path.with_suffix('.download')
            print(f'Downloading {url}', flush=True)
            urllib.request.urlretrieve(url, partial)
            partial.rename(path)
        frame = pd.read_csv(path)
        # JODIE columns: user, item, timestamp, state_label, interaction features.
        if frame.shape[1] < 5 or frame.empty:
            raise ValueError(f'Invalid JODIE input: {path}')
        users, user_values = pd.factorize(frame.iloc[:, 0], sort=True)
        items, item_values = pd.factorize(frame.iloc[:, 1], sort=True)
        times = frame.iloc[:, 2].to_numpy(dtype=np.float64)
        features = frame.iloc[:, 4:].to_numpy(dtype=np.float32)
        if not np.isfinite(times).all() or not np.isfinite(features).all():
            raise ValueError('Non-finite timestamps/features')
        order = np.argsort(times, kind='stable')
        output = pd.DataFrame(dict(u=users[order] + 1,
            i=items[order] + len(user_values) + 1,
            ts=times[order], idx=np.arange(1, len(frame) + 1),
            label=frame.iloc[:, 3].to_numpy()[order]))
        output.to_csv(processed / f'ml_{name}.csv', index=False)
        np.save(processed / f'ml_{name}.npy',
                np.concatenate([np.zeros((1, features.shape[1]), dtype=np.float32), features[order]]))
        np.save(processed / f'ml_{name}_node.npy',
                np.zeros((len(user_values) + len(item_values) + 1, 32), dtype=np.float32))
        record = dict(dataset=name, url=url, raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      events=len(frame), users=len(user_values), items=len(item_values),
                      edge_dim=features.shape[1], node_features='zero_32', bipartite=True)
        records.append(record)
        (root / 'manifest.json').write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
