"""Validated, resumable HTTP range downloads from the official JODIE host."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import time
import urllib.request

BASE = 'https://snap.stanford.edu/jodie/'
BLOCK = 1024 * 1024


def download(root, name, workers):
    root.mkdir(parents=True, exist_ok=True)
    url = BASE + name + '.csv'
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as response:
        size = int(response.headers['Content-Length'])
    destination = root / (name + '.csv')
    if destination.exists():
        if destination.stat().st_size != size:
            raise ValueError(f'Existing final file has wrong size: {destination}')
        print(f'{name}: complete, {size} bytes', flush=True)
        return
    parts = root / (name + '.parts')
    parts.mkdir(exist_ok=True)
    count = (size + BLOCK - 1) // BLOCK
    # Reuse complete blocks from the stopped single-connection download.
    prefix = root / (name + '.download')
    if prefix.exists():
        with prefix.open('rb') as handle:
            for index in range(min(prefix.stat().st_size // BLOCK, count)):
                data = handle.read(BLOCK)
                target = parts / str(index)
                if not target.exists():
                    target.write_bytes(data)

    def fetch(index):
        start = index * BLOCK
        stop = min(size, start + BLOCK) - 1
        expected = stop - start + 1
        target = parts / str(index)
        if target.exists() and target.stat().st_size == expected:
            return expected
        last_error = None
        for attempt in range(6):
            try:
                request = urllib.request.Request(url, headers={
                    'Range': f'bytes={start}-{stop}', 'Accept-Encoding': 'identity'})
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206:
                        raise ValueError('Server did not honor byte range')
                    if response.headers.get('Content-Range') != f'bytes {start}-{stop}/{size}':
                        raise ValueError('Unexpected Content-Range')
                    data = response.read(expected + 1)
                if len(data) != expected:
                    raise ValueError('Incomplete range')
                partial = parts / (str(index) + '.tmp')
                partial.write_bytes(data)
                partial.replace(target)
                return expected
            except Exception as exc:
                last_error = exc
                time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f'{name} block {index} failed: {last_error}')

    done = 0
    started = time.monotonic()
    initial = sum(p.stat().st_size for p in parts.iterdir() if p.name.isdigit())
    print(f'{name}: {workers} connections, reuse={initial}, total={size}', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, index) for index in range(count)]
        for future in as_completed(futures):
            done += future.result()
            speed = max(0, done - initial) / max(time.monotonic() - started, 0.01)
            print(f'{name}: {done}/{size} ({100*done/size:.1f}%) average={speed/1024:.1f} KiB/s', flush=True)
    assembled = root / (name + '.assembled')
    with assembled.open('wb') as output:
        for index in range(count):
            with (parts / str(index)).open('rb') as source:
                shutil.copyfileobj(source, output)
    if assembled.stat().st_size != size:
        raise ValueError('Assembled size mismatch')
    assembled.rename(destination)
    print(f'{name}: download complete, exact size verified', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError('workers must be between 1 and 16')
    for name in ('wikipedia', 'mooc', 'reddit'):
        download(Path(args.root) / 'raw', name, args.workers)


if __name__ == '__main__':
    main()
