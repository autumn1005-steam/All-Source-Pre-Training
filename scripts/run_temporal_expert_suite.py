"""Frozen development suite: two targets, three variants, three seeds each."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--revision', choices=('v1', 'v2', 'v3'), default='v1')
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    variants = (('aligned', 'paired', 'experts') if args.revision == 'v1'
                else (('ranked', 'identity_ap', 'identity_ranked', 'source_ranked')
                      if args.revision == 'v2' else ('ranked', 'source_ranked')))
    for variant in variants:
        for target in ('wikipedia', 'reddit'):
            output = root / variant / target
            output.mkdir(parents=True, exist_ok=True)
            print(f'START variant={variant} target={target}', flush=True)
            with (output / 'run.log').open('w') as log:
                subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('run_three_seed_check.py')),
                    '--processed-dir', args.processed_dir, '--output-root', str(output),
                    '--target', target, '--variant', variant], stdout=log, stderr=subprocess.STDOUT, check=True)
            summaries.append(json.loads((output / 'three_seed_summary.json').read_text()))
            (root / 'suite_summary.json').write_text(json.dumps(summaries, indent=2))
            print(f'FINISH variant={variant} target={target}', flush=True)
    print('SUITE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
