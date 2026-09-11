"""Paired three-seed diagnostic on the recovered processed dataset version."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from forward_pair.pipeline import run_pipeline
from tempofound.config import ExperimentConfig
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--target', default='wikipedia', choices=('wikipedia', 'reddit', 'mooc'))
    parser.add_argument('--variant', default='legacy', choices=('legacy', 'aligned', 'paired', 'experts', 'ranked', 'identity_ap', 'identity_ranked', 'source_ranked', 'rcsr', 'strict_ranked', 'strict_source_ranked', 'rcsr_no_regret', 'rcsr_event', 'hrcsr'))
    args = parser.parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    strict_variants = {'strict_ranked', 'strict_source_ranked', 'rcsr_no_regret', 'rcsr_event', 'hrcsr'}
    source_expert_variants = {'source_ranked', 'strict_source_ranked', 'rcsr', 'rcsr_no_regret', 'rcsr_event', 'hrcsr'}
    routed_variants = {'rcsr', 'rcsr_no_regret', 'rcsr_event', 'hrcsr'}
    ranking_variants = {'ranked', 'identity_ranked', 'source_ranked', 'rcsr', *strict_variants}
    config = replace(ExperimentConfig.from_yaml(ROOT / 'configs/rasp_real_initial.yaml'),
                     name=f'recovered_data_{args.target}_three_seed_batch_matched',
                     targets=(args.target,), seeds=(0, 1, 42),
                     consistent_candidate_weights=args.variant != 'legacy',
                     temporal_paired_gate=args.variant in ('paired', 'experts', 'ranked', 'identity_ap', 'identity_ranked', 'source_ranked', 'rcsr', *strict_variants),
                     temporal_experts=4 if args.variant in ('experts', 'identity_ap', 'identity_ranked') else 0,
                     expert_identity_init=args.variant in ('identity_ap', 'identity_ranked', *source_expert_variants),
                     proxy_metric='mrr' if args.variant in ranking_variants else 'ap',
                     source_experts=args.variant in source_expert_variants,
                     risk_controlled_routing=args.variant in routed_variants,
                     proxy_negative_policy='historical' if args.variant in strict_variants or args.variant == 'rcsr' else 'random',
                     negative_policy='historical' if args.variant in strict_variants or args.variant == 'rcsr' else 'random',
                     transfer_regret_weight=0.25 if args.variant in {'rcsr', 'rcsr_event', 'hrcsr'} else 0.0,
                     transfer_regret_mode='event_pairwise' if args.variant in {'rcsr_event', 'hrcsr'} else 'batch_bce',
                     target_gate_mode='lcb_soft' if args.variant == 'hrcsr' else 'none',
                     target_gate_windows=5,
                     target_gate_bootstrap_replicates=1000,
                     target_gate_confidence=0.95,
                     target_gate_min_gain=0.0,
                     target_gate_temperature=0.01,
                     total_batches_per_epoch=32)
    results = []
    for seed in config.seeds:
        started = time.monotonic()
        print(f'START seed={seed}', flush=True)
        result = run_pipeline(config, target=args.target, seed=seed,
                              processed_dir=args.processed_dir, output_root=output, device='cuda:0')
        diagnostics = {}
        for label in ('rasp', 'all_sources'):
            path = output / args.target / f'seed{seed}' / 'final' / label / 'pretrained.pth'
            training = torch.load(path, map_location='cpu', weights_only=False)['metadata']['pretraining']
            assert training['epoch_batch_counts'] == [32] * config.pretrain_epochs, training
            diagnostics[label] = training
        result['training_diagnostics'] = diagnostics
        result['elapsed_seconds'] = time.monotonic() - started
        results.append(result)
        summary_dir = output / args.target
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / 'paired_results.json').write_text(json.dumps(results, indent=2))
        print(f'FINISH seed={seed} delta={result["test_mrr_delta"]:.6f}', flush=True)
    deltas = [row['test_mrr_delta'] for row in results]
    summary = dict(target=args.target, variant=args.variant, seeds=list(config.seeds),
                   dataset_scope='recovered processed data; official full-data provenance unverified',
                   budget_scope='equal consumed minibatches, not equal grouped optimizer steps or FLOPs',
                   rasp_mean=statistics.mean(row['rasp_test_mrr'] for row in results),
                   all_sources_mean=statistics.mean(row['all_sources_test_mrr'] for row in results),
                   paired_delta_mean=statistics.mean(deltas),
                   paired_delta_sample_std=statistics.stdev(deltas),
                   positive_seeds=sum(delta > 0 for delta in deltas),
                   decision=('consistent positive preliminary signal' if min(deltas) > 0
                             else 'not consistently effective across three seeds'))
    summary_dir = output / args.target
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / 'three_seed_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
