"""Measure ACTUAL training update throughput for each hands/update + env batch pair.

These are full PPO/QBoost gradient-update measurements, unlike pure forward
benchmark from Phase11. Every case uses an independent run and a fresh or
identically warm-started checkpoint. The first update includes warmup.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from holdem_ai.experiments.batch_scaling import BatchExperimentConfig, run_batch_experiment


def parse_sizes(value: str) -> list[int]:
    try:
        numbers = [int(s.strip()) for s in value.split(',')]
    except ValueError as e:
        raise ValueError('comma-separated positive integers required') from e
    if not numbers or any(n < 1 for n in numbers) or len(numbers) != len(set(numbers)):
        raise ValueError('batch sizes must be unique positive integers')
    return numbers


def build_parser():
    parser = argparse.ArgumentParser(description='Phase13 real training throughput benchmark')
    parser.add_argument('--algorithm', choices=('ppo', 'qboost'), default='qboost')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--hands-per-update', default='16,64,128')
    parser.add_argument('--env-batch-sizes', default='16,32,64')
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--minibatch-size', type=int, default=64)
    parser.add_argument('--duration-minutes', type=float, default=5.0)
    parser.add_argument('--from-checkpoint', default=None)
    parser.add_argument('--root', default='runs/phase13_bench')
    parser.add_argument('--small-network', action='store_true')
    return parser


def run(args):
    if args.updates < 1:
        raise ValueError('updates must be >=1')
    h_sizes = parse_sizes(args.hands_per_update)
    env_sizes = parse_sizes(args.env_batch_sizes)
    root = Path(args.root)
    table = []
    for h in h_sizes:
        for env in env_sizes:
            if env > h:
                continue   # tests an effective, not merely nominal, env batch
            case = root / f'h{h}_b{env}'
            cfg = BatchExperimentConfig(
                algorithm=args.algorithm, device=args.device,
                duration_seconds=args.duration_minutes * 60,
                max_updates=args.updates, hands_per_update=h,
                minibatch_size=args.minibatch_size, epochs=args.epochs,
                env_batch_size=env, skip_eval=True,
                small_network=args.small_network,
                from_checkpoint=args.from_checkpoint,
                checkpoint=str(case / 'model.pt'), log_dir=str(case / 'logs'))
            print(json.dumps({'kind': 'case_start', 'hands_per_update': h,
                              'env_batch_size': env}), flush=True)
            result = run_batch_experiment(cfg)
            update_rows = [json.loads(line) for line in
                           (case / 'logs' / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            records = [x for x in update_rows if x['kind'] == 'update']
            # Drop update 1 as warmup if available.
            measured = records[1:] if len(records) > 1 else records
            total_hands = sum(x['hands'] for x in measured)
            elapsed = sum(x['elapsed_seconds'] for x in measured)
            rate = total_hands / elapsed if elapsed > 0 else 0.0
            item = {'hands_per_update': h, 'env_batch_size': env,
                    'completed_updates': len(records), 'measured_updates': len(measured),
                    'full_training_hands_per_second': round(rate, 3),
                    'cuda_peak_allocated_mb': result['cuda_peak_allocated_mb'],
                    'checkpoint': result['checkpoint']}
            table.append(item)
            print(json.dumps({'kind': 'case_end', **item}), flush=True)
    return {'phase': 13, 'algorithm': args.algorithm,
            'benchmark': 'real_training_including_optimizer',
            'results': table}


def main():
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
