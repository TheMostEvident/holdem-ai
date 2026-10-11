"""Multi-seed controlled PPO/QBoost HUNL experiment.

Example from repo root:
  uv run --extra gpu python -m scripts.phase14_study --device cuda --source-checkpoint checkpoints/phase9_ppo.pt --seeds 41,42,43 --updates 100
"""
from __future__ import annotations

import argparse
import json

from holdem_agent_phase14 import ComparisonConfig, run_study


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(x.strip()) for x in value.split(','))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError('seeds must be comma-separated integers') from exc
    if not result or len(set(result)) != len(result) or any(not 0 <= s < 100_000 for s in result):
        raise argparse.ArgumentTypeError('seeds must be unique integers in [0,100000)')
    return result


def build_parser():
    p = argparse.ArgumentParser(description='Phase14 preregistered PPO vs QBoost controlled multi-seed study')
    p.add_argument('--source-checkpoint', required=True, help='trusted FULL Phase9 checkpoint, common start')
    p.add_argument('--root', default='runs/phase14_study')
    p.add_argument('--seeds', type=parse_seeds, default=(41, 42, 43))
    p.add_argument('--updates', type=int, default=100, help='equal number of updates per arm')
    p.add_argument('--hands-per-update', type=int, default=64)
    p.add_argument('--env-batch-size', type=int, default=32)
    p.add_argument('--minibatch-size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--validation-groups', type=int, default=40)
    p.add_argument('--test-groups', type=int, default=200)
    p.add_argument('--mode', choices=('sampled','greedy'), default='sampled')
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--arm-duration-minutes', type=float, default=30.0)
    p.add_argument('--validation-seed', type=int, default=12_000_000)
    p.add_argument('--test-seed', type=int, default=22_000_000)
    p.add_argument('--entropy-warning-threshold', type=float, default=0.6)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    return p


def run(args):
    cfg = ComparisonConfig(
        source_checkpoint=args.source_checkpoint, root=args.root,
        seeds=args.seeds, updates=args.updates,
        hands_per_update=args.hands_per_update, env_batch_size=args.env_batch_size,
        minibatch_size=args.minibatch_size, epochs=args.epochs,
        eval_every=args.eval_every, validation_groups=args.validation_groups,
        test_groups=args.test_groups, eval_mode=args.mode, device=args.device,
        beta=args.beta, arm_duration_minutes=args.arm_duration_minutes,
        validation_seed=args.validation_seed, test_seed=args.test_seed,
        entropy_warning_threshold=args.entropy_warning_threshold,
        resume=args.resume, progress=not args.no_progress,
    )
    def emit(item):
        if item.get('kind') in ('validation', 'arm_end'):
            print(json.dumps(item, ensure_ascii=False), flush=True)
    result = run_study(cfg, emit=emit)
    short = {k: v for k, v in result.items() if k not in ('seed_results',)}
    print(json.dumps(short, indent=2, ensure_ascii=False), flush=True)
    return result


def main():
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()
