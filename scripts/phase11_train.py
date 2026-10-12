"""Phase 11 training CLI with real-time progress, time-left and ETA.

The JSONL log and per-update atomic checkpoints remain owned by the original
Phase-11 runner. The progress display only observes existing runner events.

Example (from project root):
  uv run --extra gpu python -m scripts.phase11_train --device cuda --duration-minutes 15 --max-updates 200

Optional tqdm visuals:
  uv add tqdm
Without tqdm, a dependency-free terminal bar is used automatically.
"""
from __future__ import annotations

import argparse
import json

from holdem_ai.experiments.time_budget import ExperimentConfig, run_experiment
from holdem_ai.utils.progress import TrainingProgress


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Phase11 batched HUNL200 training with live ETA and checkpoints')
    p.add_argument('--algorithm', choices=('ppo', 'qboost'), default='qboost')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--duration-minutes', type=float, default=300.0,
                   help='wall-clock budget for THIS invocation (default five hours)')
    p.add_argument('--max-updates', type=int, default=None,
                   help='limit ADDITIONAL updates in this invocation')
    p.add_argument('--env-batch-size', type=int, default=8,
                   help='simultaneously active hands for one actor forward pass')
    p.add_argument('--hands-per-update', type=int, default=32)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--minibatch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--eval-hands', type=int, default=80)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--log-dir', default=None)
    p.add_argument('--resume', default=None)
    p.add_argument('--from-phase9', default=None)
    p.add_argument('--from-phase10', default=None)
    p.add_argument('--small-network', action='store_true')
    p.add_argument('--no-progress', action='store_true',
                   help='disable live display and print all legacy JSON events')
    p.add_argument('--json-events', action='store_true',
                   help='print every update JSON in addition to progress (normally only saved in metrics.jsonl)')
    p.add_argument('--progress-interval', type=float, default=1.0,
                   help='live display refresh interval in seconds (>=0.1)')
    return p


def run(args: argparse.Namespace) -> dict:
    checkpoint = args.checkpoint or (args.resume if args.resume is not None
                                     else f'checkpoints/phase11_{args.algorithm}.pt')
    logs = args.log_dir or f'runs/phase11_{args.algorithm}'
    cfg = ExperimentConfig(
        algorithm=args.algorithm, device=args.device,
        duration_seconds=args.duration_minutes * 60,
        max_updates=args.max_updates, env_batch_size=args.env_batch_size,
        eval_every=args.eval_every, eval_hands=args.eval_hands,
        hands_per_update=args.hands_per_update, epochs=args.epochs,
        minibatch_size=args.minibatch_size, seed=args.seed,
        beta=args.beta, checkpoint=checkpoint, log_dir=logs,
        resume=args.resume, from_phase9=args.from_phase9,
        from_phase10=args.from_phase10, small_network=args.small_network,
    )
    if args.no_progress:
        return run_experiment(cfg, emit=lambda d: print(json.dumps(d, ensure_ascii=False), flush=True))

    with TrainingProgress(duration_seconds=cfg.duration_seconds,
                          max_updates=cfg.max_updates,
                          interval=args.progress_interval) as progress:
        def emit(event: dict) -> None:
            progress.on_event(event)
            # Full per-step data continues to be written in metrics.jsonl.
            # Print evaluations and final information, rather than a flood of
            # JSON lines that would obscure the live progress meter.
            if args.json_events or event.get('kind') in ('start', 'resume', 'evaluation', 'interrupt', 'end'):
                progress.message(json.dumps(event, ensure_ascii=False))

        return run_experiment(cfg, emit=emit)


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
