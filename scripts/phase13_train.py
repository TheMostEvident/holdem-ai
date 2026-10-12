"""Phase 13: PPO/QBoost batch scaling, checkpoint lineage, live tqdm ETA.

From repo root:
  uv run --extra gpu python -m scripts.phase13_train --device cuda --algorithm qboost --from-checkpoint checkpoints/phase11_pilot.pt --hands-per-update 64 --env-batch-size 32 --max-updates 2
"""
from __future__ import annotations

import argparse
import json

from holdem_ai.utils.progress import TrainingProgress
from holdem_ai.experiments.batch_scaling import BatchExperimentConfig, run_batch_experiment


class Phase13Progress(TrainingProgress):
    def _open_display(self) -> None:
        super()._open_display()
        if self._bar is not None:
            self._bar.set_description('Phase 13')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Phase13 HUNL PPO/QBoost training batch scaling')
    p.add_argument('--algorithm', choices=('ppo', 'qboost'), default='qboost')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--duration-minutes', type=float, default=15.0)
    p.add_argument('--max-updates', type=int, default=200, help='ADDITIONAL updates')
    p.add_argument('--hands-per-update', type=int, default=None, help='default 64 for a NEW phase13 stage')
    p.add_argument('--minibatch-size', type=int, default=None, help='default 64 for a NEW phase13 stage')
    p.add_argument('--epochs', type=int, default=None, help='default 2 for a NEW phase13 stage')
    p.add_argument('--rollout-seed', type=int, default=None, help='optional new self-play seed namespace')
    p.add_argument('--env-batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--eval-groups', type=int, default=40, help='Phase12 four-layout seed groups')
    p.add_argument('--eval-mode', choices=('sampled', 'greedy'), default='sampled')
    p.add_argument('--skip-eval', action='store_true')
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--from-checkpoint', default=None, help='warm start a NEW experiment; source never overwritten')
    p.add_argument('--resume', default=None, help='continue Phase13 run in SAME checkpoint/log-dir')
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--log-dir', default=None)
    p.add_argument('--small-network', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    p.add_argument('--json-events', action='store_true')
    p.add_argument('--progress-interval', type=float, default=1.0)
    return p


def run(args: argparse.Namespace) -> dict:
    checkpoint = args.checkpoint or args.resume or f'checkpoints/phase13_{args.algorithm}.pt'
    logs = args.log_dir or f'runs/phase13_{args.algorithm}'
    cfg = BatchExperimentConfig(
        algorithm=args.algorithm, device=args.device,
        duration_seconds=args.duration_minutes * 60,
        max_updates=args.max_updates,
        hands_per_update=args.hands_per_update,
        minibatch_size=args.minibatch_size, epochs=args.epochs,
        rollout_seed=args.rollout_seed,
        env_batch_size=args.env_batch_size, eval_every=args.eval_every,
        eval_groups=args.eval_groups, eval_mode=args.eval_mode,
        skip_eval=args.skip_eval, beta=args.beta,
        checkpoint=checkpoint, log_dir=logs,
        from_checkpoint=args.from_checkpoint, resume=args.resume,
        small_network=args.small_network)
    if args.no_progress:
        return run_batch_experiment(cfg,
            emit=lambda evt: print(json.dumps(evt, ensure_ascii=False), flush=True))
    with Phase13Progress(duration_seconds=cfg.duration_seconds,
                         max_updates=cfg.max_updates,
                         interval=args.progress_interval) as progress:
        def on_event(evt: dict):
            progress.on_event(evt)
            if args.json_events or evt.get('kind') in ('start', 'resume', 'evaluation', 'interrupt', 'end'):
                # Keep detailed update/action histograms in metrics.jsonl.
                progress.message(json.dumps(evt, ensure_ascii=False))
        return run_batch_experiment(cfg, emit=on_event)


def main():
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
