"""CLI: Phase-9 actual PPO updates, checkpoint/resume, random baseline evaluation.

From repo root:
  uv run --extra gpu python -m scripts.phase9_train --device cuda --updates 2 --hands-per-update 16 --eval-hands 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from holdem_agent_phase8 import NetworkConfig
from holdem_agent_phase9 import (PPOConfig, create_session, evaluate_vs_random,
                                 load_checkpoint, train_updates)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Phase 9 PPO baseline on HUNL200; NOT VRPO')
    p.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    p.add_argument('--updates', type=int, default=2, help='additional optimizer updates')
    p.add_argument('--hands-per-update', type=int, default=16)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--minibatch-size', type=int, default=32)
    p.add_argument('--eval-hands', type=int, default=20, help='even hand count >=2; same seed pre/post')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', default='checkpoints/phase9_ppo.pt')
    p.add_argument('--resume', default=None, help='checkpoint path; ignores fresh-run training hyperparameters')
    p.add_argument('--small-network', action='store_true', help='1-layer tiny test network instead of GTX1070 default')
    p.add_argument('--skip-eval', action='store_true')
    return p


def run(args: argparse.Namespace) -> dict:
    begin = time.perf_counter()
    if args.resume:
        session = load_checkpoint(args.resume, device=args.device)
    else:
        net = (NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1,
                             ff_dim=64, hidden_dim=48, max_tokens=256)
               if args.small_network else NetworkConfig())
        cfg = PPOConfig(hands_per_update=args.hands_per_update,
                        epochs=args.epochs, minibatch_size=args.minibatch_size,
                        seed=args.seed)
        session = create_session(device=args.device, seed=args.seed,
                                 network_config=net, ppo_config=cfg)
    eval_seed = 900000 + session.ppo_config.seed * 10
    before = None if args.skip_eval else evaluate_vs_random(
        session.actor, hands=args.eval_hands, device=session.device,
        seed=eval_seed, big_blind=session.ppo_config.big_blind)
    if before is not None:
        print(json.dumps({'evaluation': 'before', **before}, ensure_ascii=False), flush=True)
    checkpoint_path = (args.resume if args.resume and args.checkpoint == 'checkpoints/phase9_ppo.pt'
                       else args.checkpoint)
    update_reports = train_updates(session, updates=args.updates,
                                   checkpoint=checkpoint_path)
    for record in update_reports:
        print(json.dumps({'kind': 'update', **record}, ensure_ascii=False), flush=True)
    after = None if args.skip_eval else evaluate_vs_random(
        session.actor, hands=args.eval_hands, device=session.device,
        seed=eval_seed, big_blind=session.ppo_config.big_blind)
    if after is not None:
        print(json.dumps({'evaluation': 'after', **after}, ensure_ascii=False), flush=True)
    return {
        'phase': 9, 'algorithm': 'PPO_baseline_MC_returns',
        'trained_policy': True, 'device': session.device, 'torch': torch.__version__,
        'completed_updates': session.completed_updates,
        'checkpoint': str(Path(checkpoint_path)),
        'actor_parameters': sum(p.numel() for p in session.actor.parameters()),
        'critic_parameters': sum(p.numel() for p in session.critic.parameters()),
        'last_update': update_reports[-1],
        'evaluation_before': before, 'evaluation_after': after,
        'elapsed_seconds': round(time.perf_counter() - begin, 3),
        'cuda_peak_allocated_mb': round(torch.cuda.max_memory_allocated() / 2**20, 2)
        if session.device == 'cuda' else None,
    }


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
