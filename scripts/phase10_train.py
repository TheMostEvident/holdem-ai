"""Phase 10 experimental Q-Boost PPO; optional matched-start PPO comparator.

Execute at repository root: uv run --extra gpu python -m scripts.phase10_train ...
No optimizer checkpoints are committed to Git. All training uses on-policy data.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import PPOConfig, collect_self_play, evaluate_vs_random, freeze_old_policy_data, optimize_ppo
from holdem_ai.training.ppo_session import create_session, save_checkpoint
from holdem_ai.training.qboost import QBoostConfig
from holdem_ai.training.qboost_session import create_qboost_session, from_phase9_checkpoint, load_qboost_checkpoint, train_qboost_updates


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Phase10 doubly-corrected Q-Boosting PPO experiment')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--updates', type=int, default=2)
    p.add_argument('--hands-per-update', type=int, default=16)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--minibatch-size', type=int, default=32)
    p.add_argument('--eval-hands', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--checkpoint', default='checkpoints/phase10_qboost.pt')
    p.add_argument('--resume', default=None, help='Phase10 QBoost checkpoint, continues saved hyperparameters')
    p.add_argument('--from-phase9', default=None, help='warm-start from Phase9 PPO checkpoint')
    p.add_argument('--compare', action='store_true', help='train separate matched-start on-policy PPO control')
    p.add_argument('--ppo-checkpoint', default='checkpoints/phase10_ppo_control.pt')
    p.add_argument('--skip-eval', action='store_true')
    p.add_argument('--small-network', action='store_true')
    return p


def run(args: argparse.Namespace) -> dict:
    if args.resume and args.from_phase9:
        raise ValueError('--resume and --from-phase9 cannot both be set')
    if args.compare and (args.resume or args.from_phase9):
        raise ValueError('--compare requires a fresh matched-start run')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    if args.compare and Path(args.checkpoint).resolve() == Path(args.ppo_checkpoint).resolve():
        raise ValueError('PPO and QBoost checkpoints must have different paths')
    begin = time.perf_counter()
    if args.resume:
        session = load_qboost_checkpoint(args.resume, device=args.device)
        checkpoint_path = (args.resume if args.checkpoint == 'checkpoints/phase10_qboost.pt'
                           else args.checkpoint)
    elif args.from_phase9:
        session = from_phase9_checkpoint(args.from_phase9, device=args.device,
                                          qconfig=QBoostConfig(beta=args.beta))
        checkpoint_path = args.checkpoint
    else:
        net = (NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1,
                             ff_dim=64, hidden_dim=48)
               if args.small_network else NetworkConfig())
        cfg = PPOConfig(hands_per_update=args.hands_per_update,
                        epochs=args.epochs, minibatch_size=args.minibatch_size,
                        seed=args.seed)
        session = create_qboost_session(device=args.device, seed=args.seed,
                                       network_config=net, ppo_config=cfg,
                                       qconfig=QBoostConfig(beta=args.beta))
        checkpoint_path = args.checkpoint
    base = session.base
    eval_seed = 900000 + base.ppo_config.seed * 10
    def evaluate(actor):
        return evaluate_vs_random(actor, hands=args.eval_hands, device=base.device,
                                  seed=eval_seed, big_blind=base.ppo_config.big_blind)
    first = None if args.skip_eval else evaluate(base.actor)
    if first is not None:
        print(json.dumps({'kind': 'evaluation', 'agent': 'qboost', 'point': 'before', **first}), flush=True)
    # Compare: distinct on-policy actors, common initial weights and seed schedule.
    # After update 1 trajectories can diverge; do NOT train both on same stale data.
    control = None
    control_first = None
    control_reports = []
    if args.compare:
        control = create_session(device=base.device, seed=base.ppo_config.seed,
                                 network_config=base.network_config,
                                 ppo_config=base.ppo_config)
        # create_session reseeds torch but self-play samples use local Python RNG.
        control.actor.load_state_dict(base.actor.state_dict())
        control.critic.load_state_dict(base.critic.state_dict())
        control_first = None if args.skip_eval else evaluate(control.actor)
        if first is not None and control_first != first:
            raise AssertionError('comparison must start with identical evaluation')
    reports = []
    for _ in range(args.updates):
        # The comparator update uses its OWN rollout and frozen old policy.
        # Both use common seed schedule and stacks, not identical realized paths.
        reports.extend(train_qboost_updates(session, updates=1, checkpoint=checkpoint_path))
        print(json.dumps({'kind': 'update', 'agent': 'qboost', **reports[-1]}), flush=True)
        if control is not None:
            tic = time.perf_counter()
            it = control.completed_updates
            rows = collect_self_play(control.actor, control.ppo_config, iteration=it, device=control.device)
            frozen = freeze_old_policy_data(control.actor, control.critic, rows,
                                            device=control.device,
                                            chunk_size=control.ppo_config.inference_batch_size)
            stats = optimize_ppo(control.actor, control.critic, control.actor_optimizer,
                                 control.critic_optimizer, frozen, control.ppo_config,
                                 device=control.device,
                                 shuffle_seed=control.ppo_config.seed + 100003 * it)
            control.completed_updates += 1
            elapsed = time.perf_counter() - tic
            stats.update({'update': control.completed_updates, 'hands': rows.hands,
                          'elapsed_seconds': round(elapsed, 3),
                          'hands_per_second_training': round(rows.hands/max(elapsed, 1e-9), 3)})
            control_reports.append(stats)
            save_checkpoint(control, args.ppo_checkpoint)
            print(json.dumps({'kind': 'update', 'agent': 'ppo_control', **stats}), flush=True)
    last = None if args.skip_eval else evaluate(base.actor)
    control_last = None if control is None or args.skip_eval else evaluate(control.actor)
    for agent, report in (('qboost', last), ('ppo_control', control_last)):
        if report is not None:
            print(json.dumps({'kind': 'evaluation', 'agent': agent, 'point': 'after', **report}), flush=True)
    if base.device == 'cuda':
        torch.cuda.synchronize()
    return {
        'phase': 10, 'algorithm': 'qboost_ppo_doubly_corrected_lite',
        'paper_reproduction': False, 'trained_policy': True,
        'device': base.device, 'torch': torch.__version__,
        'completed_updates': base.completed_updates, 'beta': session.qconfig.beta,
        'checkpoint': str(checkpoint_path),
        'actor_parameters': sum(p.numel() for p in base.actor.parameters()),
        'critic_parameters': sum(p.numel() for p in base.critic.parameters()),
        'last_update': reports[-1] if reports else None,
        'evaluation_before': first, 'evaluation_after': last,
        'ppo_control_last_update': control_reports[-1] if control_reports else None,
        'ppo_control_evaluation_before': control_first,
        'ppo_control_evaluation_after': control_last,
        'elapsed_seconds': round(time.perf_counter() - begin, 3),
        'cuda_peak_allocated_mb': round(torch.cuda.max_memory_allocated()/2**20, 2)
        if base.device == 'cuda' else None,
    }


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
