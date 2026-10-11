"""Phase-9 on-policy PPO orchestration, safe checkpoints, and resume."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import torch

from holdem_agent_phase8 import CentralizedQCritic, NetworkConfig, TransformerActor
from .ppo import (PPOConfig, collect_self_play, evaluate_vs_random,
                  freeze_old_policy_data, optimize_ppo)


@dataclass
class TrainingSession:
    actor: TransformerActor
    critic: CentralizedQCritic
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    completed_updates: int
    network_config: NetworkConfig
    ppo_config: PPOConfig
    device: str


def create_session(*, device: str = 'cpu', seed: int = 42,
                   network_config: NetworkConfig | None = None,
                   ppo_config: PPOConfig | None = None) -> TrainingSession:
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu or install compatible CUDA torch')
    network_config = network_config or NetworkConfig()
    ppo_config = ppo_config or PPOConfig(seed=seed)
    # Phase 8 inference assumes deterministic masked logits for old log-probs.
    if network_config.dropout != 0:
        raise ValueError('PPO baseline requires dropout=0 for consistent behavior logprobs')
    torch.manual_seed(seed)
    actor = TransformerActor(network_config).to(device)
    critic = CentralizedQCritic(network_config).to(device)
    aopt = torch.optim.Adam(actor.parameters(), lr=ppo_config.learning_rate)
    copt = torch.optim.Adam(critic.parameters(), lr=ppo_config.learning_rate)
    return TrainingSession(actor, critic, aopt, copt, 0,
                           network_config, ppo_config, device)


def save_checkpoint(session: TrainingSession, path: str | Path) -> None:
    """Atomic local checkpoint; .tmp partial files are never loaded as final."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + '.tmp')
    payload = {
        'phase': 9, 'algorithm': 'ppo_mc_privileged_baseline',
        'completed_updates': session.completed_updates,
        'network_config': asdict(session.network_config),
        'ppo_config': asdict(session.ppo_config),
        'actor_state': session.actor.state_dict(),
        'critic_state': session.critic.state_dict(),
        'actor_optimizer': session.actor_optimizer.state_dict(),
        'critic_optimizer': session.critic_optimizer.state_dict(),
        'torch_rng_state': torch.get_rng_state(),
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str | Path, *, device: str = 'cpu') -> TrainingSession:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get('phase') != 9 or payload.get('algorithm') != 'ppo_mc_privileged_baseline':
        raise ValueError('this is not a Phase 9 PPO checkpoint')
    cfg = NetworkConfig(**payload['network_config'])
    hyperparams = PPOConfig(**payload['ppo_config'])
    session = create_session(device=device, seed=hyperparams.seed,
                             network_config=cfg, ppo_config=hyperparams)
    session.actor.load_state_dict(payload['actor_state'])
    session.critic.load_state_dict(payload['critic_state'])
    session.actor_optimizer.load_state_dict(payload['actor_optimizer'])
    session.critic_optimizer.load_state_dict(payload['critic_optimizer'])
    session.completed_updates = int(payload['completed_updates'])
    torch.set_rng_state(payload['torch_rng_state'].cpu())
    return session


def train_updates(session: TrainingSession, *, updates: int,
                  checkpoint: str | Path | None = None) -> list[dict]:
    """updates are ADDITIONAL iterations when resuming a saved session."""
    if type(updates) is not int or updates < 1:
        raise ValueError('updates must be positive integer')
    reports = []
    for _ in range(updates):
        tick = time.perf_counter()
        step = session.completed_updates
        rows = collect_self_play(session.actor, session.ppo_config,
                                 iteration=step, device=session.device)
        frozen = freeze_old_policy_data(
            session.actor, session.critic, rows,
            device=session.device, chunk_size=session.ppo_config.inference_batch_size)
        stats = optimize_ppo(
            session.actor, session.critic, session.actor_optimizer,
            session.critic_optimizer, frozen, session.ppo_config,
            device=session.device, shuffle_seed=session.ppo_config.seed + 100003 * step)
        session.completed_updates += 1
        elapsed = time.perf_counter() - tick
        stats.update({'update': session.completed_updates, 'hands': rows.hands,
                      'elapsed_seconds': round(elapsed, 3),
                      'hands_per_second_training': round(rows.hands / max(elapsed, 1e-9), 3)})
        reports.append(stats)
        if checkpoint is not None:
            save_checkpoint(session, checkpoint)
    return reports


def format_report(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True)
