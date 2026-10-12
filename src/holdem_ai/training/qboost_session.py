"""Phase-10 trainer, safe checkpoints and Phase-9 warm start.

A frozen Q estimate and old policy are used throughout each update's PPO epochs.
This prevents leakage from simultaneously changing critic targets into actor control
variates. Comparisons must use separate on-policy rollouts after actors diverge.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import PPOConfig, collect_self_play
from holdem_ai.training.ppo_session import create_session, load_checkpoint
from holdem_ai.training.ppo_session import TrainingSession
from holdem_ai.training.qboost import QBoostConfig, freeze_qboost_data, optimize_qboost


@dataclass
class QBoostSession:
    base: TrainingSession
    qconfig: QBoostConfig

    @property
    def completed_updates(self) -> int:
        return self.base.completed_updates


def create_qboost_session(*, device: str = 'cpu', seed: int = 42,
                          network_config: NetworkConfig | None = None,
                          ppo_config: PPOConfig | None = None,
                          qconfig: QBoostConfig | None = None) -> QBoostSession:
    return QBoostSession(create_session(device=device, seed=seed,
                         network_config=network_config, ppo_config=ppo_config),
                         qconfig or QBoostConfig())


def from_phase9_checkpoint(path: str | Path, *, device: str = 'cpu',
                           qconfig: QBoostConfig | None = None) -> QBoostSession:
    """Explicitly warm-start from user's trusted Phase9 checkpoint.

    Phase9's Adam and update counter are preserved; switching algorithms
    means the training sequence is not an exact uninterrupted PPO replay.
    """
    return QBoostSession(load_checkpoint(path, device=device), qconfig or QBoostConfig())


def save_qboost_checkpoint(session: QBoostSession, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    base = session.base
    payload = {
        'phase': 10,
        'algorithm': 'qboost_ppo_doubly_corrected_lite',
        'completed_updates': base.completed_updates,
        'network_config': asdict(base.network_config),
        'ppo_config': asdict(base.ppo_config),
        'qboost_config': asdict(session.qconfig),
        'actor_state': base.actor.state_dict(),
        'critic_state': base.critic.state_dict(),
        'actor_optimizer': base.actor_optimizer.state_dict(),
        'critic_optimizer': base.critic_optimizer.state_dict(),
        'torch_rng_state': torch.get_rng_state(),
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_qboost_checkpoint(path: str | Path, *, device: str = 'cpu') -> QBoostSession:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get('phase') != 10 or payload.get('algorithm') != 'qboost_ppo_doubly_corrected_lite':
        raise ValueError('this is not a Phase10 QBoost checkpoint')
    network_cfg = NetworkConfig(**payload['network_config'])
    ppo_cfg = PPOConfig(**payload['ppo_config'])
    qcfg = QBoostConfig(**payload['qboost_config'])
    session = create_qboost_session(device=device, seed=ppo_cfg.seed,
                                     network_config=network_cfg, ppo_config=ppo_cfg,
                                     qconfig=qcfg)
    base = session.base
    base.actor.load_state_dict(payload['actor_state'])
    base.critic.load_state_dict(payload['critic_state'])
    base.actor_optimizer.load_state_dict(payload['actor_optimizer'])
    base.critic_optimizer.load_state_dict(payload['critic_optimizer'])
    base.completed_updates = int(payload['completed_updates'])
    if base.completed_updates < 0:
        raise ValueError('negative checkpoint update counter')
    torch.set_rng_state(payload['torch_rng_state'].cpu())
    return session


def train_qboost_updates(session: QBoostSession, *, updates: int,
                         checkpoint: str | Path | None = None) -> list[dict]:
    if type(updates) is not int or updates < 1:
        raise ValueError('updates must be positive integer')
    base = session.base
    reports = []
    for _ in range(updates):
        begin = time.perf_counter()
        iteration = base.completed_updates
        rows = collect_self_play(base.actor, base.ppo_config,
                                 iteration=iteration, device=base.device)
        frozen = freeze_qboost_data(base.actor, base.critic, rows,
                                    device=base.device,
                                    chunk_size=base.ppo_config.inference_batch_size)
        stats = optimize_qboost(base.actor, base.critic,
                                base.actor_optimizer, base.critic_optimizer,
                                frozen, base.ppo_config, session.qconfig,
                                device=base.device,
                                shuffle_seed=base.ppo_config.seed + 100003 * iteration)
        base.completed_updates += 1
        elapsed = time.perf_counter() - begin
        stats.update({'update': base.completed_updates, 'hands': rows.hands,
                      'beta': session.qconfig.beta,
                      'elapsed_seconds': round(elapsed, 3),
                      'hands_per_second_training': round(rows.hands/max(elapsed, 1e-9), 3)})
        if not all(math.isfinite(float(x)) for x in stats.values()):
            raise FloatingPointError('nonfinite training metric')
        reports.append(stats)
        if checkpoint is not None:
            save_qboost_checkpoint(session, checkpoint)
    return reports
