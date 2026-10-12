"""Safe wall-clock bounded Phase11 training with crash-recoverable observability.

All optimizer updates are on-policy; rollout is batched *only across current
observations*, never reused once the actor changes. Single-process GPU batching
is not multiprocessing or a distributed replay buffer.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from holdem_agent_phase8 import NetworkConfig, TransformerActor
from holdem_agent_phase9 import (PPOConfig, create_session, freeze_old_policy_data,
                                 load_checkpoint, optimize_ppo, save_checkpoint)
from holdem_agent_phase10 import (QBoostConfig, create_qboost_session,
                                  freeze_qboost_data, from_phase9_checkpoint,
                                  load_qboost_checkpoint, optimize_qboost,
                                  save_qboost_checkpoint)
from .collector import collect_batched_self_play
from .evaluation import evaluate_balanced


@dataclass(frozen=True)
class ExperimentConfig:
    algorithm: str = 'qboost'
    device: str = 'cpu'
    duration_seconds: float = 5 * 3600
    max_updates: int | None = None
    env_batch_size: int = 8
    eval_every: int = 25
    eval_hands: int = 80
    hands_per_update: int = 32
    epochs: int = 2
    minibatch_size: int = 32
    seed: int = 42
    beta: float = 1.0
    checkpoint: str = 'checkpoints/phase11_qboost.pt'
    log_dir: str = 'runs/phase11_qboost'
    resume: str | None = None
    from_phase9: str | None = None
    from_phase10: str | None = None
    small_network: bool = False
    max_decisions: int = 200

    def __post_init__(self) -> None:
        if self.algorithm not in ('ppo', 'qboost'):
            raise ValueError('algorithm must be ppo or qboost')
        if self.device not in ('cpu', 'cuda'):
            raise ValueError('device must be cpu or cuda')
        if isinstance(self.duration_seconds, bool) or not isinstance(self.duration_seconds, (float, int)) or not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError('duration_seconds must be finite and positive')
        for field in ('env_batch_size', 'eval_every', 'hands_per_update', 'epochs',
                      'minibatch_size', 'max_decisions', 'eval_hands'):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError(f'{field} must be a positive integer')
        if self.eval_hands < 4 or self.eval_hands % 4:
            raise ValueError('eval_hands must be a multiple of four and >= 4')
        if self.max_updates is not None and (type(self.max_updates) is not int or self.max_updates < 1):
            raise ValueError('max_updates must be None or positive integer')
        if type(self.seed) is not int:
            raise ValueError('seed must be int')
        if not isinstance(self.beta, (float, int)) or isinstance(self.beta, bool) or not math.isfinite(self.beta) or not 0 <= self.beta <= 1:
            raise ValueError('beta must be in [0,1]')
        if sum(bool(x) for x in (self.resume, self.from_phase9, self.from_phase10)) > 1:
            raise ValueError('--resume, --from-phase9, --from-phase10 are mutually exclusive')
        if self.from_phase10 and self.algorithm != 'qboost':
            raise ValueError('--from-phase10 works only for qboost')
        if not self.checkpoint or not self.log_dir:
            raise ValueError('checkpoint and log directory are required')


def _write_atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _log(log_path: Path, record: dict, emit: Callable[[dict], None] | None = None) -> None:
    with log_path.open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    if emit is not None:
        emit(record)


def _session(config: ExperimentConfig):
    if config.resume:
        return (load_checkpoint(config.resume, device=config.device) if config.algorithm == 'ppo'
                else load_qboost_checkpoint(config.resume, device=config.device))
    if config.from_phase10:
        loaded = load_qboost_checkpoint(config.from_phase10, device=config.device)
        if not math.isclose(loaded.qconfig.beta, config.beta):
            raise ValueError('from-phase10 beta must match the checkpoint beta')
        return loaded
    if config.from_phase9:
        if config.algorithm == 'ppo':
            return load_checkpoint(config.from_phase9, device=config.device)
        return from_phase9_checkpoint(config.from_phase9, device=config.device,
                                      qconfig=QBoostConfig(beta=config.beta))
    network = (NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1,
                             ff_dim=64, hidden_dim=48, max_tokens=256)
               if config.small_network else NetworkConfig())
    ppo = PPOConfig(hands_per_update=config.hands_per_update,
                    epochs=config.epochs, minibatch_size=config.minibatch_size,
                    max_decisions=config.max_decisions, seed=config.seed)
    if config.algorithm == 'ppo':
        return create_session(device=config.device, seed=config.seed,
                              network_config=network, ppo_config=ppo)
    return create_qboost_session(device=config.device, seed=config.seed,
                                 network_config=network, ppo_config=ppo,
                                 qconfig=QBoostConfig(beta=config.beta))


def _base(session, algorithm: str):
    return session if algorithm == 'ppo' else session.base


def _checkpoint_save(session, algorithm: str, path: Path) -> None:
    if algorithm == 'ppo':
        save_checkpoint(session, path)
    else:
        save_qboost_checkpoint(session, path)


def _step(session, config: ExperimentConfig) -> dict:
    base = _base(session, config.algorithm)
    start = time.perf_counter()
    iteration = base.completed_updates
    rows = collect_batched_self_play(
        base.actor, base.ppo_config, iteration=iteration, device=base.device,
        env_batch_size=config.env_batch_size,
    )
    if config.algorithm == 'ppo':
        frozen = freeze_old_policy_data(base.actor, base.critic, rows,
                                        device=base.device,
                                        chunk_size=base.ppo_config.inference_batch_size)
        metrics = optimize_ppo(
            base.actor, base.critic, base.actor_optimizer,
            base.critic_optimizer, frozen, base.ppo_config,
            device=base.device,
            shuffle_seed=base.ppo_config.seed + 100003 * iteration,
        )
    else:
        frozen = freeze_qboost_data(base.actor, base.critic, rows,
                                    device=base.device,
                                    chunk_size=base.ppo_config.inference_batch_size)
        metrics = optimize_qboost(
            base.actor, base.critic, base.actor_optimizer,
            base.critic_optimizer, frozen, base.ppo_config, session.qconfig,
            device=base.device,
            shuffle_seed=base.ppo_config.seed + 100003 * iteration,
        )
        metrics['beta'] = session.qconfig.beta
    if base.device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    base.completed_updates += 1
    return {
        'kind': 'update', 'algorithm': config.algorithm,
        'update': base.completed_updates, 'hands': rows.hands,
        'elapsed_seconds': round(elapsed, 4),
        'hands_per_second_training': round(rows.hands / max(elapsed, 1e-9), 3),
        **metrics,
    }


def _load_or_create_anchor(base, *, path: Path, resume: bool, device: str):
    anchor = TransformerActor(base.network_config).to(device)
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f'frozen anchor missing: {path}')
        payload = torch.load(path, weights_only=True, map_location='cpu')
        if payload.get('network_config') != asdict(base.network_config):
            raise ValueError('anchor architecture mismatch')
        anchor.load_state_dict(payload['actor_state'])
    else:
        anchor.load_state_dict(base.actor.state_dict())
        tmp = path.with_name(path.name + '.tmp')
        torch.save({'network_config': asdict(base.network_config),
                    'actor_state': {k: v.cpu().clone() for k, v in base.actor.state_dict().items()}}, tmp)
        os.replace(tmp, path)
    anchor.eval().requires_grad_(False)
    return anchor


def _budget_remaining(deadline: float, last_update_seconds: float | None = None) -> bool:
    margin = 0.0 if last_update_seconds is None else max(0.02, 1.1 * last_update_seconds)
    return time.perf_counter() + margin < deadline


def run_experiment(config: ExperimentConfig, *,
                   emit: Callable[[dict], None] | None = None) -> dict:
    """Run until time budget or maximum extra updates; checkpoint after every update.

    Uses an *immutable frozen anchor* for all evaluations across resume. A run
    directory is exclusive: fresh runs never overwrite existing run metadata.
    Time limits apply at update/evaluation boundaries, not inside a poker hand.
    """
    if config.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable on this machine')
    checkpoint = Path(config.checkpoint)
    root = Path(config.log_dir)
    meta_path, anchor_path = root / 'run.json', root / 'anchor.pt'
    events_path, summary_path = root / 'metrics.jsonl', root / 'summary.json'
    resuming = config.resume is not None
    if resuming:
        if not meta_path.is_file() or not events_path.is_file():
            raise FileNotFoundError('resume requires existing --log-dir/run.json and metrics.jsonl')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('algorithm') != config.algorithm:
            raise ValueError('resume algorithm does not match log directory')
        if Path(meta['checkpoint']).resolve() != Path(config.resume).resolve() or checkpoint.resolve() != Path(config.resume).resolve():
            raise ValueError('resume must use the SAME checkpoint path in --checkpoint and --resume')
    else:
        if (root.exists() and any(root.iterdir())) or checkpoint.exists():
            raise FileExistsError('fresh run would overwrite log/checkpoint; choose unique paths or --resume')
        root.mkdir(parents=True, exist_ok=True)
    session = _session(config)
    base = _base(session, config.algorithm)
    if config.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    if resuming:
        if (asdict(base.network_config) != meta['network_config']
                or asdict(base.ppo_config) != meta['ppo_config']):
            raise ValueError('resume checkpoint architecture/config mismatch')
        if config.algorithm == 'qboost' and not math.isclose(session.qconfig.beta, meta['beta']):
            raise ValueError('resume beta differs from original run')
        if base.completed_updates < meta['initial_updates']:
            raise ValueError('checkpoint update count older than run start')
    else:
        meta = {
            'phase': 11, 'algorithm': config.algorithm,
            'checkpoint': str(checkpoint.resolve()),
            'initial_updates': base.completed_updates,
            'network_config': asdict(base.network_config),
            'ppo_config': asdict(base.ppo_config),
            'beta': session.qconfig.beta if config.algorithm == 'qboost' else 0.0,
            'config': asdict(config),
        }
        _write_atomic_json(meta_path, meta)
    anchor = _load_or_create_anchor(base, path=anchor_path,
                                    resume=resuming, device=config.device)
    if not resuming:
        _checkpoint_save(session, config.algorithm, checkpoint)
    started = time.perf_counter()
    deadline = started + config.duration_seconds
    updates_this_run = 0
    last_seconds = None
    evaluations = 0
    reason = 'duration_limit'
    _log(events_path, {'kind': 'resume' if resuming else 'start',
                       'completed_updates': base.completed_updates,
                       'duration_seconds': config.duration_seconds,
                       'env_batch_size': config.env_batch_size}, emit)

    def evaluate_now():
        nonlocal evaluations
        for opp in (None, anchor):
            # Same seeds and four layout combinations at every evaluation.
            report = evaluate_balanced(
                base.actor, opponent=opp, hands=config.eval_hands,
                seed=1_200_000 + base.ppo_config.seed * 10,
                device=config.device, big_blind=base.ppo_config.big_blind,
            )
            _log(events_path, {'kind': 'evaluation', 'update': base.completed_updates,
                               'algorithm': config.algorithm, **report}, emit)
            evaluations += 1

    interrupted = False
    try:
        if not resuming and _budget_remaining(deadline):
            evaluate_now()
        while True:
            if config.max_updates is not None and updates_this_run >= config.max_updates:
                reason = 'update_limit'
                break
            if not _budget_remaining(deadline, last_seconds):
                break
            report = _step(session, config)
            # Atomic checkpoint FIRST; a missed log can be recovered using update counter.
            _checkpoint_save(session, config.algorithm, checkpoint)
            _log(events_path, report, emit)
            updates_this_run += 1
            last_seconds = float(report['elapsed_seconds'])
            if (base.completed_updates % config.eval_every == 0
                    and _budget_remaining(deadline)):
                evaluate_now()
        if (updates_this_run > 0 and base.completed_updates % config.eval_every != 0
                and _budget_remaining(deadline)):
            evaluate_now()
    except KeyboardInterrupt:
        interrupted = True
        reason = 'keyboard_interrupt'
        # Optimizer checkpoint is last completed update only. A mid-update SIGINT
        # must NOT overwrite it with partially updated, inconsistent parameters.
        _log(events_path, {'kind': 'interrupt', 'completed_updates_on_disk':
                           int(torch.load(checkpoint, map_location='cpu', weights_only=True)['completed_updates'])}, emit)
    elapsed = time.perf_counter() - started
    summary = {
        'phase': 11, 'algorithm': config.algorithm, 'device': config.device,
        'stop_reason': reason, 'interrupted': interrupted,
        'updates_this_run': updates_this_run,
        'completed_updates': int(torch.load(checkpoint, map_location='cpu',
                                            weights_only=True)['completed_updates']),
        'elapsed_seconds': round(elapsed, 4),
        'evaluations': evaluations, 'checkpoint': str(checkpoint),
        'log_dir': str(root), 'env_batch_size': config.env_batch_size,
        'cuda_peak_allocated_mb': (round(torch.cuda.max_memory_allocated()/2**20, 3)
                                   if config.device == 'cuda' else None),
    }
    _write_atomic_json(summary_path, summary)
    _log(events_path, {'kind': 'end', **summary}, emit)
    return summary
