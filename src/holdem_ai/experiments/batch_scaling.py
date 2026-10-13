"""Phase 13 training-batch experiments, preserving Phase 9/10 checkpoints.

New experiments may warm-start from Phase 9/10/11/13 weights + Adam state,
then explicitly override rollout/optimization batch size.  Resume of a Phase13
experiment uses the checkpoint's SAVED PPO configuration without silent edits.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import torch

from holdem_ai.models.transformer import NetworkConfig, TransformerActor
from holdem_ai.training.ppo import PPOConfig, freeze_old_policy_data, optimize_ppo
from holdem_ai.training.ppo_session import create_session, load_checkpoint, save_checkpoint
from holdem_ai.training.qboost import QBoostConfig, freeze_qboost_data, optimize_qboost
from holdem_ai.training.qboost_session import create_qboost_session, load_qboost_checkpoint, save_qboost_checkpoint, from_phase9_checkpoint
from holdem_ai.training.batched_collector import collect_batched_self_play
from holdem_ai.evaluation.seed_blocked import EvaluationConfig, evaluate_seed_blocks
from holdem_ai.evaluation.action_diagnostics import action_diagnostics
from holdem_ai.evaluation.reward_diagnostics import summarize_rollout_rewards


@dataclass(frozen=True)
class BatchExperimentConfig:
    algorithm: str = 'qboost'
    device: str = 'cpu'
    duration_seconds: float = 15 * 60
    max_updates: int | None = 200
    hands_per_update: int | None = None
    minibatch_size: int | None = None
    epochs: int | None = None
    rollout_seed: int | None = None
    env_batch_size: int = 32
    eval_every: int = 25
    eval_groups: int = 40
    eval_mode: str = 'sampled'
    skip_eval: bool = False
    small_network: bool = False
    beta: float = 1.0
    checkpoint: str = 'checkpoints/phase13_qboost.pt'
    log_dir: str = 'runs/phase13_qboost'
    from_checkpoint: str | None = None
    resume: str | None = None

    def __post_init__(self) -> None:
        if self.algorithm not in ('ppo', 'qboost') or self.device not in ('cpu', 'cuda'):
            raise ValueError('invalid algorithm/device')
        if isinstance(self.duration_seconds, bool) or not isinstance(self.duration_seconds, (int, float)) or not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError('duration_seconds must be finite and positive')
        for name in ('hands_per_update', 'minibatch_size', 'epochs', 'rollout_seed'):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < (0 if name == 'rollout_seed' else 1)):
                raise ValueError(f'{name} must be a nonnegative/positive int or None')
        for name in ('env_batch_size', 'eval_every'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive integer')
        if type(self.eval_groups) is not int or self.eval_groups < 2:
            raise ValueError('eval_groups must be >=2')
        if self.max_updates is not None and (type(self.max_updates) is not int or self.max_updates < 1):
            raise ValueError('max_updates must be positive int or None')
        if self.eval_mode not in ('sampled', 'greedy'):
            raise ValueError('eval_mode must be sampled/greedy')
        if not isinstance(self.beta, (int, float)) or isinstance(self.beta, bool) or not 0 <= self.beta <= 1:
            raise ValueError('beta must be [0,1]')
        if self.resume and self.from_checkpoint:
            raise ValueError('--resume and --from-checkpoint are mutually exclusive')
        if self.resume and any(getattr(self, field) is not None for field in
                               ('hands_per_update', 'minibatch_size', 'epochs', 'rollout_seed')):
            raise ValueError('resume uses saved batch parameters; start a NEW stage using --from-checkpoint to reconfigure')
        if self.from_checkpoint and Path(self.from_checkpoint).resolve() == Path(self.checkpoint).resolve():
            raise ValueError('warm-start must not overwrite its source checkpoint')
        if not self.checkpoint or not self.log_dir:
            raise ValueError('checkpoint and log_dir required')


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _event(path: Path, record: dict, emit: Callable[[dict], None] | None) -> None:
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    if emit:
        emit(record)


def _base(session, algorithm):
    return session if algorithm == 'ppo' else session.base


def _load_from_path(path: str, algorithm: str, device: str, beta: float = 1.0):
    metadata = torch.load(path, weights_only=True, map_location='cpu')
    phase = metadata.get('phase')
    if algorithm == 'ppo':
        if phase != 9:
            raise ValueError(f'ppo needs phase=9 checkpoint, got {phase}')
        return load_checkpoint(path, device=device)
    if phase == 9:
        return from_phase9_checkpoint(path, device=device, qconfig=QBoostConfig(beta=beta))
    if phase == 10:
        return load_qboost_checkpoint(path, device=device)
    raise ValueError(f'qboost needs phase=9/10 checkpoint, got {phase}')


def _initialize(config: BatchExperimentConfig):
    if config.resume:
        return _load_from_path(config.resume, config.algorithm, config.device, beta=config.beta)
    if config.from_checkpoint:
        session = _load_from_path(config.from_checkpoint, config.algorithm, config.device, beta=config.beta)
    else:
        net = (NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1,
                             ff_dim=64, hidden_dim=48, max_tokens=256)
               if config.small_network else NetworkConfig())
        default_cfg = PPOConfig(seed=42)
        session = (create_session(device=config.device, seed=42,
                                  network_config=net, ppo_config=default_cfg)
                   if config.algorithm == 'ppo' else
                   create_qboost_session(device=config.device, seed=42,
                                         network_config=net, ppo_config=default_cfg,
                                         qconfig=QBoostConfig(beta=config.beta)))
    base = _base(session, config.algorithm)
    if config.algorithm == 'qboost' and not math.isclose(session.qconfig.beta, config.beta):
        raise ValueError('qboost beta must match checkpoint; do not silently alter objective')
    old = base.ppo_config
    # New warm-start experiment gets a distinct seed namespace from its source.
    seed = (config.rollout_seed if config.rollout_seed is not None else
            old.seed + 1_000_003 if config.from_checkpoint else old.seed)
    new_cfg = replace(old,
                      hands_per_update=config.hands_per_update or 64,
                      minibatch_size=config.minibatch_size or 64,
                      epochs=config.epochs or 2,
                      seed=seed)
    base.ppo_config = new_cfg
    return session


def _save(session, algorithm, path):
    if algorithm == 'ppo':
        save_checkpoint(session, path)
    else:
        save_qboost_checkpoint(session, path)


def _make_anchor(base, path: Path, resume: bool):
    model = TransformerActor(base.network_config).to(base.device).eval().requires_grad_(False)
    if resume:
        source = torch.load(path, weights_only=True, map_location='cpu')
        if source['network_config'] != asdict(base.network_config):
            raise ValueError('anchor architecture mismatch')
        model.load_state_dict(source['actor_state'])
    else:
        model.load_state_dict(base.actor.state_dict())
        tmp = path.with_name(path.name + '.tmp')
        torch.save({'network_config': asdict(base.network_config),
                    'actor_state': {k: t.detach().cpu().clone()
                                    for k, t in model.state_dict().items()}}, tmp)
        os.replace(tmp, path)
    return model


def _one_update(session, config: BatchExperimentConfig) -> dict:
    base = _base(session, config.algorithm)
    iteration = base.completed_updates
    start = time.perf_counter()
    rows = collect_batched_self_play(base.actor, base.ppo_config,
                                     iteration=iteration, device=base.device,
                                     env_batch_size=config.env_batch_size)
    diagnostics = action_diagnostics(rows, base.actor, device=base.device,
                                     chunk_size=base.ppo_config.inference_batch_size)
    reward_diagnostics = summarize_rollout_rewards(rows)
    if config.algorithm == 'ppo':
        frozen = freeze_old_policy_data(base.actor, base.critic, rows,
                                        device=base.device,
                                        chunk_size=base.ppo_config.inference_batch_size)
        stats = optimize_ppo(base.actor, base.critic,
                             base.actor_optimizer, base.critic_optimizer,
                             frozen, base.ppo_config, device=base.device,
                             shuffle_seed=base.ppo_config.seed + 100003 * iteration)
    else:
        frozen = freeze_qboost_data(base.actor, base.critic, rows,
                                    device=base.device,
                                    chunk_size=base.ppo_config.inference_batch_size)
        stats = optimize_qboost(base.actor, base.critic,
                                base.actor_optimizer, base.critic_optimizer,
                                frozen, base.ppo_config, session.qconfig,
                                device=base.device,
                                shuffle_seed=base.ppo_config.seed + 100003 * iteration)
    if base.device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    base.completed_updates += 1
    return {'kind': 'update', 'algorithm': config.algorithm,
            'update': base.completed_updates, 'hands': rows.hands,
            'elapsed_seconds': round(elapsed, 5),
            'hands_per_second_training': round(rows.hands / max(elapsed, 1e-9), 3),
            'env_batch_size': config.env_batch_size,
            'minibatch_size': base.ppo_config.minibatch_size,
            'ppo_epochs': base.ppo_config.epochs,
            'action_diagnostics': diagnostics,
            'reward_diagnostics': reward_diagnostics, **stats}


def run_batch_experiment(config: BatchExperimentConfig, *,
                         emit: Callable[[dict], None] | None = None) -> dict:
    """Atomic checkpoint each update.  Resume uses same log dir and checkpoint path.

    Duration is an invocation budget (not a global wall clock across restarts).
    The only change to PPO/QBoost math is the number of sampled hands/optimizer
    minibatches; the loss implementations are imported unchanged from Phase9/10.
    """
    if config.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    root = Path(config.log_dir)
    checkpoint = Path(config.checkpoint)
    meta_path, anchor_path = root / 'run.json', root / 'anchor.pt'
    events_path, summary_path = root / 'metrics.jsonl', root / 'summary.json'
    resuming = config.resume is not None
    if resuming:
        if not all(p.is_file() for p in (meta_path, events_path, anchor_path, checkpoint)):
            raise FileNotFoundError('resume requires run.json/metrics.jsonl/anchor.pt/checkpoint')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if (meta.get('phase') != 13 or meta.get('algorithm') != config.algorithm
                or Path(meta['checkpoint']).resolve() != checkpoint.resolve()
                or checkpoint.resolve() != Path(config.resume).resolve()):
            raise ValueError('resume metadata/checkpoint path mismatch')
    else:
        if checkpoint.exists() or (root.exists() and any(root.iterdir())):
            raise FileExistsError('fresh experiment cannot overwrite existing checkpoint/logs')
        root.mkdir(parents=True, exist_ok=True)
    session = _initialize(config)
    base = _base(session, config.algorithm)
    if resuming:
        if (asdict(base.ppo_config) != meta['ppo_config'] or
                asdict(base.network_config) != meta['network_config']):
            raise ValueError('checkpoint and run.json configuration disagree')
    else:
        meta = {'phase': 13, 'algorithm': config.algorithm,
                'checkpoint': str(checkpoint.resolve()),
                'initial_updates': base.completed_updates,
                'source_checkpoint': str(Path(config.from_checkpoint).resolve()) if config.from_checkpoint else None,
                'checkpoint_format_phase': 9 if config.algorithm == 'ppo' else 10,
                'network_config': asdict(base.network_config),
                'ppo_config': asdict(base.ppo_config),
                'beta': session.qconfig.beta if config.algorithm == 'qboost' else 0.0,
                'config': asdict(config)}
        _atomic_json(meta_path, meta)
    anchor = _make_anchor(base, anchor_path, resume=resuming)
    if not resuming:
        _save(session, config.algorithm, checkpoint)
    if config.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    end_time = started + config.duration_seconds
    completed = 0
    prev_seconds = 0.0
    evaluations = 0
    reason = 'duration_limit'
    _event(events_path, {'kind': 'resume' if resuming else 'start',
                         'completed_updates': base.completed_updates,
                         'duration_seconds': config.duration_seconds,
                         'env_batch_size': config.env_batch_size,
                         'hands_per_update': base.ppo_config.hands_per_update}, emit)

    def eval_now():
        nonlocal evaluations
        cfg = EvaluationConfig(groups=config.eval_groups, seed=12_000_000,
                               mode=config.eval_mode, device=config.device,
                               big_blind=base.ppo_config.big_blind)
        for opponent, name in ((None, 'random'), (anchor, 'frozen_anchor')):
            summary = evaluate_seed_blocks(base.actor, opponent=opponent,
                                           config=cfg).summary()
            _event(events_path, {'kind': 'evaluation', 'update': base.completed_updates,
                                 'opponent': name, **summary}, emit)
            evaluations += 1

    try:
        if not config.skip_eval and not resuming and time.perf_counter() < end_time:
            eval_now()
        while True:
            if config.max_updates is not None and completed >= config.max_updates:
                reason = 'update_limit'
                break
            if time.perf_counter() + max(prev_seconds * 1.1, 0.02) >= end_time:
                break
            report = _one_update(session, config)
            # Always save the optimizer state before declaring an update committed.
            _save(session, config.algorithm, checkpoint)
            _event(events_path, report, emit)
            completed += 1
            prev_seconds = report['elapsed_seconds']
            if (not config.skip_eval and base.completed_updates % config.eval_every == 0
                    and time.perf_counter() < end_time):
                eval_now()
        if (not config.skip_eval and completed > 0 and
                base.completed_updates % config.eval_every != 0 and
                time.perf_counter() < end_time):
            eval_now()
    except KeyboardInterrupt:
        reason = 'keyboard_interrupt'
        _event(events_path, {'kind': 'interrupt', 'update_on_disk':
                             int(torch.load(checkpoint, weights_only=True,
                                            map_location='cpu')['completed_updates'])}, emit)
    elapsed = time.perf_counter() - started
    summary = {'phase': 13, 'algorithm': config.algorithm, 'device': config.device,
               'stop_reason': reason, 'updates_this_run': completed,
               'completed_updates': int(torch.load(checkpoint, weights_only=True,
                                                   map_location='cpu')['completed_updates']),
               'elapsed_seconds': round(elapsed, 5),
               'evaluations': evaluations,
               'hands_per_update': base.ppo_config.hands_per_update,
               'env_batch_size': config.env_batch_size,
               'checkpoint': str(checkpoint), 'log_dir': str(root),
               'cuda_peak_allocated_mb': (round(torch.cuda.max_memory_allocated() / 2**20, 3)
                                          if config.device == 'cuda' else None)}
    _atomic_json(summary_path, summary)
    _event(events_path, {'kind': 'end', **summary}, emit)
    return summary
