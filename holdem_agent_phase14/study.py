"""Matched-start PPO/QBoost study with independent held-out test evaluation.

Both arms start from a trusted Phase9 checkpoint (identical actor/critic AND
Adam states). Within each training seed, on-policy rollouts have identical RNG
seed schedules but can diverge after the networks change. A validation-only
best checkpoint is saved; test seeds are untouched until all arms complete.

Phase13 supplies the unchanged PPO/QBoost loss and batched self-play step.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import torch

from holdem_agent_phase11.progress import TrainingProgress
from holdem_agent_phase12 import EvaluationConfig, compare_results, evaluate_seed_blocks
from holdem_agent_phase12.checkpoint import load_actor
from holdem_agent_phase12.evaluation import model_fingerprint
from holdem_agent_phase13.runner import (BatchExperimentConfig, _base,
                                         _load_from_path, _one_update, _save)
from .statistics import seed_level_summary


@dataclass(frozen=True)
class ComparisonConfig:
    source_checkpoint: str
    root: str = 'runs/phase14_study'
    seeds: tuple[int, ...] = (41, 42, 43)
    updates: int = 100
    hands_per_update: int = 64
    env_batch_size: int = 32
    minibatch_size: int = 64
    epochs: int = 2
    beta: float = 1.0
    eval_every: int = 25
    validation_groups: int = 40
    test_groups: int = 200
    eval_mode: str = 'sampled'
    device: str = 'cpu'
    arm_duration_minutes: float = 30.0
    validation_seed: int = 12_000_000
    test_seed: int = 22_000_000
    entropy_warning_threshold: float = 0.6
    resume: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if not self.source_checkpoint or not self.root:
            raise ValueError('source_checkpoint and root are required')
        if self.device not in ('cpu', 'cuda') or self.eval_mode not in ('sampled', 'greedy'):
            raise ValueError('invalid device/evaluation mode')
        if not self.seeds or any(type(s) is not int or not 0 <= s < 100_000 for s in self.seeds):
            raise ValueError('seeds must be distinct integers in [0,100000)')
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError('training seeds must be unique')
        for key in ('updates', 'hands_per_update', 'env_batch_size', 'minibatch_size',
                    'epochs', 'eval_every', 'validation_groups', 'test_groups'):
            val = getattr(self, key)
            if type(val) is not int or val < (2 if key.endswith('groups') else 1):
                raise ValueError(f'{key} must be positive integer (groups >= 2)')
        if self.env_batch_size > self.hands_per_update:
            raise ValueError('env_batch_size must be <= hands_per_update')
        for name in ('validation_seed', 'test_seed'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative integer')
        if self.updates * self.hands_per_update >= 1_000_000:
            raise ValueError('rollout seed namespace supports fewer than 1,000,000 hands per arm')
        for start, groups in ((self.validation_seed, self.validation_groups),
                              (self.test_seed, self.test_groups)):
            # Rollout cards occupy the 100-million+ namespace (disjoint by seed).
            for train_seed in self.seeds:
                lo = 100_000_000 + train_seed * 1_000_000
                hi = lo + 1_000_000
                if start < hi and start + groups > lo:
                    raise ValueError('evaluation seed range overlaps a training rollout seed range')
        v = set(range(self.validation_seed, self.validation_seed+self.validation_groups))
        t = set(range(self.test_seed, self.test_seed+self.test_groups))
        if v & t:
            raise ValueError('validation and test deck seeds must not overlap')
        if not 0 <= self.beta <= 1 or not math.isfinite(self.beta):
            raise ValueError('beta must be finite in [0,1]')
        if not (math.isfinite(self.arm_duration_minutes) and self.arm_duration_minutes > 0):
            raise ValueError('arm_duration_minutes must be finite positive')
        if not (math.isfinite(self.entropy_warning_threshold) and self.entropy_warning_threshold >= 0):
            raise ValueError('entropy warning threshold must be >= 0')


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _event(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _source_and_metadata(cfg: ComparisonConfig) -> tuple[Path, dict, dict]:
    source = Path(cfg.source_checkpoint).resolve(strict=True)
    payload = torch.load(source, map_location='cpu', weights_only=True)
    if payload.get('phase') != 9 or not all(k in payload for k in
            ('actor_optimizer', 'critic_optimizer', 'actor_state', 'critic_state')):
        raise ValueError('matched PPO/QBoost study requires a trusted Phase9 full checkpoint')
    digest = _sha256(source)
    frozen_cfg = asdict(cfg)
    # Only `resume` and console presentation may differ across invocations.
    frozen_cfg.pop('resume')
    frozen_cfg.pop('progress')
    frozen_cfg.pop('arm_duration_minutes')  # new wall-clock budget per resumed invocation
    frozen_cfg['source_checkpoint'] = str(source)
    frozen_cfg['seeds'] = list(cfg.seeds)
    frozen_cfg['source_sha256'] = digest
    return source, payload, frozen_cfg


def _case_paths(root: Path, seed: int, algorithm: str) -> dict[str, Path]:
    case = root / f'seed_{seed}' / algorithm
    return {'root': case, 'latest': case / 'latest.pt', 'best': case / 'best.pt',
            'state': case / 'state.json', 'events': case / 'metrics.jsonl',
            'meta': case / 'run.json'}


def _rollout_seed(seed: int) -> int:
    return 100_000_000 + seed * 1_000_000


def _make_batch_config(cfg: ComparisonConfig, path: Path, seed: int, algorithm: str) -> BatchExperimentConfig:
    return BatchExperimentConfig(
        algorithm=algorithm, device=cfg.device, duration_seconds=cfg.arm_duration_minutes*60,
        max_updates=cfg.updates, hands_per_update=cfg.hands_per_update,
        minibatch_size=cfg.minibatch_size, epochs=cfg.epochs,
        rollout_seed=_rollout_seed(seed), env_batch_size=cfg.env_batch_size,
        eval_every=cfg.eval_every, eval_groups=cfg.validation_groups,
        eval_mode=cfg.eval_mode, beta=cfg.beta,
        checkpoint=str(path), log_dir=str(path.parent), skip_eval=True,
    )


def _eval(actor, opponent, cfg: ComparisonConfig, *, partition: str):
    seed, groups = ((cfg.validation_seed, cfg.validation_groups)
                    if partition == 'validation' else (cfg.test_seed, cfg.test_groups))
    settings = EvaluationConfig(groups=groups, seed=seed, mode=cfg.eval_mode,
                                device=cfg.device)
    return evaluate_seed_blocks(actor, opponent=opponent, config=settings)


def _new_arm(cfg: ComparisonConfig, source: Path, seed: int, algorithm: str,
             paths: dict[str, Path], initial_fingerprint: str) -> None:
    if paths['root'].exists() and any(paths['root'].iterdir()):
        raise FileExistsError(f'new study cannot overwrite {paths["root"]}')
    paths['root'].mkdir(parents=True, exist_ok=True)
    session = _load_from_path(str(source), algorithm, cfg.device, beta=cfg.beta)
    base = _base(session, algorithm)
    base.ppo_config = replace(base.ppo_config, seed=_rollout_seed(seed),
                              hands_per_update=cfg.hands_per_update,
                              minibatch_size=cfg.minibatch_size, epochs=cfg.epochs)
    if model_fingerprint(base.actor) != initial_fingerprint:
        raise AssertionError('matched-start initial actor fingerprint mismatch')
    _save(session, algorithm, paths['latest'])
    _write_json(paths['meta'], {
        'phase': 14, 'algorithm': algorithm, 'seed': seed,
        'initial_update': base.completed_updates,
        'source_fingerprint': initial_fingerprint,
        'ppo_config': asdict(base.ppo_config),
        'network_config': asdict(base.network_config),
    })
    _write_json(paths['state'], {
        'best_score_bb100': None, 'best_update': None,
        'last_validation_update': None, 'entropy_warnings': 0,
    })
    _event(paths['events'], {'kind': 'start', 'algorithm': algorithm,
                             'completed_updates': base.completed_updates,
                             'duration_seconds': cfg.arm_duration_minutes*60,
                             'env_batch_size': cfg.env_batch_size})


def _load_arm(paths: dict[str, Path], cfg: ComparisonConfig, seed: int,
              algorithm: str, fingerprint: str):
    meta = json.loads(paths['meta'].read_text(encoding='utf-8'))
    state = json.loads(paths['state'].read_text(encoding='utf-8'))
    session = _load_from_path(str(paths['latest']), algorithm, cfg.device, beta=cfg.beta)
    base = _base(session, algorithm)
    if (meta['source_fingerprint'] != fingerprint or meta['seed'] != seed or
            meta['algorithm'] != algorithm or meta['ppo_config'] != asdict(base.ppo_config)
            or meta['network_config'] != asdict(base.network_config)):
        raise ValueError('arm checkpoint and study metadata diverged; refusing resume')
    if base.completed_updates < meta['initial_update'] or base.completed_updates > meta['initial_update'] + cfg.updates:
        raise ValueError('checkpoint counter outside pre-registered update budget')
    return session, meta, state


def _validation(cfg: ComparisonConfig, base, anchor, paths: dict[str, Path],
                state: dict, label: str) -> dict:
    result = _eval(base.actor, anchor, cfg, partition='validation').summary()
    score = float(result['mean_bb100'])
    update = base.completed_updates
    event = {'kind': 'validation', 'update': update, 'algorithm': label,
             'opponent': 'frozen_initial', **result}
    _event(paths['events'], event)
    if state['best_score_bb100'] is None or score > state['best_score_bb100']:
        # The saved file contains precisely the policy just evaluated.
        tmp = paths['best'].with_name('best.pt.tmp')
        shutil.copyfile(paths['latest'], tmp)
        os.replace(tmp, paths['best'])
        state['best_score_bb100'] = score
        state['best_update'] = update
    state['last_validation_update'] = update
    _write_json(paths['state'], state)
    return event


def _run_arm(cfg: ComparisonConfig, source: Path, seed: int, algorithm: str,
             root: Path, anchor, fingerprint: str,
             emit: Callable[[dict], None] | None = None) -> dict:
    paths = _case_paths(root, seed, algorithm)
    if not paths['meta'].is_file():
        if cfg.resume and paths['root'].exists() and any(paths['root'].iterdir()):
            raise FileNotFoundError(f'incomplete arm metadata: {paths["meta"]}')
        _new_arm(cfg, source, seed, algorithm, paths, fingerprint)
    session, meta, state = _load_arm(paths, cfg, seed, algorithm, fingerprint)
    base = _base(session, algorithm)
    first = meta['initial_update']
    target = first + cfg.updates
    # The loss and collector use Phase13 without changing the policy-gradient math.
    step_cfg = _make_batch_config(cfg, paths['latest'], seed, algorithm)
    deadline = time.perf_counter() + cfg.arm_duration_minutes*60
    updates_this_call = 0
    with _Phase14Progress(duration_seconds=cfg.arm_duration_minutes*60,
                          max_updates=max(1, target-base.completed_updates),
                          threaded=cfg.progress, prefer_tqdm=cfg.progress) if cfg.progress else _SilentProgress() as bar:
        bar.on_event({'kind': 'start', 'completed_updates': base.completed_updates})
        if state['last_validation_update'] is None:
            row = _validation(cfg, base, anchor, paths, state, algorithm)
            if emit:
                emit(row)
        while base.completed_updates < target:
            if time.perf_counter() + 0.05 >= deadline:
                break
            try:
                item = _one_update(session, step_cfg)
            except KeyboardInterrupt:
                _event(paths['events'], {'kind': 'interrupt', 'last_saved_update': base.completed_updates})
                break
            _save(session, algorithm, paths['latest'])
            _event(paths['events'], item)
            bar.on_event(item)
            updates_this_call += 1
            if item.get('entropy', 999) < cfg.entropy_warning_threshold:
                state['entropy_warnings'] += 1
                _event(paths['events'], {'kind':'entropy_warning', 'update':base.completed_updates,
                                         'entropy':item['entropy']})
                _write_json(paths['state'], state)
            if base.completed_updates % cfg.eval_every == 0 or base.completed_updates == target:
                row = _validation(cfg, base, anchor, paths, state, algorithm)
                if emit:
                    emit(row)
        bar.on_event({'kind': 'end', 'stop_reason': 'update_limit' if base.completed_updates == target
                      else 'duration_limit'})
    if not paths['best'].exists():
        raise AssertionError('no best validation checkpoint generated')
    return {'algorithm': algorithm, 'seed': seed,
            'training_diagnostics': _summarize_training(paths['events']),
            'updates_this_call': updates_this_call,
            'completed_updates': base.completed_updates,
            'target_updates': target, 'finished': base.completed_updates == target,
            'best_validation_update': state['best_update'],
            'best_validation_bb100': state['best_score_bb100'],
            'entropy_warnings': state['entropy_warnings'],
            'best_checkpoint': str(paths['best']), 'latest_checkpoint': str(paths['latest'])}


def _summarize_training(path: Path) -> dict:
    updates = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
               if line.strip()]
    updates = [x for x in updates if x.get('kind') == 'update']
    count = len(updates)
    def average(name):
        values = [float(u[name]) for u in updates if name in u]
        return round(math.fsum(values)/len(values), 6) if values else None
    hands = sum(int(u['hands']) for u in updates)
    seconds = math.fsum(float(u['elapsed_seconds']) for u in updates)
    metrics = {'updates_logged': count, 'training_hands_logged': hands,
               'training_seconds_logged': round(seconds, 4),
               'full_training_hands_per_second': round(hands/seconds, 3) if seconds else None,
               'average_entropy': average('entropy'),
               'average_critic_loss': average('critic_loss'),
               'average_approx_kl': average('approx_kl'),
               'average_logit_variance_ratio': average('logit_variance_ratio')}
    for a in ('fold','check','call','raise_double_pot','all_in'):
        opts = [u['action_diagnostics'] for u in updates if 'action_diagnostics' in u]
        legal = sum(o['legal_opportunities'].get(a,0) for o in opts)
        used = sum(o['actions'].get(a,0) for o in opts)
        metrics[a+'_when_legal'] = round(used/legal,6) if legal else None
    return metrics


class _Phase14Progress(TrainingProgress):
    def _open_display(self):
        super()._open_display()
        if self._bar is not None:
            self._bar.set_description('Phase 14')


class _SilentProgress:
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def on_event(self, event): pass


def _final_test(cfg: ComparisonConfig, root: Path, seed: int, anchor) -> dict:
    # Identical test deck seeds, initial anchor and evaluation mode across both arms.
    output = root / f'seed_{seed}' / 'test_results.json'
    p_paths, q_paths = (_case_paths(root, seed, x) for x in ('ppo','qboost'))
    if output.exists():
        cached = json.loads(output.read_text(encoding='utf-8'))
        if (cached.get('best_sha256_ppo') != _sha256(p_paths['best'])
                or cached.get('best_sha256_qboost') != _sha256(q_paths['best'])):
            raise ValueError('best checkpoint changed after test evaluation')
        return cached
    p_model = load_actor(p_paths['best'], device=cfg.device)
    q_model = load_actor(q_paths['best'], device=cfg.device)
    p_anchor = _eval(p_model, anchor, cfg, partition='test')
    q_anchor = _eval(q_model, anchor, cfg, partition='test')
    paired = compare_results(q_anchor, p_anchor)
    p_random = _eval(p_model, None, cfg, partition='test')
    q_random = _eval(q_model, None, cfg, partition='test')
    paired_random = compare_results(q_random, p_random)
    head_to_head = _eval(q_model, p_model, cfg, partition='test')
    result = {'seed': seed, 'phase': 14, 'partition': 'test',
              'best_sha256_ppo': _sha256(p_paths['best']),
              'best_sha256_qboost': _sha256(q_paths['best']),
              'best_checkpoint_ppo': str(p_paths['best']),
              'best_checkpoint_qboost': str(q_paths['best']),
              'ppo_vs_initial': p_anchor.summary(), 'qboost_vs_initial': q_anchor.summary(),
              'qboost_minus_ppo_vs_initial': paired,
              'ppo_vs_random': p_random.summary(), 'qboost_vs_random': q_random.summary(),
              'qboost_minus_ppo_vs_random': paired_random,
              'qboost_vs_ppo_headtohead': head_to_head.summary()}
    _write_json(output, result)
    return result


def run_study(cfg: ComparisonConfig, *, emit: Callable[[dict], None] | None = None) -> dict:
    """Run all arms; save partial results if a per-arm deadline is reached.

    `--resume` must be used to continue the SAME preregistered study. Test
    results are produced only after every PPO and QBoost arm has completed.
    """
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    source, _payload, prereg = _source_and_metadata(cfg)
    root = Path(cfg.root)
    study_file = root / 'study.json'
    if cfg.resume:
        if not study_file.is_file():
            raise FileNotFoundError('no Phase14 study.json to resume')
        original = json.loads(study_file.read_text(encoding='utf-8'))
        if original != prereg:
            raise ValueError('study configuration/source bytes differ; refusing resume')
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError('fresh Phase14 study cannot overwrite prior experiment')
        if root.resolve() == source.parent or source.is_relative_to(root.resolve()):
            raise ValueError('study directory must not contain the source checkpoint')
        _write_json(study_file, prereg)
    anchor = load_actor(source, device=cfg.device)
    fingerprint = model_fingerprint(anchor)
    outputs = []
    for seed in cfg.seeds:
        for algorithm in ('ppo','qboost'):
            item = _run_arm(cfg, source, seed, algorithm, root, anchor, fingerprint, emit=emit)
            outputs.append(item)
            if emit:
                emit({'kind':'arm_end', **item})
            if not item['finished']:
                partial = {'phase':14, 'status':'partial', 'root':str(root),
                           'arms': outputs, 'note':'Use --resume; test partition remains untouched.'}
                _write_json(root/'summary.json', partial)
                return partial
    test_results = [_final_test(cfg, root, seed, anchor) for seed in cfg.seeds]
    diffs = [r['qboost_minus_ppo_vs_initial']['delta_mean_bb100'] for r in test_results]
    diffs_random = [r['qboost_minus_ppo_vs_random']['delta_mean_bb100'] for r in test_results]
    h2h = [r['qboost_vs_ppo_headtohead']['mean_bb100'] for r in test_results]
    result = {
        'phase': 14, 'status': 'complete', 'root': str(root),
        'source_sha256': prereg['source_sha256'], 'training_seeds': list(cfg.seeds),
        'total_training_hands': 2*len(cfg.seeds)*cfg.updates*cfg.hands_per_update,
        'arms': outputs,
        'primary_qboost_minus_ppo_vs_initial': seed_level_summary(diffs),
        'secondary_qboost_minus_ppo_vs_random': seed_level_summary(diffs_random),
        'secondary_direct_headtohead_qboost_bb100': seed_level_summary(h2h),
        'seed_results': test_results,
        'notes': [
            'Preselected checkpoints by validation vs the same frozen source Actor.',
            'Test seeds are disjoint and first accessed only after all training finishes.',
            'Training-seed t intervals are conditional on the shared held-out deck bank.',
            'Policy trajectories diverge after updates; common RNG seeds are not identical trajectories.',
            'This evaluation is not GTO exploitability or proof of VRPO-paper reproduction.',
        ],
    }
    _write_json(root/'summary.json', result)
    return result
