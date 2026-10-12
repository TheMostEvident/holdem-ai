"""Phase 15: safe long-run HUNL training, validation and sealed final audit.

Reuses Phase13's on-policy collector / PPO and QBoost optimizers unchanged.
A fresh run MUST have a new directory and an independent rollout seed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import torch

from holdem_ai.utils.progress import TrainingProgress
from holdem_ai.evaluation.seed_blocked import EvaluationConfig, evaluate_seed_blocks
from holdem_ai.evaluation.checkpoint_loader import load_actor
from holdem_ai.experiments.batch_scaling import BatchExperimentConfig, _base, _load_from_path, _one_update, _save


@contextmanager
def _exclusive_lock(root: Path):
    """Non-blocking OS file lock; released automatically after process death.

    A persistent .phase15.lock is expected and is NOT a partial checkpoint.
    """
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.phase15.lock').open('a+b') as handle:
        handle.seek(0)
        handle.write(b'\0')
        handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('another Phase15 process holds this run directory') from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('another Phase15 process holds this run directory') from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


DEFAULTS = {
    'hands_per_update': 64, 'env_batch_size': 32, 'minibatch_size': 64,
    'epochs': 2, 'rollout_seed': 400_000_000, 'eval_every': 100,
    'eval_groups': 100, 'eval_mode': 'sampled', 'validation_seed': 15_000_000,
    'final_groups': 400, 'final_seed': 25_000_000,
    'entropy_alert_threshold': 0.45, 'entropy_stop_patience': 12,
    'max_approx_kl': 0.10, 'beta': 1.0,
}


@dataclass(frozen=True)
class LongRunConfig:
    root: str = 'runs/phase15_ppo'
    source_checkpoint: str | None = None
    algorithm: str | None = None
    device: str = 'cpu'
    duration_minutes: float = 300.0
    max_updates: int | None = None  # ADDITIONAL updates THIS invocation
    resume: bool = False
    progress: bool = True
    hands_per_update: int | None = None
    env_batch_size: int | None = None
    minibatch_size: int | None = None
    epochs: int | None = None
    rollout_seed: int | None = None
    eval_every: int | None = None
    eval_groups: int | None = None
    eval_mode: str | None = None
    validation_seed: int | None = None
    final_groups: int | None = None
    final_seed: int | None = None
    entropy_alert_threshold: float | None = None
    entropy_stop_patience: int | None = None
    max_approx_kl: float | None = None
    beta: float | None = None

    def __post_init__(self) -> None:
        if not self.root or self.device not in ('cpu', 'cuda'):
            raise ValueError('nonempty root and cpu/cuda device required')
        if self.algorithm is not None and self.algorithm not in ('ppo','qboost'):
            raise ValueError('algorithm must be ppo/qboost')
        if not self.resume and (not self.source_checkpoint or not self.algorithm):
            raise ValueError('fresh run requires --source-checkpoint and --algorithm')
        if not isinstance(self.duration_minutes, (int,float)) or isinstance(self.duration_minutes, bool) or not math.isfinite(self.duration_minutes) or self.duration_minutes <= 0:
            raise ValueError('duration_minutes must be finite positive')
        if self.max_updates is not None and (type(self.max_updates) is not int or self.max_updates < 1):
            raise ValueError('max_updates must be positive or omitted')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _event(path: Path, row: dict, emit: Callable[[dict],None] | None = None) -> None:
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    if emit:
        emit(row)


def _paths(root: Path) -> dict[str, Path]:
    return {k: root / v for k, v in {
        'manifest': 'run.json', 'latest': 'latest.pt', 'best': 'best.pt',
        'state': 'state.json', 'events': 'metrics.jsonl',
        'summary': 'summary.json', 'final': 'final_report.json',
    }.items()}


def _spec(cfg: LongRunConfig) -> dict:
    spec = {name: (DEFAULTS[name] if getattr(cfg,name) is None else getattr(cfg,name)) for name in DEFAULTS}
    for name in ('hands_per_update','env_batch_size','minibatch_size','epochs','eval_every',
                 'eval_groups','final_groups','entropy_stop_patience'):
        value = spec[name]
        if type(value) is not int or value < (2 if name in ('eval_groups','final_groups') else 1):
            raise ValueError(f'{name} must be a positive integer (groups >=2)')
    for name in ('rollout_seed','validation_seed','final_seed'):
        if type(spec[name]) is not int or spec[name] < 0:
            raise ValueError(f'{name} must be nonnegative integer')
    if spec['env_batch_size'] > spec['hands_per_update']:
        raise ValueError('env_batch_size cannot exceed hands_per_update')
    if spec['eval_mode'] not in ('sampled','greedy'):
        raise ValueError('eval_mode must be sampled/greedy')
    for name in ('entropy_alert_threshold','max_approx_kl','beta'):
        value = spec[name]
        if (not isinstance(value, (float,int)) or isinstance(value,bool)
                or not math.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be finite nonnegative')
    if spec['beta'] > 1 or spec['max_approx_kl'] <= 0:
        raise ValueError('beta <=1 and max_approx_kl >0 required')
    # Each population of evaluation decks must be held out of all training RNGs.
    intervals = [(spec['validation_seed'],spec['validation_seed']+spec['eval_groups']),
                 (spec['final_seed'],spec['final_seed']+spec['final_groups'])]
    if max(intervals[0][0],intervals[1][0]) < min(intervals[0][1],intervals[1][1]):
        raise ValueError('validation and final card seed ranges overlap')
    # Reserve a 20M-seed training block (up to 20M / hands_per_update updates).
    train_range = (spec['rollout_seed'],spec['rollout_seed']+20_000_000)
    for a,b in intervals:
        if max(a,train_range[0]) < min(b,train_range[1]):
            raise ValueError('held-out card seeds overlap reserved rollout seed namespace')
    return spec


def _read(root: Path) -> tuple[dict,dict,dict[str,Path]]:
    paths = _paths(root)
    if not all(paths[k].is_file() for k in ('manifest','state','latest','best','events')):
        raise FileNotFoundError('incomplete Phase15 run: missing manifest/state/latest/best/metrics')
    manifest=json.loads(paths['manifest'].read_text(encoding='utf-8'))
    state=json.loads(paths['state'].read_text(encoding='utf-8'))
    if state.get('best_sha256') != _sha256(paths['best']):
        raise ValueError('selected best.pt changed or best/state files are out of sync')
    return manifest,state,paths


def _match_resume(cfg: LongRunConfig, manifest: dict) -> dict:
    spec = manifest['settings']
    if cfg.source_checkpoint is not None and Path(cfg.source_checkpoint).resolve() != Path(manifest['source']).resolve():
        raise ValueError('resume source checkpoint differs')
    if cfg.algorithm is not None and cfg.algorithm != manifest['algorithm']:
        raise ValueError('resume algorithm differs')
    for name in DEFAULTS:
        value = getattr(cfg, name)
        if value is not None and value != spec[name]:
            raise ValueError(f'resume configuration mismatch: {name}')
    if _sha256(Path(manifest['source'])) != manifest['source_sha256']:
        raise ValueError('source checkpoint bytes changed since experiment initialization')
    return spec


def _evaluation(actor, opponent, spec: dict, device: str, *, final: bool = False, big_blind: int = 100) -> dict:
    settings = EvaluationConfig(
        groups=spec['final_groups'] if final else spec['eval_groups'],
        seed=spec['final_seed'] if final else spec['validation_seed'],
        mode=spec['eval_mode'], device=device, big_blind=big_blind,
    )
    return evaluate_seed_blocks(actor, opponent=opponent, config=settings).summary()


def _validation(base, anchor, spec: dict, device: str, baseline: dict) -> dict:
    current_anchor = _evaluation(base.actor, anchor, spec, device, big_blind=base.ppo_config.big_blind)
    current_random = _evaluation(base.actor, None, spec, device, big_blind=base.ppo_config.big_blind)
    # Equal-weighted CHANGE from initial policy against two fixed opponents.
    anchor_gain = current_anchor['mean_bb100'] - baseline['anchor']['mean_bb100']
    random_gain = current_random['mean_bb100'] - baseline['random']['mean_bb100']
    score = (anchor_gain + random_gain) / 2
    return {'kind': 'validation', 'update': base.completed_updates,
            'score_bb100': round(score,4), 'gain_vs_anchor_bb100': round(anchor_gain,4),
            'gain_vs_random_bb100': round(random_gain,4),
            'vs_anchor': current_anchor, 'vs_random': current_random}


class _Progress(TrainingProgress):
    def _open_display(self):
        super()._open_display()
        if self._bar is not None:
            self._bar.set_description('Phase 15')


def _emit_or_skip(event: dict, emit: Callable[[dict],None] | None, bar) -> None:
    bar.on_event(event)
    if emit:
        emit(event)


def _summary(root: Path, manifest: dict, state: dict, *, stop_reason: str,
             updates_this_run: int, elapsed: float, peak: float | None) -> dict:
    return {
        'phase': 15, 'algorithm': manifest['algorithm'], 'stop_reason': stop_reason,
        'updates_this_run': updates_this_run, 'completed_updates': state['completed_updates'],
        'total_updates_since_start': state['completed_updates']-manifest['initial_update'],
        'total_training_hands': (state['completed_updates']-manifest['initial_update'])*manifest['settings']['hands_per_update'],
        'elapsed_seconds_this_call': round(elapsed,3),
        'best_validation_update': state['best_update'],
        'best_validation_score_bb100': state['best_score_bb100'],
        'entropy_warnings': state['entropy_warnings'],
        'kl_warnings': state['kl_warnings'], 'latest_checkpoint': str(root/'latest.pt'),
        'best_checkpoint': str(root/'best.pt'), 'cuda_peak_allocated_mb': peak,
        'sealed': (root/'final_report.json').is_file(),
    }


def _run_longrun_impl(cfg: LongRunConfig, *, emit: Callable[[dict],None] | None = None) -> dict:
    """Train against a per-invocation wall-clock budget, with safe resume.

    A completed update is persisted atomically BEFORE logging. Unexpected power
    failure can lose the last metrics row but never silently reuse old weights.
    """
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    root = Path(cfg.root)
    paths = _paths(root)
    if cfg.resume:
        manifest,state,paths = _read(root)
        spec = _match_resume(cfg, manifest)
        if paths['final'].exists():
            raise RuntimeError('run is sealed by final evaluation; cannot resume training')
        algorithm = manifest['algorithm']
        session = _load_from_path(str(paths['latest']), algorithm, cfg.device, beta=spec['beta'])
        base = _base(session,algorithm)
        if asdict(base.ppo_config) != manifest['ppo_config'] or asdict(base.network_config) != manifest['network_config']:
            raise ValueError('checkpoint config differs from manifest')
        if base.completed_updates < state['completed_updates']:
            raise ValueError('latest checkpoint is older than committed state')
        if base.completed_updates > state['completed_updates']:
            # Latest save preceded event/state write when the previous process failed.
            _event(paths['events'], {'kind':'recovered_update','update':base.completed_updates})
            state['completed_updates'] = base.completed_updates
            _atomic_json(paths['state'],state)
    else:
        if root.exists() and any(p.name != '.phase15.lock' for p in root.iterdir()):
            raise FileExistsError('fresh Phase15 root is not empty; use --resume or a new directory')
        spec = _spec(cfg)
        algorithm = cfg.algorithm
        source = Path(cfg.source_checkpoint).resolve(strict=True)
        if source == paths['latest'].resolve() or source.is_relative_to(root.resolve()):
            raise ValueError('run directory cannot contain the source checkpoint')
        session = _load_from_path(str(source), algorithm, cfg.device, beta=spec['beta'])
        base = _base(session,algorithm)
        if algorithm == 'qboost' and not math.isclose(session.qconfig.beta, spec['beta']):
            raise ValueError('QBoost beta must match source checkpoint')
        base.ppo_config = replace(base.ppo_config,
                                  hands_per_update=spec['hands_per_update'],
                                  minibatch_size=spec['minibatch_size'],
                                  epochs=spec['epochs'],seed=spec['rollout_seed'])
        # No Future-deck or opponent-hole-card observation reaches Actor.
        manifest = {'phase': 15, 'algorithm': algorithm, 'source': str(source),
                    'source_sha256': _sha256(source),
                    'initial_update': base.completed_updates,
                    'ppo_config': asdict(base.ppo_config),
                    'network_config': asdict(base.network_config),
                    'settings': spec}
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(paths['manifest'], manifest)
        _save(session,algorithm,paths['latest'])
        shutil.copyfile(paths['latest'],paths['best'])
        anchor = load_actor(source,device=cfg.device)
        base_anchor = _evaluation(base.actor,anchor,spec,cfg.device,big_blind=base.ppo_config.big_blind)
        base_random = _evaluation(base.actor,None,spec,cfg.device,big_blind=base.ppo_config.big_blind)
        baseline = {'anchor':base_anchor,'random':base_random}
        state = {'completed_updates':base.completed_updates,
                 'best_sha256':_sha256(paths['best']),
                 'best_update':base.completed_updates, 'best_score_bb100':0.0,
                 'baseline':baseline, 'last_validation_update':base.completed_updates,
                 'entropy_warnings':0, 'kl_warnings':0, 'consecutive_safety_warnings':0,
                 'validation_count':1}
        _atomic_json(paths['state'],state)
        _event(paths['events'], {'kind':'validation','update':base.completed_updates,
                                 'score_bb100':0.0,'vs_anchor':base_anchor,
                                 'vs_random':base_random})
    # The source snapshot never mutates during this run (checked every resume).
    anchor = load_actor(manifest['source'],device=cfg.device)
    if cfg.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    deadline = started + cfg.duration_minutes*60
    updates_this_run = 0
    last_update_seconds = 0.0
    stop_reason = 'duration_limit'
    step_cfg = BatchExperimentConfig(
        algorithm=algorithm,device=cfg.device,duration_seconds=cfg.duration_minutes*60,
        max_updates=cfg.max_updates,env_batch_size=spec['env_batch_size'],
        checkpoint=str(paths['latest']),log_dir=str(root),eval_every=spec['eval_every'],
        eval_groups=spec['eval_groups'],skip_eval=True,beta=spec['beta'],
    )
    bar = (_Progress(duration_seconds=cfg.duration_minutes*60,max_updates=cfg.max_updates)
           if cfg.progress else None)
    progress_ctx = bar if bar is not None else nullcontext()
    with progress_ctx:
        event = {'kind':'resume' if cfg.resume else 'start',
                 'completed_updates':base.completed_updates,
                 'duration_seconds':cfg.duration_minutes*60,
                 'hands_per_update':spec['hands_per_update'],
                 'env_batch_size':spec['env_batch_size']}
        _event(paths['events'],event)
        if bar: bar.on_event(event)
        if emit: emit(event)
        try:
            while True:
                if cfg.max_updates is not None and updates_this_run >= cfg.max_updates:
                    stop_reason='update_limit'
                    break
                if time.perf_counter()+max(0.1, 1.1*last_update_seconds) >= deadline:
                    break
                if (base.completed_updates + 1)*spec['hands_per_update'] >= 20_000_000:
                    stop_reason='rollout_seed_namespace_limit'
                    break
                item = _one_update(session,step_cfg)
                _save(session,algorithm,paths['latest'])
                _event(paths['events'],item)
                state['completed_updates']=base.completed_updates
                updates_this_run+=1
                last_update_seconds = float(item['elapsed_seconds'])
                if bar: bar.on_event(item)
                if item.get('entropy',100) < spec['entropy_alert_threshold']:
                    state['entropy_warnings'] += 1
                    state['consecutive_safety_warnings'] += 1
                    _event(paths['events'],{'kind':'entropy_warning','update':base.completed_updates,
                                            'entropy':item['entropy']})
                elif item.get('approx_kl',0) > spec['max_approx_kl']:
                    state['kl_warnings'] += 1
                    state['consecutive_safety_warnings'] += 1
                    _event(paths['events'],{'kind':'kl_warning','update':base.completed_updates,
                                            'approx_kl':item['approx_kl']})
                else:
                    state['consecutive_safety_warnings'] = 0
                _atomic_json(paths['state'],state)
                if base.completed_updates % spec['eval_every'] == 0:
                    if time.perf_counter() < deadline:
                        v = _validation(base,anchor,spec,cfg.device,state['baseline'])
                        if v['score_bb100'] > state['best_score_bb100']:
                            temp=paths['best'].with_name('best.pt.tmp')
                            shutil.copyfile(paths['latest'],temp)
                            os.replace(temp,paths['best'])
                            state['best_score_bb100']=v['score_bb100']
                            state['best_update']=base.completed_updates
                            state['best_sha256']=_sha256(paths['best'])
                        state['last_validation_update']=base.completed_updates
                        state['validation_count']+=1
                        _atomic_json(paths['state'],state)
                        _event(paths['events'],v,emit)
                if state['consecutive_safety_warnings'] >= spec['entropy_stop_patience']:
                    stop_reason='safety_stop'
                    break
            # Evaluate last saved policy, even if the invocation stopped between periodic checks.
            if (updates_this_run and state['last_validation_update'] != base.completed_updates
                    and time.perf_counter() < deadline):
                v=_validation(base,anchor,spec,cfg.device,state['baseline'])
                if v['score_bb100'] > state['best_score_bb100']:
                    tmp=paths['best'].with_name('best.pt.tmp')
                    shutil.copyfile(paths['latest'],tmp)
                    os.replace(tmp,paths['best'])
                    state['best_score_bb100']=v['score_bb100']
                    state['best_update']=base.completed_updates
                    state['best_sha256']=_sha256(paths['best'])
                state['last_validation_update']=base.completed_updates
                state['validation_count']+=1
                _atomic_json(paths['state'],state)
                _event(paths['events'],v,emit)
        except KeyboardInterrupt:
            stop_reason='keyboard_interrupt'
            # A mid-update interruption never overwrites the previous atomic checkpoint.
            state['completed_updates']=int(torch.load(paths['latest'],weights_only=True,map_location='cpu')['completed_updates'])
            _atomic_json(paths['state'],state)
        if bar: bar.on_event({'kind':'end','stop_reason':stop_reason})
    elapsed=time.perf_counter()-started
    peak = (round(torch.cuda.max_memory_allocated()/2**20,3) if cfg.device=='cuda' else None)
    summary=_summary(root,manifest,state,stop_reason=stop_reason,
                     updates_this_run=updates_this_run,elapsed=elapsed,peak=peak)
    _atomic_json(paths['summary'],summary)
    _event(paths['events'],{'kind':'end',**summary},emit)
    return summary


def _export_curve(root: Path) -> None:
    events=root/'metrics.jsonl'
    rows=[json.loads(line) for line in events.read_text(encoding='utf-8').splitlines() if line.strip()]
    for kind,cols,filename in (
        ('update',('update','hands','elapsed_seconds','hands_per_second_training','entropy',
                   'critic_loss','approx_kl','logit_variance_ratio'), 'training_curve.csv'),
        ('validation',('update','score_bb100','gain_vs_anchor_bb100','gain_vs_random_bb100'),
         'validation_curve.csv')):
        with (root/filename).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=cols)
            writer.writeheader()
            for row in rows:
                if row.get('kind')==kind:
                    writer.writerow({key:row.get(key) for key in cols})


def _finalize_longrun_impl(root: str | Path, *, device: str = 'cpu') -> dict:
    """ONE held-out evaluation of best.pt; seals the run against further resume.

    Calling again verifies checkpoint SHA and returns the stored report.
    """
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    folder=Path(root)
    manifest,state,paths=_read(folder)
    _match_resume(LongRunConfig(root=str(folder),resume=True),manifest)
    best_sha=_sha256(paths['best'])
    if paths['final'].exists():
        cached=json.loads(paths['final'].read_text(encoding='utf-8'))
        if cached['best_sha256'] != best_sha:
            raise ValueError('best checkpoint has changed since final evaluation')
        return cached
    model=load_actor(paths['best'],device=device)
    anchor=load_actor(manifest['source'],device=device)
    spec=manifest['settings']
    big_blind=manifest['ppo_config']['big_blind']
    against_anchor=_evaluation(model,anchor,spec,device,final=True,big_blind=big_blind)
    against_random=_evaluation(model,None,spec,device,final=True,big_blind=big_blind)
    result={'phase':15,'partition':'final_held_out','algorithm':manifest['algorithm'],
            'best_checkpoint':str(paths['best']),'best_sha256':best_sha,
            'best_update':state['best_update'], 'initial_update':manifest['initial_update'],
            'validation_score_bb100':state['best_score_bb100'],
            'vs_frozen_initial':against_anchor,'vs_random':against_random,
            'final_seed':spec['final_seed'],'final_groups':spec['final_groups'],
            'note':'Test used only after checkpoint selection. Not GTO exploitability.'}
    _export_curve(folder)
    _atomic_json(paths['final'],result)  # Commit seal LAST after successful evaluation.
    return result


def run_longrun(cfg: LongRunConfig, *, emit: Callable[[dict],None] | None = None) -> dict:
    """OS-locked entry point for the crash-safe long-running training loop."""
    with _exclusive_lock(Path(cfg.root)):
        return _run_longrun_impl(cfg,emit=emit)


def finalize_longrun(root: str | Path, *, device: str = 'cpu') -> dict:
    """OS-locked held-out evaluation; no training can run concurrently."""
    with _exclusive_lock(Path(root)):
        return _finalize_longrun_impl(root,device=device)
