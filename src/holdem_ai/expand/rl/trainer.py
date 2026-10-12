"""Reproducible E3 PPO training, snapshots, safe resume and validation gates.

All training/evaluation card seeds are separated. Never report E3 as a Pluribus
blueprint or as evidence of converged multi-player equilibrium strategies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Callable

import torch

from .collector import MultiAgentCollector, RolloutConfig
from .model import ActorCritic
from .population import OpponentMix
from .ppo import PPOConfig, optimize_ppo
from .returns import prepare_gae


@dataclass(frozen=True)
class TrainConfig:
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    width: int = 64
    heads: int = 4
    layers: int = 2
    seed: int = 303000
    rollout_seed: int = 320000000
    validation_seed: int = 330000000
    eval_every: int = 2
    eval_hands: int = 8
    snapshot_every: int = 2
    max_snapshots: int = 3
    gamma: float = 1.0
    gae_lambda: float = 0.95

    def __post_init__(self):
        if self.width < 16 or self.heads < 1 or self.width % self.heads or self.layers < 1:
            raise ValueError('invalid actor network configuration')
        if any(type(x) is not int or x < 0 for x in (self.seed,self.rollout_seed,self.validation_seed)):
            raise ValueError('nonnegative seeds required')
        if self.eval_every <= 0 or self.eval_hands < 1 or self.snapshot_every <= 0 or self.max_snapshots < 0:
            raise ValueError('invalid eval/snapshot cadence')
        if not (0<=self.gamma<=1 and 0<=self.gae_lambda<=1):
            raise ValueError('invalid GAE coefficients')
        # Validation bank and training bank must not overlap for up to 1M updates.
        if abs(self.rollout_seed-self.validation_seed) < 1_000_000:
            raise ValueError('rollout and validation seed domains too close')


def config_from_dict(data: dict) -> TrainConfig:
    rd=dict(data['rollout'])
    rd['mix']=OpponentMix(**{**rd['mix'],'scripted_names':tuple(rd['mix']['scripted_names'])})
    rd['num_players']=tuple(rd['num_players'])
    if rd['random_stack_range_bb'] is not None:
        rd['random_stack_range_bb']=tuple(rd['random_stack_range_bb'])
    return TrainConfig(**{**data,'rollout':RolloutConfig(**rd),'ppo':PPOConfig(**data['ppo'])})


def _atomic_json(path: Path, record: Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp,path)


def _atomic_pt(path: Path, data: dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:
        torch.save(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp,path)


def _hash(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):
            h.update(x)
    return h.hexdigest()


def _log(path: Path, data: dict):
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def build_model(cfg: TrainConfig, device: str) -> ActorCritic:
    if device not in ('cpu','cuda'):
        raise ValueError('device cpu or cuda required')
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    return ActorCritic(width=cfg.width,heads=cfg.heads,layers=cfg.layers).to(device)


def _checkpoint(model,optimizer,cfg,update, snapshots,best_score, best_update):
    return {'format':'expand-e03-v1','model':model.state_dict(),
            'optimizer':optimizer.state_dict(),'config':asdict(cfg),
            'completed_updates':update,'snapshots':list(snapshots),
            'best_validation_score':best_score,'best_update':best_update,
            'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def load_trained_actor(path: str | Path, *, device='cpu') -> ActorCritic:
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('format') not in ('expand-e03-v1','expand-e03-snapshot-v1'):
        raise ValueError('not an Expand Phase3 trusted checkpoint')
    cfg=config_from_dict(payload['config'])
    model=build_model(cfg,device)
    model.load_state_dict(payload['model'])
    return model.eval()


def _evaluate(model: ActorCritic, cfg: TrainConfig, *, device: str,
              num_hands: int | None = None, seed: int | None = None,
              engine_factory: Callable | None = None) -> dict:
    """Frozen random/calling_station anchor; no training transitions recorded."""
    scores={}
    for j,name in enumerate(('random','calling_station')):
        collector=MultiAgentCollector(model,cfg.rollout,engine_factory=engine_factory)
        result=collector.collect(seed=(cfg.validation_seed+j*100000 if seed is None else seed+j*100000),
                                  policy_version='eval',num_hands=(num_hands or cfg.eval_hands),
                                  force_opponent=name,record_training=False)
        scores[name]=round(result.learner_payoff_bb/result.total_hands*100,5)
    return scores


def _train_impl(root: str | Path, *, config: TrainConfig | None = None, resume: bool = False,
          additional_updates: int = 3, device: str = 'cpu',
          engine_factory: Callable | None = None, emit: Callable[[dict],None] | None = None) -> dict:
    if type(additional_updates) is not int or additional_updates < 1:
        raise ValueError('additional_updates >=1 required')
    folder=Path(root)
    folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'run.json'
    latest=folder/'latest.pt'
    best=folder/'best.pt'
    state_file=folder/'state.json'
    logfile=folder/'metrics.jsonl'
    if resume:
        if not all(x.exists() for x in (manifest,latest,state_file,logfile,best)):
            raise FileNotFoundError('incomplete run; no resume possible')
        record=json.loads(manifest.read_text(encoding='utf-8'))
        cfg=config_from_dict(record['config'])
        if config is not None and asdict(cfg)!=asdict(config):
            raise ValueError('resume config differs from run.json')
        blob=torch.load(latest,map_location='cpu',weights_only=True)
        if blob.get('format')!='expand-e03-v1' or blob['config']!=asdict(cfg):
            raise ValueError('checkpoint manifest mismatch')
        # Verify validation-selected best does not change silently.
        saved_state=json.loads(state_file.read_text(encoding='utf-8'))
        if saved_state['best_sha256']!=_hash(best):
            raise ValueError('best checkpoint hash differs from stored state')
    else:
        if any(p.name != '.expand03.lock' for p in folder.iterdir()):
            raise FileExistsError('fresh training root must be empty; use --resume or a new folder')
        cfg=config or TrainConfig()
        torch.manual_seed(cfg.seed)
        if device=='cuda':torch.cuda.manual_seed_all(cfg.seed)
        _atomic_json(manifest,{'phase':'expand-03','format':'expand-e03-v1','config':asdict(cfg)})
        blob=None
    model=build_model(cfg,device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.ppo.learning_rate)
    if resume:
        model.load_state_dict(blob['model'])
        optimizer.load_state_dict(blob['optimizer'])
        # Map optimizer tensors to actual device (cpu checkpoint -> CUDA resume).
        for group in optimizer.state.values():
            for key,val in group.items():
                if torch.is_tensor(val):group[key]=val.to(device)
        torch.set_rng_state(blob['torch_rng'])
        if device=='cuda' and blob.get('cuda_rng') and len(blob['cuda_rng'])==torch.cuda.device_count():
            torch.cuda.set_rng_state_all(blob['cuda_rng'])
        updates=int(blob['completed_updates'])
        snapshots=list(blob['snapshots'])
        best_score=float(blob['best_validation_score'])
        best_update=int(blob['best_update'])
    else:
        updates=0
        snapshots=[]
        baseline=_evaluate(model,cfg,device=device,engine_factory=engine_factory)
        best_score=0.0
        best_update=0
        _atomic_json(folder/'validation_anchor.json',baseline)
        initial=_checkpoint(model,optimizer,cfg,updates,snapshots,best_score,best_update)
        _atomic_pt(latest,initial)
        _atomic_pt(best,initial)
        _atomic_json(state_file,{'completed_updates':0,'best_update':0,
                                 'best_score_bb100':0.0,'best_sha256':_hash(best)})
        _log(logfile,{'kind':'baseline','update':0,'vs_opponent_bb100':baseline})
    baseline=json.loads((folder/'validation_anchor.json').read_text(encoding='utf-8'))
    since=time.perf_counter()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    run_history=[]
    for _ in range(additional_updates):
        next_u=updates+1
        frozen=[]
        for p in snapshots:
            path=folder/p
            if not path.is_file():raise FileNotFoundError('missing frozen snapshot '+p)
            frozen.append(load_trained_actor(path,device=device))
        collector=MultiAgentCollector(model,cfg.rollout,engine_factory=engine_factory)
        data=collector.collect(seed=cfg.rollout_seed+next_u,policy_version=f'update-{updates}',
                               snapshots=tuple(frozen))
        if not data.transitions:
            raise RuntimeError('no trainable decisions from collected hands')
        samples=prepare_gae(data.transitions,gamma=cfg.gamma,lam=cfg.gae_lambda)
        stats=optimize_ppo(model,optimizer,samples,cfg.ppo)
        updates=next_u
        validated=None
        if updates % cfg.eval_every == 0:
            current=_evaluate(model,cfg,device=device,engine_factory=engine_factory)
            gain=sum(current[k]-baseline[k] for k in ('random','calling_station'))/2
            validated={'current':current,'score_delta_bb100':round(gain,5)}
            if gain>best_score:
                best_score=gain
                best_update=updates
                best_blob=_checkpoint(model,optimizer,cfg,updates,snapshots,best_score,best_update)
                _atomic_pt(best,best_blob)
        if cfg.max_snapshots and updates%cfg.snapshot_every==0:
            snap_dir=folder/'snapshots'
            snap_dir.mkdir(exist_ok=True)
            rel=f'snapshots/update_{updates:07d}.pt'
            _atomic_pt(folder/rel,{'format':'expand-e03-snapshot-v1','config':asdict(cfg),
                                   'model':model.state_dict(),'completed_updates':updates})
            snapshots.append(rel)
            if len(snapshots)>cfg.max_snapshots:
                # Old snapshots intentionally retained on disk for audit. They
                # are only removed from active opponent sampling population.
                snapshots=snapshots[-cfg.max_snapshots:]
        blob=_checkpoint(model,optimizer,cfg,updates,snapshots,best_score,best_update)
        _atomic_pt(latest,blob)
        _atomic_json(state_file,{'completed_updates':updates,'best_update':best_update,
                                 'best_score_bb100':round(best_score,5),'best_sha256':_hash(best)})
        row={'kind':'update','update':updates,'hands':data.total_hands,
             'num_players_histogram':data.hands_by_n,'warmup_hands':data.warmup_hands,
             'learner_payoff_bb':round(data.learner_payoff_bb,5),
             'illegal_actions':data.illegal_actions,**stats}
        if validated is not None:row['validation']=validated
        _log(logfile,row)
        run_history.append(row)
        if emit:emit(row)
    summary={'phase':'expand-03','completed_updates':updates,
             'updates_this_call':additional_updates,'best_update':best_update,
             'best_score_delta_bb100':round(best_score,5),
             'latest_checkpoint':str(latest),'best_checkpoint':str(best),
             'duration_seconds':round(time.perf_counter()-since,3),
             'cuda_peak_allocated_mb':round(torch.cuda.max_memory_allocated()/2**20,3) if device=='cuda' else None,
             'last_update':run_history[-1]}
    _atomic_json(folder/'summary.json',summary)
    return summary


def evaluate_checkpoint(checkpoint: str | Path, *, device: str='cpu',
                        hands: int=100, seed: int=340000000,
                        opponents: tuple[str,...]=('random','calling_station','tight','pot_pressure'),
                        num_players: tuple[int,...]=(2,3,4,5,6),
                        engine_factory: Callable | None = None) -> dict:
    if type(hands) is not int or hands < 1 or type(seed) is not int or seed < 0:
        raise ValueError('invalid evaluation settings')
    actor=load_trained_actor(checkpoint,device=device)
    raw=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=config_from_dict(raw['config'])
    results={}
    for n in num_players:
        for j,name in enumerate(opponents):
            if name not in ('random','calling_station','tight','pot_pressure'):
                raise ValueError('unknown eval opponent')
            collector=MultiAgentCollector(actor,cfg.rollout,engine_factory=engine_factory)
            outcome=collector.collect(seed=seed+int(n)*1000000+j*100000,
                          policy_version='heldout',num_hands=hands,force_n=int(n),
                          force_opponent=name,record_training=False)
            scores=[r[0] for r in outcome.hand_rewards]
            mean=100*sum(scores)/len(scores)
            if len(scores)>1:
                variance=sum((x-sum(scores)/len(scores))**2 for x in scores)/(len(scores)-1)
                se=100*math.sqrt(variance/len(scores))
            else:se=None
            results[f'{n}p_{name}']={'hands':len(scores),'mean_bb100':round(mean,5),
                 'ci95_normal':None if se is None else [round(mean-1.96*se,5),round(mean+1.96*se,5)]}
    return {'phase':'expand-03','checkpoint':str(checkpoint),'seed':seed,
            'hands_per_cell':hands,'results':results,
            'caution':'Hand-level normal CI is exploratory, not exploitability or multi-seed proof.'}


@contextmanager
def _exclusive_lock(root: Path):
    """OS-level advisory lock for one training run; .expand03.lock may persist."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.expand03.lock').open('a+b') as handle:
        handle.seek(0)
        handle.write(b'\0')
        handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Expand3 run directory is already locked') from exc
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
                raise RuntimeError('Expand3 run directory is already locked') from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def train(root: str | Path, *, config: TrainConfig | None = None, resume: bool = False,
          additional_updates: int = 3, device: str = 'cpu',
          engine_factory: Callable | None = None, emit: Callable[[dict],None] | None = None) -> dict:
    """Train one root with serialized writes and immutable resume settings."""
    with _exclusive_lock(Path(root)):
        return _train_impl(root, config=config, resume=resume,
                           additional_updates=additional_updates, device=device,
                           engine_factory=engine_factory, emit=emit)
