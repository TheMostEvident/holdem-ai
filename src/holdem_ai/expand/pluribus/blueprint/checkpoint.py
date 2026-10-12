"""Atomic gzipped JSON MCCFR checkpoints; no pickle or torch.load."""
from __future__ import annotations
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
from .mccfr import MCCFRSolver, MCCFRConfig, InfosetStore
from .toy_games import KuhnGame, LeducGame
from .holdem import HoldemGame
from ..abstraction import AbstractionConfig


def describe_game(game):
    data = {'name':game.name, 'players':game.num_players}
    if game.name == 'holdem':
        data.update({'big_blind':game.big_blind,'stack_bb':game.stack_bb,
                     'abstraction':asdict(game.abstraction),
                     'abstraction_id':game.abstraction.fingerprint})
    return data


def build_game(spec, engine_factory=None):
    if spec['name'] == 'kuhn':
        return KuhnGame(spec['players'])
    if spec['name'] == 'leduc':
        return LeducGame(spec['players'])
    if spec['name'] == 'holdem':
        abstraction = AbstractionConfig(**spec['abstraction'])
        if spec['abstraction_id'] != abstraction.fingerprint:
            raise ValueError('card/action abstraction fingerprint mismatch')
        return HoldemGame(spec['players'],big_blind=spec['big_blind'],
                          stack_bb=spec['stack_bb'],abstraction=abstraction,
                          engine_factory=engine_factory)
    raise ValueError('unsupported game in checkpoint')


def save_checkpoint(path, solver: MCCFRSolver):
    file = Path(path)
    file.parent.mkdir(parents=True,exist_ok=True)
    payload = {'phase':'expand04','schema':1,
               'game':describe_game(solver.game),
               'config':asdict(solver.config),
               'completed_iterations':solver.completed_iterations,
               'entries':solver.store.as_dict()}
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    outer = {'sha256':hashlib.sha256(raw).hexdigest(),'payload':payload}
    temp = file.with_name(file.name + '.tmp')
    with temp.open('wb') as handle:
        with gzip.GzipFile(filename='',mode='wb',fileobj=handle,mtime=0) as archive:
            archive.write(json.dumps(outer,sort_keys=True,separators=(',', ':'),allow_nan=False).encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp,file)
    return {'path':str(file),'sha256':hashlib.sha256(raw).hexdigest(),
            'completed_iterations':solver.completed_iterations,'infosets':len(solver.store.entries)}


def load_checkpoint(path, engine_factory=None):
    with gzip.open(path,'rt',encoding='utf-8') as handle:
        outer=json.load(handle)
    payload=outer['payload']
    raw=json.dumps(payload,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()
    if hashlib.sha256(raw).hexdigest()!=outer['sha256']:
        raise ValueError('checksum mismatch: checkpoint tampered or corrupt')
    if payload.get('phase')!='expand04' or payload.get('schema') != 1:
        raise ValueError('incompatible checkpoint schema')
    game=build_game(payload['game'], engine_factory=engine_factory)
    cfg=MCCFRConfig(**payload['config'])
    table=InfosetStore.from_dict(payload['entries'])
    if len(table.entries)>cfg.max_infosets:
        raise ValueError('checkpoint exceeds memory/infoset guard')
    return MCCFRSolver(game,config=cfg,store=table,
                       completed_iterations=payload['completed_iterations'])
