import copy
import gzip
import json
from pathlib import Path
import pytest
from holdem_ai.expand.pluribus.blueprint import (KuhnGame,LeducGame,MCCFRConfig,MCCFRSolver,
                                                   save_checkpoint,load_checkpoint)


def test_resume_exact_weights_and_round(tmp_path):
    a=MCCFRSolver(KuhnGame(2),MCCFRConfig(seed=9))
    b=MCCFRSolver(KuhnGame(2),MCCFRConfig(seed=9))
    a.train(200)
    b.train(100)
    target=tmp_path/'model.json.gz'
    save_checkpoint(target,b)
    restored=load_checkpoint(target)
    restored.train(100)
    assert a.completed_iterations==restored.completed_iterations
    assert a.store.as_dict()==restored.store.as_dict()


def test_checkpoint_roundtrip_leduc(tmp_path):
    s=MCCFRSolver(LeducGame(3),MCCFRConfig(seed=20))
    s.train(40)
    path=tmp_path/'l.json.gz'
    one=save_checkpoint(path,s)
    assert len(one['sha256'])==64
    loaded=load_checkpoint(path)
    assert loaded.game.num_players==3 and loaded.game.name=='leduc'
    assert loaded.store.as_dict()==s.store.as_dict()


def test_checksum_detects_tampering(tmp_path):
    s=MCCFRSolver(KuhnGame(2))
    s.train(5)
    path=tmp_path/'bad.json.gz'
    save_checkpoint(path,s)
    with gzip.open(path,'rt') as stream:content=json.load(stream)
    content['payload']['completed_iterations']=9999
    with gzip.open(path,'wt') as stream:json.dump(content,stream)
    with pytest.raises(ValueError,match='checksum'):
        load_checkpoint(path)


def test_checkpoint_incompatible_actions_rejected(tmp_path):
    s=MCCFRSolver(KuhnGame(2))
    s.train(8)
    path=tmp_path/'test.json.gz'
    save_checkpoint(path,s)
    with gzip.open(path,'rt') as stream:outer=json.load(stream)
    first=next(iter(outer['payload']['entries']))
    outer['payload']['entries'][first]['actions']=['fold']
    import hashlib
    payload=outer['payload']
    raw=json.dumps(payload,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()
    outer['sha256']=hashlib.sha256(raw).hexdigest()
    with gzip.open(path,'wt') as stream:json.dump(outer,stream)
    with pytest.raises(ValueError,match='array lengths'):
        load_checkpoint(path)
