from dataclasses import replace
import json
import pytest
import torch
from holdem_ai.expand.rl import (ActorCritic, RolloutConfig, PPOConfig, OpponentMix)
from holdem_ai.expand.rl.trainer import (TrainConfig, train,load_trained_actor,
                                          evaluate_checkpoint, config_from_dict)
from e03_fake_engine import FastFakeEngine


def config():
    return TrainConfig(width=16,heads=2,layers=1,seed=11,rollout_seed=320000001,
                       validation_seed=330000001,eval_every=1,eval_hands=2,
                       snapshot_every=1,max_snapshots=2,
                       rollout=RolloutConfig(hands_per_update=2,num_players=(2,3),
                           mix=OpponentMix(shared_policy_fraction=0.2,snapshot_fraction=0.5)),
                       ppo=PPOConfig(epochs=1,minibatch_size=8))


def test_e03_checkpoint_resume_snapshot_validation(tmp_path):
    root=tmp_path/'a'
    a=train(root,config=config(),additional_updates=1,engine_factory=FastFakeEngine)
    assert a['completed_updates']==1
    assert (root/'latest.pt').exists() and (root/'best.pt').exists()
    assert (root/'snapshots/update_0000001.pt').exists()
    assert (root/'validation_anchor.json').exists()
    assert (root/'run.json').exists()
    b=train(root,resume=True,additional_updates=1,engine_factory=FastFakeEngine)
    assert b['completed_updates']==2
    assert (root/'snapshots/update_0000002.pt').exists()
    assert load_trained_actor(root/'latest.pt').actor.width==16
    assert load_trained_actor(root/'best.pt').actor.width==16
    assert len((root/'metrics.jsonl').read_text().strip().splitlines())==3
    with pytest.raises(FileExistsError):
        train(root,config=config(),additional_updates=1,engine_factory=FastFakeEngine)
    with pytest.raises(ValueError,match='resume config'):
        train(root,config=replace(config(),eval_hands=3),resume=True,additional_updates=1,
              engine_factory=FastFakeEngine)
    results=evaluate_checkpoint(root/'best.pt',hands=2,seed=340000000,
             num_players=(2,3),opponents=('random',),engine_factory=FastFakeEngine)
    assert len(results['results'])==2
    assert results['results']['3p_random']['hands']==2


def test_e03_resume_exact_reproducible(tmp_path):
    cfg=config()
    first=tmp_path/'first'
    second=tmp_path/'second'
    train(first,config=cfg,additional_updates=2,engine_factory=FastFakeEngine)
    train(second,config=cfg,additional_updates=1,engine_factory=FastFakeEngine)
    train(second,resume=True,additional_updates=1,engine_factory=FastFakeEngine)
    x=torch.load(first/'latest.pt',weights_only=True)
    y=torch.load(second/'latest.pt',weights_only=True)
    assert x['completed_updates']==y['completed_updates']==2
    assert all(torch.equal(v,y['model'][k]) for k,v in x['model'].items())
    assert x['snapshots']==y['snapshots']


def test_e03_checkpoint_tamper_rejected(tmp_path):
    root=tmp_path/'tamper'
    train(root,config=config(),additional_updates=1,engine_factory=FastFakeEngine)
    (root/'best.pt').write_bytes(b'fake best model')
    with pytest.raises(ValueError,match='hash'):
        train(root,resume=True,additional_updates=1,engine_factory=FastFakeEngine)
