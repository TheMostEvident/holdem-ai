"""Phase14 correctness, matched-start and held-out test protection."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
import torch

from holdem_agent_phase8 import NetworkConfig
from holdem_agent_phase9 import PPOConfig, create_session, save_checkpoint
from holdem_agent_phase14 import ComparisonConfig, run_study, seed_level_summary
from scripts.phase14_study import build_parser, parse_seeds


def _source(tmp_path):
    path = tmp_path / 'source_phase9.pt'
    session = create_session(device='cpu', seed=42,
        network_config=NetworkConfig(embed_dim=16, attention_heads=2,
            transformer_layers=1, ff_dim=32, hidden_dim=24, max_tokens=256),
        ppo_config=PPOConfig(hands_per_update=3, epochs=1, minibatch_size=4))
    save_checkpoint(session, path)
    return path


def _config(tmp_path, source, **kwargs):
    return replace(ComparisonConfig(source_checkpoint=str(source),
        root=str(tmp_path / 'phase14'), seeds=(2,), updates=1,
        hands_per_update=4, env_batch_size=2, minibatch_size=4, epochs=1,
        eval_every=1, validation_groups=2, test_groups=2,
        arm_duration_minutes=1, progress=False), **kwargs)


@pytest.mark.parametrize('seeds', [(1,1),(),(-1,), (100000,)])
def test_reject_invalid_seeds(seeds):
    with pytest.raises(ValueError):
        ComparisonConfig('whatever.pt', seeds=seeds)


def test_disallow_test_overlap():
    with pytest.raises(ValueError):
        ComparisonConfig('anything.pt', validation_seed=100,
                         validation_groups=10, test_seed=105, test_groups=4)


def test_statistic_seed_level_n1_and_t():
    s = seed_level_summary([10.0]); assert s['ci95_lower_bb100'] is None
    s = seed_level_summary([2.,4.,6.]); assert s['seeds']==3
    assert s['mean_delta_bb100']==4
    assert s['ci95_lower_bb100'] < 0 < s['ci95_upper_bb100']
    assert seed_level_summary([-2.0,-2.0])['ci95_upper_bb100'] == -2.0
    with pytest.raises(ValueError): seed_level_summary([float('nan')])


def test_seed_cli_and_defaults():
    assert parse_seeds('4, 5,6') == (4,5,6)
    for x in ('', '1,1', 'hi', '1,-1'):
        with pytest.raises(Exception): parse_seeds(x)
    args = build_parser().parse_args(['--source-checkpoint','x.pt','--seeds','4,5'])
    assert args.seeds==(4,5) and args.hands_per_update==64


def test_reject_non_phase9_checkpoint(tmp_path):
    path=tmp_path/'wrong.pt'; torch.save({'phase':10},path)
    with pytest.raises(ValueError): run_study(_config(tmp_path,path))


def test_end_to_end_matched_start_and_no_test_leakage(tmp_path, monkeypatch):
    source = _source(tmp_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cfg = _config(tmp_path,source)
    from holdem_agent_phase14 import study as module
    original_eval = module._eval
    seen=[]
    def spy(actor,opponent,config,*,partition):
        seen.append(partition)
        return original_eval(actor,opponent,config,partition=partition)
    monkeypatch.setattr(module,'_eval', spy)
    output = run_study(cfg)
    assert output['status']=='complete'
    assert output['total_training_hands']==8
    assert len(output['arms'])==2
    assert output['arms'][0]['finished'] and output['arms'][1]['finished']
    assert source.read_bytes() and hashlib.sha256(source.read_bytes()).hexdigest()==digest
    assert seen[:4]==['validation']*4
    assert 'test' not in seen[:4] and len([x for x in seen if x=='test'])==5
    assert output['primary_qboost_minus_ppo_vs_initial']['seeds']==1
    assert output['primary_qboost_minus_ppo_vs_initial']['ci95_lower_bb100'] is None
    assert output['seed_results'][0]['qboost_vs_ppo_headtohead']['hands']==8
    study=json.loads((tmp_path/'phase14'/'study.json').read_text())
    assert study['source_sha256']==digest
    meta_ppo=json.loads((tmp_path/'phase14'/'seed_2'/'ppo'/'run.json').read_text())
    meta_qb=json.loads((tmp_path/'phase14'/'seed_2'/'qboost'/'run.json').read_text())
    assert meta_ppo['ppo_config']==meta_qb['ppo_config']
    assert meta_ppo['source_fingerprint']==meta_qb['source_fingerprint']
    assert meta_ppo['ppo_config']['hands_per_update']==4
    assert (tmp_path/'phase14'/'seed_2'/'ppo'/'best.pt').exists()
    assert (tmp_path/'phase14'/'seed_2'/'qboost'/'best.pt').exists()
    # Running the same study as a new experiment is intentionally blocked.
    with pytest.raises(FileExistsError): run_study(cfg)
    repeated=run_study(replace(cfg,resume=True))
    assert repeated['status']=='complete'
    assert repeated['primary_qboost_minus_ppo_vs_initial']==output['primary_qboost_minus_ppo_vs_initial']


def test_resume_rejects_seed_and_source_change(tmp_path):
    source=_source(tmp_path)
    cfg=_config(tmp_path,source)
    run_study(cfg)
    with pytest.raises(ValueError): run_study(replace(cfg,resume=True,updates=2))
    with pytest.raises(ValueError): run_study(replace(cfg,resume=True,seeds=(3,)))


def test_resumable_partial_study(tmp_path, monkeypatch):
    source = _source(tmp_path)
    cfg = _config(tmp_path,source, updates=2)
    from holdem_agent_phase14 import study as m
    original_one=m._one_update
    calls={'n':0}
    def stop_after_first(sess, config):
        calls['n']+=1
        if calls['n']==2: raise KeyboardInterrupt
        return original_one(sess,config)
    monkeypatch.setattr(m,'_one_update',stop_after_first)
    result=run_study(cfg)
    assert result['status']=='partial'
    assert not (tmp_path/'phase14'/'seed_2'/'test_results.json').exists()
    monkeypatch.setattr(m,'_one_update',original_one)
    result=run_study(replace(cfg,resume=True))
    assert result['status']=='complete'
    assert result['arms'][0]['completed_updates']==2
    assert result['arms'][1]['completed_updates']==2
    assert (tmp_path/'phase14'/'seed_2'/'test_results.json').exists()
