"""Phase15 safety, resume, validation selection, held-out sealing and CLI."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
import torch

from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import PPOConfig
from holdem_ai.training.ppo_session import create_session, save_checkpoint
from holdem_ai.experiments.longrun import LongRunConfig, run_longrun, finalize_longrun
from holdem_ai.experiments.longrun import _spec, _match_resume, _summary
from scripts.phase15_train import build_parser


def _source(tmp_path):
    path=tmp_path/'start_ppo.pt'
    sess=create_session(device='cpu',seed=3,
        network_config=NetworkConfig(embed_dim=16, attention_heads=2,
            transformer_layers=1,ff_dim=32,hidden_dim=24,max_tokens=256),
        ppo_config=PPOConfig(hands_per_update=4,epochs=1,minibatch_size=4))
    save_checkpoint(sess,path)
    return path


def _config(tmp_path, source, **changes):
    return replace(LongRunConfig(root=str(tmp_path/'longrun'),source_checkpoint=str(source),
       algorithm='ppo', duration_minutes=2,max_updates=1,progress=False,
       hands_per_update=4,env_batch_size=2,minibatch_size=4,epochs=1,
       eval_every=1,eval_groups=2,final_groups=2,rollout_seed=400_000_000),**changes)


@pytest.mark.parametrize('changes',[
    {'duration_minutes':0}, {'max_updates':0},{'algorithm':'invalid'},
    {'source_checkpoint':None}, {'env_batch_size':5},
    {'rollout_seed':-1},{'eval_groups':1},{'eval_mode':'stochastic'},
    {'final_seed':15_000_001}, {'validation_seed':400_000_000},
    {'max_approx_kl':0}, {'entropy_stop_patience':0},
])
def test_bad_config(tmp_path,changes):
    with pytest.raises((ValueError,FileNotFoundError)):
        cfg=_config(tmp_path,tmp_path/'dummy.pt',**changes)
        if changes.keys() & {'duration_minutes','max_updates','algorithm','source_checkpoint'}:
            run_longrun(cfg)
        else:
            _spec(cfg)


def test_cli_and_defaults():
    args=build_parser().parse_args(['--source-checkpoint','x.pt','--algorithm','ppo'])
    assert args.duration_minutes==300 and args.hands_per_update is None
    resume=build_parser().parse_args(['--root','r','--resume'])
    assert resume.source_checkpoint is None and resume.algorithm is None
    assert _spec(LongRunConfig(algorithm='ppo',source_checkpoint='x'))['eval_groups']==100


def test_train_resume_and_immutable_source(tmp_path):
    source=_source(tmp_path)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    cfg=_config(tmp_path,source)
    first=run_longrun(cfg)
    assert first['completed_updates']==1
    assert first['total_training_hands']==4
    assert first['best_validation_score_bb100']>=0
    assert first['stop_reason']=='update_limit'
    root=tmp_path/'longrun'
    assert (root/'run.json').is_file() and (root/'best.pt').is_file()
    assert (root/'metrics.jsonl').is_file()
    manifest=json.loads((root/'run.json').read_text())
    assert manifest['ppo_config']['hands_per_update']==4
    assert manifest['ppo_config']['seed']==400_000_000
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    second=run_longrun(LongRunConfig(root=str(root),resume=True,duration_minutes=2,max_updates=1,progress=False))
    assert second['completed_updates']==2
    assert second['total_training_hands']==8
    assert second['updates_this_run']==1
    assert second['best_validation_score_bb100']>=0
    with pytest.raises(FileExistsError): run_longrun(cfg)
    with pytest.raises(ValueError):
        run_longrun(LongRunConfig(root=str(root),resume=True,duration_minutes=2,max_updates=1,
                                 hands_per_update=128,progress=False))


def test_finalize_seals_and_does_not_retest(tmp_path,monkeypatch):
    source=_source(tmp_path)
    cfg=_config(tmp_path,source)
    run_longrun(cfg)
    root=tmp_path/'longrun'
    from holdem_ai.experiments import longrun as m
    original=m._evaluation
    seen=[]
    def spy(*args,**kw):
        seen.append(kw.get('final',False))
        return original(*args,**kw)
    monkeypatch.setattr(m,'_evaluation',spy)
    result=finalize_longrun(root)
    assert result['phase']==15 and result['partition']=='final_held_out'
    assert seen == [True,True]
    assert result['vs_frozen_initial']['hands']==8
    assert result['vs_random']['hands']==8
    assert (root/'training_curve.csv').is_file()
    assert (root/'validation_curve.csv').is_file()
    cached=finalize_longrun(root)
    assert cached==result and seen==[True,True]
    with pytest.raises(RuntimeError):
        run_longrun(LongRunConfig(root=str(root),resume=True,progress=False,max_updates=1))
    # Tamper detection on immutable selected checkpoint.
    (root/'best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError): finalize_longrun(root)


def test_source_and_checkpoint_tampering_rejected(tmp_path):
    source=_source(tmp_path)
    run_longrun(_config(tmp_path,source))
    source.write_bytes(b'changed')
    with pytest.raises(ValueError):
        run_longrun(LongRunConfig(root=str(tmp_path/'longrun'),resume=True,
                                  duration_minutes=2,max_updates=1,progress=False))


def test_keyboard_interrupt_mid_update_does_not_commit(tmp_path,monkeypatch):
    source=_source(tmp_path)
    from holdem_ai.experiments import longrun as m
    def interrupt(*args,**kw): raise KeyboardInterrupt
    with monkeypatch.context() as p:
        p.setattr(m,'_one_update',interrupt)
        result=run_longrun(_config(tmp_path,source))
    assert result['stop_reason']=='keyboard_interrupt'
    assert result['completed_updates']==0
    result=run_longrun(LongRunConfig(root=str(tmp_path/'longrun'),resume=True,
                                 duration_minutes=2,max_updates=1,progress=False))
    assert result['completed_updates']==1


def test_safety_stop_after_persistent_warning(tmp_path,monkeypatch):
    source=_source(tmp_path)
    from holdem_ai.experiments import longrun as m
    original=m._one_update
    def always_warn(sess,cfg):
        event=original(sess,cfg)
        event['entropy']=0.01
        return event
    monkeypatch.setattr(m,'_one_update',always_warn)
    cfg=_config(tmp_path,source,max_updates=5,entropy_stop_patience=2,
                entropy_alert_threshold=0.2,eval_every=99)
    result=run_longrun(cfg)
    assert result['stop_reason']=='safety_stop'
    assert result['updates_this_run']==2
    assert result['entropy_warnings']==2


def test_recover_checkpoint_one_update_ahead_of_state(tmp_path,monkeypatch):
    source=_source(tmp_path)
    run_longrun(_config(tmp_path,source))
    root=tmp_path/'longrun'
    statepath=root/'state.json'
    state=json.loads(statepath.read_text())
    state['completed_updates']-=1
    statepath.write_text(json.dumps(state))
    result=run_longrun(LongRunConfig(root=str(root),resume=True,max_updates=1,
                                     duration_minutes=2,progress=False))
    assert result['completed_updates']==2
    log=(root/'metrics.jsonl').read_text()
    assert 'recovered_update' in log


def test_source_matches_phase14_best_format(tmp_path):
    src=_source(tmp_path)
    # Phase14 PPO best.pt remains a complete Phase9-format checkpoint.
    dest=tmp_path/'best.pt'
    dest.write_bytes(src.read_bytes())
    result=run_longrun(_config(tmp_path,dest))
    assert result['completed_updates']==1


def test_qboost_from_phase14_format_and_beta_guard(tmp_path):
    from holdem_ai.training.qboost_session import from_phase9_checkpoint, save_qboost_checkpoint
    from holdem_ai.training.qboost import QBoostConfig
    source=_source(tmp_path)
    qs=from_phase9_checkpoint(source,device='cpu',qconfig=QBoostConfig(beta=1.0))
    checkpoint=tmp_path/'phase14_qboost_best.pt'
    save_qboost_checkpoint(qs,checkpoint)
    cfg=_config(tmp_path,checkpoint,algorithm='qboost')
    out=run_longrun(cfg)
    assert out['algorithm']=='qboost'
    assert out['completed_updates']==1
    assert finalize_longrun(tmp_path/'longrun')['algorithm']=='qboost'
    bad_root=tmp_path/'mismatch'
    with pytest.raises(ValueError):
        run_longrun(replace(cfg,root=str(bad_root),beta=0.2))


def test_exclusive_run_lock_refuses_parallel_job(tmp_path):
    from holdem_ai.experiments.longrun import _exclusive_lock
    root=tmp_path/'shared'
    with _exclusive_lock(root):
        with pytest.raises(RuntimeError,match='another Phase15'):
            with _exclusive_lock(root):
                pass
    # OS lock released, following invocation proceeds.
    with _exclusive_lock(root):
        assert (root/'.phase15.lock').exists()


def test_selected_checkpoint_hash_verified_before_resume(tmp_path):
    source=_source(tmp_path)
    run_longrun(_config(tmp_path,source))
    (tmp_path/'longrun'/'best.pt').write_bytes(b'bad')
    with pytest.raises(ValueError,match='selected best.pt changed'):
        run_longrun(LongRunConfig(root=str(tmp_path/'longrun'),resume=True,progress=False,
                                  max_updates=1,duration_minutes=1))
