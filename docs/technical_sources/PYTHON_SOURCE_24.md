# Complete Python source volume 24/28

[Integrated technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Pinned source commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. All code literal and complete.

## tests/test_phase13_gpu.py

Original blob `65a1da0f6a4920ecfadd062ecc3022c36d4e24f7` (800 bytes).

<a id="py-277"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase13_gpu.py" sha="65a1da0f6a4920ecfadd062ecc3022c36d4e24f7" -->

````````python
"""Actual CUDA run, skipped in environments without a CUDA GPU."""
from __future__ import annotations

import pytest
import torch

from holdem_ai.experiments.batch_scaling import BatchExperimentConfig, run_batch_experiment


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_phase13_cuda_optimization(tmp_path):
    config = BatchExperimentConfig(
        algorithm='qboost', device='cuda', duration_seconds=60,
        hands_per_update=8, env_batch_size=4, minibatch_size=8,
        epochs=1, max_updates=1, skip_eval=True,
        small_network=True,
        checkpoint=str(tmp_path/'model.pt'), log_dir=str(tmp_path/'log'))
    record = run_batch_experiment(config)
    assert record['completed_updates'] == 1
    assert record['cuda_peak_allocated_mb'] > 0

````````

<!-- END_PY_SOURCE path="tests/test_phase13_gpu.py" -->

## tests/test_phase13_scaling.py

Original blob `806fde41a3ebbfa6386d199ba8da1d7a792c769b` (9177 bytes).

<a id="py-278"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase13_scaling.py" sha="806fde41a3ebbfa6386d199ba8da1d7a792c769b" -->

````````python
"""Phase 13 backward-compatible workflow tests (real model and simulator contract)."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from holdem_ai.models.transformer import TransformerActor, NetworkConfig
from holdem_ai.training.ppo import PPOConfig
from holdem_ai.training.ppo_session import create_session
from holdem_ai.training.qboost_session import create_qboost_session
from holdem_ai.training.batched_collector import collect_batched_self_play
from holdem_ai.experiments.batch_scaling import BatchExperimentConfig, run_batch_experiment
from holdem_ai.evaluation.action_diagnostics import action_diagnostics
from scripts.phase13_benchmark import parse_sizes
from scripts.phase13_train import build_parser


@pytest.mark.parametrize('bad', [0, -2, True, 1.25])
def test_invalid_hands_per_update(bad):
    with pytest.raises(ValueError):
        BatchExperimentConfig(hands_per_update=bad)


def test_reject_resume_batch_change():
    with pytest.raises(ValueError):
        BatchExperimentConfig(resume='a.pt', hands_per_update=64)


def test_reject_source_overwrite(tmp_path):
    path = str(tmp_path / 'same.pt')
    with pytest.raises(ValueError):
        BatchExperimentConfig(from_checkpoint=path, checkpoint=path)


def test_benchmark_size_validation():
    assert parse_sizes('16,32,64') == [16, 32, 64]
    for value in ('', '3,3', '-1,2', '1,bad'):
        with pytest.raises(ValueError):
            parse_sizes(value)


def test_cli_parses_batch_options():
    args = build_parser().parse_args(['--hands-per-update', '128',
                                      '--minibatch-size', '64', '--env-batch-size', '64'])
    assert (args.hands_per_update, args.minibatch_size, args.env_batch_size) == (128, 64, 64)


def sample_rows():
    model = TransformerActor(NetworkConfig(embed_dim=16, attention_heads=2,
                                           transformer_layers=1, ff_dim=32,
                                           hidden_dim=24, max_tokens=256))
    config = PPOConfig(hands_per_update=5, epochs=1, minibatch_size=8)
    return model, collect_batched_self_play(model, config, env_batch_size=3)


def test_action_diagnostics_legal_denominators():
    model, rows = sample_rows()
    stat = action_diagnostics(rows, model, chunk_size=3)
    assert stat['decisions'] == len(rows.actions)
    assert sum(stat['actions'].values()) == len(rows.actions)
    for action, count in stat['actions'].items():
        assert count <= stat['legal_opportunities'][action]
        legal = stat['legal_opportunities'][action]
        assert stat['conditional_action_rate'][action] == pytest.approx(round(count/legal, 6)) if legal else stat['conditional_action_rate'][action] is None
    assert 0 <= stat['mean_normalized_entropy'] <= 1.001
    assert sum(x['decisions'] for x in stat['by_street'].values()) == len(rows.actions)


@pytest.mark.parametrize('algorithm', ['ppo','qboost'])
def test_new_run_and_resume(algorithm, tmp_path):
    root = tmp_path / algorithm
    cfg = BatchExperimentConfig(algorithm=algorithm, duration_seconds=90,
                                hands_per_update=6, minibatch_size=8, epochs=1,
                                env_batch_size=4, max_updates=2,
                                eval_groups=2, eval_every=2, skip_eval=True,
                                small_network=True,
                                checkpoint=str(root / 'check.pt'),
                                log_dir=str(root / 'logs'))
    first = run_batch_experiment(cfg)
    assert first['completed_updates'] == 2
    payload = torch.load(cfg.checkpoint, weights_only=True, map_location='cpu')
    assert payload['ppo_config']['hands_per_update'] == 6
    assert payload['completed_updates'] == 2
    events = [json.loads(line) for line in (root / 'logs' / 'metrics.jsonl').read_text().splitlines()]
    updates = [r for r in events if r['kind'] == 'update']
    assert len(updates) == 2
    assert sum(updates[0]['action_diagnostics']['actions'].values()) > 0
    resumed = replace(cfg, hands_per_update=None, minibatch_size=None,
                      epochs=None, resume=cfg.checkpoint, max_updates=1)
    second = run_batch_experiment(resumed)
    assert second['completed_updates'] == 3
    assert torch.load(cfg.checkpoint, weights_only=True, map_location='cpu')['completed_updates'] == 3


def test_warmstart_preserves_source_and_optimizer(tmp_path):
    import hashlib
    base_root = tmp_path / 'source'
    initial = BatchExperimentConfig(algorithm='ppo', duration_seconds=60,
                                    hands_per_update=3, minibatch_size=4,
                                    epochs=1, env_batch_size=3, max_updates=1,
                                    skip_eval=True, small_network=True,
                                    checkpoint=str(base_root/'model.pt'),
                                    log_dir=str(base_root/'logs'))
    run_batch_experiment(initial)
    with open(initial.checkpoint,'rb') as stream:
        before = hashlib.sha256(stream.read()).hexdigest()
    dst_root = tmp_path / 'scaled'
    cfg = replace(initial, from_checkpoint=initial.checkpoint,
                  checkpoint=str(dst_root/'model.pt'), log_dir=str(dst_root/'logs'),
                  hands_per_update=12, minibatch_size=8, env_batch_size=6,
                  rollout_seed=None)
    result = run_batch_experiment(cfg)
    loaded = torch.load(cfg.checkpoint, weights_only=True, map_location='cpu')
    assert loaded['ppo_config']['hands_per_update'] == 12
    assert loaded['ppo_config']['seed'] == 42 + 1000003
    assert loaded['completed_updates'] == 2
    with open(initial.checkpoint,'rb') as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == before


def test_reject_incompatible_resume_metadata(tmp_path):
    cfg = BatchExperimentConfig(algorithm='ppo', duration_seconds=60,
                                hands_per_update=2, max_updates=1,
                                env_batch_size=2, skip_eval=True,
                                small_network=True, checkpoint=str(tmp_path/'p.pt'),
                                log_dir=str(tmp_path/'run'))
    run_batch_experiment(cfg)
    wrong = replace(cfg, hands_per_update=None, checkpoint=str(tmp_path/'other.pt'),
                    resume=cfg.checkpoint)
    with pytest.raises((ValueError, FileNotFoundError)):
        run_batch_experiment(wrong)


def test_matched_warmstart_ppo_to_qboost(tmp_path):
    original = BatchExperimentConfig(algorithm='ppo', duration_seconds=60,
                                     hands_per_update=2, minibatch_size=4,
                                     epochs=1, max_updates=1, skip_eval=True,
                                     small_network=True, checkpoint=str(tmp_path/'pre.pt'),
                                     log_dir=str(tmp_path/'pre_logs'))
    run_batch_experiment(original)
    qb = BatchExperimentConfig(algorithm='qboost', duration_seconds=60,
                               hands_per_update=4, env_batch_size=4,
                               minibatch_size=4, epochs=1,
                               max_updates=1, skip_eval=True,
                               from_checkpoint=original.checkpoint,
                               checkpoint=str(tmp_path/'boost.pt'),
                               log_dir=str(tmp_path/'boost_logs'))
    report = run_batch_experiment(qb)
    payload = torch.load(qb.checkpoint, weights_only=True, map_location='cpu')
    assert payload['phase'] == 10
    assert payload['completed_updates'] == 2
    assert report['completed_updates'] == 2


def test_phase13_compare_reports_and_rejects_unmatched(tmp_path):
    from scripts.phase13_compare import compare
    source = tmp_path / 'src'
    first = BatchExperimentConfig(algorithm='ppo', duration_seconds=60,
                                  hands_per_update=2, env_batch_size=2,
                                  max_updates=1, epochs=1, skip_eval=True,
                                  small_network=True,
                                  checkpoint=str(source / 'p.pt'),
                                  log_dir=str(source / 'logs'))
    run_batch_experiment(first)
    p = tmp_path / 'ppo'
    q = tmp_path / 'qboost'
    configs = [replace(first, from_checkpoint=first.checkpoint,
                       checkpoint=str(p / 'model.pt'), log_dir=str(p / 'logs'),
                       hands_per_update=5),
               BatchExperimentConfig(algorithm='qboost', duration_seconds=60,
                                     from_checkpoint=first.checkpoint,
                                     hands_per_update=5, env_batch_size=2,
                                     epochs=1, minibatch_size=64,
                                     max_updates=1, skip_eval=True,
                                     checkpoint=str(q / 'model.pt'),
                                     log_dir=str(q / 'logs'))]
    for cfg in configs:
        run_batch_experiment(cfg)
    result = compare(p/'logs', q/'logs')
    assert result['matched_start_and_hyperparameters']
    assert result['ppo']['training_hands'] == 5
    assert result['qboost']['training_hands'] == 5

````````

<!-- END_PY_SOURCE path="tests/test_phase13_scaling.py" -->

## tests/test_phase14_gpu.py

Original blob `54eda8d142a2d19a131617d3c630ead0099f24f8` (1124 bytes).

<a id="py-279"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase14_gpu.py" sha="54eda8d142a2d19a131617d3c630ead0099f24f8" -->

````````python
"""A tiny live CUDA end-to-end check for the user's Windows GPU."""
import pytest
import torch

from holdem_ai.experiments.controlled_study import ComparisonConfig, run_study
from holdem_ai.training.ppo_session import create_session, save_checkpoint
from holdem_ai.training.ppo import PPOConfig
from holdem_ai.models.transformer import NetworkConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_phase14_cuda_smoke(tmp_path):
    source=tmp_path/'source.pt'
    save_checkpoint(create_session(device='cpu',seed=77,
        network_config=NetworkConfig(embed_dim=16,attention_heads=2,
            transformer_layers=1, ff_dim=32,hidden_dim=24),
        ppo_config=PPOConfig(hands_per_update=2,epochs=1,minibatch_size=4)), source)
    result=run_study(ComparisonConfig(source_checkpoint=str(source),
        root=str(tmp_path/'study'),seeds=(1,),updates=1,hands_per_update=4,
        env_batch_size=2,minibatch_size=4,epochs=1,eval_every=1,
        validation_groups=2,test_groups=2,device='cuda',
        arm_duration_minutes=2,progress=False))
    assert result['status']=='complete'

````````

<!-- END_PY_SOURCE path="tests/test_phase14_gpu.py" -->

## tests/test_phase14_study.py

Original blob `85db053dff0e41f43e93fdab8567476a30cd2db1` (5973 bytes).

<a id="py-280"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase14_study.py" sha="85db053dff0e41f43e93fdab8567476a30cd2db1" -->

````````python
"""Phase14 correctness, matched-start and held-out test protection."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
import torch

from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import PPOConfig
from holdem_ai.training.ppo_session import create_session, save_checkpoint
from holdem_ai.experiments.controlled_study import ComparisonConfig, run_study
from holdem_ai.evaluation.seed_statistics import seed_level_summary
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
    from holdem_ai.experiments import controlled_study as module
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
    from holdem_ai.experiments import controlled_study as m
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

````````

<!-- END_PY_SOURCE path="tests/test_phase14_study.py" -->

## tests/test_phase15_critic_diagnostics.py

Original blob `f74c4f18b35431e0e9810a78a053028c7871547a` (6084 bytes).

<a id="py-281"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase15_critic_diagnostics.py" sha="f74c4f18b35431e0e9810a78a053028c7871547a" -->

````````python
"""Tests for Phase-15 read-only critic calibration and numerical edge cases."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holdem_ai.evaluation.critic_diagnostics import (
    diagnose_critic_on_rows,
    load_ppo_critic_checkpoint,
    regression_metrics,
)
from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import PPOConfig, TrajectoryRows
from holdem_ai.training.ppo_session import create_session, save_checkpoint


class _HalfHalfActor:
    training = True

    def eval(self):
        self.training = False
        return self

    def train(self, state=True):
        self.training = state
        return self

    def distribution(self, batch):
        raw = torch.zeros(batch.legal_action_mask.shape,
                          dtype=torch.float32, device=batch.legal_action_mask.device)
        raw = raw.masked_fill(~batch.legal_action_mask, -1e9)
        return torch.distributions.Categorical(logits=raw)


class _KnownCritic:
    training = True

    def eval(self):
        self.training = False
        return self

    def train(self, state=True):
        self.training = state
        return self

    def __call__(self, batch):
        n = batch.features.shape[0]
        q0 = torch.tensor([1., 0., 3., 0., 0., 0., 0.],
                          device=batch.features.device).expand(n, -1)
        return torch.stack((q0, -q0), dim=1)


def _encoding(dim):
    f = [0.] * dim
    f[1] = 1.
    f[2] = 0.
    result = SimpleNamespace(
        features=tuple(f), token_ids=(1,), token_values=(0.,)
    )
    if dim == 20:
        result.legal_action_mask = (True, False, True, False, False, False, False)
    return result


def _rows():
    return TrajectoryRows(
        actors=(_encoding(20), _encoding(20)),
        critics=(_encoding(24), _encoding(24)),
        actions=(0, 2), acting_seats=(0, 1),
        returns_bb=(1., -3.), hands=2,
        hand_rewards=((1., -1.), (3., -3.)),
        terminal_actions=(0, 2), decisions_per_hand=(1, 1),
    )


def test_regression_metrics_bias_rmse_and_ev():
    m = regression_metrics([1., -3.], [2., -2.])
    assert m["bias_bb"] == 1.
    assert m["mae_bb"] == 1.
    assert m["rmse_bb"] == 1.
    assert m["explained_variance"] == 1.  # EV ignores constant prediction bias


def test_explained_variance_undefined_for_zero_target_variance():
    assert regression_metrics([2., 2.], [0., 3.])["explained_variance"] is None


def test_nonfinite_predictions_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        regression_metrics([0.], [float("nan")])


def test_chosen_q_and_action_independent_baseline_separate():
    actor, critic = _HalfHalfActor(), _KnownCritic()
    summary = diagnose_critic_on_rows(_rows(), actor, critic, chunk_size=1)
    q = summary["q_taken"]["overall"]
    v = summary["v_reference_policy"]["overall"]
    assert q["rmse_bb"] == 0.
    assert v["rmse_bb"] == 1.
    assert v["bias_bb"] == 1.
    assert summary["raw_advantage"]["mean_bb"] == -1.
    assert summary["q_taken"]["action:fold"]["decisions"] == 1
    assert summary["q_taken"]["action:call"]["decisions"] == 1
    assert actor.training is True and critic.training is True


def test_trusted_phase9_checkpoint_loader_preserves_models(tmp_path):
    net = NetworkConfig(embed_dim=8, attention_heads=2, transformer_layers=1,
                        ff_dim=16, hidden_dim=16)
    session = create_session(network_config=net, ppo_config=PPOConfig(seed=19))
    session.completed_updates = 125
    path = tmp_path / "best.pt"
    save_checkpoint(session, path)
    loaded = load_ppo_critic_checkpoint(path)
    assert loaded.completed_updates == 125
    assert loaded.ppo_config.seed == 19
    assert all(not p.requires_grad for p in loaded.actor.parameters())
    assert all(not p.requires_grad for p in loaded.critic.parameters())
    assert not loaded.actor.training and not loaded.critic.training
    assert path.is_file()


def test_critic_audit_never_exposes_privileged_state_to_actor():
    # Input restriction is enforced by the ActorBatch contract inside the
    # evaluation function; this numerical test checks separate batches work.
    summary = diagnose_critic_on_rows(_rows(), _HalfHalfActor(), _KnownCritic())
    assert summary["decisions"] == 2
    assert summary["v_reference_policy"]["street:preflop"]["decisions"] == 2


def test_three_rollout_protocol_uses_single_shared_reference(monkeypatch):
    from holdem_ai.evaluation import critic_diagnostics as mod

    net = NetworkConfig(embed_dim=8, attention_heads=2, transformer_layers=1,
                        ff_dim=16, hidden_dim=16)
    ppo = PPOConfig(seed=19)
    sessions = {
        "ref": mod.CriticCheckpoint(_HalfHalfActor(), _KnownCritic(), net, ppo, 104),
        "best": mod.CriticCheckpoint(_HalfHalfActor(), _KnownCritic(), net, ppo, 125),
        "latest": mod.CriticCheckpoint(_HalfHalfActor(), _KnownCritic(), net, ppo, 204),
    }
    collected_with = []

    def fake_collect(actor, config, *, iteration, device, env_batch_size):
        collected_with.append((actor, config.seed, config.hands_per_update))
        return _rows()

    monkeypatch.setattr(mod, "load_ppo_critic_checkpoint",
                        lambda path, *, device: sessions[str(path)])
    monkeypatch.setattr(mod, "collect_batched_self_play", fake_collect)
    result = mod.run_critic_audit(
        reference_checkpoint="ref", before_checkpoint="best",
        after_checkpoint="latest", hands=2, on_policy_hands=2,
        seed=570000000, env_batch_size=2)
    assert len(collected_with) == 3
    assert collected_with[0][0] is sessions["ref"].actor
    assert collected_with[1][0] is sessions["best"].actor
    assert collected_with[2][0] is sessions["latest"].actor
    assert collected_with[0][1] == 570000000
    assert collected_with[1][1] == collected_with[2][1] == 571000000
    assert result["completed_updates"] == {"before": 125, "after": 204}
    assert result["shared_reference"]["results"]["before"]["q_taken"]["overall"]["rmse_bb"] == 0.

````````

<!-- END_PY_SOURCE path="tests/test_phase15_critic_diagnostics.py" -->

## tests/test_phase15_gpu.py

Original blob `a1a2816440eafd97651bc0cb23addb49892fa5b3` (1352 bytes).

<a id="py-282"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase15_gpu.py" sha="a1a2816440eafd97651bc0cb23addb49892fa5b3" -->

````````python
"""Real CUDA invocation: only runs on CUDA-enabled hosts."""
from __future__ import annotations

import pytest
import torch

from holdem_ai.experiments.longrun import LongRunConfig, run_longrun
from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import PPOConfig
from holdem_ai.training.ppo_session import create_session, save_checkpoint


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA not available')
def test_cuda_one_update(tmp_path):
    source=tmp_path/'source.pt'
    session=create_session(device='cuda',seed=2,
        network_config=NetworkConfig(embed_dim=16,attention_heads=2,transformer_layers=1,
                                     ff_dim=32,hidden_dim=24,max_tokens=256),
        ppo_config=PPOConfig(hands_per_update=4,epochs=1,minibatch_size=4))
    save_checkpoint(session,source)
    output=run_longrun(LongRunConfig(root=str(tmp_path/'gpu'),source_checkpoint=str(source),
                                   algorithm='ppo',device='cuda',duration_minutes=1,
                                   max_updates=1,progress=False,hands_per_update=4,
                                   env_batch_size=2,minibatch_size=4,epochs=1,
                                   eval_groups=2,final_groups=2))
    assert output['completed_updates']==1
    assert output['cuda_peak_allocated_mb'] is not None

````````

<!-- END_PY_SOURCE path="tests/test_phase15_gpu.py" -->

## tests/test_phase15_initial_policy_comparison.py

Original blob `fa5ca09fc89d8c94bbdf1c94e824e8abc8458253` (6925 bytes).

<a id="py-283"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase15_initial_policy_comparison.py" sha="fa5ca09fc89d8c94bbdf1c94e824e8abc8458253" -->

````````python
"""Agent baseline all-in comparison regression tests (no environment edits)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holdem_ai.encoding.observation import EncodedActor
from holdem_ai.encoding.tensors import batch_actors
from holdem_ai.evaluation import initial_policy_comparison as audit
from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import TrajectoryRows


def _net():
    return NetworkConfig(embed_dim=8, attention_heads=2, transformer_layers=1,
                         ff_dim=16, hidden_dim=16, max_tokens=64)


def _row(stage, allin_legal=True):
    f = [0.] * 20
    f[1] = 1.
    f[2] = float(stage)
    mask = (True, True, True, True, True, True, allin_legal)
    return EncodedActor((1,), (0.0,), tuple(f), mask)


def _rows():
    actors = (_row(0), _row(0, False), _row(0, False),
              _row(0), _row(1, False), _row(1, False))
    return TrajectoryRows(
        actors=actors, critics=(None,) * 6,
        actions=(6, 2, 0, 2, 3, 1),
        acting_seats=(0, 1, 0, 0, 1, 0),
        returns_bb=(-200., 200., -0.5, 2., -2., 2.),
        hands=3,
        hand_rewards=((-200., 200.), (-0.5, 0.5), (2., -2.)),
        terminal_actions=(2, 0, 1),
        decisions_per_hand=(2, 1, 3),
    )


def test_uniform_actor_probability_mask_and_normalization():
    actor = audit.UniformLegalActor(_net())
    mask1 = (True, False, True, False, False, False, True)
    mask2 = (False, True, False, False, False, False, False)
    obs = [
        EncodedActor((1,), (0.,), (0.,) * 20, mask1),
        EncodedActor((1,), (0.,), (0.,) * 20, mask2),
    ]
    dist = actor.distribution(batch_actors(obs, device="cpu"))
    assert torch.allclose(dist.probs[0], torch.tensor([1/3, 0, 1/3, 0, 0, 0, 1/3]))
    assert torch.allclose(dist.probs[1], torch.tensor([0., 1., 0., 0., 0., 0., 0.]))


def test_random_initialization_is_reproducible_and_rng_safe():
    config = _net()
    torch.manual_seed(987)
    before = torch.get_rng_state().clone()
    a = audit.build_untrained_actor(config, seed=52)
    b = audit.build_untrained_actor(config, seed=52)
    c = audit.build_untrained_actor(config, seed=53)
    assert torch.equal(torch.get_rng_state(), before)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k])
               for k in a.state_dict())
    assert any(not torch.equal(a.state_dict()[k], c.state_dict()[k])
               for k in a.state_dict())
    assert not a.training
    assert all(not p.requires_grad for p in a.parameters())


def test_on_policy_action_rate_uses_legal_denominator_and_hand_denominator():
    metrics = audit._action_counts_on_rows(_rows())
    assert metrics["overall"]["decisions"] == 6
    assert metrics["overall"]["all_in_legal_decisions"] == 2
    assert metrics["overall"]["all_in_selected_decisions"] == 1
    assert metrics["overall"]["all_in_rate_given_legal"] == .5
    assert metrics["preflop"]["decisions"] == 4
    assert metrics["preflop"]["all_in_rate_given_legal"] == .5
    assert metrics["per_hand"]["hands"] == 3
    assert metrics["per_hand"]["hands_with_explicit_all_in"] == 1
    assert metrics["per_hand"]["hands_with_explicit_all_in_fraction"] == .333333
    assert metrics["per_hand"]["first_decision_all_in_given_legal"] == .5
    assert metrics["flop"]["all_in_rate_given_legal"] is None
    assert metrics["river"]["decisions"] == 0


def test_shared_seed_policy_order_and_phase14_fixed_observations(monkeypatch):
    trained = SimpleNamespace(config=_net())
    initial = SimpleNamespace(config=_net())
    seen = []
    phase14_obs = _rows().actors
    def fake_collect(model, conf, *, iteration, device, env_batch_size):
        seen.append((model, conf.seed, conf.hands_per_update, env_batch_size))
        return _rows()
    def fake_fixed(models, rows, *, device):
        assert rows is phase14_obs or rows == phase14_obs
        assert set(models) == {"uniform_legal", "random_init", "trained_phase14"}
        return {"sample_count": len(rows)}
    monkeypatch.setattr(audit, "load_actor", lambda path, *, device: trained)
    monkeypatch.setattr(audit, "build_untrained_actor",
                        lambda cfg, *, seed, device: initial)
    monkeypatch.setattr(audit, "collect_batched_self_play", fake_collect)
    monkeypatch.setattr(audit, "compare_actors_on_states", fake_fixed)
    report = audit.compare_three_agent_policies(
        trained_checkpoint="phase14.pt", hands=3, seed=590000000,
        env_batch_size=2, device="cpu", init_seed=52)
    assert [name for name in report["on_policy"]] == [
        "uniform_legal", "random_init", "trained_phase14"]
    assert seen[1][0] is initial and seen[2][0] is trained
    assert all(item[1:] == (590000000, 3, 2) for item in seen)
    assert report["fixed_phase14_states"]["states"] == 6
    assert "NOT_verified" in report["initial_policy_provenance"]


def test_initial_checkpoint_architecture_mismatch_rejected(monkeypatch):
    trained = SimpleNamespace(config=_net())
    wrong = SimpleNamespace(config=NetworkConfig())
    monkeypatch.setattr(audit, "load_actor",
                        lambda path, *, device: wrong if path == "initial" else trained)
    with pytest.raises(ValueError, match="architecture differs"):
        audit.compare_three_agent_policies(
            trained_checkpoint="trained", initial_checkpoint="initial",
            hands=3, env_batch_size=2)


@pytest.mark.parametrize("kwargs", [
    {"hands": 0},
    {"seed": -1},
    {"env_batch_size": 0},
    {"hands": 2, "env_batch_size": 3},
])
def test_bad_sampling_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        audit.compare_three_agent_policies(trained_checkpoint="x", **kwargs)


def test_end_to_end_three_policy_comparison_two_real_hands_cpu(tmp_path):
    """Tiny integration run: real existing Collector, masks, engine and fixed audit."""
    from dataclasses import replace
    from holdem_ai.training.ppo import PPOConfig
    from holdem_ai.training.ppo_session import create_session, save_checkpoint

    path = tmp_path / "seeded_phase14_shape.pt"
    session = create_session(
        seed=52,
        network_config=replace(_net(), max_tokens=256),
        ppo_config=PPOConfig(seed=52, hands_per_update=2),
    )
    save_checkpoint(session, path)
    result = audit.compare_three_agent_policies(
        trained_checkpoint=path,
        hands=2,
        seed=590001000,
        env_batch_size=2,
        device="cpu",
        init_seed=53,
    )
    assert result["fixed_phase14_states"]["states"] > 0
    assert set(result["on_policy"]) == {
        "uniform_legal", "random_init", "trained_phase14",
    }
    for entry in result["on_policy"].values():
        assert entry["terminal"]["hands"] == 2
        assert entry["actions"]["overall"]["decisions"] >= 2
        assert 0 <= entry["actions"]["per_hand"]["hands_with_explicit_all_in"] <= 2

````````

<!-- END_PY_SOURCE path="tests/test_phase15_initial_policy_comparison.py" -->

## tests/test_phase15_longrun.py

Original blob `9278d2829c17a4e9e2feefffe7c3e3ef5c27a0b6` (9764 bytes).

<a id="py-284"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase15_longrun.py" sha="9278d2829c17a4e9e2feefffe7c3e3ef5c27a0b6" -->

````````python
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
                entropy_alert_threshold=0.2,entropy_stop_enabled=True,eval_every=99)
    result=run_longrun(cfg)
    assert result['stop_reason']=='safety_stop'
    assert result['updates_this_run']==2
    assert result['entropy_warnings']==2



def test_low_entropy_warning_only_for_new_runs(tmp_path,monkeypatch):
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
    assert result['stop_reason']=='update_limit'
    assert result['updates_this_run']==5
    assert result['entropy_warnings']==5
    assert result['kl_warnings']==0


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

````````

<!-- END_PY_SOURCE path="tests/test_phase15_longrun.py" -->

## tests/test_phase15_policy_audit.py

Original blob `d9ad37186493e543810e6d842637f53176c4e1aa` (3422 bytes).

<a id="py-285"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase15_policy_audit.py" sha="d9ad37186493e543810e6d842637f53176c4e1aa" -->

````````python
"""Regression tests for Phase 15 *read-only* policy/optimizer diagnostics."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from holdem_ai.evaluation.fixed_state_policy_audit import compare_actors_on_states
from holdem_ai.evaluation.ppo_metric_audit import audit_metrics


class StubActor:
    def __init__(self, fold_logit: float) -> None:
        self.fold_logit = fold_logit
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, flag=True):
        self.training = flag
        return self

    def distribution(self, batch):
        b, a = batch.legal_action_mask.shape
        logits = torch.zeros(b, a)
        logits[:, 0] = self.fold_logit
        logits = logits.masked_fill(~batch.legal_action_mask, -1.0e9)
        return torch.distributions.Categorical(logits=logits)


def _row(*, stage=0, button=1.0, mask=(True, False, True, False, False, False, False)):
    features = [0.0] * 20
    features[1] = button
    features[2] = float(stage)
    return SimpleNamespace(
        token_ids=(1,), token_values=(0.0,), features=tuple(features),
        legal_action_mask=mask,
    )


def test_fixed_state_fold_probability_change_is_not_trajectory_shift():
    rows = (_row(), _row(stage=1, button=0.0),
            _row(mask=(False, True, False, False, False, False, False)))
    before, after = StubActor(0.0), StubActor(2.0)
    results = compare_actors_on_states({"before": before, "after": after}, rows)
    a, b = results["before"]["overall"], results["after"]["overall"]
    assert a["states"] == b["states"] == 3
    assert a["fold_legal_states"] == b["fold_legal_states"] == 2
    assert a["mean_fold_probability_given_legal"] == pytest.approx(0.5)
    assert b["mean_fold_probability_given_legal"] == pytest.approx(
        torch.sigmoid(torch.tensor(2.0)).item(), abs=1e-6)
    assert a["multi_legal_states"] == 2
    assert results["before"]["street:flop"]["states"] == 1
    assert before.training and after.training  # no lasting state mutation


def test_no_fold_opportunities_and_one_legal_action():
    rows = (_row(mask=(False, True, False, False, False, False, False)),)
    result = compare_actors_on_states({"x": StubActor(1)}, rows)["x"]["overall"]
    assert result["mean_fold_probability_given_legal"] is None
    assert result["multi_legal_states"] == 0
    assert result["mean_normalized_entropy_for_multi_legal"] is None


def test_optimizer_metric_audit(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text("\n".join([
        json.dumps({"kind": "update", "algorithm": "ppo", "update": 1,
                    "entropy": 0.8, "approx_kl": 0.001, "clip_fraction": 0.1}),
        json.dumps({"kind": "validation", "update": 1}),
        json.dumps({"kind": "update", "algorithm": "ppo", "update": 2,
                    "entropy": 0.2, "approx_kl": 0.002, "clip_fraction": 0.2}),
    ]), encoding="utf-8")
    summary = audit_metrics(p)
    assert summary["updates"] == 2
    assert summary["fields"]["entropy"]["last"] == 0.2
    assert summary["fields"]["grad_norm_actor"] is None


def test_optimizer_metric_audit_rejects_nonfinite(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"kind":"update","algorithm":"ppo","update":1,"entropy":NaN}\n')
    with pytest.raises(ValueError, match="nonfinite entropy"):
        audit_metrics(p)

````````

<!-- END_PY_SOURCE path="tests/test_phase15_policy_audit.py" -->

## tests/test_phase15_preflop_sequence_audit.py

Original blob `c732cbf270a5dae267d17013033fa073b7f70ae2` (5009 bytes).

<a id="py-286"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase15_preflop_sequence_audit.py" sha="c732cbf270a5dae267d17013033fa073b7f70ae2" -->

````````python
"""Phase15 agent-only preflop sequence audit checks."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from holdem_ai.evaluation import preflop_sequence_audit as audit
from holdem_ai.training.ppo import TrajectoryRows


def _obs(street, seat=0, to_call=0.0, allin=True):
    features = [0.] * 20
    features[2] = float(street)
    features[1] = float(seat == 0)
    features[4] = float(to_call)
    return SimpleNamespace(
        features=tuple(features),
        legal_action_mask=(True, True, True, True, True, True, allin),
    )


def _rows():
    # (1) preflop raise_pot -> all_in -> call (last decision preflop)
    # (2) preflop fold
    # (3) preflop call -> flop check -> flop fold
    actors = (
        _obs(0, 0, 0), _obs(0, 1, 3), _obs(0, 0, 200, allin=False),
        _obs(0, 0, 0),
        _obs(0, 0, 0), _obs(1, 1, 0, allin=False),
        _obs(1, 0, 5, allin=False),
    )
    actions = (4, 6, 2, 0, 2, 1, 0)
    return TrajectoryRows(
        actors=actors, critics=(None,) * len(actions),
        actions=actions, acting_seats=(0, 1, 0, 0, 0, 1, 0),
        returns_bb=(200., -200., 200., -0.5, -3., 3., -3.),
        hands=3, hand_rewards=((200., -200.), (-0.5, 0.5), (-3., 3.)),
        terminal_actions=(2, 0, 0),
        decisions_per_hand=(3, 1, 3),
    )


def test_preflop_sequences_exact_actions_and_first_allin():
    report = audit.summarize_preflop_sequences(_rows())
    assert report["hands"] == 3
    assert report["preflop_explicit_allin_hands"] == 1
    assert report["preflop_explicit_allin_fraction"] == .333333
    assert report["preflop_terminal_fold_hands"] == 1
    assert report["preflop_terminal_call_hands"] == 1
    assert report["first_allin_decision_number_counts"] == {"2": 1}
    assert report["immediately_before_first_allin"] == {"raise_pot": 1}
    assert report["allin_with_prior_raise_head_count"] == 1
    assert report["first_actions"] == {"call": 1, "fold": 1, "raise_pot": 1}
    assert report["preflop_decisions_per_hand"]["mean"] == pytest.approx(5 / 3, abs=1e-6)
    assert report["most_common_prefixes_through_first_allin"] == [
        {"sequence": "raise_pot > all_in", "count": 1},
    ]


def test_conditional_legal_rates_at_decision_depth():
    report = audit.summarize_preflop_sequences(_rows())
    d2 = report["action_choice_by_preflop_decision_number"]["2"]
    assert d2["all_in"]["selected"] == 1
    assert d2["all_in"]["legal_decisions"] == 1
    assert d2["all_in"]["selected_given_legal"] == 1.
    d3 = report["action_choice_by_preflop_decision_number"]["3"]
    assert d3["all_in"]["selected"] == 0
    assert d3["all_in"]["legal_decisions"] == 0
    assert d3["all_in"]["selected_given_legal"] is None
    contexts = report["action_choice_by_context"]
    assert contexts["facing:facing_bet"]["all_in"]["selected"] == 1
    assert contexts["facing:not_facing_bet"]["all_in"]["selected"] == 0


def test_boundaries_required_for_sequence_audit():
    rows = _rows()
    missing = TrajectoryRows(
        rows.actors, rows.critics, rows.actions, rows.acting_seats,
        rows.returns_bb, rows.hands, rows.hand_rewards,
    )
    with pytest.raises(ValueError, match="boundaries"):
        audit.summarize_preflop_sequences(missing)


def test_mismatched_terminal_action_is_rejected():
    rows = _rows()
    invalid = TrajectoryRows(
        rows.actors, rows.critics, rows.actions, rows.acting_seats,
        rows.returns_bb, rows.hands, rows.hand_rewards,
        (0, 0, 0), rows.decisions_per_hand,
    )
    with pytest.raises(ValueError, match="terminal action mismatch"):
        audit.summarize_preflop_sequences(invalid)


def test_seeded_three_policy_comparison_protocol(monkeypatch):
    from holdem_ai.models.transformer import NetworkConfig

    net = NetworkConfig()
    trained = SimpleNamespace(config=net)
    initial = SimpleNamespace(config=net)
    seen = []
    monkeypatch.setattr(audit, "load_actor", lambda path, *, device: trained)
    monkeypatch.setattr(audit, "build_untrained_actor",
                        lambda config, *, seed, device: initial)

    def fake_collect(actor, config, *, iteration, device, env_batch_size):
        seen.append((actor, config.seed, config.hands_per_update))
        return _rows()

    monkeypatch.setattr(audit, "collect_batched_self_play", fake_collect)
    result = audit.compare_preflop_sequences(
        trained_checkpoint="phase14.pt", hands=3,
        seed=600000000, env_batch_size=2)
    assert set(result["models"]) == {
        "uniform_legal", "random_init", "trained_phase14",
    }
    assert seen[1][0] is initial
    assert seen[2][0] is trained
    assert all(item[1:] == (600000000, 3) for item in seen)


@pytest.mark.parametrize("kwargs", [
    {"hands": 0},
    {"seed": -1},
    {"env_batch_size": 0},
    {"hands": 2, "env_batch_size": 3},
])
def test_invalid_sampling_configuration(kwargs):
    with pytest.raises(ValueError):
        audit.compare_preflop_sequences(trained_checkpoint="x", **kwargs)

````````

<!-- END_PY_SOURCE path="tests/test_phase15_preflop_sequence_audit.py" -->

## tests/test_phase15_safety_phase3.py

Original blob `14084273c8b145051a97fd7d02ed3c95e0610334` (2072 bytes).

<a id="py-287"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase15_safety_phase3.py" sha="14084273c8b145051a97fd7d02ed3c95e0610334" -->

````````python
"""Phase 15 phase-3 safety and reward diagnostics regression tests."""
from types import SimpleNamespace

from holdem_ai.experiments.ppo_safety import safety_assessment
from holdem_ai.evaluation.reward_diagnostics import summarize_rollout_rewards
from holdem_ai.training.ppo import TrajectoryRows


def test_low_entropy_is_warning_not_stop_in_new_mode():
    result = safety_assessment(entropy=0.1, approx_kl=0.001,
        entropy_threshold=0.45, max_approx_kl=0.10,
        entropy_stop_enabled=False, consecutive=8)
    assert result["low_entropy"] is True
    assert result["high_kl"] is False
    assert result["consecutive_safety_warnings"] == 0


def test_high_kl_still_counts_with_low_entropy():
    result = safety_assessment(entropy=0.1, approx_kl=0.2,
        entropy_threshold=0.45, max_approx_kl=0.10,
        entropy_stop_enabled=False, consecutive=11)
    assert result["low_entropy"] and result["high_kl"]
    assert result["consecutive_safety_warnings"] == 12


def test_legacy_entropy_stop_can_be_preserved():
    result = safety_assessment(entropy=0.1, approx_kl=0.01,
        entropy_threshold=0.45, max_approx_kl=0.10,
        entropy_stop_enabled=True, consecutive=11)
    assert result["consecutive_safety_warnings"] == 12


def test_reward_diagnostics_fold_and_payoff():
    # Constructor invariants require aligned one-decision-per-hand fake rows.
    rows = TrajectoryRows(
        actors=(SimpleNamespace(legal_action_mask=(True, False, False, False, False, False, False)),
                SimpleNamespace(legal_action_mask=(False, False, True, False, False, False, False))),
        critics=(None, None),
        actions=(0, 2), acting_seats=(0, 1),
        returns_bb=(-0.5, 0.5), hands=2,
        hand_rewards=((-0.5, 0.5), (-1.5, 1.5)),
        terminal_actions=(0, 2), decisions_per_hand=(1, 1))
    result = summarize_rollout_rewards(rows)
    assert result["fold_terminated_hands"] == 1
    assert result["fold_termination_fraction"] == 0.5
    assert result["seat0_mean_reward_bb"] == -1.0
    assert result["showdown_fraction"] is None

````````

<!-- END_PY_SOURCE path="tests/test_phase15_safety_phase3.py" -->

## tests/test_phase15_terminal_reward_audit.py

Original blob `79308df938907860e5652a93a8b98f792194e1b1` (5242 bytes).

<a id="py-288"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase15_terminal_reward_audit.py" sha="79308df938907860e5652a93a8b98f792194e1b1" -->

````````python
"""Regression tests: hand-weighted payoff tails and preflop termination types."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from holdem_ai.evaluation import terminal_reward_audit as audit
from holdem_ai.training.ppo import TrajectoryRows


def _actor(stage: int):
    f = [0.] * 20
    f[2] = float(stage)
    f[1] = 1.
    return SimpleNamespace(
        features=tuple(f),
        legal_action_mask=(True,) * 7,
    )


def _rows():
    # 1. Preflop fold
    # 2. Preflop explicit all-in then a call -> full-board automatic showdown
    # 3. Preflop call then flop fold
    # 4. Preflop/flop/turn/river decisions -> river nonfold showdown
    acts = (0, 6, 2, 2, 0, 2, 1, 1, 2)
    stages = (0, 0, 0, 0, 1, 0, 1, 2, 3)
    return TrajectoryRows(
        actors=tuple(_actor(s) for s in stages),
        critics=(None,) * len(acts),
        actions=acts,
        acting_seats=(0, 0, 1, 0, 1, 0, 1, 0, 1),
        returns_bb=(-0.5, -200., 200., -8., 8., 150., -150., 150., -150.),
        hands=4,
        hand_rewards=((-0.5, 0.5), (-200., 200.), (-8., 8.), (150., -150.)),
        terminal_actions=(0, 2, 0, 2),
        decisions_per_hand=(1, 2, 2, 4),
    )


def test_preflop_exit_and_payoff_tail_exact_counts():
    report = audit.summarize_terminal_distribution(_rows())
    assert report["hands"] == 4
    assert report["decisions"] == 9
    assert report["last_decision_street_counts"] == {
        "preflop": 2, "flop": 1, "turn": 0, "river": 1,
    }
    assert report["preflop"]["last_decision_hand_count"] == 2
    assert report["preflop"]["fold_ended_count"] == 1
    assert report["preflop"]["nonfold_showdown_count"] == 1
    assert report["preflop"]["explicit_all_in_hand_count"] == 1
    assert report["preflop"]["explicit_all_in_and_nonfold_last_preflop"] == 1
    assert report["last_decision_street_x_termination"]["flop:fold"] == 1
    assert report["decision_coverage"] == {
        "flop_hands": 2, "turn_hands": 1, "river_hands": 1,
    }
    dist = report["seat0_hand_weighted_reward"]
    assert dist["count"] == 4
    assert dist["near_full_stack_fraction"] == pytest.approx(0.25)
    assert dist["abs_ge_190bb_fraction"] == pytest.approx(0.25)
    assert dist["signed_full_stack"]["win"] == 0
    assert dist["signed_full_stack"]["loss"] == 1
    assert report["full_stack_settlements"]["hand_count"] == 1
    assert report["payoff_by_exit_group"]["preflop:showdown"]["near_full_stack_fraction"] == 1.0
    assert report["payoff_by_exit_group"]["preflop:fold"]["near_full_stack_fraction"] == 0.0
    assert report["preflop"]["last_action_counts"]["call"] == 1
    assert report["preflop"]["last_action_counts"]["fold"] == 1
    assert report["actor_decision_weighted_return"]["count"] == 9


def test_missing_boundaries_fail_without_guessing_terminations():
    rows = _rows()
    rows = TrajectoryRows(
        rows.actors, rows.critics, rows.actions, rows.acting_seats,
        rows.returns_bb, rows.hands, rows.hand_rewards)
    with pytest.raises(ValueError, match="hand boundaries"):
        audit.summarize_terminal_distribution(rows)


def test_last_decision_not_matching_terminal_action_rejected():
    rows = _rows()
    rows = TrajectoryRows(
        rows.actors, rows.critics, rows.actions, rows.acting_seats,
        rows.returns_bb, rows.hands, rows.hand_rewards,
        terminal_actions=(0, 6, 0, 2),
        decisions_per_hand=(1, 2, 2, 4))
    with pytest.raises(ValueError, match="terminal action does not match"):
        audit.summarize_terminal_distribution(rows)


def test_nonzero_sum_terminal_reward_rejected():
    rows = _rows()
    bad_rewards = list(rows.hand_rewards)
    bad_rewards[0] = (-0.5, 1.0)
    rows = TrajectoryRows(
        rows.actors, rows.critics, rows.actions, rows.acting_seats,
        rows.returns_bb, rows.hands, tuple(bad_rewards),
        rows.terminal_actions, rows.decisions_per_hand)
    with pytest.raises(ValueError, match="non-zero-sum"):
        audit.summarize_terminal_distribution(rows)


def test_two_checkpoint_sampling_protocol(monkeypatch):
    seen = []
    before = object()
    after = object()
    models = {"best.pt": before, "latest.pt": after}

    def fake_load(path, *, device):
        return models[str(path)]

    def fake_collect(actor, cfg, *, iteration, device, env_batch_size):
        seen.append((actor, cfg.seed, cfg.hands_per_update, env_batch_size))
        return _rows()

    monkeypatch.setattr(audit, "load_actor", fake_load)
    monkeypatch.setattr(audit, "collect_batched_self_play", fake_collect)
    result = audit.run_terminal_reward_audit(
        checkpoints={"best": "best.pt", "latest": "latest.pt"},
        device="cpu", hands=4, seed=580000000, env_batch_size=2)
    assert seen == [(before, 580000000, 4, 2), (after, 580000000, 4, 2)]
    assert result["models"]["latest"]["preflop"]["nonfold_showdown_count"] == 1


def test_invalid_config_and_empty_rows_are_rejected():
    with pytest.raises(ValueError, match="stack_bb"):
        audit.summarize_terminal_distribution(_rows(), stack_bb=-1)
    with pytest.raises(ValueError, match="env_batch_size"):
        audit.run_terminal_reward_audit(
            checkpoints={"best": "best.pt"}, hands=4, env_batch_size=0)

````````

<!-- END_PY_SOURCE path="tests/test_phase15_terminal_reward_audit.py" -->
