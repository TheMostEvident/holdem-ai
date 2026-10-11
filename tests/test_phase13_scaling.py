"""Phase 13 backward-compatible workflow tests (real model and simulator contract)."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from holdem_agent_phase8 import TransformerActor, NetworkConfig
from holdem_agent_phase9 import PPOConfig, create_session
from holdem_agent_phase10 import create_qboost_session
from holdem_agent_phase11 import collect_batched_self_play
from holdem_agent_phase13 import BatchExperimentConfig, action_diagnostics, run_batch_experiment
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
