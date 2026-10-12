from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from holdem_ai.experiments.time_budget import ExperimentConfig, run_experiment
from holdem_ai.experiments.time_budget import _budget_remaining
from scripts.phase11_compare import compare_logs


def config_for(tmp_path, algorithm='ppo'):
    return ExperimentConfig(algorithm=algorithm, device='cpu',
                            duration_seconds=120, max_updates=1,
                            env_batch_size=3, eval_every=1, eval_hands=4,
                            hands_per_update=3, epochs=1, minibatch_size=8,
                            small_network=True,
                            checkpoint=str(tmp_path / f'{algorithm}.pt'),
                            log_dir=str(tmp_path / f'logs_{algorithm}'))


@pytest.mark.parametrize('algorithm', ['ppo', 'qboost'])
def test_small_real_update_and_resume(algorithm, tmp_path):
    cfg = config_for(tmp_path, algorithm)
    report = run_experiment(cfg)
    assert report['completed_updates'] == 1
    assert report['updates_this_run'] == 1
    assert report['stop_reason'] == 'update_limit'
    assert report['evaluations'] >= 2
    log = tmp_path / f'logs_{algorithm}'
    events = [json.loads(x) for x in (log / 'metrics.jsonl').read_text().splitlines()]
    assert any(x['kind'] == 'update' for x in events)
    assert any(x.get('opponent') == 'frozen_anchor' for x in events)
    assert (log / 'anchor.pt').exists()
    payload = torch.load(cfg.checkpoint, map_location='cpu', weights_only=True)
    assert payload['completed_updates'] == 1
    with pytest.raises(FileExistsError):
        run_experiment(cfg)
    resumed = run_experiment(replace(cfg, resume=cfg.checkpoint))
    assert resumed['completed_updates'] == 2
    assert resumed['updates_this_run'] == 1
    assert torch.load(cfg.checkpoint, map_location='cpu', weights_only=True)['completed_updates'] == 2


def test_compare_reports_after_two_run_types(tmp_path):
    p = config_for(tmp_path, 'ppo')
    q = config_for(tmp_path, 'qboost')
    run_experiment(p)
    run_experiment(q)
    result = compare_logs(str(tmp_path / 'logs_ppo' / 'metrics.jsonl'),
                          str(tmp_path / 'logs_qboost' / 'metrics.jsonl'))
    assert result['shared_updates'] == 1
    assert result['records'][0]['update'] == 1
    assert 'qboost_minus_ppo_random_bb100' in result['records'][0]


def test_budget_and_config_validation(tmp_path):
    cfg = config_for(tmp_path)
    for key, val in [('duration_seconds', 0), ('max_updates', 0),
                     ('env_batch_size', 0), ('eval_hands', 6), ('beta', 2),
                     ('algorithm', 'vrpo'), ('seed', True)]:
        with pytest.raises(ValueError):
            replace(cfg, **{key: val})
    assert _budget_remaining(float('inf'))
    assert not _budget_remaining(0.0)


def test_tiny_duration_stops_safely(tmp_path):
    cfg = replace(config_for(tmp_path), duration_seconds=1e-6)
    result = run_experiment(cfg)
    assert result['completed_updates'] == 0
    assert result['stop_reason'] == 'duration_limit'
    assert torch.load(cfg.checkpoint, weights_only=True, map_location='cpu')['completed_updates'] == 0


def test_resume_requires_original_log_dir(tmp_path):
    cfg = config_for(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_experiment(replace(cfg, resume=cfg.checkpoint))


def test_import_phase10_checkpoint(tmp_path):
    from holdem_ai.training.qboost_session import create_qboost_session, save_qboost_checkpoint
    from holdem_ai.models.transformer import NetworkConfig
    from holdem_ai.training.ppo import PPOConfig
    prior = create_qboost_session(
        device='cpu', network_config=NetworkConfig(
            embed_dim=32, attention_heads=2, transformer_layers=1,
            ff_dim=64, hidden_dim=48),
        ppo_config=PPOConfig(hands_per_update=3, epochs=1, minibatch_size=8),
    )
    prior.base.completed_updates = 6
    source = tmp_path / 'phase10_pretrained.pt'
    save_qboost_checkpoint(prior, source)
    cfg = replace(config_for(tmp_path, 'qboost'),
                  from_phase10=str(source))
    r = run_experiment(cfg)
    assert r['completed_updates'] == 7
    assert json.loads((tmp_path / 'logs_qboost' / 'run.json').read_text())['initial_updates'] == 6
    assert torch.load(source, weights_only=True)['completed_updates'] == 6
