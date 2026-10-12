from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from holdem_ai.models.transformer import NetworkConfig, TransformerActor
from holdem_ai.evaluation.checkpoint_loader import load_actor
from scripts.phase12_evaluate import build_parser, run


def make_model_file(p):
    torch.manual_seed(10)
    cfg = NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1,
                        ff_dim=64, hidden_dim=48)
    actor = TransformerActor(cfg)
    torch.save({'phase': 10, 'network_config': asdict(cfg),
                'actor_state': actor.state_dict()}, p)


def test_local_checkpoint_and_json_csv(tmp_path):
    path = tmp_path / 'qboost.pt'
    make_model_file(path)
    loaded = load_actor(path)
    assert not loaded.training
    args = build_parser().parse_args([
        '--candidate', str(path), '--groups', '2', '--device', 'cpu',
        '--output-json', str(tmp_path / 'eval.json'),
        '--output-csv', str(tmp_path / 'eval.csv'),
    ])
    result = run(args)
    assert result['candidate']['hands'] == 8
    assert (tmp_path / 'eval.json').is_file()
    assert len((tmp_path / 'eval.csv').read_text().splitlines()) == 3
    assert json.loads((tmp_path / 'eval.json').read_text()) == result


def test_phase11_anchor_and_paired_control(tmp_path):
    candidate = tmp_path / 'actor.pt'
    anchor = tmp_path / 'anchor.pt'
    control = tmp_path / 'control.pt'
    make_model_file(candidate)
    payload = torch.load(candidate, weights_only=True)
    torch.save({'network_config': payload['network_config'],
                'actor_state': payload['actor_state']}, anchor)
    torch.save(payload, control)
    args = build_parser().parse_args([
        '--candidate', str(candidate), '--control', str(control),
        '--opponent-checkpoint', str(anchor), '--groups', '2',
    ])
    outcome = run(args)
    assert outcome['paired_comparison']['delta_mean_bb100'] == 0
    assert outcome['candidate']['mean_bb100'] == 0
    assert load_actor(anchor).config.embed_dim == 32


def test_reject_same_input_file(tmp_path):
    path = tmp_path / 'one.pt'
    make_model_file(path)
    args = build_parser().parse_args(['--candidate', str(path), '--control', str(path)])
    with pytest.raises(ValueError):
        run(args)


def test_reject_bad_checkpoint(tmp_path):
    bad = tmp_path / 'invalid.pt'
    torch.save({'other': 1}, bad)
    with pytest.raises(ValueError):
        load_actor(bad)
    wrong = tmp_path / 'wrong.pt'
    torch.save({'phase': 11, 'network_config': {}, 'actor_state': {}}, wrong)
    with pytest.raises(ValueError):
        load_actor(wrong)
