from __future__ import annotations

import json

import pytest
import torch

from holdem_ai.models.transformer import NetworkConfig, TransformerActor
from scripts.phase16_evaluate import parser, run, _specs


def checkpoint(path):
    network = NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1,
                            ff_dim=64, hidden_dim=48, max_tokens=256)
    model = TransformerActor(network)
    from dataclasses import asdict
    torch.save({'network_config': asdict(network), 'actor_state': model.state_dict()}, path)


def test_cli_smoke_creates_json_csv_without_training(tmp_path):
    model_path = tmp_path / 'phase14_best.pt'
    checkpoint(model_path)
    json_path, csv_path, md_path = tmp_path / 'out.json', tmp_path / 'out.csv', tmp_path / 'out.md'
    args = parser().parse_args(['--candidate', str(model_path), '--groups', '2',
                                 '--seed', '987664', '--opponents', 'random,tight',
                                 '--no-progress', '--output-json', str(json_path),
                                 '--output-csv', str(csv_path), '--output-md', str(md_path)])
    report = run(args)
    assert report['phase'] == 16
    assert report['pool_summary']['total_hands'] == 16
    assert json.loads(json_path.read_text(encoding='utf-8'))['phase'] == 16
    assert csv_path.read_text(encoding='utf-8').count('\n') == 3
    assert 'Phase 16' in md_path.read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):
        run(args)


def test_cli_named_checkpoint_and_invalid_spec(tmp_path):
    model_path = tmp_path / 'source.pt'
    checkpoint(model_path)
    args = parser().parse_args(['--candidate', str(model_path), '--opponents', '',
                                '--opponent-checkpoint', f'previous={model_path}'])
    assert _specs(args)[0].name == 'previous'
    args = parser().parse_args(['--candidate', str(model_path), '--opponents', '',
                                '--opponent-checkpoint', 'broken'])
    with pytest.raises(ValueError):
        _specs(args)
    args = parser().parse_args(['--candidate', str(model_path), '--opponents', ''])
    with pytest.raises(ValueError):
        _specs(args)
