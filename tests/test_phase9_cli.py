"""Phase-9 executable CLI and checkpoint interface, CPU-only fast smoke."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.phase9_train import build_parser, run


def test_phase9_cli_smoke_and_resume(tmp_path):
    out = tmp_path / 'ppo.pt'
    parser = build_parser()
    args = parser.parse_args([
        '--device', 'cpu', '--updates', '1', '--hands-per-update', '2',
        '--epochs', '1', '--minibatch-size', '4', '--eval-hands', '4',
        '--small-network', '--checkpoint', str(out), '--seed', '13',
    ])
    result = run(args)
    assert result['phase'] == 9
    assert result['trained_policy']
    assert result['completed_updates'] == 1
    assert result['evaluation_before']['hands'] == 4
    assert result['evaluation_after']['hands'] == 4
    assert out.is_file()
    args2 = parser.parse_args([
        '--resume', str(out), '--updates', '1', '--device', 'cpu',
        '--checkpoint', str(out), '--skip-eval',
    ])
    resumed = run(args2)
    assert resumed['completed_updates'] == 2
    assert resumed['evaluation_before'] is None
