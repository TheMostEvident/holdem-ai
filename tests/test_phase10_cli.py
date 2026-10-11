"""End-to-end Phase 10 CLI on CPU with mock Phase3 integration fixture."""
from __future__ import annotations

import pytest
from scripts.phase10_train import build_parser, run


def args(*extra):
    return build_parser().parse_args([
        '--device', 'cpu', '--updates', '1', '--hands-per-update', '2',
        '--epochs', '1', '--minibatch-size', '4', '--eval-hands', '4',
        '--small-network', '--seed', '22', *extra,
    ])


def test_phase10_cli_train_resume(tmp_path):
    ckpt = str(tmp_path / 'a.pt')
    result = run(args('--checkpoint', ckpt))
    assert result['phase'] == 10
    assert result['trained_policy'] is True
    assert result['completed_updates'] == 1
    assert result['evaluation_before']['hands'] == 4
    assert result['evaluation_after']['hands'] == 4
    assert result['last_update']['logit_variance_ratio'] >= 0
    result2 = run(args('--resume', ckpt, '--checkpoint', ckpt, '--skip-eval'))
    assert result2['completed_updates'] == 2
    assert result2['evaluation_before'] is None


def test_phase10_cli_fresh_comparison(tmp_path):
    out = run(args('--compare', '--checkpoint', str(tmp_path/'qb.pt'),
                   '--ppo-checkpoint', str(tmp_path/'ppo.pt')))
    assert out['ppo_control_last_update']['update'] == 1
    assert out['last_update']['update'] == 1
    assert out['evaluation_before'] == out['ppo_control_evaluation_before']
    assert out['ppo_control_evaluation_after']['hands'] == 4
    assert (tmp_path/'qb.pt').exists() and (tmp_path/'ppo.pt').exists()


def test_phase10_rejects_invalid_comparison_resume(tmp_path):
    with pytest.raises(ValueError):
        run(args('--compare', '--resume', str(tmp_path/'missing.pt')))
    with pytest.raises(ValueError):
        run(args('--resume', 'foo', '--from-phase9', 'bar'))
    with pytest.raises(ValueError):
        run(args('--compare', '--checkpoint', str(tmp_path/'same.pt'),
                 '--ppo-checkpoint', str(tmp_path/'same.pt')))


def test_phase10_rejects_invalid_beta():
    with pytest.raises(ValueError):
        run(args('--beta', '-0.1'))
