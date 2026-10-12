"""Progress UI is an observer: no dependency on the poker engine or CUDA."""
from __future__ import annotations

import io
import math
from unittest.mock import patch

import pytest

from holdem_ai.utils.progress import TrainingProgress, format_duration
from scripts.phase11_train import build_parser, run


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def create_progress(clock, *, duration=60.0, updates=4):
    screen = io.StringIO()
    progress = TrainingProgress(duration_seconds=duration, max_updates=updates,
                                stream=screen, clock=clock, threaded=False,
                                prefer_tqdm=False)
    return progress, screen


def test_eta_elapsed_resume_and_terminal_status():
    clock = Clock()
    p, output = create_progress(clock, duration=100.0, updates=4)
    with p:
        p.on_event({'kind': 'resume', 'completed_updates': 11})
        clock.now = 10.0
        p.on_event({'kind': 'update', 'update': 12, 'elapsed_seconds': 0.5,
                    'hands_per_second_training': 42.0})
        snap = p.snapshot()
        assert snap['total_updates'] == 12
        assert snap['updates'] == 1
        assert snap['remaining_updates'] == 3
        assert snap['time_left'] == pytest.approx(90.0)
        assert snap['update_eta'] == pytest.approx(30.0)
        assert snap['estimated_remaining'] == pytest.approx(30.0)
        clock.now = 20.0
        p.on_event({'kind': 'update', 'update': 13, 'hands_per_second_training': 31.0})
        assert p.snapshot()['fraction'] == pytest.approx(0.5)
        p.on_event({'kind': 'end', 'stop_reason': 'update_limit'})
        assert p.snapshot()['status'] == 'update_limit'
    text = output.getvalue()
    assert 'time_left=01:30' not in text  # uses hours:minutes:seconds
    assert 'time_left=00:01:30' in text
    assert 'ETA=00:00:30' in text
    assert 'upd=13 (+2/4)' in text
    assert 'hands/s=31.0' in text
    assert 'elapsed=00:00:20' in text


def test_time_budget_mode_heartbeat_and_ceiling():
    clock = Clock()
    p, screen = create_progress(clock, duration=80.0, updates=None)
    with p:
        p.on_event({'kind': 'start', 'completed_updates': 3})
        clock.now = 21.2
        assert p.snapshot()['fraction'] == pytest.approx(21.2 / 80.0)
        assert format_duration(p.snapshot()['time_left']) == '00:00:59'
        p.on_event({'kind': 'evaluation', 'update': 3})
        assert p.snapshot()['status'] == 'evaluated'
        clock.now = 82.0
        assert p.snapshot()['time_left'] == 0.0
        p.on_event({'kind': 'end', 'stop_reason': 'duration_limit'})
    assert '00:00:00' in screen.getvalue()


def test_zero_update_eta_is_not_fabricated():
    clock = Clock()
    p, _ = create_progress(clock)
    with p:
        p.on_event({'kind': 'start', 'completed_updates': 0})
        assert p.snapshot()['update_eta'] is None
        assert p.snapshot()['estimated_remaining'] == 60.0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        TrainingProgress(duration_seconds=0, max_updates=None)
    with pytest.raises(ValueError):
        TrainingProgress(duration_seconds=math.inf, max_updates=None)
    with pytest.raises(ValueError):
        TrainingProgress(duration_seconds=60, max_updates=0)
    with pytest.raises(ValueError):
        TrainingProgress(duration_seconds=60, max_updates=2, interval=0.01)


def test_cli_additional_flags_do_not_change_experiment_config(tmp_path):
    args = build_parser().parse_args([
        '--device', 'cpu', '--duration-minutes', '0.5',
        '--max-updates', '2', '--checkpoint', str(tmp_path / 'test.pt'),
        '--log-dir', str(tmp_path / 'run'), '--no-progress',
    ])
    assert args.no_progress is True
    assert args.progress_interval == 1.0
    with patch('scripts.phase11_train.run_experiment', return_value={'phase': 11}) as fake:
        result = run(args)
        assert result == {'phase': 11}
        assert fake.call_args.args[0].duration_seconds == 30.0
        assert fake.call_args.args[0].max_updates == 2


def test_cli_show_progress_integrates_existing_runner_emit(tmp_path, capsys):
    args = build_parser().parse_args([
        '--duration-minutes', '0.5', '--max-updates', '2',
        '--checkpoint', str(tmp_path / 'test.pt'),
        '--log-dir', str(tmp_path / 'run'),
    ])

    def fake_run(config, *, emit):
        emit({'kind': 'start', 'completed_updates': 3})
        emit({'kind': 'update', 'update': 4, 'hands_per_second_training': 42})
        emit({'kind': 'evaluation', 'opponent': 'random', 'mean_bb100': 0})
        emit({'kind': 'end', 'stop_reason': 'update_limit'})
        return {'phase': 11, 'completed_updates': 4}

    with patch('scripts.phase11_train.run_experiment', side_effect=fake_run):
        with patch('scripts.phase11_train.TrainingProgress') as mock_progress:
            ctx = mock_progress.return_value.__enter__.return_value
            result = run(args)
            assert ctx.on_event.call_count == 4
            assert ctx.message.call_count == 3
            assert result['completed_updates'] == 4


def test_progress_display_no_policy_side_effect():
    # Progress observer owns no torch/policy/engine resources.
    import inspect
    from holdem_ai.utils import progress
    source = inspect.getsource(progress)
    assert 'torch.' not in source
    assert 'collect_self_play' not in source
