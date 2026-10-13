"""Unit tests for the privacy/staging release gate, no project game imports."""
from __future__ import annotations

from pathlib import Path
import subprocess

from scripts.tournament_env17_release_audit import check_release, unsafe_reason


def _git(root: Path, *args: str) -> None:
    subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)


def _project(tmp_path: Path) -> Path:
    (tmp_path / 'frontend').mkdir()
    (tmp_path / 'backend').mkdir()
    (tmp_path / 'frontend' / 'package.json').write_text('{}', encoding='utf-8')
    (tmp_path / 'backend' / 'main.py').write_text('', encoding='utf-8')
    (tmp_path / '.gitignore').write_text(
        '.local/\nruns/\ncheckpoints/\nfrontend/node_modules/\n'
        '*.pt\n*.pth\n*.jsonl\n*.sqlite3\n', encoding='utf-8',
    )
    _git(tmp_path, 'init', '-q')
    _git(tmp_path, 'add', '--', '.gitignore', 'backend/main.py', 'frontend/package.json')
    return tmp_path


def test_unsafe_patterns_block_private_artifacts():
    assert unsafe_reason('runs/expand_phase03_pilot/best.pt')
    assert unsafe_reason('.local/tournament_history.sqlite3')
    assert unsafe_reason('frontend/node_modules/react/index.js')
    assert unsafe_reason('checkpoints/model.pt')
    assert unsafe_reason('.env.local')
    assert unsafe_reason('data/hand_histories/hand.jsonl')
    assert unsafe_reason('frontend/src/tournament/TournamentApp.tsx') is None
    assert unsafe_reason('tests/test_tournament_env17_release_audit.py') is None
    assert unsafe_reason('data/hand_histories/.gitkeep') is None


def test_read_only_audit_works_for_safe_staging(tmp_path):
    root = _project(tmp_path)
    status, lines = check_release(root, strict=True)
    assert status == 0, '\n'.join(lines)
    assert not (root / '.local').exists()


def test_audit_blocks_explicitly_force_staged_checkpoint(tmp_path):
    root = _project(tmp_path)
    folder = root / 'runs' / 'expand_phase03_pilot'
    folder.mkdir(parents=True)
    (folder / 'best.pt').write_bytes(b'fake-checkpoint')
    _git(root, 'add', '-f', '--', 'runs/expand_phase03_pilot/best.pt')
    status, lines = check_release(root, strict=True)
    assert status != 0
    assert any('staged private/generated file' in line for line in lines)
