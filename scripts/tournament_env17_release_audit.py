"""Offline Git safety audit for Tournament ENV 1.7. Read-only; no git mutations.

Usage (from repo root):
    uv run python -m scripts.tournament_env17_release_audit
    uv run python -m scripts.tournament_env17_release_audit --strict

The checker does not inspect secrets inside arbitrary source text. Manual review
of staged diffs is still required before committing and pushing.
"""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import subprocess
import sys


DANGEROUS_DIRS = (
    '.local/', 'runs/', 'checkpoints/', '.venv/', 'venv/',
    'frontend/node_modules/', 'frontend/dist/', 'frontend/coverage/',
    'data/checkpoints/', 'data/hand_histories/',
)
DANGEROUS_EXT = (
    '.pt', '.pth', '.ckpt', '.safetensors', '.sqlite', '.sqlite3',
    '.sqlite3-wal', '.sqlite3-shm', '.jsonl', '.ndjson', '.parquet',
)
SAMPLE_PRIVATE_PATHS = (
    '.local/tournament_history.sqlite3',
    'runs/expand_phase03_pilot/best.pt',
    'checkpoints/private.pt',
    'frontend/node_modules/example.js',
)


def run_git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ['git', '-C', str(root), *args], capture_output=True, check=check,
    )


def unsafe_reason(path: str) -> str | None:
    clean = path.replace('\\', '/')
    parts = PurePosixPath(clean).parts
    name = parts[-1].lower() if parts else ''
    if name == '.gitkeep' and clean in ('data/hand_histories/.gitkeep', 'data/checkpoints/.gitkeep'):
        return None
    if clean.startswith(DANGEROUS_DIRS) or any(p in ('__pycache__', '.pytest_cache') for p in parts):
        return 'generated/private directory'
    if name == '.env' or name.startswith('.env.') and name != '.env.example':
        return 'environment secrets'
    if name.endswith(DANGEROUS_EXT):
        return 'model/data/binary artifact'
    if name.endswith(('.sqlite3-journal', '.db', '.db-wal', '.db-shm')):
        return 'database artifact'
    return None


def git_paths(root: Path, *args: str) -> list[str]:
    out = run_git(root, *args).stdout
    return [x.decode('utf-8', errors='replace') for x in out.split(b'\0') if x]


def check_release(root: Path, *, strict: bool = False) -> tuple[int, list[str]]:
    log: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    try:
        actual = Path(run_git(root, 'rev-parse', '--show-toplevel').stdout.decode().strip()).resolve()
    except (OSError, subprocess.CalledProcessError):
        return 2, ['FAIL: not in a Git repository']
    if actual != root.resolve():
        return 2, [f'FAIL: root mismatch: expected {root}, Git uses {actual}']
    if not (root / 'frontend' / 'package.json').is_file() or not (root / 'backend' / 'main.py').is_file():
        return 2, ['FAIL: not the holdem-ai project root']
    branch = run_git(root, 'branch', '--show-current').stdout.decode('utf-8', errors='replace').strip()
    log.append(f'Git root: {root}')
    log.append(f'Current branch: {branch or "detached HEAD"}')
    staged = git_paths(root, 'diff', '--cached', '--name-only', '-z', '--diff-filter=ACMR')
    tracked = git_paths(root, 'ls-files', '-z')
    risky_staged = [(p, unsafe_reason(p)) for p in staged if unsafe_reason(p)]
    risky_tracked = [(p, unsafe_reason(p)) for p in tracked if unsafe_reason(p)]
    for p, why in risky_staged:
        errors.append(f'BLOCK: staged private/generated file: {p} ({why})')
    for p, why in risky_tracked:
        errors.append(f'BLOCK: private/generated file is tracked: {p} ({why})')
    for relative in SAMPLE_PRIVATE_PATHS:
        ignored = run_git(root, 'check-ignore', '-q', '--no-index', '--', relative, check=False)
        if ignored.returncode != 0:
            warnings.append(f'IGNORE MISSING: {relative}')
    if staged:
        diff = run_git(root, 'diff', '--cached', '--check', check=False)
        if diff.returncode != 0:
            errors.append('BLOCK: git diff --cached --check failed (whitespace/conflict markers)')
    elif strict:
        errors.append('BLOCK: no files staged; run the curated stage script first')
    else:
        warnings.append('No staged changes yet (normal before staging).')
    remote = run_git(root, 'remote', 'get-url', 'origin', check=False)
    if remote.returncode == 0:
        log.append('Origin: ' + remote.stdout.decode('utf-8', errors='replace').strip())
    else:
        warnings.append('No origin remote configured.')
    log.append(f'Staged files: {len(staged)}; tracked files: {len(tracked)}')
    log.extend('WARNING: ' + line for line in warnings)
    log.extend(errors)
    # In non-strict mode, report missing ignore rules as warnings so the user
    # can inspect the repository without changing it. In strict mode, block push.
    if strict and any(w.startswith('IGNORE MISSING') for w in warnings):
        log.append('BLOCK: update .gitignore before pushing')
        errors.append('missing ignore patterns')
    return (2 if errors else 0), log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', action='store_true', help='fail if no staged files or unsafe ignore patterns')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    code, lines = check_release(root, strict=args.strict)
    print('\n'.join(lines))
    print('AUDIT:', 'FAILED' if code else 'PASSED (review staged diff manually)')
    return code


if __name__ == '__main__':
    sys.exit(main())
