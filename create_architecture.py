#!/usr/bin/env python3
"""Create a non-destructive holdem-ai project scaffold.

Usage:
    python create_architecture.py
    python create_architecture.py --root path/to/holdem-ai

Existing files are never overwritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path

FOLDERS = [
    "src/holdem_ai/environment",
    "src/holdem_ai/agents",
    "src/holdem_ai/training",
    "src/holdem_ai/evaluation",
    "src/holdem_ai/ui",
    "tests/environment",
    "tests/agents",
    "data/hand_histories",
    "data/checkpoints",
    "scripts",
    "docs",
]

FILES = [
    ".gitignore",
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "main.py",
    "src/holdem_ai/__init__.py",
    "src/holdem_ai/environment/__init__.py",
    "src/holdem_ai/environment/hand_engine.py",
    "src/holdem_ai/environment/tournament_env.py",
    "src/holdem_ai/environment/game_state.py",
    "src/holdem_ai/environment/observation.py",
    "src/holdem_ai/environment/actions.py",
    "src/holdem_ai/environment/rewards.py",
    "src/holdem_ai/environment/seat_manager.py",
    "src/holdem_ai/environment/blind_manager.py",
    "src/holdem_ai/environment/pot_manager.py",
    "src/holdem_ai/agents/__init__.py",
    "src/holdem_ai/agents/base_agent.py",
    "src/holdem_ai/agents/random_agent.py",
    "src/holdem_ai/agents/cfr_agent.py",
    "src/holdem_ai/agents/rl_agent.py",
    "src/holdem_ai/training/__init__.py",
    "src/holdem_ai/training/self_play.py",
    "src/holdem_ai/training/trainer.py",
    "src/holdem_ai/evaluation/__init__.py",
    "src/holdem_ai/evaluation/benchmark.py",
    "src/holdem_ai/evaluation/tournament_eval.py",
    "src/holdem_ai/ui/README.md",
    "tests/test_scaffold.py",
    "tests/__init__.py",
    "tests/environment/__init__.py",
    "tests/environment/test_betting.py",
    "tests/environment/test_side_pot.py",
    "tests/environment/test_elimination.py",
    "tests/environment/test_position.py",
    "tests/agents/__init__.py",
    "scripts/README.md",
    "docs/architecture.md",
    "data/hand_histories/.gitkeep",
    "data/checkpoints/.gitkeep",
]

TEMPLATES = {
    ".gitignore": """# Python\n__pycache__/\n*.py[cod]\n*.egg-info/\n.pytest_cache/\n.ruff_cache/\n.venv/\nvenv/\n.env\n\n# Build and IDE\nbuild/\ndist/\n.idea/\n.vscode/\n\n# Generated training artifacts\ndata/hand_histories/*\n!data/hand_histories/.gitkeep\ndata/checkpoints/*\n!data/checkpoints/.gitkeep\n""",
    "README.md": """# holdem-ai\n\nA modular Python codebase for No-Limit Texas Hold'em environments and reinforcement-learning agents.\n\n## Objectives\n\n1. **Environment:** Build and validate the game rules, transitions, observations, payoffs, and multi-hand tournaments for heads-up and 2–6 players.\n2. **Agent:** Train and compare policies (random baseline, CFR-based methods, RL) through a stable agent interface.\n3. **Integration:** Support self-play, evaluation, replay data, and a future UI.\n\n## Setup\n\n```bash\npython -m venv .venv\n# macOS / Linux:\nsource .venv/bin/activate\n# Windows PowerShell:\n# .venv\\Scripts\\Activate.ps1\npython -m pip install -e '.[dev]'\npython -m pytest\n```\n\n> This repository is an initial **architecture scaffold**, not a working poker engine or tournament. RLCard integration and rules validation are next steps.\n\n## Directory layout\n\n- `src/holdem_ai/environment/`: hand engine, actions, observations, pots, tournament logic\n- `src/holdem_ai/agents/`: common agent interface and baseline/CFR/RL agents\n- `src/holdem_ai/training/`: training and self-play\n- `src/holdem_ai/evaluation/`: benchmarks and tournaments\n- `src/holdem_ai/ui/`: future frontend/API integration\n- `tests/`: environment and agent tests\n""",
    "requirements.txt": """# Runtime dependencies (pyproject.toml is the authoritative package config)\nrlcard==1.2.0\nnumpy>=1.24\n""",
    "pyproject.toml": """[build-system]\nrequires = [\"setuptools>=68\", \"wheel\"]\nbuild-backend = \"setuptools.build_meta\"\n\n[project]\nname = \"holdem-ai\"\nversion = \"0.1.0\"\ndescription = \"No-Limit Hold'em simulation and reinforcement learning platform\"\nreadme = \"README.md\"\nrequires-python = \">=3.10\"\ndependencies = [\n    \"rlcard==1.2.0\",\n    \"numpy>=1.24\",\n]\n\n[project.optional-dependencies]\ndev = [\"pytest>=8\"]\n\n[tool.setuptools.packages.find]\nwhere = [\"src\"]\n\n[tool.pytest.ini_options]\ntestpaths = [\"tests\"]\n""",
    "main.py": """\"\"\"Project entry point (engine integration pending).\"\"\"\n\n\ndef main() -> None:\n    print(\"holdem-ai scaffold is ready. Build and verify the environment next.\")\n\n\nif __name__ == \"__main__\":\n    main()\n""",
    "src/holdem_ai/__init__.py": """\"\"\"No-Limit Texas Hold'em AI platform.\"\"\"\n\n__version__ = \"0.1.0\"\n""",
    "src/holdem_ai/environment/__init__.py": '"""Poker hand and tournament environments."""\n',
    "src/holdem_ai/agents/__init__.py": '"""Policy and agent implementations."""\n',
    "src/holdem_ai/training/__init__.py": '"""Self-play and training pipelines."""\n',
    "src/holdem_ai/evaluation/__init__.py": '"""Game and tournament benchmarks."""\n',
    "src/holdem_ai/ui/README.md": """# UI integration\n\nFuture React frontend + Python API, separate from the training engine.\nKeep hidden cards and private information out of each player's observation.\n""",
    "tests/test_scaffold.py": """\"\"\"Test that architecture generation is repeatable and non-destructive.\"\"\"

from create_architecture import create_scaffold


def test_scaffold_creation_and_idempotency(tmp_path):
    root = tmp_path / "holdem-ai"
    created, existing = create_scaffold(root)
    assert created > 0
    assert existing == 0
    assert (root / "src/holdem_ai/environment/hand_engine.py").exists()
    assert (root / "src/holdem_ai/agents/base_agent.py").exists()

    readme = root / "README.md"
    readme.write_text("Keep my work", encoding="utf-8")
    created_again, existing_again = create_scaffold(root)
    assert created_again == 0
    assert existing_again == created
    assert readme.read_text(encoding="utf-8") == "Keep my work"
""",
    "scripts/README.md": "Helper scripts for setup, experiments, and validation will go here.\n",
    "docs/architecture.md": """# Architecture\n\n```text\nEnvironment -> player-specific observation + legal actions\nAgent       -> action\nEnvironment -> next state + payoff\nTraining    -> self-play trajectories + agent update\nEvaluation  -> benchmark and tournament results\nUI          -> sanitized game view, action input, replay\n```\n\n## Constraints\n\n- Maintain strict hidden-information boundaries.\n- Test 2–6 player betting rules, heads-up position transitions, and side-pot payouts.\n- Carry stacks across hands in tournament mode; eliminate players at zero stack.\n- Never assume the initial scaffold is a complete 6-max engine.\n""",
}


def create_scaffold(root: Path) -> tuple[int, int]:
    root = root.expanduser().resolve()
    for folder in FOLDERS:
        (root / folder).mkdir(parents=True, exist_ok=True)

    created = 0
    existing = 0
    for filename in FILES:
        target = root / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Exclusive creation: no risk of overwriting an existing file.
            with target.open("x", encoding="utf-8") as file:
                file.write(TEMPLATES.get(filename, ""))
            created += 1
        except FileExistsError:
            existing += 1
    return created, existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("holdem-ai"),
        help="Output project directory (default: ./holdem-ai)",
    )
    args = parser.parse_args()
    created, existing = create_scaffold(args.root)
    print(f"Architecture ready: {args.root.resolve()}")
    print(f"Files created: {created}; existing preserved: {existing}")


if __name__ == "__main__":
    main()
