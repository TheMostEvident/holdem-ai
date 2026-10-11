# holdem-ai

A modular Python codebase for No-Limit Texas Hold'em environments and reinforcement-learning agents.

## Objectives

1. **Environment:** Build and validate the game rules, transitions, observations, payoffs, and multi-hand tournaments for heads-up and 2–6 players.
2. **Agent:** Train and compare policies (random baseline, CFR-based methods, RL) through a stable agent interface.
3. **Integration:** Support self-play, evaluation, replay data, and a future UI.

## Setup

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python -m pytest
```

> This repository is an initial **architecture scaffold**, not a working poker engine or tournament. RLCard integration and rules validation are next steps.

## Directory layout

- `src/holdem_ai/environment/`: hand engine, actions, observations, pots, tournament logic
- `src/holdem_ai/agents/`: common agent interface and baseline/CFR/RL agents
- `src/holdem_ai/training/`: training and self-play
- `src/holdem_ai/evaluation/`: benchmarks and tournaments
- `src/holdem_ai/ui/`: future frontend/API integration
- `tests/`: environment and agent tests
