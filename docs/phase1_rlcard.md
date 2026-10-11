# Phase 1 — RLCard HUNL baseline

This patch updates the **existing** `holdem-ai` repository. It does not create another Git repository.

## Setup (PowerShell, from `holdem-ai/`)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python main.py --hands 5 --seed 42
python -m pytest -q tests/test_hand_engine.py tests/test_rlcard_integration.py
```

If PowerShell activation is blocked, run `.\.venv\Scripts\python.exe` instead of `python` for each command, without activating.

## Important limitations

- This adapter wraps **one hand of two-player** no-limit Hold'em. It is **not** a 6-player tournament.
- RLCard uses a simplified set of betting actions and needs independent testing for heads-up blinds, positions, raise rules and showdown behavior before serious strategy training.
- `get_result()` measures chip changes relative to starting stack. `bb/100 = 100 * sum(payoff_bb) / hands` for equal-length hands and fixed blinds.
- Do **not** give an agent `env.game`, `get_perfect_information()` or the opponent's hole cards; the intended agent input is RLCard's acting-player `state` only.
- The mock tests validate the Python adapter, but **real RLCard integration must be run on your local machine**.
- If you previously removed `create_architecture.py`, also remove or update the old `tests/test_scaffold.py`, which imports that script.
