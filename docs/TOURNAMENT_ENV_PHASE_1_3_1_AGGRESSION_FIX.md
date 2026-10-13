# Tournament ENV 1.3.1 — Bot Aggression Hotfix

## Why bots almost always called or folded

1. **Seat 2 (Agent placeholder):** Phase 1.3 explicitly used a `Calling Station`, which always preferred `check/call` when legal. It was **not** a trained Tournament Agent.
2. **Seats 3–6 (heuristics):** the preflop raising thresholds were too conservative after subtracting the penalty for five opponents. Several profiles rarely opened a raise.
3. **Real decision bug:** the previous expression `if value >= raise_threshold and 2 in legal` prevented a `Pot Raise` even when action `3` was legal and action `2` (`Half Pot`) was illegal. The correct test is `(2 in legal or 3 in legal)`.

## Changes

- Permit legal pot-size raise even if half-pot raise is not offered by the engine.
- Tune heuristic raise thresholds for visible, meaningfully different value-raising styles. These are **not** empirical equity estimates and no winning-performance claim is made.
- In the explicit `improved_4` lineup only, Seat 2 is still labeled `Tournament Agent placeholder`, but its **temporary untrained policy** now uses `controlled_pressure` heuristic rather than `Calling Station`.
- The original `baseline_5` lineup and its five Calling Stations remain untouched for backward compatibility.
- Updated frontend placeholder description and included synthetic raise-mix regression tests.
- No RL algorithm / trained Agent / checkpoint changes.

## Action map

| Native action | Meaning |
|---:|---|
| 0 | Fold |
| 1 | Check / Call |
| 2 | Raise half pot |
| 3 | Raise pot |
| 4 | All-in |

For a live acting Bot, the legal mask from `TournamentEnv.observe(actor)` is always authoritative:

$$a_t \in \mathcal A_t^{legal}.$$

### Diagnostic frequencies

Use:

```powershell
uv run --extra gui python -m scripts.tournament_env131_diagnose
```

The CLI enumerates 1,326 possible unordered hole-card combinations under an illustrative fixed pot, stack, and legal-action mask. The reported raise percentages are **conditional bot decisions** — they do not predict real-game raise frequency, equity, or bot strength. For a real-world benchmark, log full tournaments, positions, legal action sets, SPR, and actual actions, and compare to frozen baseline policies on matched seeds.

## Install

Close FastAPI (`Ctrl+C`), then from the project root:

```powershell
Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-3-1-aggression-fix.zip" -DestinationPath . -Force
uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_tournament_env13_improved_bots.py tests/test_tournament_env131_aggression.py
uv run --extra dev --extra gui --extra gpu python -m pytest -q
cd frontend
npm.cmd run typecheck
npm.cmd run build
npm.cmd run test
```

Start new backend and a **new tournament**; existing running sessions preserve their previously created Bot policy objects. `npm run dev` can hot reload the frontend, but the Backend must restart to load changed Python modules.

### Verification from a partial compatibility harness

- API/heuristic tests: **30 passed** (11 existing 1.1, 9 existing 1.3, 10 new 1.3.1).
- In the partial harness, 30 seeded `improved_4` tournaments (388 hands overall) all ended and preserved chips. Observed bot raise (half-pot/pot, excluding all-in) percentages by engine seat **1–5**: **29.1%, 10.4%, 19.6%, 11.2%, 28.5%**. These figures are demonstration diagnostics, not win rates, not a controlled baseline comparison, and **not** validation against the complete Windows repository.
- Supplemental frontend `tsc -p tsconfig.verify.json --noEmit` completed. The official React build/Vitest and the full Python regression suite remain to be run on the user Windows machine.
