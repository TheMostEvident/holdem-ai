# Phase 2: HUNL rules specification & diagnostic audit

## Status and scope

This phase **adds pure, testable reference rules and diagnostic tests**. It does
**not** change RLCard's internal rules or the Phase 1 `HandEngine`. Therefore
**the RLCard-backed game is not yet certified compliant** with standard HUNL.
An observed `xfail` for RLCard 1.2.0 positioning is expected. If it is an
`xpass`, inspect whether the installed RLCard version changed.

The upstream source reviewed:

- https://github.com/datamllab/rlcard/blob/master/rlcard/games/nolimitholdem/game.py
- https://github.com/datamllab/rlcard/blob/master/rlcard/games/nolimitholdem/round.py
- https://github.com/datamllab/rlcard/blob/master/rlcard/envs/nolimitholdem.py

## New files

| Module | Purpose |
|---|---|
| `src/holdem_ai/environment/seat_manager.py` | BTN/SB/BB and heads-up action order, blind posting |
| `src/holdem_ai/environment/actions.py` | Minimum bet/raise, stack cap, short all-in constraints |
| `src/holdem_ai/environment/pot_manager.py` | Side-pot layers, unmatched returns, ties and odd chips |
| `src/holdem_ai/environment/hand_evaluator.py` | Independent exact 7-card evaluator for reference showdowns |
| `src/holdem_ai/environment/rules_audit.py` | Non-mutating audit of raw RLCard state |
| `scripts/audit_rlcard.py` | CLI diagnostics (nonzero exit on upstream violations) |
| `tests/environment/test_*.py`, `tests/test_rules_audit.py` | Unit and integration tests |

## Core invariants

Initial heads-up positions (button `b`):

- `SB = b`
- `BB = 1 - b`
- `FirstActor(preflop) = b`
- `FirstActor(flop,turn,river) = 1 - b`

For no rake:

\[\sum_i \Delta C_i = 0\]

For side pots, including unmatched returns:

\[\sum_j \mathrm{Pot}_j + \sum_i U_i = \sum_i c_i\]

When the acting player has already put in `c`, current highest bet is `h`,
last full raise increment is `r`, and remaining stack is `s`:

\[\mathrm{Call}=\min(s,h-c)\]
\[\mathrm{MinRaiseTo}=\begin{cases}BB & h=0\\h+r&h>0\end{cases}\]
\[\mathrm{MaxRaiseTo}=c+s\]

A short all-in raise between the highest bet and minimum full raise can be
allowed, but it does **not** reopen action for a previously-acted player.
Tracking which player has acted in a street remains the full engine's duty.

## Run with uv

```powershell
uv sync --extra dev
uv run --extra dev python -m pytest -q
uv run python scripts/audit_rlcard.py --hands 100 --seed 42
```

Expect unit tests to pass. On the RLCard version known to have nonstandard
heads-up button/starting actor, the strict audit reports `FAIL` and exits with
code 1 **intentionally**. That result identifies work to do; it is not evidence
that our reference tests were unsuccessful.

## What remains unresolved

1. Replace or repair RLCard hand execution to match the verified rules.
2. Prove bet round closures, full/short all-in reopening and legal raises in
   hand-by-hand integration tests (pure constraints here are not enough).
3. Compare the included independent seven-card evaluator against RLCard showdown outcomes across known and randomized hands.
4. Verify all-in refunds and multi-way/side-pot settlement against actual engine.
5. Poker button rotation in a multi-hand match and 6-max elimination.

Do not start training a competitive Agent on the baseline until these engine
rules are validated; otherwise it can learn exploits of incorrectly modeled rules.
