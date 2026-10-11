# Phase 2.1 — Corrected RLCard HUNL adapter

## Key goal

RLCard 1.2.0's upstream two-player game posts blinds as if a multiplayer game were in progress. `holdem-ai` fixes the actual chip posting and first actor inside a local subclass. It leaves the installed `.venv/` untouched.

The new `HeadsUpBettingRound` also enforces a full minimum raise and non-reopening after a short all-in, while retaining action IDs `0=FOLD`, `1=CHECK_CALL`, `2=RAISE_HALF_POT`, `3=RAISE_POT`, `4=ALL_IN`. For raise-to sizing, raise half-pot/pot is calculated from the pot after calling. The local `get_payoffs()` settles unmatched chips and ties with Phase 2's `pot_manager` and `hand_evaluator`.

## Rules

For Button `b` (player ID 0 or 1), big blind is `1-b`. Preflop Button acts first, postflop big blind acts first.

```text
preflop:   BTN/SB -> BB
postflop:  BB -> BTN
```

## Running

```powershell
uv sync --extra dev
uv run --extra dev python -m pytest -q
uv run python scripts/audit_corrected.py --hands 100 --seed 42
uv run python main.py --hands 5 --seed 42
```

**Expected:** tests for the corrected engine pass. The older `tests/test_rlcard_rules_integration.py` includes an upstream `xfail`; that remains expected because upstream RLCard is deliberately unchanged. Likewise, `scripts/audit_rlcard.py` should continue to report failures for unmodified upstream RLCard. The new `scripts/audit_corrected.py` must pass.

## Repository contents changed by this patch

- Modified: `src/holdem_ai/environment/hand_engine.py`, `main.py`, `tests/test_hand_engine.py`
- New: `src/holdem_ai/environment/corrected_game.py`, `corrected_round.py`, `tests/test_corrected_round.py`, `tests/test_corrected_rlcard_integration.py`, `scripts/audit_corrected.py`, and this document.

## Limitations

- Heads-up single hands only, not 3-6 player cash games or tournaments.
- RLCard's original 5 discrete betting actions are still used; unrestricted raise sizing requires a new Agent action API.
- `obs` from RLCard remains a 54-element observation; do not rely on it as a full Information State for CFR.
- Chip integrity and selected rule scenarios are covered, but this should **not** be considered an independently certified production Hold'em engine. More property tests and integration with a verified reference engine are needed.
- Current patched implementation is designed for `rlcard==1.2.0`, with version guard, and should be verified again on actual Windows installation before merging.
