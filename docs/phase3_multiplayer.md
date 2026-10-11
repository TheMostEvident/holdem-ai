# Phase 3 - Multi-Player Hand Engine (2-6 seats)

New independent pure-Python engine: `holdem_ai.environment.multi_player_engine.MultiPlayerHandEngine`.

- Does NOT modify the existing Phase 2.1 `HandEngine` (HUNL/RLCard adapter).
- Uses existing exact `rank_showdown` and side-pot utilities (`pot_manager.py`).
- Integer chips, no rake. Five compatible action IDs: `0=fold`, `1=check/call`, `2=half-pot raise`, `3=pot raise`, `4=all-in`.
- Also supports `step_raise_to(amount)` for explicit bet sizing (street-total target). Such actions are logged with ID `5` and their actual targets appear in `action_raise_to`; preserve both for replay/ML.
- Seat indices persist; stack zero means absent player. Button must have chips. `rotate_button()` uses a moving-button model; **dead button tournament rules are not implemented**.
- Correct heads-up special case (BTN = SB; BTN acts first preflop); with 3+ players SB is left of button, BB is left of SB, preflop starts left of BB, postflop starts left of button.
- Real shuffled 52-card deck, two cards per occupied seat, one burn before flop/turn/river.
- Player-private `observe(player_id)` reveals only the acting player's hole cards. This is a different schema than RLCard's 54-dimensional encoded observation.
- Every `reset()` starts a *new hand* with the constructor's stacks; persistent tournament stacks, ante, blind levels, ranks/payouts belong in Phase 4.
- No `step_back` for CFR yet. A deep-copy/serialized state or undo-log API will be developed separately.
- Uses `build_side_pots` and `settle_pots` with individual per-layer eligibility, matched wager refunds, ties and integer odd-chip rules.

## Install and test with uv (PowerShell)

```powershell
uv sync --extra dev
uv run --extra dev python -m pytest -q
uv run python scripts/audit_multiplayer.py --players 6 --hands 1000 --seed 42
```

## Agent use

```python
from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine

env = MultiPlayerHandEngine(num_players=6, starting_stack=200, button=0, seed=42)
result = env.play_hand([RandomAgent(i) for i in range(6)])
print(result.payoffs_bb, result.final_stacks)
```

## Amount and payoff invariants

- `sum(final_stacks) == sum(initial_stacks)`
- `sum(payoffs_chips) == 0`
- `payoff[i] == awards[i] - contribution[i]`
- `final_stack[i] == initial_stack[i] + payoff[i]`

## Important limits

This is a new testing baseline, *not* a certified poker engine. Special cases requiring ongoing validation include cumulative short-all-in re-open decisions, partial blinds, tournament blind obligations when seats bust, cards burning and odd chips. Tournament orchestration and persistent stack ownership will be implemented in Phase 4.

Standard raise/betting references:
- https://www.pokertda.com/view-poker-tda-rules/

`reset()` may return `(None, -1)` if everyone is already all-in after posting blinds and the hand is automatically settled. `play_hand()` supports this case.
