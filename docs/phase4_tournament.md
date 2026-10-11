# Phase 4 — Tournament Environment

Single-table No-Limit Texas Hold'em tournament for **2–6 physical seats** using
Phase 3 `MultiPlayerHandEngine`. All amounts are integer chips.

## Main interface

```python
from holdem_ai.environment.tournament_env import TournamentEnv
from holdem_ai.agents.random_agent import RandomAgent

env = TournamentEnv(num_players=6, starting_stack=1000,
                    blind_levels=[(5, 10), (10, 20), (20, 40),
                                  (40, 80), (80, 160), (160, 320),
                                  (320, 640), (640, 1280), (1280, 2560),
                                  (2560, 5120), (5120, 10240)],
                    hands_per_level=10, seed=42)
result = env.play_tournament([RandomAgent(i) for i in range(6)])
print(result.winner, result.ranks, result.hands_played)
```

Incremental interface supports `reset()`, `state`, `current_player`,
`observe(player_id)`, `step(action_id)`, `step_raise_to(chips)`, `is_over()`,
`get_result()`; after a hand completes, `step` automatically starts the next
hand. An action is always applied to the currently acting seat.

## Invariants

For hand `h`, `stacks_before` equals prior hand `stacks_after`:

$$ C_{i,h+1}^{start} = C_{i,h}^{end} $$

No outside chips, no rake, and no rebuy:

$$ \sum_{i=0}^{n-1} C_{i,h} = \sum_{i=0}^{n-1} C_{i,0} $$

A bust has zero stack; busted seats remain as zero-valued physical seat IDs.

$$ E_{i,h} = \mathbf{1}[C_{i,h}>0], \qquad \text{done}=\left(\sum_i E_{i,h}=1\right) $$

`TournamentResult.rewards` defaults to `(1, 0, ..., 0)` permuted to the champion.
Custom rank-specific payouts may be supplied with `payouts=(..., ...)`.
Simultaneously busted entrants are ordered by **pre-hand stacks**; equal
pre-hand stacks tie and split the prizes for their occupied place slots.

## Development limitations

- **Moving button** to next live seat; not a formal dead-button tournament policy.
- **No ante, bounty, rebuy, add-on, ICM, time-based levels, or multi-table balancing.**
- Blinds advance every `hands_per_level` completed hands; last level repeats.
- `max_hands` and `max_actions` are safety limits, not termination substitutes.
- The underlying Phase 3 game engine still needs extended rules verification.
- All private hole cards are exposed only to the acting player through `observe`.
- This is not certified for real-money operations.

## Commands (Windows PowerShell / uv)

```powershell
uv sync --extra dev
uv run --extra dev python -m pytest -q
uv run python scripts/run_tournament.py --players 6 --seed 42
uv run python scripts/audit_tournament.py --players 6 --tournaments 100 --seed 42
```
