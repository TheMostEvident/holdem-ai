# Phase 3 patch (overlay into the existing `holdem-ai/` root)

Only files included in this ZIP are new. Phase 2.1 HUNL modules remain unchanged.

Files:
- `src/holdem_ai/environment/multi_player_seats.py`
- `src/holdem_ai/environment/multi_player_engine.py`
- `tests/test_multi_player_seats.py`
- `tests/test_multi_player_engine.py`
- `scripts/audit_multiplayer.py`
- `docs/phase3_multiplayer.md`

No new dependencies. The same `uv sync --extra dev` and pytest commands work.

The existing `main.py` continues to run the corrected 2-player RLCard game; the new
six-player smoke benchmark is `uv run python scripts/audit_multiplayer.py`.
