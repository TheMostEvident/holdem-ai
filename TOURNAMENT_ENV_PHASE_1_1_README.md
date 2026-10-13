# Tournament ENV Phase 1.1 — Install

**Applies on top of** GitHub `TheMostEvident/holdem-ai` commit `d818f44` (or a descendant with the same GUI backend layout).

This is **Phase 1.1: FastAPI Tournament Engine integration**, *not yet* the React 6-max GUI or the trained Tournament Agent.

## Windows PowerShell

At the `holdem-ai` repo root:

```powershell
Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-1.zip" -DestinationPath . -Force
uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_tournament_env1_api.py
uv run --extra dev --extra gui --extra gpu python -m scripts.tournament_env1_smoke
uv run --extra dev --extra gui --extra gpu python -m pytest -q
```

Start backend:

```powershell
uv run --extra gui --extra gpu python -m backend.main
```

Visit `http://127.0.0.1:8000/docs` and try the new `local-6max-tournament` endpoints. The Heads-Up React GUI remains unchanged until Phase 1.2.

Full API contract and math: `docs/TOURNAMENT_ENV_PHASE_1_1.md`.

## Files added / modified

- **MODIFIED:** `backend/main.py`
- **NEW:** `backend/tournament_router.py`, `backend/tournament_schemas.py`
- **NEW:** `src/holdem_ai/tournament_service/{__init__,baseline,session}.py`
- **NEW:** `tests/test_tournament_env1_api.py`
- **NEW:** `scripts/tournament_env1_smoke.py`
- **NEW:** `docs/TOURNAMENT_ENV_PHASE_1_1.md`

**No changes** to Phase 4 tournament engine, existing HUNL200 Actor, RL training or installed React app.
