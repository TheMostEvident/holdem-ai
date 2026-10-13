# Tournament ENV Phase 1.6 — Patch

Incremental patch after Tournament ENV Phase 1.5. Does **not** include pretrained checkpoint weights or any PPO training changes.

**New:** `.local/tournament_history.sqlite3` persistent SQLite journal; public frame replay; actor-visible NDJSON export; React archive/replay browser.

**Important:** Existing matches from before installing Phase 1.6 were memory-only and are not imported. A match interrupted by a Backend restart can be viewed but **not resumed**.

Full Korean technical guide: `docs/TOURNAMENT_ENV_PHASE_1_6.md`.

## Windows

```powershell
cd C:\Users\yiydg\Desktop\workspace\holdem-ai-starter\holdem-ai
Expand-Archive "$HOME\Downloads\holdem-ai-tournament-env-phase-1-6.zip" -DestinationPath . -Force
uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_tournament_env16_persistence.py
uv run --extra dev --extra gui --extra gpu python -m pytest -q
cd frontend
npm.cmd run typecheck
npm.cmd run build
npm.cmd run test
```

Do not push `.local` history DBs or exported actor JSONL files to GitHub.
