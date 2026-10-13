# Tournament ENV Phase 1.2 — 6-Max React GUI

**Prerequisite:** Tournament ENV Phase 1.1 installed and tested (your logs: 11 API tests passed; 713 Python tests passed; Tournament smoke passed).

**This patch changes only frontend files and adds documentation.** No Python tournament rules, agent training code, checkpoints, package.json or package-lock.json are modified.

## Installation (PowerShell, project root)

```powershell
cd C:\Users\yiydg\Desktop\workspace\holdem-ai-starter\holdem-ai
Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-2.zip" -DestinationPath . -Force
```

## Build / test

```powershell
cd frontend
npm.cmd run typecheck
npm.cmd run build
npm.cmd run test
npm.cmd run dev
```

Open http://127.0.0.1:5173 → top switch **6-Max · Tournament** → **Start tournament**.

In another PowerShell window, start the backend:

```powershell
cd C:\Users\yiydg\Desktop\workspace\holdem-ai-starter\holdem-ai
uv run --extra gui --extra gpu python -m backend.main
```

The GUI uses the existing Phase 1.1 endpoints and currently plays against **five temporary Calling Station baseline bots**. Trained Tournament Agent and four Improved Bots belong to upcoming Tournament ENV 1.4 and 1.3 respectively.

- Tournament hands **advance automatically** after the server plays baseline bots; there is no next-hand button.
- Switching GUI environments keeps current React component state; browser reload or backend stop does **not** persist sessions.
- Match history is server-memory only until Tournament ENV 1.6.
- Opponents' hole cards remain hidden; the current API does not expose them even at showdown.

## Verification boundary

- Frontend source parses as TSX with TypeScript 5.8.3, and a synthetic type-check of the combined existing HUNL/new Tournament components completed without diagnostics.
- Pure Tournament action/accounting helpers were executed in Node (8 assertions passed).
- The 7 added Vitest test cases were **written but not run** here because npm registry DNS failed (`EAI_AGAIN`). Please run the three npm commands on your machine. In particular, do not equate the synthetic type-check with a full React dependency-backed build.
- Backend Phase 1.1 / existing Python regression results are from your Windows machine, not rerun in this patch build environment.

## Phase roadmap

1.3: Improved Bot ×4 policy integration
1.4: compatible Tournament Agent checkpoint inference integration
1.5: richer elimination / level / placement UX
1.6: permanent history and replay
1.7: full Windows validation and GitHub push

Detailed Obsidian-readable contract: `docs/TOURNAMENT_ENV_PHASE_1_2.md`.
