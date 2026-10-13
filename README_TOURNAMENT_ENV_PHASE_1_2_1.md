# Tournament ENV Phase 1.2.1 - Human card visibility fix

This hotfix changes ONLY `frontend/src/styles.css` and requires Tournament ENV Phase 1.2.

What was wrong:
- `.t-seat-cards .card-corner { display:none }` hid every card's rank.
- Human hole cards were placed below the bottom seat, partly clipped by `.t-arena { overflow:hidden }`.

What changed:
- Human hole cards are displayed ABOVE the Human seat (inside the arena).
- Human rank/suit corners are visible and cards are larger.
- Opponent hole cards remain face down and no game logic or API contracts change.
- Adds responsive sizing for <=850px and <=530px layouts.

Install from repository root (PowerShell):

```powershell
Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-2-cardfix.zip" -DestinationPath . -Force
cd frontend
npm.cmd run typecheck
npm.cmd run build
npm.cmd run test
npm.cmd run dev
```

Refresh the browser with Ctrl+Shift+R. Existing FastAPI server can remain running.
Expected: Human's two hole cards show rank (A/K/Q/J/T or 2-9) and suit; no clipping.
