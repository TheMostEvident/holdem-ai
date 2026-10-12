---
title: "holdem-ai — GUI-1.2 React HUNL Poker Table"
tags: [holdem-ai, HUNL, GUI, React, FastAPI]
status: implementation-awaiting-user-validation
---

# GUI-1.2 — Heads-Up Poker Table

## Objective

Build the **read-only** browser-based HUNL poker table on top of the GUI-1.0/1.1 FastAPI contract. Keep all poker rules in the original `HUNL200Adapter` / `MultiPlayerHandEngine` and only render the explicitly projected human-visible JSON data. This phase does **not** allow betting clicks; interactive betting is GUI-1.3.

$$
\text{Environment}\xrightarrow{O_{\mathrm{hero}}}\text{FastAPI GameView}\xrightarrow{\mathrm{GET/POST}}\text{React PokerTable}
$$

## Implemented UI

- Responsive oval green-felt table with two seats: Human seat 0 and AI seat 1.
- Hero 2 hole cards, public 0/3/4/5 board cards, opponent face-down cards.
- Opponent cards reveal **only** for `phase=complete` and `hand_result.showdown=true`.
- Dealer (BTN) marker, SB/BB, pot, total stack, street bets, active seat, hand number, action history, cumulative BB and bb/100.
- Local session create, refresh, delete, and next hand (only if previous hand complete).
- Bot selection from GUI-1.1: random, calling_station, tight, pot_pressure; seed optional.
- Design preview switch with clearly marked *fictional sample fixtures* for preflop, flop, turn, river, showdown and folded finish.
- Server health/error feedback; Vite proxy /api -> http://127.0.0.1:8000.
- Vitest utility tests and a Python real-backend JSON/CORS/privacy contract test.

## Deliberate scope exclusions

- GUI-1.3: Human action submission, raise slider, keyboard shortcuts.
- GUI-1.4: Trained PPO/Q-Boost checkpoint loading.
- GUI-1.5: Interactive game history/dashboard expansion.
- GUI Phase 2: Custom Scenario Lab.
- Tournament, 6-Max, external live poker integration.

**Important:** GUI-1.2's live game starts on preflop and stays on the human decision until GUI-1.3 adds interactive betting. The view-mode demo selector is **not** a simulation and does not change backend state.

## Dependencies and local Windows run

This GUI-1.2 ZIP includes only `frontend/`, `tests/test_gui12_readonly_api.py`, and documentation. If GUI-1.0/1.1 is not installed, use the unified GUI-1.0–1.2 archive instead. From project root:

```powershell
uv add --optional gui "fastapi>=0.115,<1" "uvicorn>=0.30,<1" "httpx>=0.27,<1"
uv sync --extra dev --extra gui
uv run --extra dev --extra gui python -m pytest tests/test_gui10_hunl_contract.py tests/test_gui11_http_api.py tests/test_gui12_readonly_api.py -q
uv run --extra dev --extra gui python -m pytest -q
```

Install a supported Node.js LTS (Node 20.19+ or 22+). In PowerShell terminal **A**:

```powershell
uv run --extra gui python -m backend.main
```

Open PowerShell terminal **B** at the project root:

```powershell
cd frontend
npm install
npm run test
npm run build
npm run dev
```

Then visit http://127.0.0.1:5173. When you see *API ONLINE*, click **Deal a new game** to create a live HUNL200 hand. Use **Preview** to inspect street and showdown layouts without playing any actions. The Vite development server's `/api` proxy will forward to FastAPI.

**Runtime limitations:** The FastAPI server is an in-memory unauthenticated local development server bound to 127.0.0.1. Never expose it to the Internet. Node packages have not been installed or built in the assistant environment because its npm registry network access failed; `npm install`, `npm run test`, and `npm run build` must be run on the target PC.

## Source-of-truth and privacy

Client receives `GameResponse { session_id, state }` from GUI-1.1. It **never** receives `CriticState`, an undealt deck, opponent private cards, or arbitrary engine object dumps. Do not import `MultiPlayerHandEngine` into the frontend. `opponentCardsToRender` shows cards only when `hand_result.showdown` is true and the game is complete. A folded opponent is never exposed.

## Incremental delivery

Next: GUI-1.3 betting controls use `GameView.actions` (index 0..6), POST `/api/v1/hunl/sessions/{id}/actions` with `expected_revision`; reconcile on HTTP 409, and use engine-derived raise amounts only.
