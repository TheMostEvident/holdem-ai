# Tournament ENV Phase 1.1 — 6-Max Tournament API Integration

> Workstream: **Environment only**. Baseline commit: `d818f44`. No RL training or weight updates.

## Objective

Reuse the *existing* Phase 3 `MultiPlayerHandEngine` and Phase 4 `TournamentEnv` through a local FastAPI service. Keep the HUNL200 independent-hand training stack and its existing `/api/v1/hunl/*` endpoints unchanged.

The long-term table will be **Human (seat 0) + Tournament Agent (seat 1) + four Improved Bots (seats 2–5)**. **Phase 1.1 does not yet load any tournament-trained checkpoint or Improved Bot**: seats 1–5 temporarily use explicit `baseline_bot` policies (`calling_station`). In particular, a 2-player HUNL200 Actor with a 7-action head MUST NOT be silently used as a 6-max 5-action policy.

### Implementation delivered

- Session manager: `src/holdem_ai/tournament_service/session.py`
- Baseline policy adapter: `src/holdem_ai/tournament_service/baseline.py`
- FastAPI routes and strict JSON schemas: `backend/tournament_router.py`, `backend/tournament_schemas.py`
- Application integration: `backend/main.py`
- API acceptance tests: `tests/test_tournament_env1_api.py`
- End-to-end API smoke: `scripts/tournament_env1_smoke.py`

### Environment invariants

For a six-seat, no-rake, no-rebuy tournament:

$$
\sum_{i=0}^{5} S_{i,h}^{\mathrm{start}} = C_{\mathrm{total}},\qquad
S_{i,h+1}^{\mathrm{start}}=S_{i,h}^{\mathrm{final}}.
$$

During active betting, part of the chips belongs to the pot rather than live player stacks:

$$
\boxed{\sum_{i=0}^{5}S_{i,t}^{\mathrm{behind}}+P_t=C_{\mathrm{total}}}
$$

After pot settlement:

$$
\boxed{\sum_{i=0}^{5}S_{i,h}^{\mathrm{final}}=C_{\mathrm{total}}}
$$

Once a player's final stack is zero, their physical seat remains present but they no longer receive cards. The existing Phase 4 engine governs dealer movement, blinds, elimination ordering, side pots, short all-in handling, and transition to heads-up play.

### HTTP API

New namespace: `/api/v1/tournament`, independent of `/api/v1/hunl`.

| Method | Path | Meaning |
|---|---|---|
| GET | `/health` | Local health/version |
| POST | `/sessions` | Create 6-seat tournament |
| GET | `/sessions/{id}` | Read safe human-oriented view |
| POST | `/sessions/{id}/actions` | Submit one of the Phase-4 legal actions |
| POST | `/sessions/{id}/raise-to` | Exact human raise-to in chips |
| GET | `/sessions/{id}/history?offset=0&limit=50` | Completed-hand public summaries |
| DELETE | `/sessions/{id}` | Remove in-memory tournament |

Example session creation:

```powershell
$session = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/tournament/sessions -ContentType 'application/json' -Body '{"seed":42,"starting_stack":1000,"hands_per_level":10}'
$session.state.seats | Format-Table
$session.state.actions | Format-Table
```

Example call/check (only valid if included in the returned action offers):

```powershell
$id = $session.session_id
$rev = $session.state.revision
$body = @{action_index=1; expected_revision=$rev} | ConvertTo-Json
$session = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/tournament/sessions/$id/actions" -ContentType 'application/json' -Body $body
```

The HTTP request must always include the last observed `expected_revision`. Stale or duplicated submissions receive HTTP `409`, invalid game actions receive `422`, and missing sessions return `404`.

#### Native tournament action indices (Phase 4)

| ID | Engine action |
|---|---|
| 0 | Fold |
| 1 | Check / Call |
| 2 | Half-pot raise (if legal) |
| 3 | Pot raise (if legal) |
| 4 | All-In |
| 5 | Arbitrary raise-to (only through `/raise-to`, **not** `/actions`) |

**Important:** HUNL200 neural Actor uses a *different 7-action abstraction*. The indices are not interchangeable. The Phase 1.4 integration must explicitly translate between policy outputs and 6-max engine-legal decisions or use a natively trained tournament policy.

#### Response contract

Human-visible fields include the six public seats, behind-stacks, street bets, pot, public community cards, hero's private cards (seat 0 only), blind level, turn, legal actions, revision, completed-hands count and public hand summaries. No other player's hole cards, burn cards, shuffled deck, or training CriticState are serialized.

`TournamentEnv` automatically opens the next hand after the preceding hand finishes. There is deliberately **no `/next-hand` endpoint** in Phase 1.1; an explicit inter-hand pause is a potential 1.2/1.5 UI refinement.

### Verification

```powershell
# At repo root, after extracting the patch ZIP
uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_tournament_env1_api.py
uv run --extra dev --extra gui --extra gpu python -m scripts.tournament_env1_smoke
uv run --extra dev --extra gui --extra gpu python -m pytest -q
```

The initial local packaging environment used a partial Phase 3/4 project snapshot and shims for dependencies that exist in the user's full repository. Local 11 API tests passed in that simulated compatibility environment. **Final real-repository validation must be run on the user's Windows checkout**; do not treat partial-shim verification as complete system certification.

### Non-goals / next phases

- **Phase 1.2:** Six-seat React tournament table, betting buttons, elimination display.
- **Phase 1.3:** Wire the designated four Improved Bots, once their policy implementation is located and the action contract is verified.
- **Phase 1.4:** Load compatible *Tournament* Agent checkpoint, safely map action semantics and expose policy ID in GUI. No silent Random fallback.
- **Phase 1.5:** Tournament UX: blind progression, placements, last hand and champion.
- **Phase 1.6:** Durable public/action-observation trajectories, hand replay/reproduction and data export contracts.
- **Phase 1.7:** Windows regression / frontend build / real game acceptance / GitHub Push.

For now, all sessions are local in-memory. Server restart loses active tournaments and histories. The server is unauthenticated and should bind to `127.0.0.1`, not a public network interface.
