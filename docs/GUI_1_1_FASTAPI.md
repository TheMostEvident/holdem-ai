---
title: "GUI-1.1 — HUNL Local HTTP Game Server"
tags: [holdem-ai, gui, hunl, fastapi, research]
status: implemented-patch
---

# GUI-1.1 — FastAPI Local Game Server

## Scope

**GUI-1.0 is assumed installed and passing.** This phase adds HTTP transport to the already existing `HUNLGameSession`. It implements no new poker betting rules. A React poker table, arbitrary human `raise_to`, and trained checkpoint loading belong to later phases.

- Human fixed to physical seat 0; Bot physical seat 1.
- Two-player HUNL200: each hand starts with 200BB and rotates BTN between hands.
- Four available built-in bot baselines: `random`, `calling_station`, `tight`, `pot_pressure`.
- Revision-based optimistic concurrency; stale action and double next-hand clicks are rejected.
- Session IDs are cryptographically unpredictable; stored **in process only** (server restart destroys sessions).
- HTTP responses are schema-whitelisted player projections. Before showdown, opponent hole cards and deck are never serialized. Opponent cards are shown only after a genuine showdown.
- Bind to `127.0.0.1`, CORS permits localhost Vite development origins. This server has **no authentication and must never be published to a remote or public interface**.

## Endpoint Contract

| Method | Path | Input | Output |
| --- | --- | --- | --- |
| `GET` | `/api/v1/hunl/health` | None | Health JSON |
| `POST` | `/api/v1/hunl/sessions` | `bot`, optional `seed`, `small_blind`, `big_blind`, `stack_bb` | `{session_id, state}` with HTTP 201 |
| `GET` | `/api/v1/hunl/sessions/{id}` | None | Current `{session_id, state}` |
| `POST` | `/api/v1/hunl/sessions/{id}/actions` | `action_index`, `expected_revision` | Updated `{session_id, state}` (bot turns auto-advanced) |
| `POST` | `/api/v1/hunl/sessions/{id}/next-hand` | `expected_revision` | Next hand `{session_id, state}` |
| `DELETE` | `/api/v1/hunl/sessions/{id}` | None | HTTP 204 |

**HTTP status semantics:** `422` malformed or illegal action, `409` stale revision or invalid game phase, `404` unknown session, `503` session capacity exceeded.

## Player-Visible State

The frontend receives `{session_id, state}`. The `state` object exposes only the following properties:

`hand_number`, `revision`, `phase`, `game_type`, `hero_seat`, `button`, `small_blind`, `big_blind`, `street`, `board`, `hero_cards`, `seats`, `pot`, `to_act`, `your_turn`, `actions`, `to_call`, `min_raise_to`, `max_raise_to`, `action_log`, `completed_hands`, `cumulative_bb`, `bb_per_100`, `hand_result`.

When `your_turn` is true, `actions` offers valid **seven-head HUNL200 action indices**, not Phase-3 engine action IDs. Each raise action includes an exact `raise_to` resolved by the environment. **Do not use the GUI action index as a raw engine action ID.**

When `phase = complete`, `hand_result.showdown` controls whether `opponent_cards` is present. After fold, cards stay hidden.

The client must always send the **most recently received** `expected_revision`; never guess it. An HTTP 409 should trigger a GET refresh.

## Mathematical Definitions

For hand $h$, the human's net payoff in big blinds is

$$
R_h^{\mathrm{BB}}=\frac{C_{h,\mathrm{final}}-C_{h,\mathrm{initial}}}{BB_h}.
$$

Across $N$ completed hands, the GUI displays

$$
\widehat{\mathrm{bb/100}}=\frac{100}{N}\sum_{h=1}^{N}R_h^{\mathrm{BB}}.
$$

Per-hand stacks reset to $200BB$ as in the existing HUNL200 research environment; this is a repeated-hand study, not a continuous cash table with accumulated stacks and not a tournament.

## Installation and Tests (PowerShell; project root)

First extract GUI-1.0 and then GUI-1.1. Add the dedicated GUI optional dependencies **once** (these modify `pyproject.toml` and `uv.lock` locally and avoid overwriting existing customized manifests):

```powershell
Expand-Archive -Path "$HOME\Downloads\holdem-ai-gui-1-0.zip" -DestinationPath . -Force
Expand-Archive -Path "$HOME\Downloads\holdem-ai-gui-1-1.zip" -DestinationPath . -Force
uv add --optional gui "fastapi>=0.115,<1" "uvicorn>=0.30,<1" "httpx>=0.27,<1"
uv sync --extra dev --extra gui
uv run --extra dev --extra gui python -m pytest tests/test_gui10_hunl_contract.py tests/test_gui11_http_api.py -q
uv run --extra dev --extra gui python -m pytest -q
uv run --extra dev --extra gui python -m scripts.gui11_smoke --hands 10 --seed 42
```

Start local API server:

```powershell
uv run --extra dev --extra gui python -m backend.main
```

Open `http://127.0.0.1:8000/docs` for Swagger OpenAPI. Stop with `Ctrl+C`.

## Example Requests (PowerShell)

```powershell
$base = 'http://127.0.0.1:8000/api/v1/hunl'
$new = Invoke-RestMethod -Method Post -Uri "$base/sessions" -ContentType 'application/json' -Body '{"bot":"calling_station","seed":42}'
$new.state | ConvertTo-Json -Depth 8
$id = $new.session_id
$action = $new.state.actions | Where-Object { $_.kind -eq 'call' -or $_.kind -eq 'check' } | Select-Object -First 1
$body = @{ action_index = $action.index; expected_revision = $new.state.revision } | ConvertTo-Json
$updated = Invoke-RestMethod -Method Post -Uri "$base/sessions/$id/actions" -ContentType 'application/json' -Body $body
$updated.state | ConvertTo-Json -Depth 8
```

This example chooses Check/Call **only when the server offers it**. The state may be terminal after actions and Bot automatic continuation.

## New Code

- `backend/__init__.py`, `backend/main.py`, `backend/schemas.py`, `backend/game_router.py`
- `src/holdem_ai/game_service/session_manager.py` (GUI-1.0 file, **one additive method** `start_next_hand` for atomic revision check)
- `tests/test_gui11_http_api.py`
- `scripts/gui11_smoke.py`

## Risks and Deliberate Limitations

1. No checkpoints loaded yet (GUI-1.4). All allowed Bot types are simple baseline policies.
2. No web poker table yet (GUI-1.2 onward).
3. Only fixed seven-head Human actions; unrestricted raise-to will be added after an explicit safe HUNL200Adapter API is designed.
4. Session registry is volatile and process-local; no persistence, multi-process coordination, remote hosting, or authentication.
5. Tests establish expected behavior for exercised cases, **not** complete poker rules certification.
6. The model's strongest available inference is still strictly HUNL (two-player), not six-player.
