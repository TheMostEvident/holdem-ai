# Tournament ENV Phase 1.3 — Four Heuristic Improved Bot Slots

> **Scope:** Environment only. This patch neither trains nor updates any PPO/Q-Boost/Tournament Agent weights.
>
> **Status:** Implementation and partially isolated integration tests done. Final Windows `pytest` / React production build required.

## 1. Goal and honest scope

The GUI lineup is now:

| Physical seat (UI) | Engine seat ID | Environment role | Deployed policy |
|---|---:|---|---|
| 1 | 0 | Human | Browser actions |
| 2 | 1 | **Tournament Agent placeholder** | Calling Station baseline (**not** a trained 6-max Agent) |
| 3 | 2 | Improved Bot 1 | `tight_aggressive` heuristic |
| 4 | 3 | Improved Bot 2 | `balanced_value` heuristic |
| 5 | 4 | Improved Bot 3 | `pot_odds` heuristic |
| 6 | 5 | Improved Bot 4 | `controlled_pressure` heuristic |

**Important:** "Improved Bot" is a development role/name, **not proof** that any of these heuristic bots beats the existing Calling Station, random opponent, or a human. The next independent benchmark must establish the relative performance before claiming an improvement. No hidden card is used to select an action.

### Legacy compatibility

`TournamentGameSession()` and the API create request still default to `lineup="baseline_5"` for compatibility with Phase 1.1 tests, logs and scripts. **Only the Phase 1.3 React GUI** explicitly creates `lineup="improved_4"`.

## 2. Environment-level decision heuristic

For acting seat $i$, read only the data returned by `TournamentEnv.observe(i)`:

$$
I_t^i = (h_i, B_t, P_t, c_i, S_i, \mathcal A_t^{legal}, \text{public statuses})
$$

The simple preflop score uses private ranks, pair/suitedness and rank separation. Postflop, the score is based on a **public + own-card** best-five-card ranking and simple draw cues. It is **not** exact multiway equity and does not account fully for blockers, ICM, GTO or opponent ranges.

Call price and offered pot odds:

$$
q_t=\frac{\min(c_i,S_i)}{P_t+\min(c_i,S_i)}
$$

All profile thresholds are heuristic. Returned action must be a member of the **native Tournament action set**:

$$
a_t \in \mathcal A_t^{legal} \subseteq \{0,1,2,3,4\}
$$

Indices: `0=fold`, `1=check/call`, `2=half-pot raise`, `3=pot raise`, `4=all-in`. A free arbitrary `raise_to` action is reserved for the human and checked by the engine. The HUNL200 Actor's seven-action distribution is **never silently mapped** to this five-action interface.

Tournament-wide no-rake chip conservation remains:

$$
\sum_{i=0}^{5}S_{i,t}^{behind}+P_t=C_{total}
$$

The bot profiles use only the observation provided to them and independent seeded random generators for minor tie variation; they cannot access the engine's full deck or the other players' hidden cards.

## 3. API changes (additive)

Existing endpoints are unchanged. The creation request adds an optional `lineup`:

```json
{
  "seed": 42,
  "starting_stack": 20000,
  "button": 0,
  "hands_per_level": 100,
  "max_hands": 10000,
  "lineup": "improved_4"
}
```

`GET /api/v1/tournament/sessions/{id}` includes `state.lineup` and, per seat, `role` and `policy_name` (safe metadata only). Supported role values are `human`, `baseline_bot`, `tournament_agent_placeholder`, and `improved_bot`.

Old clients that omit `lineup` receive five Calling Station baseline seats as before. Invalid lineup values are refused with `422`.

## 4. Install, test and run (PowerShell from repository root)

```powershell
Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-3.zip" -DestinationPath . -Force
uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_tournament_env1_api.py tests/test_tournament_env13_improved_bots.py
uv run --extra dev --extra gui --extra gpu python -m scripts.tournament_env13_smoke
uv run --extra dev --extra gui --extra gpu python -m pytest -q
cd frontend
npm.cmd run typecheck
npm.cmd run build
npm.cmd run test
```

Two terminal processes for browser play:

```powershell
# from repo root
uv run --extra gui --extra gpu python -m backend.main
```

```powershell
# from frontend directory
npm.cmd run dev
```

Open `http://127.0.0.1:5173`, switch to 6-Max, and start a **new** tournament; an older session retains its originally configured lineup. The new labels must clearly distinguish the placeholder from the four heuristic bots.

## 5. Validation and limitations

- Phase 1.1 backward-compatibility tests (11) and new Phase 1.3 tests (9) were run together in a **partial local compatibility harness: 20 passed**. The harness mirrors the reference showdown evaluator but stubs some absent modules from the historical patch snapshot. It is not a complete GitHub repository run.
- TypeScript supplemental shim test: `tsc --noEmit -p tsconfig.verify.json` passed; shim files are **not** shipped. The official user's Node/React typecheck, build and Vitest run are required.
- Rule-based bots are not PPO-trained and their empirical win-rate is unverified.
- **Phase 1.4** connects a *compatible trained six-player Tournament Agent* into seat 2, after observation, legal-action and checkpoint compatibility audits.
- Human-vs-Bot full permanent replay export remains a separate milestone (Phase 1.6).
