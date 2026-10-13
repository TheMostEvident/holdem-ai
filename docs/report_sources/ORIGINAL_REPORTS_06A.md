# Original documentation volume 6A/7

[Final technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md) · exact source commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`.

## docs/TOURNAMENT_ENV_PHASE_1_1.md

Original blob `71bd90a540b0fb51334f748f55487b44b8feeacf` (6122 bytes).

<a name="md-060"></a>

<!-- BEGIN_MD_SOURCE path="docs/TOURNAMENT_ENV_PHASE_1_1.md" sha="71bd90a540b0fb51334f748f55487b44b8feeacf" -->

````````markdown
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

````````

<!-- END_MD_SOURCE path="docs/TOURNAMENT_ENV_PHASE_1_1.md" -->

## docs/TOURNAMENT_ENV_PHASE_1_2.md

Original blob `2fa9b318a01bb524ff488b9754c9a18cdcdc589c` (4703 bytes).

<a name="md-062"></a>

<!-- BEGIN_MD_SOURCE path="docs/TOURNAMENT_ENV_PHASE_1_2.md" sha="2fa9b318a01bb524ff488b9754c9a18cdcdc589c" -->

````````markdown
# Tournament ENV Phase 1.2 — 6-Max React GUI

> **Environment Only** · 2026-10-13 · Phase 1.1 API required · Agent training unchanged

## 1. 목표

Phase 4의 6인 토너먼트 엔진과 Phase 1.1 FastAPI API 위에 **Human 1명 + 임시 Calling Station Bot 5명**으로 이루어진 실제 플레이 가능 React 6인 테이블을 구축한다.

- 기존 HUNL Heads-Up GUI 및 학습 코드 변경 없음.
- 상단 `Heads-Up · HUNL` / `6-Max · Tournament` 탭으로 게임 환경 전환.
- 모드 전환 시 두 React 트리를 마운트된 상태로 유지하므로 현재 브라우저의 세션 UI 상태를 보존. 새로고침/서버 종료 후 세션 복원은 **미구현**.
- 각 좌석의 Chip Stack, 역할, BTN/SB/BB, 현재 턴, 거리/탈락/순위, 공용 카드 5장, 팟, 베팅 조작을 실시간 API 응답으로 표시.
- 마지막 핸드 정산 정보와 최근 완료 25핸드 기록을 조회. 이력은 현재 서버 **메모리 보관**이며, 영구 저장은 Phase 1.6.
- 모바일/태블릿 대응 좌석 배치 및 Responsive 패널.

## 2. Action Interface

두 환경은 추상 행동 인덱스가 **다르다**.

| Tournament Engine | ID | React UI |
|---|---:|---|
| Fold | 0 | Fold |
| Check/Call | 1 | Check / Call |
| Raise Half Pot | 2 | Raise ½ Pot |
| Raise Pot | 3 | Raise Pot |
| All-In | 4 | All-In |
| Arbitrary Raise-to | 별도 엔진 실행 | `POST /api/v1/tournament/sessions/{id}/raise-to` |

현재 Tournament의 5개 추상 행동 인덱스를 **HUNL200 모델의 7개 인덱스와 혼용하면 안 된다.** 버튼은 `TournamentView.actions`에 해당 인덱스가 있는 경우에만 활성화한다. 자유 Raise-to는 서버가 제공한 `min_raise_to`, `max_raise_to`, `can_raise_to`를 검증한 뒤 별도 API로 전달한다.

```math
A_{\text{legal}}(I_t) = \{a \mid a\in\text{TournamentView.actions}\}
```

## 3. Chip Conservation

진행 중인 핸드에서

```math
\sum_{i=0}^{5}S_{i,t}^{\text{behind}}+P_t=C_{\text{total}}.
```

현재 프론트엔드는 API의 `seats[].stack`, `pot`, `total_chips`로 위 불변조건을 재검사하여 불일치 시 경고를 표시한다. 보존 규칙의 판정·팟 분배·탈락 처리의 **최종 책임은 서버 엔진**에 있다.

## 4. HTTP Contract

Backend는 Phase 1.1에서 제공된 다음 경로를 그대로 사용한다.

```text
GET    /api/v1/tournament/health
POST   /api/v1/tournament/sessions
GET    /api/v1/tournament/sessions/{session_id}
POST   /api/v1/tournament/sessions/{session_id}/actions
POST   /api/v1/tournament/sessions/{session_id}/raise-to
GET    /api/v1/tournament/sessions/{session_id}/history?offset=0&limit=25
DELETE /api/v1/tournament/sessions/{session_id}
```

API는 `expected_revision`으로 중복/지연 액션을 거부한다 (`409` 시 다시 조회). 매번 유저 행동에 따라 서버가 Bot의 나머지 행동을 자동 처리하며 다음 핸드까지 진행한다. Tournament에는 **Next Hand 별도 버튼이 없다.**

## 5. Privacy

TournamentView가 공개한 Human `hero_cards`만 앞면을 표시한다. Bot Hole Cards는 항상 뒷면으로 표시되며, 프론트엔드는 `TournamentHandView`의 결과만 표시한다. Phase 1.1은 상대 Hole Cards를 승부 종료 시에도 사용자에게 보내지 않으므로, GUI는 상대 패 공개를 **추측하거나 생성하지 않는다**.

## 6. Installation (Windows PowerShell)

현재 Repository Root에서:

```powershell
Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-2.zip" -DestinationPath . -Force
uv run --extra dev --extra gui --extra gpu python -m pytest -q
uv run --extra gui --extra gpu python -m backend.main
```

별도 PowerShell에서:

```powershell
cd frontend
npm.cmd run typecheck
npm.cmd run build
npm.cmd run test
npm.cmd run dev
```

이후 `http://127.0.0.1:5173` 접속 → 상단 **6-Max · Tournament** → **Start tournament**.

**ZIP은 프론트엔드 새/수정 파일과 테스트·문서만 제공한다.** Phase 1.1 Backend ZIP이 먼저 설치되어 있어야 한다. `package.json`, `package-lock.json`, npm 의존성, Python 게임 엔진, 기존 학습 체크포인트는 건드리지 않는다.

## 7. 향후 Phase

- **1.3:** 기존 Improved Bot 4명 정책을 확인하고 자동 행동 Seat Adapter 연결.
- **1.4:** 6인 Tournament Policy Contract를 충족하는 Trained Tournament Agent 체크포인트 연결. *기존 Heads-Up PPO 모델을 6인 정책으로 가장하지 않음.*
- **1.5:** Blind Level UX, 탈락/순위/우승 경험 개선.
- **1.6:** Durable Tournament History, Replay 및 훈련용 Event/Observation Export.
- **1.7:** 전체 Windows/브라우저 회귀 시험 및 GitHub Push.

````````

<!-- END_MD_SOURCE path="docs/TOURNAMENT_ENV_PHASE_1_2.md" -->

## docs/TOURNAMENT_ENV_PHASE_1_3_1_AGGRESSION_FIX.md

Original blob `09efde46ffac27d0a5f39356d20c26f5cda950a5` (3715 bytes).

<a name="md-064"></a>

<!-- BEGIN_MD_SOURCE path="docs/TOURNAMENT_ENV_PHASE_1_3_1_AGGRESSION_FIX.md" sha="09efde46ffac27d0a5f39356d20c26f5cda950a5" -->

````````markdown
# Tournament ENV 1.3.1 — Bot Aggression Hotfix

## Why bots almost always called or folded

1. **Seat 2 (Agent placeholder):** Phase 1.3 explicitly used a `Calling Station`, which always preferred `check/call` when legal. It was **not** a trained Tournament Agent.
2. **Seats 3–6 (heuristics):** the preflop raising thresholds were too conservative after subtracting the penalty for five opponents. Several profiles rarely opened a raise.
3. **Real decision bug:** the previous expression `if value >= raise_threshold and 2 in legal` prevented a `Pot Raise` even when action `3` was legal and action `2` (`Half Pot`) was illegal. The correct test is `(2 in legal or 3 in legal)`.

## Changes

- Permit legal pot-size raise even if half-pot raise is not offered by the engine.
- Tune heuristic raise thresholds for visible, meaningfully different value-raising styles. These are **not** empirical equity estimates and no winning-performance claim is made.
- In the explicit `improved_4` lineup only, Seat 2 is still labeled `Tournament Agent placeholder`, but its **temporary untrained policy** now uses `controlled_pressure` heuristic rather than `Calling Station`.
- The original `baseline_5` lineup and its five Calling Stations remain untouched for backward compatibility.
- Updated frontend placeholder description and included synthetic raise-mix regression tests.
- No RL algorithm / trained Agent / checkpoint changes.

## Action map

| Native action | Meaning |
|---:|---|
| 0 | Fold |
| 1 | Check / Call |
| 2 | Raise half pot |
| 3 | Raise pot |
| 4 | All-in |

For a live acting Bot, the legal mask from `TournamentEnv.observe(actor)` is always authoritative:

$$a_t \in \mathcal A_t^{legal}.$$

### Diagnostic frequencies

Use:

```powershell
uv run --extra gui python -m scripts.tournament_env131_diagnose
```

The CLI enumerates 1,326 possible unordered hole-card combinations under an illustrative fixed pot, stack, and legal-action mask. The reported raise percentages are **conditional bot decisions** — they do not predict real-game raise frequency, equity, or bot strength. For a real-world benchmark, log full tournaments, positions, legal action sets, SPR, and actual actions, and compare to frozen baseline policies on matched seeds.

## Install

Close FastAPI (`Ctrl+C`), then from the project root:

```powershell
Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-3-1-aggression-fix.zip" -DestinationPath . -Force
uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_tournament_env13_improved_bots.py tests/test_tournament_env131_aggression.py
uv run --extra dev --extra gui --extra gpu python -m pytest -q
cd frontend
npm.cmd run typecheck
npm.cmd run build
npm.cmd run test
```

Start new backend and a **new tournament**; existing running sessions preserve their previously created Bot policy objects. `npm run dev` can hot reload the frontend, but the Backend must restart to load changed Python modules.

### Verification from a partial compatibility harness

- API/heuristic tests: **30 passed** (11 existing 1.1, 9 existing 1.3, 10 new 1.3.1).
- In the partial harness, 30 seeded `improved_4` tournaments (388 hands overall) all ended and preserved chips. Observed bot raise (half-pot/pot, excluding all-in) percentages by engine seat **1–5**: **29.1%, 10.4%, 19.6%, 11.2%, 28.5%**. These figures are demonstration diagnostics, not win rates, not a controlled baseline comparison, and **not** validation against the complete Windows repository.
- Supplemental frontend `tsc -p tsconfig.verify.json --noEmit` completed. The official React build/Vitest and the full Python regression suite remain to be run on the user Windows machine.

````````

<!-- END_MD_SOURCE path="docs/TOURNAMENT_ENV_PHASE_1_3_1_AGGRESSION_FIX.md" -->

## docs/TOURNAMENT_ENV_PHASE_1_3.md

Original blob `049e22b4cbace791b251f0cd278672cfc36ce0cc` (5192 bytes).

<a name="md-063"></a>

<!-- BEGIN_MD_SOURCE path="docs/TOURNAMENT_ENV_PHASE_1_3.md" sha="049e22b4cbace791b251f0cd278672cfc36ce0cc" -->

````````markdown
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

````````

<!-- END_MD_SOURCE path="docs/TOURNAMENT_ENV_PHASE_1_3.md" -->

## docs/TOURNAMENT_ENV_PHASE_1_4.md

Original blob `6f4f4df8fe7bbbab4df3a85ef073a0c54350c483` (6534 bytes).

<a name="md-065"></a>

<!-- BEGIN_MD_SOURCE path="docs/TOURNAMENT_ENV_PHASE_1_4.md" sha="6f4f4df8fe7bbbab4df3a85ef073a0c54350c483" -->

````````markdown
# Tournament ENV Phase 1.4 — Trained Multi-seat Actor Connection

> 환경(Environment) 전용 개발. Agent의 PPO 학습, 가중치 갱신 및 강화학습 최적화 코드는 수정하지 않는다.

## 1. 개발 목표

Phase 1.3의 6인 테이블 구성에서 **Seat 2(physical seat index 1)**의 임시 Heuristic Policy를 선택적으로 실제 학습된 2~6인 PPO 정책에 연결한다.

| 물리적 Seat | 역할 | 정책 |
| --- | --- | --- |
| 0 | Human | GUI 직접 베팅 |
| 1 | Tournament Agent | `Expand Phase3 PPO` 체크포인트 추론, 또는 명시적 미학습 Heuristic Placeholder |
| 2 | Improved Bot 1 | Tight Aggressive Heuristic |
| 3 | Improved Bot 2 | Balanced Value Heuristic |
| 4 | Improved Bot 3 | Pot Odds Heuristic |
| 5 | Improved Bot 4 | Controlled Pressure Heuristic |

**주의:** 현재 사용 가능한 `runs/expand_phase03_pilot/best.pt`는 **2~6인 캐시게임** 학습 실험의 파일이다. 토너먼트용 Actor를 학습해둔 파일이라고 주장할 수 없다. 이번 연결은 **Cash-Game → Tournament Transfer Inference**에 해당하며, 파산·블라인드 상승·잔여 인원 변화 등 학습 분포 밖 상태에서의 성능은 검증되지 않았다. `runs/phase14_pilot/.../qboost/best.pt` 같은 HUNL200 전용 2인 체크포인트는 형식 불일치로 **명시적으로 거부**한다.

## 2. 실행 방법 (Windows PowerShell)

프로젝트 루트:

```powershell
cd C:\Users\yiydg\Desktop\workspace\holdem-ai-starter\holdem-ai
Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-4.zip" -DestinationPath . -Force

# 모델 파일이 존재하는지 확인
Test-Path .\runs\expand_phase03_pilot\best.pt

# Backend 실행하는 PowerShell에서 환경변수 등록 (세션 범위)
$env:HOLDEM_AI_TOURNAMENT_CHECKPOINT = (Resolve-Path ".\runs\expand_phase03_pilot\best.pt").Path
$env:HOLDEM_AI_TOURNAMENT_DEVICE = "cpu"
$env:HOLDEM_AI_TOURNAMENT_POLICY_MODE = "sampled"  # 또는 greedy

uv run --extra gui --extra gpu python -m backend.main
```

기존 React 개발 서버가 살아 있다면 화면을 새로 고침한다. GUI의 6-Max 설정에서 **Seat 2 / Tournament Agent = Expand Phase3 · Trained Multi-seat PPO** 선택 후 새 Tournament를 생성한다. 새로 생성한 세션부터 해당 정책이 적용된다. `POST /api/v1/tournament/sessions`에서 `lineup="trained_1_improved_4"`를 지정할 때만 실제 학습된 정책을 사용한다. 기존 `lineup="improved_4"`와 `lineup="baseline_5"`는 변경되지 않는다.

**체크포인트가 없을 때:** trained 선택을 비활성화하고 이유를 표시한다. API에서 trained 라인업을 강제 요청하면 422 오류가 발생하며 **Random Bot으로 조용히 대체하지 않는다.** 형식이 맞지 않는 체크포인트도 거부한다.

## 3. 관측 및 행동 계약

Tournament Phase 4 엔진의 합법 행동:

$$
\mathcal A_T = \{0:\mathrm{Fold}, 1:\mathrm{CheckCall}, 2:\mathrm{RaiseHalfPot}, 3:\mathrm{RaisePot}, 4:\mathrm{AllIn}\}.
$$

Expand Phase3 정책 출력:

$$
\mathcal A_E = \{0:\mathrm{Fold}, 1:\mathrm{Check},2:\mathrm{Call},3:\mathrm{RaiseHalfPot},4:\mathrm{RaisePot},5:\mathrm{RaiseDoublePot},6:\mathrm{AllIn}\}.
$$

\(\mathrm{RaiseDoublePot}\)은 Tournament Native Action에 존재하지 않으므로 항상 Mask한다.

$$
\mathrm{map}(a_E) = \begin{cases}
0 & a_E=0\\
1 & a_E\in\{1,2\}\\
2 & a_E=3\\
3 & a_E=4\\
4 & a_E=6
\end{cases}
$$

Check/Call을 구분할 때는 시점의 `to_call`을 사용한다. 실제 합법성의 **유일한 권위는 Poker Engine의 `legal_actions`**이며, 모든 신경망 확률 출력에 엔진 파생 마스크를 적용한다.

$$
\pi_{\theta}^{\text{legal}}(a\mid I)
=\frac{\pi_\theta(a\mid I)\mathbf1[a\in\mathcal A^{\text{legal}}(I)]}
{\sum_b\pi_\theta(b\mid I)\mathbf1[b\in\mathcal A^{\text{legal}}(I)]}.
$$

결정 시 Actor에 전달되는 정보:

- **Seat 2 본인의** 두 Hole Cards
- 공용 카드, BTN, 스택, 공개 베팅, 팟, BB 및 합법 행동 마스크
- 해당 핸드에서 지금까지 공개된 액션 히스토리(행동 직전 Pot/Call 포함)
- **전달하지 않는 정보**: 다른 플레이어 Hole Cards, 미래 덱, 전지적 게임 상태, Critic/학습 전용 상태

명확하게 구분하기 위해 새로운 정책 인터페이스는 `TrainedTournamentActor.select_action(observation) -> int`다. 모델은 읽기 전용으로 로드하며 가중치를 업데이트하지 않는다.

## 4. 사용자 UI / Server 계약

신규 GET API:

```text
GET /api/v1/tournament/agent-status
```

`configured`, `file_found`, `checkpoint_name`, `format`, `model_family`, `device`, `mode`, `validated`를 반환한다. `validated=false`는 상태 조회가 모델 아키텍처 전체를 검증하는 과정이 아니기 때문이다. **세션 생성 시** `load_trained_actor`가 형식과 가중치 호환성을 검사한다. 설정 경로의 전체 절대경로는 HTTP 응답에 반환하지 않는다.

신규 생성 라인업:

```json
{"lineup":"trained_1_improved_4","seed":42,"starting_stack":1000,"button":0,"hands_per_level":10,"max_hands":10000}
```

Seat 2 응답 예:

```json
{"seat":1,"role":"tournament_agent","policy_name":"Expand Phase3 PPO [sha256_12...]"}
```

나머지 좌석과 기존 HUNL GUI 응답 스키마는 유지한다.

## 5. 테스트

Backend를 중지한 상태에서:

```powershell
uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_tournament_env1_api.py tests/test_tournament_env13_improved_bots.py tests/test_tournament_env131_aggression.py tests/test_tournament_env14_trained_actor.py
uv run --extra dev --extra gui --extra gpu python -m scripts.tournament_env14_smoke
uv run --extra dev --extra gui --extra gpu python -m pytest -q

cd frontend
npm.cmd run typecheck
npm.cmd run build
npm.cmd run test
```

- `tournament_env14_smoke`에는 환경변수 `HOLDEM_AI_TOURNAMENT_CHECKPOINT`가 설정되어 있어야 하며 실제 체크포인트로 행동 추론을 실행한다.
- 테스트 시 실제 프로젝트의 전체 Expand Phase2/3 모듈을 사용하는지 확인한다.
- 기존 토너먼트의 `revision`, Side Pot, 칩 보존, 파산 종료 테스트를 유지한다.

## 6. 범위 밖

- 6-Max 또는 Tournament PPO의 **새로운 훈련 및 모델 개선**은 Agent 담당.
- 지속 저장·Hand Replay는 계획대로 Tournament ENV 1.6.
- UX(순위, Blind Timer 등) 확장은 Phase 1.5.
- 학습 PPO 체크포인트를 **검증된 토너먼트 강자**라고 주장하지 않는다.

````````

<!-- END_MD_SOURCE path="docs/TOURNAMENT_ENV_PHASE_1_4.md" -->
