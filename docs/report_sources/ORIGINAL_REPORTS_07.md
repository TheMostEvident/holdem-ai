# Original documentation volume 7/7

[Final technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md) · Source commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`; exact original Markdown, including code/equations/logs.

## README_TOURNAMENT_ENV_PHASE_1_2_1.md

Original blob `91461348ec47921984049311d9cdd9202f0e9dc8` (1058 bytes).

<a name="md-003"></a>

<!-- BEGIN_MD_SOURCE path="README_TOURNAMENT_ENV_PHASE_1_2_1.md" sha="91461348ec47921984049311d9cdd9202f0e9dc8" -->

````````markdown
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

````````

<!-- END_MD_SOURCE path="README_TOURNAMENT_ENV_PHASE_1_2_1.md" -->

## README_TOURNAMENT_ENV_PHASE_1_3.md

Original blob `da54f4a530afded59cdef46d15fc4ef87646e422` (610 bytes).

<a name="md-004"></a>

<!-- BEGIN_MD_SOURCE path="README_TOURNAMENT_ENV_PHASE_1_3.md" sha="da54f4a530afded59cdef46d15fc4ef87646e422" -->

````````markdown
# Tournament ENV Phase 1.3

This incremental ZIP requires Tournament ENV **1.1 + 1.2 + 1.2.1 card-visibility fix** already installed.

- GUI creates six-seat `improved_4` lineup: Human, Tournament Agent *placeholder* (Calling Station), and 4 different heuristic bots.
- Five-action engine contract and existing HUNL GUI remain intact.
- The legacy `baseline_5` API remains the default unless a caller supplies the new `lineup` value.
- New regressions, runnable smoke command and Obsidian-compatible guide are included.

See `docs/TOURNAMENT_ENV_PHASE_1_3.md` for setup, exact roles, limitations and commands.

````````

<!-- END_MD_SOURCE path="README_TOURNAMENT_ENV_PHASE_1_3.md" -->

## README_TOURNAMENT_ENV_PHASE_1_4.md

Original blob `4ab726862cb8a0591a063d24e92da15db33e9271` (1193 bytes).

<a name="md-005"></a>

<!-- BEGIN_MD_SOURCE path="README_TOURNAMENT_ENV_PHASE_1_4.md" sha="4ab726862cb8a0591a063d24e92da15db33e9271" -->

````````markdown
# Tournament ENV 1.4 — Frozen Multi-seat PPO Adapter

**Patch over Tournament ENV 1.3.1.** Updates the Tournament backend schema, session, router, GUI selector, and adds trained_actor adapter, tests, smoke and Obsidian documentation. Existing HUNL environment and RL training files remain untouched.

Run from repository root:

```powershell
Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-4.zip" -DestinationPath . -Force
$env:HOLDEM_AI_TOURNAMENT_CHECKPOINT = (Resolve-Path '.\runs\expand_phase03_pilot\best.pt').Path
$env:HOLDEM_AI_TOURNAMENT_DEVICE = 'cpu'
$env:HOLDEM_AI_TOURNAMENT_POLICY_MODE = 'sampled'
uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_tournament_env14_trained_actor.py
uv run --extra dev --extra gui --extra gpu python -m scripts.tournament_env14_smoke
```

Open React 6-Max GUI and choose **Expand Phase3 · Trained Multi-seat PPO**. This is an **experimental 6-max cash-to-knockout transfer**: it is a *trained 2–6 player PPO* checkpoint, **not** verified tournament-trained or GTO. A missing/wrong checkpoint fails closed. Never substitute HUNL200 2-seat Q-Boost weights.

See `docs/TOURNAMENT_ENV_PHASE_1_4.md`.

````````

<!-- END_MD_SOURCE path="README_TOURNAMENT_ENV_PHASE_1_4.md" -->

## README_TOURNAMENT_ENV_PHASE_1_5.md

Original blob `6ed216eca53b9fa45c06996ca0b11d3a319be162` (1320 bytes).

<a name="md-006"></a>

<!-- BEGIN_MD_SOURCE path="README_TOURNAMENT_ENV_PHASE_1_5.md" sha="6ed216eca53b9fa45c06996ca0b11d3a319be162" -->

````````markdown
# Tournament ENV Phase 1.5 — UX / Standings / Blind Levels

## 적용 전제

Tournament ENV 1.1 → 1.2(+카드 수정) → 1.3(+1.3.1 aggression 수정) → 1.4 설치 완료.
이번 패치는 해당 버전을 기준으로 **변경 파일만 덮어쓰기** 위한 패치다.

## 설치

```powershell
cd C:\Users\yiydg\Desktop\workspace\holdem-ai-starter\holdem-ai
Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-5.zip" -DestinationPath . -Force
uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_tournament_env15_progress.py
uv run --extra dev --extra gui --extra gpu python -m pytest -q
cd frontend
npm.cmd run typecheck
npm.cmd run build
npm.cmd run test
```

**Backend 재시작 필수**. 브라우저는 새로고침(필요 시 Ctrl+Shift+R). 세션은 메모리 기반이므로 진행 중인 토너먼트를 종료하고 재시작하는 편이 좋다.

## 변경

- 6인 칩 순위(임시 `~N`)와 최종/탈락 순위(`#N`) 분리
- 실제 정산 후 생존 좌석 기반 살아 있는 플레이어 수, 올인 오판 방지
- 다음 Blind Level SB/BB와 상승까지 남은 핸드 수
- 최근 탈락 이벤트 / 토너먼트 우승자 화면
- 게임 규칙, 모델 추론/학습 가중치 불변

자세한 설계와 수식: `docs/TOURNAMENT_ENV_PHASE_1_5.md`

````````

<!-- END_MD_SOURCE path="README_TOURNAMENT_ENV_PHASE_1_5.md" -->

## README_TOURNAMENT_ENV_PHASE_1_6_1.md

Original blob `3c980c3cc62b1b9f1d464d6e6ff3352c8a51ae6a` (1310 bytes).

<a name="md-008"></a>

<!-- BEGIN_MD_SOURCE path="README_TOURNAMENT_ENV_PHASE_1_6_1.md" sha="3c980c3cc62b1b9f1d464d6e6ff3352c8a51ae6a" -->

````````markdown
# Tournament ENV 1.6.1 — Showdown 카드 공개 패치

Phase 1.6이 설치된 프로젝트 루트에 ZIP을 풀어 덮어쓰세요. 그 이전 패치는 다시 설치할 필요 없습니다.

## 실행 전

1. **Backend 종료** (`Ctrl+C`).
2. `Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-6-1-showdown.zip" -DestinationPath . -Force`
3. `uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_tournament_env161_showdown.py`
4. `uv run --extra dev --extra gui --extra gpu python -m pytest -q`
5. `cd frontend`; `npm.cmd run typecheck`; `npm.cmd run build`; `npm.cmd run test`
6. Backend 재시작 후 새 Tournament 핸드를 한 번 끝내세요.

## 확인할 곳

- **MOST RECENT SETTLEMENT**: 직전 핸드 Showdown 카드 앞면 공개.
- **Completed hands**: 과거 핸드 상세에서 같은 카드 확인.
- **Saved sessions & hand replay**: Replay 마지막 정산 프레임에서 카드 확인.

Fold한 플레이어의 카드와 정산 전 상대 카드 정보는 **절대 공개하지 않습니다**. Fold 승리에는 카드 공개가 없습니다. **기존 SQLite 기록의 과거 숨겨진 카드 데이터는 복구할 수 없습니다.**

자세한 수식과 데이터 계약은 `docs/TOURNAMENT_ENV_PHASE_1_6_1_SHOWDOWN.md`를 참고하세요.

````````

<!-- END_MD_SOURCE path="README_TOURNAMENT_ENV_PHASE_1_6_1.md" -->

## README_TOURNAMENT_ENV_PHASE_1_6.md

Original blob `6d51be0a3157f3cdc1cade936009dc453997c005` (1074 bytes).

<a name="md-007"></a>

<!-- BEGIN_MD_SOURCE path="README_TOURNAMENT_ENV_PHASE_1_6.md" sha="6d51be0a3157f3cdc1cade936009dc453997c005" -->

````````markdown
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

````````

<!-- END_MD_SOURCE path="README_TOURNAMENT_ENV_PHASE_1_6.md" -->

## README.md

Original blob `111e91de2c90f6c8e891473ffd3bea1d334c1438` (6673 bytes).

<a name="md-001"></a>

<!-- BEGIN_MD_SOURCE path="README.md" sha="111e91de2c90f6c8e891473ffd3bea1d334c1438" -->

````````markdown
# holdem-ai

**Research software for Heads-Up No-Limit Hold'em (HUNL), dynamic 2–6-player cash poker, and 6-max tournament play.**

This repository separates two responsibilities:

- **Environment:** cards, legal actions, positions, bets, side pots, showdown, chip accounting, tournament progression, and GUI-facing state.
- **Agent:** information-state encoding, masked decisions, on-policy PPO, experimental QBoost, multi-seat PPO, MCCFR/search prototypes, diagnostics, and evaluation.

> **Research status:** Experimental systems and test infrastructure; **not a solved HUNL game, verified GTO agent, replicated Pluribus, or a proven tournament-optimal policy**.

## Read first

**[Final integrated technical report — equations, purpose and technical operation of every Phase](docs/POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md)**

The technical report is the primary overview. Original granular development documents remain in \`docs/\` while their source-level evidence is preserved. A concise synthesis is not permission to delete the original reports before full lossless archiving.

## The mathematical contract

Let the complete hand state be $s_t$, legal action $a_t$, and randomized chance event $\omega_t$:

$$
s_{t+1}=T(s_t,a_t,\omega_t),\qquad a_t\in\mathcal A_{\rm legal}(s_t).
$$

The actor uses its own information state $I_{i,t}$ rather than the omniscient engine state:

$$
\pi_\theta(a\mid I_{i,t})=
\frac{\mathbf1[a\in\mathcal A(I_{i,t})]e^{\ell_\theta(I_{i,t},a)}}
{\sum_{b\in\mathcal A(I_{i,t})}e^{\ell_\theta(I_{i,t},b)}}.
$$

Chip conservation and return, assuming no rake and no mid-hand external funding:

$$
\sum_iS_{i,t}^{behind}+P_t=C_t,\qquad
R_{i,h}^{BB}=\frac{S_{i,h}^{final}-S_{i,h}^{initial}}{BB_h},\qquad
\widehat{bb/100}=\frac{100}{H}\sum_h R_{i,h}^{BB}.
$$

## Project map

| Workstream | Phases | Deliverable |
|---|---|---|
| Base Environment | 0–4 | RLCard baseline/corrections; 2–6-player native engine; tournament game rules |
| HUNL Agent | 5–14 | Seven-action HUNL200 neural Actor, PPO/QBoost, batching and seeded evaluation |
| HUNL research | 15–17 | Critic/collapse diagnostics, stability ablation, continuous-stack evaluation, GPU retraining |
| Expand | E1–E7 | Dynamic cash table, multi-seat encoding/PPO, research MCCFR/search/opponent adaptation |
| HUNL GUI | 1.0–1.6 | Human interaction, FastAPI/React, checkpoint inference, match analytics and stack carry |
| Tournament ENV | 1.1–1.7 | Six-seat UI, improved opponents, E3 PPO inference bridge, SQLite/replay, showdown reveal |

**Important distinctions:** P17 continuous-stack evaluation and P17 five-hour PPO retraining are different experiments. Base P17 HUNL weights and Expand E3 multi-seat weights are incompatible by default. Expand E3 is cash-game trained infrastructure; Tournament-specific PPO is a future training track.

## Install and test

Python 3.11 and [uv](https://docs.astral.sh/uv/) are the reference package workflow.

\`\`\`powershell
git clone https://github.com/TheMostEvident/holdem-ai.git
cd holdem-ai
uv sync --extra dev
uv run --extra dev python -m pytest -q
\`\`\`

For GUI/GPU dependencies on an appropriate CUDA host:

\`\`\`powershell
uv sync --extra dev --extra gui --extra gpu
uv run --extra dev --extra gui --extra gpu python -m pytest -q
cd frontend
npm.cmd ci
npm.cmd test
npm.cmd run build
npm.cmd run dev
\`\`\`

These are execution instructions rather than newly certified passing tests. GPU device and driver compatibility must be verified locally.

Run a corrected HUNL baseline from the repository root:

\`\`\`powershell
uv run python main.py --hands 100 --seed 42
\`\`\`

### Trained GUI checkpoints

HUNL trained PPO, experimental QBoost, and Expand E3 multi-seat PPO are **different model families** with different files and action contracts. A real locally available, trusted and compatible \`best.pt\` is required for trained-policy inference. Files inside \`runs/\` are generally excluded from Git. Use [checkpoint setup](docs/GUI_TRAINED_CHECKPOINT_SETUP.md) for local paths and \`scripts/start_gui_with_models.ps1\`; ensure the backend restarts and creates a fresh session after changing weights.

## Evaluation standards

Use paired seed-blocked seat/button configurations and statistical uncertainty, not a single high score:

$$
X_k=\frac14\sum_{j=1}^{4}R_{k,j}^{BB},\qquad
\widehat J_{bb/100}=\frac{100}{K}\sum_{k=1}^K X_k.
$$

Deck seed groups, not the four games within each group, are independent replicates. Keep validation checkpoint selection separate from a sealed held-out final test.

## Recorded milestone and limitations

The historical Phase17 PPO retrain recorded **5,000 accepted updates and 640,000 training hands**. It stopped at **12,743.274 active seconds**, before its 18,000-second target (\`complete=false\`, \`max_updates_ceiling\`). A repeated-validation best score does **not** prove independent game strength. Phase15/16 preflop and critic calibration issues and Expand native-engine/search acceptance remain research tasks.

## Repository structure

\`\`\`text
src/holdem_ai/
  environment/       # Hand engine, game rules, HUNL200, tournaments
  core/              # InformationState and policy contracts
  encoding/          # Tokenization, masks, tensors
  models/            # Transformer Actor / Q Critic
  training/          # PPO, QBoost, batched self-play
  experiments/       # Reproducible studies and long runs
  evaluation/        # Seed blocked, opponent pool, diagnostics
  expand/            # Dynamic 2-6 player research stack
  game_service/      # HUNL human GUI services
  tournament_service/# Six-seat tournament services
backend/             # FastAPI HTTP contract
frontend/            # React interface
scripts/             # Audits, smoke tests, train/evaluate CLI
tests/               # Regression, privacy and behavior tests
docs/                # Canonical report + original phase reports
\`\`\`

## Engineering priorities

1. Prove poker rule invariants and actor privacy with real-engine end-to-end tests.
2. Investigate preflop policy concentration and critic calibration before interpreting training duration as improvement.
3. Reproduce independent, position-balanced, multi-seed validation and held-out results.
4. Verify Expand E3 checkpoint compatibility and train a **tournament-specific PPO** separately.
5. Preserve original phase reports until an exact-content archived copy has been checked against every original Git blob.

See the [integrated technical report](docs/POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md) for the phase-by-phase mathematical treatment, implementation map, evidence boundaries and original references.

````````

<!-- END_MD_SOURCE path="README.md" -->

## TOURNAMENT_ENV_PHASE_1_1_README.md

Original blob `e8210483c9b90bef9415496b23fcfbc1bbe16a14` (1469 bytes).

<a name="md-009"></a>

<!-- BEGIN_MD_SOURCE path="TOURNAMENT_ENV_PHASE_1_1_README.md" sha="e8210483c9b90bef9415496b23fcfbc1bbe16a14" -->

````````markdown
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

````````

<!-- END_MD_SOURCE path="TOURNAMENT_ENV_PHASE_1_1_README.md" -->

## TOURNAMENT_ENV_PHASE_1_2_README.md

Original blob `a9106f003428131c3dc2bb13021bc9efdd986575` (2642 bytes).

<a name="md-010"></a>

<!-- BEGIN_MD_SOURCE path="TOURNAMENT_ENV_PHASE_1_2_README.md" sha="a9106f003428131c3dc2bb13021bc9efdd986575" -->

````````markdown
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

````````

<!-- END_MD_SOURCE path="TOURNAMENT_ENV_PHASE_1_2_README.md" -->

## TOURNAMENT_ENV_PHASE_1_7_README.md

Original blob `abdef5e2d2de3b3015f23718ac5ecfcb71d7f5ee` (1091 bytes).

<a name="md-011"></a>

<!-- BEGIN_MD_SOURCE path="TOURNAMENT_ENV_PHASE_1_7_README.md" sha="abdef5e2d2de3b3015f23718ac5ecfcb71d7f5ee" -->

````````markdown
# Tournament ENV Phase 1.7 · Release QA Pack

마지막 단계는 기능 변경이 아닌 **안전한 통합 검증 / 기록 보호 / GitHub Push 준비**다.

전제: Tournament ENV Phase 1.1~1.6.1까지 ZIP 패치가 로컬 프로젝트에 이미 적용되어 있어야 한다.

프로젝트 루트에서:

```powershell
Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-7.zip" -DestinationPath . -Force
powershell -ExecutionPolicy Bypass -File .\scripts\tournament_env17_verify.ps1
```

실제 체크포인트 추론을 검증하려면 문서의 환경변수 설정 후 `-WithAgentSmoke` 옵션 추가.

Push 직전:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\tournament_env17_stage.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File .\scripts\tournament_env17_stage.ps1
git diff --cached --name-status
```

Staging 스크립트는 `.gitignore`를 안전하게 덧붙이고, Tournament ENV 관련 변경 경로만 Stage한다. **Commit/Push는 자동으로 하지 않는다.**

전체 기술 문서: `docs/TOURNAMENT_ENV_PHASE_1_7_FINAL.md`

````````

<!-- END_MD_SOURCE path="TOURNAMENT_ENV_PHASE_1_7_README.md" -->
