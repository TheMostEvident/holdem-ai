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
