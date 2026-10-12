---
title: "holdem-ai GUI-1.3 — Interactive Betting"
tags:
  - holdem-ai
  - hunl
  - gui
  - fastapi
  - react
  - phase-1-3
status: "implemented; local Windows acceptance pending"
---

# GUI-1.3 — Interactive Betting

## Objective

2인 HUNL200 게임에서 Human(P0)이 실제 React 테이블에서 행동을 결정한다. `HUNLGameSession`은 사람의 베팅을 서버에서 검증하고, 상대 Bot(P1)의 자동 행동을 처리한 후 다음 공개 상태를 반환한다.

$$
\boxed{\text{Human GUI}\xrightarrow{a_t}\text{FastAPI}\xrightarrow{}\text{HUNL200Adapter}\xrightarrow{}\text{MultiPlayerHandEngine}}
$$

$$
\text{GameView}_{t+1}=\operatorname{Project}_{\mathrm{hero}}\left(T(s_t,a_t)\right)
$$

Agent는 계속해서 기존 7개 추상 행동을 사용한다. Human은 **7-Action 버튼**뿐 아니라 **임의 정수 칩 단위의 합법적 `raise_to`**도 전송할 수 있다.

## Scope

| 기능 | 방식 |
| --- | --- |
| Fold, Check, Call | 서버가 제공하는 `ActionOffer.index`를 그대로 전송 |
| Half-Pot/Pot/2x-Pot | 프런트에서 `ActionOffer.raise_to`를 프리셋 금액으로 선택 |
| All-In | 서버의 합법 `all_in` ActionOffer 전송; 콜 올인은 `Call` 처리 |
| Custom Raise-to | 새 `/raise-to` 엔드포인트에 총 베팅액 전송 |
| Betting slider & number input | `min_raise_to`, `max_raise_to`, `can_raise_to`가 있는 경우에만 활성화 |
| Short All-In | 정상 최소 레이즈 미달이어도 스택 전액에 해당하면 엔진이 허용 가능 |
| 중복 클릭·오래된 상태 | 서버 `expected_revision` 검사, HTTP 409 |
| 비공개 정보 | API whitelist projection; Bot 홀카드는 진짜 Showdown 이후에만 표시 |
| Next Hand | Hand 완결 후만 가능; 버튼 교대; 시작 스택 매번 200BB로 초기화 |

### 정확한 Raise-to 정의

상대 베팅에 추가로 필요한 Call 칩을 `c`, 현재 Street에 이미 투입한 금액을 `b_i`, 남은 스택을 `s_i`, 현재 Street의 최고 베팅액을 `b_{\max}`, 마지막 Full Raise의 증분을 `\Delta`라 하자.

$$
\text{ToCall}_i=\max(0,b_{\max}-b_i)
$$

$$
\text{MinFullRaiseTo}=b_{\max}+\Delta
$$

$$
\text{MaxRaiseTo}_i=b_i+s_i
$$

$$
\boxed{\texttt{raise\_to}=\text{이 Street의 최종 총 베팅액}} 
$$

일반적인 Full Raise는 `min_raise_to <= raise_to <= max_raise_to`이지만, Short All-In은 정확히 `raise_to == max_raise_to`인 경우 엔진이 허용할 수 있다. 그 외의 예외(재오픈, 스택 부족 등)는 **프런트가 아닌 Phase-3 엔진**이 최종 결정한다.

## Existing code reuse

- `src/holdem_ai/environment/multi_player_engine.py`: `step_raise_to(amount)` 및 합법성 검사.
- `src/holdem_ai/environment/hunl200_adapter.py`: **변경하지 않음**. 7-action Neural Agent 입출력 계약 유지.
- `src/holdem_ai/game_service/interactive_adapter.py`: 새 서브클래스. 사람의 임의 `raise_to` 실행 및 공개 Action History 기록.
- `src/holdem_ai/game_service/session_manager.py`: `play_human_raise_to()` 추가, revision과 총손익 기록 통합.
- `src/holdem_ai/game_service/state_projection.py`: 서버가 판정한 `can_raise_to` 필드만 추가.
- `backend/game_router.py`: `POST /sessions/{id}/raise-to` 추가.
- `frontend/src/components/BettingPanel.tsx`: GUI 베팅 버튼, 프리셋, 슬라이더, 정확한 금액 입력.
- `frontend/src/lib/betting.ts`: 프런트 입력 검증용 휴리스틱. 최종 판정 권한 없음.

## New HTTP endpoint

`POST /api/v1/hunl/sessions/{id}/raise-to`

```json
{
  "raise_to": 350,
  "expected_revision": 17
}
```

응답: 기존 `GameResponse`와 동일. 본인이 행동한 후 Bot은 가능한 한 자동으로 연속 행동을 실행하므로 최종 응답에는 여러 Action History 항목과 최신 revision이 포함될 수 있다.

- `200`: 성공, 최신 공개 상태 반환
- `404`: 세션 없음
- `409`: stale revision, 잘못된 시점 또는 이미 끝난 핸드
- `422`: 숫자/스키마가 잘못됐거나 엔진 규칙 위반

기존 `/actions` 엔드포인트, 카드 인코딩, 7-Action Neural Policy는 변경하지 않았다.

> [!IMPORTANT]
> GUI-1.3은 연구용 로컬 실행으로 설계했으며 외부 사이트 연동·실시간 경쟁 테이블 사용을 목표로 하지 않는다. FastAPI는 127.0.0.1에 바인딩되고 인증 및 원격 배포를 지원하지 않는다.

## Acceptance criteria

1. 로컬 세션 생성 시 Human 홀 카드 2장만 표시한다.
2. `Fold/Check/Call/All-In` 버튼은 실제 서버 행동을 발생시킨다.
3. `Min/½ Pot/Pot/2× Pot/Max` 프리셋 및 숫자 입력은 `raise_to`를 설정하고 버튼으로 제출한다.
4. 누락·분수·음수·최소 미달·최대 초과 금액을 거부하고 상태와 revision을 변경하지 않는다.
5. Bot은 기존 정보 상태와 7 Action Mask만 입력받고 자동으로 대응한다.
6. 피드에 액션 이력과 공통 카드가 갱신된다. 핸드가 끝나면 Next Hand가 활성화된다.
7. 200BB 초기화 및 BTN 교대를 보존하고, 진짜 쇼다운에서만 Bot 카드를 표시한다.
8. 전체 기존 pytest와 프런트 Vitest/빌드가 완료된다.

## Validation notes

개발 환경에서 GUI-1.0~1.3과 Phase 1~4 코드 기반 **130 passed / 3 skipped**. 신규 API 액션 및 custom raise 엔드포인트 테스트 5개 포함. Node TypeScript transpilation 문법 검사 14개 파일 오류 없음.

**제한**: npm 저장소 연결 및 cached dependency 부재로 React `npm install`, `npm test`, `npm run build`, 실제 브라우저 렌더링은 아직 실행하지 못함. 전체 Phase 0~16 GitHub 코드에 대한 일괄 회귀 테스트 역시 사용자 로컬에서 필요.

## Next

**GUI-1.4 — Trained Agent Integration**: 로컬 신뢰 가능한 Phase 15 PPO / Q-Boost PPO 체크포인트를 로드하고, HUNL200 7-Action Model로 `Human vs Trained Agent` 실행. GUI-1.3은 우선 Baseline Bots(Random/CallingStation/Tight/PotPressure)만 선택할 수 있다.
