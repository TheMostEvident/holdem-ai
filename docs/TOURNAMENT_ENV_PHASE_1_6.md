# Tournament ENV Phase 1.6 — Persistent Match History / Recorded Replay / Actor Export

> **범위: Environment.** 기존 HUNL, Tournament Engine, PPO/Q-Boost 모델 가중치, 강화학습 알고리즘은 수정하지 않는다.

## 1. 목적

기존 `/api/v1/tournament/sessions/{id}/history`는 **프로세스 메모리**에만 기록을 보관한다. 이 단계는 로컬 SQLite 저장소를 추가해서 게임 중 서버가 종료되거나 사용자가 `End session`을 눌러도 **기존에 커밋된 대국 행동/정산 기록이 남도록** 한다.

- **재현(Recorded Replay)**: 플레이 도중 저장된 프레임을 시간순으로 조회. 새로운 샘플링, 새로운 Bot 정책 실행 또는 덱 재생성이 아니다.
- **계속 플레이(Resume)**: **이번 단계에는 없음.** 서버 재시작 후 과거 토너먼트를 이어 하는 기능은 제공하지 않는다.
- **데이터 내보내기**: 각 행동의 *acting player* 기준 관측과 실제 행동을 NDJSON(JSONL)로 제공한다. 알고리즘 업데이트는 수행하지 않는다.

## 2. 저장 위치와 데이터 모델

기본값은 프로젝트 루트:

```text
.local/tournament_history.sqlite3
.local/tournament_history.sqlite3-wal
.local/tournament_history.sqlite3-shm
```

설정 변경:

```powershell
$env:HOLDEM_AI_TOURNAMENT_JOURNAL = "E:\\PrivateHoldemData\\tournament_history.sqlite3"
```

`.local/.gitignore`로 기본 기록 파일과 WAL/SHM 파일은 Git 추적 대상에서 제외한다. 사용자 지정 경로도 `git add -A` 전에 별도로 확인한다.

저장 스키마:

| 테이블 | 내용 | 기록 시점 |
| --- | --- | --- |
| `tournament_sessions` | 세션 ID, 생성 시각, 설정, 상태, 최종 우승자 | 세션 생성/행동/종료 |
| `tournament_actions` | 핸드 번호, 행동 순서, 액터의 결정 직전 정보, 행동 | **모든 합법 행동 직후** |
| `tournament_hands` | 공개 보드, 행동 목록, 배당, 최종 스택, 탈락자 | **핸드 정산 후** |

저장은 트랜잭션 단위로 커밋된다. `DELETE /sessions/{id}`는 라이브 세션만 종료하고 저장된 기록은 삭제하지 않는다.

**주의:** 시스템 강제 종료 직전에 DB로 커밋되지 못한 행동은 기록되지 않을 수 있다. WAL과 FULL 동기화를 사용하지만 디스크·하드웨어 장애에 대한 무손실 보장을 뜻하지 않는다.

## 3. 칩 보존 및 Replay 검사

실시간 시점:

$$
\boxed{\sum_{i=0}^{5} S_{i,t}^{\text{behind}} + P_t = C_{\text{total}}}
$$

핸드 정산:

$$
\boxed{\sum_{i=0}^{5} S_{i,h}^{\text{final}} = C_{\text{total}}}
$$

각 Replay 프레임은 **행동 직전** 시점의 `board`, `stacks`, `street_bets`, `contributions`, `legal_actions`, `pot`, `to_call`을 갖는다. 최종 프레임은 엔진의 `TournamentHandRecord`를 사용하므로 실제 `awards`, `payoffs_chips`, `eliminated` 기록과 일치한다.

## 4. API

기존 `/api/v1/tournament/sessions` 및 HUNL API는 그대로 유지한다.

| HTTP | Endpoint | 기능 |
| --- | --- | --- |
| GET | `/api/v1/tournament/archive` | 최근 저장된 세션 50개(페이지네이션 지원) |
| GET | `/api/v1/tournament/archive/{session_id}/hands` | 완료 핸드 최근순 조회 |
| GET | `/api/v1/tournament/archive/{session_id}/hands/{hand_number}/replay` | 시점별 행동 + 완료 시 정산 |
| GET | `/api/v1/tournament/archive/{session_id}/export` | `application/x-ndjson` 파일 다운로드 |

Swagger: `http://127.0.0.1:8000/docs`. 프론트엔드는 6-Max 탭의 **Saved sessions & hand replay** 패널에서 위 API를 사용한다. 1.5의 Live Tournament UI는 유지한다.

## 5. Actor-visible Export Contract

각 행동 $t$의 데이터는 다음과 같다.

$$
D_t=(I_t,a_t,R_h,\text{session\_id},\text{hand\_number})
$$

- $I_t$: 행동 플레이어의 **자기 홀 카드 2장**, 공개 보드, 공개 스택, 팟, 행동 가능 목록
- $a_t$: 실제 선택 행동과 정확한 `raise_to` 목표 금액
- $R_h$: 완료된 핸드의 **그 플레이어 칩 수익(라벨)**, 미정산이면 `null`

```json
{
  "schema": "tournament.actor_transition.v1",
  "session_id": "...",
  "hand_number": 1,
  "action_no": 1,
  "acting_seat": 3,
  "observation": {
    "street": "preflop",
    "board": [],
    "button": 0,
    "stacks": [1000, 995, 990, 1000, 1000, 1000],
    "pot": 15,
    "to_call": 10,
    "legal_actions": [0, 1, 2, 3, 4],
    "own_hole_cards": ["d4", "c5"]
  },
  "action": {"index": 1, "kind": "check_call", "raise_to": null},
  "hand_settled": false,
  "payoff_chips": null
}
```

이것은 필드를 간략화한 예시다. **타인 홀 카드, 전체 덱, 학습용 Critic/Optimizer 상태는 포함하지 않는다.** Replay 화면에는 Human 본인 턴에서만 Human 홀 카드가 표시되고, 다른 좌석의 홀 카드는 숨긴다. **반면 Export 파일은 각 행동자의 본인 홀 카드를 포함하므로 공개 저장소에 업로드해서는 안 된다.**

다시 강조하면 이 JSONL은 **행동 데이터 계약**이지, PPO 학습기에 곧바로 투입할 수 있는 GAE/return 확정 형식은 아니다. 신경망 구조나 학습 루프는 여기서 수정하지 않는다.

## 6. Windows 설치 및 테스트

Phase 1.1~1.5 패치가 설치된 프로젝트 루트에서:

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

Backend 재시작 전 기존 토너먼트를 마무리하는 것이 좋다. **1.6이 설치되기 전에 플레이한 과거 세션은 자동 복구되지 않는다.** 새 패치 적용 후 시작한 세션부터 DB에 저장된다.

Backend를 **체크포인트 환경변수를 설정한 동일한 PowerShell에서** 실행한다. Frontend는 별도 PowerShell에서 `npm.cmd run dev`.

## 7. 검증 범위와 제한 사항

- Python Phase 1.6 테스트: SQLite 재오픈, 완료 핸드 순서, Hero/Bot Hole 분리, 무효 행동 원자성, JSONL/HTTP, Live 세션 삭제 후 Archive 유지.
- 기존 Phase 1.1~1.5 및 프론트엔드 전체 회귀 검사는 사용자 PC에서 수행해야 한다.
- 기록 DB에는 액터 자신의 비공개 카드가 있으므로 **로컬 전용 서버를 외부 네트워크에 노출하지 않는다.**
- 파일 시스템에 같은 SQLite 경로를 쓰는 여러 서버 인스턴스를 띄우는 것은 피한다.
- 과거 Session ID로 게임을 재개하는 기능은 지원하지 않는다.
