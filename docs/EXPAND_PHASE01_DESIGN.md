---
title: "Expand Phase 1 — Dynamic 6-Max Table Environment"
project: "HUNL to 6-Max NLHE Expansion"
phase: "Expand Phase 1"
version: "1.0.0-patch"
status: "local unit tests passed; Windows real-engine acceptance pending"
date: 2026-10-12
tags:
  - holdem
  - 6max
  - cash-game
  - event-sourcing
  - expand-phase01
---

# Expand Phase 1 — Dynamic 6-Max Table Environment

> [!important] 학습과 구현의 경계
> 이 Phase는 **Cash Game의 동적 테이블/정산 코어**만 만든다. 기존 Base Phase 1~16 소스는 변경하지 않는다. PPO, MCCFR, Card Abstraction, Neural 6-Max Encoder, Pluribus Search는 이번 패치의 범위가 아니다. Tournament는 **RuleSet Stub만** 추가했고 **실제 Tournament 실행을 지원하지 않는다**.

원본 프로젝트의 `EXPAND_MASTER_DESIGN_6MAX_v1_0.md`의 E1.1~E1.5를 구현 기준으로 사용한다.

## 1. 게임 정책 및 명세

| 항목 | E1 구현값 | 비고 |
|---|---|---|
| 모드 | `cash` | Tournament는 E7 예정 |
| 좌석 수 | 물리 좌석 `0..5`, 항상 6개 | 실제 인원은 2~6 |
| 블라인드 | SB=50, BB=100 | 정수 Chip |
| 레이크 | 0 | 다른 값은 ValueError |
| 최초 Buy-In | 20~300BB | 기본 예제 100BB |
| Stack | 핸드 사이 이월 | Bust는 **쇼다운 후** 판단 |
| Button | 기존 Phase 3 Moving Button | Dead Button 미구현 |
| 참가 | `wait_for_big_blind` | 2인 테이블 성립 시 bootstrap 예외 |
| 이벤트 처리 | 핸드 경계 전용 | 핸드 중에는 예약만 가능 |
| 이벤트 로그 | append-only JSONL + deterministic replay | 로그에 RNG Seed가 있어 비공개 자료로 보관 |

### 1.1 게임 불변식

핸드 중 플레이어별 잔여 Stack $S_{i,t}$, 베팅 누적 기여금 $c_{i,t}$:

$$
\boxed{\sum_{i=0}^{5}(S_{i,t}+c_{i,t})=C_h}
$$

핸드 종료 후:

$$
\boxed{\sum_{i=0}^{5} S^{\text{after}}_{i,h}=C_h},\qquad
\boxed{\sum_i u_i(h)=0}
$$

레이크가 없는 Cash Session에서는 외부 자금 흐름을 따로 추적한다.

$$
\boxed{C_{\text{table}}=\sum B^{\text{in}}-\sum B^{\text{out}}}
$$

참가자 수 $N_h$는 **핸드 시작 시점에 고정**되며 $2\le N_h\le6$이다. 승리/패배의 기본 Hand Reward는

$$
R_i^{\mathrm{cash}}=\frac{S_{i,\mathrm{end}}-S_{i,\mathrm{start}}}{BB}
$$

이다. Buy-In과 Cash-Out을 Hand Reward에 넣지 않는다.

## 2. 기존 Base 코드와의 정확한 연동

- 재사용: `holdem_ai.environment.multi_player_engine.MultiPlayerHandEngine`
- `MultiPlayerHandEngine(stacks=(..., 6 seats), button=i, small_blind=50, big_blind=100, seed=j)`
- `reset()` 이후 `observe(current_player)` / `step(action_id)` / `step_raise_to(amount)` / `is_over()` / `get_result()`를 사용.
- `engine.seats.small_blind`, `engine.seats.big_blind`, `engine.current_player`를 활용.
- **`step_raise_to(amount)`는 현재 Street의 총 누적 베팅 금액**, 추가 지불액이 아님.
- 기존 5 Action ID: fold 0, check/call 1, half-pot 2, pot 3, all-in raise 4. **Short All-In Call**은 call(1)로 변환.
- Phase 3 Hand Engine이 Side Pot, Uncalled Wager, Odd Chips, Tie 정산을 전담한다.

> [!warning] 정보 경계
> `DynamicTable.observe_current()`는 베이스 엔진에서 **현재 행동하는 플레이어의 관측만** 반환한다. 학습용 정형 `InformationState`와 Actor/Critic 강제 분리는 Expand Phase 2의 계약이다. `ledger`/`seed`는 리플레이 목적의 특권 데이터이므로 학습 Actor의 관측에 절대로 입력하면 안 된다.

## 3. 구성 파일

```text
src/holdem_ai/expand/
  __init__.py
  environment/
    __init__.py
    table_state.py       # immutable TableConfig/PlayerSeat/Action/Settlement
    cash_rules.py        # buy-in/top-up 검증
    tournament_rules.py  # E7 extension stub (fail-closed)
    event_ledger.py      # JSONL Journal + Seq
    multiway_adapter.py  # 기존 Phase 3 Hand Engine 연결
    table_manager.py     # DynamicTable + replay_journal
scripts/
  expand_phase01_smoke.py
 tests/expand/
  conftest.py
  test_e01_table_unit.py
  test_e01_real_engine.py
 docs/
  EXPAND_PHASE01_DESIGN.md
```

추가형 Patch로서 기존 Phase 1~16 파일을 수정하거나 덮어쓰지 않는다.

## 4. 공개 객체 계약

```python
from holdem_ai.expand.environment import (
    TableConfig,
    DynamicTable,
    Action,
    TableSnapshot,
    HandHandle,
    StepEvent,
    Settlement,
    replay_journal,
)

table = DynamicTable(TableConfig(mode='cash'), session_id='sixmax-demo')
table.request_join('hero', 10000, preferred_seat=0)
table.request_join('villain', 10000, preferred_seat=1)
hand = table.start_hand(seed=123)

while table.current_player_id is not None:
    obs = table.observe_current()
    action = Action('call' if obs['raw_obs']['to_call'] else 'check')
    event = table.act(table.current_player_id, action)

settlement = table.settle_hand()
print(settlement.payoffs_bb)
print(table.snapshot())
```

`start_hand()`는 아직 2명이 성립하지 않으면 `None`을 반환한다(`WAITING_FOR_PLAYERS`). `settle_hand()`는 종료된 Hand에 한 번만 호출할 수 있다. 새로운 Hand를 시작하기 전에 이전 Hand를 반드시 정산해야 한다.

### 4.1 새 참가자

1. `request_join()`이 Seat를 즉시 `reserved`로 표시하지만 아직 자금을 테이블로 옮기지 않으며 게임에 참가하지 않는다.
2. 다음 핸드 경계에서 Buy-In을 적용한 뒤 `waiting_bb`로 만든다.
3. 기존 참가자 2명 이상이 있으면 다음 Hand에서 **BB를 내는 좌석에 해당할 때만** `seated`로 전환한다. 처음으로 테이블이 2인 구성을 갖추는 경우에는 교착 방지를 위해 즉시 입장한다.
4. Hand 시작 후에는 `participants`가 고정된다. Seat 변경 예약은 진행 중 Hand에 영향을 주지 않는다.

이 `wait_for_big_blind` 구현은 **본 프로젝트의 MVP 규칙**이며 전 세계 Casino의 모든 Seat Entry/Dead Button 규칙을 구현한 것은 아니다.

### 4.2 Leave / Sitout / Resume

`request_leave(player_id)`는 핸드 도중에도 호출할 수 있지만, 실제 Cash-Out은 Settlement 뒤에 일어난다. `request_sitout()`/`request_resume()`도 핸드 경계에만 적용된다. 복귀자는 `waiting_bb`로 들어가 BB 차례를 기다린다.

### 4.3 Bust / Rebuy / All-In

핸드 도중 Stack이 0이 되어도 **All-In Player가 무조건 탈락하는 것은 아니다**. `get_result().final_stacks`가 0인 경우에만 Bust 대상이다. 미리 Queue에 넣은 Rebuy가 있다면 정산 후 채우고 `waiting_bb`로 돌아온다. 그렇지 않으면 Cash Game에서 해당 Seat를 비운다.

Top-up 검증은 Buy-In 최대 300BB를 넘지 않도록 하며, 핸드 중 예약한 Top-up은 **핸드 시작 스택을 기준으로 보수적으로 검사**한다. 손실을 예상하여 미래의 가용 Stack을 미리 차감하지 않는다. 손실 정산이 끝난 뒤 다시 요청하면 현재 Stack 기준으로 검사된다.

### 4.4 토너먼트

`TournamentRuleSet`은 확장을 위한 데이터 타입과 명시적 `NotImplementedError`만 제공한다. 이 Phase에서 Tournament가 지원된다고 표현하면 안 된다. Expand Phase 7의 Tournament 구현에서는 추가로 Blind Level, Ante, Late Registration, Re-entry, Prize EV, Placement Rules를 결정한다.

## 5. 이벤트 로그와 리플레이

JSONL Example:

```json
{"seq":1,"event":"table_created","config":{"mode":"cash"},"session_id":"demo"}
{"seq":2,"event":"request","event_id":"demo:event:00000001","kind":"join","player_id":"hero","seat":0,"chips":10000}
```

위 JSON은 **필드 설명용 축약 예제**로, 실제 `table_created.config`에는 전체 `TableConfig`가 기록된다.

```python
live = DynamicTable(journal_path='runs/expand_phase01/session.jsonl', session_id='demo')
# ... table.request_*, table.start_hand, table.act, table.settle_hand ...
restored = replay_journal('runs/expand_phase01/session.jsonl')
assert restored.snapshot() == live.snapshot()
```

Journal replay는 사건 목록을 단순 로딩하지 않고 **같은 명령을 다시 실행해 전체 생성 이벤트를 비교**한다. 재현 가능한 Seed와 Actor의 실제 행동이 기록돼야 하므로 Journal이 유출되면 카드 배분이 재현될 수 있다. 데이터 보관 권한을 제한하고, 해당 파일을 공개 학습 관측에 섞지 않는다.

**현재 한계:** 이벤트 기록은 각 행을 fsync하는 append-only 저널이지만, 단일 프로세스 이벤트 루프 전용이다. 다중 프로세스 동시 쓰기, 임의 중간 종료 이후 부분 JSONL 행 자동 복구, 주식 거래 수준의 ACID Transaction, 이벤트 로그 스냅샷 압축은 아직 제공하지 않는다. 잘못된 사용으로 중간에 예외가 발생했다면 새 Journal로 안전하게 다시 시작한다.

## 6. 설치: Windows PowerShell

기존 프로젝트 루트 `C:\Users\yiydg\Desktop\workspace\holdem-ai-starter\holdem-ai`에서 실행한다.

ZIP 파일을 같은 디렉터리에 `holdem-ai-expand-phase01` 이름으로 압축 해제:

```powershell
Copy-Item .\holdem-ai-expand-phase01\src\holdem_ai\expand .\src\holdem_ai\ -Recurse -Force
Copy-Item .\holdem-ai-expand-phase01\scripts\expand_phase01_smoke.py .\scripts\ -Force
Copy-Item .\holdem-ai-expand-phase01\tests\expand .\tests\ -Recurse -Force
Copy-Item .\holdem-ai-expand-phase01\docs\EXPAND_PHASE01_DESIGN.md .\docs\ -Force
Copy-Item .\holdem-ai-expand-phase01\docs\EXPAND_MASTER_DESIGN_6MAX_v1_0.md .\docs\ -Force
```

### 6.1 단위 + 실제 엔진 테스트

```powershell
uv run --extra dev --extra gpu python -m pytest -q tests/expand
uv run --extra dev --extra gpu python -m pytest -q
```

테스트 범위: Event Queue, Buy-In, Rebuy, Bust, In-Hand Leave, Sitout/Resume, Duplicate IDs, Replay/Tamper Detection, 실제 Phase 3 Hand Engine, Side-Pot/Uncalled/Odd Chips.

### 6.2 짧은 실제 게임 Smoke (5×10 Hands)

```powershell
uv run --extra gpu python -m scripts.expand_phase01_smoke `
  --hands-per-n 10 `
  --seed 170001 `
  --output-json runs/expand_phase01/smoke_10.json
```

### 6.3 본 검증 (2~6인 목표 각 1,000 Hand, 총 5,000 Hand)

```powershell
uv run --extra gpu python -m scripts.expand_phase01_smoke `
  --hands-per-n 1000 `
  --seed 170101 `
  --output-json runs/expand_phase01/smoke_1000.json
```

`--hands-per-n`의 `n`은 **실제 참가자 수**로 사용한다. Bootstrap/Waiting-BB 이후 정확히 $N$명이 참가한 Hands만 1,000개까지 카운트하며, 준비 단계의 2~$(N-1)$인 핸드는 `warmup_hands`로 별도 기록한다. 따라서 전체 플레이 Hand 수는 5,000개보다 많아질 수 있다.

> [!important] 결과 파일 비덮어쓰기
> 동일 경로에 JSON이 있으면 `FileExistsError`로 중단한다. 다른 Seed/파일명으로 실행한다. `runs/`와 Checkpoint는 별도 로컬 관리한다.

## 7. E1 Acceptance Gate

- [x] 공개 API 및 6좌석 Mutable Registry / Immutable Snapshot 구현.
- [x] Join/Leave/Rebuy/Sitout/Resume의 Hand Boundary Queue 구현.
- [x] Moving Button + `wait_for_big_blind`/Bootstrap 정책 구현.
- [x] Base Phase 3의 Engine Factory로 연결, Action 변환 및 정산 코드 작성.
- [x] Seq Journal 및 전체 이벤트 결정적 Replay 구현.
- [x] 작업 컨테이너에서 isolated unit test 통과.
- [ ] Windows에서 **실제 Base Phase 3 엔진** 통합 테스트 통과.
- [ ] 2~6 Seat Smoke 각 1,000 Hands 실행 및 통계 확인.
- [ ] 기존 전체 Regression Test 통과.
- [ ] GitHub Commit & Push 확인.

다음 Expand Phase는 위 검증 및 Push 완료 후 진행한다.

## 8. Git Push (검증 완료 후)

```powershell
git status
git add src/holdem_ai/expand/ tests/expand/ scripts/expand_phase01_smoke.py docs/EXPAND_PHASE01_DESIGN.md docs/EXPAND_MASTER_DESIGN_6MAX_v1_0.md
git diff --cached --stat
git commit -m "Implement Expand Phase 1 dynamic 6-max cash table"
git push origin main
git status
```

**주의:** 이 Patch에는 기존 Base Phase 1~16을 수정하는 파일이 포함되지 않는다.
