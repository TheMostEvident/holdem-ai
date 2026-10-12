---
title: "6-Max NLHE AI — Expand Phase 1~7 Architecture Specification"
aliases:
  - "Expand Master Design"
  - "6-Max Pluribus-inspired AI Architecture"
tags:
  - poker-ai
  - 6max
  - no-limit-holdem
  - reinforcement-learning
  - mccfr
  - pluribus
  - architecture
  - cash-game
  - tournament
created: 2026-10-12
version: "1.0"
status: "Design Specified — Implementation Not Started"
project: "TheMostEvident/holdem-ai"
---

# Expand Phase 1~7 — 6-Max No-Limit Hold'em AI 상세 아키텍처 명세서

> [!abstract] 이 문서의 목적
> 기존 **Base Phase 1~16(HUNL 중심 구현)**을 보존하고, 새로운 **Expand Phase 1~7(동적 2~6인 No-Limit Hold'em)**을 **각 단계별로 설계 → 구현 → 테스트 → 실험 → GitHub Push**하는 개발 계약을 확정한다. 이 문서는 **상세 설계 문서이지, 구현 완료 보고서가 아니다.**
>
> **운영 결정:** Cash Game을 먼저 구현한다. Tournament는 동일한 Hand Engine·TableState·InformationState 위에 서로 다른 운영 규칙과 보상 구조를 추가해 지원한다.

## 0. 최신 상태·용어·설계 경계

| 항목 | 현 상태 | 주의 |
|---|---|---|
| Base Phase 1~16 | 소스 개발 및 GitHub Push 기록 존재 | HUNL 전용 학습을 6-Max 학습으로 오인하지 않는다 |
| Base Phase 15 | 학습 인프라 코드/테스트 완료 | 5시간 장시간 PPO/Q-Boost 학습은 미실행 |
| Base Phase 16 | Frozen Actor 상대 풀 평가 코드 및 소규모 GPU 파일럿 완료 | 6명이 동시에 플레이한 테스트가 아님 |
| Expand Phase 1~7 | **이 문서로 설계 기준 확정** | 소스·학습·평가를 새로 구현해야 함 |
| 연구 출처 | Brown & Sandholm, *Science* 2019 및 30p Supplement | Pluribus 원본 소스/훈련된 Blueprint 미확보 |

### 0.1 분류

- **HUNL**: 2인 Heads-Up No-Limit Texas Hold'em. Base Phase 6~16의 주요 RL 대상.
- **6-Max NLHE**: 최대 6개 물리 좌석에 2~6명의 플레이어가 참여하는 No-Limit Texas Hold'em.
- **Cash Game**: 테이블 간 칩 구매·환급이 가능한 지속 세션; 기본 보상은 칩 수익.
- **Tournament**: 블라인드 상승·탈락·상금 분배·등록 규칙이 있는 세션; 최종 보상은 상금/순위 기반.
- **Blueprint**: 오프라인 Self-Play로 학습한 추상 게임 전략.
- **Search**: 현재 공개 정보와 가능한 비공개 카드 분포를 바탕으로 제한 깊이 부분게임을 계산하는 온라인 전략 도출.
- **Expand Phase**: 기존 Base Phase 번호와 완전히 독립된 6-Max 개발 단계.

> [!warning] 논문 재현성의 한계
> Pluribus 원본의 핵심은 **External-Sampling Linear MCCFR 기반 Blueprint + Depth-Limited Imperfect-Information Search**다. 원본은 고정된 6인/100BB 시작 스택으로 평가했다. 이 문서의 Dynamic Cash/Tournament와 PPO·Opponent Adaptation은 **원본 외 확장**이다. 따라서 각 기능에 `ORIGINAL`, `ADAPTATION`, `NEW` 출처 라벨을 부여한다. 6인 CFR의 수렴이나 Pluribus 수준 성능은 보장하지 않는다.

### 0.2 설계 원칙

1. 한 번에 **한 Expand Phase**만 구현한다. 단계별 테스트/Push가 끝나야 다음 Phase로 이동.
2. 공통 `HandEngine`의 **게임 규칙**과 `TableManager`의 **운영 규칙**을 분리.
3. Hand의 공정성·칩 보존·합법 행동·비공개 정보 경계를 학습 성능보다 먼저 검증.
4. **Actor의 관측**과 (훈련할 때만 사용 가능한) **Critic용 특권 정보**를 별도 타입으로 분리.
5. 연구 구현에서 정확성(작은 게임) → 재현성 → 자원 관리 → 6-Max 확장 순으로 진행.
6. 출처의 실제 숫자와 제안한 파일럿 설정을 구분.
7. 6-Max Cash와 Tournament에서 평가 기준을 혼용하지 않는다.
8. 학습 Seed, Validation Seed, Final Test Seed는 분리하여 Test 오염을 방지.

---

# Part I. 시스템 요구사항

## 1. Product Specification

### 1.1 목표

시간에 따라 플레이어 구성·스택·상대 전략이 달라지는 테이블에서, 정보집합에 조건화된 **확률적 혼합전략**을 학습하고 합법적인 홀덤 행동을 결정한다.

$$
\pi_\theta(a\mid I_{i,t},\mathcal N_t),\qquad
2\leq |\mathcal N_t|\leq 6.
$$

- \(\mathcal N_t\): 다음 Hand에 참가할 수 있는 플레이어 집합.
- \(I_{i,t}\): 플레이어 \(i\)의 비공개 카드·공개 카드·스택·베팅 이력 등 법적으로 관측 가능한 정보.
- \(a\): Fold, Check, Call, Raise-to, All-In 등의 실제 유효 행동.

### 1.2 필수 기능

| ID | 요구사항 | 검증 방법 |
|---|---|---|
| FR-01 | 2~6인 한 손(Hand)의 완전한 규칙 진행 | 모든 N에 대한 Property Tests |
| FR-02 | 플레이어 입장·퇴장·재구매·버스트 처리 | Event Sequence Tests |
| FR-03 | Seat와 Player의 고유 식별자 분리 | ID 불변성 테스트 |
| FR-04 | All-In·Multiway Side Pot·Tie·Odd Chip 처리 | 정산 기준 사례/반례 |
| FR-05 | Cash 규칙으로 무기한 Session 진행 | 1,000+ Hand Stress Test |
| FR-06 | Tournament 등록·탈락·상금·블라인드 확장 | 별도 운영 정책 테스트 |
| FR-07 | Actor는 자신의 카드와 Public State만 열람 | Hidden-Information Noninterference Test |
| FR-08 | 2~6인 Variable-Size Input 지원 | Mask 및 Permutation Tests |
| FR-09 | 6인 Multi-Agent RL 학습·재개 가능 | Pilot Update / Resume Regression |
| FR-10 | Pluribus-inspired MCCFR와 Search가 각각 독립 평가 가능 | Toy Poker와 6-Max Pilot |
| FR-11 | 다양한 상대·스택·포지션에 대한 평가/CI | Holdout Evaluation Protocol |

### 1.3 범위

**MVP Cash:** 6좌석, 2~6인, Buy-In 20~300BB(제안 값), 기준 100BB, 고정 블라인드, 레이크 0, Ante 0, 핸드 사이 입·퇴장, 승패 후 Stack 이월. **설계 단위는 정수 칩**, 기본 SB=50, BB=100칩으로 고정해 0.5BB 문제를 피한다.

**Tournament:** 별도 RuleSet으로 Blind Schedule, Buy-In/Registration, Re-entry Window, Prize Ladder를 명세하고 Expand 1부터 인터페이스를 고정하되, 검증·확장은 단계적으로 수행한다. 금전 보상 정책은 샌드박스 환경 내 시뮬레이션으로 제한.

**초기 비범위:** 분산 멀티노드 전체 Pluribus 계산량 재현, 원본 Blueprint 복원, 원본 C++ 코드 복제, 실제 현금 게임 자동화/외부 서비스 접속, 선수 성능 우월성 선언.

---

# Part II. 수학적 게임 정의

## 2. Extensive-Form / Partially Observable Stochastic Game

핸드 내부는 고정된 참가자 수 \(N_h\)를 갖는 불완전정보 전개형 게임으로 정의한다.

$$
\mathcal G_h=(\mathcal P_h,\mathcal H,\mathcal Z,P,\{A(h)\},T,u,\{\mathcal I_i\}),
\quad 2\leq N_h=|\mathcal P_h|\leq 6.
$$

- \(\mathcal H\): Hand의 전체 History(Deck 포함).
- \(\mathcal Z\): Terminal History.
- \(P(h)\): 행동자 또는 Chance.
- \(u_i(z)\): 종료 시 플레이어 \(i\)의 순 칩 손익.
- \(\mathcal I_i\): 플레이어가 구분할 수 없는 History들의 분할.

**한 손 내부 레이크가 없다면**

$$
\sum_{i\in\mathcal P_h}u_i(z)=0.
$$

테이블의 Hand 간 운영까지 포함하면 플레이어 입퇴장과 카드 분포를 포함하는 가변 참가자 확률 게임이 된다. Hand 경계에서만 참여자 집합 \(\mathcal N_t\) 변경을 허용한다.

### 2.1 Hand 상의 칩 보존

핸드 시작 총액 \(C_h\), 시점 \(t\)의 각 플레이어 미투입 스택 \(S_{i,t}\), Pot에 투자된 칩 \(c_{i,t}\)에 대해:

$$
\boxed{\sum_i(S_{i,t}+c_{i,t})=C_h},\qquad
\boxed{\sum_iS_{i,\mathrm{after}}=C_h}\quad(\mathrm{rake}=0).
$$

Side Pot은 Pot의 분할일 뿐 새로운 칩을 생성하지 않는다. Dead Money, Uncalled Bet 반환, Short All-In을 포함해 검증해야 한다.

### 2.2 Session 상의 칩 이동

$$
C^{\mathrm{table}}_{h+1}=C^{\mathrm{table}}_h
+\sum_iB^{\mathrm{in}}_{i,h}
-\sum_iB^{\mathrm{out}}_{i,h}
-\mathrm{rake}_h.
$$

외부 Buy-In을 보상으로 해석하지 않는다. Cash Hand 보상은 **핸드 시작 스택과 종료 스택의 차이**로 정의한다.

$$
r^{\mathrm{cash}}_{i,h}
=\frac{S_{i,h}^{\mathrm{end}}-S_{i,h}^{\mathrm{start}}}{BB_h}.
$$

\(S^{start}\)는 Blind 공제 이전 값. 세션 평가에서는 관측 Hand 수 기준으로 \(100\bar r\) = bb/100을 기록한다.

**Tournament:** 최종 보상을 \(R_i^{\mathrm{tournament}}=p_{\mathrm{rank}(i)}\) (선정한 상금 단위)로 별도 정의한다. 중간 칩 변화는 Critic 보조 신호일 수 있으나 최종 상금 EV와 동치라고 가정하지 않는다.

### 2.3 게임 동안 상태 변경

- `seat_occupancy`: Seat를 점유하는 Player 인스턴스 존재 여부.
- `hand_eligible`: 다음 핸드 참여 가능 여부.
- `hand_status`: `NOT_IN_HAND | ACTIVE | FOLDED | ALL_IN | SHOWDOWN_ELIGIBLE`.
- `table_status`: `EMPTY | WAITING | SEATED | SITTING_OUT | PENDING_LEAVE | BUSTED`.
- `ALL_IN`은 **현재 Hand의 Showdown 대상**이고 `BUSTED`와 다르다.

`stack == 0`이라는 단일 조건으로 핸드 중 탈락 처리하지 않는다.

---

# Part III. 전체 소프트웨어 아키텍처

## 3. 상위 컴포넌트

```text
                       CashRuleSet       TournamentRuleSet
                            \                 /
                             \               /
                              TableManager
                                   |
               Join/Leave/Buyin <- EventLedger -> SeatRegistry
                                   |
                            MultiPlayerHandEngine [Base Phase 3]
                                   |
                     Public State / Private State Guards
                                   |
                           InformationState v1
                                   |
                         ActionTranslator / Mask
                                   |
                      +------------+-------------+
                      |                          |
                  Neural RL Track            MCCFR Track
               Seat Encoder + PPO       Abstraction + Blueprint
                      |                          |
             Multi-Agent Collector       Depth-Limited Search
                      |                          |
                      +------------+-------------+
                                   |
                          CandidatePolicy API
                                   |
                      EvaluationArena / Reports
```

### 3.1 기존 코드와의 연결점

| 기존 코드 | 검증된 역할 | Expand에서의 처리 |
|---|---|---|
| `src/holdem_ai/environment/multi_player_engine.py` | 2~6인 Hand, 합법 액션, Side Pot 정산, `step_raise_to` | **재사용·회귀 테스트**. 새 Table Manager가 Hand 사이 Stack을 전달 |
| `src/holdem_ai/environment/tournament_env.py` | 누적 Stack, 블라인드 레벨, 0-Stack 탈락 | 공통 코어에 맞춰 **정책 어댑터로 리팩터링**. 초기 버전의 종료/재진입 제한 유지 명시 |
| Base Phase 5 Agent Interface | 정책 호출 표면 | `PolicyV2` 어댑터 제공 |
| Base Phase 6~8 | HUNL Actor, Encoder, Network | 6인용 Encoder와 Mask가 신규 필수. **기존 체크포인트 직접 호환으로 가정 금지** |
| Base Phase 9~15 | PPO/Q-Boost, GPU 수집, Checkpoint | Optimizer·Checkpoint 패턴 재사용 가능; **6인 Collector는 새 구현** |
| Base Phase 16 | 상대 풀, BB/100, 조건부 행동 프로파일 | 6인용 평가에서 Seat와 상대 집합·군집 CI 반영 |

### 3.2 권장 소스 구조 — 제안, 아직 생성 전

```text
src/holdem_ai/expand/
  __init__.py
  common/
    types.py
    rng.py
    seed_registry.py
    checkpoint.py
  environment/
    table_state.py
    seat_registry.py
    table_manager.py
    cash_rules.py
    tournament_rules.py
    event_ledger.py
    multiway_adapter.py
  information/
    schema.py
    encoder.py
    masks.py
    history_encoder.py
    policy_view.py
    critic_view.py
  actions/
    translator.py
    abstraction.py
    legal.py
  rl/
    policy.py
    multiagent_collector.py
    ppo.py
    opponent_population.py
    training_runner.py
  pluribus/
    blueprint/
      infoset.py
      regret_store.py
      external_sampling.py
      blueprint_export.py
    abstraction/
      card_features.py
      card_buckets.py
      bet_sizes.py
    search/
      public_belief.py
      nested_solver.py
      continuation.py
      off_tree.py
      resource_budget.py
  adaptation/
    opponent_memory.py
    style_tracker.py
    guardrail.py
  evaluation/
    arena.py
    metrics.py
    diagnostics.py
    holdout.py
  integration/
    policy_router.py
    artifact_manifest.py
scripts/
  expand_phase01_smoke.py ... expand_phase07_benchmark.py
tests/expand/
  test_e01_*.py ... test_e07_*.py
docs/
  EXPAND_MASTER_DESIGN.md
  EXPAND_PHASE01_DESIGN.md ... EXPAND_PHASE07_DESIGN.md
```

### 3.3 공통 API 계약 (의사 타입 명세)

```python
from dataclasses import dataclass
from typing import Literal, Protocol

SeatId = int              # 0..5, stable physical seat index
PlayerId = str            # unique player identity, never recycled
HandId = str

@dataclass(frozen=True)
class TableConfig:
    mode: Literal["cash", "tournament"]
    small_blind: int
    big_blind: int
    max_seats: int = 6
    min_buyin_bb: int = 20
    max_buyin_bb: int = 300
    rake_bps: int = 0
    seating_rule: str = "moving_button"

@dataclass(frozen=True)
class PlayerSeat:
    seat_id: SeatId
    player_id: PlayerId | None
    stack_chips: int
    seat_status: str

@dataclass(frozen=True)
class TableSnapshot:
    config: TableConfig
    seats: tuple[PlayerSeat, ...]  # always length 6
    button_seat: SeatId | None
    hand_number: int
    session_id: str

@dataclass(frozen=True)
class Action:
    kind: Literal["fold", "check", "call", "raise_to", "all_in"]
    raise_to_chips: int | None = None  # total wager THIS betting street

class PokerPolicy(Protocol):
    def act(self, observation: "InformationState", rng_seed: int) -> Action: ...
```

> [!warning] `raise_to` 의미
> Base Phase 3 `step_raise_to(amount)`는 *현재 베팅 Street에 이 플레이어가 총 투입할 목표액*이다. 추가로 지불할 칩 수가 아니다. 포맷 혼동은 치명적인 버그이므로 변환 계층에서 검증한다.

---

# Part IV. Expand Phase 1~7 상세 설계

## Expand Phase 1 — Dynamic 6-Max Table Environment

**구현 대상:** Cash 운영 코어를 먼저 구현하고 Tournament RuleSet용 인터페이스를 남긴다. Base Phase 3의 Hand Engine을 교체하지 않고 래핑한다.

### E1.1 상태 전이 명세

```text
VACANT --join request--> RESERVED --hand boundary--> SEATED
SEATED --sitout--> SITTING_OUT --resume--> SEATED
SEATED --leave request during hand--> PENDING_LEAVE --settle--> VACANT
SEATED --final stack == 0--> BUSTED --cash removal--> VACANT
BUSTED --rebuy--> RESERVED (Cash) OR REENTRY_PENDING (Tournament rules)
ALL_IN (hand only) --showdown--> SEATED or BUSTED
```

**핸드 진행 중 `join`, `leave`, `rebuy`, 좌석 교체는 즉시 적용 금지.** 변경 이벤트를 Queue에 넣고 Hand Settlement 이후 원자적으로 적용한다. 참여자 수 \(<2\)이면 다음 Hand를 시작하지 않고 `WAITING_FOR_PLAYERS`에 머문다.

### E1.2 규칙 전략

- `CashRuleSet`: 고정 SB=50, BB=100; 20~300BB Buy-In; 스택 이월; 0 Stack 후 자동 자리 정리; 신규 입장 대기열; 레이크 0.
- `TournamentRuleSet`: Blind Schedule, Ante(초기 0), Registration Close, Re-entry Policy, Prize Structure, Placement Rule, Table Completion. **E1에서는 명세와 Stub만**, 별도 Tournament 전체 행동 검증은 후속 수용 시험에서 확장.
- Button: 초기에는 기존 엔진과 호환되는 `moving_button` 규칙을 명시적으로 선택. 빈 좌석·신규 참가·헤즈업 전환 때 이동을 Reference Rule과 비교 시험. **실제 Casino의 Dead Button/Waiting for BB 정책은 분리 가능한 선택형 규칙**으로 설계하고 구현·테스트 전 '공식 모든 룰을 지원한다'고 주장하지 않는다.
- 현금 게임 참가자 Entry: 다음 Hand부터 활성화; Blind를 피하기 위한 부정 참가(예: BTN 입장)를 방지하기 위해 `wait_for_big_blind` 또는 `post_to_enter`를 RuleSet에 명시적으로 둔다. **MVP 선택: `wait_for_big_blind`**. 신규 참가자는 별도 대기 마커를 가진다. **교착 방지:** 한 명만 이미 착석해 있고 신규 참가자가 처음으로 2인 게임을 성립시키는 경우, `TableManager`는 첫 핸드의 BTN/BB를 공정하게 배정하면서 두 번째 플레이어를 즉시 활성화해야 한다. 아무 핸드도 열 수 없는 상태에서 BB 차례를 영원히 기다리게 하지 않는다.
- Tournament Late Registration: 기본 비활성; 구체적인 Window와 Re-entry Stack Rule을 지정할 때만 활성.

### E1.3 핵심 함수 계약

```python
class DynamicTable:
    def request_join(self, player_id: str, buyin_chips: int, preferred_seat: int | None = None) -> str: ...
    def request_leave(self, player_id: str) -> str: ...
    def request_rebuy(self, player_id: str, amount_chips: int) -> str: ...
    def start_hand(self, seed: int) -> "HandHandle": ...
    def act(self, player_id: str, action: Action) -> "StepEvent": ...
    def settle_hand(self) -> "Settlement": ...
    def apply_boundary_events(self) -> None: ...
    def snapshot(self) -> TableSnapshot: ...
```

`request_*`는 이벤트 ID를 반환하고, 실제 변경은 Hand Boundary에서 `EventLedger`에 한 번만 기록한다(멱등성). 플레이어 ID/좌석 ID/핸드 ID를 동일한 타입으로 취급하지 않는다.

### E1.4 상태 불변식

$$
\begin{aligned}
&0\le S_i\in\mathbb Z,\qquad|\mathrm{seats}|=6,\\
&\text{one PlayerId per occupied seat, one seat per PlayerId},\\
&\sum_i(S_{i,t}+c_{i,t})=C_h,\\
&\sum_i S_{i,h}^{\rm end}=C_h\quad(\text{rake}=0),\\
&\text{changing participants} \implies \text{between hands only}.
\end{aligned}
$$

### E1.5 테스트 계획 / 완료 기준

- 최소 2~6명 각 1,000 Hand RNG Smoke; Side Pot 수작업 기준 사례.
- **Short All-in** 후 Call 및 Reopening, 여러 Short All-In 합산, Fold된 투자금 분배, Tie Odd Chips.
- Heads-Up ↔ 3인 ↔ 6인 전이, 신규 플레이어 SB/BB 우선권, Button 이동.
- 핸드 중 Leave, Join, 재구매 동시 요청, 중복 요청, Seat 충돌, Seat0 제거 후 Seat0 새 ID 입장.
- Money Ledger 정확성; `sum(stacks)+pot` 불변.
- 결정적 Seed Replay 동일 Hand History, 파일 기반 이벤트 재생.
- Base Phase 3/4/16 기존 테스트가 깨지지 않음.

**E1 산출물:** `DynamicTable`, `CashRuleSet`, `TableSnapshot`, `EventLedger`, 테스트, Smoke CLI, `EXPAND_PHASE01_DESIGN.md`. **금지:** E1에서 PPO 학습 및 Pluribus Blueprint 구현.

---

## Expand Phase 2 — 6-Max Information State, Seat Encoding & Action Abstraction

**목표:** Variable N을 처리하는 Actor Observation과 불완전정보 격리·합법 Action Translation을 만든다.

### E2.1 공개 정보와 개인 정보

플레이어 \(i\)의 관측값은

$$
I_{i,t}=\big(H_i,B_t,P_t,\mathbf S_t,\mathbf C_t,\mathbf M_t,\mathrm{Position}_i,\mathrm{ActionHistory}_t,\mathrm{LegalMask}_t\big).
$$

- `hero_hole`: 자기 카드 2장만.
- `board`: 공개 카드 0/3/4/5장.
- `pot`, `street`, `to_call`, `min_raise_to`, `max_raise_to`.
- `seats[6]`: Stack(칩 또는 BB), Street Contribution, Total Contribution, Seat/Hand 상태, 상대 공개 Player ID(옵션), Relative Position.
- `history`: Action의 **seat_id / street / action type / 실제 raise_to**.
- `legal`: 현재 Player에 대한 유효 Action 목록 + Action Size Bound.
- **제외:** 상대 Hole Cards, 미래 카드, RNG 내부 상태, 숨겨진 Showdown 전략·범위의 진실값.

타입을 분리한다.

```python
@dataclass(frozen=True)
class PublicSeatFeatures:
    occupied: bool
    dealt_in: bool
    folded: bool
    all_in: bool
    stack_bb: float
    invested_bb: float
    street_bet_bb: float
    relative_position: int

@dataclass(frozen=True)
class InformationState:
    hero_seat: int
    hero_hole: tuple[str, str]
    board: tuple[str, ...]
    street: int
    seats: tuple[PublicSeatFeatures, ...]  # 6
    action_history: tuple["PublicAction", ...]
    legal_actions: "LegalActionSpec"
    pot_bb: float
    big_blind_chips: int

@dataclass(frozen=True)
class TrainingOnlyCriticState:
    # Never passed to the acting policy, evaluation, or real-time agent.
    full_hand_state: object
```

### E2.2 Neural Input

각 Seat 특징 벡터 \(x_j\in\mathbb R^{d_s}\), Position/Street/Hand Global 벡터 \(g\), Mask \(m_j\in\{0,1\}\):

$$
Z=\operatorname{TransformerEncoder}(x_0,\dots,x_5;\,m_0,\dots,m_5),
$$
$$
\ell=f_\theta\big(\operatorname{Pool}(Z,m),g,H_i,B_t\big),
\quad
\pi_\theta(a\mid I)=\operatorname{MaskedSoftmax}(\ell,\mathcal A(I)).
$$

`seat_id`는 고정 물리 위치, `relative_position`은 Button 기준 회전된 위치로 구별한다. Occupied지만 Sitout 상태인 좌석은 Seat Token 유지하되 `eligible/dealt_in` Mask를 따로 사용한다. Tournament에서 탈락자는 `occupied=0` 또는 Tournament 정책의 다른 명시적 상태로만 처리.

### E2.3 실 Action API ↔ Abstract Action

Base Phase 3의 액션: `0=FOLD, 1=CHECK_CALL, 2=HALF_POT_RAISE, 3=POT_RAISE, 4=ALL_IN`, 그리고 `step_raise_to`(Custom Raise-to).

Expand에서 중립적 타입은 `Action(kind, raise_to_chips)`. `CHECK_CALL`을 `to_call==0`이면 CHECK, 그 외 CALL로 해석한다.

초기 RL Discrete Action Set (제안):

$$
\mathcal A_{\rm abs}=\{\mathrm{fold,check,call,raise}_{0.5P},\mathrm{raise}_{P},\mathrm{raise}_{2P},\mathrm{allin}\}.
$$

`raise_(kP)`는 **Raise-after-call Size Definition**을 정확히 하나로 고정한다. 금액 계산:

$$
\mathrm{toCall}=\max(0,b_{\max}-b_i),\qquad
P'=P+\mathrm{toCall},
$$
$$
\mathrm{targetRaiseTo}=b_i+\mathrm{toCall}+\lfloor kP'\rfloor.
$$

합법 범위는 `min_raise_to <= target <= street_bet_i+stack_i`, 그렇지 않으면 그 Abstract Action은 Mask. `ALL_IN`은 최대 베팅액을 뜻하지만 모든 All-In이 Raise 가능한 것은 아니다. Short All-In 가능 여부는 Hand Engine의 최종 Rule이 판정.

> [!important] Call/Check와 Fold 구분
> `CHECK` 가능 시 `FOLD`를 UI에 보여줄 수는 있지만 학습 목표에서는 불필요한 Fold를 마스킹할지 규칙적으로 정의해야 한다. 초기 구현에서는 **게임 엔진이 유효하다고 판단하는 Action은 모두 허용**하고 전략 편향 없이 분포를 학습한다.

### E2.4 정보 누출 검증

- 상대 Hole Cards만 변경한 두 상태에서 동일 Public + Hero Observation 여부.
- 미래 Deck을 변경해도 현재 Actor Observation 동일 여부.
- Actor Serialization에서 Critic State가 전송되지 않음.
- 합법 Action Mask가 2~6인, Fold/All-In/Short Stack 조건마다 Engine과 동일.
- 무작위 색상 Suit Permutation / Seat Rotation 테스트: 의미를 보존하는 변환 범위에서 Encoding이 일관적인지 검증.

**E2 산출물:** `InformationState v1`, `SeatEncoder`, `ActionTranslator`, `PolicyV2`, 검사 및 문서. **금지:** 카드 Bucketing이 정확한 GTO 전략을 보장한다는 주장.

---

## Expand Phase 3 — Multi-Agent PPO Self-Play Baseline

**목표:** Expand E1/E2 환경에서 실제 2~6인 Self-Play 데이터를 수집하고 PPO 기반 학습이 안정적으로 작동하는지를 검증. **PPO는 Pluribus 원본 구성요소가 아님(NEW).**

### E3.1 정책 인구 및 Seat 배정

- `trainable`: 새 6-Max Actor.
- `frozen snapshots`: 이전 버전 정책.
- `scripted`: Random/Calling Station/Tight/Pot Pressure/기초 Rule Agent.
- 한 게임 세션에 2~6명 배정; 필요하면 동일 Trainable Weight를 여러 좌석에서 공유하되 **Gradient Update는 수집 완료 후 수행**.
- \(N\sim p_N\). MVP \(p_N\): 2~6 균일. 6-Max 성능 확보를 위해 \(N=6\) 최소 비율도 따로 관리.
- Cash 초기 Stack \(S_i/BB\sim p_S\). 초기 100BB 중심에서 확장(커리큘럼).
- Snapshot Ratio는 실험 Config에 고정. 평가에서는 학습 중 사용한 Opponent Pool과 **별개의** 상대를 포함.

### E3.2 보상

한 Hand의 Cash 보상:

$$
r_{i,h}=\frac{S_{i,h}^{end}-S_{i,h}^{start}}{BB}.
$$

Multi-Agent Reward \(\sum r_{i,h}=0\) (레이크 0). 행동 시점 중간 보상은 기본 0, Terminal에서 Player별 해당 Hand 보상을 부여. Hand 도중 포기·Leave 요청하더라도 후속 정산을 반드시 포함.

### E3.3 PPO

시간 단계별 Advantage \(\hat A_t\)에 대해:

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid I_t)}{\pi_{\theta_{old}}(a_t\mid I_t)},
$$
$$
L_{\rm clip}(\theta)=\mathbb E_t\left[
\min\{\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\}\right].
$$

전체 손실(최대화 표기):

$$
J= L_{\rm clip}-\lambda_V L_V+\lambda_H\,\mathbb E[\mathcal H(\pi_\theta)].
$$

GAE는 Trajectory를 적절한 Episode로 분리하고 **Hand/Tournament Termination**, `done/truncated`, 플레이어의 행동 시점 간 간격을 구분한다. 6인 플레이어의 Decision Steps가 서로 인터리브되므로 Seat별 행동 시퀀스를 각각 재구성한다.

### E3.4 Collector 계약

```python
@dataclass(frozen=True)
class MultiAgentTransition:
    session_id: str
    hand_id: str
    player_id: str
    seat_id: int
    observation: InformationState
    action: Action
    action_logprob: float
    value: float
    reward_bb: float
    terminal_hand: bool
    terminal_session: bool
    policy_version: str

class MultiAgentCollector:
    def collect(self, *, num_hands: int, seat_policy_assignments: object, seed: int) -> list[MultiAgentTransition]: ...
```

학습용 Observation과 Real-Time Acting Observation의 Schemas는 같아야 한다. Critic만 별도 Privileged Features를 받을 수 있으나 출처와 안정성 실험을 구분해 기록.

### E3.5 학습·재개·성능 검증

- `latest.pt`, `best.pt`, `run.json`, `state.json`, `metrics.jsonl`, SHA-256, RNG States, Opponent Pool Manifest.
- GPU 메모리 측정, 1/3/6인 Rollout Time, Policy Entropy, Approx KL, Illegal Action Count.
- Pilot: 3 Updates와 Hand End-to-End.
- 2~6인과 전략 혼합 그룹에서 Checkpoint Load, Resume가 동일 설정을 복원.
- Baseline vs Random/Tight 등: 큰 차이의 통계적 해석은 충분한 독립 Seed/Cluster 수집 후.

**E3 완료 정의:** Multi-Agent PPO가 실 Hand의 보상으로 학습 업데이트를 하며, 정책·Checkpoint·평가가 재현 가능하다. **강한 게임 실력은 별도 결과.**

---

## Expand Phase 4 — Pluribus-style MCCFR Blueprint

**원본 출처:** External-Sampling Linear MCCFR, Negative-Regret Pruning, Action/Information Abstraction (`ORIGINAL`); 6인 캐시 변형과 작은 자원 설정은 `ADAPTATION`.

### E4.1 개발 순서 (반드시 작은 게임부터)

1. Kuhn Poker: External Sampling의 Regret/Strategy 축적 검증.
2. Leduc Poker: Chance Sampling, Showdown, Information Sets, 2인 게임의 전략 수렴 경향 테스트.
3. 3인 작은 추상 게임: Terminal Zero-Sum·Regret 업데이트 검증. **다인 수렴 보장 아님.**
4. 2~6인 제한된 Betting Tree: Memory/Time 예산 테스트.
5. 6-Max Blueprint 파일럿: Current Strategy 및 일정 Snapshot의 평가.

### E4.2 Regret

행동 \(a\in\mathcal A(I)\)의 누적 Counterfactual Regret:

$$
R_i^T(I,a)=\sum_{t=1}^T w_t
\left[v_i^{\sigma^t}(I,a)-v_i^{\sigma^t}(I)\right].
$$

Regret Matching:

$$
\sigma_i^{T+1}(I,a)=
\begin{cases}
\dfrac{[R_i^T(I,a)]_+}{\sum_b[R_i^T(I,b)]_+},&\sum_b[R_i^T(I,b)]_+>0,\\[4pt]
\dfrac1{|\mathcal A(I)|},&\text{otherwise}.
\end{cases}
$$

### E4.3 External Sampling MCCFR: 추상 구현 계약

```text
for iteration t:
    traverser = t mod N
    sample chance events (deals)
    recursively traverse:
        terminal -> reward vector
        chance node -> sample by true chance distribution
        other player's node -> sample according to current strategy
        traverser's node -> evaluate each legal abstract action
           compute sampled counterfactual regret difference
           update regret storage
    write iteration metrics & safe checkpoint
```

실제 Counterfactual Weight, Sampling Probability Correction은 원본 Supplement Algorithm 1과 Reference Toy Solver 테스트로 확정하며, 위 도식만으로 충분한 구현이 아니다.

### E4.4 Linear CFR 및 Pruning

원본은 처음 400분 동안 10분마다 Discounting, 200분 뒤 매우 부정적인 Regret를 가진 행동을 95% 반복에서 Prune하는 방법을 사용했다. 이 수치는 **원본의 칩 단위·Storage Scale·CPU 설정**에 묶여 있으므로 MVP에서 그대로 복사하지 않는다.

- V1: **Pruning OFF**로 정확성 확인.
- V2: Linear Weight/Discounting을 단독 도입해 Toy Game 비교.
- V3: Pruning 도입; 마지막 Street/즉시 Terminal Action은 Prune 제외.
- Pruning 활성화 전 `relative regret threshold`를 포맷과 수치 범위에 맞춰 정의.
- Negative Regret Skip Percent와 Action Coverage 로깅.

원본 참고: Supplement pp.14–17 (텍스트 p.14~16의 Regret/Snapshot/Algorithm 1 구간).

### E4.5 Card/Bet Abstraction

- Original Blueprint: Preflop Lossless, 이후 Street별 200 Buckets.
- Original Real-Time Search: 현재 Street Lossless, 이후 Street별 500 Buckets.
- 현재 하드웨어 기준 **MVP 제안**: 16/32/64 Bucket Ablation, 2~4개 Raise Size로 작동 검증. 200/500을 곧바로 달성했다고 주장 금지.
- Feature 예시: Showdown Equity Estimate, Equity Distribution, Future Hand Potential, Board Texture. 카드 Bucketing의 학습 입력/Seed/메트릭을 고정하고 중복 카드 차단.
- Abstraction Fingerprint: `hash(config + centroid arrays + schema_version)`; 학습·Search·Evaluation 모두 동일한 호환 Abstraction ID 사용.

### E4.6 저장소·자원

Regret/Strategy에 각각 float32 \(4\) bytes를 쓴다면 단순 dense 메모리 근사:

$$
M\approx 8|\mathcal I|\cdot\bar A\ \mathrm{bytes}+M_{\mathrm{metadata}}.
$$

Sparse Infoset 생성, 압축 저장, 메모리 상한 8~12GB 중 안전값 설정(시스템 RAM 16GB에 맞춰 측정 후 결정). 원본은 4바이트 정수 Regret 및 구조화된 Sparse Storage 최적화를 사용했다. Python Dict만으로 대형 실제 Blueprint를 처리한다는 보장은 없음.

### E4.7 완료 기준

- Kuhn/Leduc 수치 레퍼런스와 Regret 및 전략 sanity test.
- 무작위 시드에서 Invalid Action/Probability/NaN = 0.
- Save-Load 이후 Abstraction ID/Regret Fingerprint 동일.
- 6인 축소 게임 실행 및 제한된 메모리·시간에서 안전 종료.
- 2인 toy equilibrium metric은 참고하되 6인 전체 GTO 주장 금지.

---

## Expand Phase 5 — Depth-Limited Subgame Search

**목표:** Pluribus 원본 Supplement의 Nested Search와 Leaf Continuations를 원칙적으로 구현하고, 자원 부족·Off-Tree 상황에 안전한 Fallback을 제공한다.

### E5.1 Public Belief와 Range

플레이어 \(j\)의 가능한 홀카드 \(h_j\)에 대한 추정 분포:

$$
b_{j,t}(h_j) \propto
b_{j,0}(h_j)
\prod_{k\le t:\,\mathrm{actor}(k)=j}
\hat\pi_j(a_k\mid h_j,\mathcal H_k^{public}).
$$

**Joint Card Consistency:** 여러 상대의 Hand Range는 덱 충돌 제약으로 완전히 독립적이지 않다. 정확한 계산은 \(P(H_1,\ldots,H_N\mid\mathrm{public})\)의 Card Exclusion을 반영해야 한다. 구현은 정확한 Joint Enumeration 또는 충돌 없는 샘플링으로 근사하되 그 오류를 기록한다.

`PublicBelief`의 입력은 공개 이력, 내 카드, 공개 보드, Blueprint/상대 정책 가정만 허용. 실제 Rival Hole Cards나 future RNG를 Search 평가에 사용하지 않는다.

### E5.2 Subgame Root

- 현재 Betting Round의 **시작 Public State**를 Root로 정의.
- 상대가 현재 추상화 밖 Action을 사용하면 그 Action을 Public Tree에 삽입하고 Root부터 다시 계산.
- 이전에 실제 선택한 **내 Hole Card에 해당하는 실제 결정**의 Action 확률은 고정. 상대의 과거 행동 정책은 새 Search에서 변화 가능.
- Street가 바뀌면 Root를 새 Street 시작 Public State로 변경.
- 문제 크기 제한 시 Blueprint 또는 Safe-Fallback Policy로 전환.

### E5.3 Continuation Strategy

Pluribus의 원본 구성:

$$
\mathcal C=\{\pi_B,\pi_F,\pi_C,\pi_R\}.
$$

\(\pi_B\): Blueprint, \(\pi_F\): Fold에 가중치 5배, \(\pi_C\): Call 5배, \(\pi_R\): 모든 Raise Action 5배. 모든 재가중은 **원래 합법 Action 집합 위에서 재정규화**한다.

$$
\pi_F(a\mid I)=
\frac{w_F(a)\pi_B(a\mid I)}{\sum_{a'} w_F(a')\pi_B(a'\mid I)},\qquad
w_F(a)=\begin{cases}5,&a=\mathrm{fold}\\1,&\text{otherwise.}\end{cases}
$$

Call/raise도 동일 방식. Leaf에 도달한 각 활성 Player는 \(\mathcal C\) 중 하나 또는 혼합분포를 선택하는 확장 Action을 가진다. 이 선택은 해당 Player가 구분하지 못하는 Leaf Infoset에서 일관되어야 한다.

> [!important] 원본과 차이
> 원본 Supplement에는 **네 개의 Continuation Strategy**가 명시되어 있다. 단순히 Leaf에서 항상 Blueprint로 계속하거나 별도 Neural Value만 반환하는 것은 Pluribus의 Nested Search와 동등하지 않다.

### E5.4 Search Contract / Budget

```python
@dataclass(frozen=True)
class SearchBudget:
    max_time_ms: int
    max_nodes: int
    max_memory_mb: int
    max_iterations: int

@dataclass(frozen=True)
class SearchResult:
    action_probabilities: dict[str, float]
    status: str  # solved / timeout / resource_guard / fallback
    iterations: int
    nodes: int
    elapsed_ms: float
    search_root_id: str
    abstraction_id: str
```

시작 설정 **제안**: max_time_ms=500, max_memory_mb=1024, Max Nodes는 측정 기반. 실제 실행에선 Timeout 시 반환한 Distribution의 합이 1이어야 하며, Invalid Action은 반드시 0.

### E5.5 원본 Nested Search 최소 동작

```text
initialize current_public_state, blueprint_strategy
on opponent_action(a):
  if a is off-tree:
    add actual action to public subgame
    re-solve from betting-round root
  advance public history
  if new street: update root; re-solve
on our_turn:
  sample action from current search strategy
  mark our actual infoset/action frozen
  advance history
  if new street: update root; re-solve
```

### E5.6 완료 기준

- Private-Information Leak / Card Collision 금지.
- Toy Subgame exact EV/Strategy sanity.
- Off-Tree Raise-to Action과 Search Re-rooting 검증.
- Search Recompute 시 본인이 과거에 실제 선택한 Action 확률 보존.
- Leaf Continuation Choice에서 같은 Player Infoset 일관성.
- 동일 Seed Search 재현성, Resource Limit과 Fallback 항상 정상 작동.
- Search vs Blueprint A/B 평가. +bb/100이라도 표본 부족 시 성능 개선 주장하지 않음.

---

## Expand Phase 6 — Opponent Modeling & Adaptive Strategy

**목표:** Pluribus 원본의 고정 Blueprint 기반 Search와 별도로, 실제로 변하는 상대 전략의 추정·제어·성능 평가를 수행한다. `NEW` 연구로 표시한다.

> [!warning] Pluribus와의 구분
> 원본 Pluribus는 실전에서 상대의 장기 성향을 온라인으로 학습하는 식의 명시적 Opponent Adaptation을 목표로 하지 않았다. 따라서 본 Phase의 Opponent Modeling은 원본 설계의 복제가 아닌 별도 확장 연구다.

### E6.1 상대 행동 패턴

공개 이벤트만 저장. 플레이어 \(j\)에 대해 다음을 관찰:

$$
\hat f_j(a\mid \mathrm{street},\mathrm{position},\mathrm{SPR\ bucket},\mathrm{legal\ set})
=\frac{N_{j,a}+\alpha_a}{N_{j,\mathrm{eligible}}+\sum_{a'}\alpha_{a'}}.
$$

사전분포(Prior), 최소 샘플 수, Exponential Decay, Street별 VPIP/PFR/3-Bet 등의 진단을 명세하되 Legal Action으로 조건화한다. 플레이어 교체 시 Seat ID를 기준으로 통계를 이어붙이지 말고 **PlayerId + Session**으로 키를 관리한다.

### E6.2 Nonstationary Opponents

상대 분포가 바뀌는 구간 \(\tau\)를 정의하고 탐지:

$$
D_t=\mathrm{KL}(\hat p_{j,t}^{recent}\|\hat p_{j,t}^{long})
$$

은 후보 Change Indicator. KL은 표본이 작으면 불안정하므로 스무딩·Bootstrap/절제 기준을 강제한다. 탐지 실패보다 **잘못된 상대 추정에 과도하게 의존하는 전략 변화**를 제한하는 것이 우선.

### E6.3 Robustness Guard

- 기본 전략 = Blueprint+Search 또는 PPO Baseline.
- Adaptive Modification은 Evidence가 충분할 때만 적용.
- 행동 분포가 지나치게 치우치지 않도록 Trust Region / KL Budget:

$$
D_{\mathrm{KL}}(\pi_{\mathrm{adapt}}(\cdot|I)\|\pi_{\mathrm{base}}(\cdot|I))\le\delta.
$$

- Adversarial/Deceptive Opponent 테스트. Style Shift 발생 시 자동 Fallback.
- 실제 실시간 적응 여부는 엄격한 Holdout 상대와 Change-Point Scenario에서 검증.

### E6.4 완료 기준

- 같은 Seat에 새 Player가 들어오면 추정 데이터가 섞이지 않음.
- 적은 표본에서는 기본 정책 유지.
- Opponent Swap 이후 Change Detection 및 회복 속도 로깅.
- Adaptive vs Nonadaptive 정책을 같은 Scenario Block으로 비교하며 과적합/Exploitability Proxy를 측정.
- 이 Phase에서 학습 데이터를 Test Deck과 공유하지 않음.

---

## Expand Phase 7 — Integration, Tournament Extension & Scientific Benchmark

**목표:** 1~6 단계의 코어를 하나의 Policy API로 묶고, Cash Evaluation을 완료한 뒤 Tournament 모드를 확장·검증한다.

### E7.1 Policy Routing

```python
class IntegratedPokerAgent:
    def act(self, obs: InformationState, rng_seed: int) -> Action:
        # 1. Validate observation + legal actions.
        # 2. Determine available blueprint/search/adaptive components.
        # 3. Prefer verified policy according to routing manifest.
        # 4. Use guarded fallback on timeout/incompatibility.
        # 5. Enforce exact ActionTranslator validation.
        ...
```

정책 라우팅 예:

```text
E2/3 Neural baseline ---------------------------> fallback
E4 Blueprint --(compatible abstraction)--------> preflop/base
E5 Search -----(time & memory budget)-----------> refined policy
E6 Adaptation --(sufficient evidence & KL cap)--> guarded adjustment
```

**동일 시점에 여러 경로를 합칠 경우 구체적인 우선순위, 모든 Fallback 조건 및 Strategy Profile Provenance를 저장.**

### E7.2 Tournament Architecture

- `TournamentRuleSet`은 `TableState`와 `HandEngine`을 그대로 사용.
- 추가 정책: Rising Blinds, Antes, Late Registration Window, Re-entry Count/Stack, Payout Structure, Rank Assignment, Table Ending.
- 0 Stack이 Hand 종료 시 확정되면 Elimination Event; Re-entry가 허용되면 **새 참가 이벤트**로 취급하되 Tournament Identity/Entry ID 별도 관리.
- Winner-Take-All과 Multi-Payout의 Return Function 분리.
- ICM은 Tournament Approximation/Feature로 도입 가능하지만 정확한 Prize EV와 동일하다고 단정하지 않는다.
- E7에서는 기본 Single-Table Tournament까지를 수용 기준으로 정하고 2개 이상 테이블 병합은 확장 과제.

### E7.3 현금 보상과 토너먼트 목표 구분

Cash:

$$
J_{\rm cash}(\pi)=\mathbb E[\Delta \mathrm{chips}/BB],
\quad \widehat{bb/100}=100\overline{\Delta \mathrm{chips}/BB}.
$$

Tournament:

$$
J_{\rm tournament}(\pi)=\mathbb E[p_{\mathrm{rank}(i)}],
$$

Tournament 성과는 한 번의 우승 여부 외에 **Mean Prize EV, Win Rate, ITM Rate, Elimination Time, Risk by Stack Regime** 등을 평가한다. 한 Tournament를 하나의 독립 통계 단위로 취급하는 것을 기본으로 한다.

### E7.4 Benchmark Matrices

| Axis | 설정 |
|---|---|
| 인원 | N=2,3,4,5,6 |
| 스택 | 20/50/100/200/300BB 및 비대칭 혼합 |
| 좌석 | Button·Relative Position 균형화 |
| 상대 | Random, Calling Station, Tight, Pot Pressure, PPO Snapshot, CFR Blueprint, Adaptive Opponent |
| 변동 | 입장, 퇴장, Bust, Buy-In, Rebuy, Opponent Strategy Shift |
| 정책 | Neural PPO, Blueprint, Search, Search+Adaptive, Full Integrated |
| 모드 | Cash 주평가 → Tournament 보조 평가 |
| 검증 | Held-out Deck/Training Seed/Scenario Splits |

### E7.5 통계

Hand Cluster별 순손익을 \(X_b\)라 할 때 Cash:

$$
\hat\mu_{bb/100}=100\cdot\frac1B\sum_{b=1}^B X_b,
\quad \mathrm{SE}=100\frac{\operatorname{sd}(X_b)}{\sqrt B}.
$$

단, **동적 테이블에서는 인접 Hand가 Stack/상대/Session을 공유하여 독립이 아닐 수 있음**. 그러므로 기본 CI는 Session 단위 Cluster Bootstrap 또는 독립 Seat/Deck Rotation Block을 사용하고, 단순 Hand IID CI는 진단용으로만 표시한다.

알고리즘 A/B 비교는 **동일 시드·좌석·상대 시나리오**를 사용해 가능하면 Block 수준 대응 차이 \(D_b=X_b^A-X_b^B\)를 계산한다. 큰 손실과 Rare All-In의 높은 분산을 고려해 다중 독립 학습 Seed를 보고한다.

### E7.6 최종 완료 조건

- 통합 모델이 Cash에서 2~6인, 다양한 Stacks, Join/Leave/Bust, Opponent Shift를 처리.
- Tournament 단일 테이블이 Blind Schedule·등록·Re-entry 설정·상금 래더에 따라 종료.
- Illegal Action=0, Hidden Info Leak=0, Chip Conservation 위반=0.
- 이종 상대 풀과 새 Deck에서 Reproducible Evaluation 및 CI.
- Search-Time/Mem Budget 준수 및 Latency/Fallback 통계.
- Release Manifest에 Git Commit, Checkpoint SHA, Config, Seed Bank, Abstraction ID, Evaluation Script Hash 저장.
- 성능 우월성 주장은 통계적으로 입증되는 범위만 표시.

---

# Part V. 연구 명세·학습 실행 프로토콜

## 4. Randomness and Reproducibility

동적 Session에서 별도 RNG Namespace를 둔다.

```text
SeedRegistry
  environment_deal  [session,hand]
  seat_assignment   [session,hand]
  opponent_sample   [session,hand]
  policy_sampling   [player,hand,decision]
  model_init        [experiment]
  training_batches  [experiment,update]
  validation        [separate immutable bank]
  final_test        [sealed independent bank]
```

Seed와 Deck를 저장해 재현하더라도 **Hidden Deck가 Actor 관측으로 들어가는 일은 금지**. 학습과 평가의 동일 Card Bank 재사용은 시나리오 통제를 목적으로만 허용, 최종 Holdout Card Bank는 학습/모델 선택에 미사용.

## 5. Artifact Metadata

모든 학습 모델은 아래 Manifest를 가진다.

```yaml
artifact_schema: expand-v1
expand_phase: 3
source_commit: "<git-sha>"
environment_ruleset: "cash-v1"
observation_schema: "information-v1"
action_schema: "abstract-actions-v1"
policy_architecture: "six-seat-transformer-v1"
training_seed: 0
training_deck_namespace: "<seed-bank-id>"
validation_deck_namespace: "<seed-bank-id>"
checkpoint_sha256: "<sha256>"
```

MCCFR/Search에서는 `abstraction_id`, `regret_checkpoint`, `blueprint_version`, `search_budget`를 더 기록.

## 6. 우선순위와 리스크

| 우선순위 | 위험 | 대응 |
|---|---|---|
| P0 | Invalid Raise / Pot 정산 오류 | E1 Property/Oracle Tests |
| P0 | Opponent Hole Cards 유출 | E2 Noninterference Tests |
| P0 | Cash Buy-In을 Agent Reward로 혼동 | Hand-level Reward 계산·Ledger 경계 |
| P1 | HUNL Actor를 그대로 6인에 사용 | E2 새 Observation·Network 및 Compatibility Gate |
| P1 | Multi-Agent 학습의 불안정성 | Snapshot Opponents, KL/Entropy, 세션 Seed Holdout |
| P1 | MCCFR CPU 메모리 폭증 | Sparse Store, Toy Game, Memory Budget, Pruning 실험 |
| P1 | Search에서 합법 Off-Tree 행동 무시 | Actual Raise-to Injection + Nested Re-solving |
| P1 | Pluribus의 성능을 그대로 기대 | Algorithmic Verification와 Real-Game Evaluation 분리 |
| P2 | Adaptive Opponent Overfit | Evidence Threshold, KL Cap, Opponent Shift Benchmark |
| P2 | Tournament Cash-Equity 오용 | Prize-EV Objective 분리 |

### 6.1 개발 시스템 제약

기존 테스트 환경은 Windows / Python 3.11 / `uv`, CUDA 11.8 / GTX 1070 8GB / RAM 16GB였음. 모든 본 학습 시간·Search 처리량은 **실측 전 미확정**. 원본 Pluribus는 64-core CPU 서버 8일, 약 12,400 CPU core-hours, Blueprint 생성 시 512GB 미만 메모리를 사용했다. 따라서 Original Scale을 단일 GTX1070 시스템에서 빠르게 재현한다는 가정은 하지 않는다.

---

# Part VI. 단계별 산출물 및 개발 운영 규칙

## 7. 단계별 계약

| Expand Phase | 명칭 | 필수 주요 산출물 | 다음 단계 진입 Gate |
|---|---|---|---|
| **1** | Dynamic 6-Max Table | TableManager, CashRuleSet, EventLedger, Chip Invariants | 2~6인 Hand/입퇴장/SidePot 회귀 테스트와 Push |
| **2** | Information State & Abstraction | Public Observation, Seat Encoder, Legal Action Translator | Hidden Info Leak=0, Action Translator 일치, Push |
| **3** | Multi-Agent PPO Baseline | Collector, PPO, Opponent Population, Resume | GPU/CPU Pilot, Reproducibility, Push |
| **4** | MCCFR Blueprint | Toy MCCFR, Abstraction, Regret Store, Blueprint | Toy References / 6인 축소 게임 / Memory Guard, Push |
| **5** | Depth-Limited Nested Search | Public Belief, Leaf Continuations, Off-Tree, Resource Guard | Search Correctness/Timeout/Fallback, Push |
| **6** | Opponent Adaptation | StyleTracker, Change Detection, Robust Policy Cap | Shift Holdout/Ablation 및 과적합 점검, Push |
| **7** | Integration & Tournament | PolicyRouter, Cash Benchmark, Tournament RuleSet | Full Regression, 다양한 상대/인원 테스트, Push |

### 7.1 각 Phase의 동일한 진행 방식

```text
1. 해당 Expand Phase 설계 명세 검토
       ↓
2. 구현 파일 및 테스트 추가 (기존 Base 수정 최소화)
       ↓
3. pytest 전용 테스트 통과
       ↓
4. 전체 프로젝트 Regression 테스트 통과
       ↓
5. CPU/GPU Pilot 실제 실행 (해당 Phase에서 필요할 때)
       ↓
6. 결과 분석 및 미해결 항목 기록
       ↓
7. git add 특정 파일 → commit → push
       ↓
8. 다음 Expand Phase로 진행
```

**이번 요청으로는 설계만 완료하며 코드 수정·Push·학습은 실행하지 않는다.**

### 7.2 실행 커맨드의 계약(미래에 구현되기 전에는 실행 불가)

```powershell
# Expand Phase 1 구현이 완성된 후에만 예시로 사용할 커맨드
uv run --extra dev --extra gpu python -m pytest -q tests/expand -k e01
uv run --extra dev --extra gpu python -m pytest -q
uv run --extra gpu python -m scripts.expand_phase01_smoke

# Expand Phase 3 구현 후
uv run --extra gpu python -m scripts.expand_phase03_train --help

# Expand Phase 4 구현 후
uv run --extra gpu python -m scripts.expand_phase04_mccfr --help

# Expand Phase 7 구현 후
uv run --extra gpu python -m scripts.expand_phase07_benchmark --help
```

CLI 옵션은 해당 Phase 구현 시 상세 확정. 위 명령은 **파일 네이밍의 목표 계약이지 현재 동작하는 코드가 아니다.**

---

# Part VII. 원본 Pluribus 대조표와 참고문헌

## 8. 원본 대조표

| 항목 | Original Pluribus | Expand 설계 |
|---|---|---|
| 게임 | 6인, 시작 100BB | 2~6인, Stack 이월/가변 |
| Blueprint | External Sampling Linear MCCFR | 원리 재구현 + 메모리 제약용 축소 설정 |
| Blueprint Card Abstraction | Preflop Lossless, 이후 200 Buckets | Toy → 16/32/64 → 가능한 범위 확대 |
| Search Card Abstraction | 현재 Street Lossless, 이후 500 Buckets | 동일 원리, 초기 축소 Search |
| Action Abstraction | 상황별 여러 베팅 크기 | 정확한 Raise-to 변환·Resource-bounded action set |
| Regret Pruning | 원본 95% 일부 반복 Pruning | 정확성 검증 후만 활성 |
| Off-Tree | Off-Tree Raise 삽입 및 Nested Re-solving | E5에 그대로 설계 반영 |
| Leaf | Blueprint/Fold/Call/Raise 4 Continuations | 동일 원리, 구현·평가 후 활용 |
| PPO | 원본 핵심 아님 | 독립 Neural RL Baseline `NEW` |
| Opponent Modeling | 장기 상대 스타일 적응 중심 아님 | `NEW`, E6에서 연구 |
| Dynamic Cash | 원본 평가 실험 밖 | `NEW`, E1 |
| Tournament | 원본 평가 실험 밖 | `NEW`, E7 |

## 9. 참고문헌·재현성

1. Brown, N., & Sandholm, T. (2019). *Superhuman AI for multiplayer poker*. **Science**, 365(6456), 885–890. DOI: [10.1126/science.aay2400](https://doi.org/10.1126/science.aay2400).
2. **Original Supplement**: [Noam Brown — Supplementary Materials (30 pages)](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf) — Card Abstraction, Blueprint Algorithm 1, Nested Search Algorithm 2, Continuation Strategies.
3. **Original Paper**: [Noam Brown — Science Paper](https://noambrown.github.io/papers/19-Science-Superhuman.pdf).
4. Meta AI Research: [Pluribus: First AI to Beat Pros in 6-Player Poker](https://ai.meta.com/blog/pluribus-first-ai-to-beat-pros-in-6-player-poker/) — Original Pluribus는 인간 상대 성향을 장기 온라인 학습하는 시스템으로 제시되지 않음.
5. Repository: [TheMostEvident/holdem-ai](https://github.com/TheMostEvident/holdem-ai) — Base Phase 1~16 구현 근거.
6. 기존 Obsidian 연구 기록: `HUNL_AI_Complete_Obsidian_Phase0-16_2026-10-12.md`, `Holdem_AI_Cash_vs_Tournament_Research.md`.

## 10. 미해결·명시적 선택 사항 (구현 직전 확인)

| ID | 설계 선택 | 현 결정 | 확인 시점 |
|---|---|---|---|
| OPEN-01 | Cash 입장 Blind Rule | `wait_for_big_blind` (MVP) | Expand 1 구현 전 |
| OPEN-02 | Button Rule | 기존 Moving Button 기준, Dead Button은 후속 | Expand 1 테스트 설계 |
| OPEN-03 | Tournament Late Registration | 초기 비활성, RuleSet으로 확장 | Expand 7 |
| OPEN-04 | Rake | 최초 0; 변경 가능 Config | Expand 1 |
| OPEN-05 | Cash Buy-In | 20~300BB, 기본 100BB | Expand 1 |
| OPEN-06 | Card Bucket 규모 | Toy/16/32/64 → 자원 기준 조정 | Expand 4 |
| OPEN-07 | Search 상한 | 초기 500ms/1024MB 제안, 실측으로 조정 | Expand 5 |
| OPEN-08 | PPO-CFR 결합 | 독립 트랙으로 시작, 향후 별도 연구 | Expand 7 |
| OPEN-09 | Tournament 평가 가치 | Payout EV 우선, ICM 보조 | Expand 7 |

> [!success] 설계상의 완료 선언
> Expand Phase 1~7의 **목적·선후관계·컴포넌트 계약·수학적 정의·완료 Gate**를 이 문서에서 확정했다. 이것은 **구현 완료 선언이 아니다**. 이후 사용자가 요청할 때 **Expand Phase 1 한 단계만** 구현한다. 그 단계의 실제 테스트 결과와 Push가 확인되면 Expand Phase 2로 넘어간다.
