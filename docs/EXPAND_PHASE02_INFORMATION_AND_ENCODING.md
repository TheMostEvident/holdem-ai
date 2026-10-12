---
title: "Expand Phase 2 — 6-Max Information State, Seat Encoder & Action Translator"
project: "HUNL → Dynamic 6-Max NLHE"
phase: "Expand Phase 2"
version: "1.0.0-patch"
date: 2026-10-12
status: "standalone tests passed; real Base Phase 3 and Windows GPU acceptance pending"
tags:
  - poker
  - six-max
  - reinforcement-learning
  - obsidian
  - information-state
  - no-leakage
  - expand-phase02
---

# Expand Phase 2 — 6-Max Information State, Seat Encoder & Action Translator

> [!important] 범위와 Gate
> **본 Phase는 관측·인코딩·합법 행동 변환과 비학습 Neural Policy까지만 구현한다.** `PolicyV2`는 무작위 초기화된 네트워크이며 학습된 강력한 홀덤 AI가 아니다. 6인 PPO 학습은 Expand Phase 3, Pluribus-style MCCFR은 Expand Phase 4, Subgame Search는 Expand Phase 5, Tournament 운영은 Expand Phase 7의 별도 작업이다. **Expand Phase 1의 Windows 통합 테스트가 아직 끝나지 않았으며, 이 패치의 Windows 실제 게임/GTX 1070 검증도 아직 수행하지 않았다.** 이 두 보류 사항을 3~7단계에서 `완료`로 간주하지 않는다.

## 1. 설계 목표

최대 6개의 **물리 좌석**을 유지하고 **실제 핸드 참여 인원은 2~6명**인 Cash Game에 대해, 오직 행동권이 있는 플레이어에게 공개된 정보만 전달한다.

$$
I_{i,t}=(H_i,B_t,P_t,S_{0:5},C_{0:5},M_{0:5},p_i,h_t^{\mathrm{public}},\mathcal A(I_{i,t})).
$$

- $H_i$: 본인 Hole Cards 2장. 상대 카드 금지.
- $B_t$: 공개 Board 0/3/4/5장.
- $P_t$: Pot 크기.
- $S_j$: Seat $j$의 잔여 스택.
- $C_j$: Seat $j$의 Pot 기여금 및 현재 Street의 베팅.
- $M_j$: 물리 좌석 점유, 현재 핸드 참가, Fold, All-In, Sitout/Waiting 상태.
- $p_i$: 현재 Button 대비 상대 위치.
- $h_t^{\mathrm{public}}$: 공개 행동 이력의 행동 당시 Street·좌석·베팅 금액.
- $\mathcal A$: 합법 행동 및 Raise 금액.

> [!warning] 비공개 상태의 엄격 분리
> **금지 입력**: 다른 플레이어 Hole Cards, 미래 Deck, 덱 RNG Seed, 내부 Engine 객체, EventLedger, Hand Result가 나오기 전의 Showdown Truth, `TrainingOnlyCriticState`.
> 
> `make_information_state()`는 `DynamicTable.observe_current()`에서 얻는 **현재 행동 플레이어의 raw_obs를 allowlist로 복사**한다. 구현자가 임의의 `observation['hidden_...']`, `raw_obs['seed']`, `raw_obs['opponent_holes']`를 추가해도 반환하는 객체가 달라지지 않아야 한다.

### 1.1 데이터 구조

```python
@dataclass(frozen=True)
class PublicSeatFeatures:
    occupied: bool
    dealt_in: bool
    folded: bool
    all_in: bool
    seat_status: str       # current hand state (active/folded/allin/out)
    table_status: str      # seated/waiting_bb/sitting_out/reserved/...
    stack_bb: float
    invested_bb: float
    street_bet_bb: float
    relative_position: int # physical clockwise offset from button 0..5
    is_hero: bool

@dataclass(frozen=True)
class InformationState:
    hero_seat: int
    hero_hole: tuple[str, str]
    board: tuple[str, ...]
    street: str
    button_seat: int
    seats: tuple[PublicSeatFeatures, ...]  # always length 6
    action_history: tuple[PublicAction, ...]
    legal_actions: LegalActionSpec
    pot_bb: float
    to_call_bb: float
    big_blind_chips: int
    num_dealt_in: int
```

**세 가지 마스크는 서로 다르다.**

$$
M_j^{\mathrm{occupied}}=\mathbf 1[\text{player at physical seat }j]
$$

$$
M_j^{\mathrm{dealt}}=\mathbf 1[\text{player }j\text{ received cards in this hand}]
$$

$$
M_k^{\mathrm{legal}}=\mathbf 1[a_k\in\mathcal A(I)]
$$

예를 들어 `sitting_out` 플레이어는 `occupied=1`, `dealt_in=0`일 수 있다. 핸드 도중 Fold했다면 `dealt_in=1`, `folded=1`이다. 이 두 경우를 하나의 `active` Flag만으로 처리하면 안 된다.

> `relative_position=(seat-button)%6`는 **물리적 좌석의 시계 방향 거리**이지 UTG/HJ/CO 명칭이나 참가 플레이어 순서가 아니다. Phase 3의 시계 방향 행동순서 학습에 필요한 추가 포지션 특징은 별도 확장이 가능하다.

### 1.2 이력의 Street 보존

Base Phase 3 `raw_obs['action_history']`는 `(seat_id, action_id)`만 보유하여 과거 행동의 Street가 빠져 있다. **현재 Street를 과거 행동에 일괄 붙이면 잘못된 학습 데이터가 된다.**

`Phase2TableAdapter`는 `act()` **실행 직전** `InformationState`에서 Street / Pot / toCall / Raise-to를 기록하고, `DynamicTable.act()` 성공 후에만 `PublicAction`을 확정한다. 엔진과 추적기의 action count 또는 seat가 불일치하면 관측 생성을 **실패 처리**한다.

`Phase2TableAdapter`는 핸드 **시작 전** 연결하고, 이후 `start_hand()`, `act_abstract()`, `settle_hand()`를 반드시 이 Adapter를 통해 호출한다. 중간에 E1 `table.act()`를 우회하면, 다음 E2 Observation은 예외를 발생시킨다. 이것은 **실패를 숨기지 않는 설계**다.

---

## 2. Action Abstraction — 정확한 베팅 수식

$$
\mathcal A_{\mathrm{abs}} = \{\mathrm{Fold},\mathrm{Check},\mathrm{Call},\mathrm{Raise}_{0.5P},\mathrm{Raise}_P,\mathrm{Raise}_{2P},\mathrm{AllIn}\}.
$$

**인덱스 0..6은 순서가 고정**된다.

| Index | Name | 대상 E1 Action |
|---:|---|---|
| 0 | `fold` | `Action('fold')` |
| 1 | `check` | `Action('check')` (`to_call=0`) |
| 2 | `call` | `Action('call')` (`to_call>0`) |
| 3 | `raise_half_pot` | `Action('raise_to', target)` |
| 4 | `raise_pot` | `Action('raise_to', target)` |
| 5 | `raise_double_pot` | `Action('raise_to', target)` |
| 6 | `all_in` | `Action('all_in')` — Raising All-In만 |

**크기 계산은 정수 칩 기준으로 수행한다.**

- $b_i$: 현재 Street에서 Hero의 누적 베팅
- $c=\mathrm{toCall}$
- $P=\mathrm{pot}$ (현재까지 Pot 총액)
- $S_i$: Hero의 현재 잔여 스택

$$
P'=P+c,\qquad T(k)=b_i+c+\lfloor k P'\rfloor,\qquad k\in\{0.5,1,2\}.
$$

$$
T_{\min}=\mathrm{minRaiseTo},\qquad T_{\max}=b_i+S_i.
$$

해당 사이즈가 합법인 조건:

$$
M^{\mathrm{legal}}_{\mathrm{raise}(k)} = \mathbf{1}[\mathrm{raising\_permitted}]\cdot\mathbf{1}[T_{\min}\le T(k)\le T_{\max}]\cdot\mathbf{1}[T(k)>b_i+c].
$$

`raising_permitted`는 **기존 Phase 3 엔진의 합법 Action ID 4 (raising All-In 가능)**을 권위 있는 지표로 사용한다. Half/Pot Size에는 엔진의 ID 2/3도 동시에 확인한다. `DoublePot`은 Custom `step_raise_to()` 경로로 제공한다. 짧은 All-In **Call**은 `call` 하나로 정규화하여 행동 중복을 없앤다.

동일한 목표 칩 금액을 가진 Raise Size는 정책에서 1개만 허용한다. 우선순위 `HalfPot → Pot → DoublePot → AllIn`. 목표액이 상한/하한을 벗어나면 **조용히 Clamp하지 않고 Action Mask에서 제외**한다.

마스크가 적용된 Actor의 확률은

$$
\pi_\theta(a\mid I)=
\frac{M_a\exp(\ell_a-\max_b\ell_b)}
{\sum_{b}M_b\exp(\ell_b-\max_c\ell_c)},
\qquad M_a=0\Rightarrow\pi_\theta(a\mid I)=0.
$$

샘플링된 Action은 반드시 `to_table_action`을 거쳐 E1 Table Engine에 넘긴다. **E1의 Full `raise_to` API는 보존되지만 E2 정책의 베팅 사이즈는 3개 추상 크기 + All-In으로 제한된다.** 미래 Pluribus-style Search는 더 많은 Action Size를 별도로 설계할 수 있다.

### 2.1 Action Translation 수치 예제

$$
P=150,\quad b_i=0,\quad c=100,\quad T_{\min}=200,\quad S_i=10,000.
$$

$$
T(0.5)=225,\quad T(1)=350,\quad T(2)=600,\quad T_{\max}=10,000.
$$

---

## 3. 6-Seat Encoder 및 PolicyV2

```text
InformationState
  ├── SeatFeature[6,19], occupiedMask[6]
  ├── CardOneHot[52 + 52]   # Hero Hole / Public Board
  ├── StreetOneHot[4], GlobalFeature[6]
  ├── PublicHistory[128,21], historyMask[128]
  └── LegalMask[7]
           ↓
    SeatTransformer(width=64)
    + GRU(PublicHistory)
    + Global Card/Bet Features
           ↓
    FeedForward → logits[7]
           ↓
    MaskedSoftmax → action
```

$$
X_{\mathrm{seat}}\in\mathbb{R}^{6\times19},\quad
M_{\mathrm{seat}}\in\{0,1\}^{6},\quad
X_{\mathrm{global}}\in\mathbb{R}^{110},
$$

$$
X_{\mathrm{history}}\in\mathbb{R}^{128\times21},\quad
M_{\mathrm{history}}\in\{0,1\}^{128},\quad
M_{\mathrm{legal}}\in\{0,1\}^{7}.
$$

**좌석 Token의 입력 마스크:** `occupied=1`인 좌석은 Sitout여도 Transformer Attention에 포함한다. 빈 좌석만 Padding한다. 좌석의 절대 Index보다 Button 대비 상대 거리를 Feature로 사용하며, 공개 베팅 이력도 Button 상대 좌표로 인코딩한다.

**Card 표현:** `cA` = Club Ace, `hK` = Heart King 형식. Hero Hole과 Board를 각각 52길이 Multi-Hot으로 인코딩한다. Suit Rotation은 채널 순열에 대응하며 모델 자체가 Suit Permutation Invariant인 것은 아니다.

**History 길이:** 현재는 최근 128개의 공개 행동을 사용한다. 이전 이력을 InformationState에 보관하되 Neural 입력에서는 가장 최근 128개로 잘라 사용한다. 카드·보드 등의 수치 입력은 Big Blind로 정규화하고 큰 숫자에는 부호 보존 Log1p 압축을 적용한다.

**PolicyV2:** PyTorch TransformerEncoder + GRU + MLP. `forward(list[EncodedObservation]) -> masked_logits[B,7]`을 제공하고 모델 출력의 합법 행동 확률 합은 1이다. 무작위 초기화된 Actor만 제공하며, PPO 학습·Checkpoint·Optimizer·Self-Play Collector는 Expand Phase 3에서 구현한다. PyTorch는 `information.PolicyV2`에 접근할 때 Lazy Import된다.

> [!warning] Real-time / Pluribus 제약
> 이 Phase의 Card OneHot은 **Pluribus 200/500 Buckets**가 아니다. 이 Phase는 MCCFR, GTO, Nash Equilibrium, Exploitability를 계산하지 않는다.

---

## 4. 안전 불변식과 검증 게이트

| 검증 | 현재 로컬 검증 | 실제 Windows 엔진 검증 |
|---|---|---|
| 6좌석 정보 구조, Sitout/Waiting Mask | 단위 통과 | 보류 |
| Hero 카드 외 비공개 필드 제외 | 단위 통과 | 보류 |
| Fold/Check/Call/Half/Pot/Double/AllIn | 단위 통과 | 보류 |
| Short AllIn Call은 `call`로만 선택 | 단위 통과 | 보류 |
| `raise_to` 정수 칩 정확성 및 중복 제거 | 단위 통과 | 보류 |
| 공개 이력의 Street 추적과 불일치 감지 | Fake Engine 단위 통과 | 보류 |
| `PolicyV2` 7 Action Mask·Gradient | CPU 단위 통과 | CUDA 보류 |
| 물리 좌석 순열에 대한 Attention 일관성 | CPU 단위 통과 | CUDA 보류 |
| Base Phase 3와 통합(2~6명) | Real-Engine 전용 테스트 존재 | **보류** |
| Base Phase 1~16 전체 Regression | 현 Sandbox에 전체 소스 없음 | **보류** |

**회귀 테스트의 필수 불변식:**

$$\sum_i u_i(h)=0,\quad\sum_i S_{i,\mathrm{after}}=\sum_i S_{i,\mathrm{before}}.$$

$$\forall a\notin\mathcal A(I),\quad\pi_\theta(a\mid I)=0.$$

$$\mathrm{Obs}(H_{-i}, D_{\mathrm{future}})=\mathrm{Obs}(H'_{-i},D'_{\mathrm{future}}),$$

마지막 동등성은 본인 Hole / Public State / 행동 이력을 **고정**하고 다른 플레이어의 비공개 카드와 미래 덱만 바꾸는 조건을 의미한다.

---

## 5. 설치 및 실행 (Windows PowerShell, Phase 1이 먼저 설치되어 있어야 함)

ZIP `holdem-ai-expand-phase02.zip`을 프로젝트 루트로 이동한 후:

```powershell
Expand-Archive .\holdem-ai-expand-phase02.zip `
  -DestinationPath .\holdem-ai-expand-phase02 -Force

# Only add new Phase2 information package; never overwrite Phase1 source.
Copy-Item .\holdem-ai-expand-phase02\src\holdem_ai\expand\information `
  .\src\holdem_ai\expand\ -Recurse -Force

Copy-Item .\holdem-ai-expand-phase02\scripts\expand_phase02_smoke.py `
  .\scripts\ -Force

Copy-Item .\holdem-ai-expand-phase02\tests\expand\test_e02_*.py `
  .\tests\expand\ -Force

Copy-Item .\holdem-ai-expand-phase02\docs\EXPAND_PHASE02_INFORMATION_AND_ENCODING.md `
  .\docs\ -Force
```

> [!important] 주의: 파일 복사 순서
> 사용자 프로젝트에 **Expand Phase 1**이 아직 설치되어 있지 않다면 먼저 `holdem-ai-expand-phase01.zip`을 설치한다. 이 Phase 2 패치는 **Phase 1 소스 파일과 conftest.py를 덮어쓰지 않는다**. `pip`/`uv`가 `src/holdem_ai` 패키지를 이미 인식한다는 기존 프로젝트 구조를 사용한다.

### 5.1 Phase 2 Unit + Engine Integration

```powershell
uv run --extra dev --extra gpu python -m pytest -q -k e02
uv run --extra dev --extra gpu python -m pytest -q tests/expand
uv run --extra dev --extra gpu python -m pytest -q
```

실제 Base Phase 3 소스가 설치되어 있으면 `test_e02_real_engine.py`는 Skip하지 않고 실행된다.

### 5.2 Smoke (2~6인, 인원별 10 Hands)

```powershell
uv run --extra gpu python -m scripts.expand_phase02_smoke `
  --hands-per-n 10 `
  --seed 180201 `
  --output-json runs/expand_phase02/smoke_10.json
```

### 5.3 Acceptance (2~6인, 인원별 1,000 Hands)

```powershell
uv run --extra gpu python -m scripts.expand_phase02_smoke `
  --hands-per-n 1000 `
  --seed 180301 `
  --output-json runs/expand_phase02/smoke_1000.json
```

$$N_{\mathrm{target}}=\sum_{n=2}^6 1,000=5,000\text{ hands},$$

신규 참가자가 BB를 기다리는 준비 Hand는 별도 `entry_setup_hands`로 집계하기 때문에 총 Hand 수는 5,000보다 많을 수 있다.

> **출력 경로는 기존 파일을 덮어쓰지 않는다(`open(...,'x')`).** 재실행하려면 다른 파일명을 선택한다.

### 5.4 API 예제

```python
from holdem_ai.expand.environment import DynamicTable
from holdem_ai.expand.information import Phase2TableAdapter, encode_observation
from holdem_ai.expand.information.policy_v2 import PolicyV2

table = DynamicTable(session_id='cash-sixmax')
env = Phase2TableAdapter(table)
for i in range(6):
    table.request_join(f'player-{i}',10000,preferred_seat=i)

hand = env.start_hand(seed=123)
model = PolicyV2(width=64, heads=4, layers=2)
while table.current_player_id is not None:
    info = env.observe()                  # ONLY Acting Player's InfoState
    encoded = encode_observation(info)   # Deterministic observation vectors
    move = model.select_action(encoded, rng_seed=123 + len(info.action_history))
    env.act_abstract(move)                # Legality-preserving table action
settlement = env.settle_hand()
```

> 위 예제는 **무작위로 초기화된 모델**을 시연하므로 학습 성과를 의미하지 않는다. 6명이 처음부터 동시에 참가하지 않을 수 있으며, Phase 1 `wait_for_big_blind` 규칙이 적용된다.

---

## 6. 후속 Phase 계약

- **Expand Phase 3:** 이 `EncodedObservation`, `PolicyV2` 및 `Phase2TableAdapter`를 불변 입력 계약으로 삼아 Multi-Agent PPO Collector와 Optimizer를 구현한다. 6인 Self-Play 보상은 Terminal Hand Result에서 `bb` 단위로 계산한다.
- **Expand Phase 4:** MCCFR 및 Abstraction은 신규 Track이며 Neural PPO를 Pluribus 원본으로 오인하지 않는다.
- **Expand Phase 5:** Depth-Limited Search는 별도 Subgame API를 사용한다.
- **Expand Phase 6:** Opponent ID와 장기 통계는 현재 InformationState에 없으며, Player ID/관측 보안 정책과 함께 별도 버전으로 추가한다.
- **Expand Phase 7:** Tournament-specific 관측 상태(Blinds/Ante, Payout/ICM, Registration) 확장 및 통합 Benchmark.

> [!note] 설계 상 추가 고려사항
> `InformationState v1`에 `table_status`와 `seat_status`의 두 단계 상태를 도입했다. Phase 3 전에 **Observation Schema Version을 Checkpoint Manifest에 명시**해야 하며, 이후 필드를 변경한다면 호환 변환 또는 새 모델 버전을 발급해야 한다.
