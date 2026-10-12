---
title: Expand Phase 5 — Bounded Belief-Conditioned Subgame Search
aliases:
  - Expand E5
  - Pluribus-style Depth-Limited Search Prototype
phase: Expand Phase 5
status: isolated-tests-passed / real-engine-Windows-pending
project: 6-Max No-Limit Holdem
updated: 2026-10-12
tags:
  - poker
  - six-max
  - mccfr
  - search
  - imperfect-information
  - reinforcement-learning
---

# Expand Phase 5 — Depth-Limited Subgame Search

> [!IMPORTANT] 구현 수준
> 이 패치는 **MCCFR Blueprint를 읽어 현재 Player의 관측 정보로 비공개 상태를 샘플링하는, 자원 제한이 있는 Search 실험 구현**이다. **2019년 Pluribus 원본의 Nested Re-solving 및 Leaf Continuation Subgame을 완전히 구현한 것이 아니다.** 특히 현재 결정 지점에서 탐색하고, 이전 Street 시작점 전체의 정보집합을 재계산하지 않는다. 4개 Continuation은 Leaf Rollout의 정책 후보로 샘플링하며, **각 Player가 Leaf Continuation을 전략적으로 선택하는 원본 게임을 해결하지 않는다**. 이 차이를 해결하는 후속 연구/Phase7 게이트를 아래에 기록한다.
>
> Phase 1~4 Windows 게임 엔진/CUDA 실기 검증은 아직 수행하지 않았다. 본 로컬 검증은 기존 격리 테스트용 가짜 엔진 및 Kuhn/Leduc를 사용한다. CPU Search이며 GPU 사용은 필요하지 않다.

## 1. 목표와 경계

| 목표 | Phase 5에서 구현 | 제한 |
|---|---|---|
| Root Public Belief | `ToyRootBelief`, `HoldemObservationBelief` | 상대 행동 우도는 선택적; 기본 균등 샘플 |
| Deck Collision Exclusion | Known Hero + Board 고정, 52장 완전 재배분 | 실제 2~6인 게임 엔진 연동 Windows 미검증 |
| Phase4 Blueprint | 원본 `InfosetStore`를 읽기 전용 사용 | 미학습 Infoset은 균등분포 |
| Depth-Limited Search | Multi-player External Sampling CFR (로컬 Regret) | Iteration/Leaf Monte Carlo 노이즈 |
| Leaf Continuations | Blueprint, Fold/Call/Raise biased; 5x 재가중 | 원본 Continuation Action Subgame 아님 |
| Nested Public Trace | 이전 공개 행동/본인 선택을 기록, 변경 차단 | Street Start부터 전 트리 재탐색하지 않음 |
| Off-Tree Raise | 실제 `raise_to` 값을 엔진에 Replay 가능 | Search의 미래 임의 Raise Action 자동 주입 없음 |
| Budget | 시간/노드/Infoset/추정 테이블 메모리 | OS 실제 RSS의 엄격한 상한 아님 |
| Fallback | 유효 확률 분포 및 Last-Complete Search 평균전략 | Online Search가 Blueprint보다 좋은 EV 보장 없음 |
| A/B | Kuhn/Leduc Paired-Deal CLI | 독립적인 6-Max 통계 증거 아님 |

### 1.1 연구 자료

- [Brown & Sandholm (2019), Science — Superhuman AI for multiplayer poker](https://noambrown.github.io/papers/19-Science-Superhuman.pdf)
- [Original Supplementary Materials](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf): Search, Nested Search Algorithm 2, continuation strategies.
- [Expand Master Design](EXPAND_MASTER_DESIGN_6MAX_v1_0.md): §E5 목표/계약.

## 2. 아키텍처 / 데이터 흐름

```text
Phase 4 blueprint.json.gz (read-only)
            |
   BlueprintPolicy(store, abstraction_id)
            |                              +--- toy public observation
            |                              |
            +------> SubgameSolver <--- RootBelief.sample(rng)
                         |                  |
                    Regret Table        +--- Hold'em raw_obs (hero hole + public)
                    per Search              |   start_stacks + public action trace
                         |                  +--- conditioned collision-free deck
                         +---- bounded MCCFR iteration
                         |
                         +---- depth leaf: four continuation families
                         |
                         +---- valid normalized legal-action distribution
                                | solved / timeout / resource_guard / fallback
```

Actor Search는 `HoldemObservationBelief`에 **본인의 `raw_obs`**만 제공한다. `engine._deck`, 상대 `engine.hole_cards`, 다음 Community Card 및 Actor가 보지 못하는 CriticState를 직접 사용하지 않는다.

### 2.1 `SearchBudget` (첫 기본값)

```python
SearchBudget(
    max_time_ms=500,
    max_nodes=2500,
    max_memory_mb=64,
    max_iterations=80,
    max_infosets=1000,
    depth=2,
    max_rollout_steps=200,
    blueprint_mix=0.15,
)
```

500 ms는 *목표 시간 예산*이야. Python 시스템 호출, 엔진 `deepcopy`, Root Rejection Sampling 및 OS 스케줄링으로 실제 경과시간이 엄격히 500ms 이내라고 보장하지 않는다. `max_memory_mb`는 1 infoset=4096바이트의 **보수적 추정 테이블 카운터** 기준이므로 실제 Process RSS 상한이 아니야. 하드 실시간/메모리 격리는 별도의 프로세스 제한이 필요하다.

## 3. 수학 — Belief-Conditioned Search

Player $j$의 실제 카드 후보 $h_j$를 추정하는 일반적인 베이지안 계산은

$$
 b_{j,t}(h_j) \propto b_{j,0}(h_j)
 \prod_{k\le t:\operatorname{actor}(k)=j}
 \hat\pi_j(a_k\mid h_j,I_{j,k}).
$$

그러나 여러 Player의 카드는 독립적이지 않다.

$$
 \mathcal H_{\mathrm{valid}}
 =\{(h_0,\ldots,h_5):h_i\cap h_j=\varnothing,\;
 h_i\cap(B\cup H_{hero})=\varnothing\}.
$$

`HoldemObservationBelief`는 $H_{hero}$, 공개 보드 $B$, 핸드 시작 스택 및 이전 공개 행동들을 고정한 채 완전한 52장 덱을 **카드 중복 없이** 생성하고, 엔진으로 액션을 Replay해 해당 정보집합에 도달하는지 검증한다.

- **기본값**: 후보 덱에 대한 공개 행동의 likelihood 가중치 없이 샘플링한다(따라서 posterior의 정확한 Bayesian 갱신이 아님).
- **선택적 실험**: `opponent_likelihood(observation, action_id, raise_to)`를 전달하면 가상의 **해당 상대 관측**에서 행동 확률을 평가한 뒤 rejection-sampling으로 가중한다. 0에 가까운 likelihood에서는 재생성 시도 초과가 발생할 수 있으며, 6인 상대의 exact joint posterior라고 주장하지 않는다.
- **차단 조건**: 내 Hole Cards와 공개 카드 중복, `button`/`player_id` 이상, Public Trace와 Engine 결과 불일치 => 예외/재샘플.

### 3.1 Phase 3 덱 인덱스

현재 Base Phase 3 엔진의 카드는 Button 다음 점유 Seat부터 한 장씩 두 번, 이후 Burn/Flop, Burn/Turn, Burn/River 순서야. $n$명이 참여할 때

$$
\begin{aligned}
&\text{Hole round-1}: [0,n-1],\quad \text{Hole round-2}: [n,2n-1],\\
&\text{Burn Flop}:2n,\quad \text{Flop}:\{2n+1,2n+2,2n+3\},\\
&\text{Burn Turn}:2n+4,\quad \text{Turn}:2n+5,\\
&\text{Burn River}:2n+6,\quad \text{River}:2n+7.
\end{aligned}
$$

이는 원본 엔진의 Deal Order 계약에 종속되며, 향후 엔진 구현이 달라지면 동기화가 필요하다.

### 3.2 Regret / Blueprint 혼합

Phase4와 별도로 생성하는 일시적 Subgame Regret $R^{search}$에서

$$
\sigma^{local}(I,a)=\operatorname{RegretMatch}(R^{search}(I,a)).
$$

탐색 시 Blueprint prior를 일부 혼합한다.

$$
\sigma^{search}(I,a)=(1-\alpha)\sigma^{local}(I,a)
  +\alpha\,\pi_{blueprint}(I,a),\quad \alpha=0.15.
$$

Traverser의 Action은 전부 전개하고, 나머지 Player의 Action은 현재 전략에서 Sample한다. 정보집합 키는 Phase4 `HoldemState.infoset`을 이용한다. Iteration이 정상 완료된 뒤만 Regret/Average Table을 Commit한다. Time/Node Budget가 중단한 Iteration의 결과는 사용하지 않는다.

> [!WARNING] 이 식과 Local CFR은 **정확한 Root Counterfactual Value를 보장하지 않는다**. 특히 Leaf Rollout의 Monte Carlo 분산, Phase4의 Hash Bucket, 사전 Blueprint 혼합, 편향 없는 확률 추정기 검증 부족 등을 포함한다. 목표는 올바른 인터페이스/비공개정보 차단 및 탐색 예산 검증이다.

## 4. Leaf Continuations

$$
\mathcal C=\{\pi_B,\pi_F,\pi_C,\pi_R\}.
$$

$$
\pi_v(a\mid I)=\frac{w_v(a)\pi_B(a\mid I)}{\sum_{a'\in A(I)}w_v(a')\pi_B(a'\mid I)},\quad
w_v(a)=\begin{cases}5,&a\in\operatorname{target}(v),\\1,&\text{otherwise}.
\end{cases}
$$

- `fold_bias`: Fold만 가중치 5.
- `call_bias`: Call/Check를 가중치 5.
- `raise_bias`: Bet, Raise Half/Pot, All-In 등 Raise 유형을 **모두** 가중치 5.
- `blueprint`: 원 전략 분포.
- 합법 Action만 정규화하고 전체 확률 합이 1인지 검증한다.

동일한 탐색 Iteration/Branch에서 $(player,infoset)$이 동일하면 동일 `continuation variant`를 쓰도록 Dictionary로 묶었다. **원본 Pluribus는 이 Continuation Choice를 Leaf Subgame의 전략 행동으로 사용한다. 현재 구현은 이를 Sample해서 Rollout하는 근사**이므로 전략적 선택의 가치가 모델링되지 않는다.

## 5. Nested Session / Off-Tree 범위

- `NestedSearchSession.observe_action(...)`는 `(street,infoset,action)`으로 **내 실제 행동의 불변 기록**을 유지한다.
- 실제 `raise_to`는 Abstract Raise 크기에 없더라도 Trace에 기록된다.
- `HoldemObservationBelief`는 `action_id=5`를 `engine.step_raise_to(amount)`로 정확히 재생할 수 있다.
- Street 변경 시 Nested Session 카운터를 증가시킨다.
- **별개의 후속 작업**: 실제 Betting Round 시작 지점부터 public tree를 재구성하여 상대의 이전 정책을 재탐색하면서 내 과거 행동만 강제 고정하는 *original-style nested re-solving*, 미래 `raise_to`의 search action injection, 원본 Continuation Selection 게임. 현재 제공한 인터페이스만으로 원본 구현이 완성되었다고 볼 수 없다.

## 6. File Layout (추가형 패치)

```text
src/holdem_ai/expand/pluribus/search/
  __init__.py
  belief.py            # toy / actual Phase3 raw_obs public-consistent sampler
  continuations.py     # four leaf continuation policies
  solver.py            # SearchBudget, SearchResult, Phase4 Blueprint adapter
  nested.py            # immutable own public action trace & off-tree recording
scripts/
  expand_phase05_search.py      # current toy/holdem action distribution
  expand_phase05_evaluate.py    # toy paired blueprint vs search benchmark
tests/expand/
  test_e05_beliefs.py
  test_e05_cli.py
  test_e05_continuations.py
  test_e05_evaluate.py
  test_e05_holdem_deck.py
  test_e05_nested.py
  test_e05_real_engine.py
  test_e05_solver.py
```

**Base Phase1~16, Expand Phase1~4 코드는 수정하지 않음.**

## 7. 설치 — Windows PowerShell

`holdem-ai` 프로젝트 루트에서 Expand Phase1~4를 먼저 설치한 다음:

```powershell
Expand-Archive .\holdem-ai-expand-phase05.zip `
  -DestinationPath .\holdem-ai-expand-phase05 -Force

Copy-Item .\holdem-ai-expand-phase05\src\holdem_ai\expand\pluribus\search `
  .\src\holdem_ai\expand\pluribus\ -Recurse -Force

Copy-Item .\holdem-ai-expand-phase05\scripts\expand_phase05_*.py `
  .\scripts\ -Force

Copy-Item .\holdem-ai-expand-phase05\tests\expand\test_e05_*.py `
  .\tests\expand\ -Force

Copy-Item .\holdem-ai-expand-phase05\docs\EXPAND_PHASE05_SUBGAME_SEARCH.md `
  .\docs\ -Force
```

모든 Expanded Phase를 귀가 후 설치하는 경우, **01 → 02 → 03 → 04 → 05** 순서로 복사한다.

### 7.1 검증 게이트

```powershell
uv run --extra dev --extra gpu python -m pytest -q -k e05
uv run --extra dev --extra gpu python -m pytest -q tests/expand
uv run --extra dev --extra gpu python -m pytest -q
```

`test_e05_real_engine.py`는 Phase3 진짜 엔진이 있으면 Skip되지 않아야 한다. Windows의 이 테스트는 현재 미수행이며 **실패가 나올 경우 다음 Phase에서 넘어가기 전에 원인을 조사**해야 한다.

### 7.2 단일 Search 파일럿 (CPU)

```powershell
uv run python -m scripts.expand_phase05_search `
  --game kuhn --players 2 `
  --iterations 40 --time-ms 5000 --nodes 5000 --depth 2 `
  --output-json runs/expand_phase05/kuhn_search_40.json
```

Phase4에서 학습한 Blueprint를 이용한 Search:

```powershell
uv run python -m scripts.expand_phase05_search `
  --game kuhn --players 2 `
  --checkpoint runs/expand_phase04_kuhn_main/blueprint.json.gz `
  --iterations 100 --time-ms 1500 --nodes 5000 --depth 2 `
  --output-json runs/expand_phase05/kuhn_blueprint_search.json
```

실제 홀덤 엔진 6인용 Smoke (2BB 극소 Stack, `resource_guard`/`timeout` 가능):

```powershell
uv run python -m scripts.expand_phase05_search `
  --game holdem --players 6 --stack-bb 2 `
  --iterations 10 --time-ms 500 --nodes 500 --depth 1 `
  --output-json runs/expand_phase05/holdem6_smoke.json
```

이 명령은 본격적인 100BB 전략 훈련·성능 검증이 아니라 **엔진 연결 및 합법 행동확률의 파일럿**이야.

### 7.3 A/B Toy Poker 평가

```powershell
uv run python -m scripts.expand_phase05_evaluate `
  --game kuhn `
  --checkpoint runs/expand_phase04_kuhn_main/blueprint.json.gz `
  --hands 100 --seed 550005 --iterations 20 --time-ms 500 `
  --output-json runs/expand_phase05/kuhn_ab_100.json
```

동일 Deal과 Seat 배치를 Blueprint/검색에 반복하지만, 분기 뒤 액션이 달라질 수 있으므로 완벽한 Path Pairing은 아니다. 파일의 Normal CI는 *탐색 목적 근사치*이며 같은 Deal·좌석 사이 상관성을 완전히 반영하지 않으므로 엄밀한 유의성 추론에 사용하지 않는다. 6-Max Exploitability 보고가 아니다.

> [!NOTE] Phase 4의 `blueprint.json.gz`는 수정하지 않는다. Phase 5 실행 결과는 JSON만 생성하며 기존 JSON 파일 덮어쓰기를 방지한다.

## 8. 확인된 로컬 실험 기록

- 격리 환경: Expand Phase1~5에서 **최종 회귀 테스트 결과를 별도로 확인 후 갱신**.
- Kuhn 2인 새 Blueprint 없이 40 iteration 시험: `status=solved`, `nodes=180`, `elapsed=12.78ms`, `P(check)=0.609318`, `P(bet)=0.390682` (실행 환경의 임시 관측; Python/Seed/시스템에 따라 지연시간은 달라짐).
- 별도 2인 Kuhn Phase4 저장 Blueprint 대상 16게임 샘플: Search-Blueprint 평균 보상 차이는 `-0.125` 칩/핸드; 정규근사 95% CI `[-0.2924,+0.0424]` (아주 작은 Toy 샘플). **개선이 입증되었다는 의미가 아니다.**

## 9. Phase 5 수용 기준 및 미해결 Gate

- [x] 2~6인 Public-Consistent 카드 배분 시뮬레이션, Known Card 중복 방지
- [x] Kuhn/Leduc 자신의 카드만 고정해 Root를 재생성
- [x] Blueprint Read-Only, 유효 Legal Action 확률과 Same Seed 재현성
- [x] Four Continuations 5x Reweighting, Infoset-consistent 샘플링
- [x] Time/Node/Memory *추정치* Resource Guard/Fallback
- [x] Public Action Trace의 Off-Tree `raise_to` 기록, 내 확정 행동 변경 차단
- [ ] **Windows 실제 Phase3 Engine** 카드/베팅/Side Pot 및 `raise_to` Replay 종단 간 검증
- [ ] **원본 Pluribus Nested Search**: Street Start부터 Re-solving 및 Off-Tree Future Action Injection
- [ ] **원본 Continuation Leaf Game**: 각 Player의 Continuation Choice 최적화
- [ ] 6-Max 대규모 Blueprint/Search 성능 및 통계 유의성 확인

### 9.1 Phase 6과 연결점

Phase 6 Opponent Modeling에서는 `opponent_likelihood(...)`가 학습된 상대 모델의 후방 likelihood를 제공하도록 확장한다. Phase 7 통합 게이트 전에 카드 충돌 없는 Joint Belief, 탐색 품질 및 Best Response 성능을 검증한다.
