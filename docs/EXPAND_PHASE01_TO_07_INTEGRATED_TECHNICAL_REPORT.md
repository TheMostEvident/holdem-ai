---
title: "Dynamic 6-Max NLHE AI — Expand Phase 1~7 통합 기술 보고서"
aliases:
  - "Expanded 6-Max Technical Report"
  - "Expand Phase 1-7 Integration"
tags:
  - poker-ai/6max
  - reinforcement-learning
  - mccfr
  - subgame-search
  - opponent-modeling
  - architecture
created: 2026-10-12
version: "1.0-unified-prototype"
status: "Integrated package built and isolated tests passed; Windows/real-engine/CUDA acceptance pending"
repository: "https://github.com/TheMostEvident/holdem-ai"
---

# Expand Phase 1~7 — Dynamic 6-Max No-Limit Hold'em AI 통합 기술 보고서

> [!abstract] 목적 및 판정
> 7개의 독립적인 추가형 Expand 패치를 **하나의 소스 트리**로 병합하여 설치·검증 절차를 단일화했다. 이 보고서는 실제 병합된 Python 코드, 각 Phase의 기술 문서와 로컬 테스트에 기반한다. **Base Phase 1~16 수정 없음. 실제 Windows 엔진/CUDA 실행 및 GitHub Push는 아직 수행하지 않았다.** 특히 E4/E5는 **Pluribus 완전 재현이 아닌 연구용 프로토타입**이다.

## 0. 문서의 기준선과 검증 범위

| 항목 | 내용 / 판정 |
|---|---|
| Base Repository | `TheMostEvident/holdem-ai` (별도 설치 필요) |
| 기준 모델 | Base Phase 1~16 HUNL 시스템과 독립된 Expand 기능 |
| 통합 대상 | `holdem-ai-expand-phase01.zip` ~ `phase07.zip` |
| 설치 모델 | **단일 Additive Overlay**: `src/`, `scripts/`, `tests/`, `docs/` |
| 충돌 처리 | `README.md` 7개만 경로 충돌 → 각각 `docs/phase_original_readmes/`에 보존 |
| 나머지 소스 | **서로 다른 경로**, 변경 없이 바이너리 동일하게 병합 |
| 공급망 검증 | 모든 병합 원본의 SHA-256, 원본 ZIP SHA-256을 `SOURCE_MANIFEST.json`에 수록 |
| 이 통합본의 격리 환경 테스트 | **164 passed, 7 skipped**; 실제 엔진 없는 환경에서 통합 재실행 |
| 미실시 | 실제 Base Phase 3 엔진 통합, 전체 Base Regression, GTX 1070 CUDA, 성능 실험, Git Push |
| 훈련된 6-Max PPO 모델 | **통합 ZIP에는 포함되지 않음**. Phase 3 학습 필요 |
| Pluribus 원본 학습 Blueprint | **제공되지 않음**. Phase 4에서 새로 연구용 학습 필요 |

> [!warning] "통합"의 정확한 의미
> 하나의 ZIP 안에 E1~E7의 소스와 CLI, 테스트를 합쳤다는 뜻이다. **E4 정보집합-이력 어댑터 및 E5 Search가 실전 6인 AI 정책까지 자동으로 연결되었다는 의미가 아니다.** E7은 검증된 E3 Neural Policy를 우선 사용할 수 있으며 E4/E5 사용은 별도 호환성 검증 후 명시적 연결이 필요하다.

## 1. 프로젝트 목표와 범위

최종 목표는 플레이어 입·퇴장, 유효 스택, 포지션, 상대 행동이 변하는 **동적 2~6인 No-Limit Hold'em** 환경에서 합법적인 행동을 결정하는 정책이다.

$$
\boxed{\pi_\theta(a_t\mid I_{i,t},\mathcal N_t)},\qquad 2\le |\mathcal N_t|\le 6
$$

- **Cash Game 먼저**: 핸드 사이에 Join/Leave/Rebuy/Sit-Out/Resume, 스택 이월, 손익을 Big Blind 단위로 측정한다.
- **Tournament 별도**: 블라인드 상승, Late Registration, Re-entry, 탈락, 순위·상금 보상을 독립적으로 처리한다.
- **두 학습/의사결정 트랙**: E3 Neural PPO와 E4 MCCFR Blueprint + E5 Search는 별도 출처의 전략이다.
- **상대 적응**: E6는 공개 행동만으로 상대 모델을 구성하고, 사전에 존재하는 후보 정책만 제한적으로 선택·보간한다.

이 버전은 연구용 프로토타입이며 실제 돈을 사용하는 서비스에 바로 연결하기 위한 제품이 아니다.

## 2. 아키텍처 및 모듈 의존성

```text
Base Phase 3: MultiPlayerHandEngine (게임 규칙·카드·Side Pot·정산)
                        |
          E1 DynamicTable / TableState
             (물리적 6좌석, Join/Leave)
                        |
          E2 InformationState / Encoder
             (Hero Cards + Public State)
                        |
                 E7 Policy Router
           +------------+--------------+
           |            |              |
       E3 Neural     E4 MCCFR       E5 Search
          PPO        Blueprint     (opt-in / bounded)
           |            |              |
           +------------+--------------+
                        |
                 E6 Optional Guard
              (Evidence, Expert, KL/TV)
                        |
               E2 ActionTranslator
                        |
          Base Phase 3 Legality / Engine
                        |
       E7 Cash / Tournament Benchmark
                        |
                 JSON + Manifest
```

### 2.1 디렉터리 체계

```text
holdem-ai/                                # 기존 Git Repository
  src/holdem_ai/expand/
    environment/                         # E1: DynamicTable, EventLedger
    information/                         # E2: Public State, 7 Action, Encoder, PolicyV2
    rl/                                  # E3: Collector, ActorCritic, PPO, Trainer
    pluribus/
      abstraction/                       # E4: Card Bucketing Prototype
      blueprint/                         # E4: MCCFR / Kuhn / Leduc / Holdem
      search/                            # E5: Belief / Continuation / Solver / Nested Session
    opponent/                            # E6: Model, Bridge, Router, Adaptation
    integration/                         # E7: IntegratedPokerAgent, Tournament, Benchmark
  scripts/expand_phase01_smoke.py
  scripts/expand_phase02_smoke.py
  scripts/expand_phase03_train.py
  scripts/expand_phase03_evaluate.py
  scripts/expand_phase04_mccfr.py
  scripts/expand_phase04_evaluate.py
  scripts/expand_phase05_search.py
  scripts/expand_phase05_evaluate.py
  scripts/expand_phase06_benchmark.py
  scripts/expand_phase07_benchmark.py
  tests/expand/                           # E1~E7 테스트 전체
  docs/EXPAND_MASTER_DESIGN_6MAX_v1_0.md
  docs/EXPAND_PHASE01_DESIGN.md
  docs/EXPAND_PHASE02_INFORMATION_AND_ENCODING.md
  docs/EXPAND_PHASE03_MULTIAGENT_PPO.md
  docs/EXPAND_PHASE04_MCCFR_BLUEPRINT.md
  docs/EXPAND_PHASE05_SUBGAME_SEARCH.md
  docs/EXPAND_PHASE06_OPPONENT_MODELING.md
  docs/EXPAND_PHASE07_INTEGRATION_TOURNAMENT_BENCHMARK.md
  docs/EXPAND_PHASE01_TO_07_INTEGRATED_TECHNICAL_REPORT.md
```

> 모든 파일은 Base 코드 위에 **추가**하는 Overlay다. 기존 `src/holdem_ai/environment/*.py`, Base `holdem_agent_phase*`, `scripts/phase*`, 학습 결과 `runs/`는 ZIP에 없다.

## 3. Expand Phase 1 — Dynamic 6-Max Table Environment

**실제 소스**: `environment/table_manager.py`, `table_state.py`, `event_ledger.py`, `cash_rules.py`, `multiway_adapter.py`, `tournament_rules.py`.

- **6개 물리적 Seat**(`seat=0..5`)를 유지하면서 각 좌석의 `player_id`, `stack`, `status`를 관리한다.
- Join, Leave, Rebuy, Sit-Out, Resume 요청은 이벤트로 큐에 기록하고 **Hand Boundary에서만 적용**한다.
- 플레이어가 All-In하여 현재 스택이 0이 되어도 **핸드 진행 중에 Bust 처리하지 않는다**. Side Pot과 Showdown 정산 후 확정한다.
- 첫 참가자의 BB 등록 대기 상태 및 Moving Button 모델이 포함되며, 공식 모든 Casino Missed-Blind/Dead-Button 정책을 인증한 것은 아니다.
- 같은 플레이어 ID 중복, 예약 좌석 중복, 음수 칩, 핸드 중 이벤트 적용, 인원 부족 등의 상태를 검사한다.
- 이벤트 기록 `EventLedger`로 운영 이력 기록 및 결정적 Replay를 지원한다.

### 3.1 현금 보존 불변식

핸드 진행 중 각 Seat의 남은 스택 $S_i$와 기여금 $c_i$의 합은 변하지 않아야 한다.

$$
\sum_{i=0}^{5}(S_{i,t}+c_{i,t})=C_h, \quad
\sum_{i=0}^{5} S_{i,h}^{\mathrm{after}}=C_h
$$

핸드 사이에 신규 Buy-In $B_t$와 Cash-Out $O_t$가 있으면 테이블 전체 칩의 변화는

$$
\boxed{C_{t+1}=C_t+B_t-O_t},\qquad \mathrm{rake}=0.
$$

구현된 `DynamicTable`은 다음을 확인한다.

$$
\mathrm{LiveChips} = \mathrm{TotalBuyins}-\mathrm{TotalCashouts}.
$$

### 3.2 오픈 API

```python
from holdem_ai.expand.environment.table_manager import DynamicTable

table = DynamicTable(session_id="table-001")
table.request_join("hero", buyin_chips=20000, preferred_seat=0)
table.request_join("opponent", buyin_chips=20000, preferred_seat=3)
# table.apply_boundary_events() -> table.start_hand(...) ->
# table.act(...) -> table.settle_hand()
```

칩 값은 **정수**다. 금액의 의미는 기존 Base Engine과 동일하다. `raise_to`는 **현재 Street의 목표 누적 베팅액**이지 증분 베팅액이 아니다.

## 4. Expand Phase 2 — Information State & Action Encoding

**실제 소스**: `information/schema.py`, `builder.py`, `actions.py`, `encoder.py`, `policy_v2.py`, `adapter.py`.

### 4.1 Actor 정보경계

$$
I_{i,t}=\big(H_i,B_t,P_t,\mathbf S_t,\mathbf M_t,\mathrm{Position}_i,\mathrm{History}_t,\mathcal A(I_{i,t})\big)
$$

`InformationState`는 **행동 중인 본인의 Hole Cards**, Board, 현재 Pot, 6좌석 공개 상태, Betting History, Legal Action만 저장한다. 상대 Hole Cards·미래 Deck·Critic 전용 상태는 포함하지 않는다.

`TrainingOnlyCriticState`는 별도의 타입으로 분리되어 Actor 입력으로 전달하지 않는다. 이는 정보 누출을 방지하기 위한 인터페이스 제약이며 실제 엔진 연동 시의 비간섭 테스트는 남아 있다.

### 4.2 7개 Action 계약

실제 액션 이름의 순서는 다음과 같다.

```python
ACTION_NAMES = (
    'fold', 'check', 'call',
    'raise_half_pot', 'raise_pot', 'raise_double_pot',
    'all_in',
)
```

- `LegalActionSpec.mask`는 정확히 **7개의 bool**이다.
- 불가능한 행동의 확률은 0으로 마스킹한다.
- 짧은 Stack으로 Call하여 전체 Stack이 투입되는 경우에도 행동은 **Call**로 표현한다. All-In Raise와 중복하지 않는다.
- Raise-to 금액은 Engine에서 검증 가능한 **정수 칩 목표액**으로 전달한다.

$$
\pi_\theta(a\mid I)
=\frac{\exp z_\theta(I,a)\,\mathbf1[a\in\mathcal A(I)]}{\sum_b\exp z_\theta(I,b)\,\mathbf1[b\in\mathcal A(I)]}
$$

### 4.3 가변 참가자 인코딩

$$
X_{\mathrm{seat}}\in\mathbb R^{6\times19},\quad
X_{\mathrm{history}}\in\mathbb R^{128\times21}.
$$

`occupied`(좌석 점유)와 `dealt_in`(현재 핸드 참가)는 별개다. Fold, All-In, Sit-Out, Waiting BB 및 Vacancy를 동일 상태로 합쳐서는 안 된다. `PolicyV2`는 Seat Token과 Action History를 처리하는 Neural Actor의 초기 구조이며, **네트워크 생성 ≠ 학습된 정책 보유**이다.

## 5. Expand Phase 3 — Multi-Agent PPO Self-Play

**실제 소스**: `rl/model.py`, `collector.py`, `population.py`, `returns.py`, `ppo.py`, `trainer.py`.

### 5.1 Rollout 및 보상

- 한 핸드에서 여러 Seat의 행동이 섞이더라도 **(Hand, Seat)** 단위로 Trajectory를 보존한다.
- Learner는 하나의 Neural Policy를 공유하며 상대는 scripted 또는 **이전 PPO Snapshot**이 될 수 있다.
- 검증에는 학습과 겹치지 않는 Card Seed를 사용한다.
- 모델 갱신은 Rollout 후에 수행하며 Checkpoint는 Actor, Critic, Optimizer, Config 및 RNG 복원 자료를 기록한다.

$$
R_{i,h}^{\rm chips}=S_{i,h}^{\rm after}-S_{i,h}^{\rm before},\qquad
R_{i,h}^{\rm bb}=\frac{R_{i,h}^{\rm chips}}{BB_h}.
$$

### 5.2 GAE / PPO

$$
\delta_t^{(i)}=r_t^{(i)}+\gamma V_\phi(o_{t+1}^{(i)})-V_\phi(o_t^{(i)})
$$

$$
\hat A_t^{(i)}=\delta_t^{(i)}+\gamma\lambda(1-d_t^{(i)})\hat A_{t+1}^{(i)}
$$

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid I_t)}{\pi_{\theta_{old}}(a_t\mid I_t)},\quad
L_{\rm policy}=-\mathbb E\left[\min\left(\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right]
$$

$$
L_{\rm total}=L_{\rm policy}+c_V L_{\rm value}-c_H H(\pi_\theta).
$$

**핵심 주의**: Expand E3의 `expand-e03-v1` 체크포인트는 Base HUNL Phase 15 또는 Base Phase 14 `.pt` 체크포인트와 **호환되지 않는다**. E7은 학습된 E3 모델 체크포인트가 지정되었을 때만 이를 Neural Agent로 불러온다.

## 6. Expand Phase 4 — Pluribus-style MCCFR Blueprint

**실제 소스**: `pluribus/blueprint/mccfr.py`, `toy_games.py`, `kuhn_reference.py`, `holdem.py`, `checkpoint.py`, `evaluate.py`, `pluribus/abstraction/cards.py`.

`MCCFRSolver`는 External Sampling으로 상대·Chance 행동을 샘플링하고 Traverser의 행동만 분기하며, Linear Weighting으로 Regret/평균전략을 관리한다.

$$
R_i^{T}(I,a)=\sum_{t=1}^{T} w_t\big[\hat v_i^t(I,a)-\hat v_i^t(I)\big]
$$

$$
\sigma_i^{t+1}(I,a)=
\begin{cases}
\dfrac{[R_i^t(I,a)]_+}{\sum_b[R_i^t(I,b)]_+}, & \sum_b[R_i^t(I,b)]_+>0\\
\dfrac{1}{|\mathcal A(I)|}, & \text{otherwise}
\end{cases}
$$

`InfosetStore`는 Regret/Strategy Array를 저장하며 예산 초과 시 iteration의 부분 상태 저장을 피하도록 설계되었다.

**정확성 검증**: 2인 Kuhn Poker의 정확한 Best Response 기반 NashConv 검증 테스트를 포함한다. 기존 Phase 4 개발 기록에서 10,000 Iterations 후 약 `0.013`의 NashConv를 관측했다. 이는 **Kuhn 전용 결과**이며 6-Max 게임의 Exploitability 값이 아니다. 이번 통합 작업은 그 코드를 병합하여 테스트했지만 긴 원본 실험을 다시 학습한 것은 아니다.

### 6.1 원본 Pluribus와 다른 점

- 원본 학습된 100BB 6-Max Blueprint 미보유.
- Card Bucket은 **프로토타입 해시 기반 Abstraction**이며, 원본의 Equity-based Bucketing과 다르다.
- 원본의 Negative-Regret Pruning은 미구현.
- 큰 6-Max Tree에서 Memory/Node Budget 내 수렴 및 전략 강도 미검증.
- **다인 게임의 Nash 수렴을 HUNL과 동일하게 보장하지 않는다.**

## 7. Expand Phase 5 — Depth-Limited Subgame Search

**실제 소스**: `pluribus/search/belief.py`, `solver.py`, `continuations.py`, `nested.py`.

### 7.1 Belief와 불완전정보

현재 Hero가 알고 있는 카드와 Board는 고정한 채 가능한 상대의 Private Cards를 샘플링한다. 카드 중복은 허용되지 않는다.

$$
P(H_{-i}\mid I_i,\,a_{1:t})\propto P(a_{1:t}\mid H_{-i},I_i)P(H_{-i}\mid I_i)
$$

관측된 상대 행동으로부터 정확한 범위를 복원한다는 의미는 아니다. 정확한 행동 우도나 공동 카드 분포를 추정하는 고급 모델은 후속 작업이다.

### 7.2 Bounded Search

`SearchBudget`은 `max_time_ms`, `max_nodes`, `max_memory_mb`(**추정량**), `max_iterations`, `max_infosets`, `depth`를 가진다. 예산을 넘거나 불완전 탐색 결과이면 마지막 유효 전략/Blueprint로 Fallback한다.

$$
\pi_{\rm output}(a\mid I)=
(1-\alpha)\pi_{\rm local}(a\mid I)+\alpha\pi_{\rm blueprint}(a\mid I),\quad \alpha=0.15\;(\text{기본})
$$

Leaf Continuation에는 `blueprint / fold-biased / call-biased / raise-biased` 프로토타입을 제공한다.

> [!warning] 원본 재현 범위
> **Phase 5는 현재 의사결정에서 다시 탐색하는 근사 프로토타입**이다. 원본 Pluribus의 Betting-Round 시작점부터 완전한 Nested Re-solving과 정확한 Leaf Continuation Game을 구현한 상태가 아니다. 초기 Toy Search 비교에서 Blueprint보다 성능이 좋은 증거도 아직 없다.

## 8. Expand Phase 6 — Opponent Modeling & Adaptive Strategy

**실제 소스**: `opponent/schema.py`, `model.py`, `bridge.py`, `router.py`, `adaptive.py`, `persistence.py`.

**식별자 구조**: `(Player ID, Join Tenure ID)`를 키로 사용한다. 좌석 이동이나 신규 참가자가 이전 플레이어의 행동 통계를 가져서는 안 된다.

### 8.1 Public Action Model

상대 행동은 Street·Position·SPR·**상대 본인의 합법 행동 집합**에 조건화하여 카운트를 수집한다.

$$
\widehat P_j(a\mid c)=\frac{N_{j,c,a}+\alpha}{\sum_{b\in\mathcal A(c)}N_{j,c,b}+\alpha|\mathcal A(c)|}
$$

Phase 1/2 Bridge는 **엔진이 행동을 성공적으로 받아들인 뒤** 공개 정보를 기록한다. 성공하지 않은 Bet 명령을 상대 행동 관측으로 기록하지 않는다.

### 8.2 안전한 정책 적응

상대 스타일 분류 및 `StyleExpertRouter`는 이미 존재하는 Hero 후보 정책을 선택할 수 있지만, **Counter-Strategy를 직접 새로 학습하지 않는다**. 사전 모델의 행동 적응은 기본 전략을 제한된 범위로 혼합한다.

$$
\pi_{\rm adapt}(a)=(1-\eta)\pi_{\rm base}(a)+\eta\pi_{\rm proposal}(a)
$$

$$
\boxed{D_{KL}(\pi_{\rm adapt}\Vert\pi_{\rm base})\le0.025},\qquad
D_{TV}(\pi_{\rm adapt},\pi_{\rm base})\le0.075.
$$

이 값은 기본 `GuardConfig`의 제안 정책 변경 한계로, **GTO 보장이나 실측 수익률 개선의 증거가 아니다**. 변화 탐지나 표본 부족 시 기본 전략을 유지한다.

**격리 합성 실험**: Tight→Aggressive 전략 변화 시나리오 16개 블록에서 변화 신호 검출을 확인했다. 이는 공개 행동 예측 평가이지 실제 Poker EV 성능이 아니다.

## 9. Expand Phase 7 — Integrated Agent, Tournament & Benchmark

**실제 소스**: `integration/agent.py`, `source_bridges.py`, `tournament.py`, `benchmark.py`, `manifest.py`.

### 9.1 정책 라우팅

```text
Hero-visible InformationState
  -> E5 Search       (검증된 root belief 공급된 경우에만)
  -> E4 Blueprint    (검증된 exact infoset-key 공급된 경우에만)
  -> E3 Neural       (신뢰 가능한 E3 checkpoint 지정)
  -> Uniform Legal   (긴급 Fallback, 학습된 AI 아님)
  -> E6 Optional Adaptive Guard (기존 유효 후보 정책만)
  -> E2 Legal Action Translator
```

$$
\pi_{\rm routed}=\operatorname{FirstVerifiedValid}\{\pi_{E5},\pi_{E4},\pi_{E3},\pi_{U}\}.
$$

E2의 일반적인 `PublicAction`은 E4가 요구하는 Base Engine의 **원본 행동 ID와 정확한 Raise-to 이력**을 반드시 보존하지 않는다. 따라서 E2→E4 exact Infoset Key 변환은 아직 완전히 검증되지 않았고, **E7 기본 CLI는 E4/E5를 자동으로 활성화하지 않는다**.

### 9.2 Tournament 확장

`SingleTableTournament`는 Base Hand Engine을 재사용하고 **Hand Boundary**에서 등록·재참가·탈락을 처리한다.

| 항목 | 지원 여부 / 주의 |
|---|---|
| 2~6인 단일 테이블 | 지원 |
| Blind Levels / 상승 | 지원 |
| Late Registration | 지원, 제한된 등록 기간 |
| Re-entry | 지원, 새로운 Entry ID 부여 |
| Side Pot / All-In 정산 | Base Engine 재사용 |
| 종합 칩 EV와 상금 EV 분리 | 지원 |
| Ante | **미지원**; 설정 시 명시적인 오류 |
| 여러 테이블 / 테이블 병합 | 미지원 |
| 공식 Dead Button 및 Missed Blind 전체 인증 | 미수행 |
| ICM 기반 의사결정 | 미구현 |
| 리바이 비용을 포함한 Player-Level 순손익 | 미구현; Entry별 상금 기록만 |

Cash Reward와 Tournament Reward는 목적이 다르다.

$$
R_i^{\rm cash}=\frac{S_{i,\mathrm{after}}-S_{i,\mathrm{before}}}{BB_h},\quad
R_i^{\rm tournament}=p_{\operatorname{rank}(i)}.
$$

### 9.3 Scientific Benchmark

Cash Benchmark는 인원 $N\in\{2,3,4,5,6\}$, 초기 스택 BB, Opponent Type의 조합별로 **독립된 테이블 Session Block**을 수집한다. 신규 참가자 BB 대기·준비 핸드는 별도 집계하며 수익률 표본에 포함하지 않는다.

$$
\widehat J_{bb/100}=100\overline X,\quad
SE(\widehat J)=100\frac{s_X}{\sqrt K},\quad
CI_{95\%}\approx\widehat J\pm1.96SE.
$$

자기 전략과 비교 대상이 동일 카드 시드·자리 배치로 수집된다면 대응 차이 $D_k=X_{A,k}-X_{B,k}$로 CI를 구성해야 한다.

$$
\widehat\Delta=100\overline D,\quad
SE(\widehat\Delta)=100\frac{s_D}{\sqrt K}.
$$

**주의**: 같은 시드를 사용해도 행동 경로는 달라질 수 있다. 2명 이상의 상대가 관여하는 상태에서 **독립 세션 블록 수가 적으면** 정규근사 CI는 불안정하다. 훈련 Seed·평가 Seed·출력 파일은 분리한다.

## 10. 설치 방식과 신뢰성

### 10.1 통합 ZIP의 구조

`holdem-ai-expand-phase01-to-07-unified.zip`에는 다음이 포함된다.

- **1~7의 모든 Python 소스**, CLI, Mock/Real 통합 테스트.
- **8개**의 원문 설계·Phase 문서(마스터 설계 1 + 개별 Phase 문서 7).
- 이 **통합 기술 보고서** 및 **Phase별 원본 README 7개**.
- `SOURCE_MANIFEST.json` — 7개 원본 ZIP SHA-256 및 병합 파일별 SHA-256.
- `install_expand.py` — manifest 검증, 안전한 덮어쓰기 방지, Dry Run 및 설치.
- ZIP 루트 `README.md` — 일괄 사용법.

원본 모델 체크포인트, 학습된 가중치, `runs/`, `checkpoints/`, 기존 Base Phase 1~16 소스는 **포함하지 않는다**.

### 10.2 안전 설치 — Windows PowerShell

리포지토리 `holdem-ai` 루트에서 실행한다. ZIP을 `holdem-ai` 폴더 안에 두었다고 가정한다.

```powershell
Expand-Archive .\holdem-ai-expand-phase01-to-07-unified.zip `
  -DestinationPath .\holdem-ai-expand-unified -Force

# 1) 원본 117개 + 통합 보고서 1개, 총 118개 SHA-256 확인
uv run --extra dev --extra gpu python .\holdem-ai-expand-unified\install_expand.py --verify-only

# 2) 미리보기 — 실제 변경 없음
uv run --extra dev --extra gpu python .\holdem-ai-expand-unified\install_expand.py --repo . --dry-run

# 3) 1회 일괄 설치
uv run --extra dev --extra gpu python .\holdem-ai-expand-unified\install_expand.py --repo .
```

설치기는 다음 원칙을 따른다.

1. Base `pyproject.toml` 및 `src/holdem_ai/environment/multi_player_engine.py`가 없는 디렉터리에는 설치 거절.
2. Manifest의 `src/holdem_ai/expand/`, `scripts/expand_phase*.py`, `tests/expand/`, `docs/EXPAND_*`와 Phase별 README만 설치 허용.
3. 이미 **같은 SHA-256**의 파일이 있으면 변경 없이 건너뜀.
4. 다른 내용의 동일 경로가 있으면 기본적으로 **거절**, 임의 덮어쓰기 없음.
5. `--force`는 충돌 분석 후 **Expand 경로만** 덮어쓸 때 선택적으로 사용 가능.
6. Base 파일·학습 Checkpoint·실험 결과 폴더를 건드리지 않음.

### 10.3 필수 테스트

```powershell
# 테스트 디스커버리 및 Expand E1~E7 전체
uv run --extra dev --extra gpu python -m pytest -q tests/expand -rs

# Base Phase 1~16 포함 전체 Regression
uv run --extra dev --extra gpu python -m pytest -q
```

**중요**: 제공된 패키지 자체의 격리 환경에선 Base Phase 3 Engine이 없으므로 7개 Real Engine 테스트가 `skip`된다. **사용자의 실제 리포지토리에 설치한 다음 이 테스트가 skip되지 않는지 반드시 확인**한다.

## 11. 귀가 후 단계별 Acceptance Command

아래 명령은 `holdem-ai` 프로젝트 루트에서 실행한다. `runs/` 출력이 이미 있으면 **다른 파일명/실험 루트**를 선택한다.

### Gate A — 실제 2~6인 Table, InformationState

```powershell
uv run --extra gpu python -m scripts.expand_phase01_smoke `
  --hands-per-n 10 --seed 170001 `
  --output-json runs/expand_phase01/smoke_10.json

uv run --extra gpu python -m scripts.expand_phase02_smoke `
  --hands-per-n 10 --seed 180201 `
  --output-json runs/expand_phase02/smoke_10.json
```

문제 없으면 각각 `--hands-per-n 1000`으로 2~6인 인원별 장기 실험을 실행한다.

### Gate B — E3 PPO Pilot / Resume

```powershell
uv run --extra gpu python -m scripts.expand_phase03_train `
  --root runs/expand_phase03_pilot --device cuda `
  --max-updates 3 --hands-per-update 8 `
  --width 32 --heads 4 --layers 1 `
  --epochs 2 --minibatch-size 16 `
  --eval-every 1 --eval-hands 4 --snapshot-every 1

uv run --extra gpu python -m scripts.expand_phase03_train `
  --root runs/expand_phase03_pilot --resume --device cuda --max-updates 3
```

성공 시 E3 모델은 `runs/expand_phase03_pilot/best.pt`에 저장된다. **이 학습기와 Base Phase 15의 체크포인트는 분리**한다.

### Gate C — E4 MCCFR Kuhn Toy + 6-Max 연결

```powershell
uv run --extra gpu python -m scripts.expand_phase04_mccfr `
  --game kuhn --players 2 --iterations 10000 `
  --root runs/expand_phase04_kuhn_main --seed 42 `
  --save-every 500 --eval-hands 200

# 홀덤 연결은 실제 엔진 확인 후, 매우 작은 자원 예산부터 시작
uv run --extra gpu python -m scripts.expand_phase04_mccfr `
  --game holdem --players 6 --stack-bb 2 --iterations 2 `
  --postflop-buckets 16 --node-budget 20000 `
  --root runs/expand_phase04_holdem6_pilot
```

6-Max 2BB 실험은 **규칙/탐색 연결 확인용**이지 100BB 실전 수준의 학습이 아니다.

### Gate D — E5 Toy Search / Holdem Search

```powershell
uv run --extra gpu python -m scripts.expand_phase05_search `
  --game kuhn --players 2 --iterations 40 --time-ms 5000 `
  --nodes 5000 --depth 2 `
  --output-json runs/expand_phase05/kuhn_search_40.json

uv run --extra gpu python -m scripts.expand_phase05_search `
  --game holdem --players 6 --stack-bb 2 --iterations 10 `
  --time-ms 500 --nodes 500 --depth 1 `
  --output-json runs/expand_phase05/holdem6_smoke.json
```

### Gate E — E6 Synthetic Opponent Shift

```powershell
uv run --extra gpu python -m scripts.expand_phase06_benchmark `
  --blocks 16 --warmup 90 --post-shift 90 `
  --initial-style tight --new-style aggressive --seed 660006 `
  --output-json runs/expand_phase06/style_shift_16.json `
  --checkpoint runs/expand_phase06/registry_16.json
```

### Gate F — E7 Cash/Tournament 실행

학습된 E3 체크포인트 없이도 **Uniform-Legal Baseline**으로 먼저 엔진을 점검한다.

```powershell
uv run --extra gpu python -m scripts.expand_phase07_benchmark `
  --mode cash --groups 2 --players 2,3 --stacks-bb 100 `
  --opponents calling_station --seed 7070001 `
  --output-json runs/expand_phase07/cash_smoke_uniform.json

uv run --extra gpu python -m scripts.expand_phase07_benchmark `
  --mode tournament --groups 2 --players 2,3 `
  --opponents calling_station --seed 7074000 `
  --output-json runs/expand_phase07/tournament_smoke_uniform.json
```

실제 PPO 모델로 평가하려면 다음과 같이 **E3 모델을 명시적으로 지정**한다.

```powershell
uv run --extra gpu python -m scripts.expand_phase07_benchmark `
  --mode cash --groups 300 --players 2,3,4,5,6 `
  --stacks-bb 20,100,200 `
  --opponents random,calling_station,tight,pot_pressure `
  --checkpoint runs/expand_phase03_pilot/best.pt `
  --device cuda --seed 7071000 `
  --output-json runs/expand_phase07/cash_ppo_300.json
```

이 경우 목표 평가 핸드 수는

$$
5\;\text{player counts}\times 3\;\text{stack sizes}\times 4\;\text{opponents}\times300\;\text{blocks}
=\boxed{18,000\ \text{scored hands}}.
$$

참여 대기 준비 핸드는 추가로 진행될 수 있다.

## 12. 독립 검증 결과와 남은 Acceptance Gate

이번 통합 작업에서 실제 ZIP 원본을 병합한 폴더에서 아래 명령을 실행했다.

```bash
PYTHONPATH=src:. python -m pytest tests/expand -q -rs
```

**결과:** `164 passed, 7 skipped`. 7개 Skip은 모두 다음 모듈의 Base Engine Import 실패 때문이다.

| 보류 모듈 | 누락된 실제 의존성 |
|---|---|
| `test_e01_real_engine.py` | `holdem_ai.environment.multi_player_engine` |
| `test_e02_real_engine.py` | 동일 |
| `test_e03_real_engine.py` | 동일 |
| `test_e04_real_engine.py` | 동일 |
| `test_e05_real_engine.py` | 동일 |
| `test_e06_real_engine.py` | 동일 |
| `test_e07_real_engine.py` | 동일 |

테스트 격리 환경에서는 실제 엔진을 갖고 있지 않으며 PyTorch CPU 빌드만 확인했다. 따라서 **실제 게임 엔진과의 호환성, 6-Max 경기 품질, Windows CUDA 실행 성공은 아직 주장할 수 없다.**

### 12.1 귀가 후 보고받을 로그

- `git status` 및 설치기 Dry Run/실제 설치 출력.
- `pytest -q tests/expand -rs` 전체 출력(실제 엔진 7개 Skip 여부 포함).
- `pytest -q` 전체 Base+Expand Regression 결과.
- Phase 1·2 인원별 게임 Smoke 결과 JSON.
- E3 CUDA Pilot `summary`, `best.pt` 생성 및 `--resume` 로그.
- E4 Kuhn/6-Max 연결, E5 Search 실행 출력.
- E6 Synthetic Shift 보고서.
- E7 Cash/Tournament Smoke JSON, 플레이어 수·스택별 결과.
- 에러 로그가 있다면 원문 Traceback과 실행 커맨드.

### 12.2 다음 수용/수정 조건

- [ ] 실제 Base Phase 3 엔진에서 E1~E7 통합 테스트 **Skip 없이** 실행.
- [ ] 전체 Base+Expand Regression 통과.
- [ ] 2~6인 Cash 평가에서 칩 보존/Legal Action/Seat ID 보장.
- [ ] CUDA PPO Update/Checkpoint Resume 정상.
- [ ] E4/E5 Research Solver가 예산 초과 시 안전하게 중단하거나 Fallback.
- [ ] Tournament의 BB/Ante/등록·탈락 규칙과 상품 보상 검증.
- [ ] E7 실제 E3 모델 평가가 Uniform Baseline과 명확히 구분되어 기록.
- [ ] 로컬 결과 확인 후에만 **GitHub Commit/Push**.

## 13. 향후 연구에서 가장 중요한 미구현 과제

### 13.1 Hand-Range-Aware Opponent Pool

현재의 E3 `random`, `calling_station`, `tight`, `pot_pressure`는 간단한 Scripted Baseline이다. **상대의 카드와 Position에 따라 합리적으로 Fold/Call/Raise하는 강한 Range-Aware 상대 풀**은 아직 구현되지 않았다.

수학적으로 실제 Equity는 상대 Hole Card에 대한 사전 Uniform Range와, 행동을 관측한 **사후 Range**에서 달라질 수 있다.

$$
P(h_j\mid a,I)=\frac{P(a\mid h_j,I)P(h_j\mid I)}{\sum_{h'_j}P(a\mid h'_j,I)P(h'_j\mid I)}.
$$

이는 이후 E6를 개선하는 **Opponent Range Tracker** 및 E3 League Self-Play 확장으로 설계할 수 있다. **현재 E6는 공개 행동 스타일의 확률을 추정하지만 숨겨진 Hole Card의 정확한 Range를 산출하지 않는다.**

### 13.2 Slumbot 외부 벤치마크

Slumbot은 기본적으로 HUNL(2인)에 적합하다. 연구용 외부 HUNL 평가나 CFR 소스 참고에는 활용할 수 있지만, 그대로 6-Max의 5명 상대를 공급하는 데 적합하지 않다. **Slumbot 소스·모델·웹 API는 이 통합 ZIP에 포함되어 있지 않다.**

### 13.3 Pluribus 수준 개선

- 원본 논문: Brown & Sandholm (2019), *Superhuman AI for Multiplayer Poker*.
- 원본 Supplement: MCCFR, Action/Card Abstraction, Nested Imperfect-Information Search.
- Research References: [원본 논문](https://noambrown.github.io/papers/19-Science-Superhuman.pdf), [Supplementary Materials](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf).
- 이 패키지: MCCFR 기초·Toy 게임·근사 Search와 안전한 Route를 제공하며, **원본 Pluribus의 대규모 실전 Blueprint 및 Search 품질을 재현했다는 근거는 없다**.

## 14. 변경 이력과 출처 보존

통합 과정에서 개별 ZIP의 `src/`, `scripts/`, `tests/`, `docs/` 파일들은 **수정하지 않았다**. 동일 이름인 ZIP 루트의 `README.md`만 충돌해 7개를 다른 위치에 각각 보존했다. 패키지 밖의 새로운 산출물은 이 보고서, 통합 README, 설치기 및 SHA-256 Manifest다. 상세 Phase별 설계·실험 명세는 **원문 기술 문서**를 확인해야 한다.

| 문서 | 개발 Phase |
|---|---|
| `EXPAND_MASTER_DESIGN_6MAX_v1_0.md` | 전체 1~7 최초 상세 설계서(작성 당시 구현 전) |
| `EXPAND_PHASE01_DESIGN.md` | Cash Dynamic Table |
| `EXPAND_PHASE02_INFORMATION_AND_ENCODING.md` | Information State, Encoder |
| `EXPAND_PHASE03_MULTIAGENT_PPO.md` | Neural Self-Play |
| `EXPAND_PHASE04_MCCFR_BLUEPRINT.md` | MCCFR |
| `EXPAND_PHASE05_SUBGAME_SEARCH.md` | Search |
| `EXPAND_PHASE06_OPPONENT_MODELING.md` | Opponent Adaptation |
| `EXPAND_PHASE07_INTEGRATION_TOURNAMENT_BENCHMARK.md` | Integration, Tournament, Evaluation |

> [!success] 현재 판정
> **7개의 추가형 코드 패치는 하나의 무충돌 설치 구조로 통합되었고, 격리 단위·모의 엔진 테스트 164개가 통과했다.** 실제 Base Phase 3 엔진과 Windows/CUDA, 전체 Regression 및 전략 성능 수용 시험은 귀가 후 진행한다. 이 최종 판정이 완료되기 전에는 실전용 6-Max 강한 AI 완성으로 표시하지 않는다.
