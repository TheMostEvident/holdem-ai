---
title: Expand Phase 4 — External-Sampling MCCFR Blueprint
aliases:
  - Expand E4
  - Pluribus-style MCCFR Prototype
date: 2026-10-12
status: isolated-tests-passed / Windows-real-engine-pending
project: 6-Max NLHE Expand
phase: "Expand Phase 4"
tags:
  - poker
  - 6max
  - mccfr
  - pluribus
  - reinforcement-learning
---

# Expand Phase 4 — MCCFR Blueprint: Implementation & Acceptance Specification

> **중요:** 이것은 Pluribus **원본 코드/원본 Blueprint를 복제한 것이 아니다.** 공개된 *External-Sampling Linear MCCFR* 구조를 참고하여 단계별 수학적 검증을 할 수 있도록 제작한 **CPU 기반 연구용 Python 프로토타입**이다. Postflop Card Buckets는 **학습된 Equity/EMD Buckets가 아닌 SHA-256 재현성 검증용 임시 버킷**이다. Negative-Regret Pruning 및 온라인 Nested Search는 **아직 구현되지 않았다**. (Nested Search는 Expand Phase 5 대상.) 대규모 6-Max 완전 전략 성능 주장은 불가능하다.

## 1. 목표와 비범위

**구현 목표 E4**

1. Kuhn 2인 게임의 수학적 검증 가능한 External Sampling MCCFR.
2. Leduc 2/3인, Kuhn 3인 데이터 수집·제로섬 및 합법 행동 검사.
3. 기존 Phase 3 `MultiPlayerHandEngine`에 2~6인 6좌석 인덱스 그대로 연결.
4. 원자적 Iteration(예산 초과 시 부분 Regret 업데이트 금지), `max_nodes/max_infosets/max_depth` 자원 제한.
5. Linear MCCFR weight, Current/Average 전략, Checkpoint, 정확한 Resume, 독립 평가.
6. 6-Max 홀덤용 Abstraction ID를 모든 Checkpoint에 기록하여 호환성 검증.

**비범위**: Pluribus 원본 학습량, 원본 200 equity buckets, 95% negative-regret pruning, 500 search buckets, 실시간 Nested Search, GTO 인증, Tensor/GPU 최적화, 무제한 Action Tree. 또한 Tournament 및 DynamicTable의 join/leave는 Phase 1 및 후속 통합 과제이고 E4에서는 **단일 핸드의 고정 2~6인 샘플**을 다룬다.

## 2. 기존 Phase와의 관계

```text
Base Phase 3 MultiPlayerHandEngine   E1 DynamicTable ── E2 Info Encoder ── E3 PPO
            │                                                      │
            └──── E4 HoldemState / Adaptor ── External MCCFR ── Blueprint
                                                              │
                                             E5 Subgame Search (future)
```

- `src/holdem_ai/expand/pluribus/...`만 추가. 기존 Base Phase 1~16 / Expand 1~3 파일 수정 없음.
- Phase3의 `step(int action_id)`와 `observe(current_actor)`가 유일한 홀덤 엔진 결합 지점.
- PPO `.pt`와 MCCFR `blueprint.json.gz`는 형식이 다르며 서로 로딩 불가.
- HUNL 2인 정책을 6인 Blueprint로 주장하지 않는다.

## 3. 수식 — External-Sampling Linear MCCFR

### 3.1 Regret Matching

정보집합 $I$와 추상 행동 $a \in \mathcal A(I)$에서:

$$
\sigma_i^t(I,a)=\frac{[R_i^{t-1}(I,a)]_+}{\sum_{b\in\mathcal A(I)}[R_i^{t-1}(I,b)]_+},
$$

단 분모가 $0$이면 $\sigma=1/|\mathcal A(I)|$로 설정.

상대방·Chance 노드는 해당 현재 정책에서 **하나만 샘플링**하고, Traverser $i$의 노드는 합법 행동을 전부 탐색한다.

$$
\hat v_i(I)=\sum_{a\in\mathcal A(I)}\sigma_i^t(I,a)\hat v_i(I,a)
$$

$$
R_i^{t}(I,a)\leftarrow R_i^{t-1}(I,a)
 + w_t\big(\hat v_i(I,a)-\hat v_i(I)\big)
$$

$$
w_t=\left\lfloor\frac{t-1}{N}\right\rfloor+1
$$

여기서 $N$은 현재 게임 플레이어 수이며 Traverser를 $t\bmod N$으로 순회한다. 각 Iteration에서 Strategy를 변경하면 계산 편향이 생길 수 있으므로 **Iteration 동안 Regret/Strategy Delta를 임시로 보관하고 성공한 뒤 한 번에 Commit**한다.

### 3.2 Average Strategy의 Own-Reach Weight

Traverser $i$는 자기 선택지들을 전개하고, 상대 행동은 정책대로 샘플링한다. 이때 Traverser의 과거 자체 행동 도달확률을 $\pi_i$라 하면:

$$
S_i(I,a)\leftarrow S_i(I,a) + w_t \pi_i^{\sigma^t}(I)\sigma_i^t(I,a)
$$

상대와 Chance의 도달확률은 방문 Sampling으로 처리된다. 정책별 샘플링 확률과 추가 Weight를 중복 적용하지 않도록 구분한다.

$$
\bar\sigma_i(I,a)=\frac{S_i(I,a)}{\sum_b S_i(I,b)}
$$

Average 값이 아직 없다면 유니폼 Fallback이다. Resume는 `iteration`, `config`, `infoset tables`을 JSON.gz Checkpoint에 보존하고 Iteration RNG를 `(seed, global_iteration)` 해시로 재생성한다.

### 3.3 2인 Kuhn NashConv

$$
\operatorname{NashConv}(\bar\sigma)=
\max_{\pi_0}u_0(\pi_0,\bar\sigma_1)+
\max_{\pi_1}u_1(\bar\sigma_0,\pi_1)
$$

Kuhn 2인에서는 가능한 카드 배분 6개 및 두 플레이어의 가능한 Deterministic Best Response를 열거해 **Toy Game 한정 정확한 NashConv**를 계산한다.

격리 검증(Seed 42):

| 학습 Iterations | Kuhn NashConv |
|---|---:|
| 0 | 0.916667 |
| 3,000 | 0.050732 |
| 10,000 | 0.013073 |
| 30,000 | 0.015137 |

이 결과는 Pluribus/6-Max의 NashConv나 Exploitability를 의미하지 않는다. Monte Carlo 학습 특성상 단조 감소하지 않는다.

### 3.4 보상

핸드 정산 직후:

$$
r_i=\frac{\mathrm{payoffs\_chips}_i}{BB},\qquad
\sum_{i=0}^{5}r_i=0
$$

Seat가 비어 있어도 물리 ID는 0~5를 유지한다. 실제 Traverser는 활성 Seat 집합에서 $t\bmod N$으로 선택한다. Kuhn/Leduc toy에서는 ante 단위 칩 보상.

## 4. Abstraction 계약

### 4.1 Hold'em Abstract Actions

Base Phase3 Engine ID 기반:

| E4 Name | Phase3 ID | 조건 |
|---|---:|---|
| `fold` | 0 | legal |
| `check` | 1 | `to_call=0` |
| `call` | 1 | `to_call>0` |
| `raise_half_pot` | 2 | `--allow-half-pot` 설정 시 |
| `raise_pot` | 3 | legal |
| `all_in` | 4 | legal, 기존 Raise와 동일 금액이면 중복 제거 |

E2의 7종 Action Translator 전체와 동일하지 않다. 이번 E4는 **작은 Action Tree를 대상으로 Base3 Engine의 기존 합법 ID만 사용**한다. 다음 단계에서 Off-Tree Action을 허용하려면 별도 Action/Betting Abstraction 명세가 필요하다.

### 4.2 Information Set / Private Boundary

입력은 반드시 `engine.observe(engine.current_player)['raw_obs']`에서 허용된 필드만 가져온다. 상대 Hole Cards / 미사용 Deck / 다음 카드 / RNG State / Critic State는 입력하지 않는다.

- Preflop: Hero Hole Cards **정확히 유지** (`exact`). Preflop 'lossless'는 이 프로토타입의 자기 개인 카드에 대해 정확한 카드 문자열을 사용한다는 의미이다.
- Flop/Turn/River: `(Hero Hole + Board)`를 SHA-256으로 해시하여 `K∈{16,32,64,...}`개의 `prototype_bucket`에 매핑. **강도·Equity 보존 보장이 없다**.
- Public Info: Board, Street, Button, 전체 Seat Stack/Contribution/Status, Betting History, Target Raise를 그대로 보존한다.
- `abstraction_id=sha256(config)`는 버킷 수/Action Size 옵션을 포함. 모델 체크포인트마다 반드시 동일 ID 검증.

원본 Pluribus의 Postflop 200 Equity Buckets, Search 500 Buckets 또는 EMD 유사도를 **재현했다고 주장하지 않는다**.

## 5. 비용 및 중단 조건

`MCCFRConfig`:

- `max_nodes_per_iteration=20_000` — 초과 시 해당 Iteration 전체 **Discard**.
- `max_infosets=200_000` — Sparse Dictionary 논리적 크기 제한. 실제 Python 메모리 사용량 별도 측정.
- `max_depth=256` — 비정상적인 Action Chain 방지.
- Negative-Regret Pruning: **OFF / 미구현**. 퇴화되지 않은 Toy Reference 테스트 없이는 켜지 않는다.

MCCFR Regret/Strategy의 dense float32 구조 예상치는

$$M_{dense}\approx8\,|\mathcal I|\,\bar A\;\mathrm{bytes}$$

이지만 **현재 Python float/list/dict JSON 저장**은 훨씬 큰 메모리를 사용한다. `max_infosets`는 메모리 바이트 강제 제한이 아니며 GTX 1070의 VRAM과 직접 무관하다. 대규모 6-Max의 모든 가능 상태 탐색은 현재 범위 밖이다.

## 6. 파일 및 인터페이스

```text
src/holdem_ai/expand/pluribus/
├── __init__.py
├── abstraction/
│   ├── __init__.py
│   └── cards.py             # AbstractionConfig, hero-only key, fingerprint
└── blueprint/
    ├── __init__.py
    ├── toy_games.py         # KuhnGame, LeducGame
    ├── holdem.py            # HoldemState, HoldemGame 2..6
    ├── mccfr.py             # MCCFRSolver, InfosetStore, budgets
    ├── kuhn_reference.py    # Kuhn-only exact NashConv
    ├── checkpoint.py        # Atomic gzipped JSON save / load
    └── evaluate.py          # Seed-separated uniform opponent benchmark
scripts/
├── expand_phase04_mccfr.py
└── expand_phase04_evaluate.py
tests/expand/test_e04_*.py
docs/EXPAND_PHASE04_MCCFR_BLUEPRINT.md
```

## 7. 설치 (Windows PowerShell)

아래는 Base Phase 1~16과 Expand Phase 1~3을 **이미 설치한** 상태를 전제로 한다.

```powershell
Expand-Archive .\holdem-ai-expand-phase04.zip -DestinationPath .\holdem-ai-expand-phase04 -Force

Copy-Item .\holdem-ai-expand-phase04\src\holdem_ai\expand\pluribus .\src\holdem_ai\expand\ -Recurse -Force
Copy-Item .\holdem-ai-expand-phase04\scripts\expand_phase04_*.py .\scripts\ -Force
Copy-Item .\holdem-ai-expand-phase04\tests\expand\test_e04_*.py .\tests\expand\ -Force
Copy-Item .\holdem-ai-expand-phase04\docs\EXPAND_PHASE04_MCCFR_BLUEPRINT.md .\docs\ -Force
```

**CPU 알고리즘이다.** GPU가 필수 요소는 아니다. `uv run --extra dev --extra gpu ...`는 사용자의 기존 `uv` 환경과 일치하도록 예시로 사용한다.

```powershell
uv run --extra dev --extra gpu python -m pytest -q -k e04
uv run --extra dev --extra gpu python -m pytest -q tests/expand
uv run --extra dev --extra gpu python -m pytest -q
```

## 8. 실험 커맨드

### 8.1 작은 게임 레퍼런스 — 2인 Kuhn

```powershell
uv run --extra gpu python -m scripts.expand_phase04_mccfr `
  --game kuhn --players 2 --iterations 10000 `
  --root runs/expand_phase04_kuhn_main `
  --seed 42 --save-every 500 --eval-hands 200
```

예상: `toy_nash_conv.nash_conv`가 Uniform 초기값(0.9167)에서 낮아지는지 확인. 매 학습 실행마다 통계적 흔들림은 있다.

추가 10,000 Iterations Resume:

```powershell
uv run --extra gpu python -m scripts.expand_phase04_mccfr `
  --root runs/expand_phase04_kuhn_main `
  --resume --iterations 10000 --save-every 500
```

`--resume`에서는 모델의 Game, Players, Seed, Bucket, 알고리즘을 Checkpoint에서만 복원하고 CLI의 학습 설정 변경은 허용하지 않는다.

### 8.2 3인 Kuhn / Leduc

```powershell
uv run --extra gpu python -m scripts.expand_phase04_mccfr `
  --game kuhn --players 3 --iterations 3000 `
  --root runs/expand_phase04_kuhn3

uv run --extra gpu python -m scripts.expand_phase04_mccfr `
  --game leduc --players 2 --iterations 3000 `
  --root runs/expand_phase04_leduc2
```

### 8.3 실제 Phase3 Hold'em Engine — 2인 작은 트리

```powershell
uv run --extra gpu python -m scripts.expand_phase04_mccfr `
  --game holdem --players 2 --stack-bb 2 --iterations 5 `
  --node-budget 50000 --root runs/expand_phase04_holdem2_pilot
```

### 8.4 실제 Phase3 Hold'em Engine — 6인 자원 제한 파일럿

```powershell
uv run --extra gpu python -m scripts.expand_phase04_mccfr `
  --game holdem --players 6 --stack-bb 2 --iterations 2 `
  --postflop-buckets 16 --node-budget 20000 `
  --root runs/expand_phase04_holdem6_pilot
```

**중요:** `stopped_by_budget`가 null이 아니거나 `completed_iterations=0`이면 6인 Blueprint 학습이 성공했다고 볼 수 없다. 이 경우 `max_nodes_per_iteration`을 장비에서 감당 가능한 수준으로 늘리거나 Action Tree를 더 축소한 후 재시험해야 한다. Baseline에서는 100BB Full Tree를 처리하지 않는다.

### 8.5 결과 평가(학습 수정 없음)

```powershell
uv run --extra gpu python -m scripts.expand_phase04_evaluate `
  --root runs/expand_phase04_kuhn_main --hands 500 `
  --mode average --seed 440100000 `
  --output-json runs/expand_phase04_kuhn_main/eval_500.json
```

## 9. 실험 결과 파일

```text
runs/expand_phase04_kuhn_main/
  blueprint.json.gz   # Schema+Config+Infoset Regret+Average+SHA256
  metrics.jsonl       # 성공한 Iteration 메트릭; 체크포인트 Commit 단위
  summary.json        # 가장 최근 Train 명령 결과
  eval_500.json       # 별도 평가 명령 실행 후
```

`blueprint.json.gz`는 다른 신뢰되지 않은 외부 파일을 자동 실행하는 Pickle이 아니라 JSON이다. Checkpoint는 SHA-256 무결성 검사와 Abstraction Fingerprint 검사를 거친다. 동일 Run 폴더에는 단일 Writer만 허용한다고 가정하며, **멀티프로세스 파일 락은 아직 제공하지 않는다**.

## 10. 테스트·승인 게이트

- [x] Kuhn 2인 초기 전략 대비 Exact NashConv 감소 확인 (10,000 Iterations)
- [x] Kuhn 3인, Leduc 2/3인 Toy Zero-Sum, Infoset, Chance 검증
- [x] 2~6인 홀덤 모의 엔진 연결, 물리 Seat ID/비활성 좌석 처리
- [x] JSON.gz Checkpoint, SHA-256 변조, 재개 결정성
- [x] 추상 카드 Bucket Fingerprint, Actor의 Rival Hole Card 비노출
- [x] 예산 초과/재귀 깊이 제한 시 Partial Updates 미반영
- [ ] **Windows 실제 Phase3 Engine 2~6인 실게임 테스트**
- [ ] **Windows GTX1070/CPU 리소스, 시간/메모리 프로파일**
- [ ] **전체 기존 프로젝트 Regression** (현재 작업환경은 Base 리포지토리 전체 미설치)
- [ ] **6인 제한 Tree에서 실측 유효 학습 Iterations 및 평가**
- [ ] **Negative-Regret Pruning / Equivalence Buckets / Pluribus 규모 검증** (후속 연구)

**완료 판정:** E4는 추가형 프로토타입 구현 완료. Windows 실기 시험 및 6인 Blueprint 품질 검증 전까지 **최종 승인 아님**. GitHub Push는 사용자 검증 이후에 수행.

## 11. Pluribus 출처 및 원본과의 차이

- Brown & Sandholm, *Superhuman AI for multiplayer poker*, Science (2019). https://noambrown.github.io/papers/19-Science-Superhuman.pdf
- Brown & Sandholm, Supplementary Materials (30p), External-Sampling Linear MCCFR 및 Nested Search: https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf
- Pluribus는 오프라인 Blueprint 학습과 깊이 제한 탐색을 사용. 본 E4는 온라인 탐색 없이 MCCFR 기초 로직과 축소된 6인 게임 Adapter를 만든 단계.
- 6인 게임은 일반적으로 2인 제로섬 NashConv 수렴 보증을 그대로 사용할 수 없다.
