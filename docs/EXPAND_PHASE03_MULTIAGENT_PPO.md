---
title: "Expand Phase 3 — Dynamic 6-Max Multi-Agent PPO Self-Play"
version: "1.0.0-patch"
date: 2026-10-12
status: "implemented; isolated CPU tests passed; real Phase 3 / Windows GPU acceptance pending"
project: "HUNL to flexible 6-Max NLHE"
tags: [poker, six-max, multi-agent, PPO, self-play, reinforcement-learning, expand-phase03]
---

# Expand Phase 3 — Multi-Agent PPO Self-Play

> [!IMPORTANT] 완료 범위와 검증 상태
> **NEW neural baseline** (Pluribus 원본이 아님). E1 DynamicTable과 E2 Phase2TableAdapter / `PolicyV2`를 **그대로 재사용**한다. HUNL Base Phase 1~16 및 Expand 1~2 파일은 변경하지 않는다. 현재 실제 Base Phase 3 게임 엔진이 작업 환경에 없어서 테스트 전용 엔진을 사용한 격리 CPU 테스트만 수행했다. Windows 사용자 측에서 전체 테스트와 **실제 Hold'em 게임/GTX 1070 CUDA 수용 시험**을 해야 한다. E3의 학습 완료는 강한 실력이나 GTO 성질을 의미하지 않는다.
>
> E4의 External-Sampling Linear MCCFR Blueprint / E5 Depth-Limited Search는 별개 알고리즘 트랙이다. E3는 이들을 구현하지 않는다.

## 1. Objective와 비목표

- **지원 인원**: 한 핸드에서 $N\in\{2,3,4,5,6\}$.
- **상태**: E1의 물리 좌석 6개, E2의 개인 카드/공개 정보 허용 목록, 합법 행동 7개.
- **학습**: PPO Actor + 관측 정보만 받는 Value Critic, 복수 Seat의 공유 가중치, scripted/snapshot 상대 혼합.
- **학습 경계**: Hand 단위 Terminal 보상, 참가 Seat 단위 GAE, 모든 Rollout 수집 완료 후 1회 PPO Update.
- **지속성**: `latest.pt`, `best.pt`, `run.json`, `state.json`, `metrics.jsonl`, `validation_anchor.json`, snapshot 파일.
- **학습 중 Player 입장/퇴장**: E1의 Hand Boundary 제약을 따른다. E3 Collector 기본 세션은 초기 2~6인 로스터가 BB 입장 규칙에 따라 정착된 이후 학습을 시작한다. Collector 자체가 학습 중에 새 사람을 임의로 입장시키는 커리큘럼은 **아직 없다**. Bust 이후 세션을 새로 만든다.
- **Not implemented**: 6-Max 토너먼트, Opponent Online Adaptation, MCCFR, Search, human UI, 대용량 병렬 환경/분산 PPO, 믿을 만한 exploitability 산출.

## 2. Module 및 Interface

```text
src/holdem_ai/expand/rl/
  __init__.py
  model.py         # ActorCritic(PolicyV2 + ValueV2)
  population.py    # scripted opponents, role mixture
  collector.py     # E1/E2 real-hand 2..6p rollout, freeze during collection
  returns.py       # per-(session,hand,seat) GAE
  ppo.py           # clipped PPO
  trainer.py       # checkpoints, snapshots, resume, validation, heldout
scripts/
  expand_phase03_train.py
  expand_phase03_evaluate.py
tests/expand/
  e03_fake_engine.py  # test-only, NOT a poker rules engine
  test_e03_*.py
```

`PolicyV2` 생성 시 인수는 `width`, `heads`, `layers`이며 E2의 7-action 구조를 그대로 유지한다. Critic(`ValueV2`)은 `EncodedObservation`만 허용한다. `TrainingOnlyCriticState`, 전체 Deck, 상대 Hole Cards, 전용 RNG Seed는 두 모델에 전달하지 않는다. E2는 `InformationState`의 정의와 `encode_observation()`을 소유하며, E3에서 중복 정의하지 않는다.

### 2.1 핵심 타입

```python
@dataclass(frozen=True)
class MultiAgentTransition:
    session_id: str
    hand_id: str
    player_id: str
    seat_id: int
    observation: InformationState
    encoded: EncodedObservation
    action: str
    action_logprob: float
    value: float
    reward_bb: float
    terminal_hand: bool
    terminal_session: bool
    policy_version: str
    num_players: int
```

`terminal_session`은 현 Cash-Game per-hand rollout에서 `False`로 기록한다. Tournament / Session GAE는 E7에서 따로 설계한다. 배우는 플레이어마다 같은 가중치를 공유할 수 있지만 `seat_id`, `player_id`, `hand_id`별 Trajectory는 섞지 않는다.

### 2.2 행동/보상 경계

행동 정책은 E2의 7개 추상 Action에서만 샘플링한다.

$$
\mathcal A=\{\mathrm{Fold,Check,Call,Raise}_{0.5P},\mathrm{Raise}_{P},\mathrm{Raise}_{2P},\mathrm{AllIn}\}.
$$

E1/E2의 `to_table_action()`이 실제 합법 `raise_to`(현재 Street 누적 목표 베팅액)를 만든다. E3는 실수 Raise 금액을 만들거나 목표액을 Clamp하지 않는다.

핸드 $h$에서 Seat $i$의 보상은

$$
R_{i,h}=\frac{S^{\mathrm{end}}_{i,h}-S^{\mathrm{start}}_{i,h}}{BB}=\frac{\mathrm{payoffs\_chips}_{i,h}}{BB}.
$$

Rake가 없으면 반드시

$$\sum_{i=0}^{5}R_{i,h}=0.$$

중간 결정 $t$에는 $r_{i,t}=0$을 주고, 같은 Seat의 **마지막 의사결정**에 $r_{i,T_i}=R_{i,h}$을 준다. `Fold`한 Seat도 앞서 내린 의사결정이 있으면 Terminal 보상이 그 마지막 결정에 할당된다. 상대의 마지막 결정과 혼동하면 안 된다. Blind 강제 All-In 등으로 어떤 Seat에 **선택 의사결정이 전혀 없을 때는 해당 Seat의 PPO 샘플을 만들지 않는다**. 테이블의 총 보상 제로섬 검증은 별도로 수행한다.

## 3. Multi-Agent Collector

- `RolloutConfig.num_players=(2,3,4,5,6)`에서 `random.Random`으로 균등하게 Hand 대상 인원 $N$을 선택.
- 각 $N$에 E1 `DynamicTable` 세션을 유지. BB 대기 인원이 있는 동안 진행한 Warmup Hand는 **실제로 Settled되지만 학습 Hands 수에서 제외**.
- 로스터가 $N$명이 된 핸드를 `hands_per_update`에 포함. Bust로 인원이 줄면 새 Cash 테이블을 생성. Stack은 E1이 Hand 간에 이월한다.
- `player_0`은 Trainable, 나머지 참가 Player는 `shared_policy_fraction` 확률로 공유 Trainable, `snapshot_fraction` 확률로 이전 정책 Snapshot, 나머지는 Scripted. Snapshot이 없을 때 해당 질량은 Scripted.
- 한 Rollout 동안 Trainable Actor는 `.eval()`과 `@torch.no_grad()` 추론만 사용. PPO `optimizer.step()`은 **모든 Hand 수집 이후** 수행.
- Schedule / Seat Setup / Deck RNG / Opponent Role RNG / Action RNG는 분리하여, 행동 샘플링이 다음 대상 인원 추첨에 영향을 주지 않도록 구성. 그러나 **행동 경로와 탈락 때문에 두 서로 다른 후보 정책이 항상 동일한 게임 진행을 공유한다는 보장은 없다**.
- 자의적으로 상대 Hole Card/Deck를 읽지 않는다. `Phase2TableAdapter.observe()`가 생성한 상태만 Actor/Critic에게 전달.

Opponent 역할:

| 역할 | 구현 |
|---|---|
| Trainable | 현재 Actor / Critic 공유, 여러 Seat 가능 |
| Random | 모든 현재 합법 추상 행동 중 균등 샘플 |
| Calling Station | Check > Call > Fold |
| Tight | Check > Fold > Call (핸드 강도 모델 아님) |
| Pot Pressure | Pot > Double Pot > Half Pot > All-in > Call > Check > Fold |
| Frozen Snapshot | 예전 ActorCritic Weight를 읽기 전용으로 추론 |

기본값:

$$
P(\mathrm{shared})=0.25,\qquad P(\mathrm{snapshot})=0.25,\qquad P(\mathrm{scripted})=0.50
$$

Snapshot이 아직 없는 첫 Update에서는 공유 정책 25%, Scripted 75%가 될 수 있다. **이는 학습용 상대 분포이며, Final Evaluation Pool과 같지 않아야 한다.**

## 4. Advantage와 PPO

Seat $i$의 **본인 행동 시점**을 $\tau_0,\dots,\tau_{T_i}$라 하자. 다른 플레이어들의 결정이 그 사이에 끼어 있어도 다음 단계는 $\tau_{k+1}$이다.

$$
\delta_{i,k}=r_{i,k}+\gamma(1-d_{i,k})V_\phi(I_{i,\tau_{k+1}})-V_\phi(I_{i,\tau_k}),
$$

$$
\hat A_{i,k}=\delta_{i,k}+\gamma\lambda(1-d_{i,k})\hat A_{i,k+1},\qquad
\hat G_{i,k}=\hat A_{i,k}+V_\phi(I_{i,\tau_k}).
$$

기본 $\gamma=1$, $\lambda=0.95$. 각 Hand 마지막 의사결정은 $d_{i,T_i}=1$이므로 미래 가치가 0이다. 초기 E3에서는 Hand Boundary를 Episode Terminal로 보며, 복수 Hand에 걸친 Session GAE는 포함하지 않는다.

$$
\rho_t(\theta)=\exp\left[\log\pi_\theta(a_t|I_t)-\log\pi_{\mathrm{old}}(a_t|I_t)\right],
$$

$$
L^{clip}=\mathbb E\left[\min\{\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\}\right],
$$

$$
\mathcal L=-L^{clip}+c_V\mathbb E[(V_\phi-\hat G)^2]-c_H\,\mathbb E[\mathcal H(\pi_\theta)].
$$

기본 $\epsilon=0.2$, $c_V=0.5$, $c_H=0.01$, Adam learning rate $3\times10^{-4}$, 2 epochs, minibatch size 64, gradient clipping norm 1, approximate KL 경계 0.20. `Approx KL`은 PPO 모니터링용 경험적 추정량이지 GTO 보장이 아니다. Learned payoff의 고분산·작은 평가 샘플에는 주의해야 한다.

## 5. Checkpoint & Resume

```text
runs/expand_phase03_main/
  .expand03.lock            # 실행 충돌 방지 OS lock, 빈 폴더 검사 예외
  run.json                  # 고정 Config / 알고리즘 버전
  latest.pt                 # Model + Adam + torch CPU/CUDA RNG + snapshots + progress
  best.pt                   # Validation 승자 or 학습 전 초기 모델
  state.json                # completed_updates / best_update / best SHA-256
  validation_anchor.json    # 초기 정책 vs random/calling_station 성능
  metrics.jsonl             # 1 Record/Update and Initial baseline
  summary.json              # 마지막 호출 요약
  snapshots/update_XXXXXXX.pt
```

`--max-updates`는 **해당 호출의 추가 Update 수**이며 전체 누적 목표가 아니다. `--resume` 실행에는 `--root`, `--resume`, `--device`, `--max-updates`만 사용하고, 나머지 설정은 `run.json`에서 복원된다. Checkpoint는 `.tmp` 임시 파일에 쓰고 원자적 교체한다. 상태·Best 해시 불일치는 **멈춰서 확인**한다. 정전/프로세스 종료가 연속된 `best.pt`/`latest.pt`/`state.json` 원자 교체 사이에 끼면 자동 무손실 복구가 반드시 성공하는 것은 아니다. 백업 없이 Run 디렉터리 수정을 금지한다.

최적 정책 선택 점수는 Training과 분리된 Validation 카드 시드 도메인을 사용한다.

$$
S_t=\frac12\left[(J_R(t)-J_R(0))+(J_C(t)-J_C(0))\right],
$$

$J_R$는 Random, $J_C$는 Calling Station 상대의 평균 bb/100이다. `best.pt`는 $S_t$가 기존 최고보다 클 때만 교체한다. **초기 모델이 그대로 Best일 수 있다.** Small-Hands Validation은 노이즈가 매우 크며, Best 선택에 반복 사용했으므로 unbiased Test가 아니다. 보수적 평가를 위해 별도 `expand_phase03_evaluate`의 `seed=340000000`을 최초 한 번 사용하고 다른 미학습/미최적화 Bank에서 확인한다.

- Train rollout_seed: 기본 320,000,000
- Validation seed: 기본 330,000,000
- 별도 Heldout evaluation seed: 기본 340,000,000

기본 운영은 단일 Process/단일 GPU 또는 CPU이다. 다중 Process 수집·AMP·Distributed Synchronization은 E3 구현 범위 밖이다.

## 6. 사용법 — Windows PowerShell

> 먼저 **Expand Phase 1과 2를 순서대로 설치**해야 한다. ZIP을 프로젝트 루트로 복사해 압축을 풀고 아래를 실행한다. 기존 Base Source를 덮어쓰지 않는다.

```powershell
Expand-Archive .\holdem-ai-expand-phase03.zip -DestinationPath .\holdem-ai-expand-phase03 -Force

Copy-Item .\holdem-ai-expand-phase03\src\holdem_ai\expand\rl .\src\holdem_ai\expand\ -Recurse -Force
Copy-Item .\holdem-ai-expand-phase03\scripts\expand_phase03_*.py .\scripts\ -Force
Copy-Item .\holdem-ai-expand-phase03\tests\expand\test_e03_*.py .\tests\expand\ -Force
Copy-Item .\holdem-ai-expand-phase03\tests\expand\e03_fake_engine.py .\tests\expand\ -Force
Copy-Item .\holdem-ai-expand-phase03\docs\EXPAND_PHASE03_MULTIAGENT_PPO.md .\docs\ -Force
```

### Gate A — Unit / Regression

```powershell
uv run --extra dev --extra gpu python -m pytest -q -k e03
uv run --extra dev --extra gpu python -m pytest -q tests/expand
uv run --extra dev --extra gpu python -m pytest -q
```

### Gate B — CPU/GTX 1070 Pilot (3 Updates)

```powershell
uv run --extra gpu python -m scripts.expand_phase03_train `
  --root runs/expand_phase03_pilot `
  --device cuda `
  --max-updates 3 `
  --hands-per-update 8 `
  --width 32 --heads 4 --layers 1 `
  --epochs 2 --minibatch-size 16 `
  --eval-every 1 --eval-hands 4 `
  --snapshot-every 1
```

3 × 8 = **24개의 학습 대상 Hand**. BB 입장 대기 중 Warmup Hand는 이 수에서 제외되고 실제 총 진행 Hand는 더 많을 수 있다. 2~6인의 실제 인원 분포는 난수 표본이므로 24 Hand에서 모든 인원이 등장한다고 보장하지 않는다. CPU 파일럿은 `--device cpu`로 대체 가능하다.

### Gate C — Resume (additional 3 Updates)

```powershell
uv run --extra gpu python -m scripts.expand_phase03_train `
  --root runs/expand_phase03_pilot `
  --resume --device cuda --max-updates 3
```

### Gate D — 본 학습 새 Run (나중에 실행)

```powershell
uv run --extra gpu python -m scripts.expand_phase03_train `
  --root runs/expand_phase03_main `
  --device cuda --max-updates 100 `
  --hands-per-update 64 `
  --width 64 --heads 4 --layers 2 `
  --epochs 2 --minibatch-size 64 `
  --eval-every 10 --eval-hands 40 `
  --snapshot-every 10
```

### Gate E — 독립 평가 (학습 없이 Best 읽기)

```powershell
uv run --extra gpu python -m scripts.expand_phase03_evaluate `
  --candidate runs/expand_phase03_pilot/best.pt `
  --hands-per-cell 100 `
  --num-players 2,3,4,5,6 `
  --opponents random,calling_station,tight,pot_pressure `
  --seed 340000000 --device cuda `
  --output-json runs/expand_phase03/eval_pilot_100.json
```

$5\text{ PlayerCounts}\times4\text{ Opponents}\times100\text{ Hands}=2{,}000$ **평가 대상 Hand**이며, 준비 핸드(Warmup)를 제외한 수다. CSV/MD 파일은 이 버전에서 자동 출력하지 않고 JSON을 기록한다. 같은 출력 경로가 이미 있으면 덮어쓰지 않는다.

### Gate F — 2~6명별 실제 엔진 테스트 / CUDA

기존 테스트의 `test_e03_real_engine.py`는 실제 Base Phase 3 엔진이 존재할 때 자동으로 실행된다. GTX 1070 학습은 사용자의 로컬에서 수행해야 한다. 일부 게임 규칙 버그가 발견될 경우 **E3를 승인하지 않고 해당 현상을 재현하는 테스트를 추가한 뒤** 별도 패치로 수정한다.

## 7. 테스트 해석과 명시적 한계

| 테스트 계층 | 현재 결과 |
|---|---|
| E1/E2/E3 Sandbox isolated CPU tests | **45 passed** |
| Base Phase3 2~6 실제 핸드 통합 모듈 | **3 skipped** (해당 엔진이 Sandbox에 없음; E1/E2/E3 각 1개) |
| Windows Python 3.11 / GTX 1070 CUDA | **미실행** |
| 기존 Base Phase 1~16 regression | **미실행** |
| 3 Updates 실제 Hold'em Training | **미실행** |
| Phase4 MCCFR, Phase5 Search | **미구현** |

> [!CAUTION] 모델 성능의 해석
> `bb/100`은 해당 Scripted Opponent Pool·스택 조건·Card Seed 하의 관측 수익률이다. `CI95_normal`은 각 핸드 독립성을 단순 가정하는 탐색적 정상 근사로, Cash Session의 Stack 의존성이 있으면 과도하게 좁을 수 있다. 정식 모델 비교는 별도 Independent Seeds / Session-Cluster CI / 동일 규칙의 평가 프로토콜을 추가해야 한다. Phase 16 HUNL 평가 모델과 네트워크 형식도 직접 호환되지 않는다.

## 8. 완료 승인(Windows 로그 수령 후)

- [ ] `pytest -q -k e03`에서 실제 엔진용 테스트가 Skip 없이 통과
- [ ] `pytest -q tests/expand` 통과
- [ ] Base Phase 1~16 `pytest -q` 전체 통과
- [ ] CUDA 학습 3 Updates + Checkpoint Load/Resume 확인
- [ ] **2~6인별 게임 및 Chip Conservation** 검증
- [ ] GAE / action-masking / 불법 행동 0 / JSONL 로그 검증
- [ ] 검증 후 GitHub Commit & Push

E3가 통과할 때까지 E4 작업은 독립적으로 개발 가능하지만, **통합 시스템 완료로 표시하지 않는다**.
