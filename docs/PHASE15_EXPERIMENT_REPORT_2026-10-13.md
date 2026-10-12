# Phase 15 — HUNL PPO 장시간 학습 실험 보고서

> **작성일:** 2026-10-13  
> **프로젝트:** TheMostEvident/holdem-ai  
> **연구 범위:** 기존 2인 Heads-Up No-Limit Hold'em, HUNL200 (200 BB starting stack), Phase 15 PPO  
> **상태:** 세 차례 실행 로그 분석 완료 / 장시간 학습 안정화 미완료 / Final Held-out Test 미실행  
> **근거:** 사용자 Windows PowerShell 학습 로그와 현재 저장소의 Phase 15 구현. 이 문서는 실제로 확인되지 않은 GPU·전략 성능을 주장하지 않는다.

## 1. Executive Summary

Phase 15에서는 Phase 14 PPO 체크포인트를 이어받아 GTX 1070 CUDA에서 장시간 Self-Play를 시도했다. 최초 구성에서는 약 3분 11초 후 연속 Entropy 경고로 중지했다. 수집 Hand 수 증가와 학습 Epoch 감소를 적용한 안정성 실험에서는 처음 100 Updates를 경고 없이 완료했으나, 그 Checkpoint를 이어받아 추가 학습하자 Update 450에서 다시 Entropy 기반 Safety Stop에 도달했다.

- **첫 번째 실험:** 197 Updates / 12,608 Hands / Safety Stop / 최고 Validation 개선량 0 bb/100.
- **두 번째 실험, Pilot:** 100 Updates / 12,800 Hands / 정상 Update Limit 종료 / 최고 개선량 +98.25 bb/100.
- **두 번째 실험, Resume:** 추가 246 Updates / 31,488 Hands / Safety Stop / 최고 개선량 +413.375 bb/100 (Update 375).
- **두 번째 실험 누적:** 346 신규 Updates, 44,288 Hands. 최신 Update 450 정책에서는 전체 관찰 행동 대비 Fold 빈도가 50%를 넘었다.
- **중요한 제한:** 위 개선량은 **같은 100개 Validation Deck Groups를 반복 평가하여 선택한 상대 지표**다. 실제 승률, GTO Exploitability, 독립 Held-out Test 개선을 입증하지 않는다.

**현재 결론:** GPU·On-Policy Self-Play·Checkpoint/Resume의 동작이 관찰되었다. 그러나 동일한 학습 파이프라인의 장시간 정책 안정성은 입증되지 않았다. 기존 실행 폴더와 모델 파일을 보존한 채 Policy Collapse 원인을 조사해야 한다.

## 2. 관련 코드와 환경

| 구성 요소 | 구현 경로 | 역할 |
|---|---|---|
| Phase 15 실행 | `scripts/phase15_train.py` | 학습 CLI |
| Phase 15 장시간 학습기 | `src/holdem_ai/experiments/longrun.py` | Safety Stop, Resume, Checkpoint, Validation |
| Phase 13 Batch 학습 | `src/holdem_ai/experiments/batch_scaling.py` | 기존 PPO 학습 로직 호출 |
| Batched Self-Play | `src/holdem_ai/training/batched_collector.py` | 다중 HUNL Hand의 Batched Inference |
| PPO | `src/holdem_ai/training/ppo.py` | 정책·Critic 학습 |
| 행동 Abstraction | `src/holdem_ai/environment/bet_abstraction.py` | 7-way Action Head |
| Evaluation | `src/holdem_ai/evaluation/seed_blocked.py` | 4-layout Seed-Blocked 평가 |
| 기존 설계 문서 | `docs/PHASE15_LONG_RUN_TRAINING.md` | Phase 15 설계와 사용법 |

- 원본 모델: `runs/phase14_main/seed_52/ppo/best.pt`.
- 원본 Checkpoint의 학습 Update: **104**.
- CUDA Device: **NVIDIA GeForce GTX 1070** (사용자 로컬 PC).
- 본 문서는 **기존 HUNL PPO Phase 15** 기록이다. 나중에 추가한 `expand/rl_v2`의 2~6인 PPO와 혼동하지 말 것.
- 실험 중 기록된 PyTorch CUDA Peak Allocated Memory는 102~161 MiB 수준. 이는 CUDA 전체 장치 사용량이나 GPU 이용률이 아니다.

### 2.1 Phase 15 학습·선정 목표

학습용 Actor는 다음과 같이 행동을 샘플링한다.

$$
a_t\sim\pi_\theta(a\mid I_t),\qquad a_t\in\mathcal A_{\rm legal}(I_t).
$$

Phase 15의 Best 모델 선택은 두 고정 상대에 대한 Validation **개선량**을 동일 가중하여 산정한다.

$$
J_A(t)=\widehat J_{\rm val}(\pi_t,\pi_0),\qquad
J_R(t)=\widehat J_{\rm val}(\pi_t,\pi_{\rm Random})
$$

$$
\boxed{
S_t=\frac{1}{2}\left[
(J_A(t)-J_A(0))+(J_R(t)-J_R(0))
\right].
}
$$

여기서 $\pi_0$는 Phase 14에서 가져온 Frozen Initial Actor다. $S_t$는 Validation 모델 선택을 위한 휴리스틱이며 GTO/Nash 균형에 대한 정량적 지표가 아니다.

### 2.2 Safety Stop

코드 기본 설정:

$$
h_{\min}=0.45,\qquad k_{\max}=0.10,\qquad p=12.
$$

각 Update에서 다음 조건을 확인한다.

$$
\mathsf{warning}_t=
\mathbf 1\!\left[
H(\pi_t)<0.45\ \lor\
\widehat D_{\rm KL}(\pi_{\rm old},\pi_{\rm new})>0.10
\right].
$$

경고가 **12회 연속** 발생하면 `stop_reason=safety_stop`으로 종료한다. Entropy는 행동 확률의 집중도와 관련된 지표이지만, Legal Action 집합이나 상황 분포가 바뀌면 평균값도 달라질 수 있으므로 이것만으로 Policy Collapse를 증명하지 않는다.

## 3. 실험 A: 기존 설정의 최초 30분 시도

### 3.1 설정

```text
root              = runs/phase15_ppo_main
source_checkpoint = runs/phase14_main/seed_52/ppo/best.pt
device            = cuda
duration_minutes  = 30
hands_per_update  = 64
env_batch_size    = 32
minibatch_size    = 64
epochs            = 2
eval_every        = 100
eval_groups       = 100
final_groups      = 400
rollout_seed      = 400000000
```

### 3.2 결과

| 항목 | 실측 |
|---|---:|
| 시작 Update | 104 |
| 종료 Update | 301 |
| 추가 Updates | 197 |
| 추가 Hands | 12,608 |
| 실행 시간 | 191.406초 (3분 11초) |
| Entropy 경고 | 15 |
| Approx KL 경고 | 0 |
| 종료 원인 | `safety_stop` |
| Best Validation Update | 104 (원본) |
| Best Score | 0 bb/100 |
| CUDA Peak Allocated | 102.453 MiB |

Validation 개선량:

| Update | Score, bb/100 |
|---:|---:|
| 104 | 0 |
| 200 | -187.000 |
| 300 | -191.750 |
| 301 | -202.250 |

Entropy 관측 구간:

- Update 282: 0.754268
- Update 286: 0.432697 (임계값 이하)
- Update 289: 0.465965 (일시적으로 임계값 초과)
- Update 290~301: **12회 연속** 임계값 이하.
- Update 301: 0.201721, Approx KL 약 0, Critic Loss 0.000521.

따라서 30분 시간제한보다 **Entropy 연속 경고 한도**가 먼저 충족되었다. CUDA 충돌로 중지한 것은 아니다.

## 4. 실험 B: 안정성 설정으로 재시작하고 Resume

새 실험은 최초 실험의 `latest.pt`(Update 301)에서 시작하지 않고 **원본 Phase 14 Update 104 Checkpoint**에서 별도 경로로 시작했다.

### 4.1 변경점

| 설정 | 실험 A | 실험 B |
|---|---:|---:|
| Hands / Update | 64 | 128 |
| Env Batch Size | 32 | 32 |
| Minibatch | 64 | 128 |
| PPO Epochs | 2 | 1 |
| Rollout Seed | 400000000 | 440000000 |
| Entropy Threshold | 0.45 | 0.45 |
| Max Approx KL | 0.10 | 0.10 |

동시에 세 학습 설정을 변경했으므로 **어느 단일 변경이 안정성을 개선했는지 분리하여 식별할 수 없다.**

### 4.2 Pilot: 100 Updates

```powershell
uv run --extra gpu python -m scripts.phase15_train \`
  --algorithm ppo \`
  --source-checkpoint runs/phase14_main/seed_52/ppo/best.pt \`
  --root runs/phase15_ppo_stability_pilot \`
  --device cuda \`
  --duration-minutes 10 \`
  --max-updates 100 \`
  --hands-per-update 128 \`
  --env-batch-size 32 \`
  --minibatch-size 128 \`
  --epochs 1 \`
  --eval-every 25 \`
  --eval-groups 100 \`
  --final-groups 400 \`
  --rollout-seed 440000000
```

| 항목 | 실측 |
|---|---:|
| Update | 104 → 204 |
| 추가 Updates | 100 |
| 추가 Hands | 12,800 |
| 실행 시간 | 185.598초 (약 3분 6초) |
| Entropy/KL 경고 | 0 / 0 |
| 종료 원인 | `update_limit` |
| Best Update | 125 |
| Best Score | +98.25 bb/100 |
| CUDA Peak Allocated | 153.423 MiB |

**10분 전에 종료된 이유:** `--max-updates 100`이 `--duration-minutes 10`보다 먼저 충족되었다. 정상 종료다.

Pilot Validation:

| Update | Score, bb/100 |
|---:|---:|
| 104 | 0 |
| 125 | +98.250 |
| 150 | +40.125 |
| 175 | +98.000 |
| 200 | +64.875 |
| 204 | +7.875 |

### 4.3 Pilot Resume: 최대 10분 추가

```powershell
uv run --extra gpu python -m scripts.phase15_train \`
  --root runs/phase15_ppo_stability_pilot \`
  --resume \`
  --device cuda \`
  --duration-minutes 10
```

Resume는 `latest.pt`에서 **Actor, Critic, Optimizer 및 학습 카운터**를 이어받았다. 당시 `latest.pt`는 Update 204, `best.pt`는 Update 125였다. Best 모델이 아닌 **마지막 모델**에서 재개된다는 사실을 구분해야 한다.

| 항목 | 실측 |
|---|---:|
| 시작 Update | 204 |
| 종료 Update | 450 |
| 추가 Updates | 246 |
| 추가 Hands | 31,488 |
| 실행 시간 | 363.843초 (약 6분 4초) |
| 이번 Resume의 Entropy/KL 경고 | 12 / 0 |
| 종료 원인 | `safety_stop` |
| 실험 B 누적 추가 Updates | 346 |
| 실험 B 누적 신규 Hands | 44,288 |
| Best Update | **375** |
| Best Score | **+413.375 bb/100** |
| CUDA Peak Allocated | 161.464 MiB |
| Final 봉인 여부 | `false` |

Resume Validation:

| Update | Score, bb/100 |
|---:|---:|
| 225 | +214.625 |
| 250 | -37.375 |
| 275 | +408.000 |
| 300 | +310.875 |
| 325 | -199.750 |
| 350 | +71.250 |
| **375** | **+413.375** |
| 400 | -5.000 |
| 425 | +12.625 |
| **450** | **-520.000** |

**핵심 차이:** Update 375는 최고 Validation 모델이며, Update 450은 가장 마지막에 학습된 모델이다. 마지막 모델의 지표 악화와 Entropy 경고가 함께 관측되었다.

## 5. Action Distribution / Fold 중심 집중 현상

프로젝트의 7-way Action Head는 다음과 같다.

| Index | Action |
|---:|---|
| 0 | Fold |
| 1 | Check |
| 2 | Call |
| 3 | Raise Half Pot |
| 4 | Raise Pot |
| 5 | Raise Double Pot |
| 6 | All-in |

관측된 평가 Action Count는 `src/holdem_ai/evaluation/seed_blocked.py`의 `actor_actions` 순서에 따른다.

### 5.1 Update 375 vs Update 450

| 평가 상대 | Update 375 Fold / 모든 Actor Actions | Update 450 Fold / 모든 Actor Actions |
|---|---:|---:|
| Frozen Initial | 82 / 828 = **9.90%** | 314 / 565 = **55.58%** |
| Random Legal | 76 / 698 = **10.89%** | 268 / 493 = **54.36%** |

$$
\widehat p_{\rm fold,all}
=\frac{N(\mathrm{Fold})}{N(\mathrm{All\ Actor\ Actions})}.
$$

Fold가 가능한 상황만 조건화한 확률은 별도로 수집해야 한다.

$$
\widehat p_{\rm fold\mid legal}
=\frac{\#\{a=\mathrm{Fold}\}}{\#\{\mathrm{Fold}\in\mathcal A_{\rm legal}(I)\}}.
$$

현재 로그에는 후자의 분모가 포함되어 있지 않다. 따라서 **Fold 집중·Policy Collapse의 유력한 징후**로 해석하되, 아직 행동 마스킹과 인포스테이트별 정책 붕괴를 확정한 것은 아니다.

### 5.2 375 / 450 Validation 상세

| Update | 상대 | Mean bb/100 | 95% CI bb/100 |
|---:|---|---:|---|
| 375 | Frozen Initial | +790.750 | [-129.6283, +1711.1283] |
| 375 | Random Legal | +1372.500 | [+247.6094, +2497.3906] |
| 450 | Frozen Initial | +307.125 | [-769.7356, +1383.9856] |
| 450 | Random Legal | -10.625 | [-906.5941, +885.3441] |

설계상 Score는 상대별 절대 bb/100의 평균이 아니라 **초기 Actor 대비 개선량 평균**이다.

$$
S_{375}=\frac{790.75+36.00}{2}=413.375
$$

$$
S_{450}=\frac{307.125-1347.125}{2}=-520.000.
$$

평가당 100개 독립 Deck Group × Seat/Button 4개 레이아웃 = 상대별 400 Hands다. 그룹 내 4개 Hands는 종속되므로 표준오차는 **100개 Deck Group**을 통계 단위로 계산한다. 샘플 수 대비 구간이 넓어 강한 성능 결론은 낼 수 없다.

## 6. 왜 Approx KL은 낮은데 정책이 달라질 수 있는가?

`approx_kl`은 대체로 매 PPO Update 시점의 기존 정책 대비 변화를 나타낸다.

$$
D_{\rm KL}(\pi_{t-1}\Vert\pi_t)\approx0
\;\not\Rightarrow\;
D_{\rm KL}(\pi_{104}\Vert\pi_t)\approx0.
$$

작은 변화가 여러 번 누적되면 초기 정책과 크게 다른 행동 분포가 만들어질 수 있다. 실제 로그에서 KL 경고는 0회였지만 Fold 비율 변화와 Entropy 경고가 나타났다. 원인 후보로는 PPO 학습률/Entropy Regularization, 보상 또는 Critic Scale, On-Policy 수집 분포와 Legal Mask 변화, 정책의 Fold 편향이 있으며 **현 단계에서는 원인 확정 전**이다.

## 7. Checkpoint와 실험 재현성

현재 실험 폴더:

```text
runs/
├── phase15_ppo_main/
│   ├── latest.pt          # Update 301
│   ├── best.pt            # Update 104 (source-based)
│   ├── metrics.jsonl
│   ├── run.json
│   ├── state.json
│   └── summary.json
└── phase15_ppo_stability_pilot/
    ├── latest.pt          # Update 450
    ├── best.pt            # Update 375
    ├── metrics.jsonl
    ├── run.json
    ├── state.json
    └── summary.json
```

- 두 실행 모두 `sealed=false`: Final Held-out Evaluation 미실행.
- `latest.pt`는 **Resume 원천**, `best.pt`는 **Validation 선택 최고 모델**이다.
- `--resume`은 지정한 `root`에서 마지막 모델을 복원하며, Best Update에서 자동으로 재출발하지 않는다.
- `--duration-minutes`는 총 누적 시간이 아닌 **해당 호출에 추가되는 시간 예산**이다.
- `--max-updates`와 시간제한 중 먼저 충족되는 조건에서 끝난다. Entropy/KL Safety Stop은 그 전에 중단할 수 있다.
- 보고서만 Git에 포함한다. 학습 로그, Checkpoint, GUI, 미커밋 PPO v2 코드는 **이 보고서 Commit 대상이 아니다**.
- Local Checkpoint·실험 설정이 변경되거나 손상되지 않도록 원본 폴더를 보존하고 필요하면 별도 오프라인 백업한다.

## 8. 재현·진단용 PowerShell

### 8.1 두 실행의 최종 상태 확인

```powershell
Get-Content .\runs\phase15_ppo_main\summary.json
Get-Content .\runs\phase15_ppo_stability_pilot\summary.json
Get-Content .\runs\phase15_ppo_stability_pilot\state.json
```

### 8.2 마지막 PPO 지표 확인

```powershell
Get-Content .\runs\phase15_ppo_stability_pilot\metrics.jsonl |
  ForEach-Object { $_ | ConvertFrom-Json } |
  Where-Object { $_.kind -eq "update" } |
  Select-Object -Last 30 update, entropy, approx_kl, critic_loss
```

### 8.3 Best / Latest 파일 존재 여부

```powershell
Test-Path .\runs\phase15_ppo_stability_pilot\best.pt
Test-Path .\runs\phase15_ppo_stability_pilot\latest.pt
```

## 9. 권장 후속 연구: 학습 안정성

1. **먼저 관찰 강화:** Fold Legal Opportunity Count, Fold Conditional Rate, Street/Position별 행동 비율, Action Mask 크기별 Entropy, Actor Logit/확률 분포를 별도로 기록.
2. **원인 분리 Ablation:** Hands/Update, Epochs, Minibatch를 한 번에 바꾸지 않고 한 요소씩 통제; 반복 Seed 사용; 초기 Checkpoint 고정.
3. **PPO 강도 검증:** Learning Rate, Clip Epsilon, Entropy Coefficient, Gradient Norm, Value Target Scale 등을 개별 실험. 기존 Phase 15 CLI에서 일부 항목은 직접 변경 불가능하므로 코드 및 테스트 작업이 필요.
4. **붕괴 감시 개선:** Entropy 단독 대신 Legal-Conditioned Fold Rate와 Validation Performance를 함께 모니터링. 임계값 조정만으로 문제 해결을 선언하지 않기.
5. **평가 프로토콜:** Best 선정은 Validation에만 한정. 학습 및 모델 선택이 완전히 끝난 뒤에만 `phase15_finalize`를 실행해 독립 Final Test를 봉인. 반복적으로 Final 결과를 보고 튜닝하지 않기.
6. **5시간 학습 보류:** 단기 안정성·평가 분포가 충분히 확인되고 Safety Stop 빈도가 낮아진 뒤 별도의 장시간 실험 수행.
7. **6-Max 분리:** 이번 결과를 `expand/rl_v2` 또는 Tournament 전용 PPO의 성능 결과로 일반화하지 않기.

## 10. 기록 및 한계

- 실험 로그에는 Local 실행 일시의 신뢰할 수 있는 타임스탬프가 없으므로 문서 작성일과 실행일을 동일하다고 주장하지 않는다.
- 수집된 자료는 사용자 콘솔 출력에 의존하며 모든 실제 `metrics.jsonl` 및 Checkpoint Bytes를 직접 입수·재실행한 것은 아니다.
- 이번 Git 기록은 **실험 중간 보고서**이며, 코드 수정이나 모델 개선을 포함하지 않는다.
- 일반적인 100bb/100 임계값과 현재 HUNL200 BB 단위를 혼동하지 않는다. 모델은 200BB Stack에서 학습했으며 bb/100은 수익 단위 환산값이다.
- 관측된 Validation Score와 Entropy/Fold 징후가 인과 관계를 확정하는 것은 아니다.

---

**연구상 상태:** Phase 15 Checkpoint/Resume 및 CUDA 실행 확인, 장시간 안정성 문제 관측, Action Diagnostics·Optimizer 안정성 추가 조사 필요, 독립 Final Test 보류.