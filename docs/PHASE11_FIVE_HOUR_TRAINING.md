# Phase 11 — GTX 1070 5시간 실험과 Batched On-Policy Self-Play

> Source of truth: `TheMostEvident/holdem-ai`, Phase 10 (`main`, snapshot `e1fbfcac82231024df06e1206936e2fdaf32e3d6`).  
> **Phase 11은 VRPO 논문의 원본 구현이 아닌 VRPO-Lite의 학습 인프라 구현이다.**  
> 기존 Phase 1~10 파일은 바꾸지 않으며, CUDA 테스트는 사용자의 Windows PC에서 검증해야 한다.

## 1. 목표와 구현 범위

1. 독립적인 HUNL200 인스턴스를 **한 프로세스 안에서 여러 개 활성화**하고, 현재 행동 대기 중인 게임들의 Actor 관측을 GPU에 묶어 추론한다.
2. Phase 9 PPO 또는 Phase 10 Q-Boost를 **동일한 기존 최적화 함수**로 학습한다. 데이터는 매 업데이트 **최신 정책으로 새로 수집**하며 Replay Buffer를 재사용하지 않는다.
3. **시간 예산, 업데이트 개수, 안전한 저장, 재개, JSONL 진단 로그**를 제공한다.
4. RandomLegalPolicy와 **고정 초기 Actor snapshot**을 상대로 seat/button을 균등하게 평가한다.
5. 독립적인 PPO/Q-Boost 실험의 로그를 비교한다.

**아직 구현하지 않은 것:** 멀티프로세스 Rollout Worker 4개 + 공용 GPU 서버, 별도 Q 정확도 검증 세트, GTO Exploitability, 카드 재배분을 강제 동기화하는 Duplicate-Poker 평가, LR Schedule/AMP, 원본 VRPO 충실 재현, 5시간 학습을 이 환경에서 직접 실행하는 것.

## 2. Batched Self-Play의 정확성

동시에 활성화된 게임의 집합을 $\mathcal{E}_t$라고 할 때, 게임 $e$의 행동 플레이어 $i_e$가 관측한 $I_{e,t}^{i_e}$만 묶는다.

$$
X_t=\operatorname{Batch}\left(E_A(I_{e,t}^{i_e}):e\in\mathcal{E}_t\right)
$$

$$
\pi_{\theta_{k}}(a\mid I_{e,t}^{i_e})
=\operatorname{MaskedSoftmax}\left(f_{\theta_k}(X_t)_e\right)_a
$$

동일한 업데이트 $k$가 끝날 때까지 정책 $\theta_k$를 고정한 다음, 모든 게임을 종료하고 최종 bb 수익을 귀속한다.

$$
G_{e,t}^{i_e}=R_e^{i_e},\qquad
\sum_{i=0}^{1}R_e^i=0
$$

개별 엔진 인스턴스가 각자의 카드와 Game State를 보관한다. **Actor에는 InformationState만 전달**하며 `training_state()`는 오직 Critic 인코딩에 사용한다. 각 핸드는 독립된 RNG와 Seed (`base_seed + global_hand_index`)를 가진다. 미완료 핸드는 한 배치 사이클에서 한 번만 행동한다.

인코딩에는 전체 히스토리가 포함되므로 256 토큰 한도를 초과하면 오류를 내고, 임의로 과거 기록을 잘라내지 않는다.

## 3. PPO / Q-Boost 목적함수

Phase 9의 기본 Advantage는

$$
\widehat{A}^{\mathrm{PPO}}_t=G_t^{i_t}-V_{\mathrm{old}}^{i_t}(s_t,I_t^{i_t})
$$

이며 Actor는 PPO Clipping을 사용한다.

$$
\rho_t(\theta)=\exp\left(\log\pi_\theta(a_t\mid I_t)-\log\pi_{\mathrm{old}}(a_t\mid I_t)\right)
$$

$$
L^{\mathrm{CLIP}}=\mathbb E_t\left[\min\left(\rho_t\widehat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\widehat A_t\right)\right]
$$

Phase 10에서는 관측 행동에 대한 Q 오차 보정과 $\sum_a\pi_\theta(a\mid I)Q_{\mathrm{old}}(s,a)$ 해석적 항을 결합한다. Phase 10 결과에서 `logit_variance_ratio \approx 1`이었으므로 **분산 감소와 전략 성능 향상은 검증 목표**이다.

## 4. GTX 1070 환경 적용

기존 프로젝트 폴더에 ZIP을 `holdem-ai-phase11-training`으로 압축 해제한 다음 PowerShell에서:

```powershell
Copy-Item .\holdem-ai-phase11-training\holdem_agent_phase11 .\ -Recurse -Force
Copy-Item .\holdem-ai-phase11-training\tests\test_phase11_*.py .\tests\ -Force
Copy-Item .\holdem-ai-phase11-training\scripts\phase11_*.py .\scripts\ -Force
Copy-Item .\holdem-ai-phase11-training\docs\PHASE11_FIVE_HOUR_TRAINING.md .\docs\ -Force

uv run --extra dev --extra gpu python -m pytest -q -k phase11
uv run --extra dev --extra gpu python -m pytest -q
```

기존 `torch==2.6.0+cu118`과 GTX 1070 (Compute Capability 6.1)을 사용한다. CUDA를 지원하지 않는 실행 환경에서는 `--device cpu`로 기능 테스트가 가능하다.

## 5. 실제 GPU 배치 Benchmark (학습 없이)

```powershell
uv run --extra gpu python -m scripts.phase11_benchmark --device cuda --hands 200 --batch-sizes 1,4,8,16
```

확인해야 할 값은 `hands_per_second`, `decisions_per_second`, `peak_cuda_allocated_mb`이다. Phase 6 CPU Random 정책의 약 2,896 hands/sec와는 **동일한 작업량이 아니므로** 비교하면 안 된다. 앞선 Phase 9/10 CUDA 실험의 약 30~45 hands/sec가 비교 대상에 더 가깝지만, 배치 크기 및 시퀀스 길이에 따라 결과가 달라진다. 처음에는 `--env-batch-size 8`을 제안하고 **실측 후 결정**한다.

## 6. 첫 2 업데이트 Pilot — Phase10으로부터 이어가기

**반드시 유일한 출력 경로**를 사용한다. 이전 파일을 덮어쓰면 데이터가 유실될 수 있어, Phase 11은 기본적으로 기존 Run 경로/Checkpoint를 덮어쓰지 않는다.

```powershell
uv run --extra gpu python -m scripts.phase11_train --algorithm qboost --device cuda --from-phase10 checkpoints/phase10_qboost.pt --duration-minutes 10 --max-updates 2 --env-batch-size 8 --eval-every 1 --eval-hands 40 --checkpoint checkpoints/phase11_pilot.pt --log-dir runs/phase11_pilot
```

기존 Phase 10 체크포인트의 **Actor, Critic, Adam 상태, 업데이트 횟수, PPOConfig**를 유지한다. 예를 들어 Phase 10에서 4회 업데이트한 체크포인트라면 첫 추가 업데이트는 5가 된다. **`--hands-per-update`는 기존 체크포인트로부터 시작할 때 적용되지 않고 저장된 PPOConfig가 우선한다.**

중단한 경우의 안전한 Resume:

```powershell
uv run --extra gpu python -m scripts.phase11_train --algorithm qboost --device cuda --resume checkpoints/phase11_pilot.pt --duration-minutes 10 --max-updates 2 --env-batch-size 8 --eval-every 1 --eval-hands 40 --log-dir runs/phase11_pilot
```

`--max-updates`는 **이 명령에서 새로 추가할 업데이트 수**이다. `--duration-minutes`도 **이번 실행 세션의 시간 예산**이고, 원래 실험 전체 누적 시간이 아니다.

## 7. 전체 5시간 학습

Pilot 성능 측정이 끝난 뒤 GTX 1070에서 다음을 수행한다.

### 7.1 기존 Phase 10 모델을 계속 학습

```powershell
uv run --extra gpu python -m scripts.phase11_train --algorithm qboost --device cuda --from-phase10 checkpoints/phase10_qboost.pt --duration-minutes 300 --env-batch-size 8 --eval-every 25 --eval-hands 200 --checkpoint checkpoints/phase11_long_qboost.pt --log-dir runs/phase11_long_qboost
```

`--from-phase10`은 **기존 Phase 10 체크포인트를 읽되 수정하지 않는다.** 시간 예산이 만료되면 마지막 완료 업데이트의 상태가 Phase 11 Checkpoint에 보존된다.

### 7.2 공정한 PPO vs Q-Boost 비교 (별도의 fresh run)

같은 초기 가중치, 동일한 PPOConfig와 Seed를 사용하는 **서로 다른 두 실험**이다. 두 모델이 학습하면서 다른 행동을 할 수 있으므로 Rollout은 각자 자신의 최신 정책에서 수집된다.

```powershell
uv run --extra gpu python -m scripts.phase11_train --algorithm ppo --device cuda --duration-minutes 30 --max-updates 1000 --hands-per-update 32 --env-batch-size 8 --eval-every 25 --eval-hands 200 --seed 42 --checkpoint checkpoints/phase11_fresh_ppo.pt --log-dir runs/phase11_fresh_ppo

uv run --extra gpu python -m scripts.phase11_train --algorithm qboost --device cuda --duration-minutes 30 --max-updates 1000 --hands-per-update 32 --env-batch-size 8 --eval-every 25 --eval-hands 200 --seed 42 --checkpoint checkpoints/phase11_fresh_qboost.pt --log-dir runs/phase11_fresh_qboost
```

이 경우 두 Run은 각 30분 예산이다. 5시간 동안 각각 학습시키려면 각 명령의 `--duration-minutes`를 300으로 변경하면 된다. **전체 비교 시간은 두 실험의 합계**라는 점에 유의한다.

```powershell
uv run --extra gpu python -m scripts.phase11_compare --ppo-log runs/phase11_fresh_ppo/metrics.jsonl --qboost-log runs/phase11_fresh_qboost/metrics.jsonl
```

## 8. 산출물 및 안전장치

```
runs/phase11_pilot/
├── run.json          # immutable initial training parameters
├── anchor.pt         # frozen first-policy actor
├── metrics.jsonl     # append-only update/evaluation/status records
└── summary.json      # most recent invocation summary

checkpoints/phase11_pilot.pt  # last completed update, atomic save
```

- `metrics.jsonl`: 1줄 = JSON 객체. VS Code, Python, Pandas로 처리 가능.
- `evaluation`: `opponent=random` 또는 `frozen_anchor`의 seat/button-balanced bb/100과 대략적인 95% CI를 기록한다.
- 각 checkpoint는 **임시 파일 저장 후 `os.replace`**로 교체한다. 업데이트마다 저장하기 때문에 디스크 쓰기 빈도가 증가하지만 복구 손실을 줄인다.
- 업데이트 중 `Ctrl+C`를 누르면 RAM 내부 모델이 부분적으로 바뀔 수 있다. **Checkpoint는 직전 완료 업데이트 상태만** 보존하고 Resume 시 그 버전을 다시 로드한다.
- Crash가 `checkpoint save`와 `metrics.jsonl append` 사이에 발생하면 로그의 마지막 Update 행이 누락될 수 있다. Resume 시 checkpoint의 `completed_updates`를 신뢰하며 `resume` 행을 남긴다.
- 시간 제한은 핸드 중간이 아닌 **업데이트·평가 경계**에 적용한다. 해당 작업의 수행 시간이 길면 설정한 예산을 약간 초과할 수 있다.
- `run.json`과 Checkpoint의 알고리즘 및 하이퍼파라미터가 일치하는지 Resume에서 검증한다.
- Checkpoint에는 GPU/CPU Torch Model State와 Adam 상태가 포함되므로 **신뢰할 수 있는 자체 파일만** 로드한다.

PowerShell에서 Git에 체크포인트/로그가 들어가지 않도록 `.gitignore`의 `checkpoints/`, `runs/`, `.venv/` 항목을 확인한다. 기존 파일을 모두 추가하는 `git add .`는 권장하지 않는다.

## 9. 평가 해석의 한계

`mean_bb100`은 다음과 같다.

$$
\widehat\mu_{\mathrm{bb/100}}=\frac{100}{N}\sum_{h=1}^{N}R_h^{\mathrm{BB}}
$$

평가에서 Actor Seat 0/1과 BTN/BB를 각각 동일 횟수로 사용하지만 **Duplicate-Poker Paired Deck**을 강제하지는 않는다. 고정 `seed`도 정책이 바뀌면 액션 경로가 달라져 완전한 paired counterfactual이 되지는 않는다.

$$
\Delta\widehat\mu=\widehat\mu_{\mathrm{QBoost}}-\widehat\mu_{\mathrm{PPO}}
$$

위 차이가 양수여도 200핸드 수준의 Random/Anchor 경기만으로 VRPO가 우수하다고 말할 수 없다. $\rho_{\mathrm{var}}$ 또한 **로그잇 Gradient의 Proxy**이며 전체 파라미터의 Gradient 분산이 아니다. 다양한 Seed, 훨씬 더 큰 평가 표본, 스냅샷 리그, 더 강한 상대, exploitability 관련 별도 검증이 후속 과제다.

## 10. Git 커밋

사용자 PC 테스트가 끝난 후에만:

```powershell
git add holdem_agent_phase11/ tests/test_phase11_*.py scripts/phase11_*.py docs/PHASE11_FIVE_HOUR_TRAINING.md
git diff --cached --stat
git commit -m "Implement Phase 11 batched PPO and QBoost training infrastructure"
git push origin main
```
