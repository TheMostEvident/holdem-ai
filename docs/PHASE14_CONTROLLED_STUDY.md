# Phase 14 — PPO vs Q-Boost 다중 Seed 통제 실험

> **이 Phase는 VRPO 원 논문의 충실한 재현이 아니다.** Phase 9의 PPO와 Phase 10의 VRPO-Lite Q-Boost 제어변량을 비교한다. 테스트 카드와 모델 선택용 Validation 카드를 엄격히 분리한다.

## 1. 목표 및 실험 단위

Phase 13 측정 결과에서는 GTX 1070 기준 `hands_per_update=64`, `env_batch_size=32`의 전체 학습 속도가 상대적으로 빨랐다(91.77 hands/sec, 각 케이스는 단 2회 측정). 따라서 Phase 14의 **시작 가설**로 이를 사용하지만 최적 조합이라고 확정하지 않는다.

각 학습 Seed $s\in\mathcal S$에 대해 학습군 $a\in\{\mathrm{PPO},\mathrm{QBoost}\}$를 별도로 생성한다.

$$
\theta_{s,0}^{\mathrm{PPO}}=\theta_{s,0}^{\mathrm{QB}}=\theta_{\mathrm{source}},\qquad
\phi_{s,0}^{\mathrm{PPO}}=\phi_{s,0}^{\mathrm{QB}}=\phi_{\mathrm{source}}.
$$

$$
\mathrm{Adam}_{s,0}^{\mathrm{PPO}}=\mathrm{Adam}_{s,0}^{\mathrm{QB}}=\mathrm{Adam}_{\mathrm{source}}.
$$

**공통 소스는 반드시 완전한 Phase 9 PPO Checkpoint** (`checkpoints/phase9_ppo.pt`)여야 한다. Phase 10 Q-Boost 체크포인트를 PPO 시작점으로 혼합하지 않는다. 체크포인트 파일의 SHA-256을 실험 Manifest에 기록하며 파일 수정 뒤 재개를 거부한다.

$$
N_{\mathrm{train}}=\lvert\mathcal S\rvert\times 2\times U\times H,
$$

- $U$: 각 알고리즘당 추가 업데이트 수, 기본 100.
- $H$: 업데이트당 학습 핸드 수, 기본 64.
- 각 업데이트는 최신 Actor로 수집한 On-Policy 데이터만 사용한다. 정책이 달라지면 게임 경로도 달라진다.
- 같은 학습 Seed의 두 알고리즘은 같은 카드 시드 범위 및 Shuffle Seed 일정을 사용하지만 **동일한 행동 경로가 보장되지는 않는다.**

## 2. 학습 목적함수

PPO는 기존 Phase 9 수식을 그대로 사용한다.

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid I_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid I_t)},
$$

$$
L_{\mathrm{PPO}}=\mathbb E_t\left[\min\left(\rho_t\hat A_t,\;\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].
$$

Phase 10 Q-Boost는

$$
\hat A_t^{\mathrm{QB}}=\frac{G_t-V_{\mathrm{old}}(s_t,I_t)-\beta(Q_{\mathrm{old}}(s_t,a_t)-V_{\mathrm{old}}(s_t,I_t))-\mu_A}{\sigma_A}
$$

및

$$
C_t(\theta)=\frac{\beta}{\sigma_A}\sum_a\left(\pi_\theta(a\mid I_t)-\pi_{\mathrm{old}}(a\mid I_t)\right)Q_{\mathrm{old}}(s_t,a)
$$

을 결합한 실험적 clipped surrogate를 사용한다. Phase 14에서는 이 손실을 **수정하지 않는다.** 비교의 처리 조건만 맞춘다.

## 3. Validation / Test 분리

- Validation: 기본 `seed = 12_000_000, ..., 12_000_039`, 40개의 덱 시드.
- Test: 기본 `seed = 22_000_000, ..., 22_000_199`, 200개의 덱 시드.
- Training: `100_000_000 + 1_000_000*s`에서 시작하는 Seed별 독립 주소 공간(최대 1백만핸드 미만).

각 덱 시드 $k$를 동일한 4개 좌석·Button 배치에서 사용한다.

$$
X_{s,a,k}=\frac{1}{4}\sum_{b=0}^1\sum_{j=0}^1R_{s,a,k,b,j}.
$$

**체크포인트 선택:** Validation의 고정 초기 Actor 상대 $\widehat J_{\mathrm{val}}$ 최대값을 기준으로 각 학습군의 `best.pt`를 선택한다. 초기 Actor 자기 자신도 후보이며, 무조건 학습된 체크포인트를 선택하지 않는다.

**최종 Test:** PPO와 Q-Boost의 학습이 **모든 Seed에서 끝난 이후에만** 한 번 진행한다. 같은 Test Seed, 같은 고정 초기 Actor와 Random Agent를 상대한다. 또한 Q-Boost vs PPO 직접 대결도 수행한다.

Test Seed의 결과는 업데이트, Beta, Learning Rate, 최고 Validation 체크포인트 선정에 사용하지 않는다. Test 결과를 보고 다시 알고리즘 설정을 조정했다면 그 결과는 엄밀히는 최종 독립 테스트가 아니므로, 다음 실험을 위한 새로운 Test 카드 은행을 만들어야 한다.

## 4. 통계 추정

같은 학습 Seed $s$의 두 모델에 대해, 같은 고정 초기 Actor를 상대할 때의 차이는

$$
D_{s,k}=X_{s,\mathrm{QB},k}-X_{s,\mathrm{PPO},k}
$$

이고 Seed별 효과는

$$
\hat\Delta_s=\frac{100}{K}\sum_{k=1}^KD_{s,k}\quad[\mathrm{bb/100}].
$$

Phase 12의 `compare_results()`로 같은 카드 그룹의 **대응 차이**를 계산한다.

학습 Seed를 독립 반복 단위로 한 총 효과는

$$
\bar\Delta=\frac{1}{S}\sum_{s=1}^S\hat\Delta_s.
$$

$$
\mathrm{SE}_{\mathrm{seed}}(\bar\Delta)=\frac{\mathrm{sd}(\hat\Delta_1,\ldots,\hat\Delta_S)}{\sqrt S}.
$$

$$
\mathrm{CI}_{95\%}\approx \bar\Delta\pm t_{S-1,0.975}\,\mathrm{SE}_{\mathrm{seed}}.
$$

$S=1$이면 Seed 간 신뢰구간은 계산하지 않는다. 실험 카드 은행을 공유하므로 이 신뢰구간은 **고정된 Test 카드 조건에서 학습 Seed 간 변동성**을 표현한다. 전체 카드 분포와 학습 Seed의 동시 불확실성을 완전히 추정하는 값은 아니다. Seed가 3개뿐일 때 통계 검정력도 낮다.

Phase 14 보고서의 주요 결과는:

- `primary_qboost_minus_ppo_vs_initial`: 같은 초기 Actor 상대 대응 차이, Seed 간 평균.
- `secondary_qboost_minus_ppo_vs_random`: 같은 Random Agent 상대 대응 차이.
- `secondary_direct_headtohead_qboost_bb100`: Q-Boost가 PPO를 직접 상대했을 때의 bb/100.
- `training_diagnostics`: Actor Entropy, Critic Loss, Approx KL, Q-Boost logit-gradient **Proxy** 분산 비율, Fold/Check의 합법 시도 대비 선택률, 처리량.

이 분석은 **GTO Exploitability**를 제공하지 않으며, 원 논문의 VRPO 성능을 재현했다는 주장도 하지 않는다.

## 5. PowerShell 사용

### 먼저 검증

```powershell
uv run --extra dev --extra gpu python -m pytest -q -k phase14
uv run --extra dev --extra gpu python -m pytest -q
```

### 작은 GTX 1070 Pilot

```powershell
uv run --extra gpu python -m scripts.phase14_study --source-checkpoint checkpoints/phase9_ppo.pt --root runs/phase14_pilot --device cuda --seeds 41 --updates 2 --hands-per-update 64 --env-batch-size 32 --minibatch-size 64 --epochs 2 --eval-every 1 --validation-groups 10 --test-groups 20 --arm-duration-minutes 10
```

Pilot에서 Test가 사용되므로 Pilot은 별개 실험으로 보관한다. 본 실험에서는 다른 경로와 독립 시드를 유지한다.

### 본격적인 다중 Seed 통제 실험

```powershell
uv run --extra gpu python -m scripts.phase14_study --source-checkpoint checkpoints/phase9_ppo.pt --root runs/phase14_main --device cuda --seeds 51,52,53 --updates 100 --hands-per-update 64 --env-batch-size 32 --minibatch-size 64 --epochs 2 --eval-every 25 --validation-groups 100 --test-groups 300 --arm-duration-minutes 45 --validation-seed 12010000 --test-seed 22010000
```

실험당 각 알고리즘에 추가할 업데이트 수가 동일하고, 모든 학습이 끝나면 최종 테스트를 자동 계산한다. **`--arm-duration-minutes`는 각 학습군 호출에 적용되는 시간 예산**이다. 제한 시간에 걸리면 `status=partial`을 반환하고 Test를 수행하지 않는다.

### 동일 조건으로 재개

```powershell
uv run --extra gpu python -m scripts.phase14_study --source-checkpoint checkpoints/phase9_ppo.pt --root runs/phase14_main --device cuda --seeds 51,52,53 --updates 100 --hands-per-update 64 --env-batch-size 32 --minibatch-size 64 --epochs 2 --eval-every 25 --validation-groups 100 --test-groups 300 --arm-duration-minutes 60 --validation-seed 12010000 --test-seed 22010000 --resume
```

`--resume`은 저장된 `study.json`과 Source SHA-256을 검사한다. 학습 시간 예산은 재개 호출에서 새로 지정할 수 있고, 나머지 실험 설정은 변하지 않아야 한다.

## 6. 결과 폴더

```text
runs/phase14_main/
├── study.json                # 실험 사전 등록 설정 및 원본 SHA-256
├── summary.json              # Seed 간 통계 및 최종 보고서
├── seed_51/
│   ├── ppo/
│   │   ├── latest.pt         # 완료 업데이트의 최신 모델 및 Adam
│   │   ├── best.pt           # 최고 Validation 모델 및 Adam
│   │   ├── run.json          # 학습 설정 및 출발 가중치 지문
│   │   ├── state.json        # Validation 최고 점수/업데이트
│   │   └── metrics.jsonl     # 업데이트/Validation/Entropy 경고
│   ├── qboost/
│   │   └── ...
│   └── test_results.json     # 두 알고리즘 완료 후 생성
└── seed_52/ ...
```

코드를 Push할 때 `runs/` 및 `checkpoints/*.pt`를 Stage하지 말 것.

## 7. 주의 사항

- Test에서 양수 결과를 보였다는 것만으로 실전 고수 또는 GTO에 근접했다고 말할 수 없다.
- Self-Play 정책과 카드 배분이 달라지면 샘플 경로도 달라진다.
- 100 Updates × 64 Hands는 한 학습 Seed·알고리즘당 **6,400핸드**에 불과하다. 정책 강도 평가로는 작은 예비 실험이다.
- 기존 Phase 11 또는 Phase 13 Q-Boost 체크포인트는 **단일 학습 이어서 수행하기에는 적합하지만**, Q-Boost/PPO의 매칭된 초기 비교 소스로 사용하지 않는다. 새로운 비교는 Phase 9 원본부터 시작한다.
- `tqdm` 진행률은 알고리즘/Seed별로 표시된다. 전체 6개 학습군의 합산 ETA는 제공하지 않는다.
