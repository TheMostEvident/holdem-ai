# Source documentation volume 3/7

[Final integrated technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md) · Lossless original report content from `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`.

## docs/PHASE10_QBOOST_PPO.md

Original SHA `4cf6a06aa529540ee8b19d34c1aebc3413ef1e57` (9382 bytes).

<a id="md-030"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE10_QBOOST_PPO.md" sha="4cf6a06aa529540ee8b19d34c1aebc3413ef1e57" -->

````````markdown
# Phase 10: Experimental VRPO-Lite Q-Boosting PPO

> Project: HUNL200, `TheMostEvident/holdem-ai`, Phase 9 as base.
>
> This is **an independently constructed doubly-corrected policy-gradient control-variate PPO** inspired by Q-Boosting, not a claim of reproducing every derivation or experiment in the original VRPO paper. The comparison must decide whether the new estimator helps.

## 1. Environment and existing interfaces

- Phase 6: two-seat HUNL200 environment, normalized rewards in BB, `InformationState` and opt-in privileged `CriticState`.
- Phase 7: seven legal actions, Actor tokens and Critic tokens, masked batch encodings.
- Phase 8: decentralized Transformer Actor and centralized zero-sum `Q(s,a)` network.
- Phase 9: Monte-Carlo PPO, observed-action Huber Q regression, optimizer checkpointing and seat/button-balanced RandomLegalPolicy evaluation.
- Phase 10 changes no existing source files; it imports the above modules.

All rewards are per-hand **BB**, not bb/100. There is no temporal discount within a hand.

## 2. Why a naive $G-Q(s,a)$ substitution is wrong

For the acting seat $i=i_t$, let $s_t$ be the privileged training state, $I_t$ the actor observation, and $a_t$ its sampled action. Denote the frozen old policy by $\pi_0$ and the frozen centralized Q model by $\hat Q_0$.

$$
G_t^{i_t}=R_{i_t}, \qquad R_0+R_1=0.
$$

The Phase-9 action-independent baseline is

$$
\hat V_0(s_t,I_t)=\sum_{a\in\mathcal A(I_t)}\pi_0(a\mid I_t)\hat Q_{0,i_t}(s_t,a).
$$

The Phase-9 raw advantage is

$$
A_t^{\mathrm{MC}}=G_t^{i_t}-\hat V_0(s_t,I_t).
$$

Simply changing it to $G_t-\hat Q_0(s_t,a_t)$ and doing PPO **omits the action-dependent Q contribution** from the desired policy gradient. The correct control variate needs a separate analytic action sum.

## 3. Doubly-corrected Q control variate

Set a fixed $\beta\in[0,1]$, chosen **before collecting the rollout**.

$$
\delta_t^{(\beta)}=
G_t^{i_t}-\hat V_0(s_t,I_t)
-\beta\left[\hat Q_{0,i_t}(s_t,a_t)-\hat V_0(s_t,I_t)\right].
$$

Let the normalization constants from the **same unboosted on-policy rollout** be

$$
\mu_A=\frac1N\sum_t A_t^{\mathrm{MC}},\qquad
\sigma_A=\max\left(\sqrt{\frac1N\sum_t(A_t^{\mathrm{MC}}-\mu_A)^2},10^{-6}\right).
$$

**Do not independently standardize the residual and analytic parts.** Doing so would destroy their cancellation at the old policy.

$$
\widetilde\delta_t^{(\beta)}=
\frac{\delta_t^{(\beta)}-\mu_A}{\sigma_A}.
$$

For a candidate policy $\pi_\theta$, define the observed importance ratio

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid I_t)}{\pi_0(a_t\mid I_t)}.
$$

The surrogate objective (MAXIMIZED) is

$$
\begin{aligned}
J_{\mathrm{QBoost}}(\theta)
=\frac1N\sum_{t=1}^N \bigg[&
\min\Big(r_t\widetilde\delta_t^{(\beta)},
\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\widetilde\delta_t^{(\beta)}\Big)\\
&+\frac\beta{\sigma_A}\sum_{a\in\mathcal A(I_t)}
\left(\pi_\theta(a\mid I_t)-\pi_0(a\mid I_t)\right)
\hat Q_{0,i_t}(s_t,a)\bigg].
\end{aligned}
$$

The Actor minimizes $-J_{\mathrm{QBoost}}-c_H H(\pi_\theta)$, and the Critic continues to minimize observed-action Huber regression

$$
\mathcal L_Q=\mathbb E_t\left[\mathrm{Huber}\left(\hat Q_{\phi,i_t}(s_t,a_t)-G_t^{i_t}\right)\right].
$$

The analytic Q and old policy are **frozen before all epochs of the update**; the Critic changes only in its own optimizer and does not send privileged observations into Actor.forward.

### Old-policy policy-gradient identity

Condition on $(s_t,I_t)$ and suppose $G$ is an unbiased return sample conditional on each chosen action. Write $Q^\star(a)=\mathbb E[G\mid s_t,I_t,a]$. Then for $\beta=1$,

$$
\begin{aligned}
&\mathbb E_{a\sim\pi_0}
\left[\nabla_\theta\log\pi_\theta(a\mid I_t)
(G-\hat Q_0(a))\right]_{\theta=\theta_0}
+\sum_b \nabla_\theta\pi_\theta(b\mid I_t)\big\vert_{\theta_0}\hat Q_0(b)\\
&=\sum_a \nabla_\theta\pi_\theta(a\mid I_t)\big\vert_{\theta_0}
\left(Q^\star(a)-\hat Q_0(a)+\hat Q_0(a)\right)\\
&=\sum_a \nabla_\theta\pi_\theta(a\mid I_t)\big\vert_{\theta_0}Q^\star(a).
\end{aligned}
$$

The same identity holds for all $\beta\in[0,1]$, with the action-independent baseline making zero expected contribution. Subtracting fixed $\mu_A$ and dividing by fixed $\sigma_A$ rescales the gradient. PPO clipping introduces the usual surrogate bias after moving away from $\theta_0$; this is **not an unbiased estimator at arbitrary $\theta$**.

### Baseline limiting case

$$
\beta=0\implies \widetilde\delta_t
=\frac{A_t^{\mathrm{MC}}-\mu_A}{\sigma_A},\qquad
J_{\mathrm{QBoost}}=J_{\mathrm{PPO}}.
$$

Our tests verify that this matches the existing Phase-9 clipped objective and that the automatic gradient with complete action enumeration satisfies the identity above, even when the auxiliary Q values are inaccurate.

## 4. What is and is not evaluated

`qboost_diagnostics` reports:

- `mc_advantage_variance`: empirical variance of raw $G-\hat V$;
- `boosted_residual_variance`: empirical variance of $\delta^{(\beta)}$ (not a total policy-gradient variance);
- `logit_gradient_proxy_variance_ppo`: sum of per-coordinate empirical variance of $(e_{a_t}-\pi_0)\widetilde A_t$;
- `logit_gradient_proxy_variance_qboost`: the same **logit-gradient proxy**, including the analytic Q correction;
- `logit_variance_ratio`: boosted-to-baseline proxy ratio. Values below 1 mean lower *proxy* variance on this sample, not a guarantee of lower Transformer parameter-gradient variance.

$$
g_t^{\mathrm{PPO,logit}}=(e_{a_t}-\pi_0)\widetilde A_t
$$

$$
g_t^{\mathrm{QB,logit}}=(e_{a_t}-\pi_0)\widetilde\delta_t^{(\beta)}
+\frac\beta{\sigma_A}\pi_0\odot(\hat Q_0-\hat V_0).
$$

These are per-state logits, not the full neural-network parameter gradients. The result can be higher or lower variance; **do not claim variance reduction without measurement**. With a poorly trained Q network, improvements may be negligible or negative.

`--compare` initializes PPO and QBoost from identical weights and seeds. On update 1 they can start from identical behavior; after divergence each uses its **own on-policy rollout**, though deck seed and seat/button schedule remain matched. Reusing QBoost samples to optimize a divergent PPO control without correcting for behavior policy would not be a valid on-policy comparison. Post-training performance uses the Phase-9 seat/button-balanced, unpaired RandomLegalPolicy evaluation, with wide confidence intervals for tiny samples. It is neither GTO nor exploitability measurement.

## 5. Install on Windows PowerShell

After extracting `holdem-ai-phase10-qboost.zip` as `holdem-ai-phase10-qboost` inside your repository root:

```powershell
Copy-Item .\holdem-ai-phase10-qboost\holdem_agent_phase10 .\ -Recurse -Force
Copy-Item .\holdem-ai-phase10-qboost\tests\test_phase10_*.py .\tests\ -Force
Copy-Item .\holdem-ai-phase10-qboost\scripts\phase10_train.py .\scripts\ -Force
Copy-Item .\holdem-ai-phase10-qboost\docs\PHASE10_QBOOST_PPO.md .\docs\ -Force
```

Test Phase 10 and the entire repository:

```powershell
uv run --extra dev --extra gpu python -m pytest -q -k phase10
uv run --extra dev --extra gpu python -m pytest -q
```

First GTX 1070 training and comparison:

```powershell
uv run --extra gpu python -m scripts.phase10_train --device cuda --updates 2 --hands-per-update 16 --epochs 2 --minibatch-size 32 --eval-hands 20 --compare --beta 1 --checkpoint checkpoints/phase10_qboost.pt --ppo-checkpoint checkpoints/phase10_ppo_control.pt
```

Continue with **two additional** QBoost updates:

```powershell
uv run --extra gpu python -m scripts.phase10_train --device cuda --resume checkpoints/phase10_qboost.pt --updates 2 --eval-hands 20
```

Alternatively, warm-start QBoost from your **own trusted** Phase-9 model checkpoint (keeps existing Actor, Critic, both Adam optimizers, and update counter):

```powershell
uv run --extra gpu python -m scripts.phase10_train --device cuda --from-phase9 checkpoints/phase9_ppo.pt --updates 2 --eval-hands 20 --beta 1 --checkpoint checkpoints/phase10_from9.pt
```

You cannot use `--compare` with `--resume` or `--from-phase9`, because that would not be an identical matched-start comparison. The `--beta` flag is used for new runs and Phase-9 warm-start; resuming loads saved beta instead.

For a more informative but longer run, increase the training hands and evaluation to multiples of four, e.g. `--hands-per-update 64 --eval-hands 200`. Even 200 hands is small by poker standards.

**Checkpoints are data artifacts, not source.** Do not `git add .` while `checkpoints/` or extracted ZIP directories are present. Never unpickle or load unknown model checkpoints.

## 6. Tests and limitations

The isolated test environment includes copied Phase 6-9 interfaces with a mock Phase-3 game engine, not the user's actual Windows poker engine. Phase 10 new tests check: masking, input privacy, analytic policy-gradient identity, beta-zero equivalence, finite gradients, optimizer parameter changes, checkpoint/resume, Phase-9 checkpoint migration, CLI comparison, and guarded CUDA path. The **actual** Phase 1-9 regression and GTX 1070 tests must run locally.

Other open limitations: unoptimized per-decision inference, no exact per-action counterfactual labels, no GAE, no fully reproduced VRPO algorithm, incomplete historical exact Call/All-in amounts in Phase 6, small self-play rollout samples and uncertain evaluation returns. QBoost performance is an empirical question.

````````

<!-- END_MD_SOURCE path="docs/PHASE10_QBOOST_PPO.md" -->

## docs/PHASE11_FIVE_HOUR_TRAINING.md

Original SHA `301c3c879039e52d7d9bacbf318441c73f9b34f9` (11162 bytes).

<a id="md-031"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE11_FIVE_HOUR_TRAINING.md" sha="301c3c879039e52d7d9bacbf318441c73f9b34f9" -->

````````markdown
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
X_t=\mathrm{Batch}\left(E_A(I_{e,t}^{i_e}):e\in\mathcal{E}_t\right)
$$

$$
\pi_{\theta_{k}}(a\mid I_{e,t}^{i_e})
=\mathrm{MaskedSoftmax}\left(f_{\theta_k}(X_t)_e\right)_a
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
L^{\mathrm{CLIP}}=\mathbb E_t\left[\min\left(\rho_t\widehat A_t,\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\widehat A_t\right)\right]
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

````````

<!-- END_MD_SOURCE path="docs/PHASE11_FIVE_HOUR_TRAINING.md" -->

## docs/PHASE11_PROGRESS.md

Original SHA `332852a9115e9283137c226466f0862b30115990` (3865 bytes).

<a id="md-032"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE11_PROGRESS.md" sha="332852a9115e9283137c226466f0862b30115990" -->

````````markdown
# Phase 11 — Live progress, ETA, remaining time

This is a **small additive patch** atop Phase 11 (tested 304 passed on user's Windows / CUDA). It changes **only** the existing `scripts/phase11_train.py` and adds `holdem_agent_phase11/progress.py`, `tests/test_phase11_progress.py`. No learning code, network parameters, checkpoints or JSONL formats are changed.

## Installation (PowerShell, project root)

Unzip this patch into `holdem-ai-phase11-progress` (not directly into the repository root), then run:

```powershell
Copy-Item .\holdem-ai-phase11-progress\holdem_agent_phase11\progress.py .\holdem_agent_phase11\ -Force
Copy-Item .\holdem-ai-phase11-progress\scripts\phase11_train.py .\scripts\ -Force
Copy-Item .\holdem-ai-phase11-progress\tests\test_phase11_progress.py .\tests\ -Force
Copy-Item .\holdem-ai-phase11-progress\docs\PHASE11_PROGRESS.md .\docs\ -Force
```

Existing commands work without flags. To enable a native `tqdm` bar (optional; without it a dependency-free progress meter is shown), install with:

```powershell
uv add tqdm
```

Do not install or modify dependencies just to use the fallback progress display.

## CLI examples

```powershell
# Continue an existing 15-minute / 200-update pilot.
uv run --extra gpu python -m scripts.phase11_train --algorithm qboost --device cuda --resume checkpoints/phase11_pilot.pt --duration-minutes 15 --max-updates 200 --env-batch-size 16 --eval-every 25 --eval-hands 200 --checkpoint checkpoints/phase11_pilot.pt --log-dir runs/phase11_pilot

# Original behavior: print every event JSON and no interactive meter.
uv run --extra gpu python -m scripts.phase11_train --no-progress --algorithm qboost --device cuda --resume checkpoints/phase11_pilot.pt --duration-minutes 15 --max-updates 200 --env-batch-size 16 --eval-every 25 --eval-hands 200 --checkpoint checkpoints/phase11_pilot.pt --log-dir runs/phase11_pilot

# With bar: optionally print full per-step JSON records too.
uv run --extra gpu python -m scripts.phase11_train --json-events --progress-interval 2 --algorithm qboost --device cuda --resume checkpoints/phase11_pilot.pt --duration-minutes 15 --max-updates 200 --env-batch-size 16 --eval-every 25 --eval-hands 200 --checkpoint checkpoints/phase11_pilot.pt --log-dir runs/phase11_pilot
```

## Progress semantics

- `upd=31 (+25/200)`: overall completed update count 31, 25 new updates since this invocation began, maximum 200 added updates.
- `hands/s=...`: latest end-to-end training update throughput, not the standalone GPU forward throughput.
- `elapsed`: wall-clock elapsed since the runner's start event (includes its evaluations and training).
- `time_left`: the **remaining time budget for this CLI invocation**, not a promise the job stops at the exact second. An in-flight update or evaluation completes at a safe boundary.
- `ETA`: estimated remaining time from observed update average, **capped by** time_left, so the update target may never be reached before the time limit. Before the first update it equals time_left.
- When `--max-updates` is provided, progress bar percent refers to updates; otherwise percent refers to time. Final status shows `update_limit`, `duration_limit`, or `keyboard_interrupt`.
- The display is a background heartbeat for timing only; it does not train in the background, mutate RNG, or alter checkpoints. The `metrics.jsonl` log still saves full event JSON.
- If you resume with a 15-minute limit, `time_left` starts at 15 minutes again; it does **not** include the time previously spent on older runs.

## Validation

```powershell
uv run --extra dev --extra gpu python -m pytest -q -k phase11
uv run --extra dev --extra gpu python -m pytest -q
```

The progress display refreshes each second even while an individual update/evaluation is executing. Duration is not a hard kill timer; model state is saved after completed updates only.

````````

<!-- END_MD_SOURCE path="docs/PHASE11_PROGRESS.md" -->

## docs/PHASE12_EVALUATION.md

Original SHA `aff99548aa3ab2c1794ac4c793b4345c1586c3b3` (8509 bytes).

<a id="md-033"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE12_EVALUATION.md" sha="aff99548aa3ab2c1794ac4c793b4345c1586c3b3" -->

````````markdown
# Phase 12 — Reproducible Seed-Blocked HUNL200 Evaluation

> Project: TheMostEvident/holdem-ai. Additive patch only; Phase 1–11 remain unchanged.
>
> Important: a match against RandomLegalPolicy or a frozen Actor is **NOT** GTO exploitability.

## 1. Goal and current limitations

Phase 11 reused exactly the same 200 evaluated hands at every update. Its greedy evaluations frequently returned exactly the same `frozen_anchor` score, even when the trained weights changed. This is not necessarily a code defect: greedy `argmax` decisions often stay unchanged after small parameter updates. A normal CI that treats each game as independent also ignores correlations produced by using the same deck across seat-swapped comparisons.

Phase 12 adds a **separate post-training evaluator**. It does not change the training loop or any of the Phase-11 checkpoints. It supports:

- Common shuffled-deck seed, swapped candidate/opponent physical seats, and both button placements.
- Group-level confidence intervals; **one shuffled-deck seed is the statistical unit**.
- Greedy and sampled policies, with reproducible random seeds.
- RandomLegalPolicy or independent frozen checkpoint as opponent.
- `--control`: paired difference of candidate and control, evaluated with the same seed groups and opponent.
- `validation` / `test` disjoint seed partitions, per-action counts, per-group CSV, JSON summary, model weight fingerprint.
- Progress bar with ETA via the existing optional `tqdm` package (`--no-progress` to hide).

## 2. Mathematical specification (Obsidian Markdown)

For each independent shuffle seed $k$, play four completed HUNL hands indexed by button seat $b\in\{0,1\}$ and candidate physical seat $s\in\{0,1\}$. Both seat assignments at fixed $(k,b)$ use **the same shuffled deck**. Candidate returns are in BB, are zero-sum against the opponent, and include fold/all-in outcomes.

$$
X_k^A=\frac{1}{4}\sum_{b=0}^1\sum_{s=0}^1R_A(k,b,s).
$$

Each $X_k$ is one *seed block*, not four statistically independent observations. Estimate

$$
\hat J_A=100\bar X^A,\qquad \bar X^A=\frac1K\sum_{k=1}^K X_k^A.
$$

$$
\hat\sigma_X^2=\frac{1}{K-1}\sum_{k=1}^K(X_k^A-\bar X^A)^2,
\qquad \mathrm{SE}(\hat J_A)=100\sqrt{\frac{\hat\sigma_X^2}{K}}.
$$

$$
\mathrm{CI}_{95\%}(J_A)\approx \hat J_A\pm1.96\mathrm{SE}(\hat J_A).
$$

For two agents A and C evaluated against **the same fixed opponent B**, using **the same shuffled-deck seeds** and modes:

$$
D_k=X_k^A-X_k^C,\qquad
\widehat{\Delta J}_{A-C}=100\bar D.
$$

Compute the confidence interval from **the sample variance of $D_k$**, not by subtracting unrelated confidence intervals.

These are approximate normal intervals across independent seed blocks, requiring enough seeds for meaningful inference. Prefer $K\ge50$ for development and more for held-out final evaluation. Poker hand returns are bounded but very high-variance, so even $K=50$ is not a claim of narrow uncertainty. Confidence intervals do not correct for *repeatedly choosing policies based on the same evaluation partition*.

### Duplicate caveat

This is a *seed-matched, seat-swapped benchmark*, inspired by duplicate poker; it **does not force all future random events to match under different action histories**. Both evaluations at fixed $(k,b)$ start with the same deck shuffled by the Phase-3 engine, while the candidate and opponent exchange cards/position across seat assignments. A policy change can cause betting to end sooner or later. The method reduces variance and balances seat effects, but it does not synthesize a counterfactual outcome for unplayed actions.

**Sanity check**: Against an identically weighted independent clone of itself, the candidate must obtain exactly zero group payoff in both greedy and seeded-sampled mode, because the same physical-seat game is replayed with the same policies after swapping their *labels*. A nonzero block is a warning of asymmetry or randomness drift.

## 3. Installation (PowerShell)

Extract `holdem-ai-phase12-evaluation.zip` into the existing project root as `holdem-ai-phase12-evaluation`.

```powershell
Copy-Item .\holdem-ai-phase12-evaluation\holdem_agent_phase12 .\ -Recurse -Force
Copy-Item .\holdem-ai-phase12-evaluation\tests\test_phase12_*.py .\tests\ -Force
Copy-Item .\holdem-ai-phase12-evaluation\scripts\phase12_evaluate.py .\scripts\ -Force
Copy-Item .\holdem-ai-phase12-evaluation\docs\PHASE12_EVALUATION.md .\docs\ -Force
Copy-Item .\holdem-ai-phase12-evaluation\docs\ROADMAP_PHASE12_TO_16.md .\docs\ -Force
```

No new package dependencies.

### Tests

```powershell
uv run --extra dev --extra gpu python -m pytest -q -k phase12
uv run --extra dev --extra gpu python -m pytest -q
```

## 4. Recommended evaluations

### A. Post-training versus Random, independent of Phase 11 evaluation seeds

```powershell
uv run --extra gpu python -m scripts.phase12_evaluate --candidate checkpoints/phase11_pilot.pt --groups 100 --partition validation --mode greedy --device cuda --output-json runs/phase12/greedy_random.json --output-csv runs/phase12/greedy_random.csv
```

100 groups * 4 seat/button combinations = 400 evaluated hands. This can take longer than the Phase11 batch benchmark because evaluation uses **sequential single-hand inference** to keep the correctness contract simple.

### B. Same candidate vs Phase11 frozen Anchor

```powershell
uv run --extra gpu python -m scripts.phase12_evaluate --candidate checkpoints/phase11_pilot.pt --opponent-checkpoint runs/phase11_pilot/anchor.pt --groups 100 --partition validation --mode sampled --device cuda --output-json runs/phase12/sampled_anchor.json
```

Greedy and sampled measurements answer different questions. Sampled draws use the provided deterministic match RNG and may reveal differences hidden by greedy argmax.

### C. PPO control comparison against identical Random Agent / deck seeds

```powershell
uv run --extra gpu python -m scripts.phase12_evaluate --candidate checkpoints/phase11_pilot.pt --control checkpoints/phase11_ppo.pt --groups 100 --partition validation --mode sampled --device cuda --output-json runs/phase12/paired_qboost_minus_ppo.json
```

The PPO control path must already exist, and the PPO architecture can differ from the candidate's, because the evaluation feeds each model its own valid observation tokens.

### D. Held-out test partition

After hyperparameters are fixed using `validation`, perform **a single final** evaluation on the disjoint `test` seeds:

```powershell
uv run --extra gpu python -m scripts.phase12_evaluate --candidate checkpoints/phase11_pilot.pt --groups 500 --partition test --mode sampled --device cuda --output-json runs/phase12/heldout_test.json
```

Avoid inspecting the test scores repeatedly and then selecting hyperparameters on them; that would make the test partition no longer independent. Use independent evaluation runs/seeds to replicate findings.

## 5. Summary outputs

- `mean_bb100`: average over independent four-hand seed blocks, times 100.
- `ci95_lower_bb100`, `ci95_upper_bb100`: approximate seed-block CI.
- `standard_error_bb100`: standard error over seed blocks.
- `actor_actions`, `opponent_actions`: counts per seven abstract actions; monitor policy collapse/greedy action thresholds.
- `candidate_id`, `opponent_id`: reproducible SHA-256 prefix of model weights, not a network privacy hash.
- `paired_comparison.delta_mean_bb100`: candidate minus control, evaluated against the same opponent and seed blocks.

For no opponent checkpoint, `RandomLegalPolicy` is used. Evaluation uses only public `InformationState`, never `training_state`. GPU needs the already-installed Torch GPU extra. This patch does **not** change the Phase11 training progress bar, self-play collector, model, optimizer, or checkpoint format.

## 6. Engineering validation and limitations

The authoring environment tested this patch on PyTorch CPU with a simulated Phase-3 engine. The actual Windows HUNL200 engine and GTX1070 must be verified with the two Windows test commands before committing. CUDA unit tests will run only where CUDA is available.

Once Phase12 passes on the target PC, commit *only* the new files:

```powershell
git add holdem_agent_phase12/ tests/test_phase12_*.py scripts/phase12_evaluate.py docs/PHASE12_EVALUATION.md docs/ROADMAP_PHASE12_TO_16.md
git diff --cached --stat
git commit -m "Add Phase 12 paired seed-block HUNL evaluation"
git push origin main
```

Do not `git add .` or put raw `runs/`, training `.pt` checkpoints, or extracted ZIP folders into Git.

````````

<!-- END_MD_SOURCE path="docs/PHASE12_EVALUATION.md" -->

## docs/PHASE13_BATCH_SCALING.md

Original SHA `20147311636b06e0132923466bec69f3cae4e71e` (9451 bytes).

<a id="md-034"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE13_BATCH_SCALING.md" sha="20147311636b06e0132923466bec69f3cae4e71e" -->

````````markdown
# Phase 13 — Training Optimization 2.0

> Target: `TheMostEvident/holdem-ai`. Dependencies: **Phase 9, 10, 11 (including tqdm), 12**.
> Pure additive patch: **no existing Phase 1–12 source modification**.
> VRPO-Lite is experimental; this is **not** a faithful full VRPO paper replication.

## 1. Scope

- On-policy self-play with `hands_per_update` 16/64/128; `env_batch_size` 16/32/64.
- Reuse the **unchanged Phase 9 PPO** and **Phase 10 Q-Boost** optimizer losses.
- Import Phase9, Phase10 or Phase11 (`phase=9` / `phase=10` on disk) trusted checkpoints **without overwriting source**, preserving Actor/Critic **and Adam optimizer states**.
- Save a new atomic checkpoint each completed update and append `metrics.jsonl`; resume the exact same stage.
- Record per-action *chosen counts*, *legal opportunities*, *conditional-on-legal rates*; by street and by button/BB; pre-update empirical Actor entropy.
- Phase 12 seed-blocked evaluation against Random and frozen stage-start Actor; **sampled mode by default**. Validation seed blocks are reused within this stage for trend tracking, not final independent tests.
- Phase 13 tqdm progress, time-left, ETA; empirical training-throughput benchmark and matched-start comparison of PPO/Q-Boost.

## 2. Math

Let $H$ = hands per update, $B$ = simultaneously active environments, $M$ = optimizer minibatch size, $E$ = PPO epochs. Then:

$$
H \in \{16,64,128\},\qquad B\in\{16,32,64\},\qquad M\in\{32,64,128\}.
$$

Let $D_k$ be the actual decisions observed in update $k$ (depends on fold/raises), and $N_k=|D_k|$.

$$
N_{\text{mini},k} = E\left\lceil\frac{N_k}{M}\right\rceil.
$$

All samples are **on-policy** for the previous Actor weights $\theta_k$, collected completely before any gradient update.

$$
D_k\sim\pi_{\theta_k},\qquad
\theta_{k+1}=\mathrm{Update}(\theta_k,D_k).
$$

Throughput for training updates (including collection, diagnostics and optimization, excluding evaluation and checkpoint I/O):

$$
\widehat{T}_{\mathrm{train}}
=\frac{\sum_k H_k}{\sum_k \Delta t_{\mathrm{update},k}}
\quad[\mathrm{hands/sec}].
$$

Each action head has a legal-action mask. Define $C_a=\sum_t\mathbf{1}[a_t=a]$ and $L_a=\sum_t\mathbf{1}[a\in\mathcal{A}(I_t)]$.

$$
\hat p(a\mid\mathrm{legal})=\frac{C_a}{L_a},\quad L_a>0.
$$

When $L_a=0$, the rate is `null`, **not** zero. This is a *marginal conditional opportunity rate*, not $\pi(a|I)$ for a fixed infoset; it also varies with the population of encountered states.

$$
H(\pi(\cdot\mid I_t))=-\sum_{a\in\mathcal A(I_t)}\pi(a\mid I_t)\log\pi(a\mid I_t)
$$

$$
\overline{H}_{\mathrm{normalized}}
=\frac1N\sum_t\begin{cases}
H(\pi_t)/\log |\mathcal A(I_t)|,&|\mathcal A(I_t)|>1\\
0,&|\mathcal A(I_t)|=1.
\end{cases}
$$

These diagnostics use **Actor-only** inputs, without Critic privilege. Critic encodings are used only in the optimizer's training path.

## 3. Phase 11 to Phase 13 migration

The `checkpoints/phase11_pilot.pt` payload is a standard Phase 10 QBoost payload (phase=10, architecture/Adam/ppo_config/torch RNG). A Phase13 warm-start creates a **NEW** experiment with different PPOConfig values and new seed namespace:

$$
\mathrm{seed}_{13}=\mathrm{seed}_{old}+1{,}000{,}003
$$

unless `--rollout-seed` explicitly supplied.

**Important:** Phase 9/10/11 checkpoint formats stay unchanged; the output still has payload `phase=9` for PPO or `phase=10` for QBoost. The new run's metadata `run.json` has `phase=13` with `checkpoint_format_phase`. Phase12's checkpoint reader continues to work.

`--resume` must point to a Phase13 run's own output checkpoint and original log-dir; batch size, seed, epochs and optimizer minibatch size come **from its saved checkpoint**. To change these again, start a *new* stage with `--from-checkpoint <old-phase13-output>` and new output paths. Current stage's frozen Anchor remains unchanged on resume. Original checkpoint is never overwritten by warm-start.

The starting source can be:
- PPO Stage: Phase 9/11 PPO (`phase=9`)
- QBoost Stage: Phase 10/11 QBoost (`phase=10`)
- Matched PPO vs QBoost comparison: **same Phase 9 source (`phase=9`)**; Phase 13 automatically wraps it as QBoost using Phase10's `from_phase9_checkpoint` for the QBoost arm.

## 4. Install from ZIP (PowerShell, repo root)

```powershell
Copy-Item .\holdem-ai-phase13-scaling\holdem_agent_phase13 .\ -Recurse -Force
Copy-Item .\holdem-ai-phase13-scaling\tests\test_phase13_*.py .\tests\ -Force
Copy-Item .\holdem-ai-phase13-scaling\scripts\phase13_*.py .\scripts\ -Force
Copy-Item .\holdem-ai-phase13-scaling\docs\PHASE13_BATCH_SCALING.md .\docs\ -Force

uv run --extra dev --extra gpu python -m pytest -q -k phase13
uv run --extra dev --extra gpu python -m pytest -q
```

## 5. Warm-start the current QBoost checkpoint: first GTX 1070 Pilot

```powershell
uv run --extra gpu python -m scripts.phase13_train --algorithm qboost --device cuda --from-checkpoint checkpoints/phase11_pilot.pt --hands-per-update 64 --env-batch-size 32 --minibatch-size 64 --epochs 2 --rollout-seed 13000042 --duration-minutes 10 --max-updates 3 --eval-every 2 --eval-groups 40 --eval-mode sampled --checkpoint checkpoints/phase13_qboost_64.pt --log-dir runs/phase13_qboost_64
```

Stage resumes using original PPO hyperparameters saved within `phase13_qboost_64.pt`:

```powershell
uv run --extra gpu python -m scripts.phase13_train --algorithm qboost --device cuda --resume checkpoints/phase13_qboost_64.pt --duration-minutes 15 --max-updates 100 --env-batch-size 32 --eval-every 25 --eval-groups 40 --checkpoint checkpoints/phase13_qboost_64.pt --log-dir runs/phase13_qboost_64
```

Start another independent 128-hand stage from the original Phase11 checkpoint:

```powershell
uv run --extra gpu python -m scripts.phase13_train --algorithm qboost --device cuda --from-checkpoint checkpoints/phase11_pilot.pt --hands-per-update 128 --env-batch-size 64 --minibatch-size 64 --epochs 2 --rollout-seed 13000042 --duration-minutes 10 --max-updates 3 --eval-every 2 --eval-groups 40 --checkpoint checkpoints/phase13_qboost_128.pt --log-dir runs/phase13_qboost_128
```

**Caution:** changing hands-per-update changes `seed + iteration*hands_per_update + hand_index`; both stages have the same overall seed namespace, but **do not receive identical per-hand deal schedules**, nor identical optimizer effective gradient noise. This is a batch-scaling experiment, not a matched-deal comparison.

## 6. Actual GPU *training* benchmark, not merely inference

```powershell
uv run --extra gpu python -m scripts.phase13_benchmark --device cuda --algorithm qboost --hands-per-update 16,64,128 --env-batch-sizes 16,32,64 --updates 3 --epochs 2 --minibatch-size 64 --from-checkpoint checkpoints/phase11_pilot.pt --root runs/phase13_benchmark_qboost
```

- Runs all *effective* combinations where `env_batch_size <= hands_per_update`, not the nominal ones where `env_batch_size` cannot be fully used.
- Skips each case's first training update as warmup when two or more complete updates exist.
- Each case has a separate checkpoint/log folder. The source is untouched.
- Measures full optimizer time, unlike Phase11's actor-forward benchmark. No GPU memory/performance claim is made until the user's GTX1070 result is observed.

## 7. Matched-start PPO vs QBoost training

Both arms must use **exactly the same Phase9 source** and seed settings. After update 1 policies may diverge; each arm collects its own latest on-policy rollout.

```powershell
uv run --extra gpu python -m scripts.phase13_train --algorithm ppo --device cuda --from-checkpoint checkpoints/phase9_ppo.pt --hands-per-update 64 --env-batch-size 32 --minibatch-size 64 --epochs 2 --rollout-seed 13000999 --max-updates 100 --eval-every 25 --eval-groups 100 --checkpoint checkpoints/phase13_ppo_control.pt --log-dir runs/phase13_ppo_control

uv run --extra gpu python -m scripts.phase13_train --algorithm qboost --device cuda --from-checkpoint checkpoints/phase9_ppo.pt --hands-per-update 64 --env-batch-size 32 --minibatch-size 64 --epochs 2 --rollout-seed 13000999 --max-updates 100 --eval-every 25 --eval-groups 100 --checkpoint checkpoints/phase13_qboost_control.pt --log-dir runs/phase13_qboost_control

uv run --extra gpu python -m scripts.phase13_compare --ppo-log-dir runs/phase13_ppo_control --qboost-log-dir runs/phase13_qboost_control
```

The log comparison computes descriptive *training* metrics and only tags runs matched-start if source paths and PPO hyperparameters align. To compare **poker win rate**, use Phase12 `phase12_evaluate` against the same fixed opponent and matched seed blocks, then compare paired block scores. A training throughput/entropy advantage alone does not prove strategy improvement.

## 8. Artifacts and status

```
runs/phase13_qboost_64/
  run.json            # Phase13 stage settings and lineage
  anchor.pt           # immutable actor copy at Phase13 stage start
  metrics.jsonl       # updates, action/legal counters, evaluations
  summary.json        # run completion
checkpoints/phase13_qboost_64.pt # Phase10-format QBoost + Adam
```

The `tqdm` display shows updates, hands/s, ETA and wall-clock budget. The detailed histograms remain in `metrics.jsonl` to keep the PowerShell progress display readable.

**Do not Git-commit trained `.pt`, `runs/`, or the extracted patch folders.** Phase13 CUDA tests and performance benchmark must be verified on the user's GPU. No full VRPO convergence / exploitability claim follows from these diagnostics.

````````

<!-- END_MD_SOURCE path="docs/PHASE13_BATCH_SCALING.md" -->

## docs/PHASE14_CONTROLLED_STUDY.md

Original SHA `3a0fd210542ddb109dcb5c082b9d1e318cab7723` (9315 bytes).

<a id="md-035"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE14_CONTROLLED_STUDY.md" sha="3a0fd210542ddb109dcb5c082b9d1e318cab7723" -->

````````markdown
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

````````

<!-- END_MD_SOURCE path="docs/PHASE14_CONTROLLED_STUDY.md" -->

## docs/PHASE15_CRITIC_DIAGNOSIS.md

Original SHA `1fda4dcb7a19748df71426459b31e43da74bbccd` (4914 bytes).

<a id="md-036"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE15_CRITIC_DIAGNOSIS.md" sha="1fda4dcb7a19748df71426459b31e43da74bbccd" -->

````````markdown
# Phase 15 — PPO Critic Diagnosis (Agent Task)

**Status:** Added as a read-only diagnostic on \`agent/phase15-collapse-diagnostics\`, not merged into \`main\`. No engine, game rules, GUI, optimizer, safety defaults, historical training logs, or existing checkpoints are modified.

## 1. Targets of the diagnosis

The Phase-15 Actor can only use its own Information State \(I_i\). The centralized training-only Critic sees privileged training state \(s\) and outputs \(Q_\phi(s,i,a)\) for each player/action. It is trained with a Monte Carlo terminal reward for the **observed** action:

\[
\hat G_i^{Q}=Q_\phi(s_i, i, a_i),\qquad
L_{Q}=\operatorname{Huber}(\hat G_i^{Q}, G_i).
\]

PPO's frozen, action-independent baseline is:

\[
\hat V_\phi(s_i)=\sum_{a\in\mathcal A_{\rm legal}(I_i)}
\pi_{\rm behavior}(a\mid I_i)Q_\phi(s_i,i,a)
\]

and raw advantage is:

\[
A_i=G_i-\hat V_\phi(s_i).
\]

**Do not confuse** selected-action \(Q\) quality with action-independent \(V\) baseline quality: they use different predictions, even on the same sample.

## 2. Metrics

For prediction \(p_i\), target \(G_i\), and error \(e_i=p_i-G_i\):

\[
\mathrm{Bias}=\frac1N\sum_i e_i,\quad
\mathrm{MAE}=\frac1N\sum_i|e_i|,\quad
\mathrm{RMSE}=\sqrt{\frac1N\sum_i e_i^2}
\]

\[
\mathrm{ExplainedVariance}=1-\frac{\mathrm{Var}(e)}{\mathrm{Var}(G)}
\]

EV is \`null\` if the reward variance is zero. EV can equal 1 even when prediction bias is nonzero, so **never interpret EV without bias and RMSE**. All numerical metrics are in Big Blind units where applicable (not bb/100).

Metrics are recorded overall and split by **Street, Acting Seat, Button/Big Blind Position, and Observed Action**. Raw advantage mean/std/mean-absolute-value are included.

## 3. Evaluation protocol

### A. Shared reference rollout (controlled observations/actions/targets)

Sample a single frozen reference policy with a held-out diagnostic deck seed. Feed the identical \((s,I,a,G)\) samples to **both** critics. This controls input data and reveals differences between critic functions.

**Important limitation:** \(Q^{\pi_{\rm best}}\) and \(Q^{\pi_{\rm latest}}\) can differ legitimately because they assume different *future behavior*. Terminal returns sampled under \(\pi_{\rm ref}\) are **not** unbiased labels for a candidate policy's \(Q^{\pi_{\rm candidate}}\). Interpret paired residual differences only as a **cross-policy generalization probe**, not as proof that one Critic is incorrectly trained.

### B. Two independent on-policy rollouts (primary calibration)

Use each Checkpoint's own Actor to generate held-out trajectories. Its own Critic then predicts the realized terminal return for observed actions. PPO's baseline uses that **same generating policy**.

This evaluates the correct continuation distribution for each checkpoint but creates different state/action distributions, so the raw scores are not a controlled causal comparison.

Decision rows from the same hand share terminal rewards. This tool reports descriptive metrics, not iid significance intervals. On-policy and shared diagnostic seeds are kept outside existing training and validation namespaces. Postflop coverage can remain sparse.

## 4. Commands — Windows PowerShell

First synchronize the research branch:

\`\`\`powershell
git pull --ff-only origin agent/phase15-collapse-diagnostics
uv run --extra gpu pytest tests/test_phase15_critic_diagnostics.py -q
uv run --extra gpu pytest -q
\`\`\`

Then execute (uses existing local Checkpoints, without training):

\`\`\`powershell
uv run --extra gpu python -m scripts.phase15_critic_audit \`
  --reference runs/phase14_main/seed_52/ppo/best.pt \`
  --before runs/phase15_ppo_safety_v2_pilot/best.pt \`
  --after runs/phase15_ppo_safety_v2_pilot/latest.pt \`
  --device cuda \`
  --hands 256 \`
  --on-policy-hands 128 \`
  --seed 570000000 \`
  --output runs/phase15_audit/safety_v2_critic_audit.json
\`\`\`

Output uses exclusive creation: an existing JSON report is **not overwritten**. The CLI prints concise on-policy metrics, while the JSON contains all subsets and both protocols.

## 5. Interpretation and next phase

- Extremely poor **on-policy** \(Q(s,a_{\rm taken})\) RMSE / EV: investigate Critic target scale, architecture, optimizer and value loss before changing the Actor.
- Relatively good chosen-\(Q\) fit but poor **on-policy** \(V(s)\) calibration: investigate unsupervised alternative-action \(Q(s,a)\) and quality of \(V(s)=\sum\pi Q\).
- Reasonable critic metrics alongside declining validation scores: investigate Actor PPO updates, bet sizing, exploration, opponent/generalization, and noisy evaluation.
- The diagnostics alone cannot establish why the latest Policy's validation score is lower. Do not change PPO learning rules solely based on a noisy Pilot.

**Checkpoint safety:** Historical \`best.pt\`/\`latest.pt\` stay read-only. Do not run Phase 15 \`finalize\` for this test.

````````

<!-- END_MD_SOURCE path="docs/PHASE15_CRITIC_DIAGNOSIS.md" -->

## docs/PHASE15_EXPERIMENT_REPORT_2026-10-13.md

Original SHA `d231a20f52cd109a752451524a9891a771780011` (16335 bytes).

<a id="md-037"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE15_EXPERIMENT_REPORT_2026-10-13.md" sha="d231a20f52cd109a752451524a9891a771780011" -->

````````markdown
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
````````

<!-- END_MD_SOURCE path="docs/PHASE15_EXPERIMENT_REPORT_2026-10-13.md" -->

## docs/PHASE15_LONG_RUN_TRAINING.md

Original SHA `f950834d4fd8ae0f61007cd2ac77b09e00abbf6c` (11248 bytes).

<a id="md-038"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE15_LONG_RUN_TRAINING.md" sha="f950834d4fd8ae0f61007cd2ac77b09e00abbf6c" -->

````````markdown
# Phase 15 — 장시간 PPO / Q-Boost Self-Play 및 Best Checkpoint 관리

> Source of truth: Phase 14까지 Push된 `TheMostEvident/holdem-ai`. 이번 패치는 기존 Phase 1~14 파일을 수정하지 않는 추가형 구현이다. 현재 수행된 검증은 Linux CPU / 모의 HUNL 엔진 기반이다. Windows + GTX 1070 실기 검증은 사용자 로컬 PC에서 필요하다.

## 1. 목표

Phase 14의 통제 실험에서 PPO는 세 학습 Seed 모두 Validation 최고점이 높았으며, Q-Boost는 Frozen Initial 상대의 Test 차이가 양수였지만 Seed 3개만으로 결론을 확정할 수 없었다. Phase 15는 **알고리즘 개선이 아니라 안전한 장시간 실행과 평가의 독립성**을 목표로 한다.

- Phase 14 `best.pt`에서 **Actor / Critic / Adam**을 이어받는다.
- Phase 13의 `collect_batched_self_play`, `optimize_ppo`, `optimize_qboost` 등 학습 구현을 **수정하지 않고 재사용한다**.
- `latest.pt`를 업데이트마다 원자적으로 저장하고, Validation 최고 모델 `best.pt`를 별도로 유지한다.
- 고정 Validation 카드 묶음에서 초기 Frozen Actor와 RandomLegalPolicy **두 상대**에 대한 성능 변화를 측정한다.
- Entropy Collapse / Approx KL 비정상적 증가를 감시하고 연속 경고 시 학습을 멈춘다.
- 학습을 종료하기 전에는 **Final 카드 묶음에 접근하지 않는다**. Final 평가는 별도 CLI에서 딱 한 번 실행하고 결과를 봉인한다.
- `metrics.jsonl`과 별도의 최종 CSV / JSON 보고서 생성.

## 2. 학습 설정 및 베이스라인

권장 시작 설정(Phase 13 실측의 잠정 최적):

$$
H=64\ \text{hands/update},\quad B=32\ \text{active environments},\quad M=64\ \text{decisions/minibatch},\quad E=2\ \text{epochs}
$$

Phase 14 PPO Validation 최고점: Seed 51에서 809.50, Seed 52에서 858.125, Seed 53에서 456.75 bb/100. 그러므로 Phase 15 PPO 시작점을 **Seed 52 PPO best**로 선택할 수 있다. 이는 Phase 14 Test에서 유리하게 나온 조합을 선택한 것이 아니라 Validation 기준 선택이다.

시작 체크포인트 예시:

- PPO: `runs/phase14_main/seed_52/ppo/best.pt`
- Q-Boost: `runs/phase14_main/seed_52/qboost/best.pt`

다음과 같이 각각 별도 경로와 독립 Rollout Seed 영역을 사용한다.

$$
\mathrm{seed}_{\mathrm{rollout}}^{\mathrm{PPO}}=400,000,000,
\quad \mathrm{seed}_{\mathrm{rollout}}^{\mathrm{QB}}=500,000,000.
$$

한 업데이트에서 현재의 `completed_updates`를 $u$라고 하면 카드 RNG Seed는 기존 Phase 11/13과 동일하게

$$
\mathrm{seed}(u,k)=\mathrm{rollout\_seed}+uH+k,\quad 0\le k<H
$$

다. Phase 14의 학습 카드 Seed 네임스페이스와 다르게 설정하여 카드 재사용을 피한다. **이미 진행된 Phase 14 핸드를 새 모델에 Replay하지 않는다.**

## 3. Best Checkpoint 선택

동일한 고정 Validation 카드 시드에서, 정책 $\pi_t$를 초기 Anchor $\pi_0$와 Random 상대에 각각 평가한다.

$$
J_A(t)=\widehat J_{\mathrm{val}}(\pi_t,\pi_0),\qquad
J_R(t)=\widehat J_{\mathrm{val}}(\pi_t,\pi_{\mathrm{Random}})
$$

$$
\boxed{
S_t=\frac12\big[(J_A(t)-J_A(0))+(J_R(t)-J_R(0))\big]
}
$$

두 평가의 단위는 동일한 bb/100이다. $J_A(0)=0$은 같은 정책을 좌석 교환하여 평가한 값이다. 시작 정책 $\pi_0$의 점수는 $S_0=0$이며, 유의한 변화가 없거나 평가가 악화될 때는 **초기 `best.pt`도 유효한 선택**이다.

$$
t^*=\arg\max_{t\in\mathcal T_{\mathrm{evaluated}}} S_t
$$

**주의:** 이 점수는 연구 목적의 휴리스틱이며 특정 상대에 대한 GTO Exploitability가 아니다. 반복적으로 동일한 Validation 카드에 맞춰 `best.pt`를 선택하므로 Validation 점수에는 선택 편향이 있을 수 있다. 최종 독립 평가 결과를 별도로 보고한다.

## 4. 안전장치

정책 Entropy와 PPO Approximate KL을 매 업데이트 확인한다.

$$
\mathsf{warn}_t=\mathbf1\left[H(\pi_t)<h_{\min}\ \lor\ \widehat{KL}_t>k_{\max}\right]
$$

기본값은 $h_{\min}=0.45$, $k_{\max}=0.10$, 연속 경고 횟수 $p=12$이다. 경고가 연속 $p$회 발생하면 `stop_reason=safety_stop`을 반환한다. **일시적으로 엔트로피가 낮다는 이유만으로 모델이 나쁘다고 판정하지 않는다**. 임계값은 보호용 휴리스틱이며 성능 보장이 아니다.

- 체크포인트 저장은 Phase 9/10의 `.tmp` → atomic replace 방식을 재사용한다.
- `run.json`에는 원본 Checkpoint SHA-256과 모델/Optimizer 설정을 기록한다.
- `state.json`에는 `best.pt` SHA-256, 최고 점수, 경고 수와 진행 카운터를 기록한다.
- Resume 시 원본 SHA-256, 저장된 PPO Config, Architecture, Best Checkpoint 해시를 확인하고 변경되면 오류를 낸다.
- 같은 `root` 폴더에 두 학습기가 접근하지 못하도록 OS 파일 잠금(`.phase15.lock`)을 사용한다. 잠금 파일은 그대로 남아도 정상이다.
- `KeyboardInterrupt`가 업데이트 **도중** 발생하면 마지막 원자적 Checkpoint만 신뢰한다. 다른 프로세스/전원 중단 직후 `latest.pt`의 카운터가 `state.json`보다 크면 복구 기록 `recovered_update`를 남긴다. 복구된 업데이트의 손실값 로그는 임의로 생성하지 않는다.
- 강제 프로세스 종료·정전·파일시스템 손상 후 완전 무손실 복구를 보증하지 않는다. 큰 실험에서는 실험 폴더를 별도 백업한다.

## 5. 설치 (Windows PowerShell)

압축 해제 후 `holdem-ai` 루트에서 실행:

```powershell
Copy-Item .\holdem-ai-phase15-longrun\holdem_agent_phase15 .\ -Recurse -Force
Copy-Item .\holdem-ai-phase15-longrun\scripts\phase15_*.py .\scripts\ -Force
Copy-Item .\holdem-ai-phase15-longrun\tests\test_phase15_*.py .\tests\ -Force
Copy-Item .\holdem-ai-phase15-longrun\docs\PHASE15_LONG_RUN_TRAINING.md .\docs\ -Force

uv run --extra dev --extra gpu python -m pytest -q -k phase15
uv run --extra dev --extra gpu python -m pytest -q
```

## 6. GTX 1070 Pilot — 3 Updates

```powershell
uv run --extra gpu python -m scripts.phase15_train --algorithm ppo --source-checkpoint runs/phase14_main/seed_52/ppo/best.pt --root runs/phase15_ppo_pilot --device cuda --duration-minutes 5 --max-updates 3 --hands-per-update 64 --env-batch-size 32 --minibatch-size 64 --epochs 2 --eval-every 2 --eval-groups 40 --final-groups 200 --rollout-seed 400000000
```

3 Updates = 192개의 신규 학습 Hands. 모든 Stage의 `--max-updates`는 **이번 호출에서 추가할 업데이트 수**이고, `--duration-minutes`도 호출마다 새 예산이다. 두 제한 중 먼저 충족한 조건에서 종료한다.

## 7. PPO 본 실험 — 30분 / 5시간

아래 명령은 **새 폴더**에서 30분 학습한다. 본 실행에는 검증 가능한 카드 그룹 100개, 매 100 Updates Validation을 사용한다.

```powershell
uv run --extra gpu python -m scripts.phase15_train --algorithm ppo --source-checkpoint runs/phase14_main/seed_52/ppo/best.pt --root runs/phase15_ppo_main --device cuda --duration-minutes 30 --hands-per-update 64 --env-batch-size 32 --minibatch-size 64 --epochs 2 --eval-every 100 --eval-groups 100 --final-groups 400 --rollout-seed 400000000
```

그다음 같은 실험을 **추가 5시간** 재개한다(총 약 5.5시간 학습). 5시간 총량만 원한다면 재개 예산을 270분으로 변경한다.

```powershell
uv run --extra gpu python -m scripts.phase15_train --root runs/phase15_ppo_main --resume --device cuda --duration-minutes 270
```

`--resume`은 저장된 알고리즘/학습 Batch/평가 Seed/Entropy 한계 설정을 읽는다. 실행 시간과 이번 호출의 `--max-updates`만 자유롭게 변경할 수 있다. PPO 최신 모델은 `runs/phase15_ppo_main/latest.pt`, 가장 높은 Validation 점수의 모델은 `runs/phase15_ppo_main/best.pt`다.

Q-Boost를 새 실험으로 시작한다면 Source와 Rollout Seed를 명시적으로 구분한다.

```powershell
uv run --extra gpu python -m scripts.phase15_train --algorithm qboost --source-checkpoint runs/phase14_main/seed_52/qboost/best.pt --root runs/phase15_qboost_main --device cuda --duration-minutes 30 --hands-per-update 64 --env-batch-size 32 --minibatch-size 64 --epochs 2 --eval-every 100 --eval-groups 100 --final-groups 400 --rollout-seed 500000000
```

Phase 15는 **통제된 PPO vs Q-Boost 직접 비교가 아니라 서로 다른 모델의 개별 장시간 학습 인프라**이다. 두 실험의 출발 정책과 설정이 달라지면 알고리즘 성능의 인과적 비교는 Phase 14 다중-Seed 방식으로 별도 진행한다.

## 8. 학습 종료 후 최종 독립 평가 — 별도 명령

**더 이상 해당 모델을 학습시키지 않기로 결정했을 때만** 실행:

```powershell
uv run --extra gpu python -m scripts.phase15_finalize --root runs/phase15_ppo_main --device cuda
```

기본적으로 `final_seed=25_000_000`에서 400개의 독립 덱 시드 × 4개 좌석·Button 조합 = **1,600개의 게임**을 Frozen Initial Actor와 Random Legal Opponent 상대 각각 평가한다. 2개 상대이므로 총 3,200게임이다.

한 번 완료되면 결과를 `final_report.json`에 원자적으로 저장하고 추가 `--resume`을 차단한다. 반복 호출은 같은 Best Checkpoint SHA-256일 때 저장된 결과만 읽는다. **Final 결과를 보고 설정을 수정하려면 새 실험과 새 Final Seed를 사용해야 한다.**

## 9. 실험 디렉터리

```text
runs/phase15_ppo_main/
  .phase15.lock         # 자동 해제 OS 잠금의 영구 파일
  run.json              # Source SHA / 설정 / 고정 Seed
  state.json            # Best 선택 정보 / 경고 / 카운터 / Best SHA
  latest.pt             # 업데이트 완료 시마다 원자적 갱신
  best.pt               # 최고 Validation Checkpoint
  metrics.jsonl         # Training + Validation + 경고 + 복구 로그
  summary.json          # 마지막 실행 호출의 요약
  final_report.json     # Finalize 이후 1회 생성, sealed
  training_curve.csv    # Finalize 이후 생성
  validation_curve.csv  # Finalize 이후 생성
```

학습 도중에도 `metrics.jsonl`을 읽어 곡선을 만들 수 있다. CSV는 Finalize 시점에 생성된다.

## 10. 현재 한계 및 다음 연구

- Sampled Greedy 평가와 Fold/Check 빈도 해석은 Position/Street/Legal Action 조건을 분리해야 한다.
- Validation은 2명(Random, Frozen Initial) 뿐이다. 상대방이 학습된 전략에 적응하는 적대적 Agent Pool은 별도 Phase에서 구현해야 한다.
- Best 선정은 여러 번 같은 Validation Seed를 사용한다. 모든 Model을 평가한 뒤 Test를 반복 수행하면 최종 Test의 독립성이 사라진다.
- `entropy`, `approx_kl`, `critic_loss`는 PPO/QBoost 내부 학습 지표이지 홀덤 EV 또는 GTO Exploitability의 직접 추정량이 아니다.
- 5시간 실측 처리량은 사용자의 GTX 1070에서 확인해야 하며, 5시간 정확히 시계에 맞춰 중단하는 데에는 평가·업데이트 경계에서 지연이 발생할 수 있다.
- PyTorch CUDA 최대 메모리 값은 PyTorch 텐서 할당 메모리만 반영한다. 시스템 전체 GPU VRAM 사용량이 아니다.

````````

<!-- END_MD_SOURCE path="docs/PHASE15_LONG_RUN_TRAINING.md" -->

## docs/PHASE15_PHASE3_SAFETY_TELEMETRY.md

Original SHA `2d8cefa707ca49e2007d1e524bf2280be85f6f75` (3626 bytes).

<a id="md-039"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE15_PHASE3_SAFETY_TELEMETRY.md" sha="2d8cefa707ca49e2007d1e524bf2280be85f6f75" -->

````````markdown
# Phase 15 — Phase 3 Safety + Reward Diagnostics (Agent only)

## Status

Changes are committed to the research branch \`agent/phase15-collapse-diagnostics\`, **not** merged to \`main\`. Existing \`runs/phase15_ppo_main\` and \`runs/phase15_ppo_stability_pilot\` checkpoints are not modified. The branch code must pass tests locally before a new GPU pilot.

## Motivation

A low policy entropy alone is not evidence that a poker strategy is bad. In the observed run, however, a fixed-state comparison showed a +0.343803 increase in mean fold probability (Update 375 → 450), concentrated particularly in preflop big-blind states. An improved warning system is appropriate, while actual policy quality still requires validation.

## Safety policy

For a **new run** on this branch:

- Entropy below \`entropy_alert_threshold\` increments \`entropy_warnings\` and emits an \`entropy_warning\` log event. **It no longer advances the hard-stop counter by default.**
- Approx KL above \`max_approx_kl\` independently increments \`kl_warnings\` and advances the consecutive hard-stop counter.
- Both events are recorded when both conditions hold (previous \`elif\` could hide KL warnings).
- A high-KL streak of \`entropy_stop_patience\` updates still triggers \`safety_stop\`.
- Explicit \`entropy_stop_enabled=True\` restores entropy hard-stop behavior at configuration level.
- Resuming runs created **before** the new manifest field uses their historical entropy-hard-stop behavior. Do not use old run roots for the new safety experiment.
- NaN/Inf checks in the PPO optimizer remain intact.

Formally:

$$
w_{H,t}=\mathbf1[H_t<H_{\min}], \qquad
w_{KL,t}=\mathbf1[\widehat D_{KL,t}>KL_{\max}].
$$

$$
w_{\mathrm{stop},t}=w_{KL,t}\lor (\mathrm{entropy\_stop\_enabled}\land w_{H,t}).
$$

Warnings and stopping criteria are different. **The default has changed only for fresh runs on this research branch.**

## Per-update telemetry

The \`reward_diagnostics\` object in each new update record includes:

- Hand count; seat-0 mean/std/min/max terminal payoff (BB).
- Absolute terminal payoff mean (BB).
- Fold-terminated hand count and proportion.
- Mean decisions per hand.
- Non-fold-terminated hand count. **This is not identical to Showdown count**; the exact showdown rate remains \`null\` until an authoritative engine terminal-result field is wired.

Existing \`action_diagnostics\` already supplies legal opportunity-conditioned action rates by street/position. PPO metrics already record approximate KL, clip fraction and unclipped actor/critic gradient norms. The added telemetry augments, rather than redefines, those measurements.

## Validation command (Windows PowerShell)

\`\`\`powershell
uv run --extra gpu pytest tests/test_phase15_policy_audit.py tests/test_phase15_safety_phase3.py -q
\`\`\`

Run the ordinary project test suite too before merging. New telemetry does not retroactively appear in existing historical \`metrics.jsonl\` files.

## Next experiment

After tests pass, create a fresh Phase 15 run rooted in a **new unique directory**, initialized from the trusted Phase 14 checkpoint; do not overwrite or finalize the historical runs. Use a limited update count and an independent rollout seed, and compare against the original safety rules. Consider a second pilot seeded from the previously selected Update 375 model for a controlled continuation, taking care to distinguish preexisting optimizer-state effects.

Before claiming Policy Collapse is *resolved*, extend fixed-state sampling beyond its preflop-heavy distribution and add authoritative showdown telemetry and independent evaluation.

````````

<!-- END_MD_SOURCE path="docs/PHASE15_PHASE3_SAFETY_TELEMETRY.md" -->

## docs/PHASE15_PREFLOP_SEQUENCE_AUDIT.md

Original SHA `df3c4d6044d174788975c5a9ada10f1b1e778e2e` (3345 bytes).

<a id="md-040"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE15_PREFLOP_SEQUENCE_AUDIT.md" sha="df3c4d6044d174788975c5a9ada10f1b1e778e2e" -->

````````markdown
# Phase 15 — Preflop Sequence Diagnosis

Read-only Agent audit; existing engine/GUI, training/optimizer, previous runs and checkpoints are untouched.

## Why

The previous 512-hand-per-model comparison found:

| Metric | Uniform Legal | Random Initialization | Trained Phase14 |
| --- | ---: | ---: | ---: |
| Preflop decisions | 1397 | 1335 | 2267 |
| Explicit Preflop All-in | 218 | 218 | 438 |
| All-in legal opportunities | 1179 | 1117 | 1829 |
| Preflop terminal folds | 318 | 353 | 53 |
| Preflop terminal calls | 97 | 84 | 433 |
| Pot raises | 193 | 176 | 523 |
| Double-pot raises | 175 | 224 | 545 |

In Phase14:

\[
P(\mathrm{Preflop\ Allin\ per\ hand})=438/512\approx0.8555
\]

\[
P(\mathrm{Allin\ selected}\mid\mathrm{legal\ decision})
=438/1829\approx0.2395
\]

The denominators differ. The new audit reconstructs every preflop sampled action, for each hand.

## New diagnostic outputs

- Most common **complete preflop action-head sequences** (each hand counts once).
- Most common **action-head prefixes ending at the first explicit preflop All-in**.
- First All-in decision index histogram and prior action distribution.
- Number of actions prior to the hand's first All-in; number of actions for hands ending in a Preflop Fold.
- Selected action counts and legality-conditioned action rates by preflop decision index.
- All seven action choices conditional on public outstanding bet (to-call > 0 versus zero), and button/BB position.
- Counts of hands whose last Actor decision was Preflop and whose final action was Fold or Call.

Uses only actor-visible preflop observations from the existing frozen Agent rollout collector. No private cards, CriticState or hidden deck enter the Actor.

## Interpret carefully

- A 7-head policy action index 6 means an **explicit** all-in; an ordinary CALL can also exhaust the stack.
- A sampled Raise head sequence is NOT automatically a formal 3-bet/4-bet classification or an exact wager amount.
- A preflop last action may be followed by a full-board automatic runout before showdown.
- Each policy plays both seats against itself. Same deck seeds do not force identical action histories.
- Random Initialization is a *synthetic fresh initialization*, not necessarily the historical training-start checkpoint.

## PowerShell

First:

~~~powershell
git pull --ff-only origin agent/phase15-collapse-diagnostics
uv run --extra gpu pytest tests/test_phase15_preflop_sequence_audit.py -q
uv run --extra gpu pytest -q
~~~

If tests pass:

~~~powershell
uv run --extra gpu python -m scripts.phase15_preflop_sequence_audit --trained runs/phase14_main/seed_52/ppo/best.pt --init-seed 52 --device cuda --hands 512 --seed 600000000 --output runs/phase15_audit/phase14_preflop_sequences.json
~~~

New report output is exclusive. No training or saved-checkpoint modification.

Inspect:

~~~powershell
$r = Get-Content .\runs\phase15_audit\phase14_preflop_sequences.json -Raw | ConvertFrom-Json
$r.models.trained_phase14.first_allin_decision_number_counts
$r.models.trained_phase14.immediately_before_first_allin
$r.models.trained_phase14.most_common_prefixes_through_first_allin | Format-Table -AutoSize
$r.models.uniform_legal.most_common_prefixes_through_first_allin | Format-Table -AutoSize
~~~

Only after observing the dominant sequences should we design a controlled PPO/self-play ablation.

````````

<!-- END_MD_SOURCE path="docs/PHASE15_PREFLOP_SEQUENCE_AUDIT.md" -->
