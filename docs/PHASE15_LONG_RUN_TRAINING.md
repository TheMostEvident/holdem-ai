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
\operatorname{seed}(u,k)=\mathrm{rollout\_seed}+uH+k,\quad 0\le k<H
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
