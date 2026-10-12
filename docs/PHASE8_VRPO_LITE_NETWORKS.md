# Phase 8 — VRPO-Lite Transformer Actor & Centralized Q-Critic

> Baseline: `TheMostEvident/holdem-ai` main, Phase 7 complete (`5744df5` 확인 기준).  
> OS / hardware: Windows 10 · i7-8700 · 16 GB RAM · GTX 1070 8 GB · PyTorch 2.6.0+cu118.  
> **현재 구현은 네트워크 및 추론/Gradient 검증 단계. PPO나 VRPO 정책 최적화는 아직 미구현.**

## 1. 기존 시스템과 연결

```text
Phase 6: HUNL200Adapter
  ├── InformationState (only acting player's information)
  │    └── Phase 7: encode_actor → batch_actors
  │          └── Phase 8: TransformerActor → masked action logits [B, 7]
  └── CriticState (training_state() opt-in; includes both hole-card hands)
       └── Phase 7: encode_critic → batch_critics
             └── Phase 8: CentralizedQCritic → Q-values [B, 2, 7]
```

Actor에는 `CriticBatch`를 전달할 수 없게 인터페이스에서 거부한다. `NeuralActorPolicy.act(InformationState, rng)`는 오직 `encode_actor()`를 호출한다. `train_state()`를 Actor에서 호출하지 않는다.

## 2. 수학적 구조

Actor 입력:

$$
I_t^i=O_i(s_t,h_t^{\mathrm{obs}})
$$

$$
X_A(I_t^i)=(T_A,V_A,M_A,F_A,L_A)
$$

- $T_A$: Token IDs. Phase 7 어휘 $|\mathcal V|=73$.
- $V_A$: Token별 보조 금액 값(BB 단위).
- $M_A$: Token 위치별 Padding/Attention Mask.
- $F_A\in\mathbb{R}^{20}$: 플레이어가 관측하는 공개 수치 특징.
- $L_A\in\{0,1\}^7$: 유효한 Poker Action 마스크.

Transformer Encoder:

$$
Z=\mathrm{Transformer}_{\mathrm{causal}}\left( E_{\mathrm{token}}(T_A)+E_{\mathrm{pos}}+W_v\mathrm{asinh}(V_A)\right)
$$

$$
h_t=\mathrm{LayerNorm}(Z_{\mathrm{last\ valid}})
$$

수치적 안정성을 위해 BB 단위 값은 다음과 같이 변환한다.

$$
\mathrm{asinh}(x)=\ln\left(x+\sqrt{x^2+1}\right)
$$

Token 금액뿐 아니라 Actor/Critic의 Numeric Features에도 같은 함수를 적용한다. 200BB 스택처럼 큰 입력 때문에 초기 무작위 Logit이 지나치게 커지는 문제를 억제한다. 원본 VRPO 논문의 필수 정규화 방식이 아니라 **VRPO-Lite 고유의 구현 선택**이다.

Actor Action Logits:

$$
z_\theta=\mathrm{MLP}_A\left([h_t,\mathrm{asinh}(F_A)]\right)\in\mathbb{R}^7
$$

$$
\pi_\theta(a\mid I_t^i)=\frac{L_A(a)\exp z_\theta(a)}{\sum_b L_A(b)\exp z_\theta(b)}
$$

Critic:

$$
Q_{\phi,0}(s,a)=\mathrm{MLP}_Q\left([h^Q_t,\mathrm{asinh}(F_Q)]\right)_a
$$

$$
Q_{\phi,1}(s,a)=-Q_{\phi,0}(s,a)
$$

HUNL 제로섬 보상 조건에 맞춰 두 플레이어의 Q 값을 반대 부호로 제한한다. **이는 HUNL의 2인·무레이크·칩수익 제로섬 상황에서만 적용**한다. 6-Max/ICM 확장 시 그대로 쓰면 안 된다.

Terminal 상태에서는 Bootstrap Value가 0이어야 하므로:

$$
Q_\phi(s_T,a)=0
$$

`CentralizedQCritic`은 마지막 Feature인 `is_terminal`을 읽어 이를 명시적으로 적용한다. Critic 값은 BB 단위의 기대수익을 목표로 학습하도록 설계했지만, 아직 **실제 수익을 정확히 예측하도록 학습된 상태는 아니다**.

## 3. GTX 1070 초기 구성

| 하이퍼파라미터 | 설정 |
| --- | ---: |
| Actor Transformer Layers | 2 |
| Critic Transformer Layers | 2 |
| Embedding Dimension | 64 |
| Attention Heads | 2 |
| Feed-Forward Dimension | 128 |
| Head MLP Hidden | 128 |
| Max Tokens | 256 |
| Dropout | 0.0 |
| Precision | FP32 |
| Actor Parameters | 99,975 |
| Critic Parameters | 100,487 |
| Actor Output | `[B,7]` |
| Critic Output | `[B,2,7]` |

Actor/Critic은 서로 별개의 Token Encoder 가중치를 가진다. Causal Attention과 오른쪽 Padding을 사용하고, 마지막 유효 Token을 요약 벡터로 사용한다. Phase 7에서 입력 이력이 256개 Token을 넘으면 잘라내지 않고 예외가 발생한다.

## 4. Phase 8 프로그램 파일

```text
holdem_agent_phase8/
  __init__.py
  network.py   # NetworkConfig, CausalTokenEncoder, TransformerActor,
               # CentralizedQCritic, chosen_q
  policy.py    # NeuralActorPolicy(Phase-5 AgentPolicy-compatible)
scripts/
  phase8_smoke.py

tests/
  test_phase8_networks.py
  test_phase8_gpu.py
```

## 5. 로컬 실행 방법

ZIP은 `holdem-ai-phase8-networks` 이름으로 압축 해제하는 것을 가정한다. **프로젝트 루트에서 실행**한다.

```powershell
Copy-Item .\holdem-ai-phase8-networks\holdem_agent_phase8 .\ -Recurse -Force
Copy-Item .\holdem-ai-phase8-networks\tests\test_phase8_*.py .\tests\ -Force
Copy-Item .\holdem-ai-phase8-networks\scripts\phase8_smoke.py .\scripts\ -Force
Copy-Item .\holdem-ai-phase8-networks\docs\PHASE8_VRPO_LITE_NETWORKS.md .\docs\ -Force
```

```powershell
uv run --extra dev --extra gpu python -m pytest -q -k phase8
uv run --extra dev --extra gpu python -m pytest -q
uv run --extra dev --extra gpu python -m scripts.phase8_smoke --hands 20 --batch-size 16 --device cuda
```

검증용 `phase8_smoke`는 20개의 실제 HUNL 핸드를 수집하고 네트워크 두 개의 Forward/Backward를 한 번 실행하며, 무작위 초기 Actor로 실제 한 핸드도 플레이한다. **Optimizer Step을 실행하지 않으므로 실제 강화학습 모델의 성능을 주장할 수 없다.**

출력의 `cuda_peak_allocated_mb`는 PyTorch가 예약한 전체 VRAM이 아니라 해당 프로세스의 최대 Tensor 할당량이다. GPU 사용량 전체는 `nvidia-smi`에서도 확인할 수 있다.

### 검증 범위

- 입력/출력 차원, Legal Mask, Probability 합=1, 불법 행동 미샘플링
- Actor가 `CriticBatch`를 거부함
- Critic 제로섬 값, Terminal Value 0, 올바른 Player/Action Q 선택
- Padding 안정성, Token 금액 채널, Gradient, 가중치 저장 및 로드
- CPU 격리 시험, CUDA 가능 시 GPU 시험

### 알려진 한계

- 이 구조는 **VRPO-Lite**로, 논문 전체 네트워크나 Base-16 Action Abstraction을 재현한 것은 아니다.
- Phase 6 이력은 일부 All-in / Call 액수 정보를 잃는다. Phase 8에서 복구하지 않는다.
- 현재 `NeuralActorPolicy`는 Decision마다 한 샘플씩 GPU 추론한다. Phase 11의 대량 Self-Play에는 추론 배치화가 필요하다.
- Actor에게 상대 홀카드가 입력으로 전달되지 않도록 모듈과 계약으로 분리했지만, 프로젝트 외부의 임의 호출자가 ActorBatch를 악의적으로 조작하는 것을 방지하는 보안 경계는 아니다.
- Phase 9 PPO/GAE, Phase 10 VRPO Q-Boosting 및 Target Network/Replay는 이후 단계에서 구현한다.

## 6. Git 반영 (테스트 통과 후)

```powershell
git add holdem_agent_phase8/
git add tests/test_phase8_*.py
git add scripts/phase8_smoke.py
git add docs/PHASE8_VRPO_LITE_NETWORKS.md
git diff --cached --stat
git commit -m "Implement Phase 8 VRPO-Lite transformer actor and Q-critic"
git push origin main
```

기존 압축해제 폴더와 개인 파일을 무조건 `git add .`로 포함시키지 않는다.
