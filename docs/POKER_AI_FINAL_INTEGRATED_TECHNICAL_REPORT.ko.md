---
title: "Poker AI 최종 통합 기술보고서 — Environment · Agent · Expand · GUI · Tournament"
date: 2026-10-13
status: lossless-source-verified
tags: [HUNL, Environment, PPO, QBoost, MCCFR, Tournament, Obsidian]
---

# Poker AI 최종 통합 기술보고서

[English](POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md) | **한국어**

```math
\boxed{\mathcal G=(N,H,Z,A,P,f_c,\{\mathcal I_i\}_{i\in N},\{u_i\}_{i\in N})}
```

```math
\boxed{s_{t+1}=T(s_t,a_t,\omega_t),\quad a_t\in A_{\mathrm{legal}}(s_t),\quad a_t\sim\pi_\theta(\cdot\mid I_{i,t})}
```

$`N`$은 플레이어, $`H`$는 게임 이력, $`Z`$는 종료 이력, $`A`$는 행동, $`P`$는 행동 주체 선택, $`f_c`$는 카드 배분 확률, $`\mathcal I_i`$는 플레이어 $`i`$의 정보집합, $`u_i`$는 보상이다. 전체 게임 상태 $`s_t`$로 규칙을 계산하는 **Environment**와, 자신에게 보이는 정보 $`I_{i,t}`$로 행동을 선택하는 **Agent**를 구분한다. $`\omega_t`$는 셔플·카드 배분 등 확률 사건이다.

이 보고서는 [TheMostEvident/holdem-ai](https://github.com/TheMostEvident/holdem-ai)의 전체 개발 자료를 연결한 기술 레퍼런스다. 각 Project와 Phase의 **목적, 수식, 실제 Python 구현, 입력·출력, 실행 과정, 당시 검증 결과와 현재 한계**를 설명한다. Python **327개 전체 원문**과 Markdown **76개 전체 원문**은 아래 부록에 포함된다. 부록은 이 최종 보고서의 구성 요소이며, 별도 요약이나 별도 백업이 아니다.

## 본문 찾기

- [읽는 순서와 Project 경계](#section-2f84bd477486)
- [기준 자료와 증거 수준](#section-a030f2cc49ef)
- [모든 Phase를 연결하는 불변식](#section-7767092c3272)
- [두 기준 보고서만으로 모든 내용이 보존되는가](#section-109058370c3c)
- [Base Environment의 목적과 단위](#section-64c943b181ee)
- [Tournament ENV — 엔진을 사람이 사용하는 6인 대국 서비스로 연결](#section-28f568c64e55)
- [HUNL Agent — Base P5–P17의 수학, 구현과 검증](#section-842d126f6ae5)
- [Expand 프로젝트 — E1~E7](#section-73301c2cf7bd)
- [HUNL GUI 프로젝트 — G1.0~G1.6](#section-299f12b6cbd0)
- [원문 통합의 검증과 기존 보고서 정리](#section-c5edcf32d2ad)
- [부록 A — Python 327개 전체 원문 색인](#section-a0fe31ec5c73)
- [부록 B — 기존 Markdown 76개 전체 원문 색인](#section-27594225cc64)

<a name="section-2f84bd477486"></a>

## 읽는 순서와 Project 경계

| Project | Phase | 목적 | 입력 → 출력 | 다른 Project와의 경계 |
|---|---|---|---|---|
| Base Environment | P0–P4 | 합법적인 포커 게임과 칩 정산을 구현 | 카드·스택·합법 행동 → 다음 상태·정산·순위 | 상대 전략을 학습하지 않는다. |
| HUNL Agent | P5–P17 | 2인 의사결정·학습·평가를 구현 | 자신의 정보집합 → 합법 행동 분포·학습 체크포인트 | 게임 규칙은 Environment에 맡긴다. |
| Expand | E1–E7 | 2–6인 Cash Game 연구를 확장 | 동적 좌석·다인 정보집합 → PPO·MCCFR·Search·상대 적응 | Tournament 전용 학습과 구분한다. |
| HUNL GUI | G1.0–G1.6 | Human–Agent 경기를 관찰하고 조작 | 사용자 입력·Frozen Actor → 게임 화면·통계 | GUI는 학습기가 아니다. |
| Tournament ENV | T1.1–T1.7 및 수정판 | 6인 Knockout 경기·GUI·이력·Replay 제공 | 토너먼트 설정·사용자/봇 행동 → 칩 순위·탈락·우승·기록 | Expand E3 체크포인트 연결은 토너먼트 최적 정책의 증명이 아니다. |

**P17의 두 갈래는 분리해서 읽는다.** 연속 스택 평가는 평가 방식의 변경이고, GPU Retrain은 별도의 학습 실험이다. HUNL 7-action Actor, 다인 Expand 7-action Actor, Tournament의 5-action Native Engine도 서로 같은 인터페이스가 아니다.

<a name="section-a030f2cc49ef"></a>

## 기준 자료와 증거 수준

원문 기준 커밋은 [`6a8ae368ddc02f814140d3770bc5993eaf9a15e4`](https://github.com/TheMostEvident/holdem-ai/tree/6a8ae368ddc02f814140d3770bc5993eaf9a15e4)이다. 중단된 부록 작업이 병합된 기준은 [`e796afefaf5f44dfa8fa8fe8bf8ad0fae436547e`](https://github.com/TheMostEvident/holdem-ai/commit/e796afefaf5f44dfa8fa8fe8bf8ad0fae436547e)이다. 본문은 실제 소스와 개별 Phase 문서를 대조해 재작성했고, 원문 부록은 원래 바이트를 유지한다.

| 증거 | 이 보고서에서의 의미 |
|---|---|
| 실제 구현 | 기준 커밋의 `.py` 코드에서 확인한 동작 |
| 당시 테스트·실험 기록 | 해당 Phase 문서에 남은 결과; 이번에 재실행한 결과가 아님 |
| 이번 검증 | 원문 파일별 바이트 수·Git Blob SHA·누락·중복·문서 연결 검증 |
| 미검증·연구 한계 | 아직 입증되지 않은 학습 강도, 실전 성능, 수렴·복구 보장 |

이번 문서 정리는 학습을 새로 실행하거나 게임 엔진을 수정하는 작업이 아니다. 학습 시간·검증 점수·테스트 개수는 각 기록의 맥락을 유지한다.

<a name="section-7767092c3272"></a>

## 모든 Phase를 연결하는 불변식

```math
\boxed{\sum_i S_{i,t}^{\mathrm{behind}}+P_t=C_h}
\qquad
\boxed{R_{i,h}^{BB}=\frac{S_{i,h}^{\mathrm{final}}-S_{i,h}^{\mathrm{start}}}{BB_h}}
```

핸드 $`h`$ 내부에서 레이크·외부 자금 이동이 없으면 잔여 스택의 합과 Pot의 합은 일정하다. 동적 Cash Table의 Join·Leave·Rebuy는 핸드 경계에서 별도 회계 이벤트로 처리한다. $`R^{BB}`$는 칩 손익을 해당 핸드의 Big Blind로 정규화한 수익이며, 순위 보상·상금 효용과 구별한다.

```math
\boxed{\pi_\theta(a\mid I)=
\frac{\mathbf1[a\in A(I)]\exp\ell_\theta(I,a)}
{\sum_{b\in A(I)}\exp\ell_\theta(I,b)}}
```

$`\ell_\theta`$는 Actor Logit이고 $`A(I)`$는 관측 시점의 합법 행동 집합이다. 합법 행동 Mask를 적용한 뒤에 정규화한다. 학습용 Critic이 전체 상태를 읽을 수 있어도 Actor 입력에는 상대의 비공개 카드가 포함되어서는 안 된다.

```math
\boxed{m_{E\to T}(0,1,2,3,4,5,6)=(0,1,1,2,3,\bot,4)}
```

Expand의 Fold·Check·Call·Half-Pot·Pot·Double-Pot·All-In을 Tournament의 Fold·Check/Call·Half-Pot·Pot·All-In으로 옮기는 대응이다. $`\bot`$은 지원하지 않는 행동이다. 이름과 합법 문맥으로 대응하고, 지원하지 않는 확률은 제거한 후 다시 정규화한다. 두 모델의 정수 Index가 같다는 이유만으로 같은 행동이라고 해석하면 안 된다.

<a name="section-109058370c3c"></a>

## 두 기준 보고서만으로 모든 내용이 보존되는가

두 문서는 전체 Phase를 이해하는 출발점이지만, 나머지 보고서의 세부 인터페이스·설계·실험 기록까지 문자 그대로 포함하는 문서는 아니다. 따라서 본문의 Phase별 설명을 보충하고, 원문 전체를 검증된 부록으로 포함한 후 독립 보고서들을 정리했다.


| 원 Phase 문서와 확인 위치 | 원문에 있는 구체적인 세부사항 | Agent 통합보고서의 설명 범위 | Tournament 통합보고서의 설명 범위 / 차이 |
|---|---|---|---|
| [docs/EXPAND_PHASE02_INFORMATION_AND_ENCODING.md](report_sources/ORIGINAL_REPORTS_01.md#md-016) §2 | 동일 raise-to 목표액은 `HalfPot → Pot → DoublePot → AllIn` 우선순위로 중복 제거한다. 상하한을 벗어난 목표액은 조용히 clamp하지 않고 mask에서 제외한다. `raising_permitted`는 native ID 4를 권위로 쓰고 Half/Pot은 native ID 2/3도 확인한다. | Part II §4.2는 7-action 순서, short-all-in call 정규화, 정수 목표액과 masking을 설명한다. 이 중복 제거 우선순위·clamp 금지·native ID 판정 세부조건까지 제시하지 않는다. | §7.4는 E3의 7→5 행동 변환을 상세히 설명한다. E2의 raise 후보 생성·중복 제거 규칙은 별도 내용이다. |
| [docs/EXPAND_PHASE02_INFORMATION_AND_ENCODING.md](report_sources/ORIGINAL_REPORTS_01.md#md-016) §3 | Seat `[6,19]`, Global `[110]`, History `[128,21]` 계약; occupied sit-out 좌석도 attention에 포함; 빈 좌석만 padding; 최근 128개 이력만 neural 입력에 사용; BB 정규화와 부호 보존 Log1p; suit rotation이 모델의 suit invariance를 뜻하지 않음. | Part II §4.3은 Seat/History shape와 occupied/dealt_in 구분을 보존한다. Global 110, 최근 이력 절단, Log1p, suit invariance 제한은 그 요약에 없다. | E3 actor를 연결하는 서비스 문서로, E2 encoder의 이 수치 전처리 계약을 제시하지 않는다. |
| [docs/EXPAND_PHASE04_MCCFR_BLUEPRINT.md](report_sources/ORIGINAL_REPORTS_01.md#md-018) §3.1–3.2 | `w_t=floor((t-1)/N)+1`; 평균전략 누적 `S_i(I,a) += w_t * π_i(I) * σ_i(I,a)`의 own-reach weight; 상대/chance 방문 sampling과 weight를 중복 적용하지 않음; `(seed, global_iteration)` 해시로 iteration RNG 재생성. | Part II §6은 external sampling, linear weighting, regret matching, 부분 iteration 저장 방지와 Kuhn 검증을 설명한다. 평균전략 own-reach 식과 정확한 weight schedule/RNG 재생성 계약은 없다. | MCCFR solver 자체를 설명하는 범위가 아니며 이 평균전략·RNG 계약을 포함하지 않는다. |
| [docs/EXPAND_PHASE05_SUBGAME_SEARCH.md](report_sources/ORIGINAL_REPORTS_01.md#md-019) §2.1, §3 | 기본 belief는 공개 행동 likelihood 가중 없이 후보 덱을 sampling한다. 선택적 `opponent_likelihood(observation, action_id, raise_to)`는 해당 상대 관측에서 확률을 계산해 rejection sampling한다. 메모리 예산은 infoset당 4096바이트의 테이블 추정이며 실제 RSS 상한이 아니다. | Part II §7은 일반 Bayesian 식, 카드 일관성, 정확한 range 추정의 한계와 추정 메모리 예산을 설명한다. 기본 likelihood 비가중 경로, 선택적 callback/rejection 경로, 4096바이트 계수는 구체화하지 않는다. | Search belief와 자원 추정의 상세 구현 계약은 포함하지 않는다. |
| [docs/PHASE16_PPO_ABLATION.md](report_sources/ORIGINAL_REPORTS_04.md#md-045) §Arms, §Fairness controls | 모든 arm에서 Critic Adam moments를 reset한다. 따라서 A0도 Phase15의 bit-for-bit resume가 아니다. A2는 Q token/hidden layers를 warm-start하고 평균 Q heads로 projected V output을 초기화하며 출력 parameter 수가 줄어든다. | Part V §16.2는 A0–A3 차이, Q/200 scaling, direct V, KL rollback을 설명한다. 전 arm의 Critic Adam reset과 A2의 구체적 초기화/parameter 차이는 제시하지 않는다. | Agent optimizer ablation을 다루지 않는다. |
| [docs/PHASE16_PPO_ABLATION.md](report_sources/ORIGINAL_REPORTS_04.md#md-045) §Safety and data integrity | `final_actor.pt`에는 Actor만 있어 Critic과 두 optimizer states가 없으며 PPO resume checkpoint로 쓸 수 없다. Pilot은 auto-resume를 지원하지 않고 partial run은 보존한 채 새 root로 재시작한다. | Part V는 ablation 실행과 후속 stress의 full-state `latest.pt`/resume를 설명한다. ablation의 Actor-only 파일과 stress의 full-state 파일을 이 수준으로 구별하는 계약은 없다. | E3 tournament actor checkpoint 설명은 있지만 이 Base P16 Actor-only 파일의 resume 제한은 없다. |
| [docs/PHASE17_FIVE_HOUR_GPU_RETRAIN.md](report_sources/ORIGINAL_REPORTS_04.md#md-047) §Objective and diagnostics, §Time budget | A0는 optimizer step 이후 KL>0.10이면 `hard_stop.json`을 생성하고 automatic resume를 거부한다. A3는 KL>0.02 rollback이다. Resume 시 scientific manifest가 모두 같아야 하며 `--max-updates-this-call`만 변경할 수 있다. | Part V/VI는 A0 extreme KL stop, A3 rollback, manifest와 resume를 설명한다. P17의 `hard_stop.json`/자동 resume 거부와 허용되는 단 하나의 옵션 변경까지 규정하지 않는다. Part I의 P15 12회 연속 경고와 같은 계약으로 오인하면 안 된다. | Tournament journal/replay와는 다른 학습 재개 계약이며 포함하지 않는다. |
| [docs/GUI_1_3_INTERACTIVE_BETTING.md](report_sources/ORIGINAL_REPORTS_02.md#md-025) §New HTTP endpoint, §Acceptance criteria | HUNL `/raise-to` 요청 `{"raise_to":350,"expected_revision":17}`; 누락·분수·음수·최소 미달·최대 초과 금액은 거부하고 state와 revision을 바꾸지 않음; bot의 연속 자동 행동 때문에 한 응답에 여러 history 항목이 들어갈 수 있음. | Part VIII GUI 표는 custom raise, engine 검증, revision 409/illegal 422를 요약한다. 이 요청 예제·입력 실패 시 state/revision 불변 조건·응답 batch 의미까지 설명하지 않는다. | §1.3은 HUNL 선행 GUI를 요약하고 Tournament HTTP 계약은 별도로 상세 설명한다. 그것이 HUNL custom raise endpoint의 위 세부조건을 대체하지 않는다. |
| [docs/GUI_1_4_TRAINED_AGENT.md](report_sources/ORIGINAL_REPORTS_02.md#md-026) §Checkpoint API and file safety, §Validation and limits | `GET /bots`는 파일 존재만 광고하고 strict load는 session 생성 때 수행한다. malformed/incompatible 파일은 422이며 random fallback이 없다. Actor cache는 파일 size/mtime 변경으로 무효화하고 session별 독립 RNG를 쓴다. 당시 checkpoint tests는 mock loader여서 genuine checkpoint inference를 검증하지 않았다. | Part VIII은 frozen 추론, 모델 선택·경로 설정과 실제 로컬 체크포인트 연결 문제를 다룬다. HUNL catalog의 cache invalidation/RNG 조건과 GUI1.4 fixture의 mock loader 범위를 이와 같이 기록하지 않는다. | §7은 별도의 Tournament E3 loader와 format/action mapping을 설명한다. HUNL GUI1.4 cache/fixture 계약을 대신 포함한 것은 아니다. |
| [docs/TOURNAMENT_ENV_PHASE_1_3_1_AGGRESSION_FIX.md](report_sources/ORIGINAL_REPORTS_06A.md#md-064) §Verification from a partial compatibility harness | 30 seeded tournament/388 hand에서 engine seats 1–5의 raise(half/pot, all-in 제외) 빈도는 각각 **29.1%, 10.4%, 19.6%, 11.2%, 28.5%**다. 이는 partial harness의 조건부 행동 진단이며 win rate·완전 Windows repository 검증이 아니다. | Part VIII은 Call/Fold 편향과 pot-raise 조건 교정을 개괄한다. 이 seat별 원 측정 수치를 싣지 않는다. | §6은 `2 or 3` bug, heuristic thresholds, 1326 starting-hand 진단, 30시드/388핸드를 보존한다. 하지만 위 seat별 5개 raise 비율은 생략한다. |


이 표의 당시 문서 표기와 현재 구현이 다르면 본문에서는 실제 소스를 우선한다. 예를 들어 Expand E2의 원문 수식은 Global 110을 적었지만, 현재 Encoder의 실제 구성과 `GLOBAL_DIM`은 114다. 기존 표기 자체도 원문 부록에 그대로 남겨 추적할 수 있다.

<a name="section-64c943b181ee"></a>

## Base Environment의 목적과 단위

```math
\boxed{I_t^i=O_i(s_t),\quad a_t\in\mathcal A^{\mathrm{legal}}(I_t^i),\quad s_{t+1}=T(s_t,a_t),\quad R_i=\frac{S_i^{\mathrm{final}}-S_i^{\mathrm{initial}}}{BB}}
```

$`s_t`$는 덱·각 참가자의 홀 카드까지 포함하는 엔진 내부 상태이고, $`I_t^i`$는 행동자 $`i`$가 받을 수 있는 관측이다. $`T`$는 합법 행동을 받아 다음 게임 상태를 생성하고, $`R_i`$는 한 핸드 종료 후 BB 단위 칩 증감이다. 이 프로젝트의 Environment는 이 네 가지 계약을 먼저 만드는 작업이었다. 정책 학습은 이 계약 위에서 진행된다. 잘못된 블라인드·재레이즈·정산을 그대로 학습하면 정책은 실제 포커 대신 잘못된 시뮬레이터를 최적화하게 된다.

여기서 **Base Phase 1·2.1의 `HandEngine`**, **Base Phase 3의 `MultiPlayerHandEngine`**, **Base Phase 4의 `TournamentEnv`**는 서로 다른 실행 단위다. 첫 번째는 RLCard를 감싼 두 사람의 한 핸드, 두 번째는 독립 구현한 2~6인의 한 핸드, 세 번째는 정산 스택을 계속 승계하는 탈락형 토너먼트다. 같은 `reset()`이라도 첫 두 엔진은 한 핸드를 새로 만들고, `TournamentEnv.reset()`은 전체 토너먼트를 처음부터 시작한다.

### Base Phase 0 — 개발 기반과 책임 경계

```math
\mathrm{Run}=F(\mathrm{source\ revision},\mathrm{dependency\ lock},\mathrm{configuration},\mathrm{seed}).
```

`source revision`은 Git 커밋, `dependency lock`은 의존성 해석 결과, `configuration`은 스택·블라인드·모드, `seed`는 난수 초기값이다. 같은 실행을 다시 조사하려면 네 항목을 함께 관리해야 한다. Phase 0의 효용은 이후 엔진·학습·평가 파일이 하나의 저장소와 재현 가능한 Python 환경에서 동작하도록 만드는 데 있다.

역사적 기록은 Windows, Python 3.11, Git/GitHub, `uv`와 `.venv`를 사용했다고 설명한다. 프로젝트 이름은 `holdem-ai`, Python import 이름은 `holdem_ai`다. Environment·Agents·Training·Evaluation·UI의 책임을 분리하고, 가상환경과 모델 출력은 Git 추적에서 제외했다. 원격 저장소에 기존 이력이 있어 초기 Push 과정에서 서로 다른 Git 이력을 통합하고 README 충돌을 해결했다는 기록도 있다.

현 스냅샷에서 [src/holdem_ai/environment/game_state.py](technical_sources/PYTHON_SOURCE_07.md#py-082), [src/holdem_ai/agents/cfr_agent.py](technical_sources/PYTHON_SOURCE_06.md#py-063), [src/holdem_ai/agents/rl_agent.py](technical_sources/PYTHON_SOURCE_06.md#py-068)는 빈 파일이다. 파일이 존재한다는 이유만으로 게임 상태 클래스·CFR·RL Agent가 구현됐다고 해석하면 안 된다. 과거 문서가 언급한 `create_architecture.py`는 현 수록 소스에 없으며, 그 실행 코드를 복원해 실제 구현인 것처럼 제시하지 않는다. 패키지 초기화용 `__init__.py`와 이후의 `base_agent.py`·`random_agent.py`는 구분해서 읽어야 한다.

실행 흐름은 저장소의 의존성을 준비하고, 해당 Phase의 테스트·감사를 실행하며, 결과와 변경을 검토한 다음 커밋하는 방식이었다. 아래는 재실행 절차이며 이번 문서 작성 중 수행한 테스트 결과가 아니다.

```powershell
uv sync --extra dev
uv run --extra dev python -m pytest -q
```

Phase 0은 연구 방법과 개발 환경의 출발점이며 포커 규칙 검증이나 정책 성능 실험은 포함하지 않는다.

### Base Phase 1 — RLCard 두 사람 한 핸드의 실행 계약

```math
\pi_{\mathrm{random}}(a\mid I)=\frac{\mathbf1[a\in\mathcal A^{\mathrm{legal}}(I)]}{|\mathcal A^{\mathrm{legal}}(I)|},\qquad
\widehat J_i^{bb/100}=\frac{100}{H}\sum_{h=1}^{H}\frac{\Delta S_{i,h}}{BB}.
```

$`H`$는 플레이한 핸드 수, $`\Delta S_{i,h}`$는 해당 핸드의 칩 증감이다. Random Agent는 합법 행동 중 균등 선택한다. 이 단계는 `reset → observation → action → step → payoff`가 실제로 연결되는지 확인하기 위한 최소 기준선이다.

| 실제 Python 파일 | 담당 역할 |
|---|---|
| [src/holdem_ai/environment/hand_engine.py](technical_sources/PYTHON_SOURCE_07.md#py-083) | `HandEngine`과 불변 결과 `HandResult`; 초기화·합법 행동·한 핸드 실행·BB 수익 변환 |
| [src/holdem_ai/agents/base_agent.py](technical_sources/PYTHON_SOURCE_06.md#py-062) | `Agent.select_action(state) -> int` 프로토콜 |
| [src/holdem_ai/agents/random_agent.py](technical_sources/PYTHON_SOURCE_06.md#py-067) | 독립 seeded RNG로 합법 행동을 선택하는 `RandomAgent` |
| [main.py](technical_sources/PYTHON_SOURCE_01.md#py-007) | CLI 옵션·반복 핸드·각 좌석 수익 및 관측 bb/100 출력 |
| [tests/test_hand_engine.py](technical_sources/PYTHON_SOURCE_22.md#py-264), [tests/test_rlcard_integration.py](technical_sources/PYTHON_SOURCE_27.md#py-315) | 어댑터 계약 및 실제 RLCard 연결 검사 |

핵심 선택 코드는 실제 `random_agent.py`에서 발췌했다.

```python
    def select_action(self, state: Mapping[str, Any]) -> int:
        legal_actions = tuple(state['legal_actions'])
        if not legal_actions:
            raise ValueError('There are no legal actions in this state.')
        return int(self._rng.choice(legal_actions))
```

[main.py](technical_sources/PYTHON_SOURCE_01.md#py-007)는 두 Agent를 만든 다음 `engine.play_hand(agents)`를 반복한다. `HandEngine.play_hand()`는 행동자의 관측만 정책에 넘기고, `step()`은 받은 정수 ID가 현 합법 행동에 있는지 먼저 확인한다. `get_result()`는 종료 후 수익 합이 0인지 확인하고 칩 수익을 BB로 나눈다. 네트워크·CFR·확률 계산기는 이 루프의 필수 조건이 아니다.

현 `hand_engine.py`는 Phase 2.1 이후 버전으로, 기본 `rules_mode='corrected'`이며 `rules_mode='upstream'`일 때 원본 RLCard를 사용한다. `reset()`은 `_next_button`을 반전시켜 연속 호출에서 버튼을 교대한다. 이를 Phase 1 최초 코드의 기능과 혼동하지 않는다.

역사적 기록은 **5 tests passed**, 5핸드 smoke와 좌석 0의 $`-50\;bb/100`$을 남겼다. 기록된 수익으로는 $`100(-0.5-0.5+1-2-0.5)/5=-50`$이다. 이는 연결 확인을 위한 작은 표본이며 정책의 평균 실력 추정치로 사용할 수 없다. RLCard의 54차원 `obs`도 완전한 Information State 또는 perfect recall을 증명하지 않는다. 원래 다섯 행동은 `0=fold, 1=check/call, 2=half-pot, 3=pot, 4=all-in`이다.

### Base Phase 2 — 규칙을 독립 명세로 만들고 원본 엔진을 감사

```math
\mathrm{SB}=b,\quad\mathrm{BBseat}=1-b,\quad
\mathrm{First}_{preflop}=b,\quad\mathrm{First}_{postflop}=1-b.
```

```math
k_i=\min(S_i,h-c_i),\qquad
\ell=\begin{cases}BB,&h=0\\h+r,&h>0,\end{cases}\qquad
u_i=c_i+S_i.
```

```math
\mathcal N_q=\{i:C_i\ge v_q\},\qquad
L_q=(v_q-v_{q-1})\,|\mathcal N_q|,\qquad
E_q=\{i\in\mathcal N_q:\neg\mathrm{folded}_i\}.
```

```math
P_q=L_q\mathbf1[|\mathcal N_q|\ge2],\qquad
U_i=\sum_q L_q\mathbf1[\mathcal N_q=\{i\}],\qquad
\sum_q P_q+\sum_i U_i=\sum_i C_i.
```

$`b`$는 두 사람 게임의 버튼, $`S_i`$는 베팅 전 남은 스택, $`c_i`$는 현재 스트리트 누적 베팅, $`h`$는 스트리트 최대 베팅, $`r`$은 직전 full raise 증가량이다. $`k_i`$는 실제 콜 비용, $`\ell`$은 최소 full raise-to, $`u_i`$는 최대 raise-to다. $`C_i`$는 핸드 전체 기여액, $`v_q`$는 정렬한 양의 기여액 수준이며 $`v_0=0`$이다. $`\mathcal N_q`$는 해당 층의 기여자 집합, $`L_q`$는 층 전체 칩 수, $`E_q`$는 수상 자격자다. $`P_q`$는 기여자가 둘 이상인 층을 실제 팟으로 유지하는 금액이며, 기여자가 한 명인 층에서는 0이다. $`U_i`$는 $`i`$ 혼자 기여한 층의 금액 합으로, 해당 칩은 Fold 여부와 관계없이 $`i`$에게 반환된다. 유효한 정산에서는 기여자가 둘 이상인 각 팟의 $`E_q`$가 비어 있지 않아야 하며, 실제 `build_side_pots()`도 빈 자격자 팟을 오류로 거부한다.

Phase 2는 RLCard 결과를 그대로 신뢰하지 않고 위치·베팅·정산을 확인할 독립 기준을 만들었다. 이 시점의 helper는 upstream 내부 실행을 고치지 않는다. 감사가 실패해야 할 사례에서 실패하는 것이 진단기의 역할이다.

| 실제 Python 파일 | 제작 목적과 핵심 함수 |
|---|---|
| `environment/seat_manager.py` | `heads_up_seats`, `post_blinds`; BTN=SB와 짧은 스택의 블라인드 올인 기준 |
| `environment/actions.py` | `BetContext`, `validate_raise_to`, `legal_basic_actions`; full raise·short all-in·재오픈 허가 검증 |
| `environment/pot_manager.py` | `build_side_pots`, `settle_pots`; 층별 자격·uncalled 반환·분할팟·odd chip |
| `environment/hand_evaluator.py` | `parse_card`, `rank_five`, `rank_seven`, `rank_showdown`; 독립 족보 평가 |
| `environment/rules_audit.py` | raw RLCard 상태를 읽는 비변경 감사 |
| [scripts/audit_rlcard.py](technical_sources/PYTHON_SOURCE_01.md#py-010) | 위반 발견 시 nonzero exit를 내는 원본 엔진 CLI 감사 |
| [tests/environment/test_position.py](technical_sources/PYTHON_SOURCE_18.md#py-210), `test_betting.py`, `test_side_pot.py`, `test_hand_evaluator.py` | 각 순수 규칙 검증 |
| [tests/test_rules_audit.py](technical_sources/PYTHON_SOURCE_27.md#py-317), [tests/test_rlcard_rules_integration.py](technical_sources/PYTHON_SOURCE_27.md#py-316) | 감사기 및 upstream 통합 검사 |

위 표의 `environment/`는 `src/holdem_ai/environment/`를 뜻한다. 아래는 실제 `validate_raise_to()`의 판정 부분이다.

```python
    if not ctx.may_raise:
        raise ValueError('betting has not reopened for this player')
    if total <= ctx.highest_commitment:
        raise ValueError('raise must exceed the current highest bet')
    if total > ctx.max_total:
        raise ValueError('raise exceeds available stack')
    if total >= ctx.min_full_raise_to:
        return 'full_raise'
    if total == ctx.max_total:
        return 'short_all_in_raise'
```

Short all-in은 최소 full raise에 미달하더라도 전체 스택을 투입하는 경우 허용된다. 그것이 이전 행동자에게 재레이즈 권리를 돌려주는지는 엔진이 별도로 추적한다. 순수 helper의 `may_raise`를 올바르게 설정하는 책임도 호출자에게 있다.

족보 평가의 수학적 정의는 다음과 같다.

```math
\mathrm{rank}_7(H_i\cup B)=\max_{X\subseteq H_i\cup B,\ |X|=5}\mathrm{rank}_5(X),\qquad \binom75=21.
```

카드는 사용자 표기 `sK`, `d7`, `cT`, `hA`를 사용한다. `rank_five()`는 `(족보, 주요 rank, kicker...)` 튜플을 반환하고 더 큰 튜플이 이긴다. `rank_seven()`은 실제로 21개 조합의 최대를 구하며 A-2-3-4-5 스트레이트와 중복 카드 거부를 포함한다. `settle_pots()`는 동점 팟을 정수 나눗셈으로 분배하고 남는 칩을 버튼 왼쪽부터 시계 방향으로 준다.

역사적 기록은 **27 passed / 1 xfailed**, upstream의 위치·블라인드 문제 세 가지를 보고했다. 100핸드에서 수익 합이 0이어도 BTN·SB·선행 행동은 틀릴 수 있었다. 따라서 칩 보존과 규칙 정확성은 각각 검증해야 한다. 당시 upstream `xfail`은 알려진 위반을 표시하는 예상 실패이며, 독립 규칙 테스트의 실패와 다르다. 외부 엔진과 전수 차등 비교를 수행했다는 증거는 없다.

### Base Phase 2.1 — 프로젝트 내부의 Corrected HUNL adapter

```math
\mathrm{RaiseTo}_f=c_i+d_i+\lfloor f(P+d_i)\rfloor,\quad f\in\{\tfrac12,1\},\qquad
\Delta\mathrm{raise}\ge r\Rightarrow\mathrm{full\ raise}.
```

$`d_i`$는 해당 결정의 미충족 베팅액, $`P`$는 이미 투입된 전체 팟이다. Half-pot/Pot raise는 먼저 콜했을 때의 팟을 기준으로 목표 총액을 계산한다. 실제 소스의 칩 반올림은 `int()`이며 양수 계산에서는 내림과 같다. 베팅 ID는 Phase 1의 다섯 개를 유지했다.

목적은 앞 Phase가 찾은 문제를 실행 경로에서 고치는 것이다. 설치된 `.venv` 내부 RLCard를 수정하는 대신, 프로젝트 안의 subclass와 독립 betting round를 주입한다. RLCard 버전이 달라질 때 호환성을 재검증하도록 `rlcard==1.2.0` guard를 둔다.

| 실제 Python 파일 | 변화와 작동 역할 |
|---|---|
| `environment/corrected_game.py` | `make_corrected_rlcard_env`; local `CorrectedGame`, 블라인드 재게시·즉시 Fold 종료·독립 정산·step-back 참조 복구 |
| `environment/corrected_round.py` | `ActionId`, `HeadsUpBettingRound`; pending 행동자·full raise 이후 행동 이력·재오픈 제약 |
| `environment/hand_engine.py`, [main.py](technical_sources/PYTHON_SOURCE_01.md#py-007) | 기본 모드를 corrected로 변경하고 upstream 진단 경로 유지 |
| [scripts/audit_corrected.py](technical_sources/PYTHON_SOURCE_01.md#py-008) | 수정된 엔진 위치·수익 감사 |
| [tests/test_corrected_round.py](technical_sources/PYTHON_SOURCE_22.md#py-256), [tests/test_corrected_rlcard_integration.py](technical_sources/PYTHON_SOURCE_22.md#py-255), [tests/test_hand_engine.py](technical_sources/PYTHON_SOURCE_22.md#py-264) | betting round·실제 RLCard 연결·기본 wrapper 회귀 |

아래는 실제 `CorrectedGame.init_game()`의 블라인드 복구 및 재게시 부분이다.

```python
            for player in self.players:
                player.remained_chips += player.in_chips
                player.in_chips = 0
                player.status = PlayerStatus.ALIVE
            btn = int(self.dealer_id)
            bb = 1 - btn
            self.players[btn].bet(chips=self.small_blind)
            self.players[bb].bet(chips=self.big_blind)
            for player in self.players:
                if player.remained_chips == 0:
                    player.status = PlayerStatus.ALLIN
            self.game_pointer = btn
```

호출 흐름은 `HandEngine.__init__ → make_corrected_rlcard_env → CorrectedGame.init_game → HeadsUpBettingRound`다. `init_game()`은 upstream의 seeded deck·카드 배분·플레이어 생성은 재사용한 뒤 잘못 게시된 블라인드를 회수해 다시 게시한다. `dealer.pot`도 새 기여액 합으로 갱신해 첫 pot-size 행동의 계산을 맞춘다. `proceed_round()`는 full raise이면 이전 행동 집합을 재설정하고, short all-in이면 상대의 응답은 요구하되 이전 행동자의 재레이즈 권리를 자동 복구하지 않는다. `CorrectedGame.step(FOLD)`는 핸드를 즉시 종료해 의미 없는 다음 공용 카드를 배분하지 않는다. `get_payoffs()`는 Phase 2의 side pot·족보 평가로 배당을 계산한다.

역사적 기록은 **43 passed / 1 xfailed**, 수정 엔진의 100핸드 감사를 남겼다. upstream 자체를 바꾸지 않았으므로 원본 감사의 알려진 실패는 남는 구조다. 이 어댑터는 두 사람 한 핸드와 RLCard 1.2.0에 한정되며, 이후 6인 게임을 이 subclass로 확장하지 않았다. 남은 검증은 adversarial 재레이즈·부분 블라인드·외부 엔진과의 비교다.

### Base Phase 3 — 독립적인 2~6인 한 핸드 엔진

```math
\mathcal O=\{i:S_i^{\mathrm{initial}}>0\},\quad
\begin{cases}
\mathrm{SB}=b,\ \mathrm{BBseat}=\mathrm{next}(b),&|\mathcal O|=2,\\
\mathrm{SB}=\mathrm{next}(b),\ \mathrm{BBseat}=\mathrm{next}(\mathrm{SB}),&|\mathcal O|\ge3,
\end{cases}
```

```math
\mathrm{PreFirst}=\begin{cases}b,&|\mathcal O|=2\\\mathrm{next}(\mathrm{BBseat}),&|\mathcal O|\ge3,\end{cases}\qquad
\mathrm{PostFirst}=\mathrm{next}(b).
```

```math
\sum_i S_{i,t}^{behind}+\sum_i C_{i,t}=\sum_i S_i^{initial},\qquad
\Delta S_i=A_i-C_i,\qquad S_i^{final}=S_i^{initial}+\Delta S_i,\qquad\sum_i\Delta S_i=0.
```

$`\mathrm{next}`$는 시계 방향의 다음 점유 물리 좌석, $`C_i`$는 전체 기여액, $`A_i`$는 unmatched 반환을 포함한 배당이다. Phase 3는 upstream이 제공하지 못한 2~6인 베팅과 서로 다른 스택의 side pot을 직접 관리한다. 그 효용은 Agent·토너먼트·브라우저가 같은 정산 권위를 재사용하는 데 있다.

| 실제 Python 파일 | 담당 구현 |
|---|---|
| `environment/multi_player_engine.py` | `MultiPlayerHandEngine`, `MultiPlayerHandResult`; 덱·스트리트·pending·법적 행동·raise-to·정산 |
| `environment/multi_player_seats.py` | `next_occupied`, `multi_player_seats`, `rotate_button`; 물리 좌석 및 2인 전환 |
| `environment/hand_evaluator.py`, `environment/pot_manager.py` | Phase 2의 정확 족보와 층별 배당 재사용 |
| [scripts/audit_multiplayer.py](technical_sources/PYTHON_SOURCE_01.md#py-009) | 랜덤 핸드·칩 보존·비음수 스택·배당 합 감사 |
| [tests/test_multi_player_engine.py](technical_sources/PYTHON_SOURCE_23.md#py-265), [tests/test_multi_player_seats.py](technical_sources/PYTHON_SOURCE_23.md#py-266) | 다인 실행·위치·베팅·all-in·side pot 회귀 |

`step_raise_to(amount)`의 `amount`는 추가로 밀어 넣는 칩이 아니라 **현재 스트리트에 이미 낸 칩까지 포함한 목표 총액**이다. Native ID `0~4` 외의 임의 금액 행동은 이력에 ID `5`와 실제 target을 함께 저장한다. 두 이력을 함께 보존해야 replay나 학습에서 정확한 베팅을 복원할 수 있다.

실제 누적 short all-in 재오픈 판정은 다음과 같다.

```python
    def _raise_permitted(self, i: int) -> bool:
        # No betting into players who are already all-in/folded.
        if len(self._active()) < 2:
            return False
        if self.stacks[i] <= self._to_call(i):
            return False
        if i not in self.last_acted_high:
            return True
        # Multiple short all-ins cumulatively reopen action when >= full raise.
        return self._current_high() - self.last_acted_high[i] >= self.last_full_raise
```

이를 수식으로 쓰면 이전 행동 시점 최대 베팅 $`h_i^{last}`$에 대해 $`h_t-h_i^{last}\ge r_t`$일 때 재레이즈를 허용한다. 여러 짧은 올인 증가량의 누적이 full raise 크기에 도달했는지를 추적하는 구현이다. 단일 short all-in만으로 권리를 재설정하지 않는다.

`reset()`은 `initial_stacks`를 복원하고 52장 덱을 섞어 버튼 왼쪽부터 한 장씩 두 번 배분한다. 블라인드에는 `min(blind, stack)`을 사용한다. `_begin_action()`은 행동할 수 있는 좌석을 `pending`에 넣으며, 혼자 남은 active 플레이어는 올인 상대에게 콜 비용이 있을 때만 행동한다. `_current_high()`는 프리플랍에서 active 플레이어가 둘 이상이면 짧은 BB 올인에도 명목 BB 이상의 bring-in을 유지한다.

`step()` 또는 `step_raise_to()`는 `_apply_action()`으로 들어간다. 베팅 후 미응답자를 갱신하고 `_progress_if_closed()`가 Fold 승리·다음 스트리트·올인 보드 runout을 결정한다. `_deal_next_street()`는 플랍·턴·리버마다 한 장을 burn한다. `_settle()`은 `build_side_pots → rank_showdown → settle_pots` 순서로 배당·최종 스택을 생성하고 보존 조건을 검사한다. Blinds만으로 이미 모두 올인한 경우 `reset()`이 `(None,-1)`을 반환할 수 있고 `play_hand()`는 이 무결정 종료를 처리한다.

`observe(player_id)`는 현재 행동자에게만 허용되고 해당 좌석의 두 카드만 `raw_obs.hand`로 반환한다. 내부 `hole_cards`·덱 객체 자체는 정책 입력으로 전달하지 않는다. Native `legal_actions`는 full/short raise·재오픈 조건을 엔진이 판정한 결과다. 임의 불법 raise target을 조용히 보정하는 동작은 없다.

역사적 기록은 **74 passed / 1 xfailed**와 다인 감사 실행을 남겼다. 감사의 `len(board)==5` 집계는 완성된 보드 수이므로 엄밀히는 실제 contested showdown 수와 항상 같지 않다. 핵심 한계는 moving button 사용, 정식 dead-button 정책 부재, 직접적인 `step_back()` 부재, 외부 규칙 엔진 차등 검증 미완료다. 이 엔진의 `reset()`은 persistent stack 승계가 아니라 새 핸드 초기화다.

### Base Phase 4 — 핸드 정산을 이어 붙이는 TournamentEnv

```math
S_{i,h+1}^{start}=S_{i,h}^{final},\qquad
\mathcal A_h=\{i:S_{i,h}^{settled}>0\},\qquad
\mathrm{done}\iff |\mathcal A_h|=1.
```

```math
\ell(h)=\min\left(\left\lfloor\frac{h-1}{L}\right\rfloor,M-1\right),\qquad
R_i^{placement}=\frac1{|G_r|}\sum_{k=r}^{r+|G_r|-1}p_k\quad(i\in G_r).
```

$`h\ge1`$은 핸드 번호, $`L`$은 레벨별 핸드 수, $`M`$은 설정된 블라인드 레벨 수다. $`p_k`$는 순위 $`k`$의 지급값이며 $`G_r`$은 공동 공식 순위 $`r`$의 참가자 집합이다. 동시 탈락은 해당 핸드 시작 스택으로 정렬하고, 같은 시작 스택이면 공동 순위를 부여해 점유한 상금 순위 구간을 평균한다. 기본 지급은 우승자 1, 나머지 0이다. `placement_rewards()`의 기본 지급은 이후 Agent reward 모듈의 centered zero-sum winner 보상과 별개의 함수다.

Phase 4의 목적은 Base Phase 3의 정확한 한 핸드 결과를 장기 게임 상태로 승계하는 것이다. 매 핸드 스택을 리셋하는 cash-style 실행에서 탈락·블라인드 상승·최종 순위가 있는 단일 테이블 토너먼트로 실행 단위를 확장한다.

| 실제 Python 파일 | 담당 구현 |
|---|---|
| `environment/tournament_env.py` | `TournamentEnv`, `TournamentHandRecord`, `TournamentResult`; 핸드 승계·자동 다음 핸드·전체 종료 |
| `environment/blind_manager.py` | `BlindLevel`, `BlindManager.at_hand`; 양의 정수 SB/BB, 비감소 일정·마지막 레벨 반복 |
| `environment/player_manager.py` | `PlayerManager.update`, `alive`, `winner`; persistent stacks·총량·탈락·공동 순위 |
| `environment/rewards.py` | `chip_rewards`, `placement_rewards`; BB 변환과 최종 공동 순위 지급 |
| [scripts/run_tournament.py](technical_sources/PYTHON_SOURCE_05.md#py-054), [scripts/audit_tournament.py](technical_sources/PYTHON_SOURCE_01.md#py-011) | 토너먼트 실행 및 종료·승계·탈락 감사 |
| [tests/environment/test_elimination.py](technical_sources/PYTHON_SOURCE_18.md#py-208), [tests/test_tournament_env.py](technical_sources/PYTHON_SOURCE_27.md#py-318), [tests/test_player_manager.py](technical_sources/PYTHON_SOURCE_27.md#py-313), [tests/test_blind_manager.py](technical_sources/PYTHON_SOURCE_22.md#py-254), [tests/test_rewards_tournament.py](technical_sources/PYTHON_SOURCE_27.md#py-314) | 탈락·순위·블라인드·전체 토너먼트·순위 지급 회귀 |

실제 새 핸드 생성 부분은 `TournamentEnv._next_hand()`에 있다.

```python
            level_index, level = self.blinds.at_hand(hand_number)
            before = self.players.stacks
            hand = MultiPlayerHandEngine(
                stacks=before, button=self.button,
                small_blind=level.small_blind, big_blind=level.big_blind,
                seed=self._rng.getrandbits(64),
            )
```

`reset()`은 `PlayerManager`와 토너먼트 RNG를 초기화한다. `_next_hand()`는 직전 정산 스택을 새 엔진에 넘겨 실행한다. `_finish_hand()`는 `hand.get_result()`를 받아 `PlayerManager.update()`로 칩 보존·죽은 좌석의 부활 금지·탈락 순위를 반영하고 핸드 기록을 추가한다. 생존자가 하나면 종료하고 그렇지 않으면 `rotate_button()`으로 다음 생존 좌석에 버튼을 옮긴다. 다음 핸드가 블라인드만으로 즉시 끝나는 경우에도 `_next_hand()`의 루프가 기록과 정산을 진행한다.

완료 핸드 후 `step()`은 다음 핸드를 자동 시작한다. 행동은 항상 현재 좌석에 적용되며 탈락한 물리 좌석은 배열에서 삭제하지 않고 스택 0으로 남는다. `max_hands`·`max_actions`에 걸리면 오류를 내며 임의 우승자를 만들어 반환하지 않는다.

역사적 기록은 **113 passed / 1 xfailed**, 100토너먼트 중 100개 종료, 총 **1,086핸드·500탈락**, 평균 **10.86핸드**, 한 예시의 15핸드 종료와 우승자 6,000칩을 남겼다. 이는 Random Agent·당시 설정의 환경 감사 결과다. 범위는 2~6개 물리 좌석의 single-table knockout이며 ante·bounty·rebuy·add-on·시간 기반 레벨·다중 테이블·ICM 전략은 구현하지 않았다. 현 `tournament_env.py`의 Showdown 공개 카드 캡처는 이후 Tournament ENV 1.6.1 변경이므로 별도 단계에서 설명한다.

<a name="section-28f568c64e55"></a>

## Tournament ENV — 엔진을 사람이 사용하는 6인 대국 서비스로 연결

```math
\mathrm{NativeAction}\in\{0:F,1:C,2:\tfrac12P,3:P,4:A\},\qquad
\mathrm{HumanRaiseTo}\in\mathbb Z_{>0}\text{ subject to engine legality}.
```

Tournament ENV 1.1~1.7은 Base Phase 3·4의 규칙 엔진, 기존 HUNL GUI 설계와 이미 학습한 Expand PPO Actor를 세션·HTTP·브라우저·영구 기록에 연결한다. 이 트랙의 목적은 **실행 환경의 정확성과 사용성**이며, PPO 손실이나 가중치를 개선하는 Agent 학습 트랙과 구분한다. 학습 모드를 선택하면 좌석 0은 Human, 좌석 1은 Expand Phase 3 PPO, 좌석 2~5는 네 가지 휴리스틱이다. 기본 `baseline_5`와 명시적인 미학습 `improved_4` 라인업도 유지한다.

| 라인업 | 내부 좌석 1 | 내부 좌석 2~5 | 용도 |
|---|---|---|---|
| `baseline_5` | Calling Station | Calling Station 네 명 | 최초 API 호환·단순 기준선 |
| `improved_4` | `controlled_pressure` 미학습 placeholder | 네 가지 heuristic | 다양한 상대의 플레이 환경 |
| `trained_1_improved_4` | Frozen Expand Phase 3 PPO | 네 가지 heuristic | cash-game Actor의 tournament 전이 추론 |

### Tournament ENV 1.1 — 세션·FastAPI·자동 Bot 실행

```math
\mathrm{Accept}(a)=\big[r_{client}=r_{server}\big]\land\big[\mathrm{actor}=0\big]\land\big[a\in\mathcal A_t^{legal}\big]\land\neg\mathrm{done},\qquad
r_{server}\leftarrow r_{server}+1\ \text{per accepted engine action}.
```

$`r`$은 상태 revision이다. Human 행동뿐 아니라 자동 Bot 행동도 revision을 증가시킨다. 목적은 브라우저 중복 클릭·지연 요청·순서가 바뀐 응답이 다른 턴에 행동을 적용하지 못하도록 하고, 다음 Human 턴까지 Bot 행동을 자동 처리하는 것이다.

| 실제 Python 파일 | 최초 제작·수정 및 함수 역할 |
|---|---|
| `tournament_service/__init__.py` | 신규 서비스 패키지 |
| `tournament_service/session.py` | 신규 `TournamentPolicy`, `TournamentGameSession`, `TournamentSessionRegistry`; 이후 단계가 같은 파일을 확장 |
| `tournament_service/baseline.py` | 신규 `TournamentBaseline`; `random`, `calling_station`, `tight`, `pot_pressure` 기준 정책 |
| [backend/tournament_router.py](technical_sources/PYTHON_SOURCE_01.md#py-005) | 신규 `build_tournament_router`; 생성·조회·행동·raise-to·history·삭제 |
| [backend/tournament_schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-006) | 신규 엄격한 요청·응답; `StrictInt` 및 `extra='forbid'` |
| [backend/main.py](technical_sources/PYTHON_SOURCE_01.md#py-003) | 기존 HUNL router에 Tournament router 추가 |
| [scripts/tournament_env1_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-059), [tests/test_tournament_env1_api.py](technical_sources/PYTHON_SOURCE_28.md#py-327) | API 전체 대국 smoke 및 요청·상태·합법성 회귀 |

표의 `tournament_service/`는 `src/holdem_ai/tournament_service/`다. 실제 세션 행동 함수는 잠금 안에서 revision과 사람 차례를 먼저 검사한다.

```python
        with self._lock:
            self._check_revision(expected_revision)
            if self.env.is_over() or self.env.current_player != 0:
                raise RuntimeError("not the human's turn")
            if type(action_index) is not int or action_index not in self.env._hand.legal_actions:
                raise ValueError("illegal human action")
            self._step(action_index=action_index)
            self._run_bots()
            return self._view_unlocked()
```

호출 흐름은 `POST /sessions → TournamentSessionRegistry.create → TournamentGameSession → TournamentEnv.reset → _run_bots → Human-safe view`다. 그 후 Human 요청은 `play_action()` 또는 `play_raise_to()`로 들어가고, `_run_bots()`는 다시 Human 턴이 되거나 토너먼트가 종료될 때까지 정책을 호출한다. 각 정책 반환값은 관측의 합법 native ID에 있는지 재검사한다. `RLock`은 같은 세션의 동시 변경을 직렬화하고 Registry는 최대 128세션 기본 수용량을 관리한다. 토너먼트에 수동 Next Hand 버튼은 필요하지 않다.

`_view_unlocked()`는 보드·Human 카드·각 좌석 칩·역할·현재 차례·합법 액션을 구성하며 엔진 덱·상대 홀 카드·Critic 상태를 반환하지 않는다. 현재 진행 중이면 `sum(stacks)+pot`, 완료면 `sum(stacks)`를 토너먼트 총 칩과 비교한다. stale revision과 턴 충돌은 409, 불법 입력은 422, 없는 session은 404로 번역된다. 생성 시 자원/실행 오류는 503으로 처리된다. `Cache-Control: no-store`는 응답 캐시로 옛 revision이 사용되는 것을 줄인다.

역사적 문서 중 최초 부분 호환 harness는 **11 tests passed**, Human 25행동·10핸드 종료를 기록했다. 후속 전체 Tournament 보고서는 사용자 Windows의 **신규 11 passed**, 전체 **713 passed / 1 xfailed**, Human 25행동·**13핸드·총 600칩** smoke를 기록한다. 실행 환경과 설정이 다른 기록을 하나의 run으로 합치지 않는다. 당시 이력은 메모리에 있었고 지속 저장은 1.6에서 추가됐다.

### Tournament ENV 1.2 — React 6인 테이블

```math
\mathrm{Enabled}(a)=\mathbf1[a\in\mathrm{state.actions}],\qquad
\sum_i\mathrm{state.seats}[i].stack+\mathrm{state.pot}=\mathrm{state.total\_chips}.
```

첫 식은 버튼 활성화 조건, 둘째는 브라우저가 보조 검사하는 칩 보존 조건이다. 목적은 API를 사람들이 직접 플레이할 수 있는 원탁·카드·베팅 패널로 바꾸는 것이다. 합법성과 정산의 최종 판정은 서버 엔진이 수행한다.

이 Phase의 **새 Python 파일은 없다**. 1.1의 `session.py`, `tournament_router.py`, `tournament_schemas.py`를 그대로 소비한다. 역사적 프론트엔드 파일은 `frontend/src/App.tsx`, `HUNLApp.tsx`, `tournament/TournamentApp.tsx`, `TournamentTable.tsx`, `TournamentBetting.tsx`, `api.ts`, `types.ts`, `rules.ts`, `rules.test.ts`, `styles.css`다. 기존 HUNL 화면을 별도 트리로 분리하고 두 탭의 트리를 유지해 현재 브라우저의 UI 상태를 보존했다.

동작은 세션 생성 응답을 `TournamentApp`이 받아 테이블과 입력 패널에 전달하고, 선택 행동과 `expected_revision`을 API에 보내 새 응답을 표시하는 방식이다. 상대 카드는 뒷면으로 처리하며 당시에 서버가 보내지 않은 Showdown 카드를 생성하지 않는다. 핸드 요약과 최근 25개 기록은 당시 live session history를 조회했다. 자유 raise-to는 서버의 가능 여부·최소·최대 범위를 표시하고 별도 endpoint를 사용한다.

역사적 기록은 TypeScript typecheck·Vite build 및 **Vitest 18/18** 통과를 보고했다. 이 Python 전체 소스 부록만으로 당시 프론트엔드 구현을 새로 실행 검증했다는 의미는 아니다. 새로고침·서버 재시작 후 live session 복원은 구현하지 않았다.

### Tournament ENV 1.2.1 — Human 카드 숫자와 위치 수정

```math
\mathrm{HumanCardVisible}=\mathrm{RankVisible}\land\mathrm{SuitVisible}\land\mathrm{WithinSeatViewport}.
```

이는 카드 렌더링의 요구 조건이며 게임 확률이나 학습 식이 아니다. Human의 문양은 보이지만 A/K/Q·숫자가 사라지고 카드가 아래에서 잘리던 화면을 고친 단계다. 과거 보고서가 지목한 원인은 `.t-seat-cards .card-corner { display: none; }`와 좌석 아래의 카드 배치였다.

수정은 `frontend/src/styles.css`의 Human 카드 숫자·문양 복원, 좌석 위 배치와 반응형 크기 조정이다. **이 단계에서 제작·수정한 `.py`는 없다.** 서버 `hero_cards`와 상대 비공개 계약도 유지한다. 사용자가 화면 문제 해결을 확인한 기록이 있으나, 별도 Python 성능·정책 결과는 해당하지 않는다.

### Tournament ENV 1.3 — 네 가지 heuristic 상대 연결

```math
q_i=\frac{\min(S_i,\max(0,d_i))}{\max(1,P+\min(S_i,\max(0,d_i)))},\qquad
v_i=\mathrm{score}(H_i,B)+\epsilon-\min(0.12,0.022\max(0,n_{rival}-1)).
```

$`q_i`$는 팟 오즈, $`v_i`$는 휴리스틱 행동 점수다. $`\epsilon=(U-0.5)\,0.035`$, $`U\sim U[0,1)`$는 seeded 작은 변동이며 $`n_{rival}`$은 Fold/Out이 아닌 상대 수다. 이 점수는 showdown 승률이나 멀티웨이 equity가 아니다.

프리플랍 점수의 실제 정의를 더 구체적으로 쓰면, 높은·낮은 rank를 $`x\ge y`$라 하고 $`x_n=(x-2)/12`$, $`y_n=(y-2)/12`$일 때

```math
s_{pre}=\begin{cases}
\min(0.99,0.54+0.43x_n),&x=y,\\
\mathrm{clip}_{[0,0.94]}\big(0.13+0.40x_n+0.24y_n+0.07\mathbf1_{suited}+g(x-y)+0.04\mathbf1_{x=14}\big),&x\ne y,
\end{cases}
```

여기서 $`g(1)=0.07`$, $`g(2)=0.035`$, 그 외 간격은 0이다. 플랍 이후에는 자기 카드와 보드의 최선 5장 족보를 구하고 족보별 기준 점수·pair/high-card rank·4장 동일 suit 드로우를 반영한다. River에서 보드만으로 같은 최선 족보를 만들면 점수를 최대 0.49로 낮춘다. 정확한 상대 range나 남은 카드 조합을 통한 equity 계산은 아니다.

목적은 다섯 Calling Station뿐인 대국을 서로 구분되는 상대 네 명이 있는 환경으로 바꾸는 것이다. `improved`는 개발 슬롯 이름이며 기존 Bot 대비 승률 향상의 측정 결과가 아니다.

| 실제 Python 파일 | 변화와 역할 |
|---|---|
| `tournament_service/improved_bot.py` | 신규 `PROFILES`, `_preflop_strength`, `_postflop_strength`, `ImprovedTournamentBot.select_action` |
| `tournament_service/session.py` | 라인업에 따른 좌석 정책 조립 |
| [backend/tournament_schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-006) | `lineup`, 좌석 role·policy name 확장 |
| [backend/tournament_router.py](technical_sources/PYTHON_SOURCE_01.md#py-005) | 기존 생성 경로에서 새 라인업을 전달 |
| [scripts/tournament_env13_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-056), [tests/test_tournament_env13_improved_bots.py](technical_sources/PYTHON_SOURCE_28.md#py-320) | heuristic 합법성·라인업·대국 종료 검사 |

실제 판단 입력은 다음 코드처럼 관측에서만 읽는다.

```python
        raw = observation["raw_obs"]
        hole = tuple(raw["hand"])
        board = tuple(raw["public_cards"])
        if len(hole) != 2 or len(board) not in (0, 3, 4, 5):
            raise ValueError("acting-player hand or board is malformed")
        strength = _preflop_strength(hole) if not board else _postflop_strength(hole, board)
        stack = int(raw["stacks"][raw["player_id"]])
        call = min(stack, max(0, int(raw["to_call"])))
        pot = max(0, int(raw["pot"]))
        pot_odds = call / max(1, pot + call)
```

`TournamentGameSession.__init__()`는 좌석 2~5에 순서대로 `tight_aggressive`, `balanced_value`, `pot_odds`, `controlled_pressure`를 만든다. 원래 1.3에서 좌석 1은 Calling Station placeholder였으며, 현 소스의 `improved_4`에서는 1.3.1 변경에 따라 `controlled_pressure` placeholder다. 기본 생성 요청은 계속 `baseline_5`다. Bot은 합법 native ID만 반환하며 자유 raise-to는 Human 경로다.

역사적 부분 호환 harness는 기존 API 11개와 신규 9개, 총 **20 passed**를 기록했다. 공식 React 빌드와 전체 Windows 검증을 대신한 결과가 아니다. 네 프로필의 실제 승률·Nash exploitability·ICM 최적성은 이 단계에서 검증하지 않았다.

### Tournament ENV 1.3.1 — Raise가 누락되던 분기 수정

```math
\mathrm{RaiseCandidate}=(v_i\ge\tau_i)\land[(2\in\mathcal A^{legal})\lor(3\in\mathcal A^{legal})].
```

$`\tau_i`$는 프로필별 raise 임계값이다. 이전 코드가 ID 2의 합법성만 확인해 Half-pot은 불법이고 Pot은 합법인 상황에서도 Raise 검토를 건너뛰었다. 수정은 두 액션의 합집합을 확인하는 것이다. 다인 패널티 아래에서 과도하게 높은 임계값과 좌석 1의 Calling Station 기본 행동도 함께 조정했다.

현 `improved_bot.py`의 파라미터는 다음과 같다.

| 프로필 | Raise 임계값 | Continue 임계값 | Aggression 값 |
|---|---:|---:|---:|
| tight_aggressive | 0.675 | 0.49 | 0.12 |
| balanced_value | 0.565 | 0.40 | 0.16 |
| pot_odds | 0.635 | 0.34 | 0.06 |
| controlled_pressure | 0.495 | 0.395 | 0.22 |

Aggression 값은 소스의 `controlled_pressure` pot 선택 분기에서 사용된다. 모든 프로필에서 동일한 확률로 무작위 Raise를 실행한다는 뜻은 아니다. 매우 높은 점수 $`v_i\ge0.87`$와 $`S_i\le\max(1,P)`$를 모두 만족할 때만 합법 all-in을 우선 선택한다.

실제 수정 후 분기는 다음과 같다.

```python
        if value >= raise_threshold and (2 in legal or 3 in legal):
            if 3 in legal and (
                2 not in legal or value >= raise_threshold + 0.14 or
                (self.profile == "controlled_pressure" and self._rng.random() < aggression)
            ):
                return 3
            if 2 in legal:
                return 2
```

수정 파일은 `tournament_service/improved_bot.py`, `tournament_service/session.py`, 신규 [scripts/tournament_env131_diagnose.py](technical_sources/PYTHON_SOURCE_05.md#py-055), [tests/test_tournament_env131_aggression.py](technical_sources/PYTHON_SOURCE_28.md#py-319)다. GUI placeholder 설명도 `TournamentApp.tsx`에서 수정했다. 진단 CLI는 $`\binom{52}{2}=1326`$개 시작 패를 고정 팟·콜 비용·스택·legal mask에서 열거한다. 출력 RaiseShare는 **그 조건에서의 정책 선택 비율**이며 실전 빈도나 승률로 읽으면 안 된다.

역사적 부분 harness는 **30 tests passed**, 30시드 **388핸드**의 종료·칩 보존과 좌석 1~5의 half-pot/pot raise 비율 **29.1%, 10.4%, 19.6%, 11.2%, 28.5%**를 기록했다. All-in은 이 Raise 비율에서 제외했다. 이후 사용자의 GUI 적용 확인도 기록돼 있다. 이는 seed-matched 승률 대조 실험이 아니다. 기존 live session은 이미 생성된 정책 객체를 유지하므로 새 Backend와 새 토너먼트에서 수정 정책을 사용해야 한다.

### Tournament ENV 1.4 — Frozen Expand PPO Actor 추론

```math
g(a_E)=\begin{cases}
0,&a_E=0,\\1,&a_E\in\{1,2\},\\2,&a_E=3,\\3,&a_E=4,\\4,&a_E=6,\\\text{undefined},&a_E=5,
\end{cases}
\qquad
\pi_T(a_E\mid I)=\frac{m_E(a_E)\pi_E(a_E\mid I)}{\sum_bm_E(b)\pi_E(b\mid I)}.
```

$`a_E`$는 Expand의 일곱 행동 `(fold,check,call,half-pot,pot,double-pot,all-in)`, $`g`$는 Native 다섯 행동으로의 변환이다. $`m_E`$는 Native 엔진 합법성을 Expand 출력 공간으로 옮긴 마스크다. `to_call==0`이면 Check만, 양수면 Call만 허용한다. Native에 없는 double-pot은 항상 마스크하고 임의 다른 행동으로 바꾸지 않는다.

목적은 두 사람 HUNL 가중치를 여섯 사람 모델로 잘못 사용하는 일을 방지하면서, **Expand Phase 3 cash-game PPO를 읽기 전용으로** 좌석 1에 연결하는 것이다. knockout 전용 학습이나 tournament reward 재최적화는 하지 않는다.

| 실제 Python 파일 | 변화와 역할 |
|---|---|
| `tournament_service/trained_actor.py` | 신규 `status`, `_sha256`, `_prepare_public_state`, `TrainedTournamentActor` |
| `tournament_service/session.py` | `trained_1_improved_4` 조립, 행동 직전 public history 수집·Actor 전달 |
| [backend/tournament_router.py](technical_sources/PYTHON_SOURCE_01.md#py-005), [backend/tournament_schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-006) | `/agent-status`, trained 라인업·status 계약 |
| [scripts/tournament_env14_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-057), [tests/test_tournament_env14_trained_actor.py](technical_sources/PYTHON_SOURCE_28.md#py-321) | 실제 checkpoint smoke·관측·mask·형식 거부 |
| `expand/information/schema.py`, `encoder.py`, `expand/rl/trainer.py` | 기존 Expand InformationState·encoder·`load_trained_actor` 재사용 |

실제 변환 상수와 출력 마스크 코드 일부다.

```python
EXPAND_TO_NATIVE: tuple[int | None, ...] = (0, 1, 1, 2, 3, None, 4)
```

```python
    mask = (
        0 in legal_native,
        1 in legal_native and raw_call == 0,
        1 in legal_native and raw_call > 0,
        2 in legal_native,
        3 in legal_native,
        False,  # double-pot is NOT a native action: do not silently remap
        4 in legal_native,
    )
```

`from_environment()`는 `HOLDEM_AI_TOURNAMENT_CHECKPOINT`, `HOLDEM_AI_TOURNAMENT_DEVICE`, `HOLDEM_AI_TOURNAMENT_POLICY_MODE`를 읽는다. device는 CPU/CUDA, mode는 sampled/greedy만 허용한다. `load_trained_actor()`가 `expand-e03-v1` 호환성을 검증하고 파일 SHA-256의 앞 12글자가 policy name에 들어간다. `status()`는 경로 설정과 파일 존재만 보고하며 `validated=False`는 아직 모델 전체를 로드하지 않았다는 뜻이다. 모델 누락·형식 오류 때 trained 요청은 실패하며 Random Agent로 조용히 대체하지 않는다.

`_prepare_public_state()`는 자신의 카드·보드·여섯 물리 좌석 상태·BB로 표준화한 칩·위치·legal spec·이전 공개 행동을 Expand `InformationState`로 바꾼다. `select_action()`은 `encode_observation → model.actor.action_probabilities`를 호출하고 불법 확률을 제거한다. sampled는 남은 확률에 비례해 추출하고 greedy는 합법 최댓값을 선택하며, 마지막에 $`g`$를 적용해 Native 합법성을 확인한다. 이 경로는 모델 Critic을 호출하지 않는다. `_policy_history`는 각 행동의 **실행 직전** street·pot·call·raise target을 저장하며 새 핸드에서는 비운다. 미래 덱·다른 사람 패·training state를 이 관측에 섞지 않는다.

역사적 최종 Windows smoke는 `runs/expand_phase03_pilot/best.pt`를 로드해 **13핸드·Human 2액션·칩 보존 true**를 기록했다. 현 작성 환경에 해당 사용자 가중치가 있다는 뜻은 아니다. 주요 한계는 cash-game에서 knockout으로의 분포 이동, double-pot 제거에 따른 action-space 변화, 아직 입증하지 않은 토너먼트 성능이다.

### Tournament ENV 1.5 — 정산 스택·공식 순위·블라인드 진행 표시

```math
\mathcal A_h=\{i:S_{i,h}^{settled}>0\},\qquad
\widetilde r_i=1+\big|\{j\in\mathcal A_h:S_{j,h}^{settled}>S_{i,h}^{settled}\}\big|,
```

```math
H_{to\ next}=L-((h-1)\bmod L).
```

$`\widetilde r_i`$는 생존자의 임시 칩 순위이고 엔진의 확정 탈락 순위 $`r_i`$와 다르다. $`H_{to\ next}`$는 현재 핸드를 포함해 다음 블라인드 레벨까지 남은 핸드 수다. 마지막 레벨·완료 상태에서는 다음 레벨이 없으므로 관련 필드를 `null`로 처리한다.

목적은 `seats[i].stack==0`인 올인을 파산으로 잘못 표시하지 않고, 생존자·탈락자·chip leader·champion을 정확히 구분하는 것이다. 베팅 중 `stack`은 behind chips이고 `tournament_stacks`는 직전 핸드 정산 잔액이다. 살아 있는 All-in 플레이어는 현재 behind가 0이어도 `alive_seats`에 남는다.

수정 `.py`는 `tournament_service/session.py`, [backend/tournament_schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-006)이고 신규 회귀는 [tests/test_tournament_env15_progress.py](technical_sources/PYTHON_SOURCE_28.md#py-322)다. 프론트엔드는 `tournament/standings.ts`, `standings.test.ts`, `TournamentProgressPanel.tsx`, `TournamentApp.tsx`, `TournamentTable.tsx`, `rules.ts`, `rules.test.ts`, `types.ts`, `styles.css`를 확장했다.

실제 서비스 상태 생성 일부는 다음과 같다.

```python
            "alive_seats": list(env.players.alive),
            "tournament_stacks": list(env.players.stacks),
            "blind_progress": {
                "hands_per_level": env.blinds.hands_per_level,
                "hand_in_level": level_hand,
                "hands_to_next": (env.blinds.hands_per_level - level_hand + 1) if next_level is not None and level_hand is not None else None,
                "next_small_blind": next_level.small_blind if next_level is not None else None,
                "next_big_blind": next_level.big_blind if next_level is not None else None,
                "final_level": level_index + 1 >= level_count,
            },
```

`_view_unlocked()`가 `PlayerManager`의 공식 정보를 투영하고 브라우저 `standings`가 생존자 임시 순위를 계산한다. UI는 임시 `~N`과 확정 `#N`을 구분한다. Blinds·정산·PPO 파라미터는 여기서 바꾸지 않는다. 해당 단계의 개별 acceptance 문서는 재실행 체크리스트였고, 후속 1.7 보고서가 실제 통합 검증 완료를 기록한다. 현재 handbook에서 개별 단계의 독립 실행 시간이나 개수를 새로 만들어 주장하지 않는다.

### Tournament ENV 1.6 — SQLite 영구 이력·기록 Replay·JSONL Export

```math
\mathrm{Replay}_h=\big[(I_{h,1}^{recorded},a_{h,1}),\ldots,(I_{h,T}^{recorded},a_{h,T}),\mathrm{Settlement}_h\big],
```

```math
D_{h,t}=\big(I_{h,t}^{actor},a_{h,t},\Delta S_{actor,h},\mathrm{session\_id},h,t\big).
```

$`I^{recorded}`$는 저장된 행동 직전 프레임이며, $`I^{actor}`$는 그 행동자 본인 카드와 공개 상태다. 완료되지 않은 핸드의 $`\Delta S`$는 `null`이다. 목적은 서버가 재시작되거나 live session을 종료한 뒤에도 **이미 커밋한 행동·정산 기록**을 조회하고 분석 가능한 데이터로 내보내는 것이다.

| 실제 Python 파일 | 변화와 역할 |
|---|---|
| `tournament_service/journal.py` | 신규 `TournamentJournal`; SQLite 초기화·`begin`·`record_step`·`record_initial_hands`·`close`·`hands`·`replay`·`actor_export_rows` |
| `tournament_service/session.py` | `_step`의 행동 직전 기록·성공 후 Journal 호출·생성 실패 정리 |
| [backend/tournament_router.py](technical_sources/PYTHON_SOURCE_01.md#py-005) | archive·완료 핸드·recorded replay·NDJSON 다운로드 |
| [backend/main.py](technical_sources/PYTHON_SOURCE_01.md#py-003) | `TournamentJournal`을 Registry에 주입; 테스트용 경로 설정 |
| [tests/test_tournament_env16_persistence.py](technical_sources/PYTHON_SOURCE_28.md#py-324) | DB 재열기·세션 종료 후 보존·Replay·Export 계약 |

DB 기본 경로는 `.local/tournament_history.sqlite3`이고 `HOLDEM_AI_TOURNAMENT_JOURNAL`로 변경할 수 있다. `_connect()`는 `foreign_keys=ON`, busy timeout 15초, WAL, `synchronous=FULL`을 사용한다. `tournament_sessions`에는 설정·상태·revision·우승자, `tournament_actions`에는 `(session_id,hand_number,action_no)`와 행동 직전 데이터, `tournament_hands`에는 정산 결과를 저장한다.

다음은 실제 공개 Replay 투영이다.

```python
    @staticmethod
    def _public_replay(action: Mapping[str, Any]) -> dict[str, Any]:
        # Return hero hole only for the human's own turn. Never expose AI holes.
        result = {k: v for k, v in action.items() if k != "actor_hole"}
        result["hero_hole"] = action.get("actor_hole", []) if action["seat"] == 0 else []
        return result
```

`TournamentGameSession._step()`은 먼저 행동 직전 `board`, `stacks`, `street_bets`, `contributions`, `status`, `pot`, `to_call`, native legal actions와 행동자 자신의 카드를 캡처한다. 실제 raise target도 해석해 기록한다. 엔진이 행동을 거부하면 Journal 호출에 도달하지 않는다. 성공하면 revision을 증가시키고 `record_step()`에 행동과 해당 step에서 새로 완료된 **모든** 핸드를 전달한다. 이는 다음 핸드가 블라인드만으로 즉시 종료될 수 있기 때문이다. Journal은 액션 INSERT·정산 INSERT·세션 UPDATE를 **하나의 DB 트랜잭션**으로 커밋한다.

Public Replay는 AI 행동자의 `actor_hole`을 숨기고 Human 행동 프레임에만 `hero_hole`을 남긴다. `actor_export_rows()`는 각 행의 관측에 해당 행동자의 `own_hole_cards`를 넣고 완료 핸드와 LEFT JOIN해 그 행동자의 칩 수익을 레이블로 반환한다. export의 schema는 `tournament.actor_transition.v1`이고 Native ID와 실제 `raise_to`를 저장한다. 이 데이터는 행동 당시 자신의 정보로 구성되지만 **여러 행동자의 개인 카드가 담긴 로컬 분석 파일**이다. 즉시 PPO에서 사용하는 on-policy log probability·GAE·advantage가 완성된 rollout은 아니다.

`DELETE /sessions/{id}`는 메모리 Registry에서 live session을 제거하고 DB status를 closed/complete로 갱신하며 기록을 삭제하지 않는다. `record_initial_hands()`는 reset만으로 끝난 무결정 핸드도 저장한다. 서버 재시작 후 Archive는 DB로 다시 조회하지만 라이브 엔진을 복구하지 않는다. Recorded Replay, seeded engine re-simulation, live Resume는 서로 별개의 기능이며 뒤의 두 기능은 구현하지 않았다.

역사적 기록은 사용자 Windows의 persistence 테스트와 실제 Replay 화면 동작을 확인했다고 설명하고, 전체 결과는 1.7에 수록돼 있다. 소스상 DB transaction 이전에 메모리 엔진과 revision이 이미 변한다. 따라서 DB 쓰기 실패 때 엔진 상태까지 되돌리는 전역 atomicity나 실패 후 복구를 보장하지 않는다. 프로세스 종료 직전 커밋되지 않은 행동·디스크 장애도 WAL/FULL만으로 무손실 보장할 수 없다. 과거 메모리 이력을 소급 DB로 이전하는 기능은 없다.

### Tournament ENV 1.6.1 — 정산 후 contested Showdown 카드 공개

```math
\mathcal C_h=\{i:\neg\mathrm{folded}_{i,h}\},\qquad
\mathrm{Reveal}(i,h)=\begin{cases}
H_{i,h},&\mathrm{settled}_h\land|\mathcal C_h|\ge2\land|B_h|=5\land i\in\mathcal C_h,\\
\varnothing,&\text{otherwise}.
\end{cases}
```

$`\mathcal C_h`$는 결과에서 Fold/Out이 아닌 참가자다. 목적은 승부 후 상대 패를 볼 수 있게 하면서, 행동 중 상대 카드나 Fold한 카드가 새나가지 않도록 공개 시점을 정산 계층에 두는 것이다.

| 실제 Python 파일 | 변화와 역할 |
|---|---|
| `environment/tournament_env.py` | `TournamentHandRecord.showdown_hole_cards`; `_finish_hand()`에서 다음 핸드 전 카드 캡처 |
| `tournament_service/session.py` | `_hand_summary()`에서 `showdown_cards=[{seat,cards}]` 공개 결과 생성 |
| [backend/tournament_schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-006) | `TournamentShowdownSeat`, `TournamentHandView.showdown_cards` 엄격한 응답 계약 |
| [tests/test_tournament_env161_showdown.py](technical_sources/PYTHON_SOURCE_28.md#py-323) | 정산 전·Fold 종료 비공개, contested 공개, DB·Archive 호환 |

실제 캡처는 완료된 엔진을 읽는 `_finish_hand()` 안에서 수행한다.

```python
        contenders = [i for i, folded in enumerate(result.folded) if not folded]
        showdown_hole_cards = (
            tuple(tuple(self._hand.hole_cards[i]) if i in contenders else ()
                  for i in range(self.num_players))
            if len(contenders) >= 2 and len(result.board) == 5 else ()
        )
```

다음 핸드가 `self._hand`를 교체하기 전에 카드 tuple을 결과 기록에 저장한다. `last_hand`, live history, Archive hands와 Replay의 `settlement`에서만 공개 결과를 표시한다. Replay `actions`와 actor export 관측에 미래 정산 카드가 소급 삽입되지 않는다. 상대 전원 Fold로 끝난 승자의 카드도 공개하지 않는다. 프론트엔드는 `TournamentShowdown.tsx`, `TournamentApp.tsx`, `TournamentArchive.tsx`, `types.ts`, `styles.css`를 확장했다.

과거 DB 행에 필드가 없으면 빈 목록으로 처리하며 보관하지 않은 카드의 소급 복원은 하지 않는다. 역사적 기록은 신규·관련 회귀 통과와 사용자 실제 화면 확인을 남겼으며 최종 전체 테스트는 1.7 결과에 포함된다. 현 초기화 경로에서는 reset으로 완료된 핸드를 DB에 저장하지만 `_last_finished` 갱신 경로와 일반 step 이후 갱신 경로가 다르므로, 초기 무결정 종료의 live `last_hand` 표시도 회귀 대상으로 확인할 가치가 있다.

### Tournament ENV 1.7 — 통합 검증과 독립 Environment 릴리스

```math
\mathrm{ReleaseReady}=\mathrm{PythonRegression}\land\mathrm{TrainedActorSmoke}\land\mathrm{FrontendChecks}\land\mathrm{GitPathAudit}\land\mathrm{DiffReview}.
```

각 항목은 게임·추론·화면·배포 산출물의 서로 다른 실패를 탐지한다. Phase 1.7의 목적은 여러 단계의 패치가 HUNL과 공존하며 동작하는지 확인하고, Agent 실험 브랜치와 Environment 릴리스 변경을 구분해 GitHub에 반영하는 것이다.

| 실제 파일 | 제작 역할 |
|---|---|
| [scripts/tournament_env17_release_audit.py](technical_sources/PYTHON_SOURCE_05.md#py-058) | 신규 `unsafe_reason`, `check_release`; staged/tracked 경로·ignore·whitespace 감사 |
| [tests/test_tournament_env17_release_audit.py](technical_sources/PYTHON_SOURCE_28.md#py-326) | 위험 경로와 검사기 회귀 |
| [tests/test_tournament_env17_integration.py](technical_sources/PYTHON_SOURCE_28.md#py-325) | HUNL·Tournament 공존·Showdown·Replay 전체 계약 |
| `scripts/tournament_env17_verify.ps1` | pytest·학습 Actor smoke·TypeScript·Vite·Vitest·Git audit 통합 실행 |
| `scripts/tournament_env17_stage.ps1`, `tournament_env17_manifest.txt` | 허용 파일 목록을 사용한 선택적 Stage |
| `.gitignore` | 모델·DB·개인 기록·가상환경·빌드 출력 경로 보호 |

실제 경로 판정 부분이다.

```python
    if clean.startswith(DANGEROUS_DIRS) or any(p in ('__pycache__', '.pytest_cache') for p in parts):
        return 'generated/private directory'
    if name == '.env' or name.startswith('.env.') and name != '.env.example':
        return 'environment secrets'
    if name.endswith(DANGEROUS_EXT):
        return 'model/data/binary artifact'
```

`check_release()`는 Git root와 프로젝트 파일을 확인하고 `git diff --cached --name-only -z`·`git ls-files -z`로 staged/tracked 목록을 얻는다. 위험 경로가 추적되면 실패, `--strict`에서는 ignore 누락이나 staged 파일 없음도 실패한다. staged 변경에 `git diff --cached --check`를 적용한다. 소스 텍스트에 직접 넣은 모든 비밀을 검출하는 기능은 아니므로 path audit와 Diff review의 범위가 다르다.

역사적 릴리스 기록에는 63개 허용 파일 Stage 뒤 세 파일의 trailing whitespace를 제거한 과정, 당시 `agent/phase15-collapse-diagnostics`에서 Environment 전용 브랜치를 만들어 커밋한 과정이 포함된다. 원래 Environment 커밋 `825a507`을 최신 main에 cherry-pick한 결과가 `b2313fb`다. 같은 변경 집합도 부모 커밋이 다르면 Git commit SHA는 달라진다. 이 이력은 당시 반영 기록이며 현재 브랜치 상태를 대신 설명하지 않는다.

**2026-10-13 사용자 Windows 실행의 역사적 결과**는 다음과 같다. 이번 통합 문서를 작성하면서 새로 실행한 검증과 구분한다.

| 검증 | 기록된 결과 | 판정 범위 |
|---|---|---|
| 전체 Python | **757 passed / 1 xfailed / 1 warning**, 38.06초 | 당시 전체 회귀 |
| 실제 Expand Actor smoke | Passed, 13핸드·Human 2행동·칩 보존 true | 사용자 로컬 `best.pt` 실제 로드·추론 |
| TypeScript | `tsc -b --pretty false` 통과 | 타입 검사 |
| Vite | production build 통과, 48 modules | 프론트엔드 빌드 |
| Vitest | **25 passed / 5 files** | HUNL·Tournament 프론트엔드 회귀 |
| Git Privacy Audit | Passed | stage/tracked 경로 감사 |
| GitHub Push | `d818f44..b2313fb`, main 동기화 기록 | 당시 릴리스 반영 |

이 기록이 입증하는 것은 Human·Frozen multi-seat PPO·heuristic 네 명의 대국 실행과 기록·조회·표시 계약이다. cash-game PPO의 tournament 우월성, 실전 운영용 규칙 인증, 다중 사용자 공개 서비스 보안은 별도 검증 주제다. 기본 Backend는 `127.0.0.1:8000`에 바인딩하는 인증 없는 로컬 연구용 서버이며, 프론트엔드 기본 주소는 `127.0.0.1:5173`이다.

### Tournament ENV의 최종 HTTP 계약과 이어받기 경로

| Method | 경로 | 호출 결과 |
|---|---|---|
| GET | `/api/v1/tournament/health` | 로컬 서비스 health |
| GET | `/api/v1/tournament/agent-status` | 체크포인트 구성·존재·모드; 완전 로드 검증 아님 |
| POST | `/api/v1/tournament/sessions` | 라인업·스택·레벨·시드로 새 토너먼트 생성 |
| GET | `/api/v1/tournament/sessions/{id}` | Human-safe 현재 상태 |
| POST | `/api/v1/tournament/sessions/{id}/actions` | Native ID와 expected revision |
| POST | `/api/v1/tournament/sessions/{id}/raise-to` | 정수 street-total target과 expected revision |
| GET | `/api/v1/tournament/sessions/{id}/history` | live session의 완료 핸드 |
| DELETE | `/api/v1/tournament/sessions/{id}` | live session 종료·Journal 유지 |
| GET | `/api/v1/tournament/archive` | SQLite 세션 목록 |
| GET | `/api/v1/tournament/archive/{id}/hands` | 영구 완료 핸드 목록 |
| GET | `/api/v1/tournament/archive/{id}/hands/{h}/replay` | 기록된 행동 직전 프레임·정산 |
| GET | `/api/v1/tournament/archive/{id}/export` | actor-visible `application/x-ndjson` |

새 개발자는 게임 규칙 변경을 `multi_player_engine.py`·`pot_manager.py`·`player_manager.py`에서, 세션 행동과 프레임 캡처 변경을 `tournament_service/session.py`에서, 정책 호환 변경을 `trained_actor.py`에서, 저장 계약 변경을 `journal.py`에서 시작해야 한다. 스키마를 추가하면 Python Pydantic 응답·TypeScript 타입·live view·Archive 구형 행을 함께 점검해야 한다. native 5-action과 Expand 7-action, 현재 behind와 정산 stack, 행동 당시 관측과 종료 후 공개 결과를 각각 유지하는 것이 이 환경을 이어받는 핵심 조건이다.

<a name="section-842d126f6ae5"></a>

## HUNL Agent — Base P5–P17의 수학, 구현과 검증

이 트랙은 규칙 엔진 위에 학습 가능한 정책을 연결하고, 그 정책을 재현 가능한 조건에서 평가하는 작업이다. **Base P5–P17은 2인 HUNL 연구이며 Expand E1–E7의 2–6인 학습, Tournament ENV 1.x의 운영 환경과 구별한다.** 아래 코드는 현재 `src/holdem_ai/`의 실제 구현에서 발췌했다. 본문의 짧은 코드 블록은 읽기 편하도록 바깥 함수·클래스의 들여쓰기만 줄였으며, 해당 블록만으로 독립 실행되는 예제가 아니다. 원래 문맥과 들여쓰기를 포함한 전체 코드는 소스 부록을 참조한다. 기존 문서의 `holdem_agent_phaseN/` 경로는 초기 추가형 패키지 배포 당시의 경로이며 현재 코드의 위치로 읽으면 안 된다. 테스트 개수·처리량·GPU 학습 수치는 과거 문서와 사용자 실행 기록을 재정리한 것으로, 이번 문서 편집 중 재실행한 결과가 아니다.

### P5 — Agent 인터페이스: 엔진과 정책의 연결 계약

```math
I_t^i=O_i(s_t,h_t^{\mathrm{public}}),\quad
a_t\sim\pi(\cdot\mid I_t^i),\quad
m_a(I)=\mathbf1[a\in\mathcal A_{\mathrm{legal}}(I)].
```

```math
\pi^m(a\mid I)=
\begin{cases}
\displaystyle\frac{m_aw_a}{\sum_bm_bw_b},&\sum_bm_bw_b>0,\\
\displaystyle\frac{m_a}{\sum_bm_b},&\text{합법 행동 가중치가 모두 0}.
\end{cases}
```

`s`는 전체 엔진 상태, `I`는 행동 플레이어가 볼 수 있는 정보, `w`는 정책이 제안한 음이 아닌 가중치다. 비정상 가중치(NaN, 무한대, 음수)는 오류로 처리한다. 이 Phase의 목적은 Random, 저장된 CFR 전략, 신경망 등 여러 정책이 동일한 `act(observation, rng) -> action_index` 계약을 사용하게 만드는 것이다. `CFRTablePolicy`는 기존 전략을 읽는 샘플러이며 regret를 갱신하는 CFR solver는 아니다.

| 현재 Python 경로 | 입력 → 출력 / 역할 |
|---|---|
| [src/holdem_ai/core/schema.py](technical_sources/PYTHON_SOURCE_06.md#py-070) | `InformationState`, `Turn`, `Terminal`, `TurnBasedEnvironment`; 관측·합법성·종료 보상 계약 |
| [src/holdem_ai/agents/policies.py](technical_sources/PYTHON_SOURCE_06.md#py-066) | 관측+RNG → 합법 action; 마스킹, Random/CFR-table/ε-greedy |
| [src/holdem_ai/environment/bet_abstraction.py](technical_sources/PYTHON_SOURCE_07.md#py-078) | 엔진의 법적 베팅 문맥 → 7개 후보와 mask |
| [src/holdem_ai/training/reward_models.py](technical_sources/PYTHON_SOURCE_17.md#py-198) | 시작/종료 칩 또는 우승 좌석 → utility 벡터 |
| [src/holdem_ai/training/episode_runner.py](technical_sources/PYTHON_SOURCE_16.md#py-192) | 환경+좌석별 정책 → 전원 보상과 같은 좌석 기준 transition |
| [src/holdem_ai/environment/rlcard_adapter.py](technical_sources/PYTHON_SOURCE_08.md#py-092) | RLCard 상태 → 표준 Turn/Terminal; private-field whitelist |
| [src/holdem_ai/utils/snapshot.py](technical_sources/PYTHON_SOURCE_17.md#py-203) | 실험적 복제; 엔진별 RNG 독립성 검증 필요 |

7개 정책 action은 `fold, check, call, raise_half_pot, raise_pot, raise_double_pot, all_in`이다. 스트리트 누적 기여 `c`, 추가 콜 필요액 `k`, 행동 직전 pot `P`, 배수 `f`에 대해:

```math
B_{\mathrm{to}}^{(f)}=c+k+\mathrm{round}_{\mathrm{chip}}[f(P+k)],
\qquad f\in\{1/2,1,2\}.
```

`raise_to`는 해당 스트리트의 **총 기여 목표액**이다. 최소/최대 레이즈, short all-in, 재레이즈 재개 여부는 엔진이 결정한다. 후보는 경계에 맞추고 중복을 제거하며 일반 raise가 all-in과 겹치면 전용 all-in head를 남긴다. 실제 정수 반올림과 중복 제거 구현:

```python
# src/holdem_ai/environment/bet_abstraction.py
extra = (numer * (ctx.pot + ctx.to_call) + denom // 2) // denom
target = min(max(called_to + extra, ctx.min_raise_to), maximum)
if target <= called_to or target >= all_in_to or target in seen_targets:
    continue
actions[index] = EngineBetAction("raise_to", target)
seen_targets.add(target)
```

```math
r_i^{\mathrm{hand}}=\frac{S_{i,\mathrm{end}}-S_{i,\mathrm{start}}}{BB},
\qquad
r_i^{\mathrm{winner}}=\mathbf1[i=w]-\frac1N.
```

두 보상은 목적이 다르다. 핸드 chip EV와 토너먼트 우승 utility를 같은 값으로 섞지 않는다. `run_self_play`의 `next_observation`은 **같은 좌석의 다음 차례**이며, 그 좌석의 마지막 transition에만 terminal reward를 붙인다. 전체 `EpisodeResult.rewards`에는 행동하지 않은 좌석도 포함된다.

```python
# src/holdem_ai/training/episode_runner.py
for seat, (index, obs, action) in pending.items():
    completed.append(Transition(index, seat, obs, action, None, outcome.rewards[seat], True))
```

실행 흐름은 `reset → Turn → policy.act → assert_legal → step` 반복 후 `Terminal → pending transition 종료`다. 역사적 통합 기록은 **176 passed, 1 xfailed**를 보고하며, 초기 키트 자체 문서는 RLCard 실물 연결 일부가 skip였음을 명시한다. 따라서 서로 다른 시점의 검증 범위를 구분해야 한다. `infoset_key()`의 안정적 JSON과 NumPy 직렬화 보정은 재현성에 유용하지만 perfect recall을 증명하지 않는다. Snapshot도 실제 덱·RNG·공개 이력의 분기 독립성을 별도로 검사해야 한다.

### P6 — HUNL200 Adapter: 실제 엔진과 학습용 상태의 분리

```math
S_{0,\mathrm{start}}=S_{1,\mathrm{start}}=200BB,
\quad BB=100\text{ chips},\quad SB=50\text{ chips},
\quad R_0+R_1=0.
```

```math
I_t^i=\mathtt{observe}(i),\qquad
s_t^{C}=\mathtt{training\_state}(),\qquad
\pi_\theta\text{의 입력은 }I_t^i\text{만 허용}.
```

`I`에는 자기 카드·공개 보드·베팅·자리·mask가 들어간다. 별도 `CriticState`에는 두 좌석의 홀카드가 들어가지만 미공개 미래 덱 순서는 들어가지 않는다. 목적은 동일한 `MultiPlayerHandEngine`을 2인 모드로 사용하면서 학습 중 중앙 Critic의 특권과 실제 플레이 Actor의 정보 집합을 분리하는 것이다.

핵심 파일은 [src/holdem_ai/environment/hunl200_adapter.py](technical_sources/PYTHON_SOURCE_08.md#py-085), [src/holdem_ai/training/global_rollout.py](technical_sources/PYTHON_SOURCE_17.md#py-193), [src/holdem_ai/evaluation/benchmark.py](technical_sources/PYTHON_SOURCE_09.md#py-099) 및 [scripts/phase6_benchmark.py](technical_sources/PYTHON_SOURCE_05.md#py-049), [scripts/phase6_hardware_probe.py](technical_sources/PYTHON_SOURCE_05.md#py-050)다. Adapter는 매 `reset`에서 독립 200BB 핸드를 만든다. Policy head **4는 pot raise**, engine ID **4는 all-in**이므로 숫자만 전달하면 다른 행동이 된다.

| Policy head | 엔진 호출 |
|---:|---|
| 0 Fold | `engine.step(0)` |
| 1 Check / 2 Call | `engine.step(1)` |
| 3/4/5 Pot fractions | `engine.step_raise_to(target)` |
| 6 All-in | `engine.step(4)` |

```python
# src/holdem_ai/environment/hunl200_adapter.py
if not self.expose_training_state:
    raise PermissionError("pass expose_training_state=True only in the training collector")
engine = self._require_engine()
```

`reset → _receive → observe(current_player) → abstract_bet_actions → InformationState`의 순서로 관측을 만들고, `step → _receive`에서 엔진 종료와 zero-sum을 검사해 칩 payoff를 BB로 바꾼다. 훈련 collector만 `expose_training_state=True`와 `privileged=True`를 함께 켠다.

P5와 P6의 시간축은 다르다:

```math
\tau^{P5}_{i,n}: I_{i,n}\longrightarrow I_{i,n+1},
\qquad
\tau^{P6}_{t}: s_t\longrightarrow s_{t+1},
```

P6 `collect_global_episode`는 상대 차례도 포함한 **인접 global decision**을 저장한다. 보드 dealing은 엔진 transition 안에 포함되며 별도의 정책 action이 아니다. 이 구분은 이후 TD/GAE를 구현할 때 중요하지만, 현재 Base PPO는 terminal MC를 사용한다. 역사적 문서의 약 **2,896 random hands/s**는 CPU 무작위 정책 처리량이며 신경망·optimizer를 포함한 학습 속도로 환산할 수 없다.

### P7 — 관측 인코딩: 카드·행동 이력을 Tensor로 변환

```math
E_A(I)=(T,V,F_A,m),\quad |F_A|=20;
\qquad
E_C(s^C)=(T_C,V_C,F_C),\quad |F_C|=24.
```

`T`는 token ID, `V`는 token별 수치, `F`는 고정 길이 feature, `m`은 7-action legality다. 73-token vocabulary는 PAD와 역할/좌석 표식, 52장 카드, 4개 street, 5개 행동 종류로 구성된다. 금액은 `raise_to / BB`를 AMOUNT token의 별도 값으로 넣는다. 카드 문자열은 프로젝트 표준 `sK`, `d7`, `cT`, `hA`와 일치한다.

| Python 경로 | 역할 |
|---|---|
| [src/holdem_ai/encoding/vocabulary.py](technical_sources/PYTHON_SOURCE_07.md#py-075) | 73개 ID와 canonical card 매핑 |
| [src/holdem_ai/encoding/observation.py](technical_sources/PYTHON_SOURCE_07.md#py-073) | `encode_actor`, `encode_critic`, 엄격한 history parser |
| [src/holdem_ai/encoding/tensors.py](technical_sources/PYTHON_SOURCE_07.md#py-074) | 우측 padding, attention mask, Actor/Critic batch 타입 |
| [src/holdem_ai/encoding/actions.py](technical_sources/PYTHON_SOURCE_06.md#py-072) | one-hot action과 `[B,7]` masked categorical |
| [scripts/phase7_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-051) | 실제 관측의 인코딩 경로 확인 |

```python
# src/holdem_ai/encoding/observation.py
if len(ids) > max_tokens:
    raise ValueError(f'observation needs {len(ids)} tokens, exceeds max_tokens={max_tokens}; history not truncated')
return tuple(ids), tuple(values)
```

실행 흐름은 `InformationState → encode_actor → EncodedActor → batch_actors → ActorBatch`다. Critic에는 별도의 `encode_critic → batch_critics`만 사용한다. Actor 20 feature는 Adapter의 12 수치와 두 좌석의 4종 status one-hot 8개다. Critic 24 feature는 BTN·행동좌석·street, 두 stack/기여/bet, pot, status, terminal 표식이다.

불법 logits는 최소 유한값으로 바꾸고 categorical을 구성한다. history는 기본 256 tokens를 넘으면 예외를 내며 과거 기록을 자동 삭제하지 않는다. 다만 **원천 P6 이력은 call/all-in의 실제 지출액을 모두 기록하지 않는다.** 현재 parser도 `raise_to`에만 금액을 허용한다. 인코더가 받은 이력을 보존한다는 것과 포커의 완전한 perfect recall은 별개다. 이 인코딩과 7-action head는 VRPO 원 논문의 base-16 다단계 wager encoding을 구현한 것으로 읽어서는 안 된다.

### P8 — Transformer Actor와 중앙 Q-Critic

```math
x_j=E_{\mathrm{token}}(T_j)+E_{\mathrm{pos}}(j)
+W_v\mathrm{asinh}(V_j),
\quad z=\mathrm{LN}(\mathrm{Transformer}_{\mathrm{causal}}(x)_{L-1}).
```

```math
\pi_\theta(a\mid I)=\mathrm{MaskedSoftmax}\left(
\mathrm{MLP}_A[z,\mathrm{asinh}(F_A)]\right)_a.
```

```math
Q_{\phi,0}(s,a)=q_\phi(s,a),\quad
Q_{\phi,1}(s,a)=-q_\phi(s,a),\quad Q_\phi(s_{\mathrm{terminal}},a)=0.
```

`asinh`는 큰 수치의 범위를 압축하면서 0과 양·음의 방향을 유지한다. `L−1`은 각 샘플의 마지막 유효 token 위치이며 padding 끝이 아니다. Actor는 `ActorBatch`만 받는다. Critic의 `[B,2,7]`은 같은 global state와 같은 선택 행동에 대한 두 좌석의 수익이며, 두 플레이어가 동시에 자기 행동을 선택하는 구조가 아니다.

실제 Python은 [src/holdem_ai/models/transformer.py](technical_sources/PYTHON_SOURCE_16.md#py-183)의 `NetworkConfig`, `CausalTokenEncoder`, `TransformerActor`, `CentralizedQCritic`, `chosen_q`다. [src/holdem_ai/agents/neural_policy.py](technical_sources/PYTHON_SOURCE_06.md#py-065)는 네트워크를 P5 policy로 감싸며 [scripts/phase8_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-052)는 forward/backward를 수행한다. 기본값은 embedding=64, heads=2, layers=2, FFN=128, hidden=128, dropout=0, max_tokens=256이다. 역사적 문서의 파라미터 수는 Actor **99,975**, Critic **100,487**이다.

```python
# src/holdem_ai/models/transformer.py
last = attention_mask.long().sum(dim=1) - 1
return self.output_norm(x[torch.arange(bsz, device=x.device), last])
```

```python
# src/holdem_ai/models/transformer.py
q0 = q0.masked_fill(batch.features[:, -1:].to(dtype=torch.bool), 0.0)
return torch.stack((q0, -q0), dim=1)
```

`NeuralActorPolicy.act`는 `encode_actor → batch_actors([row]) → actor.distribution → greedy argmax 또는 Python RNG sampling`으로 실행되며 Critic을 호출하지 않는다. 목적은 훈련·저장·GUI 추론이 가능한 정책과 훈련 baseline을 마련하는 것이다. P8 smoke는 forward/backward만 수행하며 optimizer step이 없어 학습 성능을 주장할 수 없다. 제로섬 Q 제약은 여기의 2인 무레이크 chip 목표에 해당하며 다인·상금/ICM 목표에서는 다시 설계해야 한다.

### P9 — Terminal Monte-Carlo PPO 기준선

```math
G_t^{i_t}=R_{i_t},\qquad\gamma=1,
\qquad \widehat V_0(s_t,I_t)=\sum_a\pi_0(a\mid I_t)\widehat Q_{0,i_t}(s_t,a).
```

```math
A_t=G_t^{i_t}-\widehat V_0(s_t,I_t),\quad
\widetilde A_t=\frac{A_t-\mu_A}{\max(\sigma_A,10^{-6})},
\quad \rho_t(\theta)=\exp[\log\pi_\theta(a_t\mid I_t)-\log\pi_0(a_t\mid I_t)].
```

```math
J_{\mathrm{PPO}}=\frac1N\sum_t
\min\!\left(\rho_t\widetilde A_t,
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\widetilde A_t\right),
\quad \mathcal L_A=-J_{\mathrm{PPO}}-c_H\overline H(\pi_\theta).
```

```math
\mathcal L_Q=\frac1N\sum_t
\mathrm{SmoothL1}\bigl(Q_{\phi,i_t}(s_t,a_t),G_t^{i_t}\bigr).
```

`i_t`는 현재 행동 좌석, `π₀`와 `Q₀`는 update 직전 동결한 정책과 Critic, `N`은 실제 decision 수다. 현재 Base 구현은 **GAE·TD(0) 없이 terminal MC payoff를 같은 핸드의 각 행동 좌석에 귀속**한다. 상대 차례를 거쳤다고 reward 부호를 무조건 바꾸지 않고 저장된 `acting_seats`로 선택한다. Baseline은 모든 합법 action을 평균하므로 sampled action에 의존하지 않는다. 반면 selected-action Q를 baseline으로 직접 빼면 action-dependent 편향을 보정해야 한다.

[src/holdem_ai/training/ppo.py](technical_sources/PYTHON_SOURCE_17.md#py-194)는 `PPOConfig`, `TrajectoryRows`, rollout, frozen data, clipped objective, optimizer와 초기 Random 평가를 제공한다. [src/holdem_ai/training/ppo_session.py](technical_sources/PYTHON_SOURCE_17.md#py-195)는 모델·두 Adam·설정·counter·Torch RNG checkpoint와 resume, [scripts/phase9_train.py](technical_sources/PYTHON_SOURCE_05.md#py-053)는 CLI다. 기본 LR=3e-4, clip=0.2, entropy_coef=0.01, value_coef=0.5, grad_clip=0.5다.

```python
# src/holdem_ai/training/ppo.py
v = (dist.probs * q[indexes, seats]).sum(dim=1)
old_logs.append(dist.log_prob(actions).cpu())
baseline_values.append(v.cpu())
```

```python
# src/holdem_ai/training/ppo.py
ratio = torch.exp(log_prob - old_log_prob)
return torch.minimum(ratio * advantage,
                     torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage)
```

실행은 `collect_self_play → freeze_old_policy_data → optimize_ppo → save_checkpoint → evaluate_vs_random`다. Actor optimizer는 policy+entropy loss, Critic optimizer는 `value_coef × SmoothL1`를 역전파하며 둘 다 gradient clipping과 finite 검사를 한다. **Q의 회귀 target은 실제 선택한 action 하나에만 존재한다.** 다른 6개 action의 counterfactual truth를 한 핸드에서 얻지는 않는다.

중간 `approx_kl`은 `mean(ρ-1-logρ)`이며, 코드에서 optimizer step 이후 계산해도 사용되는 `new_log`는 **그 step 이전 forward 결과**다. 따라서 전체 update 완료 뒤 새로운 policy 분포를 다시 계산한 drift 지표가 아니다. P16에서 이 한계를 해결하는 별도 post-update 진단을 추가한다. 또한 경험 advantage의 중심화·표준화는 유한 표본에 의존하므로 이상적인 action-independent baseline의 무편향성 주장을 그대로 전체 실용 PPO update에 확장하지 않는다. PPO clipping과 여러 epoch는 surrogate 최적화이며 Nash 수렴 보장은 아니다.

### P10 — Q-Boost: Action-dependent control variate와 해석적 보정

```math
A_t^{MC}=G_t-V_0,
\quad \delta_t^{(\beta)}=G_t-V_0-\beta(Q_0(s_t,a_t)-V_0),
\quad \beta\in[0,1].
```

```math
J_{QB}(\theta)=\frac1N\sum_t\left[
\min\!\left(\rho_t\frac{\delta_t^{(\beta)}-\mu_A}{\sigma_A},
\mathrm{clip}(\rho_t)\frac{\delta_t^{(\beta)}-\mu_A}{\sigma_A}\right)
+\frac\beta{\sigma_A}\sum_a(\pi_\theta(a\mid I_t)-\pi_0(a\mid I_t))Q_0(s_t,a)
\right].
```

여기서 `clip(ρ)`는 P9와 같은 `[1−ε,1+ε]`, `μ_A,σ_A`는 **unboosted MC advantage에서 한 번 산정한 공통 값**이다. Residual과 analytic term을 각각 표준화하면 보정의 상쇄 관계가 깨진다. β는 rollout 이전에 고정하며 `β=0`일 때 같은 frozen data의 P9 PPO로 돌아간다.

목적은 예측 Q를 이용해 Monte-Carlo policy-gradient의 변동을 줄일 수 있는지 **측정**하는 것이다. `G−Q(a)`만 쓰면 Q가 설명한 행동 기여 자체를 빼버리므로 다음 해석적 보정이 필요하다. 고정 상태와 old-policy 점에서, advantage 정규화 상수를 고정한 이상적 기대 gradient는:

```math
\left.\mathbb E_{a\sim\pi_0}\left[
\nabla_\theta\log\pi_\theta(a\mid I)
\{G-V_0-\beta(Q_0(a)-V_0)\}\right]\right|_{\theta_0}
+\left.\beta\sum_a\nabla_\theta\pi_\theta(a\mid I)Q_0(a)\right|_{\theta_0}
=\left.\sum_a\nabla_\theta\pi_\theta(a\mid I)Q^{\pi_0}(s,a)\right|_{\theta_0}.
```

이 항등식은 Q가 정확하지 않아도 성립하는 control-variate 원리를 설명한다. 실용 PPO clipping 이후의 임의 θ에서 무편향이라는 주장, 실제 finite-batch 표준화까지 포함한 완전한 무편향 보장은 아니다.

핵심은 [src/holdem_ai/training/qboost.py](technical_sources/PYTHON_SOURCE_17.md#py-196)의 `freeze_qboost_data`, `qboost_surrogate`, `qboost_diagnostics`, `optimize_qboost`; [src/holdem_ai/training/qboost_session.py](technical_sources/PYTHON_SOURCE_17.md#py-197)는 Phase9 warm-start와 Phase10 checkpoint, [scripts/phase10_train.py](technical_sources/PYTHON_SOURCE_03.md#py-027)는 matched-start 비교 CLI다.

```python
# src/holdem_ai/training/qboost.py
observed_q = q_values.gather(1, actions[:, None]).squeeze(1)
residual = (returns - baseline - beta * (observed_q - baseline) - center) / scale
mc_clip = clipped_surrogate(new_log, old_log, residual, clip_epsilon)
analytic = beta * ((new_probs - old_probs) * q_values).sum(-1) / scale
return mc_clip + analytic
```

Actor loss는 `−J_QB−c_HH`, Critic loss는 P9와 동일하다. old probabilities와 모든 Q는 optimizer epoch 전체에서 고정된다. QBoost가 divergent PPO arm의 rollout을 그대로 재사용하면 on-policy 비교가 깨지므로 이후 각 arm은 자기 최신 policy로 새로 수집한다.

진단은 residual variance와 **logit-gradient proxy**를 구분한다:

```math
g_t^{PPO}=(e_{a_t}-\pi_0)\widetilde A_t,\quad
g_t^{QB}=(e_{a_t}-\pi_0)\widetilde\delta_t
+\frac\beta{\sigma_A}\pi_0\odot(Q_0-V_0),
\quad \rho_{var}=\frac{\sum_j\widehat{\mathrm{Var}}(g_{t,j}^{QB})}
{\sum_j\widehat{\mathrm{Var}}(g_{t,j}^{PPO})}.
```

`ρ_var<1`은 해당 표본의 logit proxy 분산 감소이며 전체 Transformer parameter-gradient 분산 감소 또는 전략 EV 개선이 아니다. 역사적 문서는 초기 비율이 약 1이었다고 기록한다. 이 구현은 독립적으로 구성한 VRPO-inspired Lite 실험이며 원 논문 전체 재현이 아니다.

### P11 — Batched Self-Play와 시간 예산 학습

```math
X_t=\mathrm{Batch}\{E_A(I_{e,t}^{i_e}):e\in\mathcal E_t\},
\quad D_k\sim\pi_{\theta_k},
\quad \theta_{k+1}=\mathrm{Update}(\theta_k,D_k).
```

`𝓔_t`는 한 프로세스에서 살아 있는 독립 핸드 집합이다. 한 forward batch마다 각 unfinished hand가 **한 decision만** 진행하고, 모두 종료한 뒤 payoff를 반환한다. 목적은 GPU의 per-decision 호출 비용을 여러 관측으로 나누면서 동일한 on-policy 수집 계약을 유지하는 것이다. 멀티프로세스 GPU inference server나 replay 학습으로 해석하면 안 된다.

[src/holdem_ai/training/batched_collector.py](technical_sources/PYTHON_SOURCE_16.md#py-191)는 `collect_batched_self_play`; [src/holdem_ai/experiments/time_budget.py](technical_sources/PYTHON_SOURCE_15.md#py-173)는 session 생성·warm-start·loop·평가·원자 저장; [src/holdem_ai/utils/progress.py](technical_sources/PYTHON_SOURCE_17.md#py-202)는 heartbeat/ETA를 담당한다. [scripts/phase11_train.py](technical_sources/PYTHON_SOURCE_03.md#py-030), `phase11_benchmark.py`, `phase11_compare.py`는 학습·추론 처리량·독립 log 비교다.

```python
# src/holdem_ai/training/batched_collector.py
with torch.inference_mode():
    batch = batch_actors(actor_rows, device=device)
    probabilities = actor.distribution(batch).probs.detach().cpu().tolist()
```

환경과 action RNG는 핸드별 `seed=base_seed+iteration×H+hand_index`, BTN은 global hand index로 교대한다. Privileged state는 별도 Critic rows에 저장하며 Actor batch에는 넣지 않는다. 모든 decision의 return은 terminal seat reward다. `dropout=0`을 강제해 수집 policy와 frozen log probability의 재현성을 유지한다.

흐름은 `run_experiment → collect_batched_self_play → freeze → 기존 PPO/QBoost optimize → atomic checkpoint → metrics append → optional evaluation`다. `run.json`은 초기 설정, `anchor.pt`는 frozen 시작 Actor, `metrics.jsonl`은 append-only events, `summary.json`은 마지막 호출 요약이다. Ctrl+C의 부분 update는 완료 checkpoint로 되돌아가며, checkpoint 저장과 log append 사이 crash는 resume event로 설명한다. 시간 제한과 `max_updates`는 **각 호출의 새 예산/추가 횟수**이고 진행 중 update·평가를 완료한 경계에서 멈춘다.

초기 GPU pilot의 **30–45 hands/s**는 당시 문서의 관측이며 P6 random CPU speed와 작업량이 다르다. `PHASE11_PROGRESS`는 사용자 Windows/CUDA **304 passed**를 기록한다. Progress heartbeat는 표시만 갱신하며 모델 학습이나 RNG를 바꾸지 않는다. 초기 repeated greedy 평가에서 frozen-anchor 값이 같았던 현상은 작은 weight 변화로 argmax가 유지될 수 있어 결함의 충분조건이 아니다. 이를 분리해 확인하는 것이 P12다.

### P12 — Seed-blocked 평가: 카드를 통계 단위로 취급

```math
X_k^A=\frac14\sum_{b=0}^{1}\sum_{s=0}^{1}R_A(k,b,s),
\quad \widehat J_A=100\bar X^A,
\quad \mathrm{SE}(\widehat J_A)=100\frac{s_X}{\sqrt K}.
```

```math
CI_{95\%}\approx\widehat J_A\pm1.96\mathrm{SE}(\widehat J_A),
\quad D_k=X_k^A-X_k^C,
\quad \widehat{\Delta J}=100\bar D.
```

`k`는 shuffled-deck seed, `b`는 BTN 좌석, `s`는 후보의 물리적 좌석이다. 같은 seed에서 네 레이아웃을 수행하므로 독립 단위는 핸드 4개가 아니라 **seed block 하나**다. Control과 비교할 때는 동일 상대·같은 config·동일 seed에서 `D_k`의 분산으로 CI를 계산한다. 별개 CI의 끝점끼리 빼지 않는다.

[src/holdem_ai/evaluation/seed_blocked.py](technical_sources/PYTHON_SOURCE_10.md#py-109)의 `EvaluationConfig`, `DealGroup`, `evaluate_seed_blocks`, `summarize_block_scores`, `compare_results`; [src/holdem_ai/evaluation/checkpoint_loader.py](technical_sources/PYTHON_SOURCE_09.md#py-100)와 [scripts/phase12_evaluate.py](technical_sources/PYTHON_SOURCE_03.md#py-031)가 핵심이다.

```python
# src/holdem_ai/evaluation/seed_blocked.py
if candidate.opponent_id != control.opponent_id:
    raise ValueError('comparison requires exactly the same opponent')
if tuple(g.seed for g in candidate.blocks) != tuple(g.seed for g in control.blocks):
    raise ValueError('comparison requires the same shuffled-deck seeds')
deltas = tuple(a - b for a, b in zip(candidate.means_bb, control.means_bb))
```

`load_actor → evaluate_seed_blocks → 4 layouts/seed → group summary → JSON/CSV`가 실행 흐름이다. `greedy`와 reproducible `sampled`, validation/test seed namespace, action counts, model fingerprint가 기록된다. Optimizer와 training_state는 평가에서 사용하지 않는다. 별개 instance의 동일 weighted policy끼리 좌석 label을 바꿔 평가하면 seed-group payoff가 정확히 0이 되는 symmetry sanity check를 제공한다.

같은 deck seed를 쓰더라도 다른 정책의 betting 경로와 종료 시점은 달라질 수 있다. 이 평가는 seed-matched 좌석 균형 실험이며 unplayed action의 counterfactual을 생성하지 않는다. 작은 K의 정규근사 CI, 반복 validation 선택 편향, 강한 상대 부재를 고려해야 한다. Random/frozen Actor 상대 성적은 exploitability가 아니다.

### P13 — Batch Scaling: 학습 처리량과 행동 조건부 진단

```math
N_{\mathrm{mini},k}=E\left\lceil\frac{N_k}{M}\right\rceil,
\quad \widehat T_{\mathrm{train}}=\frac{\sum_kH_k}{\sum_k\Delta t_{\mathrm{update},k}},
\quad \widehat p(a\mid legal)=\frac{C_a}{L_a}.
```

`H`는 hands/update, `B`는 active env 수, `M`은 decisions/minibatch, `E`는 epoch, `N_k`는 fold/raise에 따라 달라지는 실제 decision 수다. `C_a`는 action 선택 횟수, `L_a`는 합법 opportunity 횟수이며 `L_a=0`이면 `null`이다. 이 비율은 방문 상태 전체의 조건부 빈도이며 특정 infoset의 `π(a|I)`와 같지 않다.

```math
H_{norm}(I)=\begin{cases}
H(\pi(\cdot|I))/\log|\mathcal A(I)|,&|\mathcal A(I)|>1,\\0,&|\mathcal A(I)|=1.
\end{cases}
```

목적은 P11 inference benchmark에서 한 단계 나아가 **수집·진단·optimizer를 포함한 update 처리량**을 측정하고 fold/check 빈도의 분모 오류를 줄이는 것이다. [src/holdem_ai/experiments/batch_scaling.py](technical_sources/PYTHON_SOURCE_14.md#py-166), [src/holdem_ai/evaluation/action_diagnostics.py](technical_sources/PYTHON_SOURCE_09.md#py-097), `reward_diagnostics.py`와 [scripts/phase13_train.py](technical_sources/PYTHON_SOURCE_03.md#py-034), `phase13_benchmark.py`, `phase13_compare.py`가 구성된다.

```python
# src/holdem_ai/experiments/batch_scaling.py
rows = collect_batched_self_play(base.actor, base.ppo_config,
                                 iteration=iteration, device=base.device,
                                 env_batch_size=config.env_batch_size)
diagnostics = action_diagnostics(rows, base.actor, device=base.device,
                                 chunk_size=base.ppo_config.inference_batch_size)
reward_diagnostics = summarize_rollout_rewards(rows)
```

PPO/QBoost 수식은 바꾸지 않는다. Warm-start는 source를 보존하고 새 stage에 Actor/Critic/Adam을 가져오며, resume는 해당 stage의 저장 설정을 읽는다. 기본 새 seed는 `old_seed+1,000,003`이고 explicit override가 가능하다. Checkpoint payload는 계속 `phase=9/10`, run metadata는 `phase=13`이라 P12 loader와 호환된다.

역사적 GPU 기록에서 `H=64,B=32`는 **91.77 training hands/s**였지만 두 번의 측정에 기반한 시작 가설이다. H 변경은 `seed+iteration×H` 일정과 gradient noise도 바꾸므로 순수한 GPU batch 효율 효과만으로 단정하지 않는다. 수집·최적화 이외 evaluation/checkpoint I/O는 처리량 분모에서 제외되며 전체 wall time과 구별한다.

### P14 — PPO/QBoost 통제 실험: 알고리즘 효과와 Seed 변동 분리

```math
\theta_{s,0}^{PPO}=\theta_{s,0}^{QB}=\theta_{source},\quad
\phi_{s,0}^{PPO}=\phi_{s,0}^{QB}=\phi_{source},\quad
Adam_{s,0}^{PPO}=Adam_{s,0}^{QB}=Adam_{source}.
```

```math
\Delta_s=\frac{100}{K}\sum_k(X_{s,QB,k}-X_{s,PPO,k}),
\quad\bar\Delta=\frac1S\sum_s\Delta_s,
\quad CI_{95\%}\approx\bar\Delta\pm t_{S-1,0.975}\frac{s_\Delta}{\sqrt S}.
```

학습 seed `s`가 반복 실험 단위다. 두 arm은 **동일한 완전한 Phase9 checkpoint와 Adam 상태**에서 시작하며 latest on-policy rollout을 각각 새로 수집한다. 공통 shuffle schedule은 동일 action path를 의미하지 않는다. 수식과 optimizer 자체를 바꾸지 않고 시작점·예산·seed·선정 조건을 통제하는 것이 목적이다.

[src/holdem_ai/experiments/controlled_study.py](technical_sources/PYTHON_SOURCE_14.md#py-167)는 source hash 검증, arm 생성/resume, validation best 선정, 모든 arm 완료 후 test, seed 간 summary를 담당한다. [src/holdem_ai/evaluation/seed_statistics.py](technical_sources/PYTHON_SOURCE_10.md#py-110)는 seed-level 통계이고 [scripts/phase14_study.py](technical_sources/PYTHON_SOURCE_03.md#py-035)가 진입점이다.

```python
# src/holdem_ai/experiments/controlled_study.py
def _rollout_seed(seed: int) -> int:
    return 100_000_000 + seed * 1_000_000
```

실행 흐름은 `source hash + study manifest → seed별 PPO/QB arm → 반복 update/validation → best.pt → 모든 arm 완료 여부 → final test → paired deck/seed summary`다. Validation은 frozen 초기 Actor 상대 최고점을 쓰고 초기 Actor도 후보에 포함한다. Source hash나 과학적 설정이 바뀌면 resume를 거부한다. `arm-duration`에 걸려 partial이면 final test를 실행하지 않는다.

역사적 P15 문서는 P14 PPO validation 최고점을 seed51 **809.50**, seed52 **858.125**, seed53 **456.75 bb/100**로 기록했다. 이 값으로 seed52 checkpoint를 후속 출발점으로 선택했다. 세 학습 seed만으로 QBoost 우월성을 확정하지 않았으며 validation 최고점은 독립 test 결과가 아니다. 공유된 카드 은행에서 seed-level CI는 고정 test 카드 조건의 학습 변동성으로 읽어야 하고 카드·학습의 모든 불확실성을 동시에 추정하지 않는다.

### P15-A — Long-run: Checkpoint 선정과 Final 평가 봉인

```math
J_A(t)=\widehat J_{val}(\pi_t,\pi_0),\quad
J_R(t)=\widehat J_{val}(\pi_t,Random),
\quad
S_t=\frac{[J_A(t)-J_A(0)]+[J_R(t)-J_R(0)]}{2}.
```

```math
t^*=\arg\max_{t\in\mathcal T_{evaluated}}S_t,
\qquad S_0=0.
```

`π₀`는 stage 시작의 frozen Actor다. `latest.pt`는 resume 원천, `best.pt`는 반복 validation 선정 결과다. 목표는 알고리즘 개선이 아니라 장시간 학습의 기록·중단·복구·선정과 최종 평가 독립성이다.

[src/holdem_ai/experiments/longrun.py](technical_sources/PYTHON_SOURCE_14.md#py-168)의 `run_longrun`, `finalize_longrun`, `_exclusive_lock`, `_validation`; [scripts/phase15_train.py](technical_sources/PYTHON_SOURCE_04.md#py-042), `phase15_finalize.py`가 핵심이다. source/best SHA와 config를 검사하며 OS lock으로 동일 run의 동시 쓰기를 막는다. Training loop는 P13의 `_one_update`를 재사용한다. Finalize는 학습을 끝낸 best만 별도 final seeds에서 평가하고 `sealed`로 표시해 이후 resume를 차단한다.

```python
# src/holdem_ai/experiments/longrun.py
def run_longrun(cfg: LongRunConfig, *, emit: Callable[[dict],None] | None = None) -> dict:
    """OS-locked entry point for the crash-safe long-running training loop."""
    with _exclusive_lock(Path(cfg.root)):
        return _run_longrun_impl(cfg,emit=emit)
```

초기 safety는 `entropy<0.45 또는 approx_kl>0.10`이 12회 연속이면 종료였다. 이후 진단 패치 [src/holdem_ai/experiments/ppo_safety.py](technical_sources/PYTHON_SOURCE_15.md#py-171)는 경고와 hard stop을 분리한다:

```math
w_{stop,t}=w_{KL,t}\lor(\mathtt{entropy\_stop\_enabled}\land w_{H,t}).
```

```python
# src/holdem_ai/experiments/ppo_safety.py
low_entropy = entropy < entropy_threshold
high_kl = approx_kl > max_approx_kl
safety_violation = high_kl or (low_entropy and entropy_stop_enabled)
```

새 run의 낮은 entropy는 기본 경고로 기록하며, 옛 manifest에 field가 없는 resume는 당시 entropy-stop 의미를 유지한다. 이 변경은 collapse 해결의 입증이 아니라 불충분한 지표로 연구를 멈추는 문제를 줄이는 장치다. 같은 update에서 entropy와 KL 모두 경고를 기록해 한쪽이 다른 경고를 가리지 않는다.

**역사적 2026-10-13 실험 보고서의 관측**:

| 실험 | 신규 Updates / Hands | 실제 종료 | Best validation score |
|---|---:|---|---:|
| 초기 `H64/M64/E2` | 197 / 12,608 | Update301, 191.406초, safety_stop | 0, source Update104 |
| 안정성 pilot `H128/M128/E1` | 100 / 12,800 | Update204, 185.598초, update_limit | +98.25, Update125 |
| 위 pilot resume | 추가246 / 31,488 | Update450, 363.843초, safety_stop | +413.375, Update375 |

안정성 run 누적은 신규346 updates/44,288 hands다. Update375와450의 frozen-initial 평가 fold/all-action은 **82/828=9.90% → 314/565=55.58%**, Random 상대는 **76/698=10.89% → 268/493=54.36%**였다. 이는 fold-legal opportunity를 분모로 한 비율이 아니며 collapse의 원인을 확정하지 않는다. Update450의 selection score는 **−520 bb/100**였고 final은 봉인되지 않았다. `H,M,E`를 함께 바꿨기 때문에 한 요소의 인과 효과도 분리하지 못했다. 이 기록에서 CUDA peak는 102–161MiB의 PyTorch tensor allocation이지 GPU 전체 VRAM이 아니다.

### P15-B — Policy/Critic/Reward/Preflop 감사: 서로 다른 질문을 분리

```math
\overline\pi_m(a\mid legal;D)=
\frac{\sum_{I\in D}\mathbf1[a\in\mathcal A(I)]\pi_m(a\mid I)}
{\sum_{I\in D}\mathbf1[a\in\mathcal A(I)]}.
```

동일 관측 집합 `D`에서 모델별 확률을 계산하면 방문 분포의 차이를 고정해 policy 자체의 변화에 접근할 수 있다. [src/holdem_ai/evaluation/fixed_state_policy_audit.py](technical_sources/PYTHON_SOURCE_09.md#py-103)의 `compare_actors_on_states`와 [scripts/phase15_policy_audit.py](technical_sources/PYTHON_SOURCE_04.md#py-039)가 이를 수행한다. 역사적 문서는 Update375→450의 fixed-state mean fold probability **+0.343803** 변화가 주로 preflop BB에 집중됐다고 기록한다. Reference rollout 자체의 분포가 preflop 중심이면 postflop 일반화는 제한된다.

```python
# src/holdem_ai/evaluation/fixed_state_policy_audit.py
if bool(legal[i, 0]):
    bucket["fold_legal"] += 1
    bucket["fold_prob_sum"] += float(probs[i, 0])
```

Critic 감사는 taken-Q와 policy-weighted V를 별도로 비교한다:

```math
p_t^Q=Q_\phi(s_t,i_t,a_t),\quad
p_t^V=\sum_a\pi_{behavior}(a|I_t)Q_\phi(s_t,i_t,a),\quad e_t=p_t-G_t.
```

```math
Bias=\overline e,\quad MAE=\overline{|e|},\quad
RMSE=\sqrt{\overline{e^2}},\quad
EV=1-\frac{\mathrm{Var}(e)}{\mathrm{Var}(G)}.
```

[src/holdem_ai/evaluation/critic_diagnostics.py](technical_sources/PYTHON_SOURCE_09.md#py-102)의 `diagnose_critic_on_rows`, `run_critic_audit`와 [scripts/phase15_critic_audit.py](technical_sources/PYTHON_SOURCE_03.md#py-036)는 전체·street·seat·BTN/BB·action별 metric을 낸다. Reward variance가0이면 EV는 null이다. EV=1이어도 고정 bias가 클 수 있어 RMSE/Bias와 함께 읽는다.

```python
# src/holdem_ai/evaluation/critic_diagnostics.py
q_taken = chosen_q(q, seats, actions)
selected = q[torch.arange(stop - start, device=device), seats, :]
baseline = (selected * dist.probs).sum(dim=1)
```

`run_critic_audit`는 하나의 reference rollout을 두 critic에 넣는 controlled probe와, 각각 자신의 actor가 만든 별도 on-policy rollout을 모두 생성한다. Reference의 terminal return은 다른 candidate의 continuation Q에 대한 무편향 label이 아니므로 전자는 cross-policy generalization을 확인하는 보조 진단이다. 실제 calibration은 해당 candidate 자체의 on-policy 결과가 우선이다. 같은 핸드의 반복 decision은 상관돼 있어 decision row를 iid로 가정한 CI를 제공하지 않는다. 역사적 관측은 reward std 약 **193–196BB**, Q prediction이0 근처, EV≈0였으며 장기 학습 가설의 근거가 됐다.

Reward와 all-in 감사의 분모는 다음처럼 다르다:

```math
D_{hand}=\{R_{0,h}\}_{h=1}^{H},\qquad
D_{decision}=\{R_{i_{h,t},h}\}_{h,t},
```

```math
p_{hand,allin}=\frac{\#\{h:\exists t,street_t=PF,a_t=6\}}{H},
\quad p_{decision,allin|legal}=\frac{\#\{t:street_t=PF,a_t=6\}}
{\#\{t:street_t=PF,6\in\mathcal A(I_t)\}}.
```

| Python + 실행 CLI | 조사 대상 / 실제 흐름 |
|---|---|
| [src/holdem_ai/evaluation/terminal_reward_audit.py](technical_sources/PYTHON_SOURCE_10.md#py-113) + [scripts/phase15_terminal_reward_audit.py](technical_sources/PYTHON_SOURCE_04.md#py-041) | frozen rollout → hand 경계 검증 → hand/decision 보상분포·190/199/200BB 꼬리·last-street/fold 분류 |
| [src/holdem_ai/evaluation/initial_policy_comparison.py](technical_sources/PYTHON_SOURCE_09.md#py-104) + [scripts/phase15_initial_policy_comparison.py](technical_sources/PYTHON_SOURCE_04.md#py-038) | Uniform Legal·fresh random init·trained Phase14의 각자 self-play 및 공통 fixed states 확률 |
| [src/holdem_ai/evaluation/preflop_sequence_audit.py](technical_sources/PYTHON_SOURCE_09.md#py-107) + [scripts/phase15_preflop_sequence_audit.py](technical_sources/PYTHON_SOURCE_04.md#py-040) | 핸드별 ordered action → 전체 preflop sequence·첫 all-in prefix·깊이/position/to-call 조건부 rate |
| [src/holdem_ai/evaluation/reward_diagnostics.py](technical_sources/PYTHON_SOURCE_09.md#py-108) | update별 seat0 payoff·fold 종료·decision 수; authoritative 필드 없는 showdown은 null |

```python
# src/holdem_ai/evaluation/terminal_reward_audit.py
last_street = STREETS[streets[-1]]
fold = last_action == 0
preflop_all_in = any(a == 6 and st == 0 for a, st in zip(actions, streets))
any_all_in = 6 in actions
```

일반 Call도 stack을 소진할 수 있으므로 head6만 세면 실제 all-in commitment보다 적다. 마지막 decision이 preflop인 non-fold 종료는 자동 river runout 후 showdown일 수 있다. 93% decision이 preflop이었다는 것만으로 93% 핸드가 preflop fold 종료라고 결론내릴 수 없다. Head sequence도 정확한 wager 금액이나 정식 3-bet/4-bet 분류와 동일하지 않다.

과거 512-hand 비교의 보존 수치:

| 지표 | Uniform Legal | Synthetic random-init | Trained Phase14 |
|---|---:|---:|---:|
| Preflop decisions | 1,397 | 1,335 | 2,267 |
| 명시적 preflop all-in | 218 | 218 | 438 |
| All-in legal opportunities | 1,179 | 1,117 | 1,829 |
| Preflop terminal fold | 318 | 353 | 53 |
| Preflop terminal call | 97 | 84 | 433 |
| Pot raise / double-pot raise | 193 / 175 | 176 / 224 | 523 / 545 |

따라서 해당 trained 표본은 `438/512≈85.55%`의 hand-level all-in과 `438/1829≈23.95%`의 legal-decision rate를 동시에 가졌다. 다른 independent-seed sequence 감사는 **439/512=85.74%**, 대표 prefix `Pot → Double Pot → Double Pot → All-in` **115회**를 기록해 표본이 다름을 명시한다. Synthetic random-init은 실제 Phase14 학습의 첫 checkpoint와 동일하다는 보장이 없다. 이 감사는 read-only이며 checkpoint를 학습·변경·finalize하지 않는다. 단순 all-in 벌점을 추가하기 전에 abstraction branching, 반복 opportunity와 Critic calibration을 구분하는 데 효용이 있다.

### P16-A — 고정 Opponent Pool와 전략 진단

```math
\mathcal O=\{Random,CallingStation,Tight,PotPressure,FrozenActors\},
\quad J_{pool}=\frac1{|\mathcal O|}\sum_o\widehat J_o,
\quad J_{worst,observed}=\min_o\widehat J_o.
```

```math
SPR(I)=\frac{\min(S_{hero},S_{opp})}{\max(P,10^{-9})}.
```

목적은 한 상대를 이기는 정책이 과도한 call/fold/aggression 등 다른 스타일에서도 견디는지 관찰하는 것이다. 마지막 값은 관측 pool 중 최악 평균이지 minimax value·exploitability가 아니다.

[src/holdem_ai/agents/diagnostic_opponents.py](technical_sources/PYTHON_SOURCE_06.md#py-064)는 합법 fallback이 있는 규칙 정책; `evaluation/opponent_pool.py`는 opponent별 4-layout seed-block 경기; `evaluation/strategy_diagnostics.py`, `strategy_report.py`는 street·position·SPR·entropy와 JSON/CSV/Markdown; [scripts/phase16_evaluate.py](technical_sources/PYTHON_SOURCE_04.md#py-043)는 frozen checkpoint CLI다. SPR 구간은 `<1`, `[1,4)`, `[4,10)`, `≥10`이다.

```python
# src/holdem_ai/evaluation/opponent_pool.py
cfg = config or PoolEvaluationConfig()
if not opponents or len({x.name for x in opponents}) != len(opponents):
    raise ValueError('provide a nonempty pool with unique opponent names')
```

CallingStation은 check/call, Tight는 check/fold, PotPressure는 pot raise 중심의 **진단용 상대**다. 평가 중 Actor/optimizer는 갱신하지 않으며 CriticState는 전달하지 않는다. 상대별 CI는 seed-block 기준이고 다중비교 보정은 제공하지 않는다. 현재 pool evaluator만으로 PPO/QBoost 두 후보의 paired CI를 자동 제공하지는 않으며 공통 카드 group 결과를 사용하는 별도 comparison이 필요하다.

### P16-B — A0–A3 Ablation과 exact post-update KL

```math
\widehat D_{KL,post}=\frac1N\sum_{I\in D_{old}}
\sum_{a\in\mathcal A(I)}\pi_{old}(a|I)
\log\frac{\pi_{old}(a|I)}{\pi_{new}(a|I)}.
```

모든 action category를 실제로 합산한다는 의미에서 categorical exact다. 모든 가능한 poker infoset의 KL이 아니라 **동결 rollout states의 평균**이다. P9의 sampled `approx_kl`과 달리 optimizer 완료 후 Actor를 다시 forward한다.

| Arm | 수학·구현 변화 | 조사 목적 |
|---|---|---|
| A0 | 원래 PPO + 중앙 Q + BB Huber | control |
| A1 | `G′=G/200`, `Q′=Q/200`; Q 마지막 head weight/bias÷200 | target scale |
| A2 | `V(s,0)=v(s), V(s,1)=−v(s)` 직접 V head | unsupervised 대안-action Q를 평균하는 baseline 영향 |
| A3 | A0 + post KL>0.02면 Actor/Critic/두 Adam 모두 rollback | 큰 single-update policy drift |

[src/holdem_ai/experiments/ppo_ablation.py](technical_sources/PYTHON_SOURCE_15.md#py-169)의 `DirectVCritic`, `scale_q_head_`, `freeze_variant`, `optimize_variant`, `snapshot_probs`, `exact_policy_kl`, `critic_on_frozen_rows`, `run_ablation`; [scripts/phase16_ppo_ablation.py](technical_sources/PYTHON_SOURCE_04.md#py-044)가 이를 실행한다.

```python
# src/holdem_ai/experiments/ppo_ablation.py
local = torch.where(
    legal & (p > 0),
    p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log()),
    torch.zeros_like(p),
).sum(dim=1)
```

모든 arm은 동일 source Actor와 Actor Adam을 유지하되 **Critic Adam은 모두 reset**한다. 따라서 A0도 수학적 optimizer는 원래 PPO지만 옛 Phase15의 bit-for-bit resume는 아니다. A2는 구조까지 바뀌므로 단순 LR ablation으로 읽지 않는다. Pilot의 `final_actor.pt`는 Actor-only export로 Critic/Adam이 없어 학습 resume checkpoint가 아니다.

`source → variant 생성 → fresh critic Adam → own-policy rows → old freeze/probs → optimize → post KL → optional full rollback → holdout Critic + paired validation`가 흐름이다. 독립 10-update pilot 문서는 A1/A2의 확실한 개선을 입증하지 못했다고 기록했다. Gate가 구현됐다는 사실과 실제 A3가 학습을 개선했다는 증거는 구분한다.

### P16-C — A0/A3 장기 Stress와 full-state recovery

```math
H=128,\quad U_{attempt}=100,
\quad N_{train,arm}=12,800,
\quad N_{train,total}=25,600.
```

[src/holdem_ai/experiments/ppo_stress.py](technical_sources/PYTHON_SOURCE_15.md#py-172)는 A0/A3만 유지해 `run_stress`, `_one_update`, full checkpoint, log mirror 재생성, `compare_stress_runs`를 구현한다. [scripts/phase16_ppo_stress.py](technical_sources/PYTHON_SOURCE_04.md#py-045)의 campaign은 두 arm을 순차 실행하고 정렬된 validation을 비교한다.

```python
# src/holdem_ai/experiments/ppo_stress.py
session.actor.load_state_dict(actor_state)
session.critic.load_state_dict(critic_state)
session.actor_optimizer.load_state_dict(actor_opt_state)
session.critic_optimizer.load_state_dict(critic_opt_state)
```

A3가 거절한 update도 새로 수집한 card-seed block을 소비하고 **attempted counter를 증가**시킨다. 이를 되돌리면 같은 seed의 반복 수집으로 예산/비교가 왜곡된다. Accepted/rejected를 별도로 기록하며 `latest.pt`가 ledger·model·Adam·RNG의 권위다. Metrics/progress JSON은 그 checkpoint에서 재구성할 수 있고 주기별 snapshot은 변경하지 않는다. Manifest/source SHA와 lock이 동시 쓰기·잘못된 resume를 막는다.

역사적 P17 문서가 후속 기록한 100-update stress 결과는 **A3 rejection=0, frozen Phase14와 Random 상대 A3−A0=0 bb/100**이다. 그 run에서는 gate가 실제로 작동하지 않아 두 arm을 갈라놓지 않았고 Critic EV는 여전히0 근처였다. 100 attempts의 실행 성공은 장기 전략 안정성이나 GTO를 입증하지 않는다.

### P17-Eval — Continuous Stack: 가변 stack의 추론 강건성

```math
S_{0,1}+S_{1,1}=400BB,\quad
S_{i,h+1}^{start}=S_{i,h}^{final},\quad
\min_iS_{i,h}^{final}=0\Rightarrow MatchOver.
```

```math
(S_{candidate},S_{opp})=(s,400-s)BB,
\quad s\in\{50,100,200,300,350\}.
```

이는 GUI1.6와 같은 stack 승계 의미를 **offline 평가**로 확장한 Phase다. 기존 200BB-reset 학습을 변경하거나 variable-stack agent를 학습한 단계가 아니다. 첫 stack이 200/200이어도 다음 핸드에 승계되므로 P16 독립 핸드 평가와 다르다.

[src/holdem_ai/evaluation/continuous_stack.py](technical_sources/PYTHON_SOURCE_09.md#py-101)의 `_play_match`, `evaluate_continuous_stacks`; [src/holdem_ai/game_service/interactive_adapter.py](technical_sources/PYTHON_SOURCE_15.md#py-178)의 `reset_with_stacks`; [scripts/phase17_evaluate.py](technical_sources/PYTHON_SOURCE_04.md#py-046)가 핵심이다. Match마다 button을 교대하고 rebuy 없이 bankruptcy 또는 max-hands까지 실행한다. Deck/action RNG를 분리하며 decision limit의 unfinished hand는 오류를 내고 임의 payoff를 기록하지 않는다.

```python
# src/holdem_ai/evaluation/continuous_stack.py
transition = adapter.reset_with_stacks(stacks, seed=rng_deck.getrandbits(64))
```

```python
# src/holdem_ai/evaluation/continuous_stack.py
stacks = updated
hands += 1
if len(result.board) == 5 and not any(result.folded):
    showdowns += 1
```

서로 다른 match 길이를 group 안에서 실제 완료 핸드로 나눠 처리한다:

```math
b_g=100\frac{\sum_{\ell=1}^4\Delta S_{g,\ell}^{BB}}
{\sum_{\ell=1}^4H_{g,\ell}},\quad
\widehat b=\frac1G\sum_gb_g,\quad
CI_{95\%}\approx\widehat b\pm1.96\frac{s_b}{\sqrt G}.
```

각 seed group에 동일 weight를 준 평균이므로 모든 group의 net payoff/hand를 한꺼번에 합친 ratio와 일반적으로 같지 않다. Control이 있으면 같은 seed의 `b_candidate−b_control`로 paired CI를 계산한다. 추가 output은 bust/profit fraction, 평균 net BB/match, hands/match, hand win/loss/tie, authoritative showdown, **현재 uncommitted effective stack** 구간별 legal-conditioned action이다.

역사적 GUI16/P17 pilot는 **Phase14 seed41 QBoost, CallingStation, sampled, scenario별10 groups×4 matches, 최대20 hands/match**였다:

| 후보/상대 시작 BB | 완료 Hands | bb/100 | 근사95% CI | 후보 bust |
|---|---:|---:|---|---:|
| 50/350 | 151 | −621.854 | [−1554.417,+310.709] | 90.0% |
| 100/300 | 161 | −363.636 | [−1076.364,+349.091] | 77.5% |
| 200/200 | 109 | +318.182 | [−961.519,+1597.882] | 50.0% |
| 300/100 | 131 | −29.060 | [−2018.023,+1959.904] | 27.5% |
| 350/50 | 173 | −47.351 | [−1358.976,+1264.274] | 15.0% |

모든 CI가0을 포함하며 stack별 수익성이나 우열은 입증되지 않았다. 초기 bankroll 차이는 bust rate를 직접 바꾸므로 정책 강도의 단독 지표로 쓸 수 없다. 당시 Windows 통합 **690 passed, 1 xfailed, 1 warning** 역시 역사적 회귀 기록이다.

### P17-Retrain — Monitored GPU PPO: 누적 활성 시간과 update ceiling

```math
t_{active}=\sum_jt_{invocation,j},\quad t_{target}=18,000s,
\quad U_{attempt}=U_{accepted}+U_{rejected}.
```

이 Phase17은 위 평가와 **별도 작업**이다. [src/holdem_ai/experiments/ppo_five_hour.py](technical_sources/PYTHON_SOURCE_15.md#py-170)의 `FiveHourConfig`, `run_five_hour`, checkpoint/ledger/log mirrors; [scripts/phase17_gpu_retrain.py](technical_sources/PYTHON_SOURCE_04.md#py-047)는 누적5시간 목표를 관리한다. P16 stress의 원래 PPO/central Q와 post-KL 진단을 재사용한다. 출발점은 Phase14 **Update104**이며 source Actor/Actor Adam을 유지하고 Critic Adam은 reset한다.

P11/P15의 호출별 duration과 달리 여러 resume를 합친 **누적 활성 실행시간**을 센다. Human downtime은 제외하고 initial/intermediate evaluation·checkpoint I/O는 포함한다. Completed update 경계에서 저장하며 마지막 평가로 target을 조금 넘길 수 있다. Resume는 source/manifest SHA와 모든 과학적 설정을 검사한다.

```python
# src/holdem_ai/experiments/ppo_five_hour.py
if manifest.get("settings") != _settings(cfg):
    raise ValueError("scientific settings differ from saved run")
if manifest.get("source_sha256") != _sha(source):
    raise ValueError("original Phase14 source checkpoint changed")
```

A0는 post KL>0.10에서 이미 실행된 update를 보존하고 emergency halt하며 `hard_stop.json`이 있으면 자동 resume를 거부한다. A3는 선택 실험으로 >0.02에서 full rollback한다. `latest.pt`에는 Actor/Q/두 Adam/CPU+CUDA RNG/update accounting/time/manifest hash/metrics/validation ledger를 저장한다. `best.pt`는 두 frozen 상대에 대한 반복 paired change 평균이 높은 exploratory checkpoint다.

**2026-10-13 기존 통합보고서가 보존한 사용자 `summary.json`**:

| 항목 | 역사적 기록 |
|---|---:|
| Variant / protocol | A0 / `phase17_five_hour_ppo_v1` |
| Attempted / accepted / rejected | 5,000 / 5,000 / 0 |
| Global update | 5,104 |
| Training hands | 640,000 |
| Active seconds / target seconds | 12,743.274 / 18,000 |
| Validation count | 200 |
| Best global update | 2,104 |
| Best validation selection score | 1,450.4687 bb/100 |
| Stop / complete | `max_updates_ceiling` / `false` |

```math
5,000\times128=640,000\ \mathrm{hands},\quad
12,743.274s\approx3h32m23s,\quad
\frac{12,743.274}{18,000}\approx70.8\%.
```

즉 5,000-update ceiling을 완료했지만 5시간 활성 예산은 완료하지 않았다. `best.pt`는 Update2104, `latest.pt`는 Update5104로 용도가 다르다. 최고 validation score는 같은 카드 은행을 반복 사용해 선정한 값이며 독립 test EV가 아니다. 더 많은 hands를 수집했다는 사실만으로 P15의 preflop all-in/fold 편향·Critic EV≈0 문제 해결을 선언할 수 없다. 기존 sealed-final과 다른 독립 test, 여러 snapshot의 calibration/coverage, 다중 training seed 비교가 후속 과제다.

### HUNL Agent 트랙의 재현과 남은 개발 조건

`tests/test_phase5_*`부터 [tests/test_phase17_gpu_retrain.py](technical_sources/PYTHON_SOURCE_25B.md#py-296)까지 각 기능의 검증 소스 원문은 전체 Python 부록에서 확인할 수 있다. 주요 검증은 legality/mask·privacy·zero-sum, network shape/gradient, β0 equivalence·analytic correction, on-policy collector, seed-block statistics, source immutability, interrupted full-state recovery, forced unsafe-step rollback이다. 이번 편집은 이 테스트들을 실행한 GPU 연구가 아니다.

개발을 이어받을 때 먼저 **Base MC PPO와 Expand GAE**, **independent-reset와 continuous-stack**, **Actor-only export와 full training checkpoint**, **validation best와 latest resume**, **attempted와 accepted updates**를 구분해야 한다. Human GUI가 frozen Actor로 대국하고 표시용 history를 저장하는 단계와, human trajectory를 versioned on-policy 경험으로 수집해 재학습·독립 평가·승격하는 폐루프는 별개다. 현재 사람이 대국했다고 자동 PPO 학습이 진행되는 기능은 완성되지 않았다.

<a name="section-73301c2cf7bd"></a>

## Expand 프로젝트 — E1~E7

```math
\boxed{\text{E1 Table}\to\text{E2 InformationState}\to
\{\text{E3 PPO},\text{E4 Blueprint},\text{E5 Search}\}
\xrightarrow[\text{E6 public evidence}]{\text{E7 routing}}\text{legal table action}}
```

Expand의 목적은 Base의 2인 중심 학습 경로를 **물리 좌석 6개·참가자 2~6명·가변 스택**으로 확장하는 것이다. E3은 신경망 강화학습 경로, E4/E5는 MCCFR와 탐색 경로다. 이 두 경로는 서로 다른 모델·체크포인트·정보집합 계약을 사용한다. E7에서 하나의 인터페이스로 연결되지만 모든 경로가 자동으로 활성화되는 것은 아니다.

이 절의 코드 블록은 지정된 `.py` 원문에서 발췌했다. 원문 전체, 테스트, 실행 스크립트와 정확한 파일별 해시는 Python 원문 부록과 소스 매니페스트에서 해당 경로로 찾을 수 있다. 아래 실험 수치는 기존 보고서의 **역사적 기록**이며 이번 문서 통합 과정의 신규 학습·재실행 결과가 아니다.

| Phase | 개발 목적 | 얻는 효용 | 다음 단계가 요구하는 산출물 |
|---|---|---|---|
| E1 | 핸드 엔진 밖의 동적 캐시 테이블 관리 | 입퇴장·재구매·스택 이월과 칩 회계를 분리 | `DynamicTable`, `HandHandle`, `Settlement` |
| E2 | 6좌석 플레이어 관측과 합법 행동 번역 | 카드 비공개 경계와 신경망 입력을 고정 | `InformationState`, `EncodedObservation`, `LegalActionSpec` |
| E3 | 2~6인 캐시 게임 PPO | 학습 가능한 다인 신경망 기준선 | `expand-e03-v1` 체크포인트 |
| E4 | External-sampling linear MCCFR | 작은 게임에서 regret 로직을 검증 | Blueprint·추상화 fingerprint·정확한 infoset key |
| E5 | 현재 의사결정의 제한 탐색 | 공개 정보에 부합하는 세계를 샘플링해 국소 정책 계산 | `SearchResult`·시간/노드/메모리 추정량 |
| E6 | 공개 행동 상대 모델과 정책 변경 제한 | 스타일 변화 추적·증거 부족 시 기준 전략 유지 | `ModelEvidence`, `AdaptationResult` |
| E7 | 경로 통합·단일 테이블 토너먼트·평가 | 정책 출처·폴백·평가 단위를 추적 | `RoutedDecision`, tournament report, release manifest |

### E1 — Dynamic 6-Max Cash Table

```math
C_{\mathrm{live}}(t)=B_{\mathrm{cum}}(t)-W_{\mathrm{cum}}(t),\qquad
C_{\mathrm{live}}(t)=\sum_{i\in\mathrm{hand}}(s_i+c_i)+
\sum_{i\notin\mathrm{hand}}s_i.
```

```math
\sum_i S_{i,h}^{\mathrm{final}}=\sum_i S_{i,h}^{\mathrm{start}},\qquad
\sum_i r_{i,h}^{\mathrm{chips}}=0,\qquad
r_{i,h}^{\mathrm{BB}}=\frac{r_{i,h}^{\mathrm{chips}}}{BB}.
```

$`B_{\mathrm{cum}}`$은 누적 buy-in, $`W_{\mathrm{cum}}`$은 누적 cash-out, $`s_i`$는 잔여 stack, $`c_i`$는 현재 핸드의 누적 contribution이다. 핸드 도중에는 팟에 들어간 칩까지 포함하고, 종료 정산 후에는 최종 스택으로 보존식을 검사한다.

**목적·효용.** `MultiPlayerHandEngine`은 한 핸드의 베팅과 정산을 담당한다. E1의 `DynamicTable`은 여러 핸드에 걸친 참가자와 외부 자금 흐름을 담당한다. Join/leave/rebuy/sitout/resume 요청은 큐에 넣고 핸드 경계에서만 적용한다. 로스터를 진행 중에 바꾸지 않으므로 이전 참가자의 기여금·side pot과 새 참가자를 혼동하지 않는다. 일반 입장자는 자신이 BB를 낼 핸드까지 `waiting_bb`로 대기하며, 테이블을 시작하기 위한 최초 2인 구성에는 bootstrap 예외가 있다.

| `.py` 경로 — `src/holdem_ai/expand/environment/` | 역할 |
|---|---|
| `table_state.py` | `TableConfig`, `Action`, 좌석·핸들·이벤트·정산 DTO |
| `cash_rules.py` | buy-in·rebuy 금액과 캐시 규칙 검사 |
| `multiway_adapter.py` | Base 엔진 생성, surviving seat 기준 BTN·blind 이동 |
| `table_manager.py` | 요청 큐, 핸드 시작·행동·정산, 보존 검사, `replay_journal` |
| `event_ledger.py` | 순서 있는 JSON 이벤트 기록·읽기 |
| `tournament_rules.py` | E1에서 미구현 토너먼트 규칙을 명시적으로 거부 |

실제 코드 — `table_manager.py`, `apply_boundary_events()`:

```python
    def apply_boundary_events(self) -> None:
        if self._open_hand():
            raise RuntimeError('cannot apply pending changes during a hand')
        self.ledger.append('boundary_call')
        self._apply_pending()
```

**입력 → 처리 → 출력.** `request_join(player_id, chips, seat)`이 좌석을 예약하고 event ID를 반환한다. `start_hand(seed)`가 pending 요청 적용 → 로스터/BTN 선택 → Base 엔진 생성·reset → `HandHandle` 반환을 수행한다. `act(player_id, Action)`은 실제 행동자 확인 → `step`/`step_raise_to` → 보존 검사 → `StepEvent` 반환 순서다. terminal 이후 `settle_hand()`가 `Settlement`를 만들고 최종 스택을 테이블에 반영한 뒤 다음 boundary 요청을 적용한다. `replay_journal()`은 외부 명령을 다시 실행해 발생 효과를 비교한다. 재현용 seed가 들어 있는 journal은 privileged 실험 자료이며 Actor 관측에 제공하지 않는다.

**검증·한계.** [scripts/expand_phase01_smoke.py](technical_sources/PYTHON_SOURCE_01.md#py-012), [tests/expand/test_e01_table_unit.py](technical_sources/PYTHON_SOURCE_19.md#py-217), `test_e01_smoke_unit.py`, `test_e01_real_engine.py`가 경계 적용·좌석/자금 보존·실엔진 통합을 검증한다. 통합 보고서의 `164 passed, 7 skipped`에는 E1이 포함되지만 실엔진 부재로 skip된 사례가 있으므로 실제 6인 포커 규칙의 독립 인증으로 해석할 수 없다. E1은 cash 전용이며 단일 owner가 요청을 직렬화해야 한다.

### E2 — Information State, Seat Encoding, Action Translation

```math
I_i=(H_i,B,\mathrm{public\ seats},\mathrm{public\ history},\mathcal A_i^{\mathrm{legal}}),\qquad
H_{j\ne i}\notin I_i.
```

```math
x_i=\mathrm{Encode}(I_i):\quad
X_{\mathrm{seat}}\in\mathbb R^{6\times19},\quad
x_{\mathrm{global}}\in\mathbb R^{114},\quad
X_{\mathrm{history}}\in\mathbb R^{128\times21},\quad
m_{\mathrm{action}}\in\{0,1\}^{7}.
```

```math
g(x)=\mathrm{sgn}(x)\log(1+|x|),\qquad
P'=P+c,\quad T(k)=b_i+c+\lfloor kP'\rfloor,\quad k\in\{\tfrac12,1,2\}.
```

$`H_i`$는 자신의 두 카드, $`B`$는 공개 보드, $`P`$는 현재 팟, $`c`$는 콜 필요액, $`b_i`$는 현재 street에 이미 낸 금액이다. $`T(k)`$는 **이번에 추가로 내는 금액이 아니라 street의 최종 총 베팅액**이다. 일반 raise는 $`T(k)>b_i+c`$ 및 $`T_{\min}\le T(k)\le b_i+s_i`$를 만족해야 하며 엔진의 raise 가능 여부도 확인한다. 짧은 all-in call은 `call` 하나로 표현하고 동일 목표액의 중복 raise는 제거한다.

**원문과 현재 소스의 차이.** 과거 `EXPAND_PHASE02_INFORMATION_AND_ENCODING.md`의 수식은 global 차원을 110으로 적었지만 같은 문서의 구성도는 `52+52+4+6`이다. 현재 `encoder.py`는 `GLOBAL_DIM = 52 + 52 + 4 + 6`을 선언하고 실제 출력 길이도 검사하므로 **114가 실행 계약**이다. 기존 원문은 부록에 그대로 보존하고 본문은 source 확인값을 사용한다. 110차원 설명을 모델 입력 규격으로 적용해서는 안 된다.

**목적·효용.** 서로 다른 참가자 수에도 6개 물리 좌석 입력을 유지한다. 공개 좌석 상태와 private cards의 접근 범위를 명시하고, 칩/BB 단위와 action-head/engine-ID의 혼동을 막는다. 선언된 `TrainingOnlyCriticState`는 Actor encoder의 허용 타입이 아니며 E3의 value network도 현재 `EncodedObservation`만 사용한다.

| `.py` 경로 — `src/holdem_ai/expand/information/` | 역할 |
|---|---|
| `schema.py`, `builder.py` | 허용된 관측 DTO와 actor-safe engine observation의 변환 |
| `encoder.py` | 카드 one-hot, 공개 좌석·최대 128개 행동, padding masks |
| `actions.py` | 7개 행동 mask, 정수 raise-to 계산, `to_table_action` |
| `adapter.py` | E1 연결과 accepted action의 정확한 street 기록 |
| `policy_v2.py` | 좌석 Transformer pooling + 행동 이력 GRU + 7-action head |

실제 코드 — `actions.py`, `legal_action_spec()`의 raise 후보 계산:

```python
    for index, factor, engine_id in ((3, 1, 2), (4, 2, 3), (5, 4, None)):
        # exact integer math: floor(k * (pot + to_call)) where k is 1/2/2.
        post_call = pot + to_call
        amount = (post_call * factor) // 2
        target = b + to_call + amount
        permitted = (4 in ids and target > high and minimum <= target <= maximum)
        if engine_id is not None:
            permitted = permitted and engine_id in ids
        # Distinct legal actions, so no duplicate wager-size decisions.
        if permitted and target not in targets:
            allowed[index], targets[index] = True, target
```

```math
z=f_\theta(\mathrm{Pool}(\mathrm{Transformer}(X_{\mathrm{seat}})),
\mathrm{GRU}(X_{\mathrm{history}}),x_{\mathrm{global}}),\qquad
\pi_\theta(a\mid I)=\frac{m_a e^{z_a}}{\sum_bm_b e^{z_b}}.
```

**작동 순서.** 핸드 시작 전에 `Phase2TableAdapter(table)`을 부착한다. `observe()`가 E1 snapshot와 current actor observation을 검사하고 `make_information_state()`를 호출한다. `encode_observation()`이 log-scaled BB·카드·history tensor-ready 자료를 만든다. `PolicyV2.forward()`는 빈 좌석을 Transformer padding으로 제외하고 GRU에서 유효 이력 길이만 처리하며, illegal logits를 masking한다. 선택한 이름은 `to_table_action()`을 거쳐 E1 `Action`이 된다. underlying table이 먼저 행동을 수락한 뒤에만 `PublicAction`을 추가한다.

**검증·한계.** `expand_phase02_smoke.py`, `test_e02_state_and_actions.py`, `test_e02_policy.py`, `test_e02_adapter_fake.py`, `test_e02_real_engine.py`가 카드 중복·정보 경계·mask·exact wager·padding을 다룬다. E2 자체는 학습하지 않는다. E2의 공개 이력은 semantics를 보존하지만 E4가 요구하는 **원본 engine action ID trace**와 일반적으로 일대일 대응하지 않는다. 따라서 E2 관측을 E4 infoset으로 자동 변환하는 것은 미검증 상태다.

### E3 — Dynamic 2~6-Player Multi-Agent PPO

```math
r_{i,T_i}^{\mathrm{BB}}=\frac{S_{i,h}^{\mathrm{final}}-S_{i,h}^{\mathrm{start}}}{BB},\quad
r_{i,t<T_i}=0,\quad d_{i,T_i}=1.
```

```math
\delta_{i,t}=r_{i,t}+\gamma(1-d_{i,t})V(x_{i,t+1})-V(x_{i,t}),\quad
\widehat A_{i,t}=\delta_{i,t}+\gamma\lambda(1-d_{i,t})\widehat A_{i,t+1}.
```

```math
\rho_t=\frac{\pi_\theta(a_t\mid x_t)}{\pi_{\mathrm{old}}(a_t\mid x_t)},\quad
L=-\mathbb E[\min(\rho_t\widehat A_t,
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\widehat A_t)]
+c_V\mathbb E[(V_\theta-\widehat R)^2]-c_H\mathbb E[\mathcal H(\pi_\theta)].
```

$`T_i`$는 해당 플레이어의 마지막 의사결정 위치다. 다른 플레이어의 턴을 자기 trajectory의 다음 상태로 이어 붙이지 않는다. 기본값은 $`\gamma=1,\lambda=0.95,\epsilon=0.2,c_V=0.5,c_H=0.01`$이다.

**목적·효용.** 가변 인원과 공개 6좌석 관측에서 학습 가능한 기준 정책을 만든다. Collector는 update 동안 가중치를 고정해 on-policy 표본을 모은다. 자체 정책 공유·과거 snapshot·scripted 상대를 혼합하고 학습/평가 seed를 분리한다. E1의 BB 입장 warmup은 실제 정산하되 training hand 수에서 제외한다. 스택은 테이블 내에서 이월하지만 **GAE terminal은 매 핸드**다.

| `.py` 경로 — `src/holdem_ai/expand/rl/` | 역할 |
|---|---|
| `model.py` | E2 `PolicyV2`와 actor-visible `ValueV2`의 `ActorCritic` |
| `collector.py`, `population.py` | 2~6인 E1/E2 rollout, learner/snapshot/scripted 역할 |
| `returns.py` | session/hand/seat별 trajectory 그룹과 GAE |
| `ppo.py` | clipped objective, gradient clipping, nonfinite 거부, KL stop |
| `trainer.py` | 독점 run lock, 원자적 저장, validation, resume, trusted actor load |
| [scripts/expand_phase03_train.py](technical_sources/PYTHON_SOURCE_02.md#py-015), [scripts/expand_phase03_evaluate.py](technical_sources/PYTHON_SOURCE_02.md#py-014) | 학습·읽기 전용 평가 진입점 |

실제 코드 — `returns.py`, `prepare_gae()`의 trajectory 구분:

```python
    groups: dict[tuple[str, str, int], list[int]] = {}
    for k, t in enumerate(transitions):
        key = t.session_id, t.hand_id, t.seat_id
        groups.setdefault(key, []).append(k)
```

실제 코드 — `ppo.py`, `optimize_ppo()`의 objective:

```python
            ratio=log_ratio.exp()
            adv=advantages[indices]
            loss_pi=-torch.minimum(ratio*adv, ratio.clamp(1-config.clip_epsilon,1+config.clip_epsilon)*adv).mean()
            loss_v=torch.square(v-returns[indices]).mean()
            entropy=dist.entropy().mean()
            loss=loss_pi+config.value_coef*loss_v-config.entropy_coef*entropy
```

**작동 순서.** `train()` → run lock·config/checkpoint 복원 → `MultiAgentCollector.collect()` → terminal payoff를 각 seat의 마지막 표본에 기록 → `prepare_gae()` → `optimize_ppo()` → 별도 seed 기준선 평가 → `latest.pt`/`best.pt`·metrics 저장. `load_trained_actor()`는 `weights_only=True`로 읽고 `expand-e03-v1`/`expand-e03-snapshot-v1` format을 확인한 뒤 config로 모델을 만들고 weights를 적재한다.

**검증·한계.** E1~E3 초기 격리 기록은 `45 passed, 3 skipped`이며 실엔진·Windows CUDA acceptance는 pending이었다. `test_e03_returns_and_policy.py`, `test_e03_trainer.py`, `test_e03_cli.py`, `test_e03_real_engine.py`가 관련 계약을 다룬다. E3 Collector는 rake-free **cash game only**를 강제한다. 이것은 토너먼트 PPO 학습이 아니며, payout/ICM·탈락·blind schedule을 최적화한 모델도 아니다. Tournament ENV 1.4에서 이를 불러오는 것은 cash-to-knockout inference transfer다. Base HUNL200 Actor checkpoint와 E3 checkpoint는 교환할 수 없다.

### E4 — External-Sampling Linear MCCFR Blueprint

```math
\sigma_i^t(a\mid I)=
\begin{cases}
\dfrac{[R_i^{t-1}(I,a)]_+}{\sum_b[R_i^{t-1}(I,b)]_+},&\sum_b[R_i^{t-1}(I,b)]_+>0,\\
1/|\mathcal A(I)|,&\text{otherwise},
\end{cases}
```

```math
\widetilde r_i^t(I,a)=\widetilde v_i^t(I,a)-
\sum_b\sigma_i^t(b\mid I)\widetilde v_i^t(I,b),\quad
R_i^t=R_i^{t-1}+w_t\widetilde r_i^t,
\quad w_t=\left\lfloor\frac{t}{N}\right\rfloor+1.
```

```math
S_i^t(I,a)=S_i^{t-1}(I,a)+w_t\pi_i^t(I)\sigma_i^t(a\mid I),\qquad
\bar\sigma_i^t(a\mid I)=\frac{S_i^t(I,a)}{\sum_b S_i^t(I,b)}.
```

$`N`$은 참가자 수, $`t`$는 코드의 0-based completed-iteration, $`\pi_i^t(I)`$는 traverser 자신의 reach이다. Chance와 non-traverser 행동은 샘플링하고 traverser 행동은 분기한다. 따라서 상대/chance reach를 표본 선택으로 반영하면서 평균전략 누적에는 **own reach를 명시적으로 곱한다**. 여기의 linear weighting은 실제 소스처럼 regret와 strategy sum 양쪽에 적용된다.

**목적·효용.** 작은 게임에서 regret matching·own-reach 평균화·iteration 원자성·resume를 검증한 후 축소 Hold'em 트리에 연결한다. E3 PPO를 대체하는 학습 옵션이며, Pluribus의 대규모 학습량이나 성능을 재현했다는 뜻은 아니다.

| `.py` 경로 — `src/holdem_ai/expand/pluribus/` | 역할 |
|---|---|
| `blueprint/mccfr.py` | `InfosetStore`, `MCCFRSolver`, 예산 제한, external-sampling update |
| `blueprint/toy_games.py`, `blueprint/kuhn_reference.py` | Kuhn/Leduc와 2인 Kuhn exact best response/NashConv |
| `blueprint/holdem.py` | Base 엔진 상태 복사와 제한된 합법 action tree |
| `abstraction/cards.py` | preflop exact cards, postflop 임시 hash bucket, public history key |
| `blueprint/checkpoint.py`, `blueprint/evaluate.py` | checkpoint 저장/복원과 평가 |
| [scripts/expand_phase04_mccfr.py](technical_sources/PYTHON_SOURCE_02.md#py-017), [scripts/expand_phase04_evaluate.py](technical_sources/PYTHON_SOURCE_02.md#py-016) | CLI 학습·평가 |

실제 코드 — `blueprint/mccfr.py`, `MCCFRSolver.iterate()`의 traverser update:

```python
            avg = vector_for(average_deltas,key,len(actions))
            for a in range(len(actions)):
                avg[a] += weight * own_reach * policy[a]
            vals = [visit(state.next_state(action),own_reach*policy[j],depth+1)
                    for j,action in enumerate(actions)]
            mean = sum(p*v for p,v in zip(policy,vals))
            reg = vector_for(regret_deltas,key,len(actions))
            for j in range(len(actions)):
                reg[j] += weight*(vals[j]-mean)
```

**작동 순서.** `game.start(rng)`으로 deal → iteration 번호로 traverser/linear weight 선택 → `visit()`에서 actor-safe infoset 생성 → 상대는 한 행동 샘플, traverser는 모든 합법 행동 재귀 평가 → 지역 regret/average deltas 완성 → `store.update()` → completed counter 증가. 노드·깊이 budget 실패 시 traversal 중간의 delta를 store에 반영하지 않는다. Hold'em은 `next_state()`에서 `deepcopy(engine)` 후 합법 engine ID로 진행한다.

```math
\mathrm{NashConv}(\bar\sigma)=\sum_i
\left[\max_{\sigma_i'}u_i(\sigma_i',\bar\sigma_{-i})-u_i(\bar\sigma)\right].
```

| 역사적 2인 Kuhn iterations | Exact NashConv |
|---:|---:|
| 0 | 0.916667 |
| 3,000 | 0.050732 |
| 10,000 | 0.013073 |
| 30,000 | 0.015137 |

이는 **Kuhn의 수렴 진단**이며 6-Max Hold'em exploitability가 아니다. 샘플링 결과가 단조 감소하지도 않는다. 실제 postflop bucket은 `SHA256(cards) mod K` 임시 bucket이며 equity/EMD clustering이 아니다. 기본 action tree는 fold·check/call·pot·all-in에 선택적 half-pot이며 E2의 전체 7개 action과 같지 않다. Negative-regret pruning, 대규모 100BB Blueprint, 6인 균형 성질은 검증되지 않았다. 관련 테스트는 `test_e04_toy_mccfr.py`, `test_e04_holdem_mock.py`, `test_e04_checkpoint.py`, `test_e04_cli.py`, `test_e04_real_engine.py`다.

### E5 — Belief-Conditioned Bounded Search

```math
q(\omega\mid H_i,B,h_{\mathrm{pub}})\propto
\mathbf1[\omega\text{ has distinct cards and matches visible facts}]
\prod_{(j,a)\in h_{\mathrm{pub}},\,j\ne i}\ell_j(a\mid I_j(\omega)).
```

```math
\pi_{\mathrm{traverse}}=(1-\alpha)\pi_{\mathrm{local}}+\alpha\pi_{\mathrm{blueprint}},
\quad\alpha=0.15,\qquad
\widehat v_{\mathrm{leaf}}=u(\text{one continuation rollout}).
```

$`\omega`$는 관측과 일치하도록 **새로 샘플링한** 완전 세계이며 실제 상대 패를 읽어 만든 세계가 아니다. 상대 likelihood 함수가 제공되지 않으면 공개 사실·카드 제약에 조건화한 샘플링을 사용하며 action-conditioned posterior를 자동으로 학습하지 않는다. 제공되면 공개 행동 재생 중 likelihood product로 rejection sampling한다.

**목적·효용.** Blueprint보다 세밀한 현재 의사결정 계산을 제한된 CPU 예산으로 실행한다. 탐색 부족·timeout과 정상 solve를 구분해서 상위 정책이 처리할 수 있게 한다. 기본 예산은 500ms, 2,500 search nodes, 80 iterations, 1,000 infosets, depth 2, 64MB **테이블 크기 추정량**, rollout 최대 200 step이다. 64MB는 실제 프로세스 RSS의 엄격한 상한이 아니다.

| `.py` 경로 — `src/holdem_ai/expand/pluribus/search/` | 역할 |
|---|---|
| `belief.py` | hero/public facts에 맞춘 toy/실제 덱 세계 생성과 public replay |
| `continuations.py` | blueprint·fold/call/raise bias continuation 4종 |
| `solver.py` | current-root depth-limited MCCFR, budget, `SearchResult` |
| `nested.py` | public trace, 이미 실행한 hero commitment, off-tree 카운트 |
| [scripts/expand_phase05_search.py](technical_sources/PYTHON_SOURCE_02.md#py-019), [scripts/expand_phase05_evaluate.py](technical_sources/PYTHON_SOURCE_02.md#py-018) | 단일 탐색과 A/B toy 평가 |

실제 코드 — `solver.py`, `SubgameSolver.search()`의 belief root 검증:

```python
                state=belief.sample(rng)
                if state.current_player!=belief.hero or state.is_terminal():
                    raise ValueError('belief sampled an invalid hero root')
                if tuple(state.legal_actions())!=root_actions or state.infoset(belief.hero)!=root_key:
                    raise ValueError('belief root information mismatch; no safe solve possible')
```

**작동 순서.** `HoldemObservationBelief(raw_obs, initial_stacks, blinds)`는 알려진 hero 카드와 공개 board를 엔진의 deal/burn 인덱스에 배치하고 나머지 덱을 충돌 없이 섞는다. `sample()`이 초기 상태부터 공개 action ID/raise-to를 재생하고 현재 actor·스택·팟·street·카드·이력 일치를 검사한다. `search()`는 같은 hero infoset의 세계를 반복 샘플 → 깊이까지 local regret search → leaf에서 continuation rollout → iteration delta commit → root average policy와 status를 반환한다. continuation variant는 `(player, infoset)`마다 뽑아 같은 infoset에서 hidden world별로 다른 oracle 선택을 하지 않는다.

**검증·한계.** 역사적 Kuhn 40-iteration 기록은 `solved`, 180 nodes, 12.78ms, `P(check)=0.609318`, `P(bet)=0.390682`이다. 별도 16게임 toy A/B는 Search−Blueprint 평균 `−0.125` chip/hand, 95% 정규근사 CI `[-0.2924,+0.0424]`로 개선을 입증하지 못했다. 지연시간은 당시 실행환경의 값이다. `test_e05_beliefs.py`, `test_e05_holdem_deck.py`, `test_e05_continuations.py`, `test_e05_solver.py`, `test_e05_nested.py`, `test_e05_evaluate.py`, `test_e05_cli.py`, `test_e05_real_engine.py`가 각 경계를 다룬다.

`NestedSearchSession`은 **현재 의사결정 재탐색용 사실 이력**을 유지한다. betting round 시작부터 전략적으로 안전하게 다시 푸는 원본 Pluribus nested re-solving, joint leaf-continuation subgame, exploitability guarantee는 구현하지 않았다. E5 자체는 일부 완료 iteration의 결과를 timeout status와 함께 반환할 수 있으나, **E7 연결은 `status='solved'`만 허용**한다.

### E6 — Public Opponent Model and Bounded Adaptation

```math
C_{t,c,a}^{L}=\lambda_L C_{t-1,c,a}^{L}+\mathbf1[a_t=a],\quad
C_{t,c,a}^{R}=\lambda_R C_{t-1,c,a}^{R}+\mathbf1[a_t=a],\quad
\widehat p_{c}(a)=\frac{C_{c,a}+\alpha}{\sum_bC_{c,b}+\alpha|\mathcal A_c|}.
```

```math
\pi_\eta=(1-\eta)\pi_{\mathrm{base}}+\eta\pi_{\mathrm{proposal}},\quad
0\le\eta\le\eta_{\max}\frac{n_{\mathrm{eff}}}{n_{\mathrm{eff}}+\kappa},\quad
D_{KL}(\pi_\eta\Vert\pi_{\mathrm{base}})\le0.025,\quad
D_{TV}(\pi_\eta,\pi_{\mathrm{base}})\le0.075.
```

$`c`$는 street·position·SPR bucket·상대의 합법 action tuple로 구성한 context다. 기본 decay는 $`\lambda_L=0.995,\lambda_R=0.88`$, Dirichlet prior는 $`\alpha=0.5`$, 최대 혼합은 $`\eta_{\max}=0.3`$, confidence pseudocount는 $`\kappa=20`$이다. 모델 identity는 `(player_id, player_session_id)`이며 물리 좌석 하나에 이전 참가자의 성향을 계속 붙이지 않는다.

**목적·효용.** 수락된 공개 행동으로 분포와 스타일 변화를 추정한다. 별도로 제공된 hero expert의 정책을 증거에 따라 제한적으로 섞는다. E6 자체가 profitable counter-strategy를 학습하지는 않는다. 상대의 합법 action set과 현재 hero의 합법 action set은 다른 관측이므로 일치 검사를 서로 혼동하지 않는다.

| `.py` 경로 — `src/holdem_ai/expand/opponent/` | 역할 |
|---|---|
| `schema.py`, `model.py` | public event/context, 이중 decay·Dirichlet posterior·shift detection·identity registry |
| `bridge.py`, `persistence.py` | E1/E2 accepted event 수집, registry 저장·복원 |
| `router.py`, `adaptive.py` | style expert 선택, evidence gate, KL/TV 혼합 제한 |
| [scripts/expand_phase06_benchmark.py](technical_sources/PYTHON_SOURCE_02.md#py-020) | 홀덤과 분리한 합성 forecasting 실험 |

실제 코드 — `adaptive.py`, `SafeAdaptivePolicy.adjust()`의 trust-region 검색:

```python
        def blend(w):return {a:(1-w)*p[a]+w*q[a] for a in legal}
        def allowed(w):
            cand=blend(w)
            return kl_divergence(cand,p)<=self.guard.kl_budget+1e-14 and tv_distance(cand,p)<=self.guard.tv_budget+1e-14
        if not allowed(max_w):
            lo,hi=0.,max_w
            for _ in range(60):
                mid=(lo+hi)/2
                if allowed(mid):lo=mid
                else:hi=mid
            max_w=lo
```

**작동 순서.** boundary 후 `sync_seats()`로 player tenure를 바인딩 → `before=adapter.observe()` → 실제 table 행동 수락 → `record_accepted_action(before,...)` → `OpponentRegistry.observe()`로 evidence 갱신 → hero decision에서 `StyleExpertRouter.propose()` → `SafeAdaptivePolicy.adjust()`로 지원 action·표본·변화점·KL/TV 검사 → policy 반환. rejected action은 evidence에 쓰지 않는다. 기준정책 확률이 0인 행동에는 새 mass를 넣지 않는다.

변화 검사의 실제 구현은 recent와 long의 서로 다른 누적량이 거짓 shift를 만들지 않도록 $`\alpha_R=\alpha_L N_R/N_L`$로 prior/data 비율을 맞춘다. 충분한 표본에서 KL threshold를 연속 초과하면 `shift_hold`를 설정하고 long counts를 recent regime으로 재시작한다. 이후 evidence가 다시 쌓일 때까지 기준정책을 사용한다.

| 역사적 합성 16-block 평가 | 값 |
|---|---:|
| Uniform NLL / online model NLL | 1.098612 / 0.766122 |
| NLL improvement와 95% CI | +0.332491 / [+0.310901,+0.354081] |
| Shift 감지 | 16/16 blocks |
| 관측 최대 policy KL / TV | 0.012583 / 0.071326 |

이는 Tight→Aggressive 합성 공개 행동 예측 결과다. 실제 poker EV·bb/100·exploitability 측정이 아니다. 자유로운 off-tree raise-to를 E2의 7-action evidence로 변환하는 별도 계약도 필요하다. 관련 테스트는 `tests/expand/`의 `test_e06_models.py`, `test_e06_router.py`, `test_e06_adaptive.py`, `test_e06_bridge.py`, `test_e06_persistence.py`, `test_e06_state_bridge.py`, `test_e06_cli.py`, `test_e06_real_engine.py`다.

### E7 — Integration, Single-Table Tournament, Scientific Benchmark

```math
\pi_{\mathrm{selected}}=
\mathrm{FirstValid}(\pi_{\mathrm{search}},\pi_{\mathrm{blueprint}},\pi_{\mathrm{neural}}),
\qquad
a\sim\pi_{\mathrm{selected}},\quad A_{\mathrm{table}}=\mathrm{Translate}(a).
```

```math
\ell(h)=\min\!\left(\left\lfloor h/H_{\mathrm{level}}\right\rfloor,L-1\right),\quad
\sum_{i\in\mathrm{alive}}S_i=C_{\mathrm{registered}},\quad
\widehat\mu\pm1.96\frac{s_{\mathrm{block}}}{\sqrt{K}}.
```

$`h`$는 이미 완료한 핸드 수, $`H_{\mathrm{level}}`$은 blind level당 핸드 수다. $`C_{\mathrm{registered}}`$는 입장·허용된 재입장으로 투입한 총 칩이다. $`K`$는 독립 table/tournament block 수이며 한 block 안의 모든 action을 독립 표본으로 세지 않는다.

**목적·효용.** 정책 호출 계약·호환성·실패 원인을 기록하면서 플레이와 평가를 연결한다. E7의 `SingleTableTournament`는 E1 cash table과 별개로 entry/재입장·스택 이월·blind schedule·탈락·동시탈락 순위·payout을 관리한다. chips payoff와 최종 prize/rank는 서로 다른 목표이고, tournament runner를 만들었다고 tournament learner가 만들어진 것은 아니다.

| `.py` 경로 — `src/holdem_ai/expand/integration/` | 역할 |
|---|---|
| `agent.py` | `IntegratedPokerAgent`, source routing, lawful normalization, `RoutedDecision` |
| `source_bridges.py` | exact infoset E4·verified belief E5 opt-in adapter |
| `tournament.py` | boundary-only 등록·재입장, blind levels, survivor BTN, 순위/prize ledger |
| `benchmark.py` | frozen cash/tournament strata·paired comparison·block CI |
| `manifest.py` | config/seed/runtime/checkpoint SHA-256·평가 artifact 추적 |
| [scripts/expand_phase07_benchmark.py](technical_sources/PYTHON_SOURCE_02.md#py-021) | 재현 가능한 평가 CLI |

실제 코드 — `agent.py`, 선택 후보의 순서:

```python
        candidates = (
            ('search', self.search, self.config.use_search, self.config.verified_search),
            ('blueprint', self.blueprint, self.config.use_blueprint, self.config.verified_blueprint),
            ('neural', self.neural, True, True),
        )
```

실제 코드 — `source_bridges.py`, E5의 incomplete solve 처리:

```python
    def distribution(state:InformationState):
        result=solver.search(verified_root_belief(state))
        if result.status!='solved':
            raise TimeoutError(f'search incomplete: {result.status}')
        legal=tuple(a for a,yes in zip(ACTION_NAMES,state.legal_actions.mask) if yes)
        if set(result.action_probabilities)!=set(legal):
            raise ValueError('search root action abstraction incompatible')
        return dict(result.action_probabilities)
```

**실제 폴백 계약.** E4는 caller가 제공한 `exact_infoset_key`와 저장 전략의 정확한 action set을 검사한다. unseen infoset은 학습된 균등전략으로 가장하지 않고 오류를 낸다. E5는 verified belief가 필요하고 `solved` 이외 status·action mismatch는 다음 source로 넘긴다. router는 미검증·부재·예산 소진·잘못된 분포·예외의 이유를 모으면서 search→blueprint→neural을 시도한다. 모든 source가 실패하면 `uniform_legal_emergency`를 명시한다. optional budget을 소진해도 neural fallback 호출 기회는 남긴다. E6는 source 선택 후 외부 expert proposal을 제한적으로 섞는다.

**입력·출력.** `act(InformationState,rng_seed,evidence)` → strict legal-set normalization → 선택/적응 → seeded sampling → `to_table_action()` → action 이름·실제 `Action`·확률·선택 source·fallback reasons를 포함한 `RoutedDecision`이다. 캐시 benchmark는 인원/stack/style/hero seat/deck seed를 맞춘 independent table당 한 scored hand를 평가하며 admission warmup은 따로 센다. Tournament benchmark는 완결된 tournament별 우승·순위·prize를 집계하고 max-hand 중단에 임의 champion을 부여하지 않는다.

**검증·한계.** 역사적 E7 격리 테스트는 23 passed, E1~E7 합본은 `164 passed, 7 skipped`다. `test_e07_agent.py`, `test_e07_bridges_and_cli.py`, `test_e07_cash_bridge.py`, `test_e07_benchmark.py`, `test_e07_manifest.py`, `test_e07_tournament.py`, `test_e07_real_engine.py` 및 fake tournament fixture가 포함된다. exact E2→E4 key equality, 실제 Windows engine/CUDA, 실게임 opponent adaptation EV는 pending이다. Ante, multi-table director, ICM solver, 원본 Pluribus complete nested search는 구현 범위 밖이다.

<a name="section-299f12b6cbd0"></a>

## HUNL GUI 프로젝트 — G1.0~G1.6

```math
\boxed{\mathrm{Browser}\xrightarrow{(a,r)}\mathrm{FastAPI}
\xrightarrow{\mathrm{Session}}\mathrm{InteractiveHUNLAdapter}
\xrightarrow{}\mathrm{MultiPlayerHandEngine}}
```

```math
I_i=O_i(S),\quad a_i\sim\pi_\theta(\cdot\mid I_i),\quad
\mathrm{GameView}=\mathrm{Project}_{\mathrm{human}}(S),\quad
\mathrm{Accept}(a,r)\iff r=r_{\mathrm{server}}\land a\text{ is legal at the current human turn}.
```

GUI는 학습 알고리즘이 아니라 **사람과 frozen Agent가 같은 엔진에서 플레이하고 결과를 확인하는 인터페이스**다. 정책 입력 $`I_i`$와 사람에게 보내는 GameView는 서로 다른 projection이다. 브라우저에는 deck·seed·privileged critic·미공개 상대 패를 전달하지 않는다. Python backend의 현재 소스와 과거 phase 보고서를 함께 읽되, G1.6에서 바뀐 continuous default를 G1.0~1.5의 과거 독립 핸드 설명과 구분한다. 아래 frontend 파일/동작은 원문 보고서의 확정된 계약을 설명하며, 이 절의 실제 코드 발췌는 검토한 Python 파일에서 가져왔다.

| Phase | 목적 | 효용 |
|---|---|---|
| G1.0 | transport-independent session과 human-safe 관측 계약 | rules·policy·projection을 분리 |
| G1.1 | 로컬 FastAPI 요청/응답 | schema·revision·HTTP 실패 처리를 고정 |
| G1.2 | 읽기 전용 React 테이블 | server state를 시각적으로 검토 |
| G1.3 | 실제 Human 베팅과 자유 raise-to | 7-head Agent와 정수 Human 베팅 공존 |
| G1.4 | 로컬 trained checkpoint inference | 모델 선택·호환성 검사·명시적 load 실패 |
| G1.5 | 완료 핸드 기록·성과 dashboard | 누적 손익·분산·이력 추적 |
| G1.6 | continuous stack play | 이전 정산 스택 이월·파산 종료·무의사결정 정산 |

### G1.0 — Human/Agent Session Contract

```math
V_{t+1}=\mathrm{Project}_{0}(T(S_t,a_t)),\qquad
\mathrm{opponent\_cards}=
\begin{cases}H_1,&\mathrm{terminal}\land\mathrm{genuine\ showdown},\\
\mathrm{null},&\text{otherwise}.
\end{cases}
```

Human은 seat 0, Bot은 seat 1이다. genuine showdown은 현재 구현에서 `not any(result.folded)`와 `len(result.board)==5`로 판정한다. terminal fold로 끝나도 상대 패는 숨긴다.

**목적·제작 파일.** `game_service/session_manager.py`의 `HUNLGameSession`/`HUNLSessionRegistry`는 multi-hand session·독립 RNG·`RLock`·revision을 담당한다. `action_adapter.py`는 7-head Human action을 검증하고 실제 금액의 `ActionOffer`를 만든다. `bot_adapter.py`는 정책에 해당 Bot의 `InformationState`만 전달한다. `state_projection.py`는 명시한 필드만 JSON으로 만든다. 원문 전체 경로의 공통 접두사는 `src/holdem_ai/`다.

실제 코드 — 현재 `state_projection.py`의 결과 카드 공개 필드:

```python
            "showdown": is_showdown,
            # Revealing the opponent is allowed only after a genuine showdown.
            "opponent_cards": list(engine.hole_cards[opponent]) if is_showdown else None,
```

**작동 순서.** `start_hand()`가 엔진을 시작하고 Bot을 사람 턴 또는 terminal까지 자동 진행한다. `play_human_action(index,expected_revision)`은 revision→active hand→human turn→legal index를 확인한 후 `_take_action()`과 `_run_bots()`를 호출해 최신 view를 반환한다. `view()`는 핸드를 진행시키지 않는다. indices 0~6은 `[fold,check,call,half-pot,pot,double-pot,all-in]` **policy head**이고 Base 엔진의 5-ID 계약과 다르다.

**검증·한계.** `test_gui10_hunl_contract.py`, [scripts/gui10_smoke.py](technical_sources/PYTHON_SOURCE_02.md#py-022)가 privacy·legal action·revision·zero-sum을 다룬다. G1.0의 역사적 새 핸드는 매번 200BB로 초기화했고, G1.6 이후 Python session의 `independent` 기본값으로 남아 있다. session registry는 메모리 전용이다.

### G1.1 — FastAPI Local Server

```math
\mathrm{Response}=\{\mathrm{session\_id},\mathrm{state}:V\},\qquad
\mathrm{stale\ revision}\mapsto409,\quad
\mathrm{invalid\ action/schema}\mapsto422,\quad
\mathrm{unknown\ session}\mapsto404.
```

**목적·파일.** [backend/main.py](technical_sources/PYTHON_SOURCE_01.md#py-003)가 app·CORS·registry를 만들고 loopback으로 실행한다. [backend/schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-004)는 strict request/response DTO, [backend/game_router.py](technical_sources/PYTHON_SOURCE_01.md#py-002)는 `/api/v1/hunl` 라우팅을 담당한다. `session_manager.py`의 `start_next_hand(expected_revision)`는 중복 Next Hand 클릭을 원자적으로 거부한다. 라우터에서 포커 규칙을 다시 구현하지 않는다.

실제 코드 — [backend/game_router.py](technical_sources/PYTHON_SOURCE_01.md#py-002), action HTTP handler:

```python
        state = run_game_operation(
            lambda: session.play_human_action(
                request.action_index, expected_revision=request.expected_revision
            )
        )
```

| HTTP operation — `/api/v1/hunl` 아래 | 입력·출력 |
|---|---|
| `POST /sessions` | bot/seed/blinds/stack mode → 새 ID와 GameView |
| `GET /sessions/{id}` | 최신 view, 게임 진행 없음 |
| `POST /sessions/{id}/actions` | action index+revision → 처리 후 view |
| `POST /sessions/{id}/next-hand` | revision → 다음 핸드 또는 409 |
| `DELETE /sessions/{id}` | 세션 제거 |
| `GET /health` | 서비스 상태 |

**작동·검증.** Pydantic 입력 확인 → registry lookup → session lock/revision 검사 → 엔진 진행 → response projection/schema 직렬화다. [tests/test_gui11_http_api.py](technical_sources/PYTHON_SOURCE_22.md#py-258), [scripts/gui11_smoke.py](technical_sources/PYTHON_SOURCE_02.md#py-023)가 HTTP 성공·404/409/422·projection·CORS를 다룬다. process-local registry와 인증 없는 localhost 연구 서버이며 다중 worker·원격 배포·영속화를 구현한 phase가 아니다.

### G1.2 — Read-Only React Poker Table

```math
\mathrm{UI}=\mathrm{Render}(V_{\mathrm{server}}),\qquad
T_{\mathrm{backend}}(\mathrm{PreviewSelect})=\mathrm{identity}.
```

**목적·효용.** 사람 카드·공개 board·상대 카드 뒷면·BTN/SB/BB·스택·street bets·pot·행동 이력을 동일한 server projection으로 보여준다. preview fixture는 디자인 확인용 샘플이며 게임 진행이나 실적에 포함되지 않는다.

**제작 파일·실제 Python 경계.** React 테이블·카드 render helper·API client를 `frontend/`에 추가했고 [tests/test_gui12_readonly_api.py](technical_sources/PYTHON_SOURCE_22.md#py-259)가 backend JSON/CORS/privacy 계약을 검증한다. Python 서비스는 G1.0/1.1을 재사용한다. 현재 `state_projection.py`의 사람 관측 검사는 다음과 같다.

```python
        if obs.player_id != hero_seat:
            raise AssertionError("human GUI cannot receive bot observations")
```

**작동 순서.** session 생성/GET → `GameResponse.state` 수신 → 공개 필드 render → terminal 여부에 따른 Next Hand 표시다. 상대 카드는 terminal showdown일 때만 표시한다. 이 phase는 베팅 클릭을 추가하지 않았으므로 live hand는 사람의 첫 결정에서 멈추며 G1.3이 행동 전송을 추가한다.

**검증·한계.** 원문은 Node 의존성 접근 실패 때문에 `npm install/test/build`와 실제 브라우저 검증을 보류했다고 명시한다. UI 구현 설명을 production build 통과로 해석하지 않는다.

### G1.3 — Interactive Betting and Exact Human Raise-to

```math
c_i=\max(0,b_{\max}-b_i),\quad
T_{\min}=b_{\max}+\Delta_{\mathrm{last\ full\ raise}},\quad
T_{\max}=b_i+s_i,\quad
\Delta\mathrm{chips\ paid}=\mathrm{raise\_to}-b_i.
```

**목적·효용.** 사람은 서버가 제시한 7-action offer와 별도로 임의 **정수 칩** raise-to를 사용할 수 있다. Bot은 계속 기존 7-head 계약을 따른다. 최소 full raise 미달이라도 정확히 all-in maximum인 short raise는 엔진이 규칙과 재오픈 여부를 검사해 허용할 수 있으므로 frontend의 범위 검사만으로 최종 판정하지 않는다.

**제작·수정 파일.** `game_service/interactive_adapter.py`는 HUNL adapter의 Human 전용 subclass, `session_manager.py`는 `play_human_raise_to()`, `state_projection.py`는 `can_raise_to`/bounds, [backend/game_router.py](technical_sources/PYTHON_SOURCE_01.md#py-002)·`schemas.py`는 `/raise-to` request를 추가한다. React의 `BettingPanel.tsx`, `lib/betting.ts`는 presets·slider·integer 입력을 제공한다.

실제 코드 — `InteractiveHUNLAdapter.step_human_raise_to()`:

```python
        engine.step_raise_to(amount)
        self.last_engine_action = EngineBetAction("raise_to", amount)
        self._history.append(f"{street}|seat={actor}|raise_to|raise_to={amount}")
        return self._receive()
```

**입력·실행.** `POST .../raise-to {raise_to:350,expected_revision:17}` → integer/schema → session lock/revision/turn 검사 → engine legality 검사 → exact amount history 기록 → Bot 자동 행동 → 최신 GameView다. 엔진 거부 시 session revision/history는 갱신하지 않는다. UI는 409에 최신 상태를 다시 받아야 한다.

**검증·한계.** 역사적 호환 샌드박스 기록은 `130 passed,3 skipped`, 신규 API/custom raise 테스트 5개, 14개 TypeScript 파일 transpilation 문법 검사다. 이 수치는 전체 최신 repository regression·npm production build·browser E2E를 의미하지 않는다. 자유 raise-to는 학습 때의 action abstraction 밖 입력을 만들 수 있어 policy quality가 미검증이다. [tests/test_gui13_interactive_betting.py](technical_sources/PYTHON_SOURCE_22.md#py-260)가 관련 Python 계약을 다룬다.

### G1.4 — Frozen Trained-Agent Inference

```math
\theta_{\mathrm{session}}=\theta_{\mathrm{checkpoint}},\quad\Delta\theta=0,\qquad
a\sim\pi_\theta(\cdot\mid I_{\mathrm{bot}})\;(\mathrm{sampled}),\quad
a=\arg\max_a\pi_\theta(a\mid I_{\mathrm{bot}})\;(\mathrm{greedy}).
```

**목적·파일.** `game_service/checkpoint_catalog.py`가 server 환경변수의 PPO/QBoost checkpoint 슬롯·file 존재 검사·cache를 관리한다. `evaluation/checkpoint_loader.py`의 `load_actor()`와 `agents/neural_policy.py`의 `NeuralActorPolicy`를 재사용한다. [backend/game_router.py](technical_sources/PYTHON_SOURCE_01.md#py-002)/`schemas.py`는 `/bots`와 trained session 구성을 추가한다. 입력 path를 브라우저가 지정하지 않고 bot ID만 선택한다.

실제 코드 — `CheckpointCatalog.build_policy()`:

```python
                actor = load_actor(path, device=self.device)
                if actor.config.dropout != 0:
                    raise ValueError("evaluation model must have dropout=0")
                policy = NeuralActorPolicy(actor, device=self.device,
                                           big_blind=100, greedy=self.mode == "greedy")
```

**작동 순서.** `/bots`가 configured file 존재를 알림 → session 생성에서 strict load → eval-only policy cache → Bot의 private InformationState 입력 → masked 7-head inference → session별 RNG로 샘플링이다. file path/mtime/size 변화는 cache invalidation 기준이다. 파일이 없거나 format/weights/device가 틀리면 명시적 422를 반환하며 random bot으로 바꾸지 않는다. HUNL trained 시작 조건은 SB/BB=50/100·200BB다.

**검증·한계.** `test_gui14_trained_agents.py`, [scripts/gui14_smoke.py](technical_sources/PYTHON_SOURCE_02.md#py-024)와 원문은 backend 34 tests, 호환 전체 `137 passed,3 skipped`를 기록한다. checkpoint loader를 mock한 fixture가 포함되어 실제 Phase15 weights·Windows CUDA·React E2E 완료의 증거는 아니다. `ppo`/`qboost` ID는 server 구성 label이며 training algorithm을 GUI가 독립 인증하지 않는다. E3 multi-seat weights는 여기의 HUNL weights와 다른 family다.

### G1.5 — Completed-Hand Journal and Performance Dashboard

```math
R_h=\frac{S_h^{\mathrm{final}}-S_h^{\mathrm{start}}}{BB_h},\quad
C_n=\sum_{h=1}^n R_h,\quad\widehat{bb/100}=100C_n/n,
\quad\widehat p_{\mathrm{win}}=N_{\mathrm{win}}/n.
```

```math
\delta_n=R_n-\bar R_{n-1},\quad
\bar R_n=\bar R_{n-1}+\delta_n/n,\quad
M_{2,n}=M_{2,n-1}+\delta_n(R_n-\bar R_n),\quad
s_n=\sqrt{M_{2,n}/(n-1)}.
```

```math
\mathrm{CI}_{95}\approx100\bar R_n\pm1.96\frac{100s_n}{\sqrt n},\qquad
\mathrm{MaxDrawdown}=\max_n\bigl(\max_{k\le n}C_k-C_n\bigr).
```

**목적·효용.** 완료 핸드마다 딱 한 번 human-safe 기록을 저장해 이력과 통계가 중복 계산되지 않게 한다. `game_service/session_analytics.py`는 Welford 분산·누적 칩/BB·W/L/even·showdown·최대 낙폭을 관리한다. 상세 이력은 최근 500개로 제한하지만 전체 누적 통계는 유지한다. `session_manager.py`의 terminal callback, [backend/schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-004)/`game_router.py`의 `/stats`·`/history`, React `SessionAnalyticsPanel.tsx`/`lib/analytics.ts`가 연결된다.

실제 코드 — `SessionAnalytics.append_completed()`:

```python
        difference = bb - self.mean_bb
        self.mean_bb += difference / self.count
        self.m2_bb += difference * (bb - self.mean_bb)
        self.peak_bb = max(self.peak_bb, self.total_bb)
        self.max_drawdown_bb = max(self.max_drawdown_bb, self.peak_bb - self.total_bb)
```

**작동 순서.** engine terminal → `_finish_terminal()` → `project_human_view()` → `append_completed()`의 terminal/privacy/단위/연속 hand number 검사 → 누적 통계/안전 snapshot 저장 → GET stats/history → dashboard render다. history는 newest-first이고 page limit은 1~100이다. 진행 중인 핸드나 preview fixture는 완료 실적에 넣지 않는다.

**검증·한계.** [tests/test_gui15_session_analytics.py](technical_sources/PYTHON_SOURCE_22.md#py-262), [scripts/gui15_smoke.py](technical_sources/PYTHON_SOURCE_03.md#py-025)가 terminal-only·중복/번호·privacy·retention·pagination·session isolation을 다룬다. CI는 독립 hand/정규근사를 가정한 설명용 수치다. 사람의 적응과 시드·자리 불균형이 있을 수 있어 실력 인증에는 seed-blocked independent evaluation이 필요하다. 메모리 기록은 서버 재시작으로 사라지며 과거 Call/All-in history의 exact amount 일부가 없는 제약도 남는다.

### G1.6 — Continuous Stack Play

```math
S_{i,h+1}^{\mathrm{start}}=S_{i,h}^{\mathrm{final}},\quad
R_{i,h}^{\mathrm{chips}}=S_{i,h}^{\mathrm{final}}-S_{i,h}^{\mathrm{start}},\quad
\sum_{i=0}^1S_{i,h}=2\cdot200\cdot100=40{,}000,
\quad BTN_h=(h-1)\bmod2.
```

```math
\mathrm{match\_over}=\mathbf1[\min_iS_{i,h}^{\mathrm{final}}=0].
```

**목적·효용.** 브라우저 Play가 이전 핸드의 정산 스택을 다음 핸드에 넘긴다. GUI HTTP 요청의 `stack_mode` 기본값은 `continuous`, Python `HUNLGameSession()` 기본값과 Base training/evaluation은 `independent`다. 첫 핸드의 양측 200BB와 고정 blinds 50/100은 유지하되 이후 스택은 비대칭일 수 있다. rake·rebuy·top-up은 없다.

**제작·수정 파일.** `InteractiveHUNLAdapter.reset_with_stacks()`가 기존 Base 엔진의 stacks 인수로 새 핸드를 만든다. `session_manager.py`는 carryover·bust·무의사결정 terminal을 처리한다. `state_projection.py`, [backend/schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-004)/`game_router.py`는 `starting_stacks`, `stack_mode`, `match_over`, `winner_seat`를 공개한다. terminal seats에는 award 전 잔액 대신 최종 정산 스택을 표시한다.

실제 코드 — `HUNLGameSession.start_hand()`:

```python
            if self.stack_mode == "continuous" and self._adapter is not None:
                final = self._adapter.engine.get_result().final_stacks
                initial_stacks = (int(final[0]), int(final[1]))
                if min(initial_stacks) == 0:
                    raise RuntimeError("match over: a player is bankrupt; start a new session")
                if sum(initial_stacks) != 2 * self.stack_bb * self.big_blind:
                    raise AssertionError("chips changed across consecutive hands")
```

**작동 순서.** 첫 핸드는 기존 reset → terminal record → Next Hand revision 검사 → preceding final stacks 취득 → 파산/보존 검사 → `reset_with_stacks()` → BTN 교대 → Bot 자동 진행 또는 즉시 terminal이다. blind를 못 채우는 short stack이 forced all-in으로 시작해 의사결정 없이 정산돼도 완료 핸드·손익·history를 기록한다. match-over 이후 Next Hand는 409이며 새 session을 만들어야 한다. `game_type='HUNL200'`은 API/policy family 식별자이고 모든 이후 핸드가 200BB라는 뜻은 아니다.

**검증·한계.** [tests/test_gui16_continuous_stack.py](technical_sources/PYTHON_SOURCE_22.md#py-263), [scripts/gui16_smoke.py](technical_sources/PYTHON_SOURCE_03.md#py-026)는 carryover·BTN·칩 보존·independent 호환·all-in bust·revision·short-blind terminal을 검증한다. 원문 역사적 smoke는 20연속 hand와 always-fold 267-hand 파산 종료 `[0,40000]`를 기록하지만 일부 Base import에 stand-in을 쓴 격리 검증이었다. 실제 full repository Windows/browser build 통과와 별개다. 200BB 독립 핸드로 학습한 Actor의 비대칭/짧은/깊은 스택 성능은 보장되지 않으며 variable-stack curriculum은 별도 연구 과제다.


<a name="section-c5edcf32d2ad"></a>

## 원문 통합의 검증과 기존 보고서 정리

```math
\mathrm{GitBlobSHA}(b)=\mathrm{SHA1}
\left(\mathtt{blob\ }\Vert\mathrm{decimal}(|b|)\Vert\mathtt{NUL}\Vert b\right)
```

$`b`$는 원본 Git Blob의 실제 바이트이고 $`|b|`$는 바이트 수다. 문서 표시용 Fence의 구분용 LF 1바이트는 원문에 포함되지 않는다. 파일 경로, 원본 길이, Git Blob SHA를 함께 비교하고, 모든 경로가 정확히 한 번 등장하는지 확인했다.

```math
\mathrm{paths}(A_{PY})=\mathrm{paths}(S_{PY}),\quad
\mathrm{paths}(A_{MD})=\mathrm{paths}(S_{MD})
```

```math
\forall p\in S_{PY}\cup S_{MD}:\quad
\mathrm{count}_A(p)=1,\quad |A[p]|=|S[p]|,\quad
\mathrm{GitBlobSHA}(A[p])=\mathrm{GitBlobSHA}(S[p]).
```

$`S`$는 원본 스냅샷, $`A`$는 최종보고서의 원문 부록이다. 경로집합·개수·원문 크기·SHA가 일치해야 무손실로 판정한다. 스크린샷의 저장 진척이나 파일명만으로 완료를 판단하지 않았다. 재개 시 Python 부록은 300개로 확인됐고 빠진 테스트 27개를 보완했다.

| 이번에 수행한 검증 | 결과 |
|---|---:|
| Python 전체 원문 | **327/327** · 1,400,837바이트 |
| Markdown 전체 원문 | **76/76** · 1,049,564바이트 |
| 원본 경로 누락·중복·추가·SHA 불일치 | **0** |
| Python 전체 원문의 문법 검사 | **327개 통과**; 게임·학습의 런타임 테스트와 구분 |
| Python 부록 | 28 논리 권, 29 파일; 25권은 A/B 분할 |
| Markdown 부록 | 7 논리 권, 8 파일; 6권은 A/B 분할 |
| 원문 검증기의 오류 검출 검증 | 원문 변조·중복·누락·헤더 SHA·마커 SHA·Anchor·추가 원문·추가 부록 등 8종 검출 |
| 독립 원본 보고서 정리 | **73개 삭제**; 본문·README·TECH REPORT 및 구성 부록 유지 |

삭제 집합은 다음처럼 정의한다.

```math
D=S_{MD}\setminus\{\mathtt{README.md},\ \mathtt{docs/TECH\_REPORT.md},\ 
\mathtt{docs/POKER\_AI\_FINAL\_INTEGRATED\_TECHNICAL\_REPORT.md}\}.
```

두 기준 보고서, 개별 Phase 보고서, 루트의 과거 패치 README, 기존 짧은 코드 레퍼런스, 앞선 부분 원문 통합 파일도 이 73개에 포함된다. 그 내용은 Markdown 부록에 정확히 보존했다. Python·TypeScript·테스트·학습 코드·설정은 삭제·변경 대상에 포함하지 않았다.

Manifest의 SHA는 원문 기준 커밋에서 가져온 Git Blob SHA다. 최종 본문·README·TECH REPORT의 새로운 SHA와 같은 값이라고 주장하지 않는다. 이 3개 파일의 당시 원문도 부록에 포함되어 있어 재작성 전후를 비교할 수 있다.

### 무손실 검증을 재현하는 Python 코드

아래 코드를 `verify-lossless.py`로 저장하고 `python verify-lossless.py <repository-root>`로 실행한다. Python 표준 라이브러리만 사용하며, 파일을 수정하지 않는다. 원문 SHA·바이트 수·마커·Fence·Anchor와 경로 집합을 검사하고 오류가 있으면 종료 코드 1을 반환한다.

```python
#!/usr/bin/env python3
"""Read-only verification of fenced Git blobs against a preservation manifest.

Usage: python verify-lossless.py /path/to/repository
Only Python's standard library is required. No files are created or modified.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


TOKEN = re.compile(rb'<!-- (BEGIN|END)_(PY|MD)_SOURCE\b[^>]*-->')
BEGIN = re.compile(rb'<!-- BEGIN_(PY|MD)_SOURCE path="([^"]+)" sha="([^"]+)" -->')
END = re.compile(rb'<!-- END_(PY|MD)_SOURCE path="([^"]+)" -->')
OPEN = re.compile(rb'^`{8}(python|markdown)\r?\n', re.M)
CLOSE = re.compile(rb'^`{8}\r?$', re.M)
HEADER = re.compile(rb'^(?=[^\r\n]*bytes)(?:\*\*Blob SHA:\*\*|Original (?:Git blob(?: SHA)?|[Bb]lob(?: SHA)?|SHA)|[Bb]lob|Git Blob)[^\r\n]*$', re.M)
SIZE = re.compile(rb'(?:\*\*Original bytes:\*\*\s*(\d+)|(\d+)\s+bytes)')
KINDS = {'py': 'PY', 'python': 'PY', 'md': 'MD', 'markdown': 'MD'}


def blob_sha(payload):
    return hashlib.sha1(b'blob ' + str(len(payload)).encode('ascii') + b'\0' + payload).hexdigest()


def relative_file(root, value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('path must be a nonempty POSIX-relative string')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('path must remain inside the repository')
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('resolved path escapes the repository')
    return resolved


def verify(root, manifest_path):
    issues = []
    def fail(message, **details):
        issues.append(dict(error=message, **details))
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, ValueError) as exc:
        return {'ok': False, 'errors': [{'error': 'cannot read manifest', 'detail': str(exc)}]}
    if not isinstance(manifest, dict) or not isinstance(manifest.get('entries'), list) or not manifest['entries']:
        return {'ok': False, 'errors': [{'error': 'manifest.entries must be a nonempty array'}]}
    for key in ('source_commit', 'main_base_commit'):
        if not isinstance(manifest.get(key), str) or not re.fullmatch('[0-9a-f]{40}', manifest[key]):
            fail('invalid manifest commit', field=key)
    expected, archives, anchors = {}, set(), set()
    for index, item in enumerate(manifest['entries']):
        if not isinstance(item, dict):
            fail('manifest entry must be an object', index=index)
            continue
        try:
            path = item['path']
            relative_file(root, path)
            archive = item['archive_path']
            relative_file(root, archive)
            kind = KINDS.get(str(item['kind']).lower())
            if kind is None:
                raise ValueError('kind must be py/python or md/markdown')
            if not path.endswith('.py' if kind == 'PY' else '.md'):
                raise ValueError('path extension disagrees with kind')
            if not isinstance(item['sha'], str) or not re.fullmatch('[0-9a-f]{40}', item['sha']):
                raise ValueError('sha must be a 40-character Git blob SHA')
            if type(item['size']) is not int or item['size'] < 0:
                raise ValueError('size must be a nonnegative integer')
            if not isinstance(item.get('anchor'), str) or not item['anchor']:
                raise ValueError('anchor must be a nonempty string')
            if not re.fullmatch(r'[A-Za-z0-9_-]+', item['anchor']):
                raise ValueError('anchor must be a plain HTML id without a leading #')
            if item['anchor'] in anchors:
                raise ValueError('duplicate manifest anchor')
            if path in expected:
                fail('duplicate manifest source path', path=path, index=index)
                continue
            expected[path] = dict(item, marker_kind=kind)
            anchors.add(item['anchor'])
            archives.add(archive)
        except (KeyError, TypeError, ValueError) as exc:
            fail('invalid manifest entry', index=index, detail=str(exc))
    # Inspect sibling archive volumes too, so unlisted extra source blocks fail.
    for archive in list(archives):
        parent = relative_file(root, archive).parent
        for sibling in parent.glob('*.md'):
            archives.add(sibling.relative_to(root).as_posix())
    seen = Counter()
    verified = Counter()
    for archive in sorted(archives):
        try:
            data = relative_file(root, archive).read_bytes()
        except (OSError, ValueError) as exc:
            fail('cannot read archive', archive=archive, detail=str(exc))
            continue
        tokens = list(TOKEN.finditer(data))
        literal_count = sum(data.count(x) for x in (
            b'<!-- BEGIN_PY_SOURCE', b'<!-- END_PY_SOURCE',
            b'<!-- BEGIN_MD_SOURCE', b'<!-- END_MD_SOURCE'))
        if literal_count != len(tokens):
            fail('malformed source delimiter', archive=archive)
        previous_end = 0
        for pos, token in enumerate(tokens):
            if token.group(1) == b'END':
                if pos == 0 or tokens[pos - 1].group(1) != b'BEGIN':
                    fail('unpaired end marker', archive=archive, offset=token.start())
                continue
            begin = BEGIN.fullmatch(token.group())
            if begin is None:
                fail('malformed begin marker', archive=archive, offset=token.start())
                continue
            try:
                path = begin.group(2).decode('utf-8')
                sha = begin.group(3).decode('ascii')
                kind = begin.group(1).decode('ascii')
            except UnicodeDecodeError:
                fail('invalid marker encoding', archive=archive)
                continue
            seen[path] += 1
            meta = expected.get(path)
            before_errors = len(issues)
            if meta is None:
                fail('source block absent from manifest', archive=archive, path=path)
            else:
                for actual, wanted, field in ((archive, meta['archive_path'], 'archive_path'),
                                             (kind, meta['marker_kind'], 'kind'),
                                             (sha, meta['sha'], 'marker SHA')):
                    if actual != wanted:
                        fail('source metadata mismatch', archive=archive, path=path, field=field)
            if pos + 1 >= len(tokens) or tokens[pos + 1].group(1) != b'END':
                fail('missing end marker or nested begin marker', archive=archive, path=path)
                continue
            ending = tokens[pos + 1]
            end = END.fullmatch(ending.group())
            if end is None or end.group(1) != begin.group(1) or end.group(2) != begin.group(2):
                fail('end marker does not match begin marker', archive=archive, path=path)
            header = data[previous_end:token.start()]
            previous_end = ending.end()
            source_anchors = re.findall(rb'<a (?:name|id)="([^"]+)"></a>', header)
            if meta and source_anchors != [meta['anchor'].encode('ascii')]:
                fail('missing, duplicate or wrong HTML source anchor', archive=archive, path=path)
            headers = list(HEADER.finditer(header))
            if len(headers) != 1:
                fail('missing or ambiguous source byte/SHA header', archive=archive, path=path)
            else:
                shas = re.findall(rb'`([0-9a-fA-F]{40})`', headers[0].group())
                sizes = list(SIZE.finditer(headers[0].group()))
                if len(shas) != 1 or len(sizes) != 1:
                    fail('malformed source byte/SHA header', archive=archive, path=path)
                elif meta:
                    declared = int(next(g for g in sizes[0].groups() if g is not None))
                    if shas[0].decode('ascii') != meta['sha'] or declared != meta['size']:
                        fail('source byte/SHA header disagrees with manifest', archive=archive, path=path)
            wrapped = data[token.end():ending.start()]
            openings, closings = list(OPEN.finditer(wrapped)), list(CLOSE.finditer(wrapped))
            if len(openings) != 1 or len(closings) != 1 or openings[0].end() > closings[0].start():
                fail('missing, duplicate or malformed eight-backtick fence', archive=archive, path=path)
                continue
            opener, closer = openings[0], closings[0]
            language = b'python' if kind == 'PY' else b'markdown'
            if opener.group(1) != language or wrapped[:opener.start()].strip() or wrapped[closer.end():].strip():
                fail('fence language or surrounding content is invalid', archive=archive, path=path)
            raw = wrapped[opener.end():closer.start()]
            candidates = [raw]
            if raw.endswith(b'\n'):
                candidates.append(raw[:-1])
            if raw.endswith(b'\r\n'):
                candidates.append(raw[:-2])
            if meta and not any(len(p) == meta['size'] and blob_sha(p) == meta['sha'] for p in candidates):
                fail('source bytes do not match manifest Git blob', archive=archive, path=path,
                     candidate_sizes=[len(p) for p in candidates])
            if meta and len(issues) == before_errors:
                verified[kind] += 1
    for path in sorted(set(expected) - set(seen)):
        fail('missing source block', path=path)
    for path, count in sorted(seen.items()):
        if count != 1:
            fail('duplicate source block', path=path, count=count)
    return {
        'ok': not issues,
        'source_commit': manifest.get('source_commit'),
        'main_base_commit': manifest.get('main_base_commit'),
        'manifest_entries': len(manifest['entries']),
        'unique_source_blocks': len(seen),
        'verified_python': verified['PY'],
        'verified_markdown': verified['MD'],
        'errors': issues,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', default='.', help='repository root (default: current directory)')
    parser.add_argument('--manifest', default='docs/lossless_source_manifest.json', help='manifest path relative to root')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    try:
        result = verify(root, relative_file(root, args.manifest))
    except (ValueError, OSError) as exc:
        result = {'ok': False, 'errors': [{'error': str(exc)}]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())

```

<a name="section-a0fe31ec5c73"></a>

## 부록 A — Python 327개 전체 원문 색인

각 링크는 해당 파일의 **전체 원문**으로 연결한다. 역할 열은 원문 Module Docstring 또는 실제 최상위 정의·테스트 이름에서 만든 짧은 색인 설명이다. 구체적인 실행 과정은 위의 해당 Project·Phase 본문에서 설명한다. `__init__.py`·CLI·패키지 보조 파일·회귀 테스트·빈 Placeholder도 생략하지 않았다.

| 파일 전체 원문 | 역할·실행 진입점 | 원본 바이트 |
|---|---|---:|
| [backend/__init__.py](technical_sources/PYTHON_SOURCE_01.md#py-001) | Local-only HTTP transport for the GUI-1.0 HUNL game service. | 67 |
| [backend/game_router.py](technical_sources/PYTHON_SOURCE_01.md#py-002) | HTTP adapter for GUI-1.0 sessions: no poker rules implemented here | 6,122 |
| [backend/main.py](technical_sources/PYTHON_SOURCE_01.md#py-003) | Local FastAPI app for Heads-Up GUI and Tournament ENV Phase 1.1 | 2,189 |
| [backend/schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-004) | Strict API request and intentionally whitelisted response models | 4,556 |
| [backend/tournament_router.py](technical_sources/PYTHON_SOURCE_01.md#py-005) | HTTP routes for Tournament ENV 1.1; separate from the HUNL200 routes. | 6,543 |
| [backend/tournament_schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-006) | Strict, whitelisted API contract for Tournament ENV Phase 1.1. | 4,253 |
| [main.py](technical_sources/PYTHON_SOURCE_01.md#py-007) | Play independent heads-up hands with corrected positions by default. | 1,480 |
| [scripts/audit_corrected.py](technical_sources/PYTHON_SOURCE_01.md#py-008) | uv run python scripts/audit_corrected.py --hands 100 --seed 42 | 1,903 |
| [scripts/audit_multiplayer.py](technical_sources/PYTHON_SOURCE_01.md#py-009) | Quick random-benchmark smoke audit | 1,888 |
| [scripts/audit_rlcard.py](technical_sources/PYTHON_SOURCE_01.md#py-010) | Run against installed RLCard: uv run python scripts/audit_rlcard.py --hands 100 | 1,978 |
| [scripts/audit_tournament.py](technical_sources/PYTHON_SOURCE_01.md#py-011) | Single-table tournament smoke audit; run `uv run python scripts/audit_tournament.py`. | 2,399 |
| [scripts/expand_phase01_smoke.py](technical_sources/PYTHON_SOURCE_01.md#py-012) | Real-engine Expand Phase 1 acceptance smoke: each table target 2..6 seats | 5,137 |
| [scripts/expand_phase02_smoke.py](technical_sources/PYTHON_SOURCE_02.md#py-013) | Expand E2: real-engine 2..6 person acting observation & action smoke | 4,124 |
| [scripts/expand_phase03_evaluate.py](technical_sources/PYTHON_SOURCE_02.md#py-014) | Independent optional E3 heldout benchmark; never changes weights. | 1,497 |
| [scripts/expand_phase03_train.py](technical_sources/PYTHON_SOURCE_02.md#py-015) | Expand Phase 3: 2-6 player PPO self-play on the existing cash game engine | 4,063 |
| [scripts/expand_phase04_evaluate.py](technical_sources/PYTHON_SOURCE_02.md#py-016) | Read-only benchmark of a Phase4 blueprint against a uniform legal agent. | 1,280 |
| [scripts/expand_phase04_mccfr.py](technical_sources/PYTHON_SOURCE_02.md#py-017) | Expand Phase4 reference MCCFR | 5,062 |
| [scripts/expand_phase05_evaluate.py](technical_sources/PYTHON_SOURCE_02.md#py-018) | Paired-deal toy benchmark: search vs the SAME immutable Phase4 blueprint | 4,121 |
| [scripts/expand_phase05_search.py](technical_sources/PYTHON_SOURCE_02.md#py-019) | Phase5 read-only toy / Hold'em search benchmark | 3,918 |
| [scripts/expand_phase06_benchmark.py](technical_sources/PYTHON_SOURCE_02.md#py-020) | Seed-blocked synthetic change-point diagnostic: NOT poker win-rate testing | 6,571 |
| [scripts/expand_phase07_benchmark.py](technical_sources/PYTHON_SOURCE_02.md#py-021) | E7 no-training cash/tournament benchmark; never overwrites reports. | 5,102 |
| [scripts/gui10_smoke.py](technical_sources/PYTHON_SOURCE_02.md#py-022) | Transport-free GUI-1.0 HUNL smoke run; human plays check/call by default | 1,820 |
| [scripts/gui11_smoke.py](technical_sources/PYTHON_SOURCE_02.md#py-023) | HTTP smoke exercise without starting an external server | 2,129 |
| [scripts/gui14_smoke.py](technical_sources/PYTHON_SOURCE_02.md#py-024) | Smoke a trusted local Phase 15/16 neural checkpoint through GUI FastAPI | 2,785 |
| [scripts/gui15_smoke.py](technical_sources/PYTHON_SOURCE_03.md#py-025) | Exercise GUI-1.5 HTTP end-to-end without launching React/browser | 3,671 |
| [scripts/gui16_smoke.py](technical_sources/PYTHON_SOURCE_03.md#py-026) | Phase GUI-1.6 local HTTP multi-hand carryover smoke | 2,543 |
| [scripts/phase10_train.py](technical_sources/PYTHON_SOURCE_03.md#py-027) | Phase 10 experimental Q-Boost PPO; optional matched-start PPO comparator | 8,433 |
| [scripts/phase11_benchmark.py](technical_sources/PYTHON_SOURCE_03.md#py-028) | Measure inference batching gains without PPO optimizer steps. | 2,925 |
| [scripts/phase11_compare.py](technical_sources/PYTHON_SOURCE_03.md#py-029) | Compare separate PPO and QBoost JSONL logs at matched update counts | 4,163 |
| [scripts/phase11_train.py](technical_sources/PYTHON_SOURCE_03.md#py-030) | Phase 11 training CLI with real-time progress, time-left and ETA | 4,449 |
| [scripts/phase12_evaluate.py](technical_sources/PYTHON_SOURCE_03.md#py-031) | Phase12 independent paired-deck evaluator for Phase9/10/11 local checkpoints | 5,139 |
| [scripts/phase13_benchmark.py](technical_sources/PYTHON_SOURCE_03.md#py-032) | Measure ACTUAL training update throughput for each hands/update + env batch pair | 4,229 |
| [scripts/phase13_compare.py](technical_sources/PYTHON_SOURCE_03.md#py-033) | Compare Phase13 PPO and QBoost training diagnostics without claiming GTO | 4,027 |
| [scripts/phase13_train.py](technical_sources/PYTHON_SOURCE_03.md#py-034) | Phase 13: PPO/QBoost batch scaling, checkpoint lineage, live tqdm ETA | 4,257 |
| [scripts/phase14_study.py](technical_sources/PYTHON_SOURCE_03.md#py-035) | Multi-seed controlled PPO/QBoost HUNL experiment | 3,502 |
| [scripts/phase15_critic_audit.py](technical_sources/PYTHON_SOURCE_03.md#py-036) | Run a read-only diagnostic for two trusted Phase-15 HUNL PPO checkpoints | 3,142 |
| [scripts/phase15_finalize.py](technical_sources/PYTHON_SOURCE_04.md#py-037) | Phase 15 independent held-out evaluation, run ONCE after training is finished. | 639 |
| [scripts/phase15_initial_policy_comparison.py](technical_sources/PYTHON_SOURCE_04.md#py-038) | Read-only baseline action comparison for HUNL self-play, Phase 15 | 3,227 |
| [scripts/phase15_policy_audit.py](technical_sources/PYTHON_SOURCE_04.md#py-039) | CLI: fixed-state policy audit and existing PPO metrics review | 2,491 |
| [scripts/phase15_preflop_sequence_audit.py](technical_sources/PYTHON_SOURCE_04.md#py-040) | Read-only HUNL Phase15 preflop action-sequence audit | 2,636 |
| [scripts/phase15_terminal_reward_audit.py](technical_sources/PYTHON_SOURCE_04.md#py-041) | Read-only HUNL PPO per-hand reward tails and preflop exit audit | 2,694 |
| [scripts/phase15_train.py](technical_sources/PYTHON_SOURCE_04.md#py-042) | Phase 15 HUNL long-run training, from a Phase14 best.pt (or trusted PPO/QBoost checkpoint). | 2,089 |
| [scripts/phase16_evaluate.py](technical_sources/PYTHON_SOURCE_04.md#py-043) | No-training Phase16 benchmark for Phase14 and future Phase15 best.pt actors. | 6,132 |
| [scripts/phase16_ppo_ablation.py](technical_sources/PYTHON_SOURCE_04.md#py-044) | Phase16 isolated A0/A1/A2/A3 PPO ablation (NEW RUN ONLY). | 3,897 |
| [scripts/phase16_ppo_stress.py](technical_sources/PYTHON_SOURCE_04.md#py-045) | Phase16 long PPO A0/A3 stress, full-state resume, and paired comparison | 6,453 |
| [scripts/phase17_evaluate.py](technical_sources/PYTHON_SOURCE_04.md#py-046) | Phase 17 variable-stack continuous-match evaluation (no training) | 10,446 |
| [scripts/phase17_gpu_retrain.py](technical_sources/PYTHON_SOURCE_04.md#py-047) | Five-hour Phase17 GPU retraining: new run by default; resume explicitly. | 3,573 |
| [scripts/phase5_rlcard_smoke.py](technical_sources/PYTHON_SOURCE_04.md#py-048) | Run from repository root: uv run python scripts/phase5_rlcard_smoke.py | 880 |
| [scripts/phase6_benchmark.py](technical_sources/PYTHON_SOURCE_05.md#py-049) | CPU-only Phase 6 HUNL200 benchmark; no PyTorch/CUDA required | 3,422 |
| [scripts/phase6_hardware_probe.py](technical_sources/PYTHON_SOURCE_05.md#py-050) | Safe, tiny PyTorch CUDA compute probe for Pascal-class GPUs (optional). | 1,664 |
| [scripts/phase7_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-051) | Run from repo root: uv run --extra gpu python -m scripts.phase7_smoke --hands 100 --device cuda No policy optimization occurs in Phase  | 4,254 |
| [scripts/phase8_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-052) | Phase-8 integration smoke: real Phase-6 rollout -> Phase-7 encoders -> trainable nets | 6,235 |
| [scripts/phase9_train.py](technical_sources/PYTHON_SOURCE_05.md#py-053) | CLI: Phase-9 actual PPO updates, checkpoint/resume, random baseline evaluation | 4,240 |
| [scripts/run_tournament.py](technical_sources/PYTHON_SOURCE_05.md#py-054) | Run one 2-6 seat tournament (random-agent baseline) and show eliminations. | 1,534 |
| [scripts/tournament_env131_diagnose.py](technical_sources/PYTHON_SOURCE_05.md#py-055) | Show conditional preflop heuristic raise mix (NOT actual in-game raise frequency) | 1,557 |
| [scripts/tournament_env13_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-056) | Reproducible six-seat Tournament ENV 1.3 bot lineup smoke, no HTTP server. | 1,499 |
| [scripts/tournament_env14_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-057) | Real trusted Expand Phase3 checkpoint smoke for Tournament ENV 1.4 | 1,760 |
| [scripts/tournament_env17_release_audit.py](technical_sources/PYTHON_SOURCE_05.md#py-058) | Offline Git safety audit for Tournament ENV 1.7 | 5,498 |
| [scripts/tournament_env1_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-059) | API smoke: uv run --extra dev --extra gui python -m scripts.tournament_env1_smoke | 1,891 |
| [src/holdem_ai/__init__.py](technical_sources/PYTHON_SOURCE_05.md#py-060) | No-Limit Texas Hold'em AI platform. | 65 |
| [src/holdem_ai/agents/__init__.py](technical_sources/PYTHON_SOURCE_06.md#py-061) | Policy and agent implementations. | 40 |
| [src/holdem_ai/agents/base_agent.py](technical_sources/PYTHON_SOURCE_06.md#py-062) | Common interface for poker agents | 460 |
| [src/holdem_ai/agents/cfr_agent.py](technical_sources/PYTHON_SOURCE_06.md#py-063) | 빈 파일; 실행 정의 없음. | 0 |
| [src/holdem_ai/agents/diagnostic_opponents.py](technical_sources/PYTHON_SOURCE_06.md#py-064) | Deterministic benchmark opponents; all actions are legal by construction | 2,418 |
| [src/holdem_ai/agents/neural_policy.py](technical_sources/PYTHON_SOURCE_06.md#py-065) | Inference-only adapter implementing the Phase-5 AgentPolicy interface. | 1,680 |
| [src/holdem_ai/agents/policies.py](technical_sources/PYTHON_SOURCE_06.md#py-066) | Mask-correct policies usable by random, CFR and neural learners. | 3,504 |
| [src/holdem_ai/agents/random_agent.py](technical_sources/PYTHON_SOURCE_06.md#py-067) | Random legal-action baseline, for testing the environment/agent boundary. | 518 |
| [src/holdem_ai/agents/rl_agent.py](technical_sources/PYTHON_SOURCE_06.md#py-068) | 빈 파일; 실행 정의 없음. | 0 |
| [src/holdem_ai/core/__init__.py](technical_sources/PYTHON_SOURCE_06.md#py-069) | Consolidated functional package. | 39 |
| [src/holdem_ai/core/schema.py](technical_sources/PYTHON_SOURCE_06.md#py-070) | Framework-neutral, seat-relative agent decision contract (Python >= 3.11). | 3,547 |
| [src/holdem_ai/encoding/__init__.py](technical_sources/PYTHON_SOURCE_06.md#py-071) | Consolidated functional package. | 39 |
| [src/holdem_ai/encoding/actions.py](technical_sources/PYTHON_SOURCE_06.md#py-072) | Phase-5 compatible fixed 7-head action encoding and masked distributions | 1,632 |
| [src/holdem_ai/encoding/observation.py](technical_sources/PYTHON_SOURCE_07.md#py-073) | HUNL200 VRPO-Lite v1 actor/centralized-critic encoders | 8,671 |
| [src/holdem_ai/encoding/tensors.py](technical_sources/PYTHON_SOURCE_07.md#py-074) | Optional PyTorch adapters; importable without torch until batching is requested. | 2,352 |
| [src/holdem_ai/encoding/vocabulary.py](technical_sources/PYTHON_SOURCE_07.md#py-075) | Stable VRPO-Lite v1 token vocabulary (NOT the original VRPO paper encoding). | 1,074 |
| [src/holdem_ai/environment/__init__.py](technical_sources/PYTHON_SOURCE_07.md#py-076) | Poker hand and tournament environments. | 46 |
| [src/holdem_ai/environment/actions.py](technical_sources/PYTHON_SOURCE_07.md#py-077) | Independent no-limit betting constraints (amounts are integer chips) | 3,395 |
| [src/holdem_ai/environment/bet_abstraction.py](technical_sources/PYTHON_SOURCE_07.md#py-078) | Optional integer-chip action abstraction for engines with arbitrary raise-to amounts | 5,821 |
| [src/holdem_ai/environment/blind_manager.py](technical_sources/PYTHON_SOURCE_07.md#py-079) | Deterministic hand-count blind schedule for single-table tournaments | 2,482 |
| [src/holdem_ai/environment/corrected_game.py](technical_sources/PYTHON_SOURCE_07.md#py-080) | RLCard 1.2.0 game subclass fixing heads-up seats and betting round | 5,690 |
| [src/holdem_ai/environment/corrected_round.py](technical_sources/PYTHON_SOURCE_07.md#py-081) | Two-player no-limit betting round for the RLCard 1.2.0 action abstraction | 5,593 |
| [src/holdem_ai/environment/game_state.py](technical_sources/PYTHON_SOURCE_07.md#py-082) | 빈 파일; 실행 정의 없음. | 0 |
| [src/holdem_ai/environment/hand_engine.py](technical_sources/PYTHON_SOURCE_07.md#py-083) | One-hand HUNL interface | 4,869 |
| [src/holdem_ai/environment/hand_evaluator.py](technical_sources/PYTHON_SOURCE_07.md#py-084) | Independent exact seven-card Hold'em evaluator for showdown verification | 3,274 |
| [src/holdem_ai/environment/hunl200_adapter.py](technical_sources/PYTHON_SOURCE_08.md#py-085) | HUNL200 adapter: real Phase-3 hand engine -> Phase-5 turn contract | 8,695 |
| [src/holdem_ai/environment/multi_player_engine.py](technical_sources/PYTHON_SOURCE_08.md#py-086) | Standalone 2-6 player no-limit Hold'em *hand* environment | 17,555 |
| [src/holdem_ai/environment/multi_player_seats.py](technical_sources/PYTHON_SOURCE_08.md#py-087) | Moving-button seat and action order for 2-6 seat Texas Hold'em hands | 2,113 |
| [src/holdem_ai/environment/observation.py](technical_sources/PYTHON_SOURCE_08.md#py-088) | 빈 파일; 실행 정의 없음. | 0 |
| [src/holdem_ai/environment/player_manager.py](technical_sources/PYTHON_SOURCE_08.md#py-089) | Persistent chip stacks and elimination ranks across tournament hands | 2,666 |
| [src/holdem_ai/environment/pot_manager.py](technical_sources/PYTHON_SOURCE_08.md#py-090) | Reference side-pot construction and settlement with chip conservation | 3,297 |
| [src/holdem_ai/environment/rewards.py](technical_sources/PYTHON_SOURCE_08.md#py-091) | Hand-level chip rewards and terminal tournament placement rewards. | 1,789 |
| [src/holdem_ai/environment/rlcard_adapter.py](technical_sources/PYTHON_SOURCE_08.md#py-092) | Conservative RLCard Env adapter | 7,021 |
| [src/holdem_ai/environment/rules_audit.py](technical_sources/PYTHON_SOURCE_08.md#py-093) | Observational, non-mutating audit of installed RLCard HUNL baseline | 1,782 |
| [src/holdem_ai/environment/seat_manager.py](technical_sources/PYTHON_SOURCE_08.md#py-094) | Rule reference for heads-up seating and betting order | 1,643 |
| [src/holdem_ai/environment/tournament_env.py](technical_sources/PYTHON_SOURCE_08.md#py-095) | Single-table 2-6 seat knockout tournament over the Phase 3 hand engine | 10,191 |
| [src/holdem_ai/evaluation/__init__.py](technical_sources/PYTHON_SOURCE_08.md#py-096) | Game and tournament benchmarks. | 38 |
| [src/holdem_ai/evaluation/action_diagnostics.py](technical_sources/PYTHON_SOURCE_09.md#py-097) | Action-rate and legal-opportunity diagnostics on an on-policy trajectory | 4,003 |
| [src/holdem_ai/evaluation/balanced.py](technical_sources/PYTHON_SOURCE_09.md#py-098) | Seat/button-balanced held-out tests vs Random and a frozen anchor Actor. | 3,092 |
| [src/holdem_ai/evaluation/benchmark.py](technical_sources/PYTHON_SOURCE_09.md#py-099) | 빈 파일; 실행 정의 없음. | 0 |
| [src/holdem_ai/evaluation/checkpoint_loader.py](technical_sources/PYTHON_SOURCE_09.md#py-100) | Read trusted Phase-9/10/11 checkpoint or Phase-11 anchor, actor ONLY. | 950 |
| [src/holdem_ai/evaluation/continuous_stack.py](technical_sources/PYTHON_SOURCE_09.md#py-101) | Phase 17: matched-seed, variable-stack HUNL continuous-match evaluation | 14,714 |
| [src/holdem_ai/evaluation/critic_diagnostics.py](technical_sources/PYTHON_SOURCE_09.md#py-102) | Read-only HUNL PPO critic calibration diagnostics | 11,696 |
| [src/holdem_ai/evaluation/fixed_state_policy_audit.py](technical_sources/PYTHON_SOURCE_09.md#py-103) | Evaluate frozen HUNL actor checkpoints on exactly the same actor-visible states | 6,773 |
| [src/holdem_ai/evaluation/initial_policy_comparison.py](technical_sources/PYTHON_SOURCE_09.md#py-104) | Read-only Agent comparison: uniform legal, random-init, trained Phase14 | 9,620 |
| [src/holdem_ai/evaluation/opponent_pool.py](technical_sources/PYTHON_SOURCE_09.md#py-105) | Phase16 diagnostic opponent-pool benchmark, independent of Phase15 training | 6,852 |
| [src/holdem_ai/evaluation/ppo_metric_audit.py](technical_sources/PYTHON_SOURCE_09.md#py-106) | Read-only Phase 15 PPO optimizer diagnostics from existing JSONL logs. | 1,906 |
| [src/holdem_ai/evaluation/preflop_sequence_audit.py](technical_sources/PYTHON_SOURCE_09.md#py-107) | Read-only HUNL agent preflop decision-sequence diagnostics | 11,220 |
| [src/holdem_ai/evaluation/reward_diagnostics.py](technical_sources/PYTHON_SOURCE_09.md#py-108) | Summarize on-policy HUNL terminal rewards and termination actions per update | 1,717 |
| [src/holdem_ai/evaluation/seed_blocked.py](technical_sources/PYTHON_SOURCE_10.md#py-109) | Independent seed-blocked, four-layout HUNL200 evaluation (Phase 12) | 10,961 |
| [src/holdem_ai/evaluation/seed_statistics.py](technical_sources/PYTHON_SOURCE_10.md#py-110) | Training-seed-level statistical estimates for fixed independent test-deck banks | 1,889 |
| [src/holdem_ai/evaluation/strategy_diagnostics.py](technical_sources/PYTHON_SOURCE_10.md#py-111) | Actor-observation-only behavioral diagnostics for evaluated HUNL200 hands. | 5,641 |
| [src/holdem_ai/evaluation/strategy_report.py](technical_sources/PYTHON_SOURCE_10.md#py-112) | Obsidian-compatible Phase16 benchmark report from JSON-safe results. | 4,153 |
| [src/holdem_ai/evaluation/terminal_reward_audit.py](technical_sources/PYTHON_SOURCE_10.md#py-113) | Read-only per-HAND payoff and preflop-exit audit for HUNL PPO | 12,130 |
| [src/holdem_ai/evaluation/tournament_eval.py](technical_sources/PYTHON_SOURCE_10.md#py-114) | 빈 파일; 실행 정의 없음. | 0 |
| [src/holdem_ai/expand/__init__.py](technical_sources/PYTHON_SOURCE_10.md#py-115) | 6-Max expansion modules | 77 |
| [src/holdem_ai/expand/environment/__init__.py](technical_sources/PYTHON_SOURCE_10.md#py-116) | Expand Phase 1 public table API. | 561 |
| [src/holdem_ai/expand/environment/cash_rules.py](technical_sources/PYTHON_SOURCE_10.md#py-117) | E1 cash table admission rules: fixed blinds, no rake, wait for BB. | 1,073 |
| [src/holdem_ai/expand/environment/event_ledger.py](technical_sources/PYTHON_SOURCE_10.md#py-118) | Append-only deterministic journal; privileged seed log, not an actor observation. | 1,623 |
| [src/holdem_ai/expand/environment/multiway_adapter.py](technical_sources/PYTHON_SOURCE_10.md#py-119) | Late-bound adapter to the unchanged Base Phase 3 6-seat hand engine. | 943 |
| [src/holdem_ai/expand/environment/table_manager.py](technical_sources/PYTHON_SOURCE_10.md#py-120) | Expand Phase 1: dynamic 6-seat cash sessions on the unchanged Phase 3 hand engine | 22,864 |
| [src/holdem_ai/expand/environment/table_state.py](technical_sources/PYTHON_SOURCE_11.md#py-121) | Public, immutable table metadata; does not contain hole cards or hidden deck. | 3,204 |
| [src/holdem_ai/expand/environment/tournament_rules.py](technical_sources/PYTHON_SOURCE_11.md#py-122) | Protocol boundary for E7; E1 deliberately does not simulate tournaments. | 589 |
| [src/holdem_ai/expand/information/__init__.py](technical_sources/PYTHON_SOURCE_11.md#py-123) | Expand Phase 2 acting-player information, action translation and encoding. | 1,010 |
| [src/holdem_ai/expand/information/actions.py](technical_sources/PYTHON_SOURCE_11.md#py-124) | Exact legality-preserving discrete action translation, with no silent clamp. | 3,375 |
| [src/holdem_ai/expand/information/adapter.py](technical_sources/PYTHON_SOURCE_11.md#py-125) | Phase2 interface for Phase1 DynamicTable; records public action street exactly | 3,795 |
| [src/holdem_ai/expand/information/builder.py](technical_sources/PYTHON_SOURCE_11.md#py-126) | Construct an InformationState only from acting-player public observation | 6,109 |
| [src/holdem_ai/expand/information/encoder.py](technical_sources/PYTHON_SOURCE_11.md#py-127) | Stable, deterministic 6-seat tensor-ready encoding (Python only). | 3,985 |
| [src/holdem_ai/expand/information/policy_v2.py](technical_sources/PYTHON_SOURCE_11.md#py-128) | Small, trainable multi-seat neural policy | 4,370 |
| [src/holdem_ai/expand/information/schema.py](technical_sources/PYTHON_SOURCE_11.md#py-129) | Expand Phase 2: allowlisted, immutable acting-player-only information | 2,902 |
| [src/holdem_ai/expand/integration/__init__.py](technical_sources/PYTHON_SOURCE_11.md#py-130) | 패키지 초기화·공개 인터페이스. | 703 |
| [src/holdem_ai/expand/integration/agent.py](technical_sources/PYTHON_SOURCE_11.md#py-131) | E7 routing: strictly actor-visible InformationState -> one legal Phase1 Action | 7,108 |
| [src/holdem_ai/expand/integration/benchmark.py](technical_sources/PYTHON_SOURCE_11.md#py-132) | E7 reproducible *independent-session* cash and tournament benchmarks | 11,947 |
| [src/holdem_ai/expand/integration/manifest.py](technical_sources/PYTHON_SOURCE_12.md#py-133) | Reproducible experiment metadata; no checkpoint modification. | 2,055 |
| [src/holdem_ai/expand/integration/source_bridges.py](technical_sources/PYTHON_SOURCE_12.md#py-134) | Opt-in E4/E5 source bridges with explicit infoset/belief contract | 2,477 |
| [src/holdem_ai/expand/integration/tournament.py](technical_sources/PYTHON_SOURCE_12.md#py-135) | Single-table 2-6 seat tournament adapter over Base Phase3 Hand Engine | 12,012 |
| [src/holdem_ai/expand/opponent/__init__.py](technical_sources/PYTHON_SOURCE_12.md#py-136) | Expand Phase 6: public-only per-tenure opponents and bounded adaptation. | 948 |
| [src/holdem_ai/expand/opponent/adaptive.py](technical_sources/PYTHON_SOURCE_12.md#py-137) | Trust-region gate around an already valid public-information policy | 5,634 |
| [src/holdem_ai/expand/opponent/bridge.py](technical_sources/PYTHON_SOURCE_12.md#py-138) | Phase1/Phase2 public event bridge, called ONLY after a legal action commits | 4,284 |
| [src/holdem_ai/expand/opponent/model.py](technical_sources/PYTHON_SOURCE_12.md#py-139) | Per-player Bayesian-smoothed decayed action model; only public observations | 9,149 |
| [src/holdem_ai/expand/opponent/persistence.py](technical_sources/PYTHON_SOURCE_12.md#py-140) | Deterministic allowlisted JSON checkpoint, not Python pickle | 5,976 |
| [src/holdem_ai/expand/opponent/router.py](technical_sources/PYTHON_SOURCE_12.md#py-141) | Optional model-conditioned routing among PRE-EXISTING hero policies | 2,247 |
| [src/holdem_ai/expand/opponent/schema.py](technical_sources/PYTHON_SOURCE_12.md#py-142) | Allowlisted public-only events and fixed-cardinality legal-action context | 4,004 |
| [src/holdem_ai/expand/pluribus/__init__.py](technical_sources/PYTHON_SOURCE_12.md#py-143) | Pluribus-inspired research branch; NOT Pluribus original software. | 73 |
| [src/holdem_ai/expand/pluribus/abstraction/__init__.py](technical_sources/PYTHON_SOURCE_12.md#py-144) | 패키지 초기화·공개 인터페이스. | 111 |
| [src/holdem_ai/expand/pluribus/abstraction/cards.py](technical_sources/PYTHON_SOURCE_13.md#py-145) | Reproducible (deliberately *untrained*) toy 6-max hold'em abstraction | 2,718 |
| [src/holdem_ai/expand/pluribus/blueprint/__init__.py](technical_sources/PYTHON_SOURCE_13.md#py-146) | 패키지 초기화·공개 인터페이스. | 380 |
| [src/holdem_ai/expand/pluribus/blueprint/checkpoint.py](technical_sources/PYTHON_SOURCE_13.md#py-147) | Atomic gzipped JSON MCCFR checkpoints; no pickle or torch.load. | 3,419 |
| [src/holdem_ai/expand/pluribus/blueprint/evaluate.py](technical_sources/PYTHON_SOURCE_13.md#py-148) | Separate deterministic evaluation seeds; no trainer updates in evaluation. | 2,020 |
| [src/holdem_ai/expand/pluribus/blueprint/holdem.py](technical_sources/PYTHON_SOURCE_13.md#py-149) | Small-resource 2--6 seat Hold'em MCCFR adapter using the Phase3 engine | 4,709 |
| [src/holdem_ai/expand/pluribus/blueprint/kuhn_reference.py](technical_sources/PYTHON_SOURCE_13.md#py-150) | Exact 2-player Kuhn exploitability (unabstracted 3-card game only) | 2,042 |
| [src/holdem_ai/expand/pluribus/blueprint/mccfr.py](technical_sources/PYTHON_SOURCE_13.md#py-151) | Multi-player external-sampling MCCFR with linear weights and atomic iterations | 10,626 |
| [src/holdem_ai/expand/pluribus/blueprint/toy_games.py](technical_sources/PYTHON_SOURCE_13.md#py-152) | Tiny 2/3-player toy poker games for reference testing MCCFR | 6,048 |
| [src/holdem_ai/expand/pluribus/search/__init__.py](technical_sources/PYTHON_SOURCE_13.md#py-153) | Expand Phase5 bounded imperfect-information search prototype. | 558 |
| [src/holdem_ai/expand/pluribus/search/belief.py](technical_sources/PYTHON_SOURCE_13.md#py-154) | Public-knowledge-only roots for toy poker and collision-free Hold'em sampling. | 8,095 |
| [src/holdem_ai/expand/pluribus/search/continuations.py](technical_sources/PYTHON_SOURCE_13.md#py-155) | Four Pluribus-inspired continuation policy families | 1,934 |
| [src/holdem_ai/expand/pluribus/search/nested.py](technical_sources/PYTHON_SOURCE_13.md#py-156) | Public-trace session for re-searching at each decision / off-tree action | 1,862 |
| [src/holdem_ai/expand/pluribus/search/solver.py](technical_sources/PYTHON_SOURCE_14.md#py-157) | Bounded belief-conditioned external-sampling MCCFR with rollout leaf values | 9,865 |
| [src/holdem_ai/expand/rl/__init__.py](technical_sources/PYTHON_SOURCE_14.md#py-158) | Expand Phase 3: multiagent PPO on a public-information six-seat game. | 782 |
| [src/holdem_ai/expand/rl/collector.py](technical_sources/PYTHON_SOURCE_14.md#py-159) | Real-hand multiagent on-policy self-play on E1 table + E2 allowlisted adapter | 10,450 |
| [src/holdem_ai/expand/rl/model.py](technical_sources/PYTHON_SOURCE_14.md#py-160) | Expand 3: strictly actor-visible shared multi-seat Actor + Value networks. | 2,440 |
| [src/holdem_ai/expand/rl/population.py](technical_sources/PYTHON_SOURCE_14.md#py-161) | Training-time opponent selection, separate from final evaluation opponents. | 2,015 |
| [src/holdem_ai/expand/rl/ppo.py](technical_sources/PYTHON_SOURCE_14.md#py-162) | Clipped PPO on frozen collected 2-6 player trajectories. | 3,780 |
| [src/holdem_ai/expand/rl/returns.py](technical_sources/PYTHON_SOURCE_14.md#py-163) | Per-hand, per-seat GAE: decision streams are never conflated across players. | 2,513 |
| [src/holdem_ai/expand/rl/trainer.py](technical_sources/PYTHON_SOURCE_14.md#py-164) | Reproducible E3 PPO training, snapshots, safe resume and validation gates | 16,291 |
| [src/holdem_ai/experiments/__init__.py](technical_sources/PYTHON_SOURCE_14.md#py-165) | Consolidated functional package. | 39 |
| [src/holdem_ai/experiments/batch_scaling.py](technical_sources/PYTHON_SOURCE_14.md#py-166) | Phase 13 training-batch experiments, preserving Phase 9/10 checkpoints | 17,194 |
| [src/holdem_ai/experiments/controlled_study.py](technical_sources/PYTHON_SOURCE_14.md#py-167) | Matched-start PPO/QBoost study with independent held-out test evaluation | 21,714 |
| [src/holdem_ai/experiments/longrun.py](technical_sources/PYTHON_SOURCE_14.md#py-168) | Phase 15: safe long-run HUNL training, validation and sealed final audit | 26,085 |
| [src/holdem_ai/experiments/ppo_ablation.py](technical_sources/PYTHON_SOURCE_15.md#py-169) | Phase16 isolated PPO optimizer/critic ablations (AGENT ONLY) | 25,669 |
| [src/holdem_ai/experiments/ppo_five_hour.py](technical_sources/PYTHON_SOURCE_15.md#py-170) | Phase17: resumable 5-hour PPO retraining (AGENT ONLY, new run) | 24,123 |
| [src/holdem_ai/experiments/ppo_safety.py](technical_sources/PYTHON_SOURCE_15.md#py-171) | Phase 15 PPO stability safety policy | 944 |
| [src/holdem_ai/experiments/ppo_stress.py](technical_sources/PYTHON_SOURCE_15.md#py-172) | Phase16 long A0/A3 PPO stress test with transactional full-state resume | 30,107 |
| [src/holdem_ai/experiments/time_budget.py](technical_sources/PYTHON_SOURCE_15.md#py-173) | Safe wall-clock bounded Phase11 training with crash-recoverable observability | 15,737 |
| [src/holdem_ai/game_service/__init__.py](technical_sources/PYTHON_SOURCE_15.md#py-174) | Server-side HUNL human-vs-agent integration (GUI-1.0). | 182 |
| [src/holdem_ai/game_service/action_adapter.py](technical_sources/PYTHON_SOURCE_15.md#py-175) | Adapt the existing seven-action HUNL200 policy head to GUI action offers | 2,708 |
| [src/holdem_ai/game_service/bot_adapter.py](technical_sources/PYTHON_SOURCE_15.md#py-176) | Apply any Phase-5 AgentPolicy to a bot turn, including NeuralActorPolicy | 787 |
| [src/holdem_ai/game_service/checkpoint_catalog.py](technical_sources/PYTHON_SOURCE_15.md#py-177) | Trusted local checkpoint selection for GUI-1.4 HUNL evaluation | 4,986 |
| [src/holdem_ai/game_service/interactive_adapter.py](technical_sources/PYTHON_SOURCE_15.md#py-178) | Human-only exact bet sizing over the trained HUNL200 seven-action adapter | 2,755 |
| [src/holdem_ai/game_service/session_analytics.py](technical_sources/PYTHON_SOURCE_15.md#py-179) | GUI-1.5: bounded, hero-visible completed-hand ledger and online metrics | 5,496 |
| [src/holdem_ai/game_service/session_manager.py](technical_sources/PYTHON_SOURCE_15.md#py-180) | Local HUNL200 human-vs-agent game sessions; transport agnostic | 11,214 |
| [src/holdem_ai/game_service/state_projection.py](technical_sources/PYTHON_SOURCE_16.md#py-181) | Explicit whitelist of game data safe to serialize to the human browser | 4,653 |
| [src/holdem_ai/models/__init__.py](technical_sources/PYTHON_SOURCE_16.md#py-182) | Consolidated functional package. | 39 |
| [src/holdem_ai/models/transformer.py](technical_sources/PYTHON_SOURCE_16.md#py-183) | VRPO-Lite v1 trainable networks for the existing Phase-7 batch contracts | 8,612 |
| [src/holdem_ai/tournament_service/__init__.py](technical_sources/PYTHON_SOURCE_16.md#py-184) | Local-only GUI environment integration for the Phase 4 tournament engine. | 217 |
| [src/holdem_ai/tournament_service/baseline.py](technical_sources/PYTHON_SOURCE_16.md#py-185) | Temporary, engine-legal six-seat baseline policies | 1,345 |
| [src/holdem_ai/tournament_service/improved_bot.py](technical_sources/PYTHON_SOURCE_16.md#py-186) | Tournament ENV 1.3: four reproducible, observation-only heuristic opponents | 5,676 |
| [src/holdem_ai/tournament_service/journal.py](technical_sources/PYTHON_SOURCE_16.md#py-187) | Tournament ENV 1.6: append-only, local SQLite event journal | 10,830 |
| [src/holdem_ai/tournament_service/session.py](technical_sources/PYTHON_SOURCE_16.md#py-188) | Phase 1.1: six-seat, human-vs-baselines TournamentEnv session adapter | 20,805 |
| [src/holdem_ai/tournament_service/trained_actor.py](technical_sources/PYTHON_SOURCE_16.md#py-189) | Tournament ENV 1.4: frozen Expand Phase3 multi-seat PPO inference bridge | 9,173 |
| [src/holdem_ai/training/__init__.py](technical_sources/PYTHON_SOURCE_16.md#py-190) | Self-play and training pipelines. | 40 |
| [src/holdem_ai/training/batched_collector.py](technical_sources/PYTHON_SOURCE_16.md#py-191) | Batched (multi-hand, single-process) on-policy GPU inference for HUNL200 | 5,632 |
| [src/holdem_ai/training/episode_runner.py](technical_sources/PYTHON_SOURCE_16.md#py-192) | Turn-based multiagent self-play, one transition per decision by each player. | 2,963 |
| [src/holdem_ai/training/global_rollout.py](technical_sources/PYTHON_SOURCE_17.md#py-193) | Global-decision trajectories for future CTDE / VRPO learners (no training yet) | 3,127 |
| [src/holdem_ai/training/ppo.py](technical_sources/PYTHON_SOURCE_17.md#py-194) | On-policy PPO baseline for Phase-8 HUNL200 networks (NOT VRPO/Q-Boosting) | 15,577 |
| [src/holdem_ai/training/ppo_session.py](technical_sources/PYTHON_SOURCE_17.md#py-195) | Phase-9 on-policy PPO orchestration, safe checkpoints, and resume. | 5,241 |
| [src/holdem_ai/training/qboost.py](technical_sources/PYTHON_SOURCE_17.md#py-196) | Phase 10: doubly-corrected Q-control-variate PPO research baseline | 11,179 |
| [src/holdem_ai/training/qboost_session.py](technical_sources/PYTHON_SOURCE_17.md#py-197) | Phase-10 trainer, safe checkpoints and Phase-9 warm start | 5,741 |
| [src/holdem_ai/training/reward_models.py](technical_sources/PYTHON_SOURCE_17.md#py-198) | Terminal reward definitions; preserve distinction between chip and tournament utility. | 1,937 |
| [src/holdem_ai/training/self_play.py](technical_sources/PYTHON_SOURCE_17.md#py-199) | 빈 파일; 실행 정의 없음. | 0 |
| [src/holdem_ai/training/trainer.py](technical_sources/PYTHON_SOURCE_17.md#py-200) | 빈 파일; 실행 정의 없음. | 0 |
| [src/holdem_ai/utils/__init__.py](technical_sources/PYTHON_SOURCE_17.md#py-201) | Consolidated functional package. | 39 |
| [src/holdem_ai/utils/progress.py](technical_sources/PYTHON_SOURCE_17.md#py-202) | Live, optional-tqdm terminal progress for Phase11 training | 9,411 |
| [src/holdem_ai/utils/snapshot.py](technical_sources/PYTHON_SOURCE_17.md#py-203) | Safe, explicit experimental snapshot facility (not a CFR correctness guarantee). | 776 |
| [tests/__init__.py](technical_sources/PYTHON_SOURCE_17.md#py-204) | 빈 파일; 실행 정의 없음. | 0 |
| [tests/agents/__init__.py](technical_sources/PYTHON_SOURCE_18.md#py-205) | 빈 파일; 실행 정의 없음. | 0 |
| [tests/environment/__init__.py](technical_sources/PYTHON_SOURCE_18.md#py-206) | 빈 파일; 실행 정의 없음. | 0 |
| [tests/environment/test_betting.py](technical_sources/PYTHON_SOURCE_18.md#py-207) | 회귀 검증: test_bb_min_raise_preflop, test_postflop_bet_minimum_and_stack_cap, test_short_all_in_is_allowed_but_is_not_a_full_raise, tes | 1,888 |
| [tests/environment/test_elimination.py](technical_sources/PYTHON_SOURCE_18.md#py-208) | 빈 파일; 실행 정의 없음. | 0 |
| [tests/environment/test_hand_evaluator.py](technical_sources/PYTHON_SOURCE_18.md#py-209) | 회귀 검증: test_hierarchy_of_poker_hands, test_wheel_straight_loses_to_six_high, test_two_triplets_form_highest_full_house, test_board_p | 1,708 |
| [tests/environment/test_position.py](technical_sources/PYTHON_SOURCE_18.md#py-210) | 회귀 검증: test_heads_up_button_is_sb_and_first_preflop, test_blinds_are_charged_to_correct_seats, test_invalid_button_is_rejected | 942 |
| [tests/environment/test_side_pot.py](technical_sources/PYTHON_SOURCE_18.md#py-211) | 회귀 검증: test_three_way_all_in_has_correct_main_and_side_pots, test_folded_player_contribution_stays_in_pot, test_unmatched_wager_retu | 1,850 |
| [tests/expand/conftest.py](technical_sources/PYTHON_SOURCE_18.md#py-212) | Isolated stub for table-management tests; real-engine tests are separate. | 2,814 |
| [tests/expand/e03_fake_engine.py](technical_sources/PYTHON_SOURCE_18.md#py-213) | Deterministic test-only 2-6 player game | 3,340 |
| [tests/expand/e07_fake_tournament.py](technical_sources/PYTHON_SOURCE_18.md#py-214) | A deterministic side-effect-free test double, NEVER a poker engine. | 2,004 |
| [tests/expand/test_e01_real_engine.py](technical_sources/PYTHON_SOURCE_18.md#py-215) | These require the user's real Base Phase 3 modules (skipped in standalone patch). | 3,209 |
| [tests/expand/test_e01_smoke_unit.py](technical_sources/PYTHON_SOURCE_18.md#py-216) | CLI contract tested with a fake engine; the real smoke is a Windows gate. | 1,308 |
| [tests/expand/test_e01_table_unit.py](technical_sources/PYTHON_SOURCE_19.md#py-217) | 회귀 검증: test_config_rejects_unsupported_modes_and_amounts, test_initial_bootstrap_fixed_seats_and_cash_conservation, test_join_during | 9,209 |
| [tests/expand/test_e02_adapter_fake.py](technical_sources/PYTHON_SOURCE_19.md#py-218) | 회귀 검증: test_e02_tracker_carries_street_across_actions, test_e02_rejected_action_does_not_corrupt_tracker, test_e02_external_action_d | 3,925 |
| [tests/expand/test_e02_policy.py](technical_sources/PYTHON_SOURCE_19.md#py-219) | 회귀 검증: test_e02_policy_logits_gradients_and_illegal_zero, test_e02_policy_seed_and_legal_sampling, test_e02_seat_permutation_equivar | 2,999 |
| [tests/expand/test_e02_real_engine.py](technical_sources/PYTHON_SOURCE_19.md#py-220) | Only run where Base Phase3 is installed (Windows user's repository). | 2,882 |
| [tests/expand/test_e02_state_and_actions.py](technical_sources/PYTHON_SOURCE_19.md#py-221) | 회귀 검증: test_e02_only_acting_player_and_public_fields, test_e02_invariance_to_opponent_hole_and_future_deck, test_e02_street_hole_and | 7,263 |
| [tests/expand/test_e03_cli.py](technical_sources/PYTHON_SOURCE_19.md#py-222) | 회귀 검증: test_e03_cli_has_safe_defaults_and_restricted_resume, test_e03_custom_curriculum | 883 |
| [tests/expand/test_e03_real_engine.py](technical_sources/PYTHON_SOURCE_19.md#py-223) | Real 2-6 seat engine integration: deferred until the Windows full repo. | 999 |
| [tests/expand/test_e03_returns_and_policy.py](technical_sources/PYTHON_SOURCE_19.md#py-224) | 회귀 검증: test_population_probabilities_and_guards, test_rollout_all_2_to_6_and_reward_assignment, test_missing_terminal_rejected, test | 3,915 |
| [tests/expand/test_e03_trainer.py](technical_sources/PYTHON_SOURCE_19.md#py-225) | 회귀 검증: test_e03_checkpoint_resume_snapshot_validation, test_e03_resume_exact_reproducible, test_e03_checkpoint_tamper_rejected | 3,203 |
| [tests/expand/test_e04_checkpoint.py](technical_sources/PYTHON_SOURCE_19.md#py-226) | 회귀 검증: test_resume_exact_weights_and_round, test_checkpoint_roundtrip_leduc, test_checksum_detects_tampering, test_checkpoint_incomp | 2,163 |
| [tests/expand/test_e04_cli.py](technical_sources/PYTHON_SOURCE_19.md#py-227) | 회귀 검증: test_cli_kuhn_smoke_resume, test_cli_leduc_smoke, test_cli_budget_limit_saves_empty_model | 1,169 |
| [tests/expand/test_e04_holdem_mock.py](technical_sources/PYTHON_SOURCE_19.md#py-228) | 회귀 검증: test_real_interface_on_fast_deterministic_engine, test_private_rival_cards_do_not_affect_hero_infoset, test_postflop_card_buc | 2,673 |
| [tests/expand/test_e04_real_engine.py](technical_sources/PYTHON_SOURCE_20.md#py-229) | Testable on Windows after installing actual Base Phase 3 engine. | 1,393 |
| [tests/expand/test_e04_toy_mccfr.py](technical_sources/PYTHON_SOURCE_20.md#py-230) | 회귀 검증: test_toy_terminal_and_reward_zero_sum, test_mccfr_2_and_3_player_updates, test_infosets_hide_other_hole_and_future_community, | 4,100 |
| [tests/expand/test_e05_beliefs.py](technical_sources/PYTHON_SOURCE_20.md#py-231) | 회귀 검증: test_toy_belief_fix_hero_and_exclude_collisions, test_replay_history_only_public_actions, test_reject_impossible_trace | 1,406 |
| [tests/expand/test_e05_cli.py](technical_sources/PYTHON_SOURCE_20.md#py-232) | 회귀 검증: test_cli_toy_json_and_no_overwrite | 595 |
| [tests/expand/test_e05_continuations.py](technical_sources/PYTHON_SOURCE_20.md#py-233) | 회귀 검증: test_all_continuations_have_valid_legal_distribution, test_fivefold_weighting_is_over_original_mass, test_raise_bias_affects_ | 1,417 |
| [tests/expand/test_e05_evaluate.py](technical_sources/PYTHON_SOURCE_20.md#py-234) | 회귀 검증: test_evaluation_not_modifying_checkpoint | 757 |
| [tests/expand/test_e05_holdem_deck.py](technical_sources/PYTHON_SOURCE_20.md#py-235) | 회귀 검증: test_card_layout_exact_hero_visible_board_and_no_collision, test_card_collisions_and_bad_hole_rejected | 1,611 |
| [tests/expand/test_e05_nested.py](technical_sources/PYTHON_SOURCE_20.md#py-236) | 회귀 검증: test_off_tree_re_root_and_committed_own_actions, test_reject_bad_off_tree_size | 885 |
| [tests/expand/test_e05_real_engine.py](technical_sources/PYTHON_SOURCE_20.md#py-237) | Runs only on user's real base Phase3 repository install. | 2,180 |
| [tests/expand/test_e05_solver.py](technical_sources/PYTHON_SOURCE_20.md#py-238) | 회귀 검증: test_valid_probabilities_and_blueprint_unchanged, test_reproducible_with_seed_and_large_budget, test_node_cap_fallback_to_blu | 2,866 |
| [tests/expand/test_e06_adaptive.py](technical_sources/PYTHON_SOURCE_20.md#py-239) | 회귀 검증: test_no_evidence_preserves_base, test_low_samples_no_adapt, test_trust_kl_tv_bounds_and_probability_simplex, test_zero_suppor | 3,226 |
| [tests/expand/test_e06_bridge.py](technical_sources/PYTHON_SOURCE_20.md#py-240) | 회귀 검증: test_positions_six_and_four, test_prior_raises_counts_exact_raise_to, test_registry_tracks_join_tenures_from_snapshot, test_a | 2,084 |
| [tests/expand/test_e06_cli.py](technical_sources/PYTHON_SOURCE_21.md#py-241) | 회귀 검증: test_synthetic_blocked_benchmark, test_determinism | 991 |
| [tests/expand/test_e06_models.py](technical_sources/PYTHON_SOURCE_21.md#py-242) | 회귀 검증: test_spr_buckets_and_validation, test_session_isolation_across_same_seat, test_no_seat_lease_overlap, test_legal_context_and_ | 4,350 |
| [tests/expand/test_e06_persistence.py](technical_sources/PYTHON_SOURCE_21.md#py-243) | 회귀 검증: test_checkpoint_roundtrip_resume, test_checkpoint_tamper_detection, test_checkpoint_contains_no_card_or_deck | 1,702 |
| [tests/expand/test_e06_real_engine.py](technical_sources/PYTHON_SOURCE_21.md#py-244) | Runs only once Base Phase 3 engine is present in user's real repository. | 1,922 |
| [tests/expand/test_e06_router.py](technical_sources/PYTHON_SOURCE_21.md#py-245) | 회귀 검증: test_router_uses_different_experts_for_styles, test_router_uncertain_fallback | 1,797 |
| [tests/expand/test_e06_state_bridge.py](technical_sources/PYTHON_SOURCE_21.md#py-246) | Public-information bridge test: Phase2 private hero hole must never pass to Phase6. | 1,643 |
| [tests/expand/test_e07_agent.py](technical_sources/PYTHON_SOURCE_21.md#py-247) | 회귀 검증: test_e07_uniform_reproducible_and_legal, test_e07_priority_and_fallback, test_e07_unverified_abstraction_never_executed, test | 3,183 |
| [tests/expand/test_e07_benchmark.py](technical_sources/PYTHON_SOURCE_21.md#py-248) | 회귀 검증: test_e07_cash_scenarios_and_independent_clustering, test_e07_cash_paired_rejects_different_seed, test_e07_tournament_sessions | 1,777 |
| [tests/expand/test_e07_bridges_and_cli.py](technical_sources/PYTHON_SOURCE_21.md#py-249) | 회귀 검증: test_e07_search_source_rejects_timeout_without_mask_override, test_e07_search_source_rejects_nonmatching_action_support, test | 2,195 |
| [tests/expand/test_e07_cash_bridge.py](technical_sources/PYTHON_SOURCE_21.md#py-250) | 회귀 검증: test_e07_e1_e2_e6_accepts_public_actions | 584 |
| [tests/expand/test_e07_manifest.py](technical_sources/PYTHON_SOURCE_21.md#py-251) | 회귀 검증: test_e07_manifest_checkpoint_hash_and_no_overwrite | 847 |
| [tests/expand/test_e07_real_engine.py](technical_sources/PYTHON_SOURCE_21.md#py-252) | Gated: run only when Base Phase3 engine is installed in user's repository. | 1,008 |
| [tests/expand/test_e07_tournament.py](technical_sources/PYTHON_SOURCE_22.md#py-253) | 회귀 검증: test_e07_tournament_knockouts_chip_ledger_and_prizes, test_e07_reentry_new_id_late_registration, test_e07_tournament_guard_ba | 2,716 |
| [tests/test_blind_manager.py](technical_sources/PYTHON_SOURCE_22.md#py-254) | 회귀 검증: test_blind_levels_repeat_last_level, test_invalid_blinds, test_blind_hand_number_validation | 1,089 |
| [tests/test_corrected_rlcard_integration.py](technical_sources/PYTHON_SOURCE_22.md#py-255) | End-to-end corrected engine tests; run with RLCard 1.2.0 installed. | 2,786 |
| [tests/test_corrected_round.py](technical_sources/PYTHON_SOURCE_22.md#py-256) | Independent betting scenarios requiring no RLCard installation. | 3,360 |
| [tests/test_gui10_hunl_contract.py](technical_sources/PYTHON_SOURCE_22.md#py-257) | GUI-1.0 behavior, privacy and action compatibility with the real HUNL200 API. | 7,544 |
| [tests/test_gui11_http_api.py](technical_sources/PYTHON_SOURCE_22.md#py-258) | GUI-1.1 HTTP contract tests (requires GUI-1.0 + FastAPI/httpx). | 9,066 |
| [tests/test_gui12_readonly_api.py](technical_sources/PYTHON_SOURCE_22.md#py-259) | GUI-1.2 read-only display contract against the real GUI-1.1 API. | 2,754 |
| [tests/test_gui13_interactive_betting.py](technical_sources/PYTHON_SOURCE_22.md#py-260) | GUI-1.3: actual engine transitions through browser-facing HTTP contracts. | 4,882 |
| [tests/test_gui14_trained_agents.py](technical_sources/PYTHON_SOURCE_22.md#py-261) | GUI-1.4 local trained-agent selection, loading, privacy, and HTTP regression. | 7,689 |
| [tests/test_gui15_session_analytics.py](technical_sources/PYTHON_SOURCE_22.md#py-262) | GUI-1.5: completed-hand ledger, summary, privacy, pagination, API isolation. | 8,219 |
| [tests/test_gui16_continuous_stack.py](technical_sources/PYTHON_SOURCE_22.md#py-263) | GUI-1.6 end-to-end stack carryover, cash accounting, bust and mode isolation | 6,973 |
| [tests/test_hand_engine.py](technical_sources/PYTHON_SOURCE_22.md#py-264) | Offline tests for the public adapter API with a stand-in RLCard Env. | 2,191 |
| [tests/test_multi_player_engine.py](technical_sources/PYTHON_SOURCE_23.md#py-265) | 회귀 검증: test_preflop_and_postflop_order_three_players, test_six_players_betting_order, test_headsup_compatible_and_private_hole_cards | 11,045 |
| [tests/test_multi_player_seats.py](technical_sources/PYTHON_SOURCE_23.md#py-266) | 회귀 검증: test_heads_up_seat_rule_and_sparse_empty_seats, test_three_and_six_players, test_rotate_button_skips_empty_seats, test_invali | 1,202 |
| [tests/test_phase10_cli.py](technical_sources/PYTHON_SOURCE_23.md#py-267) | End-to-end Phase 10 CLI on CPU with mock Phase3 integration fixture. | 2,073 |
| [tests/test_phase10_gpu.py](technical_sources/PYTHON_SOURCE_23.md#py-268) | GTX1070 smoke; skipped in environments without PyTorch CUDA. | 907 |
| [tests/test_phase10_qboost.py](technical_sources/PYTHON_SOURCE_23.md#py-269) | Phase 10: algebra, information boundary, policy-gradient and integration. | 11,558 |
| [tests/test_phase11_batched.py](technical_sources/PYTHON_SOURCE_23.md#py-270) | 회귀 검증: test_batched_rows_are_legal_and_seat_specific, test_batch_one_matches_legacy_sequential, test_batched_reproducible_and_change | 3,130 |
| [tests/test_phase11_gpu.py](technical_sources/PYTHON_SOURCE_23.md#py-271) | CUDA-only batched rollout integration, runs on the user's GTX 1070. | 863 |
| [tests/test_phase11_progress.py](technical_sources/PYTHON_SOURCE_23.md#py-272) | Progress UI is an observer: no dependency on the poker engine or CUDA. | 5,397 |
| [tests/test_phase11_runner.py](technical_sources/PYTHON_SOURCE_23.md#py-273) | 회귀 검증: test_small_real_update_and_resume, test_compare_reports_after_two_run_types, test_budget_and_config_validation, test_tiny_dur | 4,381 |
| [tests/test_phase12_cli.py](technical_sources/PYTHON_SOURCE_23.md#py-274) | 회귀 검증: test_local_checkpoint_and_json_csv, test_phase11_anchor_and_paired_control, test_reject_same_input_file, test_reject_bad_chec | 2,673 |
| [tests/test_phase12_evaluation.py](technical_sources/PYTHON_SOURCE_23.md#py-275) | Phase12 robust evaluation: same-deck, opposite-seat, unit tests. | 5,558 |
| [tests/test_phase12_gpu.py](technical_sources/PYTHON_SOURCE_23.md#py-276) | GPU smoke on user's GTX1070; deliberately skipped on CPU-only runners. | 667 |
| [tests/test_phase13_gpu.py](technical_sources/PYTHON_SOURCE_24.md#py-277) | Actual CUDA run, skipped in environments without a CUDA GPU. | 800 |
| [tests/test_phase13_scaling.py](technical_sources/PYTHON_SOURCE_24.md#py-278) | Phase 13 backward-compatible workflow tests (real model and simulator contract). | 9,177 |
| [tests/test_phase14_gpu.py](technical_sources/PYTHON_SOURCE_24.md#py-279) | A tiny live CUDA end-to-end check for the user's Windows GPU. | 1,124 |
| [tests/test_phase14_study.py](technical_sources/PYTHON_SOURCE_24.md#py-280) | Phase14 correctness, matched-start and held-out test protection. | 5,973 |
| [tests/test_phase15_critic_diagnostics.py](technical_sources/PYTHON_SOURCE_24.md#py-281) | Tests for Phase-15 read-only critic calibration and numerical edge cases. | 6,084 |
| [tests/test_phase15_gpu.py](technical_sources/PYTHON_SOURCE_24.md#py-282) | Real CUDA invocation: only runs on CUDA-enabled hosts. | 1,352 |
| [tests/test_phase15_initial_policy_comparison.py](technical_sources/PYTHON_SOURCE_24.md#py-283) | Agent baseline all-in comparison regression tests (no environment edits). | 6,925 |
| [tests/test_phase15_longrun.py](technical_sources/PYTHON_SOURCE_24.md#py-284) | Phase15 safety, resume, validation selection, held-out sealing and CLI. | 9,764 |
| [tests/test_phase15_policy_audit.py](technical_sources/PYTHON_SOURCE_24.md#py-285) | Regression tests for Phase 15 *read-only* policy/optimizer diagnostics. | 3,422 |
| [tests/test_phase15_preflop_sequence_audit.py](technical_sources/PYTHON_SOURCE_24.md#py-286) | Phase15 agent-only preflop sequence audit checks. | 5,009 |
| [tests/test_phase15_safety_phase3.py](technical_sources/PYTHON_SOURCE_24.md#py-287) | Phase 15 phase-3 safety and reward diagnostics regression tests. | 2,072 |
| [tests/test_phase15_terminal_reward_audit.py](technical_sources/PYTHON_SOURCE_24.md#py-288) | Regression tests: hand-weighted payoff tails and preflop termination types. | 5,242 |
| [tests/test_phase16_cli.py](technical_sources/PYTHON_SOURCE_25A.md#py-289) | 회귀 검증: test_cli_smoke_creates_json_csv_without_training, test_cli_named_checkpoint_and_invalid_spec | 2,192 |
| [tests/test_phase16_diagnostics.py](technical_sources/PYTHON_SOURCE_25A.md#py-290) | 회귀 검증: test_effective_spr_public_observable, test_conditional_rate_not_raw_rate_and_strata, test_illegal_selection_rejected, test_en | 3,119 |
| [tests/test_phase16_evaluation.py](technical_sources/PYTHON_SOURCE_25A.md#py-291) | 회귀 검증: test_multiplayer_pool_reproducible_and_no_mutation, test_empty_and_duplicate_opponents_rejected, test_invalid_config, test_ne | 3,714 |
| [tests/test_phase16_opponents.py](technical_sources/PYTHON_SOURCE_25A.md#py-292) | 회귀 검증: test_opponents_choose_legal_actions, test_behavior_semantics, test_invalid_builtins, test_checkpoint_name_allowed, test_facto | 1,570 |
| [tests/test_phase16_ppo_ablation.py](technical_sources/PYTHON_SOURCE_25A.md#py-293) | Phase16 isolated PPO ablations: objectives, KL, checkpoints and safety. | 7,515 |
| [tests/test_phase16_ppo_stress.py](technical_sources/PYTHON_SOURCE_25A.md#py-294) | Phase16 A0/A3 long-stress regression and real-HUNL mini integration tests. | 9,100 |
| [tests/test_phase17_continuous_stack.py](technical_sources/PYTHON_SOURCE_25B.md#py-295) | Phase17 integration regression against the real GUI-1.6 hand adapter. | 7,905 |
| [tests/test_phase17_gpu_retrain.py](technical_sources/PYTHON_SOURCE_25B.md#py-296) | Phase17 5-hour PPO runner: immutable source, full resume and KL safeguards. | 7,873 |
| [tests/test_phase5_betting.py](technical_sources/PYTHON_SOURCE_25B.md#py-297) | 회귀 검증: test_street_raise_targets_are_integer_post_call_pot, test_raise_rounds_up_to_min_and_deduplicates, test_short_all_in_allowed_ | 3,059 |
| [tests/test_phase5_numpy_json.py](technical_sources/PYTHON_SOURCE_25B.md#py-298) | Regression tests: RLCard public state may include nested NumPy scalars. | 2,125 |
| [tests/test_phase5_rewards.py](technical_sources/PYTHON_SOURCE_25B.md#py-299) | 회귀 검증: test_cash_zero_sum_in_bb_units, test_chip_deltas_six_seat_and_bust, test_chip_delta_chips_not_conserved, test_bad_scale_and_s | 1,406 |
| [tests/test_phase5_rlcard_adapter.py](technical_sources/PYTHON_SOURCE_25B.md#py-300) | 회귀 검증: test_observation_never_exposes_opponents_hidden_cards_or_engine_data, test_rlcard_adapter_seed_masks_and_selfplay, test_rlcar | 5,089 |
| [tests/test_phase5_schema_policy.py](technical_sources/PYTHON_SOURCE_26.md#py-301) | 회귀 검증: test_infoset_key_stable_and_private_distinguished, test_legal_mask_rejects_illegal_and_type, test_nonterminal_must_have_legal | 3,631 |
| [tests/test_phase5_self_play.py](technical_sources/PYTHON_SOURCE_26.md#py-302) | 회귀 검증: test_selfplay_6_max_trajectories_link_next_same_seat, test_deterministic_selfplay_actions_seed, test_selfplay_missing_policy, | 3,544 |
| [tests/test_phase6_benchmark.py](technical_sources/PYTHON_SOURCE_26.md#py-303) | Small speed-test smoke; does not assert hardware-dependent throughput. | 640 |
| [tests/test_phase6_hunl_adapter.py](technical_sources/PYTHON_SOURCE_26.md#py-304) | Integration tests run against the actual Phase 3 engine in the repository. | 6,125 |
| [tests/test_phase7_encoding.py](technical_sources/PYTHON_SOURCE_26.md#py-305) | Phase 7 contracts: actor privacy, history, stable cards, critic feature schema. | 6,764 |
| [tests/test_phase7_rollout.py](technical_sources/PYTHON_SOURCE_26.md#py-306) | End-to-end HUNL200 -> global episodes -> actor/critic tensors | 1,772 |
| [tests/test_phase7_tensors_actions.py](technical_sources/PYTHON_SOURCE_26.md#py-307) | Action legality, padding masks, numeric tensors and gradients (GPU optional). | 3,709 |
| [tests/test_phase8_gpu.py](technical_sources/PYTHON_SOURCE_26.md#py-308) | Runs a minimal real Phase-7 batch on CUDA when available (e.g | 1,530 |
| [tests/test_phase8_networks.py](technical_sources/PYTHON_SOURCE_26.md#py-309) | Phase-8 independent unit tests using real Phase-7 batch structures. | 9,837 |
| [tests/test_phase9_cli.py](technical_sources/PYTHON_SOURCE_27.md#py-310) | Phase-9 executable CLI and checkpoint interface, CPU-only fast smoke. | 1,112 |
| [tests/test_phase9_gpu.py](technical_sources/PYTHON_SOURCE_27.md#py-311) | Run on user's GTX1070; skip on the CPU-only packaging machine. | 1,296 |
| [tests/test_phase9_ppo.py](technical_sources/PYTHON_SOURCE_27.md#py-312) | Phase-9 unit/integration tests against real HUNL200 engine in the user's repo. | 8,613 |
| [tests/test_player_manager.py](technical_sources/PYTHON_SOURCE_27.md#py-313) | 회귀 검증: test_elimination_and_pre_hand_stack_tiebreak, test_different_prehand_stacks_same_hand_bust, test_equal_start_busts_joint_rank | 1,675 |
| [tests/test_rewards_tournament.py](technical_sources/PYTHON_SOURCE_27.md#py-314) | 회귀 검증: test_chip_rewards_and_winner_take_all, test_joint_placing_split_prize_slots, test_invalid_rank_sequences, test_invalid_payout | 836 |
| [tests/test_rlcard_integration.py](technical_sources/PYTHON_SOURCE_27.md#py-315) | Runs only where RLCard is installed; verifies a real hand end-to-end. | 543 |
| [tests/test_rlcard_rules_integration.py](technical_sources/PYTHON_SOURCE_27.md#py-316) | Checks the installed RLCard (1.2.0) rather than synthetic game objects. | 1,036 |
| [tests/test_rules_audit.py](technical_sources/PYTHON_SOURCE_27.md#py-317) | 회귀 검증: test_audit_accepts_a_correct_heads_up_implementation, test_audit_detects_a_reversed_blind_and_first_actor | 881 |
| [tests/test_tournament_env.py](technical_sources/PYTHON_SOURCE_27.md#py-318) | 회귀 검증: test_reproducible_two_player_complete_tournament, test_tournament_random_end_to_end, test_incremental_step_observation_no_pri | 8,761 |
| [tests/test_tournament_env131_aggression.py](technical_sources/PYTHON_SOURCE_28.md#py-319) | ENV 1.3.1 regression tests for multiway bot aggression and action safety. | 2,720 |
| [tests/test_tournament_env13_improved_bots.py](technical_sources/PYTHON_SOURCE_28.md#py-320) | Tournament ENV 1.3: explicit six-seat lineup with four heuristic policies. | 5,331 |
| [tests/test_tournament_env14_trained_actor.py](technical_sources/PYTHON_SOURCE_28.md#py-321) | ENV 1.4: checkpoint gating, action contract and no silent baseline fallback. | 7,407 |
| [tests/test_tournament_env15_progress.py](technical_sources/PYTHON_SOURCE_28.md#py-322) | Tournament ENV 1.5 public, read-only standings and blind level contract. | 4,064 |
| [tests/test_tournament_env161_showdown.py](technical_sources/PYTHON_SOURCE_28.md#py-323) | Phase 1.6.1: show only legitimate showdown hands, and only after settlement. | 7,741 |
| [tests/test_tournament_env16_persistence.py](technical_sources/PYTHON_SOURCE_28.md#py-324) | Phase 1.6: durable actor-visible transitions and public replay. | 6,901 |
| [tests/test_tournament_env17_integration.py](technical_sources/PYTHON_SOURCE_28.md#py-325) | Tournament ENV 1.7: release contract across both GUIs and durable replay | 4,731 |
| [tests/test_tournament_env17_release_audit.py](technical_sources/PYTHON_SOURCE_28.md#py-326) | Unit tests for the privacy/staging release gate, no project game imports. | 2,253 |
| [tests/test_tournament_env1_api.py](technical_sources/PYTHON_SOURCE_28.md#py-327) | Tournament ENV 1.1 -- API/engine integration and privacy boundary. | 10,183 |

<a name="section-27594225cc64"></a>

## 부록 B — 기존 Markdown 76개 전체 원문 색인

당시 Frontmatter·본문·수식·Python/PowerShell 코드·실험 로그·링크를 원문 그대로 포함한다. 원문 안의 과거 링크와 경로는 당시 문맥의 기록이며, 현재 읽기 경로는 이 색인에서 제공한다. 두 원 기준 보고서와 모든 세부 보고서를 여기서 찾을 수 있다.

| 당시 문서의 전체 원문 | 원본 바이트 | 정리 결과 |
|---|---:|---|
| [README.md](report_sources/ORIGINAL_REPORTS_07.md#md-001) | 6,673 | 현행 문서 재작성; 당시 원문 부록 보존 |
| [README_TOURNAMENT_ENV_1_3_1.md](report_sources/ORIGINAL_REPORTS_06B.md#md-002) | 399 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [README_TOURNAMENT_ENV_PHASE_1_2_1.md](report_sources/ORIGINAL_REPORTS_07.md#md-003) | 1,058 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [README_TOURNAMENT_ENV_PHASE_1_3.md](report_sources/ORIGINAL_REPORTS_07.md#md-004) | 610 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [README_TOURNAMENT_ENV_PHASE_1_4.md](report_sources/ORIGINAL_REPORTS_07.md#md-005) | 1,193 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [README_TOURNAMENT_ENV_PHASE_1_5.md](report_sources/ORIGINAL_REPORTS_07.md#md-006) | 1,320 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [README_TOURNAMENT_ENV_PHASE_1_6.md](report_sources/ORIGINAL_REPORTS_07.md#md-007) | 1,074 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [README_TOURNAMENT_ENV_PHASE_1_6_1.md](report_sources/ORIGINAL_REPORTS_07.md#md-008) | 1,310 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [TOURNAMENT_ENV_PHASE_1_1_README.md](report_sources/ORIGINAL_REPORTS_07.md#md-009) | 1,469 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [TOURNAMENT_ENV_PHASE_1_2_README.md](report_sources/ORIGINAL_REPORTS_07.md#md-010) | 2,642 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [TOURNAMENT_ENV_PHASE_1_7_README.md](report_sources/ORIGINAL_REPORTS_07.md#md-011) | 1,091 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/AGENT_DEVELOPMENT_TECHNICAL_REPORT_2026-10-13.md](report_sources/ORIGINAL_REPORTS_01.md#md-012) | 93,299 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/EXPAND_MASTER_DESIGN_6MAX_v1_0.md](report_sources/ORIGINAL_REPORTS_01.md#md-013) | 50,610 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/EXPAND_PHASE01_DESIGN.md](report_sources/ORIGINAL_REPORTS_01.md#md-014) | 12,011 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/EXPAND_PHASE01_TO_07_INTEGRATED_TECHNICAL_REPORT.md](report_sources/ORIGINAL_REPORTS_01.md#md-015) | 31,992 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/EXPAND_PHASE02_INFORMATION_AND_ENCODING.md](report_sources/ORIGINAL_REPORTS_01.md#md-016) | 15,623 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/EXPAND_PHASE03_MULTIAGENT_PPO.md](report_sources/ORIGINAL_REPORTS_01.md#md-017) | 15,621 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/EXPAND_PHASE04_MCCFR_BLUEPRINT.md](report_sources/ORIGINAL_REPORTS_01.md#md-018) | 14,494 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/EXPAND_PHASE05_SUBGAME_SEARCH.md](report_sources/ORIGINAL_REPORTS_01.md#md-019) | 15,379 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/EXPAND_PHASE06_OPPONENT_MODELING.md](report_sources/ORIGINAL_REPORTS_01.md#md-020) | 16,222 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/EXPAND_PHASE07_INTEGRATION_TOURNAMENT_BENCHMARK.md](report_sources/ORIGINAL_REPORTS_02.md#md-021) | 20,388 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/GUI_1_0_HUNL_CONTRACT.md](report_sources/ORIGINAL_REPORTS_02.md#md-022) | 7,076 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/GUI_1_1_FASTAPI.md](report_sources/ORIGINAL_REPORTS_02.md#md-023) | 6,246 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/GUI_1_2_REACT_TABLE.md](report_sources/ORIGINAL_REPORTS_02.md#md-024) | 4,148 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/GUI_1_3_INTERACTIVE_BETTING.md](report_sources/ORIGINAL_REPORTS_02.md#md-025) | 6,003 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/GUI_1_4_TRAINED_AGENT.md](report_sources/ORIGINAL_REPORTS_02.md#md-026) | 6,531 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/GUI_1_5_PERFORMANCE_DASHBOARD.md](report_sources/ORIGINAL_REPORTS_02.md#md-027) | 6,040 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/GUI_1_6_CONTINUOUS_STACK.md](report_sources/ORIGINAL_REPORTS_02.md#md-028) | 7,436 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/GUI_TRAINED_CHECKPOINT_SETUP.md](report_sources/ORIGINAL_REPORTS_02.md#md-029) | 5,914 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE10_QBOOST_PPO.md](report_sources/ORIGINAL_REPORTS_03.md#md-030) | 9,382 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE11_FIVE_HOUR_TRAINING.md](report_sources/ORIGINAL_REPORTS_03.md#md-031) | 11,162 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE11_PROGRESS.md](report_sources/ORIGINAL_REPORTS_03.md#md-032) | 3,865 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE12_EVALUATION.md](report_sources/ORIGINAL_REPORTS_03.md#md-033) | 8,509 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE13_BATCH_SCALING.md](report_sources/ORIGINAL_REPORTS_03.md#md-034) | 9,451 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE14_CONTROLLED_STUDY.md](report_sources/ORIGINAL_REPORTS_03.md#md-035) | 9,315 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE15_CRITIC_DIAGNOSIS.md](report_sources/ORIGINAL_REPORTS_03.md#md-036) | 4,914 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE15_EXPERIMENT_REPORT_2026-10-13.md](report_sources/ORIGINAL_REPORTS_03.md#md-037) | 16,335 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE15_LONG_RUN_TRAINING.md](report_sources/ORIGINAL_REPORTS_03.md#md-038) | 11,248 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE15_PHASE3_SAFETY_TELEMETRY.md](report_sources/ORIGINAL_REPORTS_03.md#md-039) | 3,626 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE15_PREFLOP_SEQUENCE_AUDIT.md](report_sources/ORIGINAL_REPORTS_03.md#md-040) | 3,345 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE15_TERMINAL_REWARD_AUDIT.md](report_sources/ORIGINAL_REPORTS_04.md#md-041) | 4,206 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE15_THREE_POLICY_COMPARISON.md](report_sources/ORIGINAL_REPORTS_04.md#md-042) | 3,940 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE16_LONG_STRESS_TEST.md](report_sources/ORIGINAL_REPORTS_04.md#md-043) | 9,005 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE16_OPPONENT_POOL_AND_STRATEGY.md](report_sources/ORIGINAL_REPORTS_04.md#md-044) | 8,643 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE16_PPO_ABLATION.md](report_sources/ORIGINAL_REPORTS_04.md#md-045) | 8,324 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE17_CONTINUOUS_STACK_EVALUATION.md](report_sources/ORIGINAL_REPORTS_04.md#md-046) | 8,630 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE17_FIVE_HOUR_GPU_RETRAIN.md](report_sources/ORIGINAL_REPORTS_04.md#md-047) | 7,944 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE5_AGENT_INTERFACE.md](report_sources/ORIGINAL_REPORTS_05.md#md-048) | 11,848 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE6_HUNL200_INTEGRATION.md](report_sources/ORIGINAL_REPORTS_05.md#md-049) | 6,229 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE7_VRPO_LITE_ENCODING.md](report_sources/ORIGINAL_REPORTS_05.md#md-050) | 7,323 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE8_VRPO_LITE_NETWORKS.md](report_sources/ORIGINAL_REPORTS_05.md#md-051) | 6,923 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE9_PPO_BASELINE.md](report_sources/ORIGINAL_REPORTS_05.md#md-052) | 8,515 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PHASE_VERIFIED_PYTHON_CODE_REFERENCE.md](report_sources/ORIGINAL_REPORTS_02.md#md-053) | 14,753 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md](report_sources/ORIGINAL_REPORTS_05.md#md-054) | 24,744 | 현행 문서 재작성; 당시 원문 부록 보존 |
| [docs/PROJECT_PROGRESS_GUI16_PHASE17.md](report_sources/ORIGINAL_REPORTS_05.md#md-055) | 16,132 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/PROJECT_REPORT.md](report_sources/ORIGINAL_REPORTS_05.md#md-056) | 28,236 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/ROADMAP_PHASE12_TO_16.md](report_sources/ORIGINAL_REPORTS_05.md#md-057) | 3,492 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/TECH_REPORT.md](report_sources/ORIGINAL_REPORTS_05.md#md-058) | 2,369 | 현행 문서 재작성; 당시 원문 부록 보존 |
| [docs/TOURNAMENT_ENV_PHASE1_FULL_TECHNICAL_REPORT.md](report_sources/ORIGINAL_REPORTS_06B.md#md-059) | 52,111 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/TOURNAMENT_ENV_PHASE_1_1.md](report_sources/ORIGINAL_REPORTS_06A.md#md-060) | 6,122 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/TOURNAMENT_ENV_PHASE_1_1_KO.md](report_sources/ORIGINAL_REPORTS_05.md#md-061) | 4,487 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/TOURNAMENT_ENV_PHASE_1_2.md](report_sources/ORIGINAL_REPORTS_06A.md#md-062) | 4,703 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/TOURNAMENT_ENV_PHASE_1_3.md](report_sources/ORIGINAL_REPORTS_06A.md#md-063) | 5,192 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/TOURNAMENT_ENV_PHASE_1_3_1_AGGRESSION_FIX.md](report_sources/ORIGINAL_REPORTS_06A.md#md-064) | 3,715 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/TOURNAMENT_ENV_PHASE_1_4.md](report_sources/ORIGINAL_REPORTS_06A.md#md-065) | 6,534 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/TOURNAMENT_ENV_PHASE_1_5.md](report_sources/ORIGINAL_REPORTS_06B.md#md-066) | 7,204 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/TOURNAMENT_ENV_PHASE_1_6.md](report_sources/ORIGINAL_REPORTS_06B.md#md-067) | 6,756 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/TOURNAMENT_ENV_PHASE_1_6_1_SHOWDOWN.md](report_sources/ORIGINAL_REPORTS_06B.md#md-068) | 4,890 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/TOURNAMENT_ENV_PHASE_1_7_FINAL.md](report_sources/ORIGINAL_REPORTS_06B.md#md-069) | 10,078 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/architecture.md](report_sources/ORIGINAL_REPORTS_01.md#md-070) | 601 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/archive/INTEGRATED_SOURCE_SNAPSHOT_01.md](report_sources/ORIGINAL_REPORTS_01.md#md-071) | 298,011 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/phase1_rlcard.md](report_sources/ORIGINAL_REPORTS_02.md#md-072) | 1,393 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/phase2_1_correction.md](report_sources/ORIGINAL_REPORTS_04.md#md-073) | 2,474 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/phase2_rule_validation.md](report_sources/ORIGINAL_REPORTS_04.md#md-074) | 3,369 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/phase3_multiplayer.md](report_sources/ORIGINAL_REPORTS_04.md#md-075) | 3,006 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |
| [docs/phase4_tournament.md](report_sources/ORIGINAL_REPORTS_04.md#md-076) | 2,638 | 독립 문서 삭제; 전체 내용 최종보고서 부록으로 통합 |

[파일별 SHA·바이트 수·부록 위치·정리 목록 Manifest](lossless_source_manifest.json)
