---
title: Poker AI — Technical Report Index
date: 2026-10-13
status: lossless-source-verified
---

# TECH REPORT — 목적별 읽기 경로

$$
\boxed{\text{게임 규칙 }T\ \longrightarrow\ \text{관측 }I\ \longrightarrow\ \text{정책 }\pi_\theta\ \longrightarrow\ \text{행동 }a\ \longrightarrow\ \text{정산 }u}
$$

게임 규칙이 상태를 만들고, 관측이 플레이어의 정보 범위를 제한하며, 정책이 합법 행동을 선택하고, 정산이 학습·평가의 보상을 제공한다. 이 문서는 [최종 통합 기술보고서](POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md)의 진입점이다. 모든 Phase의 상세 설명과 전체 원문은 최종 보고서 및 그 부록에 연결돼 있다.

## 목적별 Project·Phase

| 알고 싶은 것 | 읽을 Project·Phase | 핵심 확인 사항 |
|---|---|---|
| 게임 규칙과 상태 전이 | Base Environment P0–P4 | 카드·행동 순서·Raise·Side Pot·Showdown·칩 보존 |
| 2인 Actor와 Critic 구현 | HUNL Agent P5–P8 | InformationState·7-action Mask·Token Encoder·Transformer·정보 경계 |
| PPO와 QBoost의 차이 | HUNL Agent P9–P10 | Terminal MC Return·Old Policy·Clipped Loss·Control Variate |
| 학습 효율과 평가 설계 | HUNL Agent P11–P14 | Batched Collector·시간 예산·Checkpoint·Seed Block·Held-Out |
| Policy Collapse·Critic 문제 | HUNL Agent P15–P16 | 행동 분모·보상·KL·A0–A3·Stress·Rollback |
| 연속 스택과 재학습 | HUNL Agent P17a/P17b | 평가 인터페이스와 GPU 학습 실험 분리 |
| 2–6인 Cash Table | Expand E1–E3 | 핸드 경계 이벤트·변동 좌석·다인 관측·PPO/GAE |
| CFR·Search·상대 적응 | Expand E4–E7 | External Sampling·Belief·Budget·Policy Routing·실제 엔진 검증 |
| Human–HUNL Agent GUI | HUNL GUI G1.0–G1.6 | Frozen Actor·API Revision·Statistics·Stack Carry |
| 6인 Tournament GUI | Tournament ENV T1.1–T1.7 | Bots·Actor Bridge·탈락 순위·Blind Level·SQLite·Replay·카드 공개 |
| 코드 또는 문서 전체 원문 | 최종 보고서의 원문 색인 | Python 327개·Markdown 76개를 파일별로 열기 |

## 각 Phase를 읽는 순서

$$
\boxed{\text{수식}\rightarrow\text{변수}\rightarrow\text{목적·효용}\rightarrow\text{실제 코드}\rightarrow\text{실행 과정}\rightarrow\text{검증·한계}}
$$

수식은 구현해야 할 계약을 정의한다. 변수 설명은 어떤 데이터가 필요한지 알려준다. 목적은 그 Phase가 해결하는 문제를 명시한다. 실제 코드와 호출 순서는 입력이 결과로 바뀌는 과정을 보여준다. 검증과 한계는 구현·과거 테스트 기록·실전 강도를 구분한다.

Base P9/P10은 Terminal Monte-Carlo Return을 사용하는 HUNL 학습이고, Expand E3의 GAE 학습과 구별한다. P17a는 연속 스택 평가이며 P17b는 GPU Retrain이다. Expand E3의 Tournament 연결은 Cash-Game Checkpoint의 Frozen Inference다.

## 공통 수학 계약

$$
s_{t+1}=T(s_t,a_t,\omega_t),\quad a_t\in A_{\mathrm{legal}}(s_t)
$$

Environment는 합법 행동으로 상태를 전이하고, 카드 배분 확률 사건 $\omega_t$를 별도로 처리한다.

$$
\pi_\theta(a\mid I)=\frac{\mathbf1[a\in A(I)]e^{\ell_\theta(I,a)}}{\sum_{b\in A(I)}e^{\ell_\theta(I,b)}}
$$

Actor는 자신의 정보집합 $I$와 합법 행동 Mask로 확률을 만든다. Training-Only Critic의 비공개 정보가 Actor의 추론 입력으로 넘어가지 않아야 한다.

$$
\sum_i S_{i,t}^{\mathrm{behind}}+P_t=C_h,\quad
R_{i,h}^{BB}=\frac{S_{i,h}^{\mathrm{final}}-S_{i,h}^{\mathrm{start}}}{BB_h},\quad
\widehat{bb/100}=\frac{100}{H}\sum_h R_{i,h}^{BB}
$$

레이크·핸드 중 외부 자금 이동이 없는 경우 칩은 보존된다. $BB$로 정규화한 수익과 순위 보상을 구분하며, 평가의 독립 단위는 실험 설계의 Seed Block에 맞춘다.

## 전체 원문과 무손실 검증

$$
\operatorname{GitBlobSHA}(b)=\operatorname{SHA1}(\texttt{blob }\Vert\operatorname{decimal}(|b|)\Vert\texttt{NUL}\Vert b)
$$

$b$는 원본 파일의 UTF-8 바이트다. 원문 뒤의 Fence 구분용 LF를 제외하고 길이와 SHA를 계산한다. 경로별 원문이 한 번씩 들어 있고, 바이트 수와 SHA가 원본 트리와 같아야 한다.

| 검증 항목 | 결과 |
|---|---:|
| Python 원문 | 327/327 파일 · 1,400,837바이트 |
| Markdown 원문 | 76/76 파일 · 1,049,564바이트 |
| 누락·중복·원문 SHA 불일치 | 0 |
| 기존 독립 보고서 정리 | 73개 삭제; 원문은 최종보고서 부록에 포함 |
| Python 부록 | 28 논리 권 / 29 파일; 25권은 A·B 분할 |
| Markdown 부록 | 7 논리 권 / 8 파일; 6권은 A·B 분할 |

원문 기준은 `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`이다. [Manifest](lossless_source_manifest.json)는 모든 원본 경로·SHA·바이트 수·부록 경로·Anchor와 삭제 목록을 제공한다. 최종보고서에는 동일 검증을 재현할 수 있는 Python 코드가 있다.

이번 문서 작업에서 실행한 것은 원문·문법·링크·변경 범위 검증이다. 기존 문서의 Pytest·Frontend·학습 수치는 당시 기록이며, 이번에 해당 학습과 전체 런타임 테스트를 다시 수행한 결과가 아니다.
