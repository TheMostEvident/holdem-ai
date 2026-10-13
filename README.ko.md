# holdem-ai

[English](README.md) | **한국어**

```math
\boxed{s_{t+1}=T(s_t,a_t,\omega_t)}
\qquad
\boxed{a_t\sim\pi_\theta(\cdot\mid I_{i,t}),\ a_t\in A_{\mathrm{legal}}(s_t)}
```

**Environment는 게임 규칙을 실행하고, Agent는 자신에게 보이는 정보로 행동을 결정한다.** 이 저장소는 Heads-Up No-Limit Hold'em, 동적 2–6인 Cash Game, 6인 Knockout Tournament의 시뮬레이션·학습·평가·GUI를 구현하는 연구 프로젝트다.

가장 먼저 [최종 통합 기술보고서](docs/POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.ko.md)를 읽는다. Phase별 목적과 효용, 수식, 실제 Python 코드, 실행 과정, 검증 결과와 남은 문제를 한곳에서 연결한다. [TECH REPORT](docs/TECH_REPORT.ko.md)는 목적별 읽기 경로와 검증 기준을 제공한다.

## Project와 Phase

| Project | Phase | 제공하는 기능 |
|---|---|---|
| Base Environment | P0–P4 | RLCard 기준 환경·Heads-up 순서 보정·독립 2–6인 엔진·Side Pot·토너먼트 진행 |
| HUNL Agent | P5–P10 | 정보집합·합법 행동·Transformer Actor/Q Critic·PPO·실험적 QBoost |
| HUNL 학습·평가 | P11–P17 | Batch·Resume·Seed-Blocked 평가·Critic/Policy 진단·Ablation·연속 스택·GPU Retrain |
| Expand | E1–E7 | 동적 다인 Table·다인 PPO·MCCFR Blueprint·제한 Search·상대 모델·통합 라우팅 |
| HUNL GUI | G1.0–G1.6 | Human–Agent 경기·FastAPI/React·Checkpoint 선택·통계·연속 스택 |
| Tournament ENV | T1.1–T1.7 | 6인 GUI·Improved Bots·다인 Frozen Actor·블라인드/순위·SQLite·Replay·Showdown 공개 |

HUNL Actor와 Expand Actor는 모델·관측·Checkpoint 계약이 다르다. Expand E3의 Cash-Game 학습 모델을 Tournament에 연결하는 기능과 Tournament 전용 정책 학습을 구분한다. MCCFR/Search는 연구 구현이며, 현재 기록만으로 HUNL 해결·GTO·Pluribus 재현·Tournament 최적 정책을 입증하지 않는다.

## 정확성의 기준

```math
\sum_i S_{i,t}^{\mathrm{behind}}+P_t=C_h,
\qquad R_{i,h}^{BB}=\frac{S_{i,h}^{\mathrm{final}}-S_{i,h}^{\mathrm{start}}}{BB_h}
```

핸드 내부의 칩 보존과 정산을 먼저 확인한다. $`R^{BB}`$는 칩 손익을 Big Blind로 나눈 값이다. 순위 보상·상금 효용은 별도의 목적함수다. 합법 행동 Mask와 Actor의 비공개 정보 경계도 모든 학습·평가·GUI에 동일하게 적용한다.

```math
\widehat{bb/100}=\frac{100}{H}\sum_{h=1}^{H}R_h^{BB}
```

성능 비교에는 Seed별 짝지은 Deck·Seat·Button 조건과 신뢰구간을 사용한다. 같은 Deck을 재사용한 경기들을 독립 표본으로 세지 않는다. Validation으로 Checkpoint를 선택한 결과와 최종 Held-Out Test를 구별한다.

## 설치와 기본 실행

`pyproject.toml`의 Python 요구조건은 **3.10 이상**이다. Windows의 기존 실행 기준은 Python 3.11이며, Frontend의 Node 요구조건은 **20.19.0 이상**이다.

```powershell
git clone https://github.com/TheMostEvident/holdem-ai.git
cd holdem-ai
uv sync --extra dev
uv run python main.py --hands 100 --seed 42
```

Windows에서 GUI·PyTorch 학습 기능까지 사용할 경우:

```powershell
uv sync --extra dev --extra gui --extra gpu
uv run --extra dev --extra gui --extra gpu python -m pytest -q
```

GPU Extra의 현재 PyTorch 설정은 Windows 및 Python 3.14 미만에 조건부로 적용된다. 다른 운영체제의 PyTorch 환경은 별도로 맞춰야 한다. 위 명령은 실행 절차이며, 이번 문서 정리에서 전체 게임·학습 테스트를 다시 실행했다는 뜻은 아니다.

## GUI 실행

Repository 루트의 터미널에서 Backend를 실행한다.

```powershell
.\scripts\start_gui_with_models.ps1
```

다른 터미널에서 Frontend를 실행한다.

```powershell
cd frontend
npm.cmd ci
npm.cmd run typecheck
npm.cmd test
npm.cmd run build
npm.cmd run dev
```

실제 학습 모델을 지정하려면 로컬의 해당 모델 계열 `best.pt`를 전달한다.

```powershell
.\scripts\start_gui_with_models.ps1 `
  -PpoCheckpoint ".\runs\my_hunl_run\best.pt" `
  -TournamentCheckpoint ".\runs\my_expand_run\best.pt" `
  -Device cpu `
  -PolicyMode sampled
```

`my_hunl_run`과 `my_expand_run`은 실제 로컬 실행 폴더로 바꾼다. 체크포인트 파일은 저장소에 포함되지 않는다. 모델을 변경하면 Backend를 재시작하고 새 세션을 생성한다. 시작 스크립트는 명시 경로·환경변수·후보 폴더 순서로 파일을 찾으며, `last.pt`/`latest.pt` 입력은 같은 Run의 `best.pt`로 연결한다. 모델 호환성은 실제 게임 시작 시 검사된다.

## 전체 원문과 개발 자료

| 자료 | 범위 | 읽는 곳 |
|---|---|---|
| 최종 통합 본문 | 모든 Project·Phase의 수식·구현·실행 설명 | [통합 기술보고서](docs/POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.ko.md) |
| Python 원문 부록 | **327개 전체 파일**, CLI·패키지·테스트 포함 | 보고서의 Python 원문 색인 및 `docs/technical_sources/` |
| 기존 문서 원문 부록 | **76개 전체 문서**, 코드·수식·로그 포함 | 보고서의 Markdown 원문 색인 및 `docs/report_sources/` |
| 원문 검증 명세 | 파일 경로·SHA·바이트 수·부록 위치 | [lossless_source_manifest.json](docs/lossless_source_manifest.json) |

기준 커밋 `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`의 전체 원문은 **Python 327/327, Markdown 76/76** Git Blob SHA로 검증했다. 기존 독립 보고서 73개는 내용을 이 최종보고서의 부록으로 통합한 뒤 삭제했다. 실행 코드·설정·Frontend 파일은 문서 정리 대상에 포함하지 않았다.

## 현재 실험 결과를 해석하는 기준

P17 GPU Retrain의 당시 기록은 **5,000 accepted updates / 640,000 hands / 12,743.274 active seconds**다. 18,000초 목표 전에 `max_updates_ceiling`으로 종료했으며 `complete=false`였다. 이는 독립 Held-Out 성능이나 5시간 학습 완료를 증명하지 않는다.

다음 개발 판단은 실제 엔진의 규칙·정보 경계, PPO의 학습 안정성, 독립 평가, Expand의 실제 다인 엔진 연결, Tournament 전용 학습을 각각 검증하면서 진행한다. 자세한 판단 근거와 Phase별 코드는 최종 기술보고서에 있다.
