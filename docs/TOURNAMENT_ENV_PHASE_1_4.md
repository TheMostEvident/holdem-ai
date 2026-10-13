# Tournament ENV Phase 1.4 — Trained Multi-seat Actor Connection

> 환경(Environment) 전용 개발. Agent의 PPO 학습, 가중치 갱신 및 강화학습 최적화 코드는 수정하지 않는다.

## 1. 개발 목표

Phase 1.3의 6인 테이블 구성에서 **Seat 2(physical seat index 1)**의 임시 Heuristic Policy를 선택적으로 실제 학습된 2~6인 PPO 정책에 연결한다.

| 물리적 Seat | 역할 | 정책 |
| --- | --- | --- |
| 0 | Human | GUI 직접 베팅 |
| 1 | Tournament Agent | `Expand Phase3 PPO` 체크포인트 추론, 또는 명시적 미학습 Heuristic Placeholder |
| 2 | Improved Bot 1 | Tight Aggressive Heuristic |
| 3 | Improved Bot 2 | Balanced Value Heuristic |
| 4 | Improved Bot 3 | Pot Odds Heuristic |
| 5 | Improved Bot 4 | Controlled Pressure Heuristic |

**주의:** 현재 사용 가능한 `runs/expand_phase03_pilot/best.pt`는 **2~6인 캐시게임** 학습 실험의 파일이다. 토너먼트용 Actor를 학습해둔 파일이라고 주장할 수 없다. 이번 연결은 **Cash-Game → Tournament Transfer Inference**에 해당하며, 파산·블라인드 상승·잔여 인원 변화 등 학습 분포 밖 상태에서의 성능은 검증되지 않았다. `runs/phase14_pilot/.../qboost/best.pt` 같은 HUNL200 전용 2인 체크포인트는 형식 불일치로 **명시적으로 거부**한다.

## 2. 실행 방법 (Windows PowerShell)

프로젝트 루트:

```powershell
cd C:\Users\yiydg\Desktop\workspace\holdem-ai-starter\holdem-ai
Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-4.zip" -DestinationPath . -Force

# 모델 파일이 존재하는지 확인
Test-Path .\runs\expand_phase03_pilot\best.pt

# Backend 실행하는 PowerShell에서 환경변수 등록 (세션 범위)
$env:HOLDEM_AI_TOURNAMENT_CHECKPOINT = (Resolve-Path ".\runs\expand_phase03_pilot\best.pt").Path
$env:HOLDEM_AI_TOURNAMENT_DEVICE = "cpu"
$env:HOLDEM_AI_TOURNAMENT_POLICY_MODE = "sampled"  # 또는 greedy

uv run --extra gui --extra gpu python -m backend.main
```

기존 React 개발 서버가 살아 있다면 화면을 새로 고침한다. GUI의 6-Max 설정에서 **Seat 2 / Tournament Agent = Expand Phase3 · Trained Multi-seat PPO** 선택 후 새 Tournament를 생성한다. 새로 생성한 세션부터 해당 정책이 적용된다. `POST /api/v1/tournament/sessions`에서 `lineup="trained_1_improved_4"`를 지정할 때만 실제 학습된 정책을 사용한다. 기존 `lineup="improved_4"`와 `lineup="baseline_5"`는 변경되지 않는다.

**체크포인트가 없을 때:** trained 선택을 비활성화하고 이유를 표시한다. API에서 trained 라인업을 강제 요청하면 422 오류가 발생하며 **Random Bot으로 조용히 대체하지 않는다.** 형식이 맞지 않는 체크포인트도 거부한다.

## 3. 관측 및 행동 계약

Tournament Phase 4 엔진의 합법 행동:

$$
\mathcal A_T = \{0:\mathrm{Fold}, 1:\mathrm{CheckCall}, 2:\mathrm{RaiseHalfPot}, 3:\mathrm{RaisePot}, 4:\mathrm{AllIn}\}.
$$

Expand Phase3 정책 출력:

$$
\mathcal A_E = \{0:\mathrm{Fold}, 1:\mathrm{Check},2:\mathrm{Call},3:\mathrm{RaiseHalfPot},4:\mathrm{RaisePot},5:\mathrm{RaiseDoublePot},6:\mathrm{AllIn}\}.
$$

\(\mathrm{RaiseDoublePot}\)은 Tournament Native Action에 존재하지 않으므로 항상 Mask한다.

$$
\mathrm{map}(a_E) = \begin{cases}
0 & a_E=0\\
1 & a_E\in\{1,2\}\\
2 & a_E=3\\
3 & a_E=4\\
4 & a_E=6
\end{cases}
$$

Check/Call을 구분할 때는 시점의 `to_call`을 사용한다. 실제 합법성의 **유일한 권위는 Poker Engine의 `legal_actions`**이며, 모든 신경망 확률 출력에 엔진 파생 마스크를 적용한다.

$$
\pi_{\theta}^{\text{legal}}(a\mid I)
=\frac{\pi_\theta(a\mid I)\mathbf1[a\in\mathcal A^{\text{legal}}(I)]}
{\sum_b\pi_\theta(b\mid I)\mathbf1[b\in\mathcal A^{\text{legal}}(I)]}.
$$

결정 시 Actor에 전달되는 정보:

- **Seat 2 본인의** 두 Hole Cards
- 공용 카드, BTN, 스택, 공개 베팅, 팟, BB 및 합법 행동 마스크
- 해당 핸드에서 지금까지 공개된 액션 히스토리(행동 직전 Pot/Call 포함)
- **전달하지 않는 정보**: 다른 플레이어 Hole Cards, 미래 덱, 전지적 게임 상태, Critic/학습 전용 상태

명확하게 구분하기 위해 새로운 정책 인터페이스는 `TrainedTournamentActor.select_action(observation) -> int`다. 모델은 읽기 전용으로 로드하며 가중치를 업데이트하지 않는다.

## 4. 사용자 UI / Server 계약

신규 GET API:

```text
GET /api/v1/tournament/agent-status
```

`configured`, `file_found`, `checkpoint_name`, `format`, `model_family`, `device`, `mode`, `validated`를 반환한다. `validated=false`는 상태 조회가 모델 아키텍처 전체를 검증하는 과정이 아니기 때문이다. **세션 생성 시** `load_trained_actor`가 형식과 가중치 호환성을 검사한다. 설정 경로의 전체 절대경로는 HTTP 응답에 반환하지 않는다.

신규 생성 라인업:

```json
{"lineup":"trained_1_improved_4","seed":42,"starting_stack":1000,"button":0,"hands_per_level":10,"max_hands":10000}
```

Seat 2 응답 예:

```json
{"seat":1,"role":"tournament_agent","policy_name":"Expand Phase3 PPO [sha256_12...]"}
```

나머지 좌석과 기존 HUNL GUI 응답 스키마는 유지한다.

## 5. 테스트

Backend를 중지한 상태에서:

```powershell
uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_tournament_env1_api.py tests/test_tournament_env13_improved_bots.py tests/test_tournament_env131_aggression.py tests/test_tournament_env14_trained_actor.py
uv run --extra dev --extra gui --extra gpu python -m scripts.tournament_env14_smoke
uv run --extra dev --extra gui --extra gpu python -m pytest -q

cd frontend
npm.cmd run typecheck
npm.cmd run build
npm.cmd run test
```

- `tournament_env14_smoke`에는 환경변수 `HOLDEM_AI_TOURNAMENT_CHECKPOINT`가 설정되어 있어야 하며 실제 체크포인트로 행동 추론을 실행한다.
- 테스트 시 실제 프로젝트의 전체 Expand Phase2/3 모듈을 사용하는지 확인한다.
- 기존 토너먼트의 `revision`, Side Pot, 칩 보존, 파산 종료 테스트를 유지한다.

## 6. 범위 밖

- 6-Max 또는 Tournament PPO의 **새로운 훈련 및 모델 개선**은 Agent 담당.
- 지속 저장·Hand Replay는 계획대로 Tournament ENV 1.6.
- UX(순위, Blind Timer 등) 확장은 Phase 1.5.
- 학습 PPO 체크포인트를 **검증된 토너먼트 강자**라고 주장하지 않는다.
