# Tournament ENV Phase 1.1 — 개발·검증 요약

> 기준 커밋: `d818f44`
> 담당 범위: **Environment Only**
> 현 단계: **6-Max Tournament Engine ↔ FastAPI 연동**
> 주의: **6인 React 화면과 Tournament Agent/Improved Bot 연결은 아직 아님.**

## 1. Objective

기존 Phase 3의 `MultiPlayerHandEngine`과 Phase 4의 `TournamentEnv`를 재사용하여 Human이 참가하는 6인 토너먼트의 서버 API를 구축한다.

$$
\mathrm{TournamentENV}_{1.1}
=\mathrm{Phase4\ Engine}+\mathrm{SessionAdapter}+\mathrm{FastAPI}
$$

`/api/v1/hunl/*`는 기존 그대로 유지하고, `/api/v1/tournament/*`를 추가한다.

## 2. Tournament Invariants

6명의 물리적 좌석을 유지하며, 리바이·레이크가 없을 때 총 칩 수는 일정하다.

$$
C_{\mathrm{total}}=\sum_{i=0}^{5}S_{i,0}=6S_0
$$

진행 중인 핸드에서 베팅된 금액은 팟에 포함되므로:

$$
\boxed{\sum_{i=0}^{5} S_{i,t}^{\mathrm{behind}}+P_t=C_{\mathrm{total}}}
$$

핸드 종료 시 팟을 분배하고 스택을 다음 핸드로 승계한다.

$$
\boxed{S_{i,h+1}^{\mathrm{start}}=S_{i,h}^{\mathrm{final}}}
$$

탈락한 플레이어는 스택 0으로 유지되며 다음 핸드에서 카드를 받지 않는다.

$$
S_{i,h}^{\mathrm{final}}=0\Rightarrow i\notin\mathcal P_{h+1}^{\mathrm{alive}}
$$

## 3. API Contract

| Endpoint | 역할 |
|---|---|
| `GET /api/v1/tournament/health` | 서버 동작 점검 |
| `POST /api/v1/tournament/sessions` | 새 6인 토너먼트 생성 |
| `GET /api/v1/tournament/sessions/{id}` | Human에게 공개할 상태만 반환 |
| `POST /api/v1/tournament/sessions/{id}/actions` | Human 행동 입력 |
| `POST /api/v1/tournament/sessions/{id}/raise-to` | Human이 임의 금액 Raise |
| `GET /api/v1/tournament/sessions/{id}/history` | 완료된 핸드 기록 조회 |
| `DELETE /api/v1/tournament/sessions/{id}` | 세션 삭제 |

Phase 4 엔진의 행동 인덱스는 다음과 같다.

| Index | 행동 |
|---:|---|
| 0 | Fold |
| 1 | Check / Call |
| 2 | Half-Pot Raise |
| 3 | Pot Raise |
| 4 | All-In |

`raise-to`는 별도 엔드포인트를 통해 실행한다. **HUNL200 PPO의 7개 Action Head 인덱스와 다르므로 자동 변환하지 않는다.**

## 4. Player Configuration — 현재와 목표

| Seat | 현재 Phase 1.1 | 향후 목표 |
|---:|---|---|
| 0 | Human | Human |
| 1 | Calling Station (임시 기준 Bot) | Tournament Agent |
| 2–5 | Calling Station (임시 기준 Bot) | Improved Bot ×4 |

지금은 서버 연결 테스트를 위한 **Baseline Bot**만 사용한다. 새 PPO/Q-Boost 학습과 가중치 업데이트는 여기서 하지 않는다.

## 5. 실행 순서

```powershell
# GitHub 프로젝트 루트
Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-1.zip" -DestinationPath . -Force

# Tournament API 신규 테스트
uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_tournament_env1_api.py

# 실제 엔진을 이용한 6인 API 스모크
uv run --extra dev --extra gui --extra gpu python -m scripts.tournament_env1_smoke

# 전체 회귀 테스트
uv run --extra dev --extra gui --extra gpu python -m pytest -q

# Backend 실행
uv run --extra gui --extra gpu python -m backend.main
```

API를 테스트하려면 `http://127.0.0.1:8000/docs`에 접속한다. **6인 React GUI 화면은 아직 생성되지 않는다.**

## 6. 테스트 및 한계

- API 계약 관련 신규 테스트 **11개 통과** (부분 호환 검증 환경)
- API Smoke: Human 25회 행동, 총 10핸드, 6인 매치 종료 확인 (부분 호환 검증 환경)
- 별도 30개 시드의 6인 매치를 반복 실행하여 칩 보존 및 핸드 승계 확인
- **중요:** 검증 환경에 없는 기존 저장소 모듈 일부를 검증용 대체 코드로 처리했기 때문에, 사용자 Windows 전체 프로젝트에서 신규 테스트와 전체 테스트를 다시 실행해야 한다. 대체 코드는 배포 ZIP에 포함하지 않는다.
- 현재 서버 세션은 메모리에만 존재하므로 재시작하면 대국 기록은 사라진다. 영구 기록은 Phase 1.6에서 구현한다.

## 7. Remaining Phase 1 Roadmap

- [x] **1.1** Tournament Engine ↔ FastAPI
- [ ] **1.2** React 6인 Tournament 테이블
- [ ] **1.3** Improved Bot ×4 연결
- [ ] **1.4** Tournament Agent 연결
- [ ] **1.5** 블라인드 상승·탈락·우승 화면
- [ ] **1.6** 영구 대국 기록 및 Replay
- [ ] **1.7** Windows 통합 테스트 및 GitHub Push
