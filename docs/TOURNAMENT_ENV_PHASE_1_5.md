# Tournament ENV Phase 1.5 — 6-Max Tournament Progress UX

> 담당 영역: **Environment Only**
> 기준: Tournament ENV Phase 1.1–1.4 + Improved Bot Aggression 1.3.1
> 목적: 6인 토너먼트의 **생존/탈락**, **블라인드 레벨 진행**, **정산 기준 칩 순위**, **공식 순위**, **최종 우승**을 혼동 없이 표시한다.

## 1. 기존 구현과 추가 기능

기존 Phase 1.4의 **학습된 Expand Phase 3 PPO**(6인 Cash-game 학습 정책의 Tournament 전이 추론), **Human 1명**, **Improved Heuristic Bot 4명**, 엔진 베팅 규칙은 변경하지 않는다. React/FastAPI 응답의 표시 정보를 확장하는 작업이다.

| 기능 | 기존 | Phase 1.5 |
| --- | --- | --- |
| 생존자 정보 | 현재 핸드 `stack`/`status`에 의존 | `alive_seats` (마지막 핸드 정산 이후 생존 좌석) |
| 칩 순위 | 별도 순위표 없음 | `tournament_stacks` 기준 임시 칩 순위(~N) |
| 확정 순위 | 각 좌석 `rank` | 생존자의 임시 칩 순위와 탈락자의 공식 배치(`#N`) 구분 |
| 블라인드 레벨 | 현재 SB/BB 및 `level_index` | 다음 SB/BB, 레벨 내 진행, 다음 레벨까지 남은 핸드 수 |
| 탈락 이벤트 | 완료된 핸드 History 열람 | 최근 탈락자와 해당 핸드 번호 |
| 우승 | 완료 메시지 | 명시적인 Champion 화면·공식 #1 |

## 2. 토너먼트 스택과 핸드 내부 칩 구분

홀덤 핸드 중 베팅된 칩은 `stack`에서 차감되어 `pot`에 들어간다. 따라서 **현재 핸드에서 `stack=0`이라고 바로 탈락은 아니다.** Short Stack의 강제 올인이나 자발적 올인을 잘못 탈락 처리하면 안 된다.

### 실시간 칩 보존

$$\boxed{\sum_{i=0}^5 S_{i,t}^{\mathrm{behind}}+P_t=C_{\mathrm{total}}}$$

- `seats[i].stack`: 현재 액션 직후 **팟에 투입하고 남은 칩**.
- `pot`: 현재 핸드의 전체 기여 칩.
- `tournament_stacks[i]`: 직전 완료 핸드 이후 정산된 해당 좌석 스택. 현재 핸드가 끝나기 전까지 **리더보드에서 고정**.
- `alive_seats`: 직전 정산 기준 생존 중인 물리적 좌석 번호. 현재 핸드의 올인은 탈락이 아님.
- `seats[i].rank`: 엔진이 제공한 **확정 토너먼트 순위**. 일반적으로 생존 플레이어는 `null`, 탈락한 플레이어는 확정 순위, 최종 우승자는 `1`.

### 순위 계산

생존자 집합은

$$\mathcal A_h=\{i\mid S_{i,h}^{\mathrm{settled}}>0\}.$$

**임시 칩 순위**는 생존자에 대해서만 정산 기준 스택을 내림차순으로 정렬한다. 동점이면 공동 순위(competition ranking)를 준다.

$$\widetilde{r}_i=1+\#\{j\in\mathcal A_h:S_{j,h}^{\mathrm{settled}}>S_{i,h}^{\mathrm{settled}}\}.$$

이는 토너먼트 상금·탈락 순위가 아니며 GUI에 `~N`으로 표시한다. 반대로 엔진의 탈락 확정 순위는 `#N`으로 표시한다. 동시 탈락의 순위 결정은 기존 Phase 4 `PlayerManager` 규칙을 그대로 따른다.

## 3. 블라인드 레벨 진행

`BlindManager`의 공식 일정표를 사용한다. `hands_per_level=L`, 현재 1-based 핸드 번호가 `h`라면, 마지막 레벨을 제외한 다음 레벨 시작까지 필요한 핸드 수(현재 핸드 포함)는

$$H_{\mathrm{to\ next}}=L-((h-1)\bmod L).$$

API에서 이 값을 서버가 계산해 넘겨 주기 때문에, 브라우저 화면이 독립적인 블라인드 일정표를 만들 필요가 없다. 마지막 레벨에서는 다음 블라인드가 존재하지 않으므로 거짓 카운트다운을 보여주지 않는다.

### 읽기 전용 API 변경

기존 응답 `GET /api/v1/tournament/sessions/{id}` 및 액션 반환 `state`에 다음을 추가한다.

```json
{
  "alive_seats": [0, 1, 2, 3, 4, 5],
  "tournament_stacks": [1000, 1000, 1000, 1000, 1000, 1000],
  "blind_progress": {
    "hands_per_level": 10,
    "hand_in_level": 1,
    "hands_to_next": 10,
    "next_small_blind": 10,
    "next_big_blind": 20,
    "final_level": false
  }
}
```

최종 완료 상태에서 `hands_to_next`, `hand_in_level`, `next_small_blind`, `next_big_blind`는 `null` 가능하다. 이전 Phase 1.1 API의 모든 필드는 유지한다.

## 4. 파일 변경

```text
backend/tournament_schemas.py                   # API 응답 Pydantic 확장
src/holdem_ai/tournament_service/session.py      # 읽기 전용 정산 기반 정보 공개
frontend/src/tournament/types.ts                 # TS API 계약 확장
frontend/src/tournament/standings.ts             # 임시/확정 순위, 탈락 기록, 블라인드 문구
frontend/src/tournament/standings.test.ts        # 신규 유닛 테스트
frontend/src/tournament/TournamentProgressPanel.tsx
frontend/src/tournament/TournamentApp.tsx
frontend/src/tournament/TournamentTable.tsx
frontend/src/tournament/rules.ts
frontend/src/tournament/rules.test.ts
frontend/src/styles.css
```

학습용 `src/holdem_ai/expand/rl/...`은 수정하지 않는다. 체크포인트 파일도 건드리지 않는다.

## 5. 적용 및 검증 (Windows PowerShell)

프로젝트 루트에서 ZIP을 덮어쓴 다음 **Backend를 재시작**해야 한다. 기존 세션 데이터는 서버 메모리에만 저장되므로, 가능한 토너먼트를 끝낸 뒤 재시작한다.

```powershell
cd C:\Users\yiydg\Desktop\workspace\holdem-ai-starter\holdem-ai
Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-5.zip" -DestinationPath . -Force

uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_tournament_env15_progress.py
uv run --extra dev --extra gui --extra gpu python -m pytest -q

cd frontend
npm.cmd run typecheck
npm.cmd run build
npm.cmd run test
```

기존 모델을 사용하려면 같은 Backend 터미널에서:

```powershell
cd C:\Users\yiydg\Desktop\workspace\holdem-ai-starter\holdem-ai
$env:HOLDEM_AI_TOURNAMENT_CHECKPOINT = (Resolve-Path '.\runs\expand_phase03_pilot\best.pt').Path
$env:HOLDEM_AI_TOURNAMENT_DEVICE = 'cpu'
$env:HOLDEM_AI_TOURNAMENT_POLICY_MODE = 'sampled'
uv run --extra gui --extra gpu python -m backend.main
```

프론트엔드는 별도 터미널 `cd frontend; npm.cmd run dev`, 브라우저 `http://127.0.0.1:5173`.

### Acceptance checklist

- [ ] Tournament Agent 실제 체크포인트 추론 기능 유지
- [ ] 6인 Seat 및 기존 자유로운 Raise-to 정상 동작
- [ ] 올인으로 현재 Behind Stack이 0이어도 탈락 처리하지 않음
- [ ] 토너먼트 칩 보존 검증 통과
- [ ] 생존자의 임시 칩 순위와 탈락자의 공식 순위가 서로 다르게 표시됨
- [ ] 블라인드 레벨 변경 시 다음 레벨과 남은 핸드 수가 갱신됨
- [ ] 최종 레벨에서 가짜 다음 레벨을 표시하지 않음
- [ ] 탈락 이벤트 및 최종 Champion 표시
- [ ] 기존 HUNL GUI·RL 학습 파이프라인은 불변

## 6. 한계와 다음 단계

현재 History는 세션 메모리에 유지된다. **지속 저장과 Hand Replay는 Phase 1.6**이다. 기존 Expand Phase 3 PPO는 캐시게임에서 학습된 모델이며, Tournament Transfer 추론이 더 강한 전략임을 보장하지 않는다. 이번 Phase 1.5는 **Environment 표시·관측 정확성 개선**이며 학습·전략 개선은 수행하지 않는다.
