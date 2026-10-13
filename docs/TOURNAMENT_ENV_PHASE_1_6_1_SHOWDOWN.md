# Tournament ENV Phase 1.6.1 — Showdown Hole Cards 공개

> Scope: **Environment only**. No Agent training, model update, or change to betting rules.

## 요구사항

- 실제 Showdown이 발생하고 **핸드 정산이 완료된 뒤**에만 카드 공개.
- Fold한 플레이어는 본인의 홀 카드를 공개하지 않음.
- Fold 승리(무경쟁 팟)는 승자라도 홀 카드 공개하지 않음.
- 마지막 핸드 결과, 완료 핸드 목록, SQLite Replay의 **최종 정산 프레임**에서 동일한 정보를 표시.
- 진행 중인 핸드의 모든 상대 홀 카드, 덱, 번 카드는 비공개.
- 기존 JSONL Actor-visible Export 계약은 변경하지 않음.

## 1. 공개 조건

`result.folded[i]`는 해당 플레이어가 Fold했거나 핸드에 참가하지 않았는지를 나타냅니다.

$$
C_h = \{ i \mid \operatorname{folded}_{i,h} = \mathrm{False} \}
$$

$$
\boxed{
\operatorname{Reveal}(i,h)
= \begin{cases}
\operatorname{HoleCards}(i,h), & |C_h|\ge 2 \land i\in C_h \land |B_h|=5\\
\varnothing, & \text{otherwise}
\end{cases}
}
$$

카드는 `TournamentEnv._finish_hand()`에서 정산 완료 시에 캡처하므로, 다음 핸드가 자동 시작돼도 직전 핸드의 Showdown을 볼 수 있습니다. 이 데이터는 행동 선택을 위한 `observe()`에는 전달하지 않습니다.

## 2. HTTP 계약

기존 Hand Settlement JSON에 **선택적** 필드 `showdown_cards`가 추가됩니다.

```json
{
  "hand_number": 5,
  "board": ["sA", "dK", "h7", "c3", "d2"],
  "showdown_cards": [
    {"seat": 0, "cards": ["sQ", "sJ"]},
    {"seat": 1, "cards": ["cK", "dQ"]}
  ]
}
```

위 JSON은 **구조 예시이며 실제 핸드 데이터가 아닙니다**. 반환되는 다른 핸드 결과 필드는 유지됩니다.

API 반영 위치:

- `GET /api/v1/tournament/sessions/{session_id}` → `state.last_hand.showdown_cards`
- `GET /api/v1/tournament/sessions/{session_id}/history` → `hands[].showdown_cards`
- `GET /api/v1/tournament/archive/{session_id}/hands` → `hands[].showdown_cards`
- `GET /api/v1/tournament/archive/{session_id}/hands/{hand_number}/replay` → `settlement.showdown_cards`

Replay의 `actions[]`에는 `showdown_cards`가 들어가지 않습니다. **Showdown 카드는 최종 정산(`settlement`)에서만 공개됩니다.**

오래된 SQLite 기록에는 `showdown_cards`가 없을 수 있으며, 화면은 이를 빈 목록으로 처리합니다. 이미 저장된 기록을 소급 복원하지 않습니다.

## 3. UI 동작

1. 핸드가 완료되면 `MOST RECENT SETTLEMENT`에서 **SHOWDOWN · REVEALED HANDS**에 카드 앞면을 표시.
2. `Completed hands`의 핸드를 펼쳐도 같은 카드를 표시.
3. `Saved sessions & hand replay`의 마지막 프레임에서 Showdown을 표시.
4. Fold 승리 또는 오래된 기록에는 `NO CARD REVEAL` 안내를 표시.

현재 UI에서는 엔진이 **다음 핸드를 자동으로 시작**합니다. 따라서 쇼다운 결과는 새로운 핸드의 좌석 카드 자리에 섞지 않고 별도의 **이전 핸드 결과 패널**에 표시합니다.

## 4. 변경 파일

- `src/holdem_ai/environment/tournament_env.py`: 완료 핸드별 showdown 공개 카드 보존.
- `src/holdem_ai/tournament_service/session.py`: 완료 핸드 요약에 `showdown_cards` 직렬화.
- `backend/tournament_schemas.py`: 엄격한 응답 모델 확장.
- `frontend/src/tournament/types.ts`: 타입 필드 확장(구형 아카이브 호환).
- `frontend/src/tournament/TournamentShowdown.tsx`: 재사용 카드 컴포넌트.
- `frontend/src/tournament/TournamentApp.tsx`: 최근 결과와 히스토리 카드 공개.
- `frontend/src/tournament/TournamentArchive.tsx`: Replay 마지막 프레임에 카드 공개.
- `frontend/src/styles.css`: 숫자·문양·반응형 카드 표시.
- `tests/test_tournament_env161_showdown.py`: 6개 신규 테스트.

## 5. Windows 설치

Backend를 Ctrl+C로 종료한 뒤 프로젝트 루트에서:

```powershell
Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-6-1-showdown.zip" -DestinationPath . -Force

uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_tournament_env161_showdown.py
uv run --extra dev --extra gui --extra gpu python -m pytest -q

cd frontend
npm.cmd run typecheck
npm.cmd run build
npm.cmd run test
```

Frontend는 Vite dev 서버가 이미 실행 중이라면 변경 사항이 자동 반영됩니다. Backend는 새 Python 코드를 읽도록 **반드시 재시작**해야 합니다.

## 6. 참고

원래 Phase 1.6에서 저장하던 `actor_hole` 기록은 **행동한 선수 본인에게만 관측 가능한 정보**로 별도 보관하며, 그대로 유지합니다. 이번 변경은 공개 가능한 쇼다운 결과에 대한 읽기 전용 확장입니다.

본 기능은 토너먼트 Agent 체크포인트나 학습 알고리즘을 변경하지 않습니다.
