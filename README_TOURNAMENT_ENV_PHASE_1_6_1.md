# Tournament ENV 1.6.1 — Showdown 카드 공개 패치

Phase 1.6이 설치된 프로젝트 루트에 ZIP을 풀어 덮어쓰세요. 그 이전 패치는 다시 설치할 필요 없습니다.

## 실행 전

1. **Backend 종료** (`Ctrl+C`).
2. `Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-6-1-showdown.zip" -DestinationPath . -Force`
3. `uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_tournament_env161_showdown.py`
4. `uv run --extra dev --extra gui --extra gpu python -m pytest -q`
5. `cd frontend`; `npm.cmd run typecheck`; `npm.cmd run build`; `npm.cmd run test`
6. Backend 재시작 후 새 Tournament 핸드를 한 번 끝내세요.

## 확인할 곳

- **MOST RECENT SETTLEMENT**: 직전 핸드 Showdown 카드 앞면 공개.
- **Completed hands**: 과거 핸드 상세에서 같은 카드 확인.
- **Saved sessions & hand replay**: Replay 마지막 정산 프레임에서 카드 확인.

Fold한 플레이어의 카드와 정산 전 상대 카드 정보는 **절대 공개하지 않습니다**. Fold 승리에는 카드 공개가 없습니다. **기존 SQLite 기록의 과거 숨겨진 카드 데이터는 복구할 수 없습니다.**

자세한 수식과 데이터 계약은 `docs/TOURNAMENT_ENV_PHASE_1_6_1_SHOWDOWN.md`를 참고하세요.
