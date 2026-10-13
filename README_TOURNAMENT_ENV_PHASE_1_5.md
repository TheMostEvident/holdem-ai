# Tournament ENV Phase 1.5 — UX / Standings / Blind Levels

## 적용 전제

Tournament ENV 1.1 → 1.2(+카드 수정) → 1.3(+1.3.1 aggression 수정) → 1.4 설치 완료.
이번 패치는 해당 버전을 기준으로 **변경 파일만 덮어쓰기** 위한 패치다.

## 설치

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

**Backend 재시작 필수**. 브라우저는 새로고침(필요 시 Ctrl+Shift+R). 세션은 메모리 기반이므로 진행 중인 토너먼트를 종료하고 재시작하는 편이 좋다.

## 변경

- 6인 칩 순위(임시 `~N`)와 최종/탈락 순위(`#N`) 분리
- 실제 정산 후 생존 좌석 기반 살아 있는 플레이어 수, 올인 오판 방지
- 다음 Blind Level SB/BB와 상승까지 남은 핸드 수
- 최근 탈락 이벤트 / 토너먼트 우승자 화면
- 게임 규칙, 모델 추론/학습 가중치 불변

자세한 설계와 수식: `docs/TOURNAMENT_ENV_PHASE_1_5.md`
