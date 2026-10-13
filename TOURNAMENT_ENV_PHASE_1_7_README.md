# Tournament ENV Phase 1.7 · Release QA Pack

마지막 단계는 기능 변경이 아닌 **안전한 통합 검증 / 기록 보호 / GitHub Push 준비**다.

전제: Tournament ENV Phase 1.1~1.6.1까지 ZIP 패치가 로컬 프로젝트에 이미 적용되어 있어야 한다.

프로젝트 루트에서:

```powershell
Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-7.zip" -DestinationPath . -Force
powershell -ExecutionPolicy Bypass -File .\scripts\tournament_env17_verify.ps1
```

실제 체크포인트 추론을 검증하려면 문서의 환경변수 설정 후 `-WithAgentSmoke` 옵션 추가.

Push 직전:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\tournament_env17_stage.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File .\scripts\tournament_env17_stage.ps1
git diff --cached --name-status
```

Staging 스크립트는 `.gitignore`를 안전하게 덧붙이고, Tournament ENV 관련 변경 경로만 Stage한다. **Commit/Push는 자동으로 하지 않는다.**

전체 기술 문서: `docs/TOURNAMENT_ENV_PHASE_1_7_FINAL.md`
