# hold'em-ai Phase 2.1 patch notes

## Apply (Windows PowerShell)

Run these from the `holdem-ai` repository root. Prefer to commit/backup any local changes before using `-Force`.

```powershell
Expand-Archive -Path "$HOME\Downloads\holdem-ai-phase2-1.zip" -DestinationPath . -Force
uv sync --extra dev
uv run --extra dev python -m pytest -q
uv run python scripts/audit_corrected.py --hands 100 --seed 42
uv run python main.py --hands 5 --seed 42
```

`PATCH_NOTES.md` is a patch document (safe to leave in repository or delete).

## Notes

- `HandEngine` defaults to `rules_mode="corrected"`; the original `rules_mode="upstream"` remains selectable for comparison.
- `scripts/audit_rlcard.py` still tests the **unmodified** RLCard upstream and is expected to fail. Use `scripts/audit_corrected.py` to validate the patch.
- `tests/test_rlcard_rules_integration.py` retains an expected upstream failure (`xfail`) intentionally.
- The patched five-action abstraction is *not* arbitrary bet sizing; six-player tournaments remain out of scope.
- This patch has offline unit test coverage. Actual RLCard 1.2.0 integration must still be run on the local PC.
