# Phase 5 RLCard NumPy JSON Hotfix

## Why

RLCard public state can include `numpy.int64` inside a list/dict. The original
`_json_safe` converted the outer object only, then `json.dumps` raised
`TypeError: Object of type int64 is not JSON serializable`.

## Scope

- REPLACE `holdem_agent_phase5/rlcard_adapter.py` (one Phase 5 module only).
- ADD `tests/test_phase5_numpy_json.py` (regression tests).
- No files in existing `src/`, `docs/`, or `scripts/` are changed.

## From repository root, in PowerShell

Download this archive into the `holdem-ai` repository root, then run:

```powershell
Copy-Item .\holdem_agent_phase5\rlcard_adapter.py .\holdem_agent_phase5\rlcard_adapter.py.bak
Expand-Archive -LiteralPath .\holdem-ai-phase5-numpy-patch.zip -DestinationPath . -Force
uv run python -m pytest -q -k phase5
uv run python -m pytest -q
```

This ZIP deliberately has no outer wrapper folder. `-Force` will overwrite the
Phase 5 adapter file; its backup is the `.bak` copy. It will not overwrite
pre-existing Phase 1-4 files because those paths are not in this ZIP.

If a test fails, do not commit; send the full traceback.

After verification, delete `rlcard_adapter.py.bak` and the copied ZIP (if they
reside in the repository root) before committing.
