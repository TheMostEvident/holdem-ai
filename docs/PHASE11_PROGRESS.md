# Phase 11 — Live progress, ETA, remaining time

This is a **small additive patch** atop Phase 11 (tested 304 passed on user's Windows / CUDA). It changes **only** the existing `scripts/phase11_train.py` and adds `holdem_agent_phase11/progress.py`, `tests/test_phase11_progress.py`. No learning code, network parameters, checkpoints or JSONL formats are changed.

## Installation (PowerShell, project root)

Unzip this patch into `holdem-ai-phase11-progress` (not directly into the repository root), then run:

```powershell
Copy-Item .\holdem-ai-phase11-progress\holdem_agent_phase11\progress.py .\holdem_agent_phase11\ -Force
Copy-Item .\holdem-ai-phase11-progress\scripts\phase11_train.py .\scripts\ -Force
Copy-Item .\holdem-ai-phase11-progress\tests\test_phase11_progress.py .\tests\ -Force
Copy-Item .\holdem-ai-phase11-progress\docs\PHASE11_PROGRESS.md .\docs\ -Force
```

Existing commands work without flags. To enable a native `tqdm` bar (optional; without it a dependency-free progress meter is shown), install with:

```powershell
uv add tqdm
```

Do not install or modify dependencies just to use the fallback progress display.

## CLI examples

```powershell
# Continue an existing 15-minute / 200-update pilot.
uv run --extra gpu python -m scripts.phase11_train --algorithm qboost --device cuda --resume checkpoints/phase11_pilot.pt --duration-minutes 15 --max-updates 200 --env-batch-size 16 --eval-every 25 --eval-hands 200 --checkpoint checkpoints/phase11_pilot.pt --log-dir runs/phase11_pilot

# Original behavior: print every event JSON and no interactive meter.
uv run --extra gpu python -m scripts.phase11_train --no-progress --algorithm qboost --device cuda --resume checkpoints/phase11_pilot.pt --duration-minutes 15 --max-updates 200 --env-batch-size 16 --eval-every 25 --eval-hands 200 --checkpoint checkpoints/phase11_pilot.pt --log-dir runs/phase11_pilot

# With bar: optionally print full per-step JSON records too.
uv run --extra gpu python -m scripts.phase11_train --json-events --progress-interval 2 --algorithm qboost --device cuda --resume checkpoints/phase11_pilot.pt --duration-minutes 15 --max-updates 200 --env-batch-size 16 --eval-every 25 --eval-hands 200 --checkpoint checkpoints/phase11_pilot.pt --log-dir runs/phase11_pilot
```

## Progress semantics

- `upd=31 (+25/200)`: overall completed update count 31, 25 new updates since this invocation began, maximum 200 added updates.
- `hands/s=...`: latest end-to-end training update throughput, not the standalone GPU forward throughput.
- `elapsed`: wall-clock elapsed since the runner's start event (includes its evaluations and training).
- `time_left`: the **remaining time budget for this CLI invocation**, not a promise the job stops at the exact second. An in-flight update or evaluation completes at a safe boundary.
- `ETA`: estimated remaining time from observed update average, **capped by** time_left, so the update target may never be reached before the time limit. Before the first update it equals time_left.
- When `--max-updates` is provided, progress bar percent refers to updates; otherwise percent refers to time. Final status shows `update_limit`, `duration_limit`, or `keyboard_interrupt`.
- The display is a background heartbeat for timing only; it does not train in the background, mutate RNG, or alter checkpoints. The `metrics.jsonl` log still saves full event JSON.
- If you resume with a 15-minute limit, `time_left` starts at 15 minutes again; it does **not** include the time previously spent on older runs.

## Validation

```powershell
uv run --extra dev --extra gpu python -m pytest -q -k phase11
uv run --extra dev --extra gpu python -m pytest -q
```

The progress display refreshes each second even while an individual update/evaluation is executing. Duration is not a hard kill timer; model state is saved after completed updates only.
