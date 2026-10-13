# GUI trained policy checkpoints (Windows)

Both GUI policy selectors stay clickable, even when no local model file is configured. Selecting a policy does **not** make missing weights available: the API rejects attempted creation with HTTP 422 and preserves the active session. There is no silent fallback to a heuristic bot.

## Required model types

| UI choice | Server variable | Compatible weights |
| --- | --- | --- |
| HUNL: Trained PPO | `HOLDEM_AI_PPO_CHECKPOINT` | HUNL200 Transformer Actor/PPO (Phase 9-compatible checkpoint) |
| HUNL: Trained Q-Boost PPO | `HOLDEM_AI_QBOOST_CHECKPOINT` | HUNL200 Transformer Actor/Q-Boost checkpoint |
| Tournament: Trained PPO with other Improved Bots | `HOLDEM_AI_TOURNAMENT_CHECKPOINT` | **Expand Phase 3 2–6-seat PPO** cash-game checkpoint |

**HUNL and Expand Phase 3 checkpoint formats are not interchangeable.** The tournament policy was not optimized specifically for knockout tournament ICM; it is a research cash-to-tournament transfer. The heuristic seats are independent of the chosen trained policy.

## Quick launch with local checkpoint discovery

Run the script **from a Windows PowerShell at the project root**:

```powershell
git switch main
git pull --ff-only origin main
uv sync --extra gui --extra gpu
.\scripts\start_gui_with_models.ps1
```

By default it checks these **exact conventional local paths** in order:

- HUNL PPO: `runs/phase14_main/seed_52/ppo/best.pt`, `runs/phase14_pilot/seed_41/ppo/best.pt`, `runs/phase17_ppo_5h_seed_01/best.pt`.
- HUNL Q-Boost: `runs/phase14_pilot/seed_41/qboost/best.pt`.
- Tournament: `runs/expand_phase03_pilot/best.pt`.

If the weights are stored elsewhere, specify their **real paths** explicitly. The script validates that each supplied path exists and points to a `.pt` file, then configures the API process:

```powershell
.\scripts\start_gui_with_models.ps1 `
  -PpoCheckpoint ".\runs\my_hunl_training\best.pt" `
  -TournamentCheckpoint ".\runs\my_expand_training\best.pt" `
  -Device cpu -PolicyMode sampled
```

Run the React application in a **separate terminal**:

```powershell
cd frontend
npm ci
npm test
npm run build
npm run dev
```

Open `http://127.0.0.1:5173` and choose HUNL or Tournament. **Restart the backend**, not just Vite, whenever checkpoint environment variables change. The script uses the current process environment and does not download or create weights.

### Verify the two independent model status APIs

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/hunl/bots | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8000/api/v1/tournament/agent-status | Format-List
```

HUNL `available: true` and Tournament `file_found: true` indicate **file presence, not model compatibility**. The loader validates the trusted local model **on game creation**. An incompatible checkpoint produces an explicit error. No training occurs in the GUI.

## Revert a GUI bot from last.pt/latest.pt to best.pt

The GUI backend reads its model path from a PowerShell environment variable at **server start**, not from the browser. Stop the existing FastAPI server first. The launcher now recognizes an old environment variable or argument ending in `last.pt` or `latest.pt` and redirects to the **sibling** `best.pt` in that SAME run directory. If that sibling is missing, it aborts instead of silently loading last or a different run.

To set a specific known-good best model explicitly:

```powershell
.\scripts\start_gui_with_models.ps1 `
  -PpoCheckpoint ".\runs\phase14_main\seed_52\ppo\best.pt" `
  -TournamentCheckpoint ".\runs\expand_phase03_pilot\best.pt"
```

If it was your Q-Boost slot using last.pt, supply `-QBoostCheckpoint "ACTUAL_RUN_DIRECTORY\best.pt"` too. All paths must exist locally. Restarting the browser alone does not update the running Backend's cached Actor. **Switching checkpoints affects new games only**; already-started sessions keep their frozen policy instance.

## Troubleshooting

- **Trained choice clickable, but session creation fails:** Ensure the matching file exists; check the two status APIs; stop and restart the Backend with the proper variables. The error message will identify which variable or model format is required.
- **Wrong model architecture:** The HUNL 7-action Transformer Actor is not an Expand Phase 3 multi-seat Actor. Use the correct source model.
- **Torch/device error:** Launch with `-Device cpu`. CUDA depends on GPU architecture, driver, and an appropriate PyTorch build; the PowerShell script cannot repair a mismatched CUDA installation.
- **No local `.pt` at all:** Source and experiments are merged on GitHub, but training weights under `runs/` are git-ignored and **must be provided by a completed local training run**. The GUI is not a model downloader.
