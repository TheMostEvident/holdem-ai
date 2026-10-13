# Tournament ENV 1.4 — Frozen Multi-seat PPO Adapter

**Patch over Tournament ENV 1.3.1.** Updates the Tournament backend schema, session, router, GUI selector, and adds trained_actor adapter, tests, smoke and Obsidian documentation. Existing HUNL environment and RL training files remain untouched.

Run from repository root:

```powershell
Expand-Archive -Path "$HOME\Downloads\holdem-ai-tournament-env-phase-1-4.zip" -DestinationPath . -Force
$env:HOLDEM_AI_TOURNAMENT_CHECKPOINT = (Resolve-Path '.\runs\expand_phase03_pilot\best.pt').Path
$env:HOLDEM_AI_TOURNAMENT_DEVICE = 'cpu'
$env:HOLDEM_AI_TOURNAMENT_POLICY_MODE = 'sampled'
uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_tournament_env14_trained_actor.py
uv run --extra dev --extra gui --extra gpu python -m scripts.tournament_env14_smoke
```

Open React 6-Max GUI and choose **Expand Phase3 · Trained Multi-seat PPO**. This is an **experimental 6-max cash-to-knockout transfer**: it is a *trained 2–6 player PPO* checkpoint, **not** verified tournament-trained or GTO. A missing/wrong checkpoint fails closed. Never substitute HUNL200 2-seat Q-Boost weights.

See `docs/TOURNAMENT_ENV_PHASE_1_4.md`.
