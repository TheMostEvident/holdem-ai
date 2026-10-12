# Phase 6 — HUNL200 Environment Integration and CPU Benchmark

> [!important] Status
> This patch is an additive adapter and benchmark, **not VRPO training**. It targets `TheMostEvident/holdem-ai` at Git commit `894a466afe48309ebf01c334c3bdad7471923eb3`.
> Run all tests on the user's local repository after applying. No GitHub changes are made automatically by this ZIP.

## Objectives

- Use existing real `MultiPlayerHandEngine` in 2-player mode; keep Phase 1–5 unchanged.
- Provide HUNL200 (200BB effective stack, 100-chip BB, 50-chip SB, button selectable).
- Adapt engine's arbitrary `step_raise_to` to Phase-5 7-action policy head.
- Keep Actor observations distinct from privileged Q-Critic training states.
- Record per-**global** poker decision state-action-next-state transitions for VRPO.
- Measure CPU self-play hands/s for 1, 2, and 4 processes; estimate an achievable 5-hour training budget after benchmarking.
- Optionally check CUDA/PyTorch forward and backward on GTX 1070 8GB.

## Existing / new boundary

```text
Phase 3 MultiPlayerHandEngine (2 seats, HUNL200)
     |
     v
HUNL200Adapter (NEW) <--------- Phase 5 abstract_bet_actions (7 heads)
     |                                           |
     +-- Turn(InformationState) ----------------> Phase 5 AgentPolicy
     |                                                   |
     +<------------------------------ action index 0..6 -+
     |
     +-- Terminal(rewards in BB) -------> Phase 5 run_self_play
     |
     +-- training_state() (OPT-IN) -----> Phase 6 global trajectory for future Q-Critic

Phase 6 has no Actor network, Q-Critic network, PPO, Q-Boosting, or GPU training loop.
```

### Reward and legality

For each terminal seat $i$:

$$
r_i = \frac{C_i^{\mathrm{final}}-C_i^{\mathrm{initial}}}{BB}
$$

No-rake two-player chip conservation requires:

$$
r_0+r_1=0.
$$

Engine legal constraints are authoritative, not the adapter. Our action IDs are:

| Phase-5 ID | Meaning | Engine call |
|---|---|---|
| 0 | Fold | `engine.step(0)` |
| 1 | Check | `engine.step(1)` |
| 2 | Call | `engine.step(1)` |
| 3 | Half-pot raise | `engine.step_raise_to(amount)` |
| 4 | Pot raise | `engine.step_raise_to(amount)` |
| 5 | Double-pot raise | `engine.step_raise_to(amount)` |
| 6 | All-in raise | `engine.step(4)` |

The 7-head index 4 **does not** mean the engine's all-in ID 4. These mappings MUST remain separate. All-in short-calls are represented by `call`; an all-in *raise* is ID 6 only when legal and meaningfully exceeds call.

At decision time, `abstract_bet_actions` uses the **post-call pot** for fractional raises:

$$
B_{\mathrm{raise\ to}}=c+k+\mathrm{round}\left[f(P+k)\right],\quad f\in\{0.5,1,2\}
$$

Min/max boundaries, deduplication, and short all-in cases are supplied or validated by the real engine.

## Observation privacy / logging

`InformationState` is constructed **only** from the acting player's `engine.observe(player)` plus adapter-recorded *public* street/action/raise history. No other player's cards or hidden deck are passed to the Actor.

`CriticState` is a different dataclass, retrieved through `HUNL200Adapter(expose_training_state=True).training_state()`. It includes both hole-card pairs for **training only**. It intentionally excludes undealt deck and unrevealed future board cards, preventing trivial future-card leakage. Do not attach it to any Actor feature encoder.

`collect_global_episode` records adjacent **global** poker decisions; Phase 5 `run_self_play` records next decisions of the **same** seat. Do not confuse these two time axes when implementing VRPO temporal-difference residuals.

## Apply ZIP (PowerShell)

Open PowerShell in the **project root** (where `pyproject.toml` exists), then:

```powershell
# 1) Ensure correct baseline
 git status
 git log -1 --oneline

# 2) Make optional local development branch
 git switch -c phase6/hunl200-integration

# 3) Extract the delivered ZIP contents directly into the repository root:
# Example: Expand-Archive -Path "$HOME\Downloads\holdem-ai-phase6.zip" -DestinationPath . -Force

# 4) Test Phase 6, then the full old regression suite
uv run python -m pytest -q -k phase6
uv run python -m pytest -q

# 5) Run small and medium random-rollout throughput benchmarks
uv run python scripts/phase6_benchmark.py --hands 100 --workers 1,2,4
uv run python scripts/phase6_benchmark.py --hands 1000 --workers 1,2,4 --json-out data/phase6_rollout_speed.json

# 6) Optional hardware test (requires compatible installed PyTorch)
uv run python scripts/phase6_hardware_probe.py

# 7) Commit only when all regression tests pass
 git status
 git add holdem_agent_phase6 scripts/phase6_benchmark.py scripts/phase6_hardware_probe.py tests/test_phase6_hunl_adapter.py tests/test_phase6_benchmark.py docs/PHASE6_HUNL200_INTEGRATION.md
 git commit -m "Add Phase 6 HUNL200 agent adapter and rollout benchmark"
 git push -u origin phase6/hunl200-integration
```

Avoid committing `data/phase6_rollout_speed.json` unless desired. A report file is generated only if `--json-out` is provided. Add it to `.gitignore` before large-scale logs.

## Acceptance criteria

1. Both player seats receive correct SB/BB, and SB acts first preflop.
2. Illegal action IDs are rejected *before* changing the underlying engine.
3. 7-head actions map to legal exact engine calls, including `raise_to` and all-in.
4. Actor's `private_cards` includes only the acting player's hole cards; opponent cards never appear in public_state/history/features.
5. Critic private data can only be retrieved with explicit opt-in.
6. Zero-sum rewards in BB and all chip-conservation tests hold over 200 sampled games.
7. Phase 1–5 tests continue to pass (baseline was 176 passed, 1 xfailed before applying this patch).
8. 1/2/4-worker benchmark prints reproducible hands/s and decisions/s on the **local machine**.

## What comes next

Phase 7 must match the paper's token observation and Base-16 bet encoding to approach a faithful VRPO reproduction. This Phase 6 adapter deliberately retains the 7-head action abstraction from Phase 5 for low-cost smoke runs. Baseline performance will **not** be directly comparable to paper's HUNL results until rule/encoding/evaluation differences are addressed.
