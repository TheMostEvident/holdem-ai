# holdem-ai

**English** | [한국어](README.ko.md)

```math
\boxed{s_{t+1}=T(s_t,a_t,\omega_t)}
\qquad
\boxed{a_t\sim\pi_\theta(\cdot\mid I_{i,t}),\ a_t\in A_{\mathrm{legal}}(s_t)}
```

**The environment executes the game rules; the agent decides how to act using the information visible to it.** This research repository implements simulation, training, evaluation, and GUIs for heads-up no-limit Hold'em, dynamic cash games with 2–6 players, and six-player knockout tournaments.

Start with the [Final Integrated Technical Report](docs/POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). It connects each phase's purpose and value, equations, actual Python code, execution flow, validation results, and unresolved issues. [TECH REPORT](docs/TECH_REPORT.md) provides reading paths by objective and verification criteria.

## Projects and phases

| Project | Phase | Capabilities |
|---|---|---|
| Base Environment | P0–P4 | RLCard reference environment, corrected heads-up action order, independent 2–6-player engine, side pots, and tournament progression |
| HUNL Agent | P5–P10 | Information sets, legal actions, Transformer actor/Q critic, PPO, and experimental QBoost |
| HUNL training and evaluation | P11–P17 | Batching, resume, seed-blocked evaluation, critic/policy diagnostics, ablations, continuous stacks, and GPU retraining |
| Expand | E1–E7 | Dynamic multiplayer tables, multiplayer PPO, MCCFR blueprint, bounded search, opponent modeling, and integrated policy routing |
| HUNL GUI | G1.0–G1.6 | Human–agent games, FastAPI/React, checkpoint selection, statistics, and continuous stacks |
| Tournament ENV | T1.1–T1.7 | Six-player GUI, improved bots, multiplayer frozen actor, blinds/standings, SQLite, replay, and showdown disclosure |

HUNL and Expand actors have different model, observation, and checkpoint contracts. Connecting an Expand E3 cash-game model to a tournament differs from training a tournament-specific policy. MCCFR/search are research implementations; the current records do not establish that HUNL is solved, that a policy is GTO, that Pluribus has been reproduced, or that a tournament policy is optimal.

## Correctness criteria

```math
\sum_i S_{i,t}^{\mathrm{behind}}+P_t=C_h,
\qquad R_{i,h}^{BB}=\frac{S_{i,h}^{\mathrm{final}}-S_{i,h}^{\mathrm{start}}}{BB_h}
```

Verify chip conservation and settlement within each hand first. $`R^{BB}`$ is chip profit or loss divided by the big blind. Placement rewards and prize utility require separate objective functions. The legal-action mask and the actor's private-information boundary apply consistently across training, evaluation, and GUIs.

```math
\widehat{bb/100}=\frac{100}{H}\sum_{h=1}^{H}R_h^{BB}
```

Performance comparisons use paired deck, seat, and button conditions for each seed, together with confidence intervals. Games reusing the same deck do not count as independent samples. Distinguish checkpoint selection on validation data from the final held-out test.

## Installation and basic execution

`pyproject.toml` requires **Python 3.10 or later**. The existing Windows execution reference uses Python 3.11; the frontend requires **Node 20.19.0 or later**.

```powershell
git clone https://github.com/TheMostEvident/holdem-ai.git
cd holdem-ai
uv sync --extra dev
uv run python main.py --hands 100 --seed 42
```

For GUI and PyTorch training features on Windows:

```powershell
uv sync --extra dev --extra gui --extra gpu
uv run --extra dev --extra gui --extra gpu python -m pytest -q
```

The GPU extra's current PyTorch configuration applies conditionally to Windows and Python versions below 3.14. Configure PyTorch separately for other operating systems. These commands document the execution procedure; this documentation work did not rerun the complete game and training test suite.

## Running the GUI

Start the backend from a terminal at the repository root.

```powershell
.\scripts\start_gui_with_models.ps1
```

Start the frontend in another terminal.

```powershell
cd frontend
npm.cmd ci
npm.cmd run typecheck
npm.cmd test
npm.cmd run build
npm.cmd run dev
```

To select trained models, supply a local `best.pt` checkpoint for each corresponding model family.

```powershell
.\scripts\start_gui_with_models.ps1 `
  -PpoCheckpoint ".\runs\my_hunl_run\best.pt" `
  -TournamentCheckpoint ".\runs\my_expand_run\best.pt" `
  -Device cpu `
  -PolicyMode sampled
```

Replace `my_hunl_run` and `my_expand_run` with actual local run directories. Checkpoint files are not included in the repository. After changing a model, restart the backend and create a new session. The startup script searches explicit paths, environment variables, and candidate directories in that order; a `last.pt`/`latest.pt` input resolves to the same run's `best.pt`. Model compatibility is checked when an actual game starts.

## Complete sources and development records

| Material | Scope | Location |
|---|---|---|
| Final integrated report body | Equations, implementation, and execution explanations for all projects and phases | [Integrated Technical Report](docs/POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md) |
| Original Python source appendix | **327 complete files**, including CLI, packages, and tests | The report's Python source index and `docs/technical_sources/` |
| Original document appendix | **76 complete documents**, including code, equations, and logs | The report's Markdown source index and `docs/report_sources/` |
| Source-verification manifest | File paths, SHAs, byte counts, and appendix locations | [lossless_source_manifest.json](docs/lossless_source_manifest.json) |

The complete original sources from reference commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4` were verified using Git blob SHAs: **Python 327/327 and Markdown 76/76**. The 73 superseded standalone reports were deleted after their contents were integrated into the final report's appendices. Executable code, configuration, and frontend files were outside the documentation cleanup scope.

## Interpreting the current experimental results

The historical P17 GPU retraining record reports **5,000 accepted updates / 640,000 hands / 12,743.274 active seconds**. It stopped at `max_updates_ceiling` before the 18,000-second target, with `complete=false`. This does not establish independent held-out performance or completion of five hours of training.

Further development decisions require separate validation of engine rules and information boundaries, PPO training stability, independent evaluation, Expand's connection to the actual multiplayer engine, and tournament-specific training. The final technical report contains the detailed evidence and code for each phase.
