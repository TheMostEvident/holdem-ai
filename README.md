# holdem-ai

**Research software for Heads-Up No-Limit Hold'em (HUNL), dynamic 2–6-player cash poker, and 6-max tournament play.**

This repository separates two responsibilities:

- **Environment:** cards, legal actions, positions, bets, side pots, showdown, chip accounting, tournament progression, and GUI-facing state.
- **Agent:** information-state encoding, masked decisions, on-policy PPO, experimental QBoost, multi-seat PPO, MCCFR/search prototypes, diagnostics, and evaluation.

> **Research status:** Experimental systems and test infrastructure; **not a solved HUNL game, verified GTO agent, replicated Pluribus, or a proven tournament-optimal policy**.

## Read first

**[Final integrated technical report — equations, purpose and technical operation of every Phase](docs/POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md)**

The technical report is the primary overview. Original granular development documents remain in \`docs/\` while their source-level evidence is preserved. A concise synthesis is not permission to delete the original reports before full lossless archiving.

## The mathematical contract

Let the complete hand state be $s_t$, legal action $a_t$, and randomized chance event $\omega_t$:

$$
s_{t+1}=T(s_t,a_t,\omega_t),\qquad a_t\in\mathcal A_{\rm legal}(s_t).
$$

The actor uses its own information state $I_{i,t}$ rather than the omniscient engine state:

$$
\pi_\theta(a\mid I_{i,t})=
\frac{\mathbf1[a\in\mathcal A(I_{i,t})]e^{\ell_\theta(I_{i,t},a)}}
{\sum_{b\in\mathcal A(I_{i,t})}e^{\ell_\theta(I_{i,t},b)}}.
$$

Chip conservation and return, assuming no rake and no mid-hand external funding:

$$
\sum_iS_{i,t}^{behind}+P_t=C_t,\qquad
R_{i,h}^{BB}=\frac{S_{i,h}^{final}-S_{i,h}^{initial}}{BB_h},\qquad
\widehat{bb/100}=\frac{100}{H}\sum_h R_{i,h}^{BB}.
$$

## Project map

| Workstream | Phases | Deliverable |
|---|---|---|
| Base Environment | 0–4 | RLCard baseline/corrections; 2–6-player native engine; tournament game rules |
| HUNL Agent | 5–14 | Seven-action HUNL200 neural Actor, PPO/QBoost, batching and seeded evaluation |
| HUNL research | 15–17 | Critic/collapse diagnostics, stability ablation, continuous-stack evaluation, GPU retraining |
| Expand | E1–E7 | Dynamic cash table, multi-seat encoding/PPO, research MCCFR/search/opponent adaptation |
| HUNL GUI | 1.0–1.6 | Human interaction, FastAPI/React, checkpoint inference, match analytics and stack carry |
| Tournament ENV | 1.1–1.7 | Six-seat UI, improved opponents, E3 PPO inference bridge, SQLite/replay, showdown reveal |

**Important distinctions:** P17 continuous-stack evaluation and P17 five-hour PPO retraining are different experiments. Base P17 HUNL weights and Expand E3 multi-seat weights are incompatible by default. Expand E3 is cash-game trained infrastructure; Tournament-specific PPO is a future training track.

## Install and test

Python 3.11 and [uv](https://docs.astral.sh/uv/) are the reference package workflow.

\`\`\`powershell
git clone https://github.com/TheMostEvident/holdem-ai.git
cd holdem-ai
uv sync --extra dev
uv run --extra dev python -m pytest -q
\`\`\`

For GUI/GPU dependencies on an appropriate CUDA host:

\`\`\`powershell
uv sync --extra dev --extra gui --extra gpu
uv run --extra dev --extra gui --extra gpu python -m pytest -q
cd frontend
npm.cmd ci
npm.cmd test
npm.cmd run build
npm.cmd run dev
\`\`\`

These are execution instructions rather than newly certified passing tests. GPU device and driver compatibility must be verified locally.

Run a corrected HUNL baseline from the repository root:

\`\`\`powershell
uv run python main.py --hands 100 --seed 42
\`\`\`

### Trained GUI checkpoints

HUNL trained PPO, experimental QBoost, and Expand E3 multi-seat PPO are **different model families** with different files and action contracts. A real locally available, trusted and compatible \`best.pt\` is required for trained-policy inference. Files inside \`runs/\` are generally excluded from Git. Use [checkpoint setup](docs/GUI_TRAINED_CHECKPOINT_SETUP.md) for local paths and \`scripts/start_gui_with_models.ps1\`; ensure the backend restarts and creates a fresh session after changing weights.

## Evaluation standards

Use paired seed-blocked seat/button configurations and statistical uncertainty, not a single high score:

$$
X_k=\frac14\sum_{j=1}^{4}R_{k,j}^{BB},\qquad
\widehat J_{bb/100}=\frac{100}{K}\sum_{k=1}^K X_k.
$$

Deck seed groups, not the four games within each group, are independent replicates. Keep validation checkpoint selection separate from a sealed held-out final test.

## Recorded milestone and limitations

The historical Phase17 PPO retrain recorded **5,000 accepted updates and 640,000 training hands**. It stopped at **12,743.274 active seconds**, before its 18,000-second target (\`complete=false\`, \`max_updates_ceiling\`). A repeated-validation best score does **not** prove independent game strength. Phase15/16 preflop and critic calibration issues and Expand native-engine/search acceptance remain research tasks.

## Repository structure

\`\`\`text
src/holdem_ai/
  environment/       # Hand engine, game rules, HUNL200, tournaments
  core/              # InformationState and policy contracts
  encoding/          # Tokenization, masks, tensors
  models/            # Transformer Actor / Q Critic
  training/          # PPO, QBoost, batched self-play
  experiments/       # Reproducible studies and long runs
  evaluation/        # Seed blocked, opponent pool, diagnostics
  expand/            # Dynamic 2-6 player research stack
  game_service/      # HUNL human GUI services
  tournament_service/# Six-seat tournament services
backend/             # FastAPI HTTP contract
frontend/            # React interface
scripts/             # Audits, smoke tests, train/evaluate CLI
tests/               # Regression, privacy and behavior tests
docs/                # Canonical report + original phase reports
\`\`\`

## Engineering priorities

1. Prove poker rule invariants and actor privacy with real-engine end-to-end tests.
2. Investigate preflop policy concentration and critic calibration before interpreting training duration as improvement.
3. Reproduce independent, position-balanced, multi-seed validation and held-out results.
4. Verify Expand E3 checkpoint compatibility and train a **tournament-specific PPO** separately.
5. Preserve original phase reports until an exact-content archived copy has been checked against every original Git blob.

See the [integrated technical report](docs/POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md) for the phase-by-phase mathematical treatment, implementation map, evidence boundaries and original references.
