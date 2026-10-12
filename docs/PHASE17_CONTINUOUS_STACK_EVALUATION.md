# Phase 17 — Continuous Stack HUNL Evaluation

> Obsidian Markdown. This is an **evaluation-only** expansion of GUI-1.6.
> **Phase 0–16 code, PPO/Q-Boost training equations, and the default independent 200BB reset remain unchanged.**

## 1. Objective and environment

For a trained 7-action Transformer Actor $\pi_\theta$, test behavioral robustness when the match's initial stacks are unequal and subsequent hands inherit the previous terminal stacks. The opponent can be a rule-based baseline or another **frozen** trained Actor.

The **total bank** of a two-player GUI-1.6 match is fixed at $400BB$.

$$
\begin{aligned}
S_{0,1}^{\mathrm{start}}+S_{1,1}^{\mathrm{start}}&=400BB,\\
S_{i,h+1}^{\mathrm{start}}&=S_{i,h}^{\mathrm{final}},\\
S_{0,h}^{\mathrm{final}}+S_{1,h}^{\mathrm{final}}&=400BB.
\end{aligned}
$$

If either stack reaches zero, the match terminates. No rebuy is permitted. The button changes each hand, regardless of which candidate seat is being tested.

The Phase17 200/200BB scenario **does not reset stacks every hand**. It is not equivalent to the Phase16 fixed-stack experiment.

## 2. Experiment grid

For a candidate-start value $s\in\{50,100,200,300,350\}$ BB, initialize:

$$
(S_{\mathrm{candidate}},S_{\mathrm{opponent}})=(s,400-s)BB.
$$

For every independent seed group $g$, replay four physical layouts:

| Layout | Candidate physical seat | First-hand BTN |
|---|---:|---:|
| 1 | 0 | 0 |
| 2 | 1 | 0 |
| 3 | 0 | 1 |
| 4 | 1 | 1 |

Within each group, the same shuffled-deck seed sequence is used for all four layouts. Actions and chip distributions may differ, so this is **not a guarantee of exact duplicate poker**. Stochastic policy RNG streams are deterministic and separate from the shuffle RNG.

For each layout, execute at most $H_{\max}$ hands, ending early on bankruptcy. An unfinished hand at the decision limit **fails the experiment** rather than inventing a payoff.

## 3. Reward, bb/100, and uncertainty

For physical candidate seat $i$ in one full match $m$:

$$
\Delta S_m^{BB}=\frac{S_{i,m}^{\mathrm{final}}-S_{i,m}^{\mathrm{initial}}}{BB}.
$$

For each seed group, **pool the actual completed hands across its four layouts**:

$$
\boxed{b_g=100\frac{\sum_{\ell=1}^{4}\Delta S_{g,\ell}^{BB}}{\sum_{\ell=1}^{4}H_{g,\ell}}}
$$

The report assigns equal weight to independent seed groups:

$$
\bar b=\frac1G\sum_{g=1}^{G}b_g,
\qquad
\widehat{SE}(\bar b)=\frac{s_b}{\sqrt G},
\qquad
CI_{95\%}^{\mathrm{approx}}=\bar b\pm 1.96\widehat{SE}(\bar b).
$$

The interval uses an approximate normal critical value. **Only the seed group** is treated as a replicate; the many hands in a continuous match are *not* independent samples.

If a frozen **control** policy is supplied (for example PPO vs Q-Boost PPO, tested against an identical Calling Station opponent), compute within-seed paired differences:

$$
\boxed{d_g=b_g^{\mathrm{candidate}}-b_g^{\mathrm{control}}}
$$

and compute the corresponding approximate interval from the distribution of $d_g$. The two policies use the same seed schedule but can follow different action paths, so this is **seed-matched** rather than exact counterfactual comparison.

## 4. Additional diagnostics

- **Hand win fraction:** fraction of completed hands with positive chip payoff (loss/tie fractions also reported).
- **Match profit fraction:** $\Pr(\Delta S_m^{BB}>0)$ (not per-hand win rate).
- **Candidate bust fraction:** fraction of whole matches ending in candidate bankruptcy.
- **Opponent bust fraction:** fraction of whole matches ending in opponent bankruptcy.
- **Mean net BB per match:** $\frac1M\sum_m\Delta S_m^{BB}$; easier to interpret when match durations vary.
- **Mean hands per match:** inspect this together with bb/100; short matches can create extreme returns.
- **Showdown fraction:** actual showdowns / all completed hands.
- **Action rates by remaining-effective-stack band:** actions selected divided by times each action was *legal*, using the smaller of the two players' **remaining, not initial** stacks at the decision moment:

$$
E_t^{\mathrm{remaining}}=\frac{\min(S_{0,t}^{\mathrm{uncommitted}},S_{1,t}^{\mathrm{uncommitted}})}{BB}.
$$

Action bands: $E<40$, $40\le E<100$, $100\le E<200$, $E\ge200$ BB. The exact beginning-of-hand poker effective stack differs from the 'remaining' diagnostic; the latter is used here only for plotting action tendencies.

If no decisions in a band legally permit a particular action, its conditional rate is `null`, not falsely zero.

## 5. Installation

Extract `holdem-ai-phase17.zip` **into the existing project root**, where `pyproject.toml`, `backend/`, and `frontend/` already exist. Do **not** create a nested `holdem-ai/holdem-ai/` folder.

Run in PowerShell:

```powershell
cd C:\Users\yiydg\Desktop\workspace\holdem-ai-starter\holdem-ai
Expand-Archive -Path "$HOME\Downloads\holdem-ai-phase17.zip" -DestinationPath . -Force
uv run --extra dev --extra gui --extra gpu python -m pytest -q tests/test_phase17_continuous_stack.py
uv run --extra dev --extra gui --extra gpu python -m pytest -q
```

No `npm install`, FastAPI server startup, or extra GUI package is needed for Phase 17: it is a Python offline batch experiment.

### A. Quick baseline smoke (no checkpoint required)

```powershell
uv run --extra dev --extra gui --extra gpu python -m scripts.phase17_evaluate `
  --candidate-builtin random `
  --opponent calling_station `
  --groups 2 --hands-per-match 3 `
  --candidate-stacks-bb 50,200,350 `
  --output-dir runs/phase17/smoke-random-v1
```

### B. Test your Phase15 PPO checkpoint

First find the actual model path:

```powershell
Get-ChildItem .\runs -Recurse -Filter best.pt | Select-Object FullName
```

With the paths replaced by existing files, run:

```powershell
uv run --extra dev --extra gui --extra gpu python -m scripts.phase17_evaluate `
  --candidate .\runs\phase15_ppo\best.pt `
  --opponent calling_station `
  --groups 30 --hands-per-match 50 `
  --candidate-stacks-bb 50,100,200,300,350 `
  --mode sampled --device cpu `
  --output-dir runs/phase17/ppo-vs-calling-v1
```

This can take substantial CPU time: the pretrained Transformer runs once per bot decision, and Python's reference showdown evaluator is exact rather than optimized. Start with the smoke test and increase groups gradually.

### C. Paired candidate-versus-control experiment

```powershell
uv run --extra dev --extra gui --extra gpu python -m scripts.phase17_evaluate `
  --candidate .\runs\phase15_qboost\best.pt `
  --control .\runs\phase15_ppo\best.pt `
  --opponent calling_station `
  --groups 30 --hands-per-match 50 `
  --mode sampled --device cpu `
  --output-dir runs/phase17/qboost-vs-ppo-matched-v1
```

For a trained opponent instead of a baseline, use `--opponent-checkpoint <trusted-local-actor.pt>` and omit `--opponent`.

## 6. Output

Each run creates **a new, empty directory** containing:

```text
runs/phase17/<run-id>/
├── results.json   # complete machine-readable metrics, seed blocks, action frequencies
├── summary.csv    # stack scenarios and compact metrics
└── report.md      # Obsidian-compatible experiment report
```

Existing nonempty directories are rejected rather than overwritten, preserving independent experiments. Model checkpoint bytes **are not** copied into these outputs. The `.pt` file must be a trusted local model file. The Phase16 loader uses `torch.load(..., weights_only=True)`; this remains a *local/trusted weights* interface, not an arbitrary public upload endpoint.

## 7. Interpretation, caveats, and next Phase

1. A strong positive measured mean is **not** a GTO exploitability certificate or population-strength guarantee.
2. Equal weighting of seed groups (with four layouts per group) matters more than treating each correlated hand as an independent observation.
3. An **initially asymmetric stack** grid is a deliberate stress test, not a sample of stack positions naturally reached by PPO self-play.
4. This report addresses **evaluation**, not variable-stack agent training. The original Actor's input encoding was designed around HUNL200; distribution shift is a scientific hypothesis that must be measured, not assumed resolved.
5. If the sample mean differs across stack scenarios, inspect bootstrap/CI width, match length, bust rate, and the conditional action profiles before drawing conclusions.
6. The system remains a **local research and play environment**; this experiment does not interact with an online poker platform.

**Possible Phase 18:** train (and independently test) a variable-stack Actor, keeping original fixed-200BB evaluation benchmarks as an untouched control.
