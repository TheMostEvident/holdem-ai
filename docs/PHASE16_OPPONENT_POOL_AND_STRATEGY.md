# Phase 16 — Opponent Pool Evaluation & Strategy Diagnostics

> **Implementation status:** Phase 16 is evaluation-only. It does **not** train, run, modify or resume Phase 15. You can install and test it now, and evaluate Phase 15's `best.pt` later.
>
> Obsidian-compatible Markdown. Existing Phase 1–15 files are unchanged.

## 1. Objective

Evaluate one frozen HUNL200 Actor against a fixed opponent pool; diagnose **conditional-on-legal** action choices by street, position and effective stack-to-pot ratio (SPR).

$$
\mathcal{O}=\{\text{Random},\text{CallingStation},\text{Tight},\text{PotPressure},\ldots\}
$$

Additional opponents can be frozen TransformerActor checkpoints (e.g. `runs/phase14_main/seed_51/qboost/best.pt`). Phase 14 and Phase 15 `best.pt` are **compatible because their checkpoint formats are Phase 9 or 10**, already supported by `holdem_agent_phase12.checkpoint.load_actor`.

No Phase 15 model is required to perform Phase 16 smoke tests.

### 1.1 Engine, policy and information boundary

```text
HUNL200Adapter (Phase6 engine)
        |
        +-- own private cards + public information + legal mask --> Candidate Actor
        |
        +-- other player's private view + public information --> Opponent
        |
        +-- zero-sum terminal BB payoffs --> evaluation report
```

- Actor receives only `InformationState`, **not** `CriticState`, hidden deck or rival hole cards.
- `privileged=False` for every Phase 16 game.
- Neither the Actor nor checkpoint optimizer state is modified.
- Original opponent policies are intentionally simple diagnostic baselines, not trained equilibrium strategies.

## 2. Opponent Pool

| Name | Action preference | Purpose |
|---|---|---|
| `random` | uniform among currently legal actions | baseline |
| `calling_station` | Check, else Call, then legal fallback | test overcalling |
| `tight` | Check, else Fold, then legal fallback | test overfolding |
| `pot_pressure` | Pot Raise > 2x Pot > Half Pot > All-In > Call > Check | test aggression |
| `NAME=PATH` | frozen TransformerActor loaded from checkpoint | evaluate against previous trained models |

Opponent actions always respect the adapter's legal-action mask. Fixed baselines are crude by design: high bb/100 against them does **not** imply strong HUNL strategy.

## 3. Evaluation protocol

For each opponent and independent shuffled deck seed $k$, play all four physical seat/button layouts:

$$
(b,s)\in\{(0,0),(0,1),(1,0),(1,1)\}
$$

$$
X_{o,k}=\frac14\sum_{b=0}^1\sum_{s=0}^1R_{o,k,b,s}^{\mathrm{candidate}}\quad[\mathrm{bb}]
$$

$$
\widehat J_o=100\cdot \frac1K\sum_{k=1}^K X_{o,k}\quad[\mathrm{bb/100}]
$$

The **one deck seed block** is the independent statistical unit, not each of the four hands. The approximate normal confidence interval is:

$$
\mathrm{SE}(\widehat J_o)=100\cdot \frac{s(X_{o,1},\ldots,X_{o,K})}{\sqrt K}
$$

$$
\mathrm{CI}_{95\%}(\widehat J_o)\approx \widehat J_o\pm1.96\cdot \mathrm{SE}(\widehat J_o)
$$

Candidate and opponent always play the same deck seed and both button/seat combinations. This **does not** ensure the paths after different actions are identical. Evaluation uses the pre-existing Phase 12 `summarize_block_scores` formula.

Across a **predeclared** opponent pool of size $M$:

$$
J_{\mathrm{pool}}=\frac1M\sum_{o=1}^M \widehat J_o,\qquad
J_{\mathrm{worst\_observed}}=\min_o\widehat J_o.
$$

The worst observed pool mean is *not* minimax game value or exploitability. Pool comparisons and selecting the worst score among opponents can increase selection bias. Only opponent-specific CIs are reported; they are not multiplicity-adjusted.

## 4. Action diagnostics

For action $a$:

$$
\widehat P(a\mid a\in\mathcal A(I))=
\frac{\#\{t:a_t=a\}}{\#\{t:a\in\mathcal A(I_t)\}}.
$$

This separates **Check never selected** from **Check never legal**.

- Street: `preflop`, `flop`, `turn`, `river`.
- Position: `button`, `big_blind` (actor-visible feature 1).
- SPR: `spr_lt_1`, `spr_1_to_4`, `spr_4_to_10`, `spr_ge_10`.

Effective SPR uses only existing observable Phase6 features:

$$
\mathrm{SPR}(I)=\frac{\min(\mathrm{hero\_stack},\mathrm{opponent\_stack})}{\max(\mathrm{pot},10^{-9})}
$$

All chip values here are already normalized to BB. This is a descriptive instant-of-decision SPR, not a formal postflop GTO range definition.

Policy entropy (natural logs) is computed from the Actor distribution **on the actual observed decision states**:

$$
H(\pi\mid I)=-\sum_{a\in\mathcal A(I)}\pi(a\mid I)\log\pi(a\mid I)
$$

$$
H_{\mathrm{normalized}}(\pi\mid I)=
\begin{cases}
H(\pi\mid I)/\log |\mathcal A(I)|,&|\mathcal A(I)|>1\\
0,&|\mathcal A(I)|=1.
\end{cases}
$$

The diagnostics are observational, not a new loss function and not opponent adaptation training.

## 5. Installation (Windows PowerShell)

Extract `holdem-ai-phase16-evaluation.zip` into a folder named `holdem-ai-phase16-evaluation` alongside the repository root's files. From the repository root:

```powershell
Copy-Item .\holdem-ai-phase16-evaluation\holdem_agent_phase16 .\ -Recurse -Force
Copy-Item .\holdem-ai-phase16-evaluation\tests\test_phase16_*.py .\tests\ -Force
Copy-Item .\holdem-ai-phase16-evaluation\scripts\phase16_evaluate.py .\scripts\ -Force
Copy-Item .\holdem-ai-phase16-evaluation\docs\PHASE16_OPPONENT_POOL_AND_STRATEGY.md .\docs\ -Force

uv run --extra dev --extra gpu python -m pytest -q -k phase16
uv run --extra dev --extra gpu python -m pytest -q
```

### 5.1 Run now using Phase 14 — GPU pilot

No Phase 15 training required:

```powershell
uv run --extra gpu python -m scripts.phase16_evaluate `
  --candidate runs/phase14_main/seed_52/ppo/best.pt `
  --opponents random,calling_station,tight,pot_pressure `
  --groups 10 --seed 16000000 --mode sampled --device cuda `
  --output-json runs/phase16/phase14_ppo_pilot.json `
  --output-csv runs/phase16/phase14_ppo_pilot.csv `
  --output-md runs/phase16/phase14_ppo_pilot.md
```

4 opponents × 10 groups × 4 layouts = **160 evaluation hands**. No gradient updates.

### 5.2 Frozen checkpoint opponent

```powershell
uv run --extra gpu python -m scripts.phase16_evaluate `
  --candidate runs/phase14_main/seed_52/ppo/best.pt `
  --opponents random,calling_station,tight,pot_pressure `
  --opponent-checkpoint qboost51=runs/phase14_main/seed_51/qboost/best.pt `
  --groups 100 --seed 16010000 --mode sampled --device cuda `
  --output-json runs/phase16/phase14_ppo_full.json `
  --output-csv runs/phase16/phase14_ppo_full.csv `
  --output-md runs/phase16/phase14_ppo_full.md
```

5 opponents × 100 groups × 4 = **2,000 hands**.

### 5.3 Later: Phase 15 best checkpoint

After finishing Phase 15 and before or after its independent final report:

```powershell
uv run --extra gpu python -m scripts.phase16_evaluate `
  --candidate runs/phase15_ppo_main/best.pt `
  --groups 100 --seed 16100000 --mode sampled --device cuda `
  --output-json runs/phase16/phase15_ppo_eval.json `
  --output-csv runs/phase16/phase15_ppo_eval.csv `
  --output-md runs/phase16/phase15_ppo_eval.md
```

**Do not reuse final held-out Phase 15 test seeds to tune future models.** Use Phase 16 as a diagnostics/validation bank, or choose a truly fresh holdout at the end.

### 5.4 Fair PPO vs Q-Boost comparison

Run separate Phase 16 evaluations of both candidates with **identical opponent pool, mode and seed interval**, but **different output filenames**. Their descriptive bb/100 values can be compared; for a hypothesis test, use a *paired block-level comparison* with independent Training Seeds (as in Phase 14). Phase 16 currently does not provide paired cross-candidate confidence intervals.

## 6. Output

```text
runs/phase16/
  phase14_ppo_pilot.json    # detailed results and nested street/position/SPR metrics
  phase14_ppo_pilot.csv     # one row per opponent, performance & action summary
  phase14_ppo_pilot.md      # Obsidian-ready report with tables and formulas
```

The outputs deliberately **do not contain raw hands, player hole cards or critic state**. Paths must not already exist: each benchmark run uses distinct output files to prevent accidental overwrite.

## 7. Development validation and limitations

Development regression was executed in a Linux CPU **mock Phase3 engine** harness. Run both test commands on the real Windows HUNL200 engine and GTX1070 before interpreting the findings. CUDA-only test is skipped on systems without CUDA.

An observed positive bb/100 only measures performance against the selected opponents on the selected seed blocks; it is not proof of GTO strength or exploitation resistance. This Phase is not CFR, Nash solving, meta-game policy training, or adaptive opponent modeling.
