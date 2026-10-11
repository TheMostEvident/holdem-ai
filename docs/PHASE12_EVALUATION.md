# Phase 12 — Reproducible Seed-Blocked HUNL200 Evaluation

> Project: TheMostEvident/holdem-ai. Additive patch only; Phase 1–11 remain unchanged.
>
> Important: a match against RandomLegalPolicy or a frozen Actor is **NOT** GTO exploitability.

## 1. Goal and current limitations

Phase 11 reused exactly the same 200 evaluated hands at every update. Its greedy evaluations frequently returned exactly the same `frozen_anchor` score, even when the trained weights changed. This is not necessarily a code defect: greedy `argmax` decisions often stay unchanged after small parameter updates. A normal CI that treats each game as independent also ignores correlations produced by using the same deck across seat-swapped comparisons.

Phase 12 adds a **separate post-training evaluator**. It does not change the training loop or any of the Phase-11 checkpoints. It supports:

- Common shuffled-deck seed, swapped candidate/opponent physical seats, and both button placements.
- Group-level confidence intervals; **one shuffled-deck seed is the statistical unit**.
- Greedy and sampled policies, with reproducible random seeds.
- RandomLegalPolicy or independent frozen checkpoint as opponent.
- `--control`: paired difference of candidate and control, evaluated with the same seed groups and opponent.
- `validation` / `test` disjoint seed partitions, per-action counts, per-group CSV, JSON summary, model weight fingerprint.
- Progress bar with ETA via the existing optional `tqdm` package (`--no-progress` to hide).

## 2. Mathematical specification (Obsidian Markdown)

For each independent shuffle seed $k$, play four completed HUNL hands indexed by button seat $b\in\{0,1\}$ and candidate physical seat $s\in\{0,1\}$. Both seat assignments at fixed $(k,b)$ use **the same shuffled deck**. Candidate returns are in BB, are zero-sum against the opponent, and include fold/all-in outcomes.

$$
X_k^A=\frac{1}{4}\sum_{b=0}^1\sum_{s=0}^1R_A(k,b,s).
$$

Each $X_k$ is one *seed block*, not four statistically independent observations. Estimate

$$
\hat J_A=100\bar X^A,\qquad \bar X^A=\frac1K\sum_{k=1}^K X_k^A.
$$

$$
\hat\sigma_X^2=\frac{1}{K-1}\sum_{k=1}^K(X_k^A-\bar X^A)^2,
\qquad \operatorname{SE}(\hat J_A)=100\sqrt{\frac{\hat\sigma_X^2}{K}}.
$$

$$
\operatorname{CI}_{95\%}(J_A)\approx \hat J_A\pm1.96\operatorname{SE}(\hat J_A).
$$

For two agents A and C evaluated against **the same fixed opponent B**, using **the same shuffled-deck seeds** and modes:

$$
D_k=X_k^A-X_k^C,\qquad
\widehat{\Delta J}_{A-C}=100\bar D.
$$

Compute the confidence interval from **the sample variance of $D_k$**, not by subtracting unrelated confidence intervals.

These are approximate normal intervals across independent seed blocks, requiring enough seeds for meaningful inference. Prefer $K\ge50$ for development and more for held-out final evaluation. Poker hand returns are bounded but very high-variance, so even $K=50$ is not a claim of narrow uncertainty. Confidence intervals do not correct for *repeatedly choosing policies based on the same evaluation partition*.

### Duplicate caveat

This is a *seed-matched, seat-swapped benchmark*, inspired by duplicate poker; it **does not force all future random events to match under different action histories**. Both evaluations at fixed $(k,b)$ start with the same deck shuffled by the Phase-3 engine, while the candidate and opponent exchange cards/position across seat assignments. A policy change can cause betting to end sooner or later. The method reduces variance and balances seat effects, but it does not synthesize a counterfactual outcome for unplayed actions.

**Sanity check**: Against an identically weighted independent clone of itself, the candidate must obtain exactly zero group payoff in both greedy and seeded-sampled mode, because the same physical-seat game is replayed with the same policies after swapping their *labels*. A nonzero block is a warning of asymmetry or randomness drift.

## 3. Installation (PowerShell)

Extract `holdem-ai-phase12-evaluation.zip` into the existing project root as `holdem-ai-phase12-evaluation`.

```powershell
Copy-Item .\holdem-ai-phase12-evaluation\holdem_agent_phase12 .\ -Recurse -Force
Copy-Item .\holdem-ai-phase12-evaluation\tests\test_phase12_*.py .\tests\ -Force
Copy-Item .\holdem-ai-phase12-evaluation\scripts\phase12_evaluate.py .\scripts\ -Force
Copy-Item .\holdem-ai-phase12-evaluation\docs\PHASE12_EVALUATION.md .\docs\ -Force
Copy-Item .\holdem-ai-phase12-evaluation\docs\ROADMAP_PHASE12_TO_16.md .\docs\ -Force
```

No new package dependencies.

### Tests

```powershell
uv run --extra dev --extra gpu python -m pytest -q -k phase12
uv run --extra dev --extra gpu python -m pytest -q
```

## 4. Recommended evaluations

### A. Post-training versus Random, independent of Phase 11 evaluation seeds

```powershell
uv run --extra gpu python -m scripts.phase12_evaluate --candidate checkpoints/phase11_pilot.pt --groups 100 --partition validation --mode greedy --device cuda --output-json runs/phase12/greedy_random.json --output-csv runs/phase12/greedy_random.csv
```

100 groups * 4 seat/button combinations = 400 evaluated hands. This can take longer than the Phase11 batch benchmark because evaluation uses **sequential single-hand inference** to keep the correctness contract simple.

### B. Same candidate vs Phase11 frozen Anchor

```powershell
uv run --extra gpu python -m scripts.phase12_evaluate --candidate checkpoints/phase11_pilot.pt --opponent-checkpoint runs/phase11_pilot/anchor.pt --groups 100 --partition validation --mode sampled --device cuda --output-json runs/phase12/sampled_anchor.json
```

Greedy and sampled measurements answer different questions. Sampled draws use the provided deterministic match RNG and may reveal differences hidden by greedy argmax.

### C. PPO control comparison against identical Random Agent / deck seeds

```powershell
uv run --extra gpu python -m scripts.phase12_evaluate --candidate checkpoints/phase11_pilot.pt --control checkpoints/phase11_ppo.pt --groups 100 --partition validation --mode sampled --device cuda --output-json runs/phase12/paired_qboost_minus_ppo.json
```

The PPO control path must already exist, and the PPO architecture can differ from the candidate's, because the evaluation feeds each model its own valid observation tokens.

### D. Held-out test partition

After hyperparameters are fixed using `validation`, perform **a single final** evaluation on the disjoint `test` seeds:

```powershell
uv run --extra gpu python -m scripts.phase12_evaluate --candidate checkpoints/phase11_pilot.pt --groups 500 --partition test --mode sampled --device cuda --output-json runs/phase12/heldout_test.json
```

Avoid inspecting the test scores repeatedly and then selecting hyperparameters on them; that would make the test partition no longer independent. Use independent evaluation runs/seeds to replicate findings.

## 5. Summary outputs

- `mean_bb100`: average over independent four-hand seed blocks, times 100.
- `ci95_lower_bb100`, `ci95_upper_bb100`: approximate seed-block CI.
- `standard_error_bb100`: standard error over seed blocks.
- `actor_actions`, `opponent_actions`: counts per seven abstract actions; monitor policy collapse/greedy action thresholds.
- `candidate_id`, `opponent_id`: reproducible SHA-256 prefix of model weights, not a network privacy hash.
- `paired_comparison.delta_mean_bb100`: candidate minus control, evaluated against the same opponent and seed blocks.

For no opponent checkpoint, `RandomLegalPolicy` is used. Evaluation uses only public `InformationState`, never `training_state`. GPU needs the already-installed Torch GPU extra. This patch does **not** change the Phase11 training progress bar, self-play collector, model, optimizer, or checkpoint format.

## 6. Engineering validation and limitations

The authoring environment tested this patch on PyTorch CPU with a simulated Phase-3 engine. The actual Windows HUNL200 engine and GTX1070 must be verified with the two Windows test commands before committing. CUDA unit tests will run only where CUDA is available.

Once Phase12 passes on the target PC, commit *only* the new files:

```powershell
git add holdem_agent_phase12/ tests/test_phase12_*.py scripts/phase12_evaluate.py docs/PHASE12_EVALUATION.md docs/ROADMAP_PHASE12_TO_16.md
git diff --cached --stat
git commit -m "Add Phase 12 paired seed-block HUNL evaluation"
git push origin main
```

Do not `git add .` or put raw `runs/`, training `.pt` checkpoints, or extracted ZIP folders into Git.
