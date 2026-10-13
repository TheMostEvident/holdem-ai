# Source documentation volume 4/7

[Final integrated technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md) · Lossless original report content from `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`.

## docs/PHASE15_TERMINAL_REWARD_AUDIT.md

Original SHA `a0bac55cc794db1f442b1fc9c0f389b25935bac4` (4206 bytes).

<a id="md-041"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE15_TERMINAL_REWARD_AUDIT.md" sha="a0bac55cc794db1f442b1fc9c0f389b25935bac4" -->

````````markdown
# Phase 15 — Terminal Reward Distribution and Preflop Hand Exit Audit

Scope: **Agent diagnostics only**. The HUNL engine, GUI, PPO optimizer, past checkpoints, runs and safety settings remain untouched.

## Research question

The previous Critic evaluation returned a terminal-return standard deviation near 193–196 BB. Over 93% of Actor *decisions* were preflop. Neither result alone proves that most hands terminate on preflop, or that nearly all rewards are ±200BB. This per-hand audit answers:

1. What proportion of hands has |R₀| at or above 190, 199 and 200 BB?
2. What fraction of hands ends immediately following a **preflop last Actor decision**?
3. Of these hands, how many terminate via **Fold** vs non-Fold showdown (including automatic board runout)?
4. How often does explicit preflop All-in (Action index 6) occur?
5. Are full-stack payoffs concentrated in particular termination categories?
6. How different are **hand-weighted** rewards and **decision-weighted** PPO training targets?

### Distributions

Hand-weighted payoffs:

\[
D_\mathrm{hand}=\{R_{0,h}\}_{h=1}^H,\qquad
\mu_\mathrm{hand}=\frac{1}{H}\sum_h R_{0,h}.
\]

Decision-weighted acting-seat PPO targets repeat terminal rewards within each hand:

\[
D_\mathrm{decision}=\{G_{h,t}=R_{\mathrm{seat}(h,t),h}\}_{h,t}.
\]

The denominators differ: hands containing many decisions contribute more PPO targets than one-action hands.

### Termination semantics

- **Preflop last decision + Fold:** a player folded without proceeding to a new Actor decision on a later street.
- **Preflop last decision + non-Fold:** a subsequent terminal showdown can have an automatically completed River board. This does not mean showdown occurred on a preflop board.
- **Explicit Preflop All-in:** Actor head action index 6 selected preflop. A CALL may exhaust all remaining stack chips as well, so this is an **under-count** of true all-in commitments.
- Classification uses existing per-hand decision boundaries and final action in the Agent Collector. The verified HUNL heads-up engine resolves non-fold terminal states by showdown and has no separate terminal reason field.
- The engine and its actor-observation path are not modified. No private cards are passed into the actor.

### Report

- Seat-0 hand-weighted: payoff mean, standard deviation, min/max, absolute quantiles, histogram, near-full-stack counts.
- Actor-decision-weighted returns: analogous summary with the different denominator.
- Counts by last-decision Street × Fold/Showdown.
- Conditional payoff distributions for Preflop Fold, Preflop Showdown, Postflop Fold and Postflop Showdown.
- Explicit preflop/all-street All-in, terminal action counts, per-street decision coverage.

## Windows PowerShell

On the same diagnostics branch:

~~~powershell
git pull --ff-only origin agent/phase15-collapse-diagnostics
uv run --extra gpu pytest tests/test_phase15_terminal_reward_audit.py -q
uv run --extra gpu pytest -q
~~~

If tests pass, execute this **read-only** audit:

~~~powershell
uv run --extra gpu python -m scripts.phase15_terminal_reward_audit `
  --before runs/phase15_ppo_safety_v2_pilot/best.pt `
  --after runs/phase15_ppo_safety_v2_pilot/latest.pt `
  --device cuda `
  --hands 512 `
  --seed 580000000 `
  --output runs/phase15_audit/safety_v2_terminal_rewards.json
~~~

Inspect relevant parts of the resulting JSON:

~~~powershell
$r = Get-Content .\runs\phase15_audit\safety_v2_terminal_rewards.json -Raw | ConvertFrom-Json
$r.models.before.seat0_hand_weighted_reward
$r.models.after.seat0_hand_weighted_reward
$r.models.before.preflop
$r.models.after.preflop
$r.models.before.payoff_by_exit_group
$r.models.after.payoff_by_exit_group
~~~

The CLI creates a new report and refuses overwrite. It does not train, save checkpoints, alter old metrics, or finalize evaluation.

### Limits

Each policy's independent sample uses the same deck-seed range, but different sampled actions may lead to different hand trajectories. This is descriptive, not a causal or exploitability test. There may be very few postflop decisions because a preflop All-in allows an automatic River board runout. No new PPO Hyperparameter or baseline architecture is changed here.

````````

<!-- END_MD_SOURCE path="docs/PHASE15_TERMINAL_REWARD_AUDIT.md" -->

## docs/PHASE15_THREE_POLICY_COMPARISON.md

Original SHA `8824c060b8ce7ef01852809fa00bdde489a4d174` (3940 bytes).

<a id="md-042"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE15_THREE_POLICY_COMPARISON.md" sha="8824c060b8ce7ef01852809fa00bdde489a4d174" -->

````````markdown
# Phase 15 — Three-Policy Preflop All-in Baseline Comparison

**Agent Task / Read-only**. No engine/GUI changes, no PPO training updates, no old-run/checkpoint mutation.

## Hypotheses

Both Phase14 Update104 and Phase15 Best Update125 self-play contain ~84% of hands with explicit Preflop All-in. Does this occur without PPO optimization?

We compare three self-play policies, always same policy on **both seats**:

1. **uniform_legal:** choose uniformly from the current legal 7-head action indices. For legal actions, probability = 1 / number-of-legal-actions.
2. **random_init:** a fresh initialized TransformerActor using the trained Phase14 architecture, with a reproducible initialization seed (default 52). This is *NOT* necessarily the actual first weights used in Phase14. An optional real initial checkpoint can replace this synthetic control.
3. **trained_phase14:** frozen trained Phase14 Update104 actor.

Deck and action sampling seed blocks are the same across policies, but action sequences and states are not identical. No cross-policy matchup or hand EV is inferred.

### Distinct denominators

Hand-level Preflop All-in occurrence:

\[
p_{\mathrm{hand}} =
\frac{\#\{\text{hands with at least one explicit Preflop action 6}\}}{H}.
\]

Action frequency conditional on All-in legality:

\[
p_{\mathrm{decision}\mid\mathrm{legal}} =
\frac{\#\{\text{Preflop decisions selecting action 6}\}}
{\#\{\text{Preflop decisions where action 6 is legal}\}}.
\]

The second metric is NOT the first. An ordinary CALL can exhaust the final stack without explicit action index 6.

The audit also measures first-decision all-in rates, action counts by street, per-hand any-All-in, payoff tails, Preflop exits and Flop/Turn/River decision coverage.

### Fixed-state experiment

One Phase14 reference policy rollout supplies exactly the same Actor-observable states to all three policies. The tool evaluates average expected All-in probabilities for states where All-in is legal:

\[
\bar\pi_m =
\frac{1}{|D^{\mathrm{Phase14}}_{\mathrm{allin\ legal}}|}
\sum_{I\in D^{\mathrm{Phase14}}_{\mathrm{allin\ legal}}}
\pi_m(\mathrm{allin}\mid I).
\]

This controls visited observations, but the distribution of sampled states is still Phase14-dependent.

## Windows PowerShell

~~~powershell
git pull --ff-only origin agent/phase15-collapse-diagnostics
uv run --extra gpu pytest tests/test_phase15_initial_policy_comparison.py -q
uv run --extra gpu pytest -q
~~~

After tests pass, execute the command (one line shown for copy safety):

~~~powershell
uv run --extra gpu python -m scripts.phase15_initial_policy_comparison --trained runs/phase14_main/seed_52/ppo/best.pt --init-seed 52 --device cuda --hands 512 --seed 590000000 --output runs/phase15_audit/phase14_uniform_initial_comparison.json
~~~

Use the optional --initial-checkpoint PATH only if a **verified real initialization checkpoint** is available. This suppresses the fresh synthetic initialization.

Output is new-only (exclusive creation), read-only with respect to checkpoints, old runs and metrics.

## How to interpret

| Pattern | Priority |
| --- | --- |
| Uniform legal and random-init also show high per-hand All-in | Check action-abstraction branching and compounding across decisions. PPO is not sufficient to explain it. |
| Uniform low, random-init high | Investigate untrained logits, random seeds, policy-head initialization. |
| Both low, trained Phase14 high | Inspect earlier training checkpoints and PPO advantage / rewards. |
| Hand-level All-in high but conditional action rate low | Multiple All-in opportunities increase probability of observing at least one per hand. |
| Fixed-state probabilities similar but on-policy outcomes differ | State visitation, action sequencing and opponent response are relevant. |

Do not impose arbitrary All-in penalties or constraints from this audit. Uniform Legal is a diagnostic baseline, not a competitive HUNL opponent.

````````

<!-- END_MD_SOURCE path="docs/PHASE15_THREE_POLICY_COMPARISON.md" -->

## docs/PHASE16_LONG_STRESS_TEST.md

Original SHA `1d283f7fa49d15ea2c80b4a44acc7326da9212f9` (9005 bytes).

<a id="md-043"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE16_LONG_STRESS_TEST.md" sha="1d283f7fa49d15ea2c80b4a44acc7326da9212f9" -->

````````markdown
# Phase 16 — 100-Update A0/A3 Long Stability Stress Test

**Status: executable implementation, NOT yet locally GPU-validated.** This remains an **Agent-only** research-branch experiment. The game engine, betting rules, GUI, original PPO implementation, Phase14/15 checkpoints and existing logs are unchanged.

## What is being tested?

The two Phase16 10-update pilot seed runs found that scaled-Q (A1) and Direct-V (A2) were not demonstrably better than A0. Even though exact *post-update* KL was small over 10 updates, that does NOT establish long-run stability.

The remaining controlled arms are:

| Arm | Training | Safety |
| --- | --- | --- |
| **A0** | Existing original PPO clipped-surrogate, Huber centralized Q, Actor Adam and source Actor weights | Measure exact post-update KL; optional emergency stop at KL > 0.10 (model already saved for inspection) |
| **A3** | Identical original PPO and Q-Critic | If exact post-update KL > 0.02, roll back Actor, Critic, and BOTH Adam Optimizer states for that proposed update |

**Both arms** start with the same trusted Phase14 full PPO checkpoint, retain its Actor Adam moments, and RESET Critic Adam moments as in the previous Phase16 A0/A3 ablation. All scientific configs must match between the arms: 128 hands/update, 100 attempted updates, one PPO epoch, identical card seed blocks and seat/button alternation. **Only A3's KL rejection gate changes.**

\[
N_{\mathrm{train,arm}}=128\times100=12{,}800\ \mathrm{hands}.
\]

\[
N_{\mathrm{train,total}}=2\times12{,}800=25{,}600\ \mathrm{hands}.
\]

A3 *rejected attempts still consume their card-seed blocks*, but roll back the entire optimizer/model update. This protects on-policy data integrity. Thus matched seeds between arms only guarantee matched **initial decks**, not that both policies visit the same states after diverging.

### Exact Post-Update KL

\[
\widehat{D}_{KL}(\pi_{\mathrm{old}}\|\pi_{\mathrm{new}})=
\frac{1}{N}\sum_{I \in D_{\mathrm{old}}}
\sum_{a\in A_{\mathrm{legal}}(I)}
\pi_{\mathrm{old}}(a|I)\log\frac{\pi_{\mathrm{old}}(a|I)}{\pi_{\mathrm{new}}(a|I)}.
\]

The gate applies to the **batch-average** divergence on sampled on-policy states. It is NOT a guarantee of bounded worst-case divergence for all poker information states.

## Durability/resume semantics

Each attempted update is committed with a single **atomic latest.pt** containing:

- the Actor and Critic parameters,
- both Adam Optimizer states and Torch CPU RNG state,
- the full PPOConfig and global completed-update count,
- attempted/accepted/rejected update accounting,
- per-update entropy, policy/critic losses, gradient norms, KL, all-in/postflop statistics,
- periodic validation reports and Critic holdout metrics.

The full checkpoint remains Phase9-compatible for load_checkpoint() and for a future separate training resume.

Additional immutable snapshots are retained every 10 attempted updates at `snapshots/attempt_00010.pt`, etc. The authoritative state is **latest.pt**; JSON metrics/progress files are regenerated from the committed ledger during Resume.

If a process terminates *after* latest.pt was committed but before metrics.jsonl is updated, Resume reconstructs logs. If it terminates after an update at a scheduled evaluation boundary but before evaluation finishes, Resume executes the missing validation first. It does **not** replay previously committed training hands.

- `--updates 100` is the TOTAL number of **attempted** updates SINCE the original source checkpoint, not a further 100 after Resume.
- `--max-updates-this-call 20` optionally processes at most 20 attempts per invocation. When resuming, use the *same scientific arguments* plus `--resume`. The max-per-call value may change.
- Runs reject overwritten roots, mismatched configs, mutated source SHA, and unsafe A0 resume after a hard-stop event.
- The checkpoint and evaluation metrics **do not prove** that strategy improved or converged.

## Evaluation protocol

At start, save the untouched Phase14 source Actor's performance against (1) uniform Random Legal and (2) a frozen source Actor clone.

Every 20 attempted updates (and at the end), evaluate current Actor against both fixed opponents using **four seat/button layouts per deck seed**, reporting paired change from the original Actor and block-level approximate 95% CIs:

\[
\Delta J_{k}=\frac{1}{4}\sum_{j=1}^4
\left(R_{\mathrm{candidate},k,j}-R_{\mathrm{source},k,j}\right),
\]

\[
\widehat{\Delta\mathrm{bb}/100}=100\cdot
\frac{1}{K}\sum_{k=1}^{K}\Delta J_k.
\]

Also collect an independently seeded on-policy Critic/behavioral diagnostic: decision-weighted Critic RMSE/EV; hand-weighted Preflop explicit All-in, Flop decision coverage, Fold exits, and full-stack reward tails.

Use the SAME 40 independent validation seed groups for A0/A3 to enable paired A3-minus-A0 comparisons. However, repeated reuse of these validation seeds to choose a winning checkpoint induces selection bias; a final truly held-out test would be required for claims of superiority.

## FIRST: run local tests

~~~powershell
git pull --ff-only origin agent/phase15-collapse-diagnostics
uv run --extra gpu pytest tests/test_phase16_ppo_stress.py -q
uv run --extra gpu pytest -q
~~~

Do not proceed if either command fails.

## Smoke test (2 additional updates per arm; NEW root)

~~~powershell
uv run --extra gpu python -m scripts.phase16_ppo_stress campaign --source runs/phase14_main/seed_52/ppo/best.pt --root runs/phase16_stress_smoke_01 --device cuda --updates 2 --hands-per-update 8 --minibatch-size 8 --epochs 1 --env-batch-size 4 --eval-every 1 --eval-groups 2 --critic-eval-hands 8 --save-every 1
~~~

Confirm `runs/phase16_stress_smoke_01/paired_comparison.json` exists and both arms show `completed: true`.

## FULL 100-Update GPU campaign (NEW root)

One command runs A0 (100 updates) then A3 (100 updates) and automatically compares their aligned validations:

~~~powershell
uv run --extra gpu python -m scripts.phase16_ppo_stress campaign --source runs/phase14_main/seed_52/ppo/best.pt --root runs/phase16_stress_100_seed_01 --device cuda --updates 100 --hands-per-update 128 --minibatch-size 128 --epochs 1 --env-batch-size 32 --eval-every 20 --eval-groups 40 --critic-eval-hands 128 --rollout-seed 680000000 --validation-seed 78000000 --critic-seed 690000000 --max-post-kl 0.02 --hard-stop-kl 0.10 --save-every 10
~~~

If interrupted, run **the same command** with `--resume` appended; it reuses no already-committed rollout seed.

### Optional segmented run (to pause at 20 updates/arm per invocation)

On the first invocation, add `--max-updates-this-call 20`. On subsequent invocations, append `--resume` and keep the same `--max-updates-this-call 20`. The campaign finishes automatically as soon as both arms reach 100.

Do NOT launch two processes for the same arm root; process-level locking prevents this.

## Important outputs

~~~text
runs/phase16_stress_100_seed_01/
  a0/
    run.json
    latest.pt                # FULL PPO/Adam/RNG/ledger
    metrics.jsonl
    progress.json
    summary.json
    snapshots/attempt_00010.pt, ..., attempt_00100.pt
  a3/
    run.json
    latest.pt
    metrics.jsonl
    progress.json
    summary.json
    snapshots/attempt_00010.pt, ..., attempt_00100.pt
  paired_comparison.json     # A3 - A0 paired deck-group scores
~~~

Read concise outcome after completion:

~~~powershell
$c = Get-Content .\runs\phase16_stress_100_seed_01\paired_comparison.json -Raw | ConvertFrom-Json
$c.a0_attempted
$c.a3_attempted
$c.a0_rejected
$c.a3_rejected
$c.paired_validations.'100'.frozen_phase14 | Format-List
$c.paired_validations.'100'.random_legal | Format-List
~~~

Inspect individual update metrics and validation:

~~~powershell
$a0 = Get-Content .\runs\phase16_stress_100_seed_01\a0\progress.json -Raw | ConvertFrom-Json
$a3 = Get-Content .\runs\phase16_stress_100_seed_01\a3\progress.json -Raw | ConvertFrom-Json
$a0, $a3 | Format-List
~~~

**Important:** no tool in this chat can execute a 100-update program on your personal Windows GTX1070/RTX5060. The preceding GitHub commits are executable code; passing your local tests and GPU launch is still required before claiming empirical results or stability.

## Criteria to allow formal GPU retraining

- No NaN/Inf, corrupted checkpoint, seed reuse, KL logging mismatch, or broken full-state Resume.
- A3 gate must reject a forced unsafe step in testing. Real-world zero rejections mean **A3 == A0** in that run, not that A3 fixed collapse.
- Plot/compare distributions of entropy, exact KL, Q-Critic RMSE and EV, Preflop All-in, postflop coverage across 100 updates, not just final values.
- Compare A3 and A0 on *paired* held-out deck groups vs **both** opponents, report uncertainty. Prefer repeating 100-update stress with an independent train/eval/critic seed family before promotion to full GPU retraining.
- Judge actual poker profitability, not an arbitrary target for fewer All-ins; avoid claiming HUNL exploitability or GTO from small sampled matches.

````````

<!-- END_MD_SOURCE path="docs/PHASE16_LONG_STRESS_TEST.md" -->

## docs/PHASE16_OPPONENT_POOL_AND_STRATEGY.md

Original SHA `fed9a3b8906193aebc0899ce4966fff5c9a1ed38` (8643 bytes).

<a id="md-044"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE16_OPPONENT_POOL_AND_STRATEGY.md" sha="fed9a3b8906193aebc0899ce4966fff5c9a1ed38" -->

````````markdown
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

````````

<!-- END_MD_SOURCE path="docs/PHASE16_OPPONENT_POOL_AND_STRATEGY.md" -->

## docs/PHASE16_PPO_ABLATION.md

Original SHA `508e9b335a7a64c2ff31ade25e9b68b86f1fbe10` (8324 bytes).

<a id="md-045"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE16_PPO_ABLATION.md" sha="508e9b335a7a64c2ff31ade25e9b68b86f1fbe10" -->

````````markdown
# Phase 16 — PPO Optimizer/Critic Ablation Before GPU Retraining

Status: ablation implementation, **not** demonstration that learning is stable. On research branch `agent/phase15-collapse-diagnostics`. **Agent-only**. HUNL engine, game rules, GUI, existing PPO/QBoost training functions, Phase14/15 checkpoints, earlier runs and final held-out seeds are unchanged.

## Motivating measurements from completed Phase15

- Preflop explicit All-in appears in 439/512 = 85.74% of Phase14 on-policy hands (another independent seed block).
- Dominant First-All-in sequence: Pot Raise → Double-Pot Raise → Double-Pot Raise → All-in, observed 115 times.
- Terminal payoff is commonly ±200BB; previous Critic on-policy out-of-sample Q prediction was almost constant near 0BB, with EV around zero.
- Earlier PPO Approx KL was computed before the minibatch optimizer step; it is not a reliable **post-update drift** statistic.

These observations do not prove that Adam or PPO objective is incorrect. This experiment changes one factor at a time.

## Arms

| Arm | Actor PPO | Critic/target | Additional safety |
| --- | --- | --- | --- |
| A0 | original clipped surrogate, entropy coefficient, advantage normalization | original privileged Q, Huber target in BB | measure exact post-update KL only |
| A1 | same PPO update formula | privileged Q in 200BB scaled units: target G/200, Q head reparameterized from original to Q/200 | measure exact post-update KL |
| A2 | same PPO update formula | privileged direct V(s), warm-starting token+hidden layers from Q; projected output initialized from mean Q heads; Huber raw BB target | measure exact post-update KL |
| A3 | original A0 PPO update | original privileged Q, Huber raw BB target | **rollback entire update** if exact KL(old||new) exceeds limit |

### Scaling and gradient

\[
G^\prime=G/200,\qquad
L_Q=\mathrm{SmoothL1}(Q^\prime_\phi,G^\prime),\qquad
A^\prime=G^\prime-\sum_a\pi_{\mathrm{old}}(a\mid I)Q^\prime_\phi(s,a).
\]

\[
\hat A^\prime = \frac{A^\prime-\bar A^\prime}
{\mathrm{Std}(A^\prime)+\epsilon}.
\]

A0/A3 use the original unscaled BB return and old privileged-Q state baseline.

A2 replaces the action-conditional critic with an **action-independent privileged baseline**:

\[
V_\phi(s,i) = \begin{cases}
v_\phi(s),& i=0,\\
-v_\phi(s),&i=1.
\end{cases}
\qquad A=G_i-V_\phi(s,i).
\]

The centralized critic can observe both private hands **for training only**; the Actor never receives that information. The V model has fewer output parameters than the 7-action Q head: this is an architectural change, not a pure optimization-hyperparameter test.

A3 computes exact categorical divergence of distributions on **the frozen old-policy rollout states**:

\[
\widehat D_{\mathrm{KL}} =
\frac{1}{N}\sum_{n=1}^N
\sum_{a\in\mathcal A(I_n)}
\pi_\mathrm{old}(a\mid I_n)
\log \frac{\pi_\mathrm{old}(a\mid I_n)}
{\pi_\mathrm{new}(a\mid I_n)}.
\]

If KL > cutoff, restore Actor, Critic, Actor Adam moments and Critic Adam moments from immediately BEFORE that attempted update. This **sample-based** gate is not a global trust-region or formal PPO convergence guarantee.

## Fairness controls

- All arms start from exactly the SAME trusted Phase14 PPO checkpoint and original Actor Adam state.
- **Every arm resets Critic Adam moments**, avoiding old optimizer moments inconsistent with the scaled and changed-shape critic. Thus A0 is a fresh-Critic-Adam continuation of the original *mathematical* PPO optimizer, not a bit-for-bit Phase15 resume.
- Same number of sampled hands, attempted updates, PPO epochs, minibatches, model architecture (except V head), card seed block and seat/button alternation for every arm.
- A3 can **accept fewer updates** because it rolls back unsafe proposals; attempted sample budget still matches.
- Identical separate valuation deck groups, separate critic holdout hand seed block. Frozen Phase14 and Random Legal are both used as opponents.
- The same four-layout seat/button balanced benchmark is used to compute per-deck paired \(\Delta\mathrm{bb}/100\) CIs vs the unchanged initial Phase14 Actor.

Evaluation CIs can be extremely wide at 12 groups; pilot results only justify a larger confirmatory run, not an inference of poker strength. The Critic holdout metrics are on-policy **decision-weighted** (correlated within hands), so no independent-row confidence intervals are implied.

## Safety and data integrity

- Run root must be a brand NEW empty path. A completed run cannot be silently overwritten, and auto-resume is not supported by this pilot.
- Saved **final_actor.pt** is a checkpoint for evaluating Actor weights only. It is deliberately **NOT** a PPO resume checkpoint; it lacks both Optimizer states and the Critic.
- Existing best/latest/source checkpoints are never rewritten or read via a partially saved path.
- Per-arm result JSON and final summary JSON are new-only. A partial run must remain intact; restart under a NEW root.
- Phase15 final held-out card seeds are **not** used for ablation selection.

## Windows PowerShell: first validate

~~~powershell
git pull --ff-only origin agent/phase15-collapse-diagnostics
uv run --extra gpu pytest tests/test_phase16_ppo_ablation.py -q
uv run --extra gpu pytest -q
~~~

## Smoke test (32 hands x 2 attempted updates x 4 arms)

~~~powershell
uv run --extra gpu python -m scripts.phase16_ppo_ablation --source runs/phase14_main/seed_52/ppo/best.pt --root runs/phase16_ablation_smoke_01 --device cuda --variants a0,a1,a2,a3 --hands-per-update 32 --updates 2 --minibatch-size 32 --epochs 1 --env-batch-size 16 --eval-groups 4 --critic-eval-hands 32 --rollout-seed 620000000 --validation-seed 75000000 --critic-seed 660000000 --max-post-kl 0.02
~~~

Each policy has its **own** rollout opponent (itself), and evaluation uses two specified opponents. No model change is allowed during evaluation. Smoke results primarily validate logging, shapes, NaNs, source immutability and A3 gate correctness.

## Exploratory pilot after smoke passes (new run root, not reuse)

~~~powershell
uv run --extra gpu python -m scripts.phase16_ppo_ablation --source runs/phase14_main/seed_52/ppo/best.pt --root runs/phase16_ablation_pilot_seed_01 --device cuda --variants a0,a1,a2,a3 --hands-per-update 64 --updates 10 --minibatch-size 64 --epochs 1 --env-batch-size 32 --eval-groups 32 --critic-eval-hands 128 --rollout-seed 621000000 --validation-seed 75100000 --critic-seed 661000000 --max-post-kl 0.02
~~~

Repeat with **independent rollout, critic and validation seed blocks** and another new Run root, preferably on another day or session. Never choose the final model from the sealed final evaluation partition.

## What to inspect

~~~powershell
$r = Get-Content .\runs\phase16_ablation_pilot_seed_01\summary.json -Raw | ConvertFrom-Json
$table = @(
  foreach ($name in @("a0", "a1", "a2", "a3")) {
    $v = $r.variants.$name
    $d = $v.evaluation.frozen_phase14.paired_change_from_source
    [PSCustomObject]@{
      Arm = $name
      CriticRMSEBB = $v.critic_holdout.rmse_bb
      CriticEV = $v.critic_holdout.explained_variance
      MaxPostKL = $v.post_kl_max
      Rejected = $v.rejected_updates
      PFAllIn = $v.on_policy_behavior.preflop_explicit_allin_fraction
      FlopCoverage = $v.on_policy_behavior.postflop_decision_hand_fraction
      DeltaVsSourceBB100 = $d.delta_mean_bb100
      CI95Low = $d.delta_ci95_lower_bb100
      CI95High = $d.delta_ci95_upper_bb100
    }
  }
)
$table | Format-Table -AutoSize
~~~

## Decision rules before GPU long-run retraining

1. No crashes, NaN/Inf, unexplained checkpoint mutation, or broken rollback.
2. Exact post-update KL is observed after the optimizer step, and bounded/rejected correctly in A3. A3 with zero rejections is **not evidence** it improved stability.
3. Compare Critic out-of-training-rollout RMSE and EV: a higher EV is not a sufficient condition for profitable play.
4. Compare paired matched-deck \(\Delta\mathrm{bb}/100\) against BOTH source and Random Legal, and discuss 95% CI uncertainty, not just point estimates.
5. Track postflop decision coverage and Preflop All-in without optimizing for a fixed arbitrary All-in quota.
6. Confirm promising candidates over independent rollout/evaluation seeds BEFORE longer GPU retraining. A short pilot cannot establish converged or Nash-equilibrium play.

````````

<!-- END_MD_SOURCE path="docs/PHASE16_PPO_ABLATION.md" -->

## docs/PHASE17_CONTINUOUS_STACK_EVALUATION.md

Original SHA `596dacd13bcbce6e9a54e8027fbf45c221b29f23` (8630 bytes).

<a id="md-046"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE17_CONTINUOUS_STACK_EVALUATION.md" sha="596dacd13bcbce6e9a54e8027fbf45c221b29f23" -->

````````markdown
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

````````

<!-- END_MD_SOURCE path="docs/PHASE17_CONTINUOUS_STACK_EVALUATION.md" -->

## docs/PHASE17_FIVE_HOUR_GPU_RETRAIN.md

Original SHA `330433ec78d33411571c6beb0b8a9c4792df7561` (7944 bytes).

<a id="md-047"></a>

<!-- BEGIN_MD_SOURCE path="docs/PHASE17_FIVE_HOUR_GPU_RETRAIN.md" sha="330433ec78d33411571c6beb0b8a9c4792df7561" -->

````````markdown
# Phase 17 — Monitored 5-hour HUNL PPO GPU Retrain

## Project decision and limitations

Phase15 diagnosed repeated preflop Pot/Double-Pot → All-in trajectories. Phase16 compared A0 original PPO, scaled-Q (A1), Direct-V (A2), and KL gate (A3). Independent 10-update seeds showed no reliable A1/A2 improvement. A0 and A3 completed 100 additional updates each with **zero A3 rejections**; matched-deck final A3 minus A0 was **0 bb/100** against both frozen Phase14 and Random Legal. Critic EV remained near 0. Thus **5 more hours is authorized research training, NOT established convergence**.

Start from the unmodified **Phase14 Update104** `runs/phase14_main/seed_52/ppo/best.pt` (trusted complete PPO checkpoint), NOT the Phase16 Update204 Actor. Keep the original Transformer Actor, centralized Q-Critic and unmodified PPO loss.

The default A0 continues source Actor and Actor Adam moments. As in the Phase16 A0/A3 ablation, **reset Critic Adam state** identically; this differs from a bit-for-bit continuation of original Phase14 Adam, and the protocol records it. A3 rollback remains an opt-in experiment; default A0 monitors exact KL and uses only an extreme emergency halt (>0.10).

No modifications to poker engine, betting rules, GUI, historical checkpoints, reports, existing Phase15/16 PPO functions, or prior training roots.

## Objective and diagnostics

\[
L_\mathrm{PPO} = -
\mathbb E_t\left[
\min(r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t)
\right]-\alpha H(\pi_\theta).
\]

\[
L_\mathrm{critic} = \operatorname{SmoothL1}
(Q_{\phi}(s_t, i_t, a_t), G_{i_t});
\quad \hat A_t =
\operatorname{Normalize}\left(G_{i_t}-\sum_a
\pi_\mathrm{old}(a\mid I_t)Q_{\phi_\mathrm{old}}(s_t,i_t,a)\right).
\]

Post-update exact empirical categorical KL:

\[
\widehat{\mathrm{KL}}_\mathrm{post} =
\frac{1}{N}\sum_{t=1}^N\sum_{a\in\mathcal A(I_t)}
\pi_\mathrm{old}(a\mid I_t)
\log\frac{\pi_\mathrm{old}(a\mid I_t)}
{\pi_\mathrm{new}(a\mid I_t)}.
\]

A0 observes KL after the optimizer has stepped. If KL > 0.10, it creates `hard_stop.json` and refuses automatic resume, preserving the latest checkpoint for human examination. **This is not an algorithmic fix to PPO**: the model has already completed the last step when the alarm fires. No automatic adjustment or remedial learning is made. With `--variant a3`, rollback is applied when KL > 0.02 (Actor, Critic and both Adam states). Only one variant per NEW run.

Log at each update: KL, entropy, clipped PPO diagnostic, Actor/Critic losses, gradient norms, explicit preflop All-in fraction, Flop decision coverage, stack reward tails and elapsed seconds. Every 25 updates, matched-deck validation includes 40/80 independent deal seeds (4 seat/button-balanced hands each, default 80) versus (1) Frozen Phase14 and (2) Random Legal, paired Δbb/100 and 95% CI, plus independent on-policy Critic RMSE/EV and behavioral diagnostics.

The same validation seed groups are reused across checkpoints. **Best checkpoint selection from repeated validation scores is exploratory**, not an unbiased performance claim. Keep a separately sealed final test for future candidate ranking.

## Time budget

`--duration-hours 5.0` means **approximately 5 hours of cumulative active program execution** across uninterrupted or multiple resumed sessions. Human downtime is NOT counted. Includes initial baseline evaluation, intermediate validations, checkpoint saving and the PPO training loop. Program stops between updates; a scheduled final evaluation after the last update may extend past this target; total runtime is not exactly 5:00:00.

Each complete update atomically saves the **entire original Phase9-compatible PPO checkpoint** with additional `retrain` ledger:

- Actor and centralized Q-Critic weights,
- Actor and Critic Adam Optimizer state,
- CPU + CUDA Torch RNG state,
- number of attempted/accepted/rejected updates,
- cumulative active elapsed time and scientific manifest SHA256,
- complete update metrics and validations.

`latest.pt` is authoritative. `best.pt` is an exploratory validation winner, updated atomically only on a better average paired change against the fixed opponents. `snapshots/attempt_00010.pt` etc. are immutable periodic checkpoint snapshots. `metrics.jsonl` and `progress.json` can be reconstructed from `latest.pt` after a process interruption.

All output folders must be NEW. Never delete existing runs to reuse a path. When Resume is requested, ALL scientific parameters must match the original manifest. Changing only `--max-updates-this-call` is allowed.

## Step 1. Pull and test — Windows PowerShell

~~~powershell
git pull --ff-only origin agent/phase15-collapse-diagnostics

uv run --extra gpu pytest tests/test_phase17_gpu_retrain.py -q
uv run --extra gpu pytest -q
~~~

**Do not launch 5 hours unless tests pass.** The code was written via GitHub; tests still need to be run on the local machine.

## Step 2. GPU smoke test with a NEW run

~~~powershell
uv run --extra gpu python -m scripts.phase17_gpu_retrain `
  --source runs/phase14_main/seed_52/ppo/best.pt `
  --root runs/phase17_ppo_5h_smoke_01 `
  --device cuda `
  --duration-hours 0.10 `
  --hands-per-update 8 `
  --minibatch-size 8 `
  --epochs 1 `
  --env-batch-size 4 `
  --eval-every 1 `
  --eval-groups 2 `
  --critic-eval-hands 8 `
  --max-updates 2 `
  --save-every 1
~~~

For smoke tests, reaching a `max_updates_ceiling` stop reason is expected; 5-hour cumulative completion is **not** expected. Check `latest.pt`, `best.pt`, `progress.json`, `metrics.jsonl`.

## Step 3. Five-hour GPU run — NEW root

After all tests and smoke pass:

~~~powershell
uv run --extra gpu python -m scripts.phase17_gpu_retrain `
  --source runs/phase14_main/seed_52/ppo/best.pt `
  --root runs/phase17_ppo_5h_seed_01 `
  --device cuda `
  --duration-hours 5.0 `
  --variant a0 `
  --hands-per-update 128 `
  --minibatch-size 128 `
  --epochs 1 `
  --env-batch-size 32 `
  --eval-every 25 `
  --eval-groups 80 `
  --critic-eval-hands 128 `
  --rollout-seed 700000000 `
  --validation-seed 80000000 `
  --critic-seed 710000000 `
  --max-post-kl 0.02 `
  --hard-stop-kl 0.10 `
  --save-every 10 `
  --max-updates 5000
~~~

Resume after Ctrl+C/power interruption using **exactly the same command plus `--resume`**, and verify no `hard_stop.json`. Never start the program concurrently with itself on the same root. Completed full-state checkpoints are not overwritten with a partial model.

## Compact progress/results inspection

~~~powershell
$root = ".\runs\phase17_ppo_5h_seed_01"
Get-Content "$root\progress.json" -Raw | ConvertFrom-Json | Format-List
Get-Content "$root\summary.json" -Raw | ConvertFrom-Json | Format-List
Get-Content "$root\metrics.jsonl" -Tail 4
~~~

To view all periodic evaluations without dumping large arrays:

~~~powershell
$records = Get-Content "$root\metrics.jsonl" |
  ForEach-Object { $_ | ConvertFrom-Json } |
  Where-Object { $_.kind -eq "validation" }
$records | ForEach-Object {
  [PSCustomObject]@{
    Attempt = $_.attempted
    GlobalUpdate = $_.global_update
    CriticEV = $_.critic_holdout.explained_variance
    CriticRMSE = $_.critic_holdout.rmse_bb
    PreflopAllIn = $_.behavior.preflop_explicit_allin_fraction
    FlopDecisions = $_.behavior.flop_decision_fraction
    DeltaVsPhase14BB100 = $_.opponents.frozen_phase14.paired_vs_source.delta_mean_bb100
    CI95Lo = $_.opponents.frozen_phase14.paired_vs_source.delta_ci95_lower_bb100
    CI95Hi = $_.opponents.frozen_phase14.paired_vs_source.delta_ci95_upper_bb100
  }
} | Format-Table -AutoSize
~~~

## End-of-run decision

Do NOT assume stable HUNL strategy if the 5-hour run finishes. Compare several snapshots on an independent final evaluation partition, and audit Critic calibration, probability of preflop All-in, and postflop visitation across time; wide paired CIs imply an inconclusive bb/100 result. Preserve all checkpoints and reports for inspection.

````````

<!-- END_MD_SOURCE path="docs/PHASE17_FIVE_HOUR_GPU_RETRAIN.md" -->

## docs/phase2_1_correction.md

Original SHA `c371132babdb894207b6b05a25e3cd007ce920a8` (2474 bytes).

<a id="md-073"></a>

<!-- BEGIN_MD_SOURCE path="docs/phase2_1_correction.md" sha="c371132babdb894207b6b05a25e3cd007ce920a8" -->

````````markdown
# Phase 2.1 — Corrected RLCard HUNL adapter

## Key goal

RLCard 1.2.0's upstream two-player game posts blinds as if a multiplayer game were in progress. `holdem-ai` fixes the actual chip posting and first actor inside a local subclass. It leaves the installed `.venv/` untouched.

The new `HeadsUpBettingRound` also enforces a full minimum raise and non-reopening after a short all-in, while retaining action IDs `0=FOLD`, `1=CHECK_CALL`, `2=RAISE_HALF_POT`, `3=RAISE_POT`, `4=ALL_IN`. For raise-to sizing, raise half-pot/pot is calculated from the pot after calling. The local `get_payoffs()` settles unmatched chips and ties with Phase 2's `pot_manager` and `hand_evaluator`.

## Rules

For Button `b` (player ID 0 or 1), big blind is `1-b`. Preflop Button acts first, postflop big blind acts first.

```text
preflop:   BTN/SB -> BB
postflop:  BB -> BTN
```

## Running

```powershell
uv sync --extra dev
uv run --extra dev python -m pytest -q
uv run python scripts/audit_corrected.py --hands 100 --seed 42
uv run python main.py --hands 5 --seed 42
```

**Expected:** tests for the corrected engine pass. The older `tests/test_rlcard_rules_integration.py` includes an upstream `xfail`; that remains expected because upstream RLCard is deliberately unchanged. Likewise, `scripts/audit_rlcard.py` should continue to report failures for unmodified upstream RLCard. The new `scripts/audit_corrected.py` must pass.

## Repository contents changed by this patch

- Modified: `src/holdem_ai/environment/hand_engine.py`, `main.py`, `tests/test_hand_engine.py`
- New: `src/holdem_ai/environment/corrected_game.py`, `corrected_round.py`, `tests/test_corrected_round.py`, `tests/test_corrected_rlcard_integration.py`, `scripts/audit_corrected.py`, and this document.

## Limitations

- Heads-up single hands only, not 3-6 player cash games or tournaments.
- RLCard's original 5 discrete betting actions are still used; unrestricted raise sizing requires a new Agent action API.
- `obs` from RLCard remains a 54-element observation; do not rely on it as a full Information State for CFR.
- Chip integrity and selected rule scenarios are covered, but this should **not** be considered an independently certified production Hold'em engine. More property tests and integration with a verified reference engine are needed.
- Current patched implementation is designed for `rlcard==1.2.0`, with version guard, and should be verified again on actual Windows installation before merging.

````````

<!-- END_MD_SOURCE path="docs/phase2_1_correction.md" -->

## docs/phase2_rule_validation.md

Original SHA `742adda19602030e3650fd568ccd3718ad95a41a` (3369 bytes).

<a id="md-074"></a>

<!-- BEGIN_MD_SOURCE path="docs/phase2_rule_validation.md" sha="742adda19602030e3650fd568ccd3718ad95a41a" -->

````````markdown
# Phase 2: HUNL rules specification & diagnostic audit

## Status and scope

This phase **adds pure, testable reference rules and diagnostic tests**. It does
**not** change RLCard's internal rules or the Phase 1 `HandEngine`. Therefore
**the RLCard-backed game is not yet certified compliant** with standard HUNL.
An observed `xfail` for RLCard 1.2.0 positioning is expected. If it is an
`xpass`, inspect whether the installed RLCard version changed.

The upstream source reviewed:

- https://github.com/datamllab/rlcard/blob/master/rlcard/games/nolimitholdem/game.py
- https://github.com/datamllab/rlcard/blob/master/rlcard/games/nolimitholdem/round.py
- https://github.com/datamllab/rlcard/blob/master/rlcard/envs/nolimitholdem.py

## New files

| Module | Purpose |
|---|---|
| `src/holdem_ai/environment/seat_manager.py` | BTN/SB/BB and heads-up action order, blind posting |
| `src/holdem_ai/environment/actions.py` | Minimum bet/raise, stack cap, short all-in constraints |
| `src/holdem_ai/environment/pot_manager.py` | Side-pot layers, unmatched returns, ties and odd chips |
| `src/holdem_ai/environment/hand_evaluator.py` | Independent exact 7-card evaluator for reference showdowns |
| `src/holdem_ai/environment/rules_audit.py` | Non-mutating audit of raw RLCard state |
| `scripts/audit_rlcard.py` | CLI diagnostics (nonzero exit on upstream violations) |
| `tests/environment/test_*.py`, `tests/test_rules_audit.py` | Unit and integration tests |

## Core invariants

Initial heads-up positions (button `b`):

- `SB = b`
- `BB = 1 - b`
- `FirstActor(preflop) = b`
- `FirstActor(flop,turn,river) = 1 - b`

For no rake:

\[\sum_i \Delta C_i = 0\]

For side pots, including unmatched returns:

\[\sum_j \mathrm{Pot}_j + \sum_i U_i = \sum_i c_i\]

When the acting player has already put in `c`, current highest bet is `h`,
last full raise increment is `r`, and remaining stack is `s`:

\[\mathrm{Call}=\min(s,h-c)\]
\[\mathrm{MinRaiseTo}=\begin{cases}BB & h=0\\h+r&h>0\end{cases}\]
\[\mathrm{MaxRaiseTo}=c+s\]

A short all-in raise between the highest bet and minimum full raise can be
allowed, but it does **not** reopen action for a previously-acted player.
Tracking which player has acted in a street remains the full engine's duty.

## Run with uv

```powershell
uv sync --extra dev
uv run --extra dev python -m pytest -q
uv run python scripts/audit_rlcard.py --hands 100 --seed 42
```

Expect unit tests to pass. On the RLCard version known to have nonstandard
heads-up button/starting actor, the strict audit reports `FAIL` and exits with
code 1 **intentionally**. That result identifies work to do; it is not evidence
that our reference tests were unsuccessful.

## What remains unresolved

1. Replace or repair RLCard hand execution to match the verified rules.
2. Prove bet round closures, full/short all-in reopening and legal raises in
   hand-by-hand integration tests (pure constraints here are not enough).
3. Compare the included independent seven-card evaluator against RLCard showdown outcomes across known and randomized hands.
4. Verify all-in refunds and multi-way/side-pot settlement against actual engine.
5. Poker button rotation in a multi-hand match and 6-max elimination.

Do not start training a competitive Agent on the baseline until these engine
rules are validated; otherwise it can learn exploits of incorrectly modeled rules.

````````

<!-- END_MD_SOURCE path="docs/phase2_rule_validation.md" -->

## docs/phase3_multiplayer.md

Original SHA `54412e7aca492b3865d1dc222586483c0dae107c` (3006 bytes).

<a id="md-075"></a>

<!-- BEGIN_MD_SOURCE path="docs/phase3_multiplayer.md" sha="54412e7aca492b3865d1dc222586483c0dae107c" -->

````````markdown
# Phase 3 - Multi-Player Hand Engine (2-6 seats)

New independent pure-Python engine: `holdem_ai.environment.multi_player_engine.MultiPlayerHandEngine`.

- Does NOT modify the existing Phase 2.1 `HandEngine` (HUNL/RLCard adapter).
- Uses existing exact `rank_showdown` and side-pot utilities (`pot_manager.py`).
- Integer chips, no rake. Five compatible action IDs: `0=fold`, `1=check/call`, `2=half-pot raise`, `3=pot raise`, `4=all-in`.
- Also supports `step_raise_to(amount)` for explicit bet sizing (street-total target). Such actions are logged with ID `5` and their actual targets appear in `action_raise_to`; preserve both for replay/ML.
- Seat indices persist; stack zero means absent player. Button must have chips. `rotate_button()` uses a moving-button model; **dead button tournament rules are not implemented**.
- Correct heads-up special case (BTN = SB; BTN acts first preflop); with 3+ players SB is left of button, BB is left of SB, preflop starts left of BB, postflop starts left of button.
- Real shuffled 52-card deck, two cards per occupied seat, one burn before flop/turn/river.
- Player-private `observe(player_id)` reveals only the acting player's hole cards. This is a different schema than RLCard's 54-dimensional encoded observation.
- Every `reset()` starts a *new hand* with the constructor's stacks; persistent tournament stacks, ante, blind levels, ranks/payouts belong in Phase 4.
- No `step_back` for CFR yet. A deep-copy/serialized state or undo-log API will be developed separately.
- Uses `build_side_pots` and `settle_pots` with individual per-layer eligibility, matched wager refunds, ties and integer odd-chip rules.

## Install and test with uv (PowerShell)

```powershell
uv sync --extra dev
uv run --extra dev python -m pytest -q
uv run python scripts/audit_multiplayer.py --players 6 --hands 1000 --seed 42
```

## Agent use

```python
from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine

env = MultiPlayerHandEngine(num_players=6, starting_stack=200, button=0, seed=42)
result = env.play_hand([RandomAgent(i) for i in range(6)])
print(result.payoffs_bb, result.final_stacks)
```

## Amount and payoff invariants

- `sum(final_stacks) == sum(initial_stacks)`
- `sum(payoffs_chips) == 0`
- `payoff[i] == awards[i] - contribution[i]`
- `final_stack[i] == initial_stack[i] + payoff[i]`

## Important limits

This is a new testing baseline, *not* a certified poker engine. Special cases requiring ongoing validation include cumulative short-all-in re-open decisions, partial blinds, tournament blind obligations when seats bust, cards burning and odd chips. Tournament orchestration and persistent stack ownership will be implemented in Phase 4.

Standard raise/betting references:
- https://www.pokertda.com/view-poker-tda-rules/

`reset()` may return `(None, -1)` if everyone is already all-in after posting blinds and the hand is automatically settled. `play_hand()` supports this case.

````````

<!-- END_MD_SOURCE path="docs/phase3_multiplayer.md" -->

## docs/phase4_tournament.md

Original SHA `db65736ec017c7fe7de7b38f45fd2a9327687117` (2638 bytes).

<a id="md-076"></a>

<!-- BEGIN_MD_SOURCE path="docs/phase4_tournament.md" sha="db65736ec017c7fe7de7b38f45fd2a9327687117" -->

````````markdown
# Phase 4 — Tournament Environment

Single-table No-Limit Texas Hold'em tournament for **2–6 physical seats** using
Phase 3 `MultiPlayerHandEngine`. All amounts are integer chips.

## Main interface

```python
from holdem_ai.environment.tournament_env import TournamentEnv
from holdem_ai.agents.random_agent import RandomAgent

env = TournamentEnv(num_players=6, starting_stack=1000,
                    blind_levels=[(5, 10), (10, 20), (20, 40),
                                  (40, 80), (80, 160), (160, 320),
                                  (320, 640), (640, 1280), (1280, 2560),
                                  (2560, 5120), (5120, 10240)],
                    hands_per_level=10, seed=42)
result = env.play_tournament([RandomAgent(i) for i in range(6)])
print(result.winner, result.ranks, result.hands_played)
```

Incremental interface supports `reset()`, `state`, `current_player`,
`observe(player_id)`, `step(action_id)`, `step_raise_to(chips)`, `is_over()`,
`get_result()`; after a hand completes, `step` automatically starts the next
hand. An action is always applied to the currently acting seat.

## Invariants

For hand `h`, `stacks_before` equals prior hand `stacks_after`:

$$ C_{i,h+1}^{start} = C_{i,h}^{end} $$

No outside chips, no rake, and no rebuy:

$$ \sum_{i=0}^{n-1} C_{i,h} = \sum_{i=0}^{n-1} C_{i,0} $$

A bust has zero stack; busted seats remain as zero-valued physical seat IDs.

$$ E_{i,h} = \mathbf{1}[C_{i,h}>0], \qquad \text{done}=\left(\sum_i E_{i,h}=1\right) $$

`TournamentResult.rewards` defaults to `(1, 0, ..., 0)` permuted to the champion.
Custom rank-specific payouts may be supplied with `payouts=(..., ...)`.
Simultaneously busted entrants are ordered by **pre-hand stacks**; equal
pre-hand stacks tie and split the prizes for their occupied place slots.

## Development limitations

- **Moving button** to next live seat; not a formal dead-button tournament policy.
- **No ante, bounty, rebuy, add-on, ICM, time-based levels, or multi-table balancing.**
- Blinds advance every `hands_per_level` completed hands; last level repeats.
- `max_hands` and `max_actions` are safety limits, not termination substitutes.
- The underlying Phase 3 game engine still needs extended rules verification.
- All private hole cards are exposed only to the acting player through `observe`.
- This is not certified for real-money operations.

## Commands (Windows PowerShell / uv)

```powershell
uv sync --extra dev
uv run --extra dev python -m pytest -q
uv run python scripts/run_tournament.py --players 6 --seed 42
uv run python scripts/audit_tournament.py --players 6 --tournaments 100 --seed 42
```

````````

<!-- END_MD_SOURCE path="docs/phase4_tournament.md" -->
