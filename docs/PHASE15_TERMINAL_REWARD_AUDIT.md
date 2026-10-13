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
