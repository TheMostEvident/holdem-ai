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
