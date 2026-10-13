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
