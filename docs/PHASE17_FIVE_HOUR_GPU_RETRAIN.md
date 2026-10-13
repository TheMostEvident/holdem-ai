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
