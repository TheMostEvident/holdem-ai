# Phase 13 — Training Optimization 2.0

> Target: `TheMostEvident/holdem-ai`. Dependencies: **Phase 9, 10, 11 (including tqdm), 12**.
> Pure additive patch: **no existing Phase 1–12 source modification**.
> VRPO-Lite is experimental; this is **not** a faithful full VRPO paper replication.

## 1. Scope

- On-policy self-play with `hands_per_update` 16/64/128; `env_batch_size` 16/32/64.
- Reuse the **unchanged Phase 9 PPO** and **Phase 10 Q-Boost** optimizer losses.
- Import Phase9, Phase10 or Phase11 (`phase=9` / `phase=10` on disk) trusted checkpoints **without overwriting source**, preserving Actor/Critic **and Adam optimizer states**.
- Save a new atomic checkpoint each completed update and append `metrics.jsonl`; resume the exact same stage.
- Record per-action *chosen counts*, *legal opportunities*, *conditional-on-legal rates*; by street and by button/BB; pre-update empirical Actor entropy.
- Phase 12 seed-blocked evaluation against Random and frozen stage-start Actor; **sampled mode by default**. Validation seed blocks are reused within this stage for trend tracking, not final independent tests.
- Phase 13 tqdm progress, time-left, ETA; empirical training-throughput benchmark and matched-start comparison of PPO/Q-Boost.

## 2. Math

Let $H$ = hands per update, $B$ = simultaneously active environments, $M$ = optimizer minibatch size, $E$ = PPO epochs. Then:

$$
H \in \{16,64,128\},\qquad B\in\{16,32,64\},\qquad M\in\{32,64,128\}.
$$

Let $D_k$ be the actual decisions observed in update $k$ (depends on fold/raises), and $N_k=|D_k|$.

$$
N_{\text{mini},k} = E\left\lceil\frac{N_k}{M}\right\rceil.
$$

All samples are **on-policy** for the previous Actor weights $\theta_k$, collected completely before any gradient update.

$$
D_k\sim\pi_{\theta_k},\qquad
\theta_{k+1}=\operatorname{Update}(\theta_k,D_k).
$$

Throughput for training updates (including collection, diagnostics and optimization, excluding evaluation and checkpoint I/O):

$$
\widehat{T}_{\mathrm{train}}
=\frac{\sum_k H_k}{\sum_k \Delta t_{\mathrm{update},k}}
\quad[\mathrm{hands/sec}].
$$

Each action head has a legal-action mask. Define $C_a=\sum_t\mathbf{1}[a_t=a]$ and $L_a=\sum_t\mathbf{1}[a\in\mathcal{A}(I_t)]$.

$$
\hat p(a\mid\mathrm{legal})=\frac{C_a}{L_a},\quad L_a>0.
$$

When $L_a=0$, the rate is `null`, **not** zero. This is a *marginal conditional opportunity rate*, not $\pi(a|I)$ for a fixed infoset; it also varies with the population of encountered states.

$$
H(\pi(\cdot\mid I_t))=-\sum_{a\in\mathcal A(I_t)}\pi(a\mid I_t)\log\pi(a\mid I_t)
$$

$$
\overline{H}_{\mathrm{normalized}}
=\frac1N\sum_t\begin{cases}
H(\pi_t)/\log |\mathcal A(I_t)|,&|\mathcal A(I_t)|>1\\
0,&|\mathcal A(I_t)|=1.
\end{cases}
$$

These diagnostics use **Actor-only** inputs, without Critic privilege. Critic encodings are used only in the optimizer's training path.

## 3. Phase 11 to Phase 13 migration

The `checkpoints/phase11_pilot.pt` payload is a standard Phase 10 QBoost payload (phase=10, architecture/Adam/ppo_config/torch RNG). A Phase13 warm-start creates a **NEW** experiment with different PPOConfig values and new seed namespace:

$$
\mathrm{seed}_{13}=\mathrm{seed}_{old}+1{,}000{,}003
$$

unless `--rollout-seed` explicitly supplied.

**Important:** Phase 9/10/11 checkpoint formats stay unchanged; the output still has payload `phase=9` for PPO or `phase=10` for QBoost. The new run's metadata `run.json` has `phase=13` with `checkpoint_format_phase`. Phase12's checkpoint reader continues to work.

`--resume` must point to a Phase13 run's own output checkpoint and original log-dir; batch size, seed, epochs and optimizer minibatch size come **from its saved checkpoint**. To change these again, start a *new* stage with `--from-checkpoint <old-phase13-output>` and new output paths. Current stage's frozen Anchor remains unchanged on resume. Original checkpoint is never overwritten by warm-start.

The starting source can be:
- PPO Stage: Phase 9/11 PPO (`phase=9`)
- QBoost Stage: Phase 10/11 QBoost (`phase=10`)
- Matched PPO vs QBoost comparison: **same Phase 9 source (`phase=9`)**; Phase 13 automatically wraps it as QBoost using Phase10's `from_phase9_checkpoint` for the QBoost arm.

## 4. Install from ZIP (PowerShell, repo root)

```powershell
Copy-Item .\holdem-ai-phase13-scaling\holdem_agent_phase13 .\ -Recurse -Force
Copy-Item .\holdem-ai-phase13-scaling\tests\test_phase13_*.py .\tests\ -Force
Copy-Item .\holdem-ai-phase13-scaling\scripts\phase13_*.py .\scripts\ -Force
Copy-Item .\holdem-ai-phase13-scaling\docs\PHASE13_BATCH_SCALING.md .\docs\ -Force

uv run --extra dev --extra gpu python -m pytest -q -k phase13
uv run --extra dev --extra gpu python -m pytest -q
```

## 5. Warm-start the current QBoost checkpoint: first GTX 1070 Pilot

```powershell
uv run --extra gpu python -m scripts.phase13_train --algorithm qboost --device cuda --from-checkpoint checkpoints/phase11_pilot.pt --hands-per-update 64 --env-batch-size 32 --minibatch-size 64 --epochs 2 --rollout-seed 13000042 --duration-minutes 10 --max-updates 3 --eval-every 2 --eval-groups 40 --eval-mode sampled --checkpoint checkpoints/phase13_qboost_64.pt --log-dir runs/phase13_qboost_64
```

Stage resumes using original PPO hyperparameters saved within `phase13_qboost_64.pt`:

```powershell
uv run --extra gpu python -m scripts.phase13_train --algorithm qboost --device cuda --resume checkpoints/phase13_qboost_64.pt --duration-minutes 15 --max-updates 100 --env-batch-size 32 --eval-every 25 --eval-groups 40 --checkpoint checkpoints/phase13_qboost_64.pt --log-dir runs/phase13_qboost_64
```

Start another independent 128-hand stage from the original Phase11 checkpoint:

```powershell
uv run --extra gpu python -m scripts.phase13_train --algorithm qboost --device cuda --from-checkpoint checkpoints/phase11_pilot.pt --hands-per-update 128 --env-batch-size 64 --minibatch-size 64 --epochs 2 --rollout-seed 13000042 --duration-minutes 10 --max-updates 3 --eval-every 2 --eval-groups 40 --checkpoint checkpoints/phase13_qboost_128.pt --log-dir runs/phase13_qboost_128
```

**Caution:** changing hands-per-update changes `seed + iteration*hands_per_update + hand_index`; both stages have the same overall seed namespace, but **do not receive identical per-hand deal schedules**, nor identical optimizer effective gradient noise. This is a batch-scaling experiment, not a matched-deal comparison.

## 6. Actual GPU *training* benchmark, not merely inference

```powershell
uv run --extra gpu python -m scripts.phase13_benchmark --device cuda --algorithm qboost --hands-per-update 16,64,128 --env-batch-sizes 16,32,64 --updates 3 --epochs 2 --minibatch-size 64 --from-checkpoint checkpoints/phase11_pilot.pt --root runs/phase13_benchmark_qboost
```

- Runs all *effective* combinations where `env_batch_size <= hands_per_update`, not the nominal ones where `env_batch_size` cannot be fully used.
- Skips each case's first training update as warmup when two or more complete updates exist.
- Each case has a separate checkpoint/log folder. The source is untouched.
- Measures full optimizer time, unlike Phase11's actor-forward benchmark. No GPU memory/performance claim is made until the user's GTX1070 result is observed.

## 7. Matched-start PPO vs QBoost training

Both arms must use **exactly the same Phase9 source** and seed settings. After update 1 policies may diverge; each arm collects its own latest on-policy rollout.

```powershell
uv run --extra gpu python -m scripts.phase13_train --algorithm ppo --device cuda --from-checkpoint checkpoints/phase9_ppo.pt --hands-per-update 64 --env-batch-size 32 --minibatch-size 64 --epochs 2 --rollout-seed 13000999 --max-updates 100 --eval-every 25 --eval-groups 100 --checkpoint checkpoints/phase13_ppo_control.pt --log-dir runs/phase13_ppo_control

uv run --extra gpu python -m scripts.phase13_train --algorithm qboost --device cuda --from-checkpoint checkpoints/phase9_ppo.pt --hands-per-update 64 --env-batch-size 32 --minibatch-size 64 --epochs 2 --rollout-seed 13000999 --max-updates 100 --eval-every 25 --eval-groups 100 --checkpoint checkpoints/phase13_qboost_control.pt --log-dir runs/phase13_qboost_control

uv run --extra gpu python -m scripts.phase13_compare --ppo-log-dir runs/phase13_ppo_control --qboost-log-dir runs/phase13_qboost_control
```

The log comparison computes descriptive *training* metrics and only tags runs matched-start if source paths and PPO hyperparameters align. To compare **poker win rate**, use Phase12 `phase12_evaluate` against the same fixed opponent and matched seed blocks, then compare paired block scores. A training throughput/entropy advantage alone does not prove strategy improvement.

## 8. Artifacts and status

```
runs/phase13_qboost_64/
  run.json            # Phase13 stage settings and lineage
  anchor.pt           # immutable actor copy at Phase13 stage start
  metrics.jsonl       # updates, action/legal counters, evaluations
  summary.json        # run completion
checkpoints/phase13_qboost_64.pt # Phase10-format QBoost + Adam
```

The `tqdm` display shows updates, hands/s, ETA and wall-clock budget. The detailed histograms remain in `metrics.jsonl` to keep the PowerShell progress display readable.

**Do not Git-commit trained `.pt`, `runs/`, or the extracted patch folders.** Phase13 CUDA tests and performance benchmark must be verified on the user's GPU. No full VRPO convergence / exploitability claim follows from these diagnostics.
