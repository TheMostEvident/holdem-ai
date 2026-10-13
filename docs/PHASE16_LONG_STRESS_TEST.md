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
