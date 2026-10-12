# Phase 9 — HUNL200 PPO Baseline Training

> Target: `TheMostEvident/holdem-ai`, branch `main`, Phase 8 baseline commit `25f47f4`.
> **This is Monte-Carlo PPO, not the full VRPO algorithm and not Q-Boosting.**
> Standalone additive files; Phase 1–8 are unchanged.

## 1. Design

- One **shared decentralized Transformer Actor** (Phase 8) controls both seats during on-policy self-play. The inference policy receives only `InformationState` from Phase 6.
- One **centralized Q-Critic** (Phase 8) trains on the opt-in `CriticState` with both players' cards; this information is never fed to Actor.
- Terminal zero-sum chip rewards are measured in **big blinds**, not bb/100.
- All actor actions are masked to the existing seven legal abstract poker actions.
- Self-play is sequential, with single-decision GPU inference. **Do not extrapolate Phase 6 random CPU benchmark throughput to Phase 9.** Multiprocessing/batched model serving is future work.
- PPO uses terminal Monte-Carlo returns, **not GAE**, not TD(0), and not discounting. In an episodic zero-sum cash-game hand, `gamma = 1`.

## 2. Mathematical derivation (Obsidian Markdown)

Let $T$ be the last global poker action in one hand and $i_t\in\{0,1\}$ the acting seat. The environment returns $R_i\in[-200,200]$ BB and $R_0+R_1=0$.

$$
G_t^{i_t}=R_{i_t}, \qquad t\le T.
$$

The old **decentralized** policy is $\pi_{\theta_{\mathrm{old}}}(a\mid I_t^{i_t})$ and its legal support is $\mathcal A(I_t^{i_t})$.

$$
\ell_t^{\mathrm{old}} = \log \pi_{\theta_{\mathrm{old}}}(a_t\mid I_t^{i_t})
$$

The centralized critic predicts seven action values for each seat. Its outputs obey the zero-sum relation

$$
Q_{\phi,0}(s_t,a)=-Q_{\phi,1}(s_t,a).
$$

**Do not use $Q_\phi(s_t,a_t)$ directly as an actor baseline:** it depends on the sampled action. Instead use a frozen, action-independent expectation,

$$
V_{\mathrm{old}}^{i_t}(s_t,I_t^{i_t})
=\sum_{a\in\mathcal A(I_t^{i_t})}
\pi_{\theta_{\mathrm{old}}}(a\mid I_t^{i_t})\,Q_{\phi_{\mathrm{old}},i_t}(s_t,a).
$$

This baseline may use privileged training information **only in the advantage computation**, not in the Actor forward pass. It is valid as an action-independent control variate for policy-gradient estimation, although its variance-reduction quality depends on critic accuracy.

$$
\widehat A_t=G_t^{i_t}-V_{\mathrm{old}}^{i_t}(s_t,I_t^{i_t})
$$

Advantages are standardized across the collected decisions:

$$
\widetilde A_t=\frac{\widehat A_t-\mu_A}{\max(\sigma_A,10^{-6})}.
$$

For an epoch of PPO updates,

$$
r_t(\theta)
=\exp\left[\log\pi_\theta(a_t\mid I_t^{i_t})-\ell_t^{\mathrm{old}}\right].
$$

$$
L^{\mathrm{CLIP}}(\theta)
=\mathbb E_t\left[\min\left(
 r_t(\theta)\widetilde A_t,
 \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\widetilde A_t
\right)\right].
$$

Actor *minimizes*

$$
\mathcal L_{\mathrm{actor}}=-L^{\mathrm{CLIP}}(\theta)-c_H\,\mathbb E_t[H(\pi_\theta(\cdot\mid I_t))].
$$

Critic is trained on the **observed action only**, using terminal Monte-Carlo targets:

$$
\mathcal L_Q=\mathbb E_t\left[
\mathrm{Huber}\left(Q_{\phi,i_t}(s_t,a_t)-G_t^{i_t}\right)
\right].
$$

No claim is made that the Q-values for unobserved counterfactual actions are supervised by that single hand.

## 3. Modules

| File | Purpose |
| --- | --- |
| `holdem_agent_phase9/ppo.py` | Self-play collection, frozen old policy/baseline, PPO clipping, Huber Q loss, seat-balanced random evaluation |
| `holdem_agent_phase9/trainer.py` | Optimizers, step loop, atomic checkpoint, weights-only resume |
| `holdem_agent_phase9/__init__.py` | Public functions |
| `scripts/phase9_train.py` | Real training CLI, JSON per-update metrics |
| `tests/test_phase9_ppo.py` | PPO math, legal masks, gradients, reward and seat handling, checkpoint/resume |
| `tests/test_phase9_cli.py` | End-to-end CLI test |
| `tests/test_phase9_gpu.py` | CUDA optimizer smoke (skipped without CUDA) |

`checkpoints/phase9_ppo.pt` stores Actor/Critic state, both Adam optimizers, config, current update and CPU Torch RNG. It is **not** an exact bitwise cross-device/Windows resume guarantee. Only load checkpoints created by you, and keep large checkpoint files out of Git.

## 4. Install patch on Windows (PowerShell)

From your existing project root after extracting the ZIP as `holdem-ai-phase9-ppo`:

```powershell
Copy-Item .\holdem-ai-phase9-ppo\holdem_agent_phase9 .\ -Recurse -Force
Copy-Item .\holdem-ai-phase9-ppo\tests\test_phase9_*.py .\tests\ -Force
Copy-Item .\holdem-ai-phase9-ppo\scripts\phase9_train.py .\scripts\ -Force
Copy-Item .\holdem-ai-phase9-ppo\docs\PHASE9_PPO_BASELINE.md .\docs\ -Force
```

No PyTorch dependencies are added; your installed `torch==2.6.0+cu118` GPU extra is used.

## 5. Tests and actual first GPU training

```powershell
uv run --extra dev --extra gpu python -m pytest -q -k phase9
uv run --extra dev --extra gpu python -m pytest -q
uv run --extra gpu python -m scripts.phase9_train --device cuda --updates 2 --hands-per-update 16 --epochs 2 --minibatch-size 32 --eval-hands 20 --checkpoint checkpoints/phase9_ppo.pt
```

If initial GPU per-decision inference is too slow, use `--hands-per-update 8 --updates 1 --eval-hands 4`. You can verify on CPU by replacing `cuda` with `cpu`. An even evaluation count is insufficient: **eval hands must be divisible by four** so each combination of neural seat and button is sampled equally.

To continue from a checkpoint, adding two **further** updates (not to a total of two):

```powershell
uv run --extra gpu python -m scripts.phase9_train --device cuda --resume checkpoints/phase9_ppo.pt --updates 2 --eval-hands 20
```

The program reuses `--resume` as the output path when `--checkpoint` has not been overridden. When resuming, saved PPO/network config takes precedence over fresh-run flags. The evaluation RNG seed is unchanged before and after, but **poker decisions differ with policy**, so this is not a duplicated-card paired-outcome evaluator.

For faster CLI integration testing:

```powershell
uv run --extra gpu python -m scripts.phase9_train --device cuda --updates 1 --hands-per-update 2 --epochs 1 --minibatch-size 4 --eval-hands 4 --small-network --checkpoint checkpoints/phase9_tiny.pt
```

## 6. Interpret the JSON

- `update`: training iterations completed.
- `decisions`: number of observed self-play decisions.
- `actor_loss`: clipped PPO surrogate loss for the sampled data; **not** a monotonic convergence metric.
- `critic_loss`: Huber regression to own-seat BB return.
- `approx_kl`, `clip_fraction`: policy change diagnostics.
- `entropy`: policy distribution entropy, ideally monitored for premature collapse.
- `grad_norm_actor`, `grad_norm_critic`: norms prior to gradient clipping.
- `mean_bb100`, `ci95_lower_bb100`, `ci95_upper_bb100`: neural policy returns against RandomLegalPolicy, not GTO exploitability. Tiny 20-hand CI can be huge and normal approximation is unreliable at small samples.
- `cuda_peak_allocated_mb`: program-local PyTorch allocated peak, not all VRAM or future five-hour train memory.
- `trained_policy: true`: optimizer updates were applied, **not** proof of poker improvement.

## 7. Limitations and next phases

1. One GPU policy inference call **per action**; the 4-worker Phase 6 CPU-only throughput is not applicable.
2. Random-vs-random and self-play are **weak baselines**; measure Random, prior self, frozen snapshots, and seats at larger evaluation sizes before making improvement claims.
3. `Q(s,a)` is trained only on actually sampled actions and is a **variance-reduction baseline**, not Q-Boosting or an accurate counterfactual oracle.
4. Phase 6 history logs exact `raise_to` but omit amount for `all_in`/`call`; history representation can be improved before high-fidelity training.
5. PPO is not generally guaranteed to converge to a Nash equilibrium in imperfect-information zero-sum games. Exploitability/GTO evaluation is a later independent workstream.
6. The pilot uses no GAE, no recurrent belief tracker, no parallel neural rollout workers and no full VRPO Q-Boosting; these are future enhancements.

To commit the patch after your actual GTX 1070 tests:

```powershell
git add holdem_agent_phase9/ tests/test_phase9_*.py scripts/phase9_train.py docs/PHASE9_PPO_BASELINE.md
git diff --cached --stat
git commit -m "Implement Phase 9 on-policy PPO baseline training"
git push origin main
```

Do **not** `git add .` while extracted ZIP folders and `.pt` model checkpoints are in your local project. Consider adding `checkpoints/` to `.gitignore` separately.
