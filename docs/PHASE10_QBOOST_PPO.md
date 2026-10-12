# Phase 10: Experimental VRPO-Lite Q-Boosting PPO

> Project: HUNL200, `TheMostEvident/holdem-ai`, Phase 9 as base.
>
> This is **an independently constructed doubly-corrected policy-gradient control-variate PPO** inspired by Q-Boosting, not a claim of reproducing every derivation or experiment in the original VRPO paper. The comparison must decide whether the new estimator helps.

## 1. Environment and existing interfaces

- Phase 6: two-seat HUNL200 environment, normalized rewards in BB, `InformationState` and opt-in privileged `CriticState`.
- Phase 7: seven legal actions, Actor tokens and Critic tokens, masked batch encodings.
- Phase 8: decentralized Transformer Actor and centralized zero-sum `Q(s,a)` network.
- Phase 9: Monte-Carlo PPO, observed-action Huber Q regression, optimizer checkpointing and seat/button-balanced RandomLegalPolicy evaluation.
- Phase 10 changes no existing source files; it imports the above modules.

All rewards are per-hand **BB**, not bb/100. There is no temporal discount within a hand.

## 2. Why a naive $G-Q(s,a)$ substitution is wrong

For the acting seat $i=i_t$, let $s_t$ be the privileged training state, $I_t$ the actor observation, and $a_t$ its sampled action. Denote the frozen old policy by $\pi_0$ and the frozen centralized Q model by $\hat Q_0$.

$$
G_t^{i_t}=R_{i_t}, \qquad R_0+R_1=0.
$$

The Phase-9 action-independent baseline is

$$
\hat V_0(s_t,I_t)=\sum_{a\in\mathcal A(I_t)}\pi_0(a\mid I_t)\hat Q_{0,i_t}(s_t,a).
$$

The Phase-9 raw advantage is

$$
A_t^{\mathrm{MC}}=G_t^{i_t}-\hat V_0(s_t,I_t).
$$

Simply changing it to $G_t-\hat Q_0(s_t,a_t)$ and doing PPO **omits the action-dependent Q contribution** from the desired policy gradient. The correct control variate needs a separate analytic action sum.

## 3. Doubly-corrected Q control variate

Set a fixed $\beta\in[0,1]$, chosen **before collecting the rollout**.

$$
\delta_t^{(\beta)}=
G_t^{i_t}-\hat V_0(s_t,I_t)
-\beta\left[\hat Q_{0,i_t}(s_t,a_t)-\hat V_0(s_t,I_t)\right].
$$

Let the normalization constants from the **same unboosted on-policy rollout** be

$$
\mu_A=\frac1N\sum_t A_t^{\mathrm{MC}},\qquad
\sigma_A=\max\left(\sqrt{\frac1N\sum_t(A_t^{\mathrm{MC}}-\mu_A)^2},10^{-6}\right).
$$

**Do not independently standardize the residual and analytic parts.** Doing so would destroy their cancellation at the old policy.

$$
\widetilde\delta_t^{(\beta)}=
\frac{\delta_t^{(\beta)}-\mu_A}{\sigma_A}.
$$

For a candidate policy $\pi_\theta$, define the observed importance ratio

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid I_t)}{\pi_0(a_t\mid I_t)}.
$$

The surrogate objective (MAXIMIZED) is

$$
\begin{aligned}
J_{\mathrm{QBoost}}(\theta)
=\frac1N\sum_{t=1}^N \bigg[&
\min\Big(r_t\widetilde\delta_t^{(\beta)},
\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\widetilde\delta_t^{(\beta)}\Big)\\
&+\frac\beta{\sigma_A}\sum_{a\in\mathcal A(I_t)}
\left(\pi_\theta(a\mid I_t)-\pi_0(a\mid I_t)\right)
\hat Q_{0,i_t}(s_t,a)\bigg].
\end{aligned}
$$

The Actor minimizes $-J_{\mathrm{QBoost}}-c_H H(\pi_\theta)$, and the Critic continues to minimize observed-action Huber regression

$$
\mathcal L_Q=\mathbb E_t\left[\mathrm{Huber}\left(\hat Q_{\phi,i_t}(s_t,a_t)-G_t^{i_t}\right)\right].
$$

The analytic Q and old policy are **frozen before all epochs of the update**; the Critic changes only in its own optimizer and does not send privileged observations into Actor.forward.

### Old-policy policy-gradient identity

Condition on $(s_t,I_t)$ and suppose $G$ is an unbiased return sample conditional on each chosen action. Write $Q^\star(a)=\mathbb E[G\mid s_t,I_t,a]$. Then for $\beta=1$,

$$
\begin{aligned}
&\mathbb E_{a\sim\pi_0}
\left[\nabla_\theta\log\pi_\theta(a\mid I_t)
(G-\hat Q_0(a))\right]_{\theta=\theta_0}
+\sum_b \nabla_\theta\pi_\theta(b\mid I_t)\big\vert_{\theta_0}\hat Q_0(b)\\
&=\sum_a \nabla_\theta\pi_\theta(a\mid I_t)\big\vert_{\theta_0}
\left(Q^\star(a)-\hat Q_0(a)+\hat Q_0(a)\right)\\
&=\sum_a \nabla_\theta\pi_\theta(a\mid I_t)\big\vert_{\theta_0}Q^\star(a).
\end{aligned}
$$

The same identity holds for all $\beta\in[0,1]$, with the action-independent baseline making zero expected contribution. Subtracting fixed $\mu_A$ and dividing by fixed $\sigma_A$ rescales the gradient. PPO clipping introduces the usual surrogate bias after moving away from $\theta_0$; this is **not an unbiased estimator at arbitrary $\theta$**.

### Baseline limiting case

$$
\beta=0\implies \widetilde\delta_t
=\frac{A_t^{\mathrm{MC}}-\mu_A}{\sigma_A},\qquad
J_{\mathrm{QBoost}}=J_{\mathrm{PPO}}.
$$

Our tests verify that this matches the existing Phase-9 clipped objective and that the automatic gradient with complete action enumeration satisfies the identity above, even when the auxiliary Q values are inaccurate.

## 4. What is and is not evaluated

`qboost_diagnostics` reports:

- `mc_advantage_variance`: empirical variance of raw $G-\hat V$;
- `boosted_residual_variance`: empirical variance of $\delta^{(\beta)}$ (not a total policy-gradient variance);
- `logit_gradient_proxy_variance_ppo`: sum of per-coordinate empirical variance of $(e_{a_t}-\pi_0)\widetilde A_t$;
- `logit_gradient_proxy_variance_qboost`: the same **logit-gradient proxy**, including the analytic Q correction;
- `logit_variance_ratio`: boosted-to-baseline proxy ratio. Values below 1 mean lower *proxy* variance on this sample, not a guarantee of lower Transformer parameter-gradient variance.

$$
g_t^{\mathrm{PPO,logit}}=(e_{a_t}-\pi_0)\widetilde A_t
$$

$$
g_t^{\mathrm{QB,logit}}=(e_{a_t}-\pi_0)\widetilde\delta_t^{(\beta)}
+\frac\beta{\sigma_A}\pi_0\odot(\hat Q_0-\hat V_0).
$$

These are per-state logits, not the full neural-network parameter gradients. The result can be higher or lower variance; **do not claim variance reduction without measurement**. With a poorly trained Q network, improvements may be negligible or negative.

`--compare` initializes PPO and QBoost from identical weights and seeds. On update 1 they can start from identical behavior; after divergence each uses its **own on-policy rollout**, though deck seed and seat/button schedule remain matched. Reusing QBoost samples to optimize a divergent PPO control without correcting for behavior policy would not be a valid on-policy comparison. Post-training performance uses the Phase-9 seat/button-balanced, unpaired RandomLegalPolicy evaluation, with wide confidence intervals for tiny samples. It is neither GTO nor exploitability measurement.

## 5. Install on Windows PowerShell

After extracting `holdem-ai-phase10-qboost.zip` as `holdem-ai-phase10-qboost` inside your repository root:

```powershell
Copy-Item .\holdem-ai-phase10-qboost\holdem_agent_phase10 .\ -Recurse -Force
Copy-Item .\holdem-ai-phase10-qboost\tests\test_phase10_*.py .\tests\ -Force
Copy-Item .\holdem-ai-phase10-qboost\scripts\phase10_train.py .\scripts\ -Force
Copy-Item .\holdem-ai-phase10-qboost\docs\PHASE10_QBOOST_PPO.md .\docs\ -Force
```

Test Phase 10 and the entire repository:

```powershell
uv run --extra dev --extra gpu python -m pytest -q -k phase10
uv run --extra dev --extra gpu python -m pytest -q
```

First GTX 1070 training and comparison:

```powershell
uv run --extra gpu python -m scripts.phase10_train --device cuda --updates 2 --hands-per-update 16 --epochs 2 --minibatch-size 32 --eval-hands 20 --compare --beta 1 --checkpoint checkpoints/phase10_qboost.pt --ppo-checkpoint checkpoints/phase10_ppo_control.pt
```

Continue with **two additional** QBoost updates:

```powershell
uv run --extra gpu python -m scripts.phase10_train --device cuda --resume checkpoints/phase10_qboost.pt --updates 2 --eval-hands 20
```

Alternatively, warm-start QBoost from your **own trusted** Phase-9 model checkpoint (keeps existing Actor, Critic, both Adam optimizers, and update counter):

```powershell
uv run --extra gpu python -m scripts.phase10_train --device cuda --from-phase9 checkpoints/phase9_ppo.pt --updates 2 --eval-hands 20 --beta 1 --checkpoint checkpoints/phase10_from9.pt
```

You cannot use `--compare` with `--resume` or `--from-phase9`, because that would not be an identical matched-start comparison. The `--beta` flag is used for new runs and Phase-9 warm-start; resuming loads saved beta instead.

For a more informative but longer run, increase the training hands and evaluation to multiples of four, e.g. `--hands-per-update 64 --eval-hands 200`. Even 200 hands is small by poker standards.

**Checkpoints are data artifacts, not source.** Do not `git add .` while `checkpoints/` or extracted ZIP directories are present. Never unpickle or load unknown model checkpoints.

## 6. Tests and limitations

The isolated test environment includes copied Phase 6-9 interfaces with a mock Phase-3 game engine, not the user's actual Windows poker engine. Phase 10 new tests check: masking, input privacy, analytic policy-gradient identity, beta-zero equivalence, finite gradients, optimizer parameter changes, checkpoint/resume, Phase-9 checkpoint migration, CLI comparison, and guarded CUDA path. The **actual** Phase 1-9 regression and GTX 1070 tests must run locally.

Other open limitations: unoptimized per-decision inference, no exact per-action counterfactual labels, no GAE, no fully reproduced VRPO algorithm, incomplete historical exact Call/All-in amounts in Phase 6, small self-play rollout samples and uncertain evaluation returns. QBoost performance is an empirical question.
