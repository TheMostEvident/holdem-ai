# Phase 15 — PPO Critic Diagnosis (Agent Task)

**Status:** Added as a read-only diagnostic on \`agent/phase15-collapse-diagnostics\`, not merged into \`main\`. No engine, game rules, GUI, optimizer, safety defaults, historical training logs, or existing checkpoints are modified.

## 1. Targets of the diagnosis

The Phase-15 Actor can only use its own Information State \(I_i\). The centralized training-only Critic sees privileged training state \(s\) and outputs \(Q_\phi(s,i,a)\) for each player/action. It is trained with a Monte Carlo terminal reward for the **observed** action:

\[
\hat G_i^{Q}=Q_\phi(s_i, i, a_i),\qquad
L_{Q}=\operatorname{Huber}(\hat G_i^{Q}, G_i).
\]

PPO's frozen, action-independent baseline is:

\[
\hat V_\phi(s_i)=\sum_{a\in\mathcal A_{\rm legal}(I_i)}
\pi_{\rm behavior}(a\mid I_i)Q_\phi(s_i,i,a)
\]

and raw advantage is:

\[
A_i=G_i-\hat V_\phi(s_i).
\]

**Do not confuse** selected-action \(Q\) quality with action-independent \(V\) baseline quality: they use different predictions, even on the same sample.

## 2. Metrics

For prediction \(p_i\), target \(G_i\), and error \(e_i=p_i-G_i\):

\[
\mathrm{Bias}=\frac1N\sum_i e_i,\quad
\mathrm{MAE}=\frac1N\sum_i|e_i|,\quad
\mathrm{RMSE}=\sqrt{\frac1N\sum_i e_i^2}
\]

\[
\mathrm{ExplainedVariance}=1-\frac{\mathrm{Var}(e)}{\mathrm{Var}(G)}
\]

EV is \`null\` if the reward variance is zero. EV can equal 1 even when prediction bias is nonzero, so **never interpret EV without bias and RMSE**. All numerical metrics are in Big Blind units where applicable (not bb/100).

Metrics are recorded overall and split by **Street, Acting Seat, Button/Big Blind Position, and Observed Action**. Raw advantage mean/std/mean-absolute-value are included.

## 3. Evaluation protocol

### A. Shared reference rollout (controlled observations/actions/targets)

Sample a single frozen reference policy with a held-out diagnostic deck seed. Feed the identical \((s,I,a,G)\) samples to **both** critics. This controls input data and reveals differences between critic functions.

**Important limitation:** \(Q^{\pi_{\rm best}}\) and \(Q^{\pi_{\rm latest}}\) can differ legitimately because they assume different *future behavior*. Terminal returns sampled under \(\pi_{\rm ref}\) are **not** unbiased labels for a candidate policy's \(Q^{\pi_{\rm candidate}}\). Interpret paired residual differences only as a **cross-policy generalization probe**, not as proof that one Critic is incorrectly trained.

### B. Two independent on-policy rollouts (primary calibration)

Use each Checkpoint's own Actor to generate held-out trajectories. Its own Critic then predicts the realized terminal return for observed actions. PPO's baseline uses that **same generating policy**.

This evaluates the correct continuation distribution for each checkpoint but creates different state/action distributions, so the raw scores are not a controlled causal comparison.

Decision rows from the same hand share terminal rewards. This tool reports descriptive metrics, not iid significance intervals. On-policy and shared diagnostic seeds are kept outside existing training and validation namespaces. Postflop coverage can remain sparse.

## 4. Commands — Windows PowerShell

First synchronize the research branch:

\`\`\`powershell
git pull --ff-only origin agent/phase15-collapse-diagnostics
uv run --extra gpu pytest tests/test_phase15_critic_diagnostics.py -q
uv run --extra gpu pytest -q
\`\`\`

Then execute (uses existing local Checkpoints, without training):

\`\`\`powershell
uv run --extra gpu python -m scripts.phase15_critic_audit \`
  --reference runs/phase14_main/seed_52/ppo/best.pt \`
  --before runs/phase15_ppo_safety_v2_pilot/best.pt \`
  --after runs/phase15_ppo_safety_v2_pilot/latest.pt \`
  --device cuda \`
  --hands 256 \`
  --on-policy-hands 128 \`
  --seed 570000000 \`
  --output runs/phase15_audit/safety_v2_critic_audit.json
\`\`\`

Output uses exclusive creation: an existing JSON report is **not overwritten**. The CLI prints concise on-policy metrics, while the JSON contains all subsets and both protocols.

## 5. Interpretation and next phase

- Extremely poor **on-policy** \(Q(s,a_{\rm taken})\) RMSE / EV: investigate Critic target scale, architecture, optimizer and value loss before changing the Actor.
- Relatively good chosen-\(Q\) fit but poor **on-policy** \(V(s)\) calibration: investigate unsupervised alternative-action \(Q(s,a)\) and quality of \(V(s)=\sum\pi Q\).
- Reasonable critic metrics alongside declining validation scores: investigate Actor PPO updates, bet sizing, exploration, opponent/generalization, and noisy evaluation.
- The diagnostics alone cannot establish why the latest Policy's validation score is lower. Do not change PPO learning rules solely based on a noisy Pilot.

**Checkpoint safety:** Historical \`best.pt\`/\`latest.pt\` stay read-only. Do not run Phase 15 \`finalize\` for this test.
