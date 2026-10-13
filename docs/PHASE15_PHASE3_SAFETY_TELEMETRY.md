# Phase 15 — Phase 3 Safety + Reward Diagnostics (Agent only)

## Status

Changes are committed to the research branch \`agent/phase15-collapse-diagnostics\`, **not** merged to \`main\`. Existing \`runs/phase15_ppo_main\` and \`runs/phase15_ppo_stability_pilot\` checkpoints are not modified. The branch code must pass tests locally before a new GPU pilot.

## Motivation

A low policy entropy alone is not evidence that a poker strategy is bad. In the observed run, however, a fixed-state comparison showed a +0.343803 increase in mean fold probability (Update 375 → 450), concentrated particularly in preflop big-blind states. An improved warning system is appropriate, while actual policy quality still requires validation.

## Safety policy

For a **new run** on this branch:

- Entropy below \`entropy_alert_threshold\` increments \`entropy_warnings\` and emits an \`entropy_warning\` log event. **It no longer advances the hard-stop counter by default.**
- Approx KL above \`max_approx_kl\` independently increments \`kl_warnings\` and advances the consecutive hard-stop counter.
- Both events are recorded when both conditions hold (previous \`elif\` could hide KL warnings).
- A high-KL streak of \`entropy_stop_patience\` updates still triggers \`safety_stop\`.
- Explicit \`entropy_stop_enabled=True\` restores entropy hard-stop behavior at configuration level.
- Resuming runs created **before** the new manifest field uses their historical entropy-hard-stop behavior. Do not use old run roots for the new safety experiment.
- NaN/Inf checks in the PPO optimizer remain intact.

Formally:

$$
w_{H,t}=\mathbf1[H_t<H_{\min}], \qquad
w_{KL,t}=\mathbf1[\widehat D_{KL,t}>KL_{\max}].
$$

$$
w_{\mathrm{stop},t}=w_{KL,t}\lor (\mathrm{entropy\_stop\_enabled}\land w_{H,t}).
$$

Warnings and stopping criteria are different. **The default has changed only for fresh runs on this research branch.**

## Per-update telemetry

The \`reward_diagnostics\` object in each new update record includes:

- Hand count; seat-0 mean/std/min/max terminal payoff (BB).
- Absolute terminal payoff mean (BB).
- Fold-terminated hand count and proportion.
- Mean decisions per hand.
- Non-fold-terminated hand count. **This is not identical to Showdown count**; the exact showdown rate remains \`null\` until an authoritative engine terminal-result field is wired.

Existing \`action_diagnostics\` already supplies legal opportunity-conditioned action rates by street/position. PPO metrics already record approximate KL, clip fraction and unclipped actor/critic gradient norms. The added telemetry augments, rather than redefines, those measurements.

## Validation command (Windows PowerShell)

\`\`\`powershell
uv run --extra gpu pytest tests/test_phase15_policy_audit.py tests/test_phase15_safety_phase3.py -q
\`\`\`

Run the ordinary project test suite too before merging. New telemetry does not retroactively appear in existing historical \`metrics.jsonl\` files.

## Next experiment

After tests pass, create a fresh Phase 15 run rooted in a **new unique directory**, initialized from the trusted Phase 14 checkpoint; do not overwrite or finalize the historical runs. Use a limited update count and an independent rollout seed, and compare against the original safety rules. Consider a second pilot seeded from the previously selected Update 375 model for a controlled continuation, taking care to distinguish preexisting optimizer-state effects.

Before claiming Policy Collapse is *resolved*, extend fixed-state sampling beyond its preflop-heavy distribution and add authoritative showdown telemetry and independent evaluation.
