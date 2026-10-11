# HUNL AI roadmap — Phase 12 to 16

> Priority order: establish a trustworthy evaluation protocol **before** increasing expensive training budgets. No phase claims Nash equilibrium or GTO exploitability from Random-vs-agent scores.

| Phase | Scope | Implementation acceptance conditions | Gate before next phase |
|---|---|---|---|
| **12. Evaluation 2.0** | Common shuffled-deck seeds, 4 seat×button hands/seed, seed-block 95% CI, PPO-QBoost paired difference, greedy vs sampled, validation/test partitions, action histograms. | 4-layout test; identical independent actor clone has zero grouped return; summary JSON/CSV; GTX1070 tests and held-out scores. | Run 100+ seed validation, record limitations. |
| **13. Training Scale + Optimizer Controls** | Increase `hands_per_update` to 64/128 and `env_batch_size` to 32/64 **even on checkpoint resume** through explicitly documented safe config migration; gradient clipping, LR scheduler, batch throughput, memory and per-update decisions. | Checkpoint optimizer compatibility; batch shape and reward invariants; benchmark end-to-end **including optimizer**. | Compare 16/64/128 hands-update at matched total training hands. |
| **14. Controlled PPO vs QBoost Study** | Identical initial weights, 3+ random seeds, separately collected on-policy self-play, held-out paired evaluation at matched hands, critic calibration and sample-variance reduction diagnostics. | Publish means, between-seed spread, win/loss by position; no conflation of proxy logit variance with full-gradient variance. | Determine whether QBoost adds measurable value beyond PPO. |
| **15. Five-hour Training** | Config freeze, ETA/tqdm, regular checkpoints, early-stop safety, time-budget accounting, snapshot league of opponents, periodic validation, **test exactly once**. | Reproducible 5-hour runs; checkpoint resumption; learning curve and independent held-out evaluation of selected snapshot. | Training stability and no leaks/strategy collapse. |
| **16. Poker Strategy Diagnostics** | Policy action statistics by street, position, SPR, bet action size, opening ranges; detect illegal-policy and pathological-fold/raise patterns; exploitability proxy if validated against equilibrium methods. | Interpretable strategy report, action distributions, game-rule regression, stronger fixed-policy benchmark, next RL/CFR design recommendations. | Define the next research iteration from measured weaknesses. |

## Essential experimental quantities

`K`: independent deck seed blocks (not fourfold number of games). Each candidate/opponent matchup takes `4*K` hands.

$$
X_k=\frac14\sum_{b,s\in\{0,1\}}R_A(k,b,s), \quad \hat J=100\overline X.
$$

$$
D_k=X_k^{\mathrm{QBoost}}-X_k^{\mathrm{PPO}},\qquad
\widehat{\Delta J}=100\overline D.
$$

Uncertainty estimates use the empirical variance *across independent deck seed blocks*; if training replicas share initializations/seed schedules, report also across-run variation.

Batch scaling compares `hands/sec` **including** actor forward, critic, optimizer and checkpoint overhead on the target GPU. The Phase11 `phase11_benchmark` is forward/rollout only and cannot stand in for training throughput.

### Phase12 restrictions

This package does not modify Phase11. An evaluation performed repeatedly during a training run is a *validation* score. Reserve `test` seed blocks for a final, precommitted assessment. Multiple comparisons on validation require caution as they can overfit the checkpoint-selection process.
