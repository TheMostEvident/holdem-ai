---
title: "Poker AI — Unified Engineering & Research Technical Report"
date: 2026-10-13
status: "source-grounded consolidated technical reference"
aliases: ["Poker AI 최종 통합 기술보고서"]
tags: [HUNL, tournament, PPO, MCCFR, game-engine, reinforcement-learning]
---

# Poker AI — Final Integrated Technical Report

> [!note] Verified implementation excerpts
> The [Phase-verified Python code reference](PHASE_VERIFIED_PYTHON_CODE_REFERENCE.md) contains **literal GitHub source excerpts** for the native multiplayer engine, PPO trainer, MCCFR prototype, Tournament actor bridge, HUNL session manager and GPU retrain scheduler. Conceptual execution sketches in this document are separately labeled pseudocode.

> [!important] Evidence and coverage
> This report synthesizes the checked-in \`AGENT_DEVELOPMENT_TECHNICAL_REPORT_2026-10-13.md\`, \`TOURNAMENT_ENV_PHASE1_FULL_TECHNICAL_REPORT.md\`, \`PROJECT_REPORT.md\`, Expand E1–E7 documentation and GUI / P15–17 research notes. It is a **navigable engineering synthesis, not a verbatim replacement of every historical source file**. Historical logs and detailed code remain in version control. A phase being documented is not proof of a trained strong or GTO-equivalent agent. Source code excerpts below are deliberately short and labeled by path or API; pseudocode is expressly identified.

## 1. Objective and formal mathematical contract

\`\`\`mermaid
flowchart TD
    C[Chance: shuffle and deal] --> E[Environment: hand rules / pot / payoff]
    E --> I[Actor-visible information state and legal mask]
    I --> P[Agent: frozen or trainable policy]
    P --> A[Action abstraction and native-action translation]
    A --> E
    E --> D[Rollout / rewards]
    D --> U[PPO / QBoost / MCCFR research]
    U --> P
    E --> G[HUNL or Tournament GUI / replay]
\`\`\`

The Environment and Agent are independent components of an imperfect-information extensive-form stochastic game:

$$
\mathcal G=(N,H,Z,A,P,f_c,\{\mathcal I_i\},\{u_i\}),
\quad s_{t+1}=T(s_t,a_t,\omega_t), \quad a_t\in A_{\rm legal}(s_t).
$$

Here $N$ denotes participants, $H$ histories, $Z$ terminal histories, $A$ admissible actions, $P$ actor selection, $f_c$ chance/card allocation, $\mathcal I_i$ information partitions and $u_i$ payoff. The acting policy must use only its own hole cards and public board, stacks, position and betting history. Training-only centralized critics can access privileged state, but cannot leak it into inference.

$$
\boxed{\sum_{i\in N}S_{i,t}^{behind}+Pot_t=C_t},\quad
\boxed{R_{i,h}^{BB}=\frac{S_{i,h}^{final}-S_{i,h}^{initial}}{BB_h}},\quad
\boxed{\widehat{bb/100}=\frac{100}{H}\sum_{h=1}^H R_{i,h}^{BB}}.
$$

Chip conservation holds during a hand without rake; for dynamic cash tables, join/rebuy/cash-out events change the table total at explicitly recorded **hand boundaries**. Continuous-stack sessions obey $S_{i,h+1}^{start}=S_{i,h}^{final}$. In two-player rake-free play, $\sum_iR_{i,h}^{BB}=0$.

The masked actor policy is

$$
\pi_\theta(a\mid I)=
\frac{\mathbf1[a\in A(I)]\,e^{\ell_\theta(I,a)}}
{\sum_{b\in A(I)}e^{\ell_\theta(I,b)}}.
$$

An invalid action must receive probability zero. No high win-rate measurement can compensate for a broken game rule or private information leak.

## 2. Project boundaries and development-phase matrix

| Track | Phase | Why it exists | Core source files | Acceptance / limitations |
|---|---|---|---|---|
| Base Environment | P0 | Reproducible package and CLI | \`main.py\` | Python / uv setup |
| Base Environment | P1 | RLCard minimal HUNL baseline | \`environment/rlcard_adapter.py\`, \`hand_engine.py\` | Initial smoke, not game-rule certification |
| Base Environment | P2/2.1 | Audit and correct button, blinds, round order | \`environment/rules_audit.py\`, \`corrected_game.py\`, \`corrected_round.py\` | Heads-up SB acts first preflop, BB postflop |
| Base Environment | P3 | Native 2–6 person betting, all-ins, side pots | \`environment/multi_player_engine.py\`, \`multi_player_seats.py\` | Pot and valid raise invariants |
| Base Environment | P4 | Knockout tournament state, blind progression | \`environment/tournament_env.py\`, \`blind_manager.py\`, \`player_manager.py\`, \`rewards.py\` | Elimination and rank correctness |
| Base Agent | P5 | Shared legal policy contract | \`core/schema.py\`, \`agents/policies.py\`, \`training/episode_runner.py\` | Actor legal mask |
| Base Agent | P6 | HUNL200 adapter + actor/critic privacy | \`environment/hunl200_adapter.py\`, \`training/global_rollout.py\` | 7-action → native mapping |
| Base Agent | P7 | Token/action encoder | \`encoding/observation.py\`, \`actions.py\`, \`tensors.py\` | Padding, public info privacy |
| Base Agent | P8 | Transformer actor and centralized Q critic | \`models/transformer.py\`, \`agents/neural_policy.py\` | Frozen inference |
| Base Agent | P9 | On-policy Monte-Carlo PPO | \`training/ppo.py\`, \`ppo_session.py\` | Masked clipped loss, checkpoint |
| Base Agent | P10 | Experimental QBoost action-dependent control variate | \`training/qboost.py\`, \`qboost_session.py\` | Baseline equivalence for zero correction |
| Base Agent | P11 | Batch collector and time budget | \`training/batched_collector.py\`, \`experiments/time_budget.py\` | Checkpoint/resume |
| Base Agent | P12 | Seat/button matched-deck evaluation | \`evaluation/seed_blocked.py\`, \`seed_statistics.py\` | Seed groups, not games, independent |
| Base Agent | P13 | Batch scaling and action diagnostics | \`experiments/batch_scaling.py\`, \`evaluation/action_diagnostics.py\` | Throughput vs learning |
| Base Agent | P14 | Multi-seed paired PPO-vs-QBoost | \`experiments/controlled_study.py\` | Validation vs sealed test |
| Base Agent | P15 | Long-run, critic and preflop-collapse diagnosis | \`experiments/longrun.py\`, \`evaluation/critic_diagnostics.py\`, \`preflop_sequence_audit.py\` | Per-hand vs per-legal-action denominators |
| Base Agent | P16 | Frozen opponent pool, A0–A3 ablation and long stress | \`experiments/ppo_ablation.py\`, \`ppo_stress.py\`, \`evaluation/opponent_pool.py\` | Paired CI, update KL, stable PPO |
| Base Agent | P17a | Continuous-stack test only | \`evaluation/continuous_stack.py\` | No 5h training claim |
| Base Agent | P17b | 5h target GPU retrain | \`experiments/ppo_five_hour.py\`, \`scripts/phase17_gpu_retrain.py\` | 5h target **not completed** |
| Expand 6-max | E1 | Join/leave/rebuy boundary state | \`expand/environment/table_manager.py\`, \`event_ledger.py\` | Real engine acceptance needed |
| Expand 6-max | E2 | Variable-player information/action encoding | \`expand/information/builder.py\`, \`encoder.py\`, \`actions.py\` | Legal / private boundaries |
| Expand 6-max | E3 | Multi-seat PPO/G​AE | \`expand/rl/collector.py\`, \`ppo.py\`, \`trainer.py\` | Cash → tournament transfer is not tournament training |
| Expand 6-max | E4 | External-sampling MCCFR research blueprint | \`expand/pluribus/blueprint/mccfr.py\`, \`holdem.py\` | Toy NashConv ≠ six-player exploitability |
| Expand 6-max | E5 | Budgeted belief-conditioned subgame search | \`expand/pluribus/search/belief.py\`, \`solver.py\` | True 6-max search performance unproven |
| Expand 6-max | E6 | Public-action opponent modeling | \`expand/opponent/model.py\`, \`adaptive.py\`, \`router.py\` | Expert switching with divergence guards |
| Expand 6-max | E7 | Policy routing, tournament/benchmark bridges | \`expand/integration/agent.py\`, \`tournament.py\`, \`benchmark.py\` | E4/E5 use needs explicit verification |
| HUNL GUI | G1.0–G1.6 | Human gameplay, HTTP, React, betting, trained checkpoints, analytics, continuous stacks | \`game_service/*.py\`, \`backend/game_router.py\`, \`frontend/src/HUNLApp.tsx\` | Freeze actor; do not treat GUI as trainer |
| Tournament ENV | T1.1–T1.7 | API, React table, improved bots, trained 6-seat actor, rankings, SQLite, reveal, QA | \`tournament_service/*.py\`, \`backend/tournament_router.py\`, \`frontend/src/tournament/TournamentApp.tsx\` | Replay and showdown privacy; no proven tournament-optimized PPO |

All relative \`environment/\`, \`training/\`, etc. paths above (except \`scripts/\`, \`backend/\`, \`frontend/\`) are relative to \`src/holdem_ai/\`.

## 3. Environment implementation: input → rule → state → reward

### 3.1 Hand lifecycle and rule corrections (P1–P4)

$$
\operatorname{start\_hand}\to\operatorname{post\_blinds}\to
\operatorname{deal}\to\operatorname{legal\_action}\to
\operatorname{act}\to\operatorname{advance\_street}\to
\operatorname{settle}.
$$

The principal \`src/holdem_ai/environment/multi_player_engine.py\` owns the hand state and rules. \`multi_player_seats.py\` determines actor/seat order, \`actions.py\` validates betting operations, \`pot_manager.py\` forms side pots, and \`hand_evaluator.py\` resolves ranked hands. The test and audit scripts are \`scripts/audit_multiplayer.py\`, \`tests/test_multi_player_engine.py\`, \`tests/environment/test_side_pot.py\`. P2.1 corrected a crucial heads-up order difference:

$$
\operatorname{first}_{preflop}=\mathrm{SB/BTN},\qquad
\operatorname{first}_{postflop}=\mathrm{BB}.
$$

Without this distinction, even flawless RL optimization would target a different game. P4 \`tournament_env.py\` adds hand-to-hand stack carry, blind levels, elimination and ranking, not a decision-making neural policy.

The following is an **execution-flow illustration, not a direct source quotation**:

\`\`\`python
# Conceptual interface; use the native engine's actual methods/contracts.
state = environment.start_hand(seed=seed)
while not environment.is_terminal():
    information = environment.observe_current()
    legal = environment.legal_actions()
    action = acting_policy(information, legal)
    state = environment.act(action)
result = environment.settle_hand()
\`\`\`

### 3.2 Native and abstract action mapping

HUNL Head indices $0..6$ mean respectively Fold, Check, Call, Half-Pot Raise, Pot Raise, Double-Pot Raise, Explicit All-In. Some native engines index action 4 as All-In, so matching integer indices alone is invalid.

For Tournament, the 7-action Expand Head must be translated into its 5-action native contract:

$$
\mathcal A_E=\{F,X,C,\tfrac12P,P,2P,\mathrm{AllIn}\},\ 
\mathcal A_T=\{F,X/C,\tfrac12P,P,\mathrm{AllIn}\}
$$

$$
\boxed{m_{E\to T}(0,1,2,3,4,5,6)=(0,1,1,2,3,\bot,4)}
$$

Index 5, Double-Pot, is excluded if unsupported; Check and Call are distinguished through **current legal context**. The probability mass is renormalized over compatible legal native actions. The implementation reference is \`src/holdem_ai/tournament_service/trained_actor.py\` and \`docs/TOURNAMENT_ENV_PHASE_1_4.md\`.

## 4. PPO, critic, QBoost and diagnostics (P5–P17)

### 4.1 Actor–Critic execution

\`\`\`mermaid
flowchart LR
    E[HUNL200 Environment] --> O[Actor observation + legal mask]
    O --> A[Transformer logits]
    A --> S[Sample legal native action]
    S --> E
    E --> C[Training-only privileged critic state]
    C --> Q[Centralized Q critic]
    E --> R[Terminal returns]
    R --> L[On-policy update]
    Q --> L
    L --> A
\`\`\`

P7 encoder transforms cards, streets, bet history and numeric quantities into masked tensors. P8 \`models/transformer.py\` computes policy logits and an action-value head. P9 \`training/ppo.py\` calculates

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid I_t)}{\pi_{\theta_{\rm old}}(a_t\mid I_t)}
$$

$$
\boxed{L_\mathrm{PPO}=
-\mathbb E_t\left[\min\{\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\}\right]
-\alpha \mathcal H(\pi_\theta)}
$$

$$
V_\phi(s_t,i)=\sum_{a\in A(I_t)}\pi_{\rm old}(a\mid I_t)Q_\phi(s_t,i,a),\quad
\hat A_t=\operatorname{Normalize}(G_{i_t}-V_\phi(s_t,i))
$$

$$
L_Q=\operatorname{SmoothL1}(Q_\phi(s_t,i,a_t),G_{i_t}).
$$

P10 QBoost retains a critic but adds an **experimental action-dependent control variate and analytic correction term** to the policy gradient; it is not a reproduction-certified full VRPO implementation. The original \`docs/PHASE10_QBOOST_PPO.md\` and the checked-in \`training/qboost.py\` remain authoritative for the implemented formula.

The following is **schematic pseudocode**, not a quotation from \`training/ppo.py\`:

\`\`\`python
for update in range(num_updates):
    trajectories = collect_on_policy(current_actor)
    returns = compute_terminal_returns(trajectories)
    advantages = normalize(returns - frozen_old_critic_values)
    update_actor_with_clipped_objective(trajectories, advantages)
    update_critic_on_observed_action_returns(trajectories, returns)
    save_checkpoint_if_needed()
\`\`\`

### 4.2 Experimental controls (P11–P14)

$$
X_k=\tfrac14\sum_{j=1}^4R^{BB}_{k,j},\quad
\widehat J_{bb/100}=100\frac1K\sum_{k=1}^K X_k,\quad
CI_{95\%}\simeq\widehat J\pm1.96\frac{s_{100X}}{\sqrt K}.
$$

The statistical replicate is a **deck seed block** containing four balanced seat/button layouts, not each hand. P11 batching improves sequential sampling efficiency. P12 evaluates reproducibly; P13 compares batch sizing; P14 controls matched starting actor/critic/optimizer, training seed, and a separate validation/test division. Historical P14 validation scores do not imply GTO strength or generalize to live human play.

### 4.3 Collapse, preflop and reward audits (P15–P16)

$$
\widehat D_{\mathrm{KL}}=
\frac1M\sum_{I\in D_{\mathrm{old}}}\sum_{a\in A(I)}
\pi_{\mathrm{old}}(a\mid I)\log
\frac{\pi_{\mathrm{old}}(a\mid I)}
{\pi_{\mathrm{new}}(a\mid I)}
$$

$$
p_{\mathrm{allin,hand}}=
\frac{\#\text{hands with all-in}}{\#\text{hands}},\quad
p_{\mathrm{allin\mid legal}}=
\frac{\#\text{selected all-in at eligible decisions}}{\#\text{eligible decisions}}
$$

Historical 438/512 preflop-all-in hands is a **per-hand count**, not policy probability conditional on a legal all-in opportunity. \`evaluation/preflop_sequence_audit.py\` and \`terminal_reward_audit.py\` trace when hand exits occur; \`critic_diagnostics.py\` compares calibration quantities:

$$
e_i=\hat V_\phi(s_i)-G_i,\ 
\mathrm{RMSE}=\sqrt{\frac1M\sum_i e_i^2},\ 
\mathrm{EV}=1-\frac{\mathrm{Var}(e)}{\mathrm{Var}(G)}.
$$

P16 separates A0–A3 optimizer experiments and A0/A3 100-update stress from real strategy validation. Exact sampled-state post-update KL and diverse frozen opponents are diagnostics, not a global Nash guarantee. No definitive root cause of collapse is established simply from run duration.

### 4.4 P17 is two different tracks

**P17 Continuous-Stack:** \`evaluation/continuous_stack.py\` tests unequal initial stacks and match continuation; it is not training. **P17 GPU Retrain:** \`experiments/ppo_five_hour.py\` and \`scripts/phase17_gpu_retrain.py\` manage active-runtime accounting, update-boundary safety, atomic checkpoint, resume, validation and PPO diagnostics.

Historical 2026-10-13 run:

| Metric | Recorded |
|---|---:|
| Attempted / accepted updates | 5000 / 5000 |
| Training hands | 640000 |
| Global update | 5104 |
| Active duration | 12743.274 seconds |
| Target duration | 18000 seconds |
| Stop reason | \`max_updates_ceiling\` |
| \`complete\` | \`false\` |
| Best global update | 2104 |
| Best repeated-validation bb/100 | 1450.4687 |

$$
640000=5000\times128,\qquad
\frac{12743.274}{18000}\approx 0.708.
$$

The target five active hours were **not completed**. Best validation was a repeated model-selection criterion, **not independent held-out performance**.

## 5. Expanded 6-max research track (E1–E7)

E1 \`expand/environment/table_manager.py\` and \`event_ledger.py\` add join/leave/sit-out/rebuy only at hand boundaries and preserve chip ledger. E2 \`expand/information/builder.py\`, \`encoder.py\`, \`actions.py\` encode 2–6 seats without revealing opponents' private cards. E3 \`expand/rl/collector.py\`, \`returns.py\`, \`ppo.py\`, \`trainer.py\` implement independent multi-seat neural PPO:

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),\quad
\hat A_t^{GAE}=\sum_{\ell\ge0}(\gamma\lambda)^\ell\delta_{t+\ell}.
$$

E4 \`expand/pluribus/blueprint/mccfr.py\` implements a research-scale **external-sampling MCCFR** blueprint. Its regret-matching foundation is

$$
R_i^T(I,a)=\sum_{t=1}^Tr_i^t(I,a),\qquad
\sigma_i^{T+1}(I,a)=
\begin{cases}
[R_i^T(I,a)]_+/\sum_b[R_i^T(I,b)]_+,&\sum_b[R_i^T(I,b)]_+>0,\\
1/|A(I)|,&\text{otherwise}.
\end{cases}
$$

Kuhn/Leduc toy validations do **not** certify six-player no-limit exploitability or reproduce original Pluribus. E5 \`expand/pluribus/search/belief.py\`, \`solver.py\` support depth-limited belief-dependent search and fallback under node/time constraints:

$$
b(h_1,\dots,h_n\mid h_{\rm public})\propto
\mathbf1[\text{card-consistent}]\,b_0(h_1,\dots,h_n)
\prod_{j,k}\hat\pi_j(a_k\mid h_j,I_{j,k}).
$$

E6 \`expand/opponent/model.py\`, \`adaptive.py\`, \`router.py\` infer opponent tendencies only from **public actions** and route among known experts with bounded policy divergence:

$$
\pi_{\rm adapt}=(1-\eta)\pi_{\rm base}+\eta q,\quad
D_{\rm KL}(\pi_{\rm adapt}\Vert\pi_{\rm base})\le\epsilon_{\rm KL}.
$$

E7 \`expand/integration/agent.py\` and \`benchmark.py\` combine compatible source policies using **FirstVerifiedValid** fallbacks, but E4/E5 information-set adapters and search are **not automatically certified end-to-end**. The previously recorded 164 passed / 7 skipped were isolated-package tests; full Windows/native-engine/CUDA acceptance remained pending.

## 6. HUNL GUI 1.0–1.6: functional evolution

| GUI phase | Purpose | Main implementation and processing |
|---|---|---|
| 1.0 | Human-vs-Bot single-hand interaction | \`game_service/session_manager.py\` owns state, \`interactive_adapter.py\` selects legal bot actions |
| 1.1 | HTTP contract | \`backend/main.py\`, \`game_router.py\`; revision conflict 409 / illegal 422 |
| 1.2 | Poker table view | \`frontend/src/HUNLApp.tsx\`, \`PokerTable.tsx\`; uses projected information |
| 1.3 | User-defined raises | \`game_service/action_adapter.py\`; validates raise-to in engine |
| 1.4 | PPO/QBoost frozen agents | \`game_service/checkpoint_catalog.py\`, \`agents/neural_policy.py\`; inference only |
| 1.5 | Match history/performance | \`game_service/session_analytics.py\`; history/bb/100/showdown |
| 1.6 | Persistent stacks across hands | \`game_service/session_manager.py\`; bankruptcy ends match |

$$
\operatorname{GUIObservation}_i=
\operatorname{PublicProjection}(s)\cup\operatorname{PrivateOwnCards}(s,i).
$$

**Model selection is not model execution.** Checkpoint configured, file found, format compatible, actor instantiated, legal action sampled, browser request succeeded, and completed match are separate acceptance conditions. GUI human play does not automatically update training weights.

## 7. Tournament ENV 1.1–1.7: functional evolution

| ENV phase | Purpose | Primary code and flow |
|---|---|---|
| 1.1 | Tournament engine ↔ HTTP | \`tournament_service/session.py\`, \`backend/tournament_router.py\` |
| 1.2/1.2.1 | React six-seat table/card display fixes | \`frontend/src/tournament/TournamentApp.tsx\` |
| 1.3 | Four improved heuristic bots | \`tournament_service/improved_bot.py\`, turn-state policy |
| 1.3.1 | Fix over-calling/folding | Evaluate raise when **either** native ID 2 or 3 is legal |
| 1.4 | Connect trained multi-seat actor | \`tournament_service/trained_actor.py\`; 7→5 action bridge |
| 1.5 | Blind/ranking/champion UX | Tournament service state and frontend controls |
| 1.6 | Persistent match history | \`tournament_service/journal.py\`: SQLite/WAL/replay/JSONL |
| 1.6.1 | Post-settlement showdown cards | \`environment/tournament_env.py\`; reveal only non-folded showdown hands |
| 1.7 | Release and regression audit | \`scripts/tournament_env17_release_audit.py\`; Git safety checks |

The allowed showdown reveal rule is:

$$
C_h=\{i:\neg\mathrm{folded}_{i,h}\},\quad
\mathrm{Reveal}(i,h)=
\begin{cases}
\mathrm{HoleCards}(i,h)&i\in C_h,\ |C_h|\ge2,\ |\mathrm{Board}|=5,\ \text{after settlement},\\
\varnothing&\text{otherwise}.
\end{cases}
$$

SQLite persistence and recorded replay do not mean every uncommitted action survives a hardware crash. Folded hole cards are never exposed through public replay, and private actor transitions remain separate from public showdown data.

The Tournament trained PPO was originally **multi-seat cash-game E3**, not a tournament-specific optimized model. Tournament-specific PPO must address blind progression, elimination and payout objectives; prospective prize objective:

$$
J_i^{prize}(\theta)=\mathbb E_{\pi_\theta,\pi_{-i}}[\mathrm{Payout}(\mathrm{FinalRank}_i)].
$$

This is an **objective proposal**, not proof of an existing ICM-aware implementation.

## 8. Acceptance gates and research priorities

1. **Rules Gate:** heads-up button/SB order, side pots, all-in/short-call, min-raise, settlements, elimination, showdown public/private projection.
2. **Actor Gate:** masked legality, model/encoder/action-contract fingerprint, no privileged critic data at inference, checkpoint file SHA, loaded actor inference.
3. **Agent Training Gate:** on-policy reproducibility, resume integrity, critic calibration, preflop/postflop coverage, stability over fixed updates.
4. **Evaluation Gate:** seed-blocked paired results, multiple train seeds, confidence intervals, held-out final tests, diverse frozen opponent pool.
5. **Expand Gate:** native 2–6-engine operation, MCCFR toy-vs-Holdem boundary, belief and source routing compatibility.
6. **Tournament Gate:** actual trained E3 actor smoke and separate tournament-specific retraining/payout evaluation.
7. **GUI Gate:** live browser click-to-action loop, HTTP codes, history/replay database consistency and privacy, frozen checkpoint selection.

$$
\widehat{\Delta J}_{bb/100}
=\frac{100}{K}\sum_{k=1}^K
(X_{candidate,k}-X_{baseline,k}),\quad
SE(\widehat{\Delta J})=\frac{100s_{\Delta X}}{\sqrt K}.
$$

Independent deck seed groups (not four matching games) are the experimental replicate. Evaluation against random/calling/tight bots does not prove equilibrium strength.

## 9. How to inspect the actual Python implementation

This report identifies real tracked paths and interfaces, but uses **execution-flow pseudocode rather than fabricated verbatim source snippets** where a direct code-level excerpt has not been independently matched. Treat the Git file as the implementation truth. Each phase has its own source document and \`tests/\` cases and CLI under \`scripts/\`.

\`\`\`powershell
git switch main
git pull --ff-only origin main
uv sync --extra dev --extra gui --extra gpu
uv run --extra dev --extra gui --extra gpu python -m pytest -q
cd frontend
npm.cmd ci
npm.cmd test
npm.cmd run build
\`\`\`

These commands are reproducibility instructions, **not tests newly run for this report**. The model weights in \`runs/*.pt\` are generally Git-ignored. A code push does not transfer neural checkpoints. HUNL P17, QBoost and Expand E3 models must not be interchanged without architecture/action/encoder checks.

## 10. Source registry and archival policy

- \`docs/AGENT_DEVELOPMENT_TECHNICAL_REPORT_2026-10-13.md\`: Base P0–17, Expand E1–E7, P15/16 diagnostics, GUI PR chronology.
- \`docs/TOURNAMENT_ENV_PHASE1_FULL_TECHNICAL_REPORT.md\`: detailed T1.1–1.7 including UI bugfixes, improved bot, actor mapping, SQLite and release QA.
- \`docs/PROJECT_REPORT.md\`: Base P0–16 canonical historical summary.
- \`docs/EXPAND_MASTER_DESIGN_6MAX_v1_0.md\` and \`docs/EXPAND_PHASE0*.md\`: originally proposed/implemented designs; preserve source/implementation distinctions.
- \`docs/PHASE15_*.md\`, \`PHASE16_*.md\`, \`PHASE17_*.md\`: original measurements, controls and unresolved issues.
- \`docs/GUI_1_*.md\` and \`TOURNAMENT_ENV_PHASE_1_*.md\`: detailed HTTP/GUI contracts and acceptance commands.
- \`docs/archive/INTEGRATED_SOURCE_SNAPSHOT_01.md\`: partial source archive created in earlier work; it is **not** evidence that all 72 originals have been archived.

**Deletion policy:** Do not delete historical markdown before its full original source text, provenance, and deep links have been verified in a complete archive. A consolidated synopsis is not a lossless replacement. Work branch/PR review should separately approve any cleanup. 

## 11. Next engineering change contract

For every future phase, document (1) Objective, (2) mathematical specification, (3) exact code changed with file path/function references, (4) input→call graph→output with legal/private invariants, (5) run commands and expected artifacts, (6) real test evidence, (7) limitations/remaining acceptance. Avoid replacing actual technical evidence with generic conceptual code. Every model checkpoint must record training seed, architecture, action contract, model hash, prior weights, validation-selection protocol and independent test ID.
