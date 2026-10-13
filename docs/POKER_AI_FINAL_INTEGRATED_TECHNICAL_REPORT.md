---
title: "Poker AI Final Integrated Technical Report — Environment · Agent · Expand · GUI · Tournament"
date: 2026-10-13
status: lossless-source-verified
tags: [HUNL, Environment, PPO, QBoost, MCCFR, Tournament, Obsidian]
---

# Poker AI Final Integrated Technical Report

**English** | [한국어](POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.ko.md)

```math
\boxed{\mathcal G=(N,H,Z,A,P,f_c,\{\mathcal I_i\}_{i\in N},\{u_i\}_{i\in N})}
```

```math
\boxed{s_{t+1}=T(s_t,a_t,\omega_t),\quad a_t\in A_{\mathrm{legal}}(s_t),\quad a_t\sim\pi_\theta(\cdot\mid I_{i,t})}
```

$`N`$ denotes the players; $`H`$, game histories; $`Z`$, terminal histories; $`A`$, actions; $`P`$, the function that selects the acting player; $`f_c`$, card-dealing probabilities; $`\mathcal I_i`$, player $`i`$'s information sets; and $`u_i`$, utility. The **Environment** applies the rules to the complete game state $`s_t`$, while the **Agent** selects an action from the information $`I_{i,t}`$ visible to it. $`\omega_t`$ denotes chance events such as shuffling and dealing cards.

This report is a technical reference that connects the complete development record of [TheMostEvident/holdem-ai](https://github.com/TheMostEvident/holdem-ai). It explains the **purpose, equations, actual Python implementation, inputs and outputs, execution flow, historical validation results, and current limitations** of each project and phase. The appendices contain the **complete original text of 327 Python files** and **76 Markdown documents**. These appendices form part of the final report; they are neither separate summaries nor separate backups.

## Contents

- [Reading order and project boundaries](#section-2f84bd477486)
- [Reference sources and evidence levels](#section-a030f2cc49ef)
- [Invariants shared by all phases](#section-7767092c3272)
- [Do the two reference reports preserve all the details?](#section-109058370c3c)
- [Base Environment: Purpose and Units of Execution](#section-64c943b181ee)
- [Tournament ENV — Connecting the Engine to a Six-Player Service for Human Play](#section-28f568c64e55)
- [HUNL Agent — Mathematics, Implementation, and Validation of Base P5–P17](#section-842d126f6ae5)
- [Expand Project — E1–E7](#section-73301c2cf7bd)
- [HUNL GUI Project — G1.0–G1.6](#section-299f12b6cbd0)
- [Verifying source integration and removing superseded reports](#section-c5edcf32d2ad)
- [Appendix A — Complete Source Index for All 327 Python Files](#section-a0fe31ec5c73)
- [Appendix B — Complete Original Text Index for All 76 Markdown Files](#section-27594225cc64)

<a name="section-2f84bd477486"></a>

## Reading order and project boundaries

| Project | Phase | Purpose | Input → output | Boundary with other projects |
|---|---|---|---|---|
| Base Environment | P0–P4 | Implement legal poker play and chip settlement | Cards, stacks, and legal actions → next state, settlement, and standings | Does not learn opponent strategies. |
| HUNL Agent | P5–P17 | Implement heads-up decisions, training, and evaluation | The player's information set → legal-action distribution and training checkpoint | Delegates game rules to the environment. |
| Expand | E1–E7 | Extend research to cash games with 2–6 players | Dynamic seats and multiplayer information sets → PPO, MCCFR, search, and opponent adaptation | Distinct from training a tournament-specific policy. |
| HUNL GUI | G1.0–G1.6 | Observe and control human–agent games | User input and frozen actor → game display and statistics | The GUI does not train models. |
| Tournament ENV | T1.1–T1.7 and revisions | Provide six-player knockout games, a GUI, history, and replay | Tournament configuration and human/bot actions → chip standings, eliminations, winner, and records | Connecting an Expand E3 checkpoint does not establish tournament-policy optimality. |

**Read the two P17 branches separately.** Continuous-stack evaluation changes the evaluation protocol; GPU retraining is a separate training experiment. The HUNL 7-action actor, multiplayer Expand 7-action actor, and Tournament 5-action native engine also use different interfaces.

<a name="section-a030f2cc49ef"></a>

## Reference sources and evidence levels

The reference commit for the original sources is [`6a8ae368ddc02f814140d3770bc5993eaf9a15e4`](https://github.com/TheMostEvident/holdem-ai/tree/6a8ae368ddc02f814140d3770bc5993eaf9a15e4). The interrupted appendix work was merged at [`e796afefaf5f44dfa8fa8fe8bf8ad0fae436547e`](https://github.com/TheMostEvident/holdem-ai/commit/e796afefaf5f44dfa8fa8fe8bf8ad0fae436547e). The report body was rewritten by cross-checking actual source code against individual phase documents; the source appendices retain the original bytes.

| Evidence | Meaning in this report |
|---|---|
| Actual implementation | Behavior verified in the reference commit's `.py` files |
| Historical tests and experiments | Results recorded in the relevant phase document; not results rerun for this documentation work |
| Verification for this integration | Checks of original file byte counts, Git blob SHAs, omissions, duplicates, and document links |
| Unverified claims and research limitations | Training strength, practical performance, convergence, or recovery guarantees that remain unproven |

This documentation work did not rerun training or modify the game engine. Training times, validation scores, and test counts retain the context of their original records.

<a name="section-7767092c3272"></a>

## Invariants shared by all phases

```math
\boxed{\sum_i S_{i,t}^{\mathrm{behind}}+P_t=C_h}
\qquad
\boxed{R_{i,h}^{BB}=\frac{S_{i,h}^{\mathrm{final}}-S_{i,h}^{\mathrm{start}}}{BB_h}}
```

Within hand $`h`$, the sum of remaining stacks and the pot stays constant when there is no rake or external transfer of funds. Joins, departures, and rebuys at a dynamic cash table are separate accounting events at hand boundaries. $`R^{BB}`$ is chip profit or loss normalized by that hand's big blind; it differs from placement rewards and prize utility.

```math
\boxed{\pi_\theta(a\mid I)=
\frac{\mathbf1[a\in A(I)]\exp\ell_\theta(I,a)}
{\sum_{b\in A(I)}\exp\ell_\theta(I,b)}}
```

$`\ell_\theta`$ is the actor logit, and $`A(I)`$ is the legal-action set at the time of observation. Apply the legal-action mask before normalization. Even if a training critic can access the full state, the actor's input must exclude opponents' private cards.

```math
\boxed{m_{E\to T}(0,1,2,3,4,5,6)=(0,1,1,2,3,\bot,4)}
```

This maps Expand's Fold, Check, Call, Half-Pot, Pot, Double-Pot, and All-In actions to Tournament's Fold, Check/Call, Half-Pot, Pot, and All-In actions. $`\bot`$ denotes an unsupported action. Map actions by name and legal context, discard probability mass for unsupported actions, and renormalize. Equal integer indices across the two models do not imply equal actions.

<a name="section-109058370c3c"></a>

## Do the two reference reports preserve all the details?

The two reports provide a starting point for understanding all phases, but they do not contain every interface detail, design decision, and experimental record from the other reports verbatim. The phase explanations were therefore expanded, and all original text was included in verified appendices before the standalone reports were removed.


| Original phase document and location | Specific details in the original | Coverage in the integrated Agent report | Coverage or difference in the integrated Tournament report |
|---|---|---|---|
| [docs/EXPAND_PHASE02_INFORMATION_AND_ENCODING.md](report_sources/ORIGINAL_REPORTS_01.md#md-016) §2 | Identical raise-to targets are deduplicated in the priority order `HalfPot → Pot → DoublePot → AllIn`. Targets outside the bounds are excluded from the mask rather than silently clamped. `raising_permitted` treats native ID 4 as authoritative; Half/Pot also check native IDs 2/3. | Part II §4.2 explains the 7-action order, short-all-in call normalization, integer targets, and masking. It does not specify this deduplication priority, the prohibition on clamping, or the detailed native-ID checks. | §7.4 explains E3's 7→5 action mapping in detail. E2's raise-candidate generation and deduplication rules are separate material. |
| [docs/EXPAND_PHASE02_INFORMATION_AND_ENCODING.md](report_sources/ORIGINAL_REPORTS_01.md#md-016) §3 | Seat `[6,19]`, Global `[110]`, and History `[128,21]` contracts; occupied sit-out seats remain in attention; only empty seats are padded; neural input uses only the most recent 128 history entries; BB normalization and sign-preserving Log1p; suit rotation does not imply model suit invariance. | Part II §4.3 retains the Seat/History shapes and the occupied/dealt_in distinction. Its summary omits Global 110, history truncation, Log1p, and the suit-invariance limitation. | This service report connects the E3 actor and does not specify these numerical preprocessing contracts for the E2 encoder. |
| [docs/EXPAND_PHASE04_MCCFR_BLUEPRINT.md](report_sources/ORIGINAL_REPORTS_01.md#md-018) §3.1–3.2 | `w_t=floor((t-1)/N)+1`; own-reach weighting in average-strategy accumulation, `S_i(I,a) += w_t * π_i(I) * σ_i(I,a)`; no double application of opponent/chance visitation sampling and weighting; iteration RNG reconstructed from a hash of `(seed, global_iteration)`. | Part II §6 explains external sampling, linear weighting, regret matching, prevention of partial-iteration saves, and Kuhn validation. It omits the average-strategy own-reach equation and the exact weight-schedule/RNG-reconstruction contracts. | The MCCFR solver is outside its scope; it omits these average-strategy and RNG contracts. |
| [docs/EXPAND_PHASE05_SUBGAME_SEARCH.md](report_sources/ORIGINAL_REPORTS_01.md#md-019) §2.1, §3 | The default belief samples candidate decks without weighting by public-action likelihood. The optional `opponent_likelihood(observation, action_id, raise_to)` computes probabilities from the corresponding opponent observation and uses rejection sampling. The memory budget estimates table storage at 4096 bytes per infoset; it is not an actual RSS limit. | Part II §7 explains the general Bayesian equation, card consistency, limitations of exact range estimation, and an estimated memory budget. It does not specify the default unweighted path, optional callback/rejection path, or 4096-byte coefficient. | It omits the detailed implementation contracts for search belief and resource estimation. |
| [docs/PHASE16_PPO_ABLATION.md](report_sources/ORIGINAL_REPORTS_04.md#md-045) §Arms, §Fairness controls | Critic Adam moments are reset in every arm. A0 therefore is not a bit-for-bit resume of Phase 15. A2 warm-starts Q token/hidden layers and initializes the projected V output from averaged Q heads, reducing the output parameter count. | Part V §16.2 explains the A0–A3 differences, Q/200 scaling, direct V, and KL rollback. It omits the critic Adam reset shared by all arms and A2's specific initialization/parameter differences. | It does not cover agent-optimizer ablations. |
| [docs/PHASE16_PPO_ABLATION.md](report_sources/ORIGINAL_REPORTS_04.md#md-045) §Safety and data integrity | `final_actor.pt` contains only the actor, without the critic or either optimizer state, and cannot serve as a PPO resume checkpoint. The pilot does not support auto-resume; a partial run is retained and a new run starts in a new root directory. | Part V explains ablation execution and the subsequent stress run's full-state `latest.pt`/resume. It does not distinguish the ablation's actor-only file from the stress run's full-state file at this level of detail. | It explains E3 tournament actor checkpoints but omits this Base P16 actor-only resume restriction. |
| [docs/PHASE17_FIVE_HOUR_GPU_RETRAIN.md](report_sources/ORIGINAL_REPORTS_04.md#md-047) §Objective and diagnostics, §Time budget | After an optimizer step, A0 creates `hard_stop.json` and refuses automatic resume if KL>0.10. A3 rolls back if KL>0.02. On resume, the scientific manifest must match in full; only `--max-updates-this-call` may change. | Part V/VI explains the A0 extreme-KL stop, A3 rollback, manifest, and resume. It does not specify P17's `hard_stop.json`, refusal of automatic resume, or the single permitted option change. This contract must not be confused with P15's 12 consecutive warnings in Part I. | This training-resume contract differs from Tournament journal/replay and is not included. |
| [docs/GUI_1_3_INTERACTIVE_BETTING.md](report_sources/ORIGINAL_REPORTS_02.md#md-025) §New HTTP endpoint, §Acceptance criteria | HUNL `/raise-to` request: `{"raise_to":350,"expected_revision":17}`. Missing, fractional, negative, below-minimum, and above-maximum amounts are rejected without changing state or revision. A response can contain several history entries because the bot may take consecutive automatic actions. | The Part VIII GUI table summarizes custom raises, engine validation, revision 409, and illegal-action 422. It omits this request example, state/revision invariance on invalid input, and the meaning of batched responses. | §1.3 summarizes the preceding HUNL GUI and separately details Tournament HTTP contracts. Those contracts do not replace these HUNL custom-raise endpoint details. |
| [docs/GUI_1_4_TRAINED_AGENT.md](report_sources/ORIGINAL_REPORTS_02.md#md-026) §Checkpoint API and file safety, §Validation and limits | `GET /bots` advertises only file existence; strict loading occurs at session creation. Malformed/incompatible files produce 422 with no random fallback. Actor caches are invalidated on file size/mtime changes; each session has an independent RNG. The checkpoint tests used a mock loader and did not validate inference with a genuine checkpoint. | Part VIII discusses frozen inference, model selection/path settings, and connecting actual local checkpoints. It does not record the HUNL catalog's cache-invalidation/RNG conditions or the GUI 1.4 fixture's mock-loader scope at this level. | §7 explains a separate Tournament E3 loader and format/action mapping. It does not include the HUNL GUI 1.4 cache/fixture contracts instead. |
| [docs/TOURNAMENT_ENV_PHASE_1_3_1_AGGRESSION_FIX.md](report_sources/ORIGINAL_REPORTS_06A.md#md-064) §Verification from a partial compatibility harness | Across 30 seeded tournaments/388 hands, raise frequencies (half/pot, excluding all-in) for engine seats 1–5 were **29.1%, 10.4%, 19.6%, 11.2%, and 28.5%**, respectively. These are conditional action diagnostics from a partial harness, not win rates or verification of the full Windows repository. | Part VIII outlines the Call/Fold bias and corrected pot-raise conditions. It omits the original measurements for each seat. | §6 retains the `2 or 3` bug, heuristic thresholds, diagnostics for 1326 starting hands, and the 30-seed/388-hand sample. It nevertheless omits these five seat-specific raise rates. |


Where historical notation in this table differs from the current implementation, the report body follows the actual source. For example, Expand E2's original equation specifies Global 110, whereas the current encoder's actual composition and `GLOBAL_DIM` are 114. The original notation remains unchanged in the source appendix for traceability.

<a name="section-64c943b181ee"></a>

## Base Environment: Purpose and Units of Execution

```math
\boxed{I_t^i=O_i(s_t),\quad a_t\in\mathcal A^{\mathrm{legal}}(I_t^i),\quad s_{t+1}=T(s_t,a_t),\quad R_i=\frac{S_i^{\mathrm{final}}-S_i^{\mathrm{initial}}}{BB}}
```

$`s_t`$ is the engine's internal state, including the deck and every participant's hole cards; $`I_t^i`$ is the observation available to the acting player $`i`$. $`T`$ accepts a legal action and produces the next game state, while $`R_i`$ is the player's chip change, measured in BB, after a hand ends. Building the Environment meant establishing these four contracts first. Policy training operates on top of them. If training incorporates incorrect blinds, reraising, or settlement, the policy optimizes a faulty simulator rather than actual poker.

**`HandEngine` in Base Phases 1 and 2.1**, **`MultiPlayerHandEngine` in Base Phase 3**, and **`TournamentEnv` in Base Phase 4** have distinct units of execution. The first wraps RLCard for a single heads-up hand; the second independently implements a single hand for two to six players; the third runs an elimination tournament that carries settled stacks forward. Although all three expose `reset()`, the first two start a new hand, while `TournamentEnv.reset()` restarts the entire tournament.

### Base Phase 0 — Development Foundation and Responsibility Boundaries

```math
\mathrm{Run}=F(\mathrm{source\ revision},\mathrm{dependency\ lock},\mathrm{configuration},\mathrm{seed}).
```

`source revision` is the Git commit, `dependency lock` is the resolved dependency set, `configuration` specifies stacks, blinds, and modes, and `seed` initializes randomness. All four must be tracked together to investigate the same run later. Phase 0 established a single repository and a reproducible Python environment in which subsequent engine, training, and evaluation files could operate.

The historical record describes Windows, Python 3.11, Git/GitHub, `uv`, and `.venv`. The project is named `holdem-ai`, and its Python import name is `holdem_ai`. Responsibilities were separated across Environment, Agents, Training, Evaluation, and UI; virtual environments and model outputs were excluded from Git tracking. The record also describes reconciling separate Git histories and resolving a README conflict during the initial push because the remote repository already had a history.

In the current snapshot, [src/holdem_ai/environment/game_state.py](technical_sources/PYTHON_SOURCE_07.md#py-082), [src/holdem_ai/agents/cfr_agent.py](technical_sources/PYTHON_SOURCE_06.md#py-063), and [src/holdem_ai/agents/rl_agent.py](technical_sources/PYTHON_SOURCE_06.md#py-068) are empty files. Their existence does not establish that a game-state class, CFR, or an RL Agent has been implemented. `create_architecture.py`, mentioned in earlier documents, is absent from the included source; its execution code is not reconstructed and presented as an actual implementation. Package-initialization files named `__init__.py` must also be distinguished from the later `base_agent.py` and `random_agent.py` implementations.

The workflow was to prepare the repository's dependencies, run the relevant phase's tests and audits, review the results and changes, and then commit. The commands below are instructions for rerunning verification, not test results obtained while writing this document.

```powershell
uv sync --extra dev
uv run --extra dev python -m pytest -q
```

Phase 0 established the research workflow and development environment; it did not validate poker rules or conduct policy-performance experiments.

### Base Phase 1 — Execution Contract for a Single RLCard Heads-Up Hand

```math
\pi_{\mathrm{random}}(a\mid I)=\frac{\mathbf1[a\in\mathcal A^{\mathrm{legal}}(I)]}{|\mathcal A^{\mathrm{legal}}(I)|},\qquad
\widehat J_i^{bb/100}=\frac{100}{H}\sum_{h=1}^{H}\frac{\Delta S_{i,h}}{BB}.
```

$`H`$ is the number of hands played, and $`\Delta S_{i,h}`$ is the chip change in the corresponding hand. The Random Agent samples uniformly from legal actions. This phase provides the minimal baseline needed to verify that `reset → observation → action → step → payoff` is connected end to end.

| Actual Python file | Responsibility |
|---|---|
| [src/holdem_ai/environment/hand_engine.py](technical_sources/PYTHON_SOURCE_07.md#py-083) | `HandEngine` and immutable `HandResult`; initialization, legal actions, single-hand execution, and BB payoff conversion |
| [src/holdem_ai/agents/base_agent.py](technical_sources/PYTHON_SOURCE_06.md#py-062) | The `Agent.select_action(state) -> int` protocol |
| [src/holdem_ai/agents/random_agent.py](technical_sources/PYTHON_SOURCE_06.md#py-067) | `RandomAgent`, which selects legal actions using an independent seeded RNG |
| [main.py](technical_sources/PYTHON_SOURCE_01.md#py-007) | CLI options, repeated hands, per-seat payoffs, and observed bb/100 output |
| [tests/test_hand_engine.py](technical_sources/PYTHON_SOURCE_22.md#py-264), [tests/test_rlcard_integration.py](technical_sources/PYTHON_SOURCE_27.md#py-315) | Adapter-contract checks and integration with actual RLCard |

The core selection code is excerpted from the actual `random_agent.py`.

```python
    def select_action(self, state: Mapping[str, Any]) -> int:
        legal_actions = tuple(state['legal_actions'])
        if not legal_actions:
            raise ValueError('There are no legal actions in this state.')
        return int(self._rng.choice(legal_actions))
```

[main.py](technical_sources/PYTHON_SOURCE_01.md#py-007) creates two Agents and repeatedly calls `engine.play_hand(agents)`. `HandEngine.play_hand()` passes only the acting player's observation to the policy, and `step()` first checks that the supplied integer ID belongs to the current legal actions. After termination, `get_result()` checks that the payoffs sum to zero and divides chip payoffs by BB. A neural network, CFR, or a probability calculator is not a prerequisite for this loop.

The current `hand_engine.py` is the version from Phase 2.1 onward: its default is `rules_mode='corrected'`, while `rules_mode='upstream'` selects the original RLCard implementation. `reset()` toggles `_next_button` to alternate the button across consecutive calls. These features must not be attributed to the initial Phase 1 code.

The historical record reports **5 tests passed**, a five-hand smoke test, and $`-50\;bb/100`$ for seat 0. The recorded payoffs yield $`100(-0.5-0.5+1-2-0.5)/5=-50`$. This small sample verifies connectivity; it cannot estimate the policy's average playing strength. RLCard's 54-dimensional `obs` also does not establish a complete Information State or perfect recall. The original five actions are `0=fold, 1=check/call, 2=half-pot, 3=pot, 4=all-in`.

### Base Phase 2 — Independent Rule Specification and Audit of the Upstream Engine

```math
\mathrm{SB}=b,\quad\mathrm{BBseat}=1-b,\quad
\mathrm{First}_{preflop}=b,\quad\mathrm{First}_{postflop}=1-b.
```

```math
k_i=\min(S_i,h-c_i),\qquad
\ell=\begin{cases}BB,&h=0\\h+r,&h>0,\end{cases}\qquad
u_i=c_i+S_i.
```

```math
\mathcal N_q=\{i:C_i\ge v_q\},\qquad
L_q=(v_q-v_{q-1})\,|\mathcal N_q|,\qquad
E_q=\{i\in\mathcal N_q:\neg\mathrm{folded}_i\}.
```

```math
P_q=L_q\mathbf1[|\mathcal N_q|\ge2],\qquad
U_i=\sum_q L_q\mathbf1[\mathcal N_q=\{i\}],\qquad
\sum_q P_q+\sum_i U_i=\sum_i C_i.
```

$`b`$ is the button in a heads-up game, $`S_i`$ is the stack remaining before the bet, $`c_i`$ is the player's cumulative commitment on the current street, $`h`$ is the highest commitment on that street, and $`r`$ is the size of the last full raise. $`k_i`$ is the actual cost to call, $`\ell`$ is the minimum full raise-to, and $`u_i`$ is the maximum raise-to. $`C_i`$ is the player's total contribution to the hand; $`v_q`$ denotes sorted positive contribution levels, with $`v_0=0`$. $`\mathcal N_q`$ is the set of contributors to a layer, $`L_q`$ is its total chip amount, and $`E_q`$ is the set of eligible recipients. $`P_q`$ is the amount retained as an actual pot when at least two players contributed to the layer; it is zero for a layer with only one contributor. $`U_i`$ is the sum of layers contributed solely by $`i`$, and those chips are returned to $`i`$ regardless of whether the player folded. A valid settlement requires a nonempty $`E_q`$ for every pot with at least two contributors; the actual `build_side_pots()` implementation rejects pots with no eligible recipient.

Phase 2 established an independent reference for checking positions, betting, and settlement rather than trusting RLCard's results. The helpers at this stage do not modify upstream execution. A diagnostic is expected to fail on cases that should fail its audit.

| Actual Python file | Purpose and core functions |
|---|---|
| `environment/seat_manager.py` | `heads_up_seats`, `post_blinds`; BTN=SB and blind all-ins for short stacks |
| `environment/actions.py` | `BetContext`, `validate_raise_to`, `legal_basic_actions`; validation of full raises, short all-ins, and permission to reopen betting |
| `environment/pot_manager.py` | `build_side_pots`, `settle_pots`; layer eligibility, uncalled returns, split pots, and odd chips |
| `environment/hand_evaluator.py` | `parse_card`, `rank_five`, `rank_seven`, `rank_showdown`; independent hand-ranking evaluation |
| `environment/rules_audit.py` | A read-only audit of raw RLCard state |
| [scripts/audit_rlcard.py](technical_sources/PYTHON_SOURCE_01.md#py-010) | CLI audit of the upstream engine, returning a nonzero exit code when violations are found |
| [tests/environment/test_position.py](technical_sources/PYTHON_SOURCE_18.md#py-210), `test_betting.py`, `test_side_pot.py`, `test_hand_evaluator.py` | Validation of the corresponding pure rule functions |
| [tests/test_rules_audit.py](technical_sources/PYTHON_SOURCE_27.md#py-317), [tests/test_rlcard_rules_integration.py](technical_sources/PYTHON_SOURCE_27.md#py-316) | Auditor checks and upstream integration tests |

In the table above, `environment/` denotes `src/holdem_ai/environment/`. The following is the actual decision logic from `validate_raise_to()`.

```python
    if not ctx.may_raise:
        raise ValueError('betting has not reopened for this player')
    if total <= ctx.highest_commitment:
        raise ValueError('raise must exceed the current highest bet')
    if total > ctx.max_total:
        raise ValueError('raise exceeds available stack')
    if total >= ctx.min_full_raise_to:
        return 'full_raise'
    if total == ctx.max_total:
        return 'short_all_in_raise'
```

A short all-in is permitted below the minimum full raise when it commits the player's entire stack. The engine separately tracks whether it restores a previous actor's right to reraise. The caller is also responsible for setting the pure helper's `may_raise` correctly.

Hand evaluation is defined mathematically as follows.

```math
\mathrm{rank}_7(H_i\cup B)=\max_{X\subseteq H_i\cup B,\ |X|=5}\mathrm{rank}_5(X),\qquad \binom75=21.
```

Cards use the user's notation: `sK`, `d7`, `cT`, and `hA`. `rank_five()` returns a `(hand category, principal rank, kicker...)` tuple, and the larger tuple wins. `rank_seven()` actually computes the maximum over 21 combinations and handles the A-2-3-4-5 straight while rejecting duplicate cards. `settle_pots()` uses integer division to distribute tied pots and assigns remaining chips clockwise, starting to the button's left.

The historical record reports **27 passed / 1 xfailed** and three upstream position/blind issues. Even if payoffs sum to zero across 100 hands, BTN, SB, and the first actor can still be wrong. Chip conservation and rule correctness therefore require separate checks. The upstream `xfail` at the time marks an expected failure caused by a known violation; it differs from a failure in the independent rule tests. There is no evidence of exhaustive differential testing against an external engine.

### Base Phase 2.1 — Corrected HUNL Adapter Within the Project

```math
\mathrm{RaiseTo}_f=c_i+d_i+\lfloor f(P+d_i)\rfloor,\quad f\in\{\tfrac12,1\},\qquad
\Delta\mathrm{raise}\ge r\Rightarrow\mathrm{full\ raise}.
```

$`d_i`$ is the unmatched amount to call at the current decision, and $`P`$ is the total pot already committed. Half-pot/pot raises calculate the target total from the pot after calling. The actual source rounds chips with `int()`, equivalent to flooring for positive calculations. The five betting IDs from Phase 1 were retained.

The purpose was to correct the preceding phase's findings in the execution path. Rather than editing RLCard inside the installed `.venv`, the project injects a local subclass and an independent betting round. A `rlcard==1.2.0` guard requires compatibility to be rechecked if the RLCard version changes.

| Actual Python file | Change and runtime responsibility |
|---|---|
| `environment/corrected_game.py` | `make_corrected_rlcard_env`; local `CorrectedGame`, reposted blinds, immediate fold termination, independent settlement, and restored step-back references |
| `environment/corrected_round.py` | `ActionId`, `HeadsUpBettingRound`; pending actors, action history after full raises, and reopening restrictions |
| `environment/hand_engine.py`, [main.py](technical_sources/PYTHON_SOURCE_01.md#py-007) | Default mode changed to corrected, with the upstream diagnostic path retained |
| [scripts/audit_corrected.py](technical_sources/PYTHON_SOURCE_01.md#py-008) | Position and payoff audit of the corrected engine |
| [tests/test_corrected_round.py](technical_sources/PYTHON_SOURCE_22.md#py-256), [tests/test_corrected_rlcard_integration.py](technical_sources/PYTHON_SOURCE_22.md#py-255), [tests/test_hand_engine.py](technical_sources/PYTHON_SOURCE_22.md#py-264) | Betting-round, actual RLCard-integration, and default-wrapper regressions |

The following is the actual blind-recovery and reposting logic in `CorrectedGame.init_game()`.

```python
            for player in self.players:
                player.remained_chips += player.in_chips
                player.in_chips = 0
                player.status = PlayerStatus.ALIVE
            btn = int(self.dealer_id)
            bb = 1 - btn
            self.players[btn].bet(chips=self.small_blind)
            self.players[bb].bet(chips=self.big_blind)
            for player in self.players:
                if player.remained_chips == 0:
                    player.status = PlayerStatus.ALLIN
            self.game_pointer = btn
```

The call flow is `HandEngine.__init__ → make_corrected_rlcard_env → CorrectedGame.init_game → HeadsUpBettingRound`. `init_game()` reuses the upstream seeded deck, card dealing, and player creation, then recovers the incorrectly posted blinds and reposts them. It also updates `dealer.pot` to the sum of the new contributions so that the first pot-size action is calculated correctly. `proceed_round()` resets the previous-actor set after a full raise; after a short all-in it requires the opponent to respond without automatically restoring a previous actor's right to reraise. `CorrectedGame.step(FOLD)` terminates the hand immediately instead of dealing unnecessary community cards. `get_payoffs()` computes awards using Phase 2's side-pot and hand-evaluation functions.

The historical record reports **43 passed / 1 xfailed** and a 100-hand audit of the corrected engine. The known failure in the upstream audit remains because upstream itself was not changed. This adapter is restricted to single heads-up hands and RLCard 1.2.0; the later six-player game was not built by extending this subclass. Remaining validation includes adversarial reraising, partial blinds, and comparison with external engines.

### Base Phase 3 — Independent Single-Hand Engine for Two to Six Players

```math
\mathcal O=\{i:S_i^{\mathrm{initial}}>0\},\quad
\begin{cases}
\mathrm{SB}=b,\ \mathrm{BBseat}=\mathrm{next}(b),&|\mathcal O|=2,\\
\mathrm{SB}=\mathrm{next}(b),\ \mathrm{BBseat}=\mathrm{next}(\mathrm{SB}),&|\mathcal O|\ge3,
\end{cases}
```

```math
\mathrm{PreFirst}=\begin{cases}b,&|\mathcal O|=2\\\mathrm{next}(\mathrm{BBseat}),&|\mathcal O|\ge3,\end{cases}\qquad
\mathrm{PostFirst}=\mathrm{next}(b).
```

```math
\sum_i S_{i,t}^{behind}+\sum_i C_{i,t}=\sum_i S_i^{initial},\qquad
\Delta S_i=A_i-C_i,\qquad S_i^{final}=S_i^{initial}+\Delta S_i,\qquad\sum_i\Delta S_i=0.
```

$`\mathrm{next}`$ denotes the next occupied physical seat clockwise, $`C_i`$ is the player's total contribution, and $`A_i`$ is the award including returns of unmatched chips. Phase 3 directly manages betting for two to six players and side pots for unequal stacks, which upstream could not provide. It allows Agents, tournaments, and the browser to reuse a single authoritative settlement implementation.

| Actual Python file | Implementation responsibility |
|---|---|
| `environment/multi_player_engine.py` | `MultiPlayerHandEngine`, `MultiPlayerHandResult`; deck, streets, pending actors, legal actions, raise-to, and settlement |
| `environment/multi_player_seats.py` | `next_occupied`, `multi_player_seats`, `rotate_button`; physical seating and transition to heads-up |
| `environment/hand_evaluator.py`, `environment/pot_manager.py` | Reuse of Phase 2's exact hand rankings and layer-based awards |
| [scripts/audit_multiplayer.py](technical_sources/PYTHON_SOURCE_01.md#py-009) | Audit of random hands, chip conservation, nonnegative stacks, and total awards |
| [tests/test_multi_player_engine.py](technical_sources/PYTHON_SOURCE_23.md#py-265), [tests/test_multi_player_seats.py](technical_sources/PYTHON_SOURCE_23.md#py-266) | Multiplayer execution, position, betting, all-in, and side-pot regressions |

The `amount` in `step_raise_to(amount)` is **the target total committed on the current street, including chips already committed on that street**, rather than the additional chips pushed forward. Arbitrary-amount actions outside native IDs `0~4` are recorded with ID `5` and their actual target. Both history fields must be preserved to reconstruct the exact bet during replay or training.

The actual reopening check for cumulative short all-ins is as follows.

```python
    def _raise_permitted(self, i: int) -> bool:
        # No betting into players who are already all-in/folded.
        if len(self._active()) < 2:
            return False
        if self.stacks[i] <= self._to_call(i):
            return False
        if i not in self.last_acted_high:
            return True
        # Multiple short all-ins cumulatively reopen action when >= full raise.
        return self._current_high() - self.last_acted_high[i] >= self.last_full_raise
```

In mathematical terms, let $`h_i^{last}`$ be the highest bet when the player last acted; reraising is permitted when $`h_t-h_i^{last}\ge r_t`$. The implementation tracks whether the cumulative increases from several short all-ins have reached the size of a full raise. A single short all-in does not reset that right by itself.

`reset()` restores `initial_stacks`, shuffles a 52-card deck, and deals one card at a time over two rounds, starting to the button's left. Blinds use `min(blind, stack)`. `_begin_action()` adds seats capable of acting to `pending`; the sole remaining active player acts only when there is a call amount owed against an all-in opponent. On the preflop street, if there are at least two active players, `_current_high()` retains a bring-in of at least the nominal BB even when the BB posts a short all-in.

`step()` and `step_raise_to()` both enter `_apply_action()`. After betting, the engine updates players still required to respond, and `_progress_if_closed()` determines a fold victory, a transition to the next street, or an all-in board runout. `_deal_next_street()` burns one card before each of the flop, turn, and river. `_settle()` generates awards and final stacks through `build_side_pots → rank_showdown → settle_pots` and checks conservation. If everyone is already all-in from posting blinds, `reset()` may return `(None,-1)`; `play_hand()` handles this terminal hand with no decision.

`observe(player_id)` is available only to the current actor and returns only that seat's two cards in `raw_obs.hand`. The internal `hole_cards` and deck objects themselves are not passed to policy inputs. Native `legal_actions` reflect the engine's full/short-raise and reopening checks. Arbitrary illegal raise targets are not silently corrected.

The historical record reports **74 passed / 1 xfailed** and execution of the multiplayer audit. The audit's `len(board)==5` count measures completed boards, which is not always identical to the number of actual contested showdowns. Principal limitations are the moving button, the absence of a formal dead-button policy, no direct `step_back()`, and incomplete differential validation against external rule engines. This engine's `reset()` initializes a new hand rather than carrying persistent stacks forward.

### Base Phase 4 — TournamentEnv Carries Hand Settlements Forward

```math
S_{i,h+1}^{start}=S_{i,h}^{final},\qquad
\mathcal A_h=\{i:S_{i,h}^{settled}>0\},\qquad
\mathrm{done}\iff |\mathcal A_h|=1.
```

```math
\ell(h)=\min\left(\left\lfloor\frac{h-1}{L}\right\rfloor,M-1\right),\qquad
R_i^{placement}=\frac1{|G_r|}\sum_{k=r}^{r+|G_r|-1}p_k\quad(i\in G_r).
```

$`h\ge1`$ is the hand number, $`L`$ is the number of hands per level, and $`M`$ is the number of configured blind levels. $`p_k`$ is the payout for place $`k`$, and $`G_r`$ is the set of participants tied at official place $`r`$. Simultaneous eliminations are ordered by the stacks at the start of that hand; equal starting stacks receive a shared place and the average payout across the places they occupy. The default payout is 1 for the winner and 0 for everyone else. The default payout in `placement_rewards()` is a separate function from the centered zero-sum winner reward in the later Agent reward module.

Phase 4 carries the accurate single-hand result from Base Phase 3 into long-term game state. It extends the unit of execution from cash-style play that resets stacks every hand to a single-table tournament with eliminations, rising blinds, and final places.

| Actual Python file | Implementation responsibility |
|---|---|
| `environment/tournament_env.py` | `TournamentEnv`, `TournamentHandRecord`, `TournamentResult`; carrying state across hands, automatic next hands, and tournament termination |
| `environment/blind_manager.py` | `BlindLevel`, `BlindManager.at_hand`; positive integer SB/BB, a nondecreasing schedule, and repetition of the final level |
| `environment/player_manager.py` | `PlayerManager.update`, `alive`, `winner`; persistent stacks, total chips, eliminations, and tied places |
| `environment/rewards.py` | `chip_rewards`, `placement_rewards`; BB conversion and final payouts for tied places |
| [scripts/run_tournament.py](technical_sources/PYTHON_SOURCE_05.md#py-054), [scripts/audit_tournament.py](technical_sources/PYTHON_SOURCE_01.md#py-011) | Tournament execution and audits of termination, stack continuity, and elimination |
| [tests/environment/test_elimination.py](technical_sources/PYTHON_SOURCE_18.md#py-208), [tests/test_tournament_env.py](technical_sources/PYTHON_SOURCE_27.md#py-318), [tests/test_player_manager.py](technical_sources/PYTHON_SOURCE_27.md#py-313), [tests/test_blind_manager.py](technical_sources/PYTHON_SOURCE_22.md#py-254), [tests/test_rewards_tournament.py](technical_sources/PYTHON_SOURCE_27.md#py-314) | Elimination, placement, blind, full-tournament, and placement-payout regressions |

The actual creation of a new hand appears in `TournamentEnv._next_hand()`.

```python
            level_index, level = self.blinds.at_hand(hand_number)
            before = self.players.stacks
            hand = MultiPlayerHandEngine(
                stacks=before, button=self.button,
                small_blind=level.small_blind, big_blind=level.big_blind,
                seed=self._rng.getrandbits(64),
            )
```

`reset()` initializes `PlayerManager` and the tournament RNG. `_next_hand()` passes the previous settled stacks into a new engine. `_finish_hand()` obtains `hand.get_result()`, applies chip conservation, the prohibition on reviving eliminated seats, and elimination places through `PlayerManager.update()`, and appends the hand record. The tournament ends when one player remains; otherwise, `rotate_button()` moves the button to the next surviving seat. The `_next_hand()` loop also records and settles subsequent hands that end immediately from posting blinds.

After a hand finishes, `step()` automatically starts the next hand. Actions always apply to the current seat, and eliminated physical seats remain in the array with a zero stack rather than being removed. Reaching `max_hands` or `max_actions` raises an error instead of inventing and returning a winner.

The historical record reports **113 passed / 1 xfailed**; all 100 of 100 tournaments terminated, with **1,086 hands and 500 eliminations**, averaging **10.86 hands**. It also records an example that ended after 15 hands with 6,000 chips for the winner. These are environment-audit results for Random Agents under the settings used at the time. The scope is a single-table knockout tournament with two to six physical seats; antes, bounties, rebuys, add-ons, time-based levels, multiple tables, and ICM strategy are not implemented. The current `tournament_env.py` captures cards for showdown disclosure, a later change from Tournament ENV 1.6.1 discussed separately below.

<a name="section-28f568c64e55"></a>

## Tournament ENV — Connecting the Engine to a Six-Player Service for Human Play

```math
\mathrm{NativeAction}\in\{0:F,1:C,2:\tfrac12P,3:P,4:A\},\qquad
\mathrm{HumanRaiseTo}\in\mathbb Z_{>0}\text{ subject to engine legality}.
```

Tournament ENV 1.1–1.7 connects the rule engine from Base Phases 3 and 4, the existing HUNL GUI design, and the already trained Expand PPO Actor to sessions, HTTP, a browser, and persistent records. This track concerns **execution-environment correctness and usability**; it is distinct from the Agent-training track that improves PPO losses or weights. When the trained mode is selected, seat 0 is Human, seat 1 is the Expand Phase 3 PPO, and seats 2–5 use four heuristic policies. The default `baseline_5` lineup and the explicitly untrained `improved_4` lineup are also retained.

| Lineup | Internal seat 1 | Internal seats 2–5 | Purpose |
|---|---|---|---|
| `baseline_5` | Calling Station | Four Calling Stations | Initial API compatibility and a simple baseline |
| `improved_4` | Untrained `controlled_pressure` placeholder | Four heuristic policies | Play against varied opponents |
| `trained_1_improved_4` | Frozen Expand Phase 3 PPO | Four heuristic policies | Tournament-transfer inference with a cash-game Actor |

### Tournament ENV 1.1 — Sessions, FastAPI, and Automatic Bot Execution

```math
\mathrm{Accept}(a)=\big[r_{client}=r_{server}\big]\land\big[\mathrm{actor}=0\big]\land\big[a\in\mathcal A_t^{legal}\big]\land\neg\mathrm{done},\qquad
r_{server}\leftarrow r_{server}+1\ \text{per accepted engine action}.
```

$`r`$ is the state revision. Automatic Bot actions increment the revision, as do Human actions. The purpose is to prevent duplicate browser clicks, delayed requests, or reordered responses from applying an action to a different turn, while automatically executing Bot actions until the next Human turn.

| Actual Python file | Initial creation/modification and function responsibilities |
|---|---|
| `tournament_service/__init__.py` | New service package |
| `tournament_service/session.py` | New `TournamentPolicy`, `TournamentGameSession`, and `TournamentSessionRegistry`; later phases extend the same file |
| `tournament_service/baseline.py` | New `TournamentBaseline`; baseline policies `random`, `calling_station`, `tight`, and `pot_pressure` |
| [backend/tournament_router.py](technical_sources/PYTHON_SOURCE_01.md#py-005) | New `build_tournament_router`; creation, retrieval, actions, raise-to, history, and deletion |
| [backend/tournament_schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-006) | New strict requests/responses; `StrictInt` and `extra='forbid'` |
| [backend/main.py](technical_sources/PYTHON_SOURCE_01.md#py-003) | Addition of the Tournament router alongside the existing HUNL router |
| [scripts/tournament_env1_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-059), [tests/test_tournament_env1_api.py](technical_sources/PYTHON_SOURCE_28.md#py-327) | Full-game API smoke test and request, state, and legality regressions |

In the table, `tournament_service/` denotes `src/holdem_ai/tournament_service/`. The actual session action function first checks the revision and the Human turn under a lock.

```python
        with self._lock:
            self._check_revision(expected_revision)
            if self.env.is_over() or self.env.current_player != 0:
                raise RuntimeError("not the human's turn")
            if type(action_index) is not int or action_index not in self.env._hand.legal_actions:
                raise ValueError("illegal human action")
            self._step(action_index=action_index)
            self._run_bots()
            return self._view_unlocked()
```

The call flow is `POST /sessions → TournamentSessionRegistry.create → TournamentGameSession → TournamentEnv.reset → _run_bots → Human-safe view`. Subsequent Human requests enter `play_action()` or `play_raise_to()`, and `_run_bots()` calls policies until the Human turn returns or the tournament ends. Each policy's returned value is checked again against the legal native IDs in its observation. `RLock` serializes concurrent changes to the same session, and the Registry manages a default capacity of up to 128 sessions. The tournament does not require a manual Next Hand button.

`_view_unlocked()` assembles the board, Human cards, per-seat chips, roles, current turn, and legal actions; it does not return the engine deck, opponents' hole cards, or Critic state. During play it compares `sum(stacks)+pot` with the tournament's total chips; after completion it compares `sum(stacks)` instead. Stale revisions and turn conflicts become HTTP 409, illegal input 422, and missing sessions 404. Resource/execution errors during creation are handled as 503. `Cache-Control: no-store` reduces the use of old revisions from cached responses.

The initial partial-compatibility harness in the historical documents reports **11 tests passed**, 25 Human actions, and termination after 10 hands. The later complete Tournament report records the user's Windows results: **11 new tests passed**, **713 passed / 1 xfailed** overall, and a smoke test with 25 Human actions, **13 hands, and 600 total chips**. Records from different environments and configurations must not be merged into a single run. History was held in memory at this stage; persistence was added in 1.6.

### Tournament ENV 1.2 — React Six-Player Table

```math
\mathrm{Enabled}(a)=\mathbf1[a\in\mathrm{state.actions}],\qquad
\sum_i\mathrm{state.seats}[i].stack+\mathrm{state.pot}=\mathrm{state.total\_chips}.
```

The first expression defines when an action button is enabled; the second is the browser's supplementary chip-conservation check. The purpose is to turn the API into a circular table, cards, and a betting panel that people can play directly. The server engine remains the final authority on legality and settlement.

This phase has **no new Python files**. It consumes `session.py`, `tournament_router.py`, and `tournament_schemas.py` from 1.1 unchanged. The historical frontend files are `frontend/src/App.tsx`, `HUNLApp.tsx`, `tournament/TournamentApp.tsx`, `TournamentTable.tsx`, `TournamentBetting.tsx`, `api.ts`, `types.ts`, `rules.ts`, `rules.test.ts`, and `styles.css`. The existing HUNL screen was separated into its own component tree, and both tabs' trees were retained to preserve UI state in the current browser.

`TournamentApp` receives the session-creation response and passes it to the table and input panel. It sends the selected action and `expected_revision` to the API, then displays the new response. Opponents' cards are rendered face down; the frontend does not fabricate showdown cards that the server did not provide at the time. Hand summaries and the most recent 25 records were retrieved from live session history. Free-form raise-to displays server-reported availability and minimum/maximum bounds and uses a separate endpoint.

The historical record reports successful TypeScript type checking and a Vite build, with **18/18 Vitest tests** passing. Inclusion in this appendix of complete Python source does not mean the frontend implementation was rerun and verified. Live-session restoration after a refresh or server restart was not implemented.

### Tournament ENV 1.2.1 — Human Card Rank and Position Fix

```math
\mathrm{HumanCardVisible}=\mathrm{RankVisible}\land\mathrm{SuitVisible}\land\mathrm{WithinSeatViewport}.
```

This is a card-rendering requirement, not a game-probability or training equation. This phase fixed a screen where the Human card suits remained visible but A/K/Q or numeric ranks disappeared and the cards were clipped at the bottom. The earlier report identified `.t-seat-cards .card-corner { display: none; }` and the placement of cards below the seat as the causes.

The changes in `frontend/src/styles.css` restored the Human card ranks and suits, placed cards above the seat, and adjusted responsive sizing. **No `.py` files were created or modified in this phase.** The server's `hero_cards` and opponent-card privacy contracts were retained. The historical record includes the user's confirmation that the display issue was resolved; separate Python-performance or policy results do not apply.

### Tournament ENV 1.3 — Four Heuristic Opponents

```math
q_i=\frac{\min(S_i,\max(0,d_i))}{\max(1,P+\min(S_i,\max(0,d_i)))},\qquad
v_i=\mathrm{score}(H_i,B)+\epsilon-\min(0.12,0.022\max(0,n_{rival}-1)).
```

$`q_i`$ is the pot odds, and $`v_i`$ is a heuristic action score. $`\epsilon=(U-0.5)\,0.035`$, with $`U\sim U[0,1)`$, is a small seeded perturbation; $`n_{rival}`$ is the number of opponents who are neither Fold nor Out. This score is not a showdown win probability or multiway equity.

More specifically, let the higher and lower preflop card ranks be $`x\ge y`$, with $`x_n=(x-2)/12`$ and $`y_n=(y-2)/12`$. The actual preflop score is

```math
s_{pre}=\begin{cases}
\min(0.99,0.54+0.43x_n),&x=y,\\
\mathrm{clip}_{[0,0.94]}\big(0.13+0.40x_n+0.24y_n+0.07\mathbf1_{suited}+g(x-y)+0.04\mathbf1_{x=14}\big),&x\ne y,
\end{cases}
```

Here, $`g(1)=0.07`$, $`g(2)=0.035`$, and other gaps contribute zero. After the flop, the policy finds the best five-card hand from its own cards and the board, incorporating a category-specific baseline score, pair/high-card ranks, and draws with four cards of one suit. On the river, if the board alone makes the same best hand, the score is capped at 0.49. This is not an equity calculation over an exact opponent range or combinations of remaining cards.

The purpose is to replace a game with five Calling Stations with an environment containing four distinct opponents. `improved` names a development slot; it does not report a measured win-rate improvement over the existing Bots.

| Actual Python file | Change and responsibility |
|---|---|
| `tournament_service/improved_bot.py` | New `PROFILES`, `_preflop_strength`, `_postflop_strength`, and `ImprovedTournamentBot.select_action` |
| `tournament_service/session.py` | Assembly of seat policies according to the lineup |
| [backend/tournament_schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-006) | Extension of `lineup`, seat roles, and policy names |
| [backend/tournament_router.py](technical_sources/PYTHON_SOURCE_01.md#py-005) | Passing the new lineup through the existing creation route |
| [scripts/tournament_env13_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-056), [tests/test_tournament_env13_improved_bots.py](technical_sources/PYTHON_SOURCE_28.md#py-320) | Checks of heuristic legality, lineups, and game termination |

Actual decision inputs are read solely from the observation, as shown below.

```python
        raw = observation["raw_obs"]
        hole = tuple(raw["hand"])
        board = tuple(raw["public_cards"])
        if len(hole) != 2 or len(board) not in (0, 3, 4, 5):
            raise ValueError("acting-player hand or board is malformed")
        strength = _preflop_strength(hole) if not board else _postflop_strength(hole, board)
        stack = int(raw["stacks"][raw["player_id"]])
        call = min(stack, max(0, int(raw["to_call"])))
        pot = max(0, int(raw["pot"]))
        pot_odds = call / max(1, pot + call)
```

`TournamentGameSession.__init__()` creates `tight_aggressive`, `balanced_value`, `pot_odds`, and `controlled_pressure` in that order for seats 2–5. In the original 1.3 implementation, seat 1 was a Calling Station placeholder; in the current source's `improved_4` lineup, it is a `controlled_pressure` placeholder following the 1.3.1 change. The default creation request remains `baseline_5`. Bots return only legal native IDs; free-form raise-to is a Human action path.

The historical partial-compatibility harness reports 11 existing API tests and 9 new tests, for **20 passed** in total. This does not replace the official React build or full Windows validation. This phase did not validate the four profiles' actual win rates, Nash exploitability, or ICM optimality.

### Tournament ENV 1.3.1 — Fixing the Branch That Omitted Raises

```math
\mathrm{RaiseCandidate}=(v_i\ge\tau_i)\land[(2\in\mathcal A^{legal})\lor(3\in\mathcal A^{legal})].
```

$`\tau_i`$ is the profile-specific raise threshold. The earlier code checked only the legality of ID 2, skipping consideration of a raise even when half-pot was illegal and pot was legal. The fix checks the union of both actions. Excessively high thresholds under the multiplayer penalty and seat 1's default Calling Station behavior were adjusted at the same time.

The current parameters in `improved_bot.py` are as follows.

| Profile | Raise threshold | Continue threshold | Aggression value |
|---|---:|---:|---:|
| tight_aggressive | 0.675 | 0.49 | 0.12 |
| balanced_value | 0.565 | 0.40 | 0.16 |
| pot_odds | 0.635 | 0.34 | 0.06 |
| controlled_pressure | 0.495 | 0.395 | 0.22 |

The aggression value is used in the source's pot-selection branch for `controlled_pressure`. It does not mean all profiles perform random raises with the same probability. A legal all-in is prioritized only when both the very high score condition $`v_i\ge0.87`$ and $`S_i\le\max(1,P)`$ hold.

The actual branch after the fix is as follows.

```python
        if value >= raise_threshold and (2 in legal or 3 in legal):
            if 3 in legal and (
                2 not in legal or value >= raise_threshold + 0.14 or
                (self.profile == "controlled_pressure" and self._rng.random() < aggression)
            ):
                return 3
            if 2 in legal:
                return 2
```

The modified files are `tournament_service/improved_bot.py` and `tournament_service/session.py`, with new [scripts/tournament_env131_diagnose.py](technical_sources/PYTHON_SOURCE_05.md#py-055) and [tests/test_tournament_env131_aggression.py](technical_sources/PYTHON_SOURCE_28.md#py-319). The GUI's placeholder description was also updated in `TournamentApp.tsx`. The diagnostic CLI enumerates $`\binom{52}{2}=1326`$ starting hands under fixed pot, call-cost, stack, and legal-mask conditions. The reported RaiseShare is **the proportion of policy choices under those conditions**, not a live-play frequency or win rate.

The historical partial harness reports **30 tests passed**, termination and chip conservation across **388 hands** over 30 seeds, and half-pot/pot raise rates of **29.1%, 10.4%, 19.6%, 11.2%, and 28.5%** for seats 1–5. All-ins are excluded from these raise rates. The record also includes the user's subsequent confirmation of the GUI update. This was not a seed-matched win-rate comparison. Existing live sessions retain their previously created policy objects, so the revised policy requires the updated Backend and a new tournament.

### Tournament ENV 1.4 — Inference with the Frozen Expand PPO Actor

```math
g(a_E)=\begin{cases}
0,&a_E=0,\\1,&a_E\in\{1,2\},\\2,&a_E=3,\\3,&a_E=4,\\4,&a_E=6,\\\text{undefined},&a_E=5,
\end{cases}
\qquad
\pi_T(a_E\mid I)=\frac{m_E(a_E)\pi_E(a_E\mid I)}{\sum_bm_E(b)\pi_E(b\mid I)}.
```

$`a_E`$ is one of Expand's seven actions `(fold,check,call,half-pot,pot,double-pot,all-in)`, and $`g`$ maps them to the five native actions. $`m_E`$ is a mask that transfers native-engine legality into Expand's output space. Only Check is permitted when `to_call==0`; only Call is permitted when it is positive. Double-pot, which has no native counterpart, is always masked and is not remapped to an arbitrary alternative.

The purpose is to connect the **Expand Phase 3 cash-game PPO in read-only mode** to seat 1 while preventing heads-up HUNL weights from being used incorrectly as a six-player model. This phase performs neither knockout-specific training nor reoptimization against tournament rewards.

| Actual Python file | Change and responsibility |
|---|---|
| `tournament_service/trained_actor.py` | New `status`, `_sha256`, `_prepare_public_state`, and `TrainedTournamentActor` |
| `tournament_service/session.py` | Assembly of `trained_1_improved_4`, collection of public history immediately before each action, and delivery to the Actor |
| [backend/tournament_router.py](technical_sources/PYTHON_SOURCE_01.md#py-005), [backend/tournament_schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-006) | `/agent-status`, trained lineup, and status contracts |
| [scripts/tournament_env14_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-057), [tests/test_tournament_env14_trained_actor.py](technical_sources/PYTHON_SOURCE_28.md#py-321) | Actual-checkpoint smoke test, observations, masks, and rejection of invalid formats |
| `expand/information/schema.py`, `encoder.py`, `expand/rl/trainer.py` | Reuse of the existing Expand InformationState, encoder, and `load_trained_actor` |

The following excerpts show the actual mapping constant and output mask.

```python
EXPAND_TO_NATIVE: tuple[int | None, ...] = (0, 1, 1, 2, 3, None, 4)
```

```python
    mask = (
        0 in legal_native,
        1 in legal_native and raw_call == 0,
        1 in legal_native and raw_call > 0,
        2 in legal_native,
        3 in legal_native,
        False,  # double-pot is NOT a native action: do not silently remap
        4 in legal_native,
    )
```

`from_environment()` reads `HOLDEM_AI_TOURNAMENT_CHECKPOINT`, `HOLDEM_AI_TOURNAMENT_DEVICE`, and `HOLDEM_AI_TOURNAMENT_POLICY_MODE`. Only CPU/CUDA devices and sampled/greedy modes are accepted. `load_trained_actor()` validates compatibility with `expand-e03-v1`, and the first 12 characters of the file's SHA-256 appear in the policy name. `status()` reports only path configuration and file existence; `validated=False` means the full model has not yet been loaded. A trained request fails if the model is missing or has an invalid format; it does not silently fall back to a Random Agent.

`_prepare_public_state()` converts the player's own cards, board, six physical seats' states, BB-normalized chips, position, legal specification, and preceding public actions into an Expand `InformationState`. `select_action()` calls `encode_observation → model.actor.action_probabilities` and removes illegal probability mass. Sampled mode draws in proportion to the remaining probabilities, while greedy mode selects the legal maximum; finally, $`g`$ is applied and native legality is checked. This path does not call the model's Critic. `_policy_history` stores the street, pot, call amount, and raise target **immediately before execution** of each action and is cleared for a new hand. Future deck contents, other players' cards, and training state are not included in this observation.

The historical final Windows smoke test loaded `runs/expand_phase03_pilot/best.pt` and recorded **13 hands, 2 Human actions, and chip conservation true**. This does not mean the user's weights were present in the environment used to write this document. Principal limitations are the distribution shift from cash games to knockout tournaments, the action-space change caused by removing double-pot, and tournament performance that has not yet been demonstrated.

### Tournament ENV 1.5 — Displaying Settled Stacks, Official Places, and Blind Progress

```math
\mathcal A_h=\{i:S_{i,h}^{settled}>0\},\qquad
\widetilde r_i=1+\big|\{j\in\mathcal A_h:S_{j,h}^{settled}>S_{i,h}^{settled}\}\big|,
```

```math
H_{to\ next}=L-((h-1)\bmod L).
```

$`\widetilde r_i`$ is a surviving player's provisional chip rank, distinct from the engine's confirmed elimination place $`r_i`$. $`H_{to\ next}`$ is the number of hands remaining until the next blind level, including the current hand. At the final level or after completion there is no next level, so related fields are `null`.

The purpose is to distinguish surviving players, eliminated players, chip leaders, and champions correctly, without misclassifying an all-in player with `seats[i].stack==0` as eliminated. During betting, `stack` means chips behind, while `tournament_stacks` holds the balances settled at the end of the previous hand. A surviving all-in player remains in `alive_seats` even when their current chips behind are zero.

The modified `.py` files are `tournament_service/session.py` and [backend/tournament_schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-006); the new regression suite is [tests/test_tournament_env15_progress.py](technical_sources/PYTHON_SOURCE_28.md#py-322). The frontend extends `tournament/standings.ts`, `standings.test.ts`, `TournamentProgressPanel.tsx`, `TournamentApp.tsx`, `TournamentTable.tsx`, `rules.ts`, `rules.test.ts`, `types.ts`, and `styles.css`.

The following is an excerpt of actual service-state construction.

```python
            "alive_seats": list(env.players.alive),
            "tournament_stacks": list(env.players.stacks),
            "blind_progress": {
                "hands_per_level": env.blinds.hands_per_level,
                "hand_in_level": level_hand,
                "hands_to_next": (env.blinds.hands_per_level - level_hand + 1) if next_level is not None and level_hand is not None else None,
                "next_small_blind": next_level.small_blind if next_level is not None else None,
                "next_big_blind": next_level.big_blind if next_level is not None else None,
                "final_level": level_index + 1 >= level_count,
            },
```

`_view_unlocked()` projects the authoritative information from `PlayerManager`, and the browser's `standings` computes provisional ranks among survivors. The UI distinguishes provisional `~N` from confirmed `#N`. This phase does not change blinds, settlement, or PPO parameters. Its individual acceptance document was a rerun checklist; the subsequent 1.7 report records completed integration verification. This handbook does not invent independent execution times or test counts for the individual phase.

### Tournament ENV 1.6 — Persistent SQLite History, Recorded Replay, and JSONL Export

```math
\mathrm{Replay}_h=\big[(I_{h,1}^{recorded},a_{h,1}),\ldots,(I_{h,T}^{recorded},a_{h,T}),\mathrm{Settlement}_h\big],
```

```math
D_{h,t}=\big(I_{h,t}^{actor},a_{h,t},\Delta S_{actor,h},\mathrm{session\_id},h,t\big).
```

$`I^{recorded}`$ is the stored frame immediately before an action, while $`I^{actor}`$ contains that actor's own cards and the public state. $`\Delta S`$ is `null` for unfinished hands. The purpose is to retain access to **already committed action and settlement records** after a server restart or live-session closure, and to export them as analyzable data.

| Actual Python file | Change and responsibility |
|---|---|
| `tournament_service/journal.py` | New `TournamentJournal`; SQLite initialization, `begin`, `record_step`, `record_initial_hands`, `close`, `hands`, `replay`, and `actor_export_rows` |
| `tournament_service/session.py` | Pre-action capture in `_step`, Journal call after success, and cleanup after creation failure |
| [backend/tournament_router.py](technical_sources/PYTHON_SOURCE_01.md#py-005) | Archive, completed hands, recorded replay, and NDJSON downloads |
| [backend/main.py](technical_sources/PYTHON_SOURCE_01.md#py-003) | Injection of `TournamentJournal` into the Registry and path configuration for tests |
| [tests/test_tournament_env16_persistence.py](technical_sources/PYTHON_SOURCE_28.md#py-324) | DB reopening, preservation after session closure, Replay, and Export contracts |

The default DB path is `.local/tournament_history.sqlite3`, configurable through `HOLDEM_AI_TOURNAMENT_JOURNAL`. `_connect()` uses `foreign_keys=ON`, a 15-second busy timeout, WAL, and `synchronous=FULL`. `tournament_sessions` stores configuration, status, revision, and winner; `tournament_actions` stores `(session_id,hand_number,action_no)` and pre-action data; `tournament_hands` stores settlement results.

The following is the actual public Replay projection.

```python
    @staticmethod
    def _public_replay(action: Mapping[str, Any]) -> dict[str, Any]:
        # Return hero hole only for the human's own turn. Never expose AI holes.
        result = {k: v for k, v in action.items() if k != "actor_hole"}
        result["hero_hole"] = action.get("actor_hole", []) if action["seat"] == 0 else []
        return result
```

`TournamentGameSession._step()` first captures the pre-action `board`, `stacks`, `street_bets`, `contributions`, `status`, `pot`, `to_call`, native legal actions, and the actor's own cards. It also resolves and records the actual raise target. If the engine rejects the action, execution never reaches the Journal call. On success, it increments the revision and passes the action and **all** hands newly completed during that step to `record_step()`. Multiple hands may complete because the next hand can terminate immediately from posting blinds. The Journal commits the action INSERT, settlement INSERTs, and session UPDATE in **one DB transaction**.

Public Replay hides an AI actor's `actor_hole` and retains `hero_hole` only in Human action frames. `actor_export_rows()` includes the corresponding actor's `own_hole_cards` in each row's observation and LEFT JOINs completed hands to return that actor's chip payoff as a label. The export schema is `tournament.actor_transition.v1`, storing the native ID and actual `raise_to`. Although each observation is built from its actor's information at decision time, this is **a local analysis file containing private cards from multiple actors**. It is not a complete rollout ready for direct PPO use with on-policy log probabilities, GAE, and advantages.

`DELETE /sessions/{id}` removes the live session from the in-memory Registry and updates its DB status to closed/complete without deleting its records. `record_initial_hands()` also stores hands that terminate on reset without a decision. After a server restart, Archive can query the DB again, but the live engine is not restored. Recorded Replay, seeded engine resimulation, and live Resume are separate features; the latter two are not implemented.

The historical record describes successful persistence tests on the user's Windows system and confirmation that the actual Replay screen worked; the full results appear in 1.7. In the source, the in-memory engine and revision have already changed before the DB transaction. Global atomicity that rolls back engine state on a DB write failure, and recovery after such a failure, are therefore not guaranteed. WAL/FULL alone also cannot guarantee lossless preservation of uncommitted actions just before process termination or in the event of disk failure. There is no retroactive migration of earlier in-memory history into the DB.

### Tournament ENV 1.6.1 — Revealing Contested Showdown Cards After Settlement

```math
\mathcal C_h=\{i:\neg\mathrm{folded}_{i,h}\},\qquad
\mathrm{Reveal}(i,h)=\begin{cases}
H_{i,h},&\mathrm{settled}_h\land|\mathcal C_h|\ge2\land|B_h|=5\land i\in\mathcal C_h,\\
\varnothing,&\text{otherwise}.
\end{cases}
```

$`\mathcal C_h`$ contains participants whose result is neither Fold nor Out. The purpose is to make opponents' hands visible after the contest while preventing leaks of opponents' cards during play or of folded cards. Disclosure timing is therefore placed in the settlement layer.

| Actual Python file | Change and responsibility |
|---|---|
| `environment/tournament_env.py` | `TournamentHandRecord.showdown_hole_cards`; capture in `_finish_hand()` before the next hand |
| `tournament_service/session.py` | Creation of the public `showdown_cards=[{seat,cards}]` result in `_hand_summary()` |
| [backend/tournament_schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-006) | Strict response contracts for `TournamentShowdownSeat` and `TournamentHandView.showdown_cards` |
| [tests/test_tournament_env161_showdown.py](technical_sources/PYTHON_SOURCE_28.md#py-323) | Privacy before settlement and on fold termination, contested-showdown disclosure, and DB/Archive compatibility |

The actual capture occurs inside `_finish_hand()`, which reads the completed engine.

```python
        contenders = [i for i, folded in enumerate(result.folded) if not folded]
        showdown_hole_cards = (
            tuple(tuple(self._hand.hole_cards[i]) if i in contenders else ()
                  for i in range(self.num_players))
            if len(contenders) >= 2 and len(result.board) == 5 else ()
        )
```

The card tuple is stored in the result record before the next hand replaces `self._hand`. Public results are displayed only in `last_hand`, live history, Archive hands, and Replay's `settlement`. Future settlement cards are not inserted retroactively into Replay `actions` or actor-export observations. A winner's cards are also not revealed when all opponents fold. The frontend extends `TournamentShowdown.tsx`, `TournamentApp.tsx`, `TournamentArchive.tsx`, `types.ts`, and `styles.css`.

Older DB rows without this field are treated as empty lists; cards that were not stored are not reconstructed retroactively. The historical record reports successful new and related regressions and the user's confirmation on the actual screen; the final full test run is included in the 1.7 results. In the current initialization path, hands completed on reset are saved to the DB, but the `_last_finished` update path differs from the one after an ordinary step. The live `last_hand` display for an initial hand ending without a decision is therefore also worth checking as a regression case.

### Tournament ENV 1.7 — Integration Verification and Independent Environment Release

```math
\mathrm{ReleaseReady}=\mathrm{PythonRegression}\land\mathrm{TrainedActorSmoke}\land\mathrm{FrontendChecks}\land\mathrm{GitPathAudit}\land\mathrm{DiffReview}.
```

The individual checks detect different failures in gameplay, inference, the display, and release artifacts. Phase 1.7 verifies that patches from multiple phases work alongside HUNL and publishes the Environment release changes to GitHub separately from the Agent experimentation branch.

| Actual file | Purpose |
|---|---|
| [scripts/tournament_env17_release_audit.py](technical_sources/PYTHON_SOURCE_05.md#py-058) | New `unsafe_reason`, `check_release`; audits of staged/tracked paths, ignore rules, and whitespace |
| [tests/test_tournament_env17_release_audit.py](technical_sources/PYTHON_SOURCE_28.md#py-326) | Unsafe-path and auditor regressions |
| [tests/test_tournament_env17_integration.py](technical_sources/PYTHON_SOURCE_28.md#py-325) | Full contracts for HUNL/Tournament coexistence, Showdown, and Replay |
| `scripts/tournament_env17_verify.ps1` | Combined execution of pytest, trained-Actor smoke, TypeScript, Vite, Vitest, and Git audit |
| `scripts/tournament_env17_stage.ps1`, `tournament_env17_manifest.txt` | Selective staging using an allowed-file manifest |
| `.gitignore` | Protection of paths for models, DBs, private records, virtual environments, and build outputs |

The actual path-classification logic includes the following.

```python
    if clean.startswith(DANGEROUS_DIRS) or any(p in ('__pycache__', '.pytest_cache') for p in parts):
        return 'generated/private directory'
    if name == '.env' or name.startswith('.env.') and name != '.env.example':
        return 'environment secrets'
    if name.endswith(DANGEROUS_EXT):
        return 'model/data/binary artifact'
```

`check_release()` checks the Git root and project files, then obtains staged/tracked file lists with `git diff --cached --name-only -z` and `git ls-files -z`. Tracked unsafe paths cause failure; under `--strict`, missing ignore rules or an empty staged-file set also cause failure. It applies `git diff --cached --check` to staged changes. It does not detect every secret embedded directly in source text, so path auditing and diff review have different scopes.

The historical release record describes staging 63 allowed files, removing trailing whitespace from three files, and creating an Environment-specific branch from the then-current `agent/phase15-collapse-diagnostics` branch to commit the changes. Cherry-picking the original Environment commit `825a507` onto the latest main produced `b2313fb`. The same change set yields a different Git commit SHA when its parent commit differs. This is a record of publication at that time, not a description of the current branch state.

The following are **historical results from the user's Windows run on 2026-10-13**, distinguished from any verification rerun while preparing this integrated document.

| Verification | Recorded result | Scope of the result |
|---|---|---|
| Full Python suite | **757 passed / 1 xfailed / 1 warning**, 38.06 seconds | Full regression suite at the time |
| Actual Expand Actor smoke | Passed; 13 hands, 2 Human actions, chip conservation true | Actual loading and inference with the user's local `best.pt` |
| TypeScript | `tsc -b --pretty false` passed | Type checking |
| Vite | Production build passed, 48 modules | Frontend build |
| Vitest | **25 passed / 5 files** | HUNL/Tournament frontend regressions |
| Git Privacy Audit | Passed | Staged/tracked path audit |
| GitHub Push | `d818f44..b2313fb`, main synchronization recorded | Release publication at the time |

This record establishes gameplay execution and the recording, retrieval, and display contracts for a Human, a frozen multi-seat PPO, and four heuristic opponents. Tournament superiority of the cash-game PPO, certification of rules for production play, and security for a public multiuser service require separate validation. The default Backend is an unauthenticated local research server bound to `127.0.0.1:8000`; the frontend's default address is `127.0.0.1:5173`.

### Final Tournament ENV HTTP Contract and Development Handoff

| Method | Path | Result |
|---|---|---|
| GET | `/api/v1/tournament/health` | Local-service health |
| GET | `/api/v1/tournament/agent-status` | Checkpoint configuration, existence, and mode; not full model-load validation |
| POST | `/api/v1/tournament/sessions` | New tournament from lineup, stacks, levels, and seed |
| GET | `/api/v1/tournament/sessions/{id}` | Current Human-safe state |
| POST | `/api/v1/tournament/sessions/{id}/actions` | Native ID and expected revision |
| POST | `/api/v1/tournament/sessions/{id}/raise-to` | Integer street-total target and expected revision |
| GET | `/api/v1/tournament/sessions/{id}/history` | Completed hands in a live session |
| DELETE | `/api/v1/tournament/sessions/{id}` | Close the live session and retain the Journal |
| GET | `/api/v1/tournament/archive` | SQLite session list |
| GET | `/api/v1/tournament/archive/{id}/hands` | Persistently stored completed hands |
| GET | `/api/v1/tournament/archive/{id}/hands/{h}/replay` | Recorded pre-action frames and settlement |
| GET | `/api/v1/tournament/archive/{id}/export` | Actor-visible `application/x-ndjson` |

A developer taking over should begin rule changes in `multi_player_engine.py`, `pot_manager.py`, and `player_manager.py`; session-action and frame-capture changes in `tournament_service/session.py`; policy-compatibility changes in `trained_actor.py`; and storage-contract changes in `journal.py`. Adding a schema requires coordinated checks of Python Pydantic responses, TypeScript types, live views, and older Archive rows. Preserving the distinctions between the native five-action space and Expand's seven-action space, current chips behind and settled stacks, and decision-time observations and post-termination public results is essential to continuing work on this environment.

<a name="section-842d126f6ae5"></a>

## HUNL Agent — Mathematics, Implementation, and Validation of Base P5–P17

This track connects trainable policies to the rules engine and evaluates them under reproducible conditions. **Base P5–P17 covers two-player HUNL research, distinct from the two- to six-player training in Expand E1–E7 and the operational environment in Tournament ENV 1.x.** The code below is excerpted from the actual current implementations in `src/holdem_ai/`. In the short code blocks, only the indentation inherited from the enclosing function or class has been reduced for readability; these blocks are not standalone executable examples. See the source appendices for complete code with its original context and indentation. The `holdem_agent_phaseN/` paths in earlier documents refer to the initial add-on package distribution and must not be interpreted as current source locations. Test counts, throughput measurements, and GPU training results below consolidate earlier documents and user execution records; they were not rerun during this documentation edit.

### P5 — Agent Interface: The Contract Between the Engine and a Policy

```math
I_t^i=O_i(s_t,h_t^{\mathrm{public}}),\quad
a_t\sim\pi(\cdot\mid I_t^i),\quad
m_a(I)=\mathbf1[a\in\mathcal A_{\mathrm{legal}}(I)].
```

```math
\pi^m(a\mid I)=
\begin{cases}
\displaystyle\frac{m_aw_a}{\sum_bm_bw_b},&\sum_bm_bw_b>0,\\
\displaystyle\frac{m_a}{\sum_bm_b},&\text{all legal-action weights are zero}.
\end{cases}
```

`s` is the complete engine state, `I` is the information visible to the acting player, and `w` contains the nonnegative weights proposed by the policy. Invalid weights—NaN, infinity, or negative values—raise an error. This phase establishes a common `act(observation, rng) -> action_index` contract for random policies, saved CFR strategies, neural networks, and other policies. `CFRTablePolicy` samples an existing strategy; it is not a CFR solver that updates regrets.

| Current Python path | Input → output / role |
|---|---|
| [src/holdem_ai/core/schema.py](technical_sources/PYTHON_SOURCE_06.md#py-070) | `InformationState`, `Turn`, `Terminal`, `TurnBasedEnvironment`; contracts for observations, legality, and terminal rewards |
| [src/holdem_ai/agents/policies.py](technical_sources/PYTHON_SOURCE_06.md#py-066) | Observation + RNG → legal action; masking and Random/CFR-table/ε-greedy policies |
| [src/holdem_ai/environment/bet_abstraction.py](technical_sources/PYTHON_SOURCE_07.md#py-078) | Legal betting context from the engine → seven candidates and a mask |
| [src/holdem_ai/training/reward_models.py](technical_sources/PYTHON_SOURCE_17.md#py-198) | Starting/ending chips or the winning seat → utility vector |
| [src/holdem_ai/training/episode_runner.py](technical_sources/PYTHON_SOURCE_16.md#py-192) | Environment + per-seat policies → rewards for all seats and transitions between turns of the same seat |
| [src/holdem_ai/environment/rlcard_adapter.py](technical_sources/PYTHON_SOURCE_08.md#py-092) | RLCard state → standard Turn/Terminal; private-field whitelist |
| [src/holdem_ai/utils/snapshot.py](technical_sources/PYTHON_SOURCE_17.md#py-203) | Experimental cloning; RNG independence requires verification for each engine |

The seven policy actions are `fold, check, call, raise_half_pot, raise_pot, raise_double_pot, all_in`. For cumulative street contribution `c`, additional amount to call `k`, pot `P` immediately before the action, and multiplier `f`:

```math
B_{\mathrm{to}}^{(f)}=c+k+\mathrm{round}_{\mathrm{chip}}[f(P+k)],
\qquad f\in\{1/2,1,2\}.
```

`raise_to` is the **target total contribution on the current street**. The engine determines minimum and maximum raises, short all-ins, and whether raising is reopened. Candidates are adjusted to the legal bounds and deduplicated; if a regular raise overlaps an all-in, the dedicated all-in head is retained. The actual integer-rounding and deduplication implementation is:

```python
# src/holdem_ai/environment/bet_abstraction.py
extra = (numer * (ctx.pot + ctx.to_call) + denom // 2) // denom
target = min(max(called_to + extra, ctx.min_raise_to), maximum)
if target <= called_to or target >= all_in_to or target in seen_targets:
    continue
actions[index] = EngineBetAction("raise_to", target)
seen_targets.add(target)
```

```math
r_i^{\mathrm{hand}}=\frac{S_{i,\mathrm{end}}-S_{i,\mathrm{start}}}{BB},
\qquad
r_i^{\mathrm{winner}}=\mathbf1[i=w]-\frac1N.
```

These rewards serve different objectives. Hand-level chip EV and tournament-winning utility must not be mixed as though they were the same quantity. In `run_self_play`, `next_observation` refers to **the same seat's next turn**, and the terminal reward is attached only to that seat's final transition. The full `EpisodeResult.rewards` includes seats that never acted.

```python
# src/holdem_ai/training/episode_runner.py
for seat, (index, obs, action) in pending.items():
    completed.append(Transition(index, seat, obs, action, None, outcome.rewards[seat], True))
```

Execution repeats `reset → Turn → policy.act → assert_legal → step`, then finishes with `Terminal → close pending transitions`. The historical integration record reports **176 passed, 1 xfailed**, while the initial kit's own documentation explicitly states that some integration with a real RLCard engine was skipped. Validation scopes from different points in time therefore need to be distinguished. Stable JSON in `infoset_key()` and corrected NumPy serialization help reproducibility but do not prove perfect recall. Snapshots also require separate tests of branch independence for the actual deck, RNG, and public history.

### P6 — HUNL200 Adapter: Separating the Actual Engine from Training State

```math
S_{0,\mathrm{start}}=S_{1,\mathrm{start}}=200BB,
\quad BB=100\text{ chips},\quad SB=50\text{ chips},
\quad R_0+R_1=0.
```

```math
I_t^i=\mathtt{observe}(i),\qquad
s_t^{C}=\mathtt{training\_state}(),\qquad
\pi_\theta\text{ takes only  }I_t^i\text{ as input}.
```

`I` contains the player's own cards, public board, bets, position, and mask. A separate `CriticState` contains both seats' hole cards but excludes the unrevealed future deck order. The purpose is to run the same `MultiPlayerHandEngine` in two-player mode while separating the centralized Critic's training privileges from the Actor's information set during actual play.

The core files are [src/holdem_ai/environment/hunl200_adapter.py](technical_sources/PYTHON_SOURCE_08.md#py-085), [src/holdem_ai/training/global_rollout.py](technical_sources/PYTHON_SOURCE_17.md#py-193), [src/holdem_ai/evaluation/benchmark.py](technical_sources/PYTHON_SOURCE_09.md#py-099), [scripts/phase6_benchmark.py](technical_sources/PYTHON_SOURCE_05.md#py-049), and [scripts/phase6_hardware_probe.py](technical_sources/PYTHON_SOURCE_05.md#py-050). Each adapter `reset` creates an independent 200BB hand. Policy head **4 means a pot raise**, whereas engine ID **4 means all-in**; passing the number alone would execute a different action.

| Policy head | Engine call |
|---:|---|
| 0 Fold | `engine.step(0)` |
| 1 Check / 2 Call | `engine.step(1)` |
| 3/4/5 Pot fractions | `engine.step_raise_to(target)` |
| 6 All-in | `engine.step(4)` |

```python
# src/holdem_ai/environment/hunl200_adapter.py
if not self.expose_training_state:
    raise PermissionError("pass expose_training_state=True only in the training collector")
engine = self._require_engine()
```

Observations are constructed through `reset → _receive → observe(current_player) → abstract_bet_actions → InformationState`. On `step → _receive`, the adapter checks engine termination and zero-sum rewards, then converts chip payoffs into BB. Only the training collector enables both `expose_training_state=True` and `privileged=True`.

P5 and P6 use different timelines:

```math
\tau^{P5}_{i,n}: I_{i,n}\longrightarrow I_{i,n+1},
\qquad
\tau^{P6}_{t}: s_t\longrightarrow s_{t+1},
```

P6 `collect_global_episode` stores **adjacent global decisions**, including the opponent's turns. Board dealing occurs within engine transitions rather than as a separate policy action. This distinction matters when implementing TD/GAE later, although current Base PPO uses terminal MC returns. The historical figure of approximately **2,896 random hands/s** measures CPU throughput with random policies and cannot be converted directly into training speed that includes neural networks and an optimizer.

### P7 — Observation Encoding: Turning Cards and Action History into Tensors

```math
E_A(I)=(T,V,F_A,m),\quad |F_A|=20;
\qquad
E_C(s^C)=(T_C,V_C,F_C),\quad |F_C|=24.
```

`T` contains token IDs, `V` contains per-token numerical values, `F` contains fixed-length features, and `m` is the legality mask for seven actions. The 73-token vocabulary comprises PAD and role/seat markers, 52 cards, four streets, and five action types. An amount is represented as `raise_to / BB` in the AMOUNT token's separate numerical value. Card strings follow the project standard: `sK`, `d7`, `cT`, and `hA`.

| Python path | Role |
|---|---|
| [src/holdem_ai/encoding/vocabulary.py](technical_sources/PYTHON_SOURCE_07.md#py-075) | 73 IDs and canonical card mappings |
| [src/holdem_ai/encoding/observation.py](technical_sources/PYTHON_SOURCE_07.md#py-073) | `encode_actor`, `encode_critic`, and a strict history parser |
| [src/holdem_ai/encoding/tensors.py](technical_sources/PYTHON_SOURCE_07.md#py-074) | Right padding, attention masks, and Actor/Critic batch types |
| [src/holdem_ai/encoding/actions.py](technical_sources/PYTHON_SOURCE_06.md#py-072) | One-hot actions and `[B,7]` masked categorical distributions |
| [scripts/phase7_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-051) | Verification of the encoding path for actual observations |

```python
# src/holdem_ai/encoding/observation.py
if len(ids) > max_tokens:
    raise ValueError(f'observation needs {len(ids)} tokens, exceeds max_tokens={max_tokens}; history not truncated')
return tuple(ids), tuple(values)
```

The execution flow is `InformationState → encode_actor → EncodedActor → batch_actors → ActorBatch`. The Critic uses only its separate `encode_critic → batch_critics` path. The Actor's 20 features consist of the adapter's 12 numerical values plus eight one-hot status entries covering four status types for each of two seats. The Critic's 24 features cover the BTN, acting seat, street, both stacks/contributions/bets, pot, statuses, and a terminal flag.

Illegal logits are replaced with the minimum finite value before constructing the categorical distribution. History exceeding the default 256 tokens raises an exception; past records are not automatically dropped. However, **the source P6 history does not record all actual expenditures for calls and all-ins.** The current parser also permits amounts only on `raise_to`. Preserving the history received by the encoder is distinct from complete perfect recall in poker. This encoding and seven-action head must not be interpreted as implementing the original VRPO paper's base-16 multistep wager encoding.

### P8 — Transformer Actor and Centralized Q-Critic

```math
x_j=E_{\mathrm{token}}(T_j)+E_{\mathrm{pos}}(j)
+W_v\mathrm{asinh}(V_j),
\quad z=\mathrm{LN}(\mathrm{Transformer}_{\mathrm{causal}}(x)_{L-1}).
```

```math
\pi_\theta(a\mid I)=\mathrm{MaskedSoftmax}\left(
\mathrm{MLP}_A[z,\mathrm{asinh}(F_A)]\right)_a.
```

```math
Q_{\phi,0}(s,a)=q_\phi(s,a),\quad
Q_{\phi,1}(s,a)=-q_\phi(s,a),\quad Q_\phi(s_{\mathrm{terminal}},a)=0.
```

`asinh` compresses large numerical magnitudes while preserving zero and the distinction between positive and negative values. `L−1` indexes each sample's last valid token rather than the end of its padding. The Actor accepts only `ActorBatch`. The Critic's `[B,2,7]` outputs represent both seats' returns for the same global state and the same selected action; they do not represent both players simultaneously choosing their own actions.

The actual Python implementation in [src/holdem_ai/models/transformer.py](technical_sources/PYTHON_SOURCE_16.md#py-183) provides `NetworkConfig`, `CausalTokenEncoder`, `TransformerActor`, `CentralizedQCritic`, and `chosen_q`. [src/holdem_ai/agents/neural_policy.py](technical_sources/PYTHON_SOURCE_06.md#py-065) wraps the network as a P5 policy, and [scripts/phase8_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-052) runs forward and backward passes. Defaults are embedding=64, heads=2, layers=2, FFN=128, hidden=128, dropout=0, and max_tokens=256. Historical documents report **99,975** Actor parameters and **100,487** Critic parameters.

```python
# src/holdem_ai/models/transformer.py
last = attention_mask.long().sum(dim=1) - 1
return self.output_norm(x[torch.arange(bsz, device=x.device), last])
```

```python
# src/holdem_ai/models/transformer.py
q0 = q0.masked_fill(batch.features[:, -1:].to(dtype=torch.bool), 0.0)
return torch.stack((q0, -q0), dim=1)
```

`NeuralActorPolicy.act` executes `encode_actor → batch_actors([row]) → actor.distribution → greedy argmax or Python RNG sampling` without calling the Critic. This phase supplies a policy that supports training, saving, and GUI inference, together with a training baseline. The P8 smoke test runs forward and backward passes only, without an optimizer step, and therefore cannot establish learning performance. The zero-sum Q constraint applies to this two-player, rake-free chip objective; multiplayer and prize/ICM objectives require a redesigned constraint.

### P9 — Terminal Monte-Carlo PPO Baseline

```math
G_t^{i_t}=R_{i_t},\qquad\gamma=1,
\qquad \widehat V_0(s_t,I_t)=\sum_a\pi_0(a\mid I_t)\widehat Q_{0,i_t}(s_t,a).
```

```math
A_t=G_t^{i_t}-\widehat V_0(s_t,I_t),\quad
\widetilde A_t=\frac{A_t-\mu_A}{\max(\sigma_A,10^{-6})},
\quad \rho_t(\theta)=\exp[\log\pi_\theta(a_t\mid I_t)-\log\pi_0(a_t\mid I_t)].
```

```math
J_{\mathrm{PPO}}=\frac1N\sum_t
\min\!\left(\rho_t\widetilde A_t,
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\widetilde A_t\right),
\quad \mathcal L_A=-J_{\mathrm{PPO}}-c_H\overline H(\pi_\theta).
```

```math
\mathcal L_Q=\frac1N\sum_t
\mathrm{SmoothL1}\bigl(Q_{\phi,i_t}(s_t,a_t),G_t^{i_t}\bigr).
```

`i_t` is the currently acting seat, `π₀` and `Q₀` are the policy and Critic frozen immediately before the update, and `N` is the actual number of decisions. The current Base implementation **assigns the terminal MC payoff to the acting seat at each decision in the same hand, without GAE or TD(0)**. It selects the reward using the stored `acting_seats` rather than automatically changing its sign after an opponent's turn. The baseline averages over all legal actions and therefore does not depend on the sampled action. In contrast, subtracting the selected-action Q directly as a baseline requires correction for action-dependent bias.

[src/holdem_ai/training/ppo.py](technical_sources/PYTHON_SOURCE_17.md#py-194) provides `PPOConfig`, `TrajectoryRows`, rollouts, frozen data, the clipped objective, optimizers, and initial evaluation against Random. [src/holdem_ai/training/ppo_session.py](technical_sources/PYTHON_SOURCE_17.md#py-195) handles checkpointing and resuming the models, both Adam optimizers, configuration, counters, and Torch RNG; [scripts/phase9_train.py](technical_sources/PYTHON_SOURCE_05.md#py-053) is the CLI. Defaults are LR=3e-4, clip=0.2, entropy_coef=0.01, value_coef=0.5, and grad_clip=0.5.

```python
# src/holdem_ai/training/ppo.py
v = (dist.probs * q[indexes, seats]).sum(dim=1)
old_logs.append(dist.log_prob(actions).cpu())
baseline_values.append(v.cpu())
```

```python
# src/holdem_ai/training/ppo.py
ratio = torch.exp(log_prob - old_log_prob)
return torch.minimum(ratio * advantage,
                     torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage)
```

Execution follows `collect_self_play → freeze_old_policy_data → optimize_ppo → save_checkpoint → evaluate_vs_random`. The Actor optimizer backpropagates the policy-plus-entropy loss; the Critic optimizer backpropagates `value_coef × SmoothL1`. Both apply gradient clipping and finite-value checks. **A Q regression target exists only for the action actually selected.** A single hand does not provide counterfactual ground truth for the other six actions.

The intermediate `approx_kl` is `mean(ρ-1-logρ)`. Although it is computed after an optimizer step, the `new_log` used in that calculation comes from **the forward pass before that step**. It therefore does not measure drift obtained by recomputing the new policy distribution after the complete update. P16 adds separate post-update diagnostics to address this limitation. Empirical advantage centering and standardization also depend on a finite sample; the unbiasedness claim for an ideal action-independent baseline should not be extended unchanged to the entire practical PPO update. PPO clipping and multiple epochs optimize a surrogate objective and do not guarantee convergence to a Nash equilibrium.

### P10 — Q-Boost: An Action-Dependent Control Variate and Analytic Correction

```math
A_t^{MC}=G_t-V_0,
\quad \delta_t^{(\beta)}=G_t-V_0-\beta(Q_0(s_t,a_t)-V_0),
\quad \beta\in[0,1].
```

```math
J_{QB}(\theta)=\frac1N\sum_t\left[
\min\!\left(\rho_t\frac{\delta_t^{(\beta)}-\mu_A}{\sigma_A},
\mathrm{clip}(\rho_t)\frac{\delta_t^{(\beta)}-\mu_A}{\sigma_A}\right)
+\frac\beta{\sigma_A}\sum_a(\pi_\theta(a\mid I_t)-\pi_0(a\mid I_t))Q_0(s_t,a)
\right].
```

Here, `clip(ρ)` uses the same `[1−ε,1+ε]` interval as P9; `μ_A,σ_A` are **shared values computed once from the unboosted MC advantage**. Standardizing the residual and analytic term separately breaks the cancellation underlying the correction. β is fixed before the rollout; `β=0` recovers P9 PPO on the same frozen data.

The purpose is to **measure** whether predicted Q can reduce Monte-Carlo policy-gradient variability. Using only `G−Q(a)` would remove the action contribution explained by Q itself, so the following analytic correction is required. At a fixed state and the old-policy point, with advantage-normalization constants held fixed, the ideal expected gradient satisfies:

```math
\left.\mathbb E_{a\sim\pi_0}\left[
\nabla_\theta\log\pi_\theta(a\mid I)
\{G-V_0-\beta(Q_0(a)-V_0)\}\right]\right|_{\theta_0}
+\left.\beta\sum_a\nabla_\theta\pi_\theta(a\mid I)Q_0(a)\right|_{\theta_0}
=\left.\sum_a\nabla_\theta\pi_\theta(a\mid I)Q^{\pi_0}(s,a)\right|_{\theta_0}.
```

This identity explains the control-variate principle, which holds even when Q is inaccurate. It does not establish unbiasedness at arbitrary θ after practical PPO clipping or fully guarantee unbiasedness when finite-batch standardization is included.

The core functions in [src/holdem_ai/training/qboost.py](technical_sources/PYTHON_SOURCE_17.md#py-196) are `freeze_qboost_data`, `qboost_surrogate`, `qboost_diagnostics`, and `optimize_qboost`. [src/holdem_ai/training/qboost_session.py](technical_sources/PYTHON_SOURCE_17.md#py-197) handles Phase9 warm-starting and Phase10 checkpoints, while [scripts/phase10_train.py](technical_sources/PYTHON_SOURCE_03.md#py-027) is the matched-start comparison CLI.

```python
# src/holdem_ai/training/qboost.py
observed_q = q_values.gather(1, actions[:, None]).squeeze(1)
residual = (returns - baseline - beta * (observed_q - baseline) - center) / scale
mc_clip = clipped_surrogate(new_log, old_log, residual, clip_epsilon)
analytic = beta * ((new_probs - old_probs) * q_values).sum(-1) / scale
return mc_clip + analytic
```

The Actor loss is `−J_QB−c_HH`, and the Critic loss is unchanged from P9. Old probabilities and all Q values remain fixed throughout the optimizer epochs. Reusing a diverged PPO arm's rollouts for QBoost would invalidate the on-policy comparison; each arm therefore collects fresh data with its own latest policy thereafter.

Diagnostics distinguish residual variance from a **logit-gradient proxy**:

```math
g_t^{PPO}=(e_{a_t}-\pi_0)\widetilde A_t,\quad
g_t^{QB}=(e_{a_t}-\pi_0)\widetilde\delta_t
+\frac\beta{\sigma_A}\pi_0\odot(Q_0-V_0),
\quad \rho_{var}=\frac{\sum_j\widehat{\mathrm{Var}}(g_{t,j}^{QB})}
{\sum_j\widehat{\mathrm{Var}}(g_{t,j}^{PPO})}.
```

`ρ_var<1` indicates reduced variance in the logit proxy for that sample; it does not establish reduced variance in the full Transformer's parameter gradients or improved strategic EV. Historical documents record an initial ratio of approximately 1. This is an independently constructed VRPO-inspired Lite experiment, not a full reproduction of the original paper.

### P11 — Batched Self-Play and Training Under a Time Budget

```math
X_t=\mathrm{Batch}\{E_A(I_{e,t}^{i_e}):e\in\mathcal E_t\},
\quad D_k\sim\pi_{\theta_k},
\quad \theta_{k+1}=\mathrm{Update}(\theta_k,D_k).
```

`𝓔_t` is the set of independent hands still active in one process. Each forward batch advances each unfinished hand by **one decision**, then returns payoffs after all hands finish. The purpose is to amortize per-decision GPU-call costs across multiple observations while retaining the same on-policy collection contract. This is neither a multiprocess GPU inference server nor replay-based training.

[src/holdem_ai/training/batched_collector.py](technical_sources/PYTHON_SOURCE_16.md#py-191) provides `collect_batched_self_play`; [src/holdem_ai/experiments/time_budget.py](technical_sources/PYTHON_SOURCE_15.md#py-173) handles session creation, warm-starting, the loop, evaluation, and atomic saving; [src/holdem_ai/utils/progress.py](technical_sources/PYTHON_SOURCE_17.md#py-202) handles heartbeats and ETA. [scripts/phase11_train.py](technical_sources/PYTHON_SOURCE_03.md#py-030), `phase11_benchmark.py`, and `phase11_compare.py` provide training, inference-throughput measurement, and comparison of independent logs, respectively.

```python
# src/holdem_ai/training/batched_collector.py
with torch.inference_mode():
    batch = batch_actors(actor_rows, device=device)
    probabilities = actor.distribution(batch).probs.detach().cpu().tolist()
```

The environment and action RNGs use the per-hand seed `seed=base_seed+iteration×H+hand_index`; the BTN alternates according to the global hand index. Privileged state is stored in separate Critic rows and excluded from the Actor batch. Every decision's return is the terminal reward for its seat. `dropout=0` is enforced to preserve reproducibility between the collection policy and frozen log probabilities.

The flow is `run_experiment → collect_batched_self_play → freeze → existing PPO/QBoost optimization → atomic checkpoint → append metrics → optional evaluation`. `run.json` records the initial configuration, `anchor.pt` contains the frozen starting Actor, `metrics.jsonl` contains append-only events, and `summary.json` summarizes the most recent invocation. A partial update interrupted by Ctrl+C is restored to the last completed checkpoint; a crash between saving the checkpoint and appending the log is explained by a resume event. The time limit and `max_updates` provide **a new budget/additional update count for each invocation**, with stopping at a boundary after the current update or evaluation finishes.

The initial GPU pilot's **30–45 hands/s** is a historical observation with a different workload from P6's CPU random-policy speed. `PHASE11_PROGRESS` records **304 passed** in the user's Windows/CUDA environment. Progress heartbeats only refresh the display; they do not change training or RNG state. Repeated identical greedy evaluations against the frozen anchor do not alone establish a defect: small weight changes can leave the argmax unchanged. P12 is designed to investigate this distinction.

### P12 — Seed-Blocked Evaluation: Treating the Deal as the Statistical Unit

```math
X_k^A=\frac14\sum_{b=0}^{1}\sum_{s=0}^{1}R_A(k,b,s),
\quad \widehat J_A=100\bar X^A,
\quad \mathrm{SE}(\widehat J_A)=100\frac{s_X}{\sqrt K}.
```

```math
CI_{95\%}\approx\widehat J_A\pm1.96\mathrm{SE}(\widehat J_A),
\quad D_k=X_k^A-X_k^C,
\quad \widehat{\Delta J}=100\bar D.
```

`k` is the shuffled-deck seed, `b` the BTN seat, and `s` the candidate's physical seat. Four layouts are played with the same seed, so the independent unit is **one seed block**, not four individual hands. Comparisons with a control calculate a CI from the variance of `D_k` using the same opponent, configuration, and seeds. They do not subtract endpoints of separate confidence intervals.

The core components are `EvaluationConfig`, `DealGroup`, `evaluate_seed_blocks`, `summarize_block_scores`, and `compare_results` in [src/holdem_ai/evaluation/seed_blocked.py](technical_sources/PYTHON_SOURCE_10.md#py-109), together with [src/holdem_ai/evaluation/checkpoint_loader.py](technical_sources/PYTHON_SOURCE_09.md#py-100) and [scripts/phase12_evaluate.py](technical_sources/PYTHON_SOURCE_03.md#py-031).

```python
# src/holdem_ai/evaluation/seed_blocked.py
if candidate.opponent_id != control.opponent_id:
    raise ValueError('comparison requires exactly the same opponent')
if tuple(g.seed for g in candidate.blocks) != tuple(g.seed for g in control.blocks):
    raise ValueError('comparison requires the same shuffled-deck seeds')
deltas = tuple(a - b for a, b in zip(candidate.means_bb, control.means_bb))
```

Execution follows `load_actor → evaluate_seed_blocks → 4 layouts/seed → group summary → JSON/CSV`. It records `greedy` or reproducible `sampled` play, validation/test seed namespaces, action counts, and model fingerprints. Evaluation uses neither an optimizer nor `training_state`. A symmetry sanity check evaluates separate instances of the same weighted policy with seat labels swapped; their seed-group payoff is exactly zero.

Even with the same deck seed, different policies can follow different betting paths and terminate at different times. This is a seat-balanced experiment with matched seeds; it does not generate counterfactual outcomes for unplayed actions. Limitations include normal-approximation CIs at small K, selection bias from repeated validation, and the absence of strong opponents. Results against Random or a frozen Actor do not measure exploitability.

### P13 — Batch Scaling: Training Throughput and Action-Conditional Diagnostics

```math
N_{\mathrm{mini},k}=E\left\lceil\frac{N_k}{M}\right\rceil,
\quad \widehat T_{\mathrm{train}}=\frac{\sum_kH_k}{\sum_k\Delta t_{\mathrm{update},k}},
\quad \widehat p(a\mid legal)=\frac{C_a}{L_a}.
```

`H` is hands/update, `B` the number of active environments, `M` decisions/minibatch, `E` the epoch count, and `N_k` the actual number of decisions, which varies with folds and raises. `C_a` is the action-selection count, and `L_a` is the count of legal opportunities; the ratio is `null` when `L_a=0`. This is a conditional frequency across visited states, not `π(a|I)` at a particular infoset.

```math
H_{norm}(I)=\begin{cases}
H(\pi(\cdot|I))/\log|\mathcal A(I)|,&|\mathcal A(I)|>1,\\0,&|\mathcal A(I)|=1.
\end{cases}
```

Building on P11's inference benchmark, this phase measures **update throughput including collection, diagnostics, and optimizer work** and reduces denominator errors in fold/check frequencies. Its components are [src/holdem_ai/experiments/batch_scaling.py](technical_sources/PYTHON_SOURCE_14.md#py-166), [src/holdem_ai/evaluation/action_diagnostics.py](technical_sources/PYTHON_SOURCE_09.md#py-097), `reward_diagnostics.py`, [scripts/phase13_train.py](technical_sources/PYTHON_SOURCE_03.md#py-034), `phase13_benchmark.py`, and `phase13_compare.py`.

```python
# src/holdem_ai/experiments/batch_scaling.py
rows = collect_batched_self_play(base.actor, base.ppo_config,
                                 iteration=iteration, device=base.device,
                                 env_batch_size=config.env_batch_size)
diagnostics = action_diagnostics(rows, base.actor, device=base.device,
                                 chunk_size=base.ppo_config.inference_batch_size)
reward_diagnostics = summarize_rollout_rewards(rows)
```

PPO/QBoost equations remain unchanged. Warm-starting preserves the source and imports the Actor, Critic, and Adam states into a new stage; resuming reads the saved configuration of that stage. The default new seed is `old_seed+1,000,003`, with an explicit override available. The checkpoint payload remains `phase=9/10`, while run metadata uses `phase=13`, preserving compatibility with the P12 loader.

The historical GPU result for `H=64,B=32` was **91.77 training hands/s**, but it is an initial hypothesis based on two measurements. Changing H also changes the `seed+iteration×H` schedule and gradient noise, so the difference cannot be attributed solely to GPU batching efficiency. Evaluation and checkpoint I/O outside collection and optimization are excluded from the throughput denominator; this measure is distinct from total wall-clock time.

### P14 — Controlled PPO/QBoost Study: Separating Algorithm Effects from Seed Variation

```math
\theta_{s,0}^{PPO}=\theta_{s,0}^{QB}=\theta_{source},\quad
\phi_{s,0}^{PPO}=\phi_{s,0}^{QB}=\phi_{source},\quad
Adam_{s,0}^{PPO}=Adam_{s,0}^{QB}=Adam_{source}.
```

```math
\Delta_s=\frac{100}{K}\sum_k(X_{s,QB,k}-X_{s,PPO,k}),
\quad\bar\Delta=\frac1S\sum_s\Delta_s,
\quad CI_{95\%}\approx\bar\Delta\pm t_{S-1,0.975}\frac{s_\Delta}{\sqrt S}.
```

Training seed `s` is the replicate unit. Both arms start from **the same complete Phase9 checkpoint and Adam states**, and each collects fresh rollouts with its latest policy. A shared shuffle schedule does not imply identical action paths. The purpose is to control the starting point, budget, seeds, and selection criteria without changing the equations or optimizers themselves.

[src/holdem_ai/experiments/controlled_study.py](technical_sources/PYTHON_SOURCE_14.md#py-167) verifies the source hash, creates or resumes arms, selects validation-best checkpoints, runs tests only after all arms finish, and summarizes across seeds. [src/holdem_ai/evaluation/seed_statistics.py](technical_sources/PYTHON_SOURCE_10.md#py-110) provides seed-level statistics, and [scripts/phase14_study.py](technical_sources/PYTHON_SOURCE_03.md#py-035) is the entry point.

```python
# src/holdem_ai/experiments/controlled_study.py
def _rollout_seed(seed: int) -> int:
    return 100_000_000 + seed * 1_000_000
```

The flow is `source hash + study manifest → PPO/QB arms per seed → repeated updates/validation → best.pt → check completion of all arms → final test → paired deck/seed summary`. Validation selects the highest score against the frozen starting Actor and includes that starting Actor among the candidates. Resume is rejected if the source hash or scientific settings change. A run that remains partial because it reaches `arm-duration` does not execute the final test.

The historical P15 document records P14 PPO validation peaks of **809.50** for seed51, **858.125** for seed52, and **456.75 bb/100** for seed53. These values led to selecting the seed52 checkpoint as the subsequent starting point. Three training seeds did not establish QBoost superiority, and peak validation values are not independent test results. With a shared deal bank, seed-level CIs describe training variability conditional on fixed test deals; they do not jointly estimate all uncertainty from deals and training.

### P15-A — Long-Run Training: Checkpoint Selection and Sealing Final Evaluation

```math
J_A(t)=\widehat J_{val}(\pi_t,\pi_0),\quad
J_R(t)=\widehat J_{val}(\pi_t,Random),
\quad
S_t=\frac{[J_A(t)-J_A(0)]+[J_R(t)-J_R(0)]}{2}.
```

```math
t^*=\arg\max_{t\in\mathcal T_{evaluated}}S_t,
\qquad S_0=0.
```

`π₀` is the frozen Actor at the stage's start. `latest.pt` is the source for resuming, whereas `best.pt` is the result of repeated validation-based selection. The goal is logging, interruption, recovery, selection, and independent final evaluation for long training runs, rather than an algorithmic improvement.

The core components are `run_longrun`, `finalize_longrun`, `_exclusive_lock`, and `_validation` in [src/holdem_ai/experiments/longrun.py](technical_sources/PYTHON_SOURCE_14.md#py-168), together with [scripts/phase15_train.py](technical_sources/PYTHON_SOURCE_04.md#py-042) and `phase15_finalize.py`. They verify source/best SHAs and configuration, and use an OS lock to prevent simultaneous writes to the same run. The training loop reuses P13's `_one_update`. Finalization evaluates only the selected best checkpoint after training has ended, using separate final seeds, and marks the run `sealed` to prevent further resuming.

```python
# src/holdem_ai/experiments/longrun.py
def run_longrun(cfg: LongRunConfig, *, emit: Callable[[dict],None] | None = None) -> dict:
    """OS-locked entry point for the crash-safe long-running training loop."""
    with _exclusive_lock(Path(cfg.root)):
        return _run_longrun_impl(cfg,emit=emit)
```

The original safety rule stopped a run after 12 consecutive updates satisfying `entropy<0.45 or approx_kl>0.10`. A subsequent diagnostic patch in [src/holdem_ai/experiments/ppo_safety.py](technical_sources/PYTHON_SOURCE_15.md#py-171) separates warnings from hard stops:

```math
w_{stop,t}=w_{KL,t}\lor(\mathtt{entropy\_stop\_enabled}\land w_{H,t}).
```

```python
# src/holdem_ai/experiments/ppo_safety.py
low_entropy = entropy < entropy_threshold
high_kl = approx_kl > max_approx_kl
safety_violation = high_kl or (low_entropy and entropy_stop_enabled)
```

New runs record low entropy as a warning by default; resuming an old manifest without the field retains its original entropy-stop semantics. This change does not prove that collapse has been resolved; it reduces premature termination based on an insufficient metric. Entropy and KL warnings are both recorded within the same update, so neither hides the other.

**Observations in the historical experiment report dated 2026-10-13**:

| Experiment | New updates / hands | Actual termination | Best validation score |
|---|---:|---|---:|
| Initial `H64/M64/E2` | 197 / 12,608 | Update301, 191.406 seconds, safety_stop | 0, source Update104 |
| Stability pilot `H128/M128/E1` | 100 / 12,800 | Update204, 185.598 seconds, update_limit | +98.25, Update125 |
| Resume of that pilot | Additional 246 / 31,488 | Update450, 363.843 seconds, safety_stop | +413.375, Update375 |

The stability run accumulated 346 new updates and 44,288 hands. Between Update375 and Update450, fold/all-action counts in evaluation against the frozen initial Actor changed from **82/828=9.90% to 314/565=55.58%**; against Random, they changed from **76/698=10.89% to 268/493=54.36%**. These denominators are not fold-legal opportunities, and the ratios do not establish the cause of collapse. Update450's selection score was **−520 bb/100**, and final evaluation was not sealed. Since `H,M,E` changed together, the causal effect of an individual factor could not be isolated. The reported CUDA peak of 102–161MiB measures PyTorch tensor allocation rather than total GPU VRAM.

### P15-B — Policy/Critic/Reward/Preflop Audits: Separating Distinct Questions

```math
\overline\pi_m(a\mid legal;D)=
\frac{\sum_{I\in D}\mathbf1[a\in\mathcal A(I)]\pi_m(a\mid I)}
{\sum_{I\in D}\mathbf1[a\in\mathcal A(I)]}.
```

Computing each model's probabilities on the same observation set `D` holds the visitation distribution fixed, allowing changes in the policy itself to be examined. `compare_actors_on_states` in [src/holdem_ai/evaluation/fixed_state_policy_audit.py](technical_sources/PYTHON_SOURCE_09.md#py-103) and [scripts/phase15_policy_audit.py](technical_sources/PYTHON_SOURCE_04.md#py-039) perform this audit. Historical documents record a **+0.343803** change in fixed-state mean fold probability from Update375 to Update450, concentrated primarily in the preflop BB position. If the reference rollout distribution is itself dominated by preflop states, conclusions about postflop generalization are limited.

```python
# src/holdem_ai/evaluation/fixed_state_policy_audit.py
if bool(legal[i, 0]):
    bucket["fold_legal"] += 1
    bucket["fold_prob_sum"] += float(probs[i, 0])
```

The Critic audit compares taken-action Q and policy-weighted V separately:

```math
p_t^Q=Q_\phi(s_t,i_t,a_t),\quad
p_t^V=\sum_a\pi_{behavior}(a|I_t)Q_\phi(s_t,i_t,a),\quad e_t=p_t-G_t.
```

```math
Bias=\overline e,\quad MAE=\overline{|e|},\quad
RMSE=\sqrt{\overline{e^2}},\quad
EV=1-\frac{\mathrm{Var}(e)}{\mathrm{Var}(G)}.
```

`diagnose_critic_on_rows` and `run_critic_audit` in [src/holdem_ai/evaluation/critic_diagnostics.py](technical_sources/PYTHON_SOURCE_09.md#py-102), together with [scripts/phase15_critic_audit.py](technical_sources/PYTHON_SOURCE_03.md#py-036), produce metrics overall and by street, seat, BTN/BB, and action. EV is null when reward variance is zero. Even EV=1 can coexist with a large constant bias, so it must be read alongside RMSE and Bias.

```python
# src/holdem_ai/evaluation/critic_diagnostics.py
q_taken = chosen_q(q, seats, actions)
selected = q[torch.arange(stop - start, device=device), seats, :]
baseline = (selected * dist.probs).sum(dim=1)
```

`run_critic_audit` generates both a controlled probe that passes one reference rollout to two Critics and separate on-policy rollouts generated by each Critic's own Actor. A reference terminal return is not an unbiased label for another candidate's continuation Q, so the controlled probe is an auxiliary diagnostic of cross-policy generalization. Actual calibration should prioritize the candidate's own on-policy results. Repeated decision rows within a hand are correlated; the audit does not provide CIs that assume those rows are iid. Historical observations included reward standard deviations of approximately **193–196BB**, Q predictions near zero, and EV≈0, motivating the long-training hypothesis.

Reward and all-in audits use different denominators:

```math
D_{hand}=\{R_{0,h}\}_{h=1}^{H},\qquad
D_{decision}=\{R_{i_{h,t},h}\}_{h,t},
```

```math
p_{hand,allin}=\frac{\#\{h:\exists t,street_t=PF,a_t=6\}}{H},
\quad p_{decision,allin|legal}=\frac{\#\{t:street_t=PF,a_t=6\}}
{\#\{t:street_t=PF,6\in\mathcal A(I_t)\}}.
```

| Python + CLI | Audit target / actual flow |
|---|---|
| [src/holdem_ai/evaluation/terminal_reward_audit.py](technical_sources/PYTHON_SOURCE_10.md#py-113) + [scripts/phase15_terminal_reward_audit.py](technical_sources/PYTHON_SOURCE_04.md#py-041) | Frozen rollout → hand-boundary validation → hand/decision reward distributions, 190/199/200BB tails, and last-street/fold classifications |
| [src/holdem_ai/evaluation/initial_policy_comparison.py](technical_sources/PYTHON_SOURCE_09.md#py-104) + [scripts/phase15_initial_policy_comparison.py](technical_sources/PYTHON_SOURCE_04.md#py-038) | Separate self-play by Uniform Legal, fresh random initialization, and trained Phase14, plus probabilities on shared fixed states |
| [src/holdem_ai/evaluation/preflop_sequence_audit.py](technical_sources/PYTHON_SOURCE_09.md#py-107) + [scripts/phase15_preflop_sequence_audit.py](technical_sources/PYTHON_SOURCE_04.md#py-040) | Ordered actions per hand → complete preflop sequences, prefixes ending at the first all-in, and rates conditional on depth/position/to-call |
| [src/holdem_ai/evaluation/reward_diagnostics.py](technical_sources/PYTHON_SOURCE_09.md#py-108) | Per-update seat0 payoffs, fold terminations, and decision counts; showdown is null without an authoritative field |

```python
# src/holdem_ai/evaluation/terminal_reward_audit.py
last_street = STREETS[streets[-1]]
fold = last_action == 0
preflop_all_in = any(a == 6 and st == 0 for a, st in zip(actions, streets))
any_all_in = 6 in actions
```

A regular Call can also exhaust a stack, so counting only head6 understates actual all-in commitments. A non-fold termination whose last decision is preflop may reach showdown after an automatic river runout. The observation that 93% of decisions were preflop does not imply that 93% of hands ended with a preflop fold. Head sequences are also not equivalent to exact wager amounts or formal 3-bet/4-bet classifications.

Preserved figures from the earlier 512-hand comparison:

| Metric | Uniform Legal | Synthetic random-init | Trained Phase14 |
|---|---:|---:|---:|
| Preflop decisions | 1,397 | 1,335 | 2,267 |
| Explicit preflop all-ins | 218 | 218 | 438 |
| All-in legal opportunities | 1,179 | 1,117 | 1,829 |
| Preflop terminal fold | 318 | 353 | 53 |
| Preflop terminal call | 97 | 84 | 433 |
| Pot raise / double-pot raise | 193 / 175 | 176 / 224 | 523 / 545 |

The trained sample therefore simultaneously had a hand-level all-in rate of `438/512≈85.55%` and a legal-decision rate of `438/1829≈23.95%`. A separate sequence audit with independent seeds recorded **439/512=85.74%** and **115 occurrences** of the representative prefix `Pot → Double Pot → Double Pot → All-in`; these are explicitly different samples. Synthetic random initialization is not guaranteed to match the actual first checkpoint of Phase14 training. This audit is read-only: it does not train, modify, or finalize a checkpoint. Its value is in distinguishing abstraction branching, repeated opportunities, and Critic calibration before introducing a simple all-in penalty.

### P16-A — Fixed Opponent Pool and Strategy Diagnostics

```math
\mathcal O=\{Random,CallingStation,Tight,PotPressure,FrozenActors\},
\quad J_{pool}=\frac1{|\mathcal O|}\sum_o\widehat J_o,
\quad J_{worst,observed}=\min_o\widehat J_o.
```

```math
SPR(I)=\frac{\min(S_{hero},S_{opp})}{\max(P,10^{-9})}.
```

The purpose is to observe whether a policy that beats one opponent also withstands other styles, including excessive calling, folding, or aggression. The final quantity is the lowest observed mean within the pool, not a minimax value or exploitability.

[src/holdem_ai/agents/diagnostic_opponents.py](technical_sources/PYTHON_SOURCE_06.md#py-064) provides rule-based policies with legal fallbacks; `evaluation/opponent_pool.py` runs four-layout seed-block matches against each opponent; `evaluation/strategy_diagnostics.py` and `strategy_report.py` handle street, position, SPR, entropy, and JSON/CSV/Markdown output; [scripts/phase16_evaluate.py](technical_sources/PYTHON_SOURCE_04.md#py-043) is the frozen-checkpoint CLI. SPR buckets are `<1`, `[1,4)`, `[4,10)`, and `≥10`.

```python
# src/holdem_ai/evaluation/opponent_pool.py
cfg = config or PoolEvaluationConfig()
if not opponents or len({x.name for x in opponents}) != len(opponents):
    raise ValueError('provide a nonempty pool with unique opponent names')
```

CallingStation emphasizes check/call, Tight check/fold, and PotPressure pot raises; all are **diagnostic opponents**. Evaluation does not update the Actor or optimizer and does not pass a CriticState to the policy. Opponent-specific CIs use seed blocks, with no multiple-comparison correction. The current pool evaluator alone does not automatically provide a paired CI for two PPO/QBoost candidates; a separate comparison using common deal-group results is required.

### P16-B — A0–A3 Ablation and Exact Post-Update KL

```math
\widehat D_{KL,post}=\frac1N\sum_{I\in D_{old}}
\sum_{a\in\mathcal A(I)}\pi_{old}(a|I)
\log\frac{\pi_{old}(a|I)}{\pi_{new}(a|I)}.
```

This is exact categorical KL in the sense that it explicitly sums over all action categories. It is an **average over the frozen rollout states**, not KL over all possible poker infosets. Unlike P9's sampled `approx_kl`, it performs another Actor forward pass after optimization finishes.

| Arm | Mathematical / implementation change | Question examined |
|---|---|---|
| A0 | Original PPO + centralized Q + BB Huber | Control |
| A1 | `G′=G/200`, `Q′=Q/200`; divide the final Q-head weights/bias by 200 | Target scale |
| A2 | Direct V head with `V(s,0)=v(s), V(s,1)=−v(s)` | Effect of averaging unsupervised alternative-action Q values in the baseline |
| A3 | A0 + rollback of the Actor, Critic, and both Adam optimizers when post KL>0.02 | Large policy drift in a single update |

[src/holdem_ai/experiments/ppo_ablation.py](technical_sources/PYTHON_SOURCE_15.md#py-169) provides `DirectVCritic`, `scale_q_head_`, `freeze_variant`, `optimize_variant`, `snapshot_probs`, `exact_policy_kl`, `critic_on_frozen_rows`, and `run_ablation`; [scripts/phase16_ppo_ablation.py](technical_sources/PYTHON_SOURCE_04.md#py-044) runs the experiment.

```python
# src/holdem_ai/experiments/ppo_ablation.py
local = torch.where(
    legal & (p > 0),
    p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log()),
    torch.zeros_like(p),
).sum(dim=1)
```

All arms retain the same source Actor and Actor Adam state, but **reset the Critic Adam state**. A0 therefore uses the original PPO optimizer mathematically, yet is not a bit-for-bit resume of the earlier Phase15 run. A2 also changes architecture and must not be read as a simple LR ablation. The pilot's `final_actor.pt` is an Actor-only export without Critic or Adam state, so it is not a training-resume checkpoint.

The flow is `source → create variant → fresh Critic Adam → own-policy rows → old freeze/probabilities → optimize → post KL → optional full rollback → holdout Critic evaluation + paired validation`. The independent 10-update pilot document records no conclusive improvement from A1/A2. Implementing a gate is distinct from evidence that A3 actually improves learning.

### P16-C — Long A0/A3 Stress Runs and Full-State Recovery

```math
H=128,\quad U_{attempt}=100,
\quad N_{train,arm}=12,800,
\quad N_{train,total}=25,600.
```

[src/holdem_ai/experiments/ppo_stress.py](technical_sources/PYTHON_SOURCE_15.md#py-172) retains only A0/A3 and implements `run_stress`, `_one_update`, full checkpoints, regeneration of log mirrors, and `compare_stress_runs`. The campaign in [scripts/phase16_ppo_stress.py](technical_sources/PYTHON_SOURCE_04.md#py-045) runs both arms sequentially and compares aligned validation results.

```python
# src/holdem_ai/experiments/ppo_stress.py
session.actor.load_state_dict(actor_state)
session.critic.load_state_dict(critic_state)
session.actor_optimizer.load_state_dict(actor_opt_state)
session.critic_optimizer.load_state_dict(critic_opt_state)
```

An update rejected by A3 still consumes its freshly collected card-seed block and **increments the attempted-update counter**. Reversing that counter would repeatedly collect the same seeds and distort the budget and comparison. Accepted and rejected updates are recorded separately, and `latest.pt` is authoritative for the ledger, models, Adam states, and RNG. Metrics/progress JSON can be reconstructed from that checkpoint; periodic snapshots remain unchanged. Manifest/source SHAs and a lock prevent concurrent writes and invalid resumes.

The historical P17 document subsequently records the 100-update stress result as **A3 rejection=0, with A3−A0=0 bb/100 against both frozen Phase14 and Random**. In that run, the gate never activated and therefore did not separate the arms; Critic EV remained near zero. Successfully executing 100 attempts does not establish long-term strategic stability or GTO play.

### P17-Eval — Continuous Stacks: Inference Robustness to Variable Stacks

```math
S_{0,1}+S_{1,1}=400BB,\quad
S_{i,h+1}^{start}=S_{i,h}^{final},\quad
\min_iS_{i,h}^{final}=0\Rightarrow MatchOver.
```

```math
(S_{candidate},S_{opp})=(s,400-s)BB,
\quad s\in\{50,100,200,300,350\}.
```

This phase extends the stack carryover semantics of GUI1.6 to **offline evaluation**. It does not modify the existing 200BB-reset training or train a variable-stack agent. Even when a match starts at 200/200, stacks carry forward to the next hand, distinguishing it from P16's independent-hand evaluation.

The core components are `_play_match` and `evaluate_continuous_stacks` in [src/holdem_ai/evaluation/continuous_stack.py](technical_sources/PYTHON_SOURCE_09.md#py-101), `reset_with_stacks` in [src/holdem_ai/game_service/interactive_adapter.py](technical_sources/PYTHON_SOURCE_15.md#py-178), and [scripts/phase17_evaluate.py](technical_sources/PYTHON_SOURCE_04.md#py-046). Within each match, the button alternates and play continues without rebuy until bankruptcy or the maximum hand count. Deck and action RNGs are separate; a hand left unfinished at the decision limit raises an error rather than receiving an arbitrary payoff.

```python
# src/holdem_ai/evaluation/continuous_stack.py
transition = adapter.reset_with_stacks(stacks, seed=rng_deck.getrandbits(64))
```

```python
# src/holdem_ai/evaluation/continuous_stack.py
stacks = updated
hands += 1
if len(result.board) == 5 and not any(result.folded):
    showdowns += 1
```

Different match lengths are handled within each group by dividing by the number of hands actually completed:

```math
b_g=100\frac{\sum_{\ell=1}^4\Delta S_{g,\ell}^{BB}}
{\sum_{\ell=1}^4H_{g,\ell}},\quad
\widehat b=\frac1G\sum_gb_g,\quad
CI_{95\%}\approx\widehat b\pm1.96\frac{s_b}{\sqrt G}.
```

This average gives each seed group equal weight and is generally not the same as pooling every group's net payoff and hand count into a single ratio. With a control, a paired CI is computed from `b_candidate−b_control` for the same seed. Additional outputs include bust/profit fractions, mean net BB/match, hands/match, hand wins/losses/ties, authoritative showdown counts, and legal-conditioned actions by **current uncommitted effective-stack** bucket.

The historical GUI16/P17 pilot used **Phase14 seed41 QBoost, CallingStation, sampled play, 10 groups×4 matches per scenario, and at most 20 hands/match**:

| Candidate/opponent starting BB | Completed hands | bb/100 | Approximate 95% CI | Candidate bust |
|---|---:|---:|---|---:|
| 50/350 | 151 | −621.854 | [−1554.417,+310.709] | 90.0% |
| 100/300 | 161 | −363.636 | [−1076.364,+349.091] | 77.5% |
| 200/200 | 109 | +318.182 | [−961.519,+1597.882] | 50.0% |
| 300/100 | 131 | −29.060 | [−2018.023,+1959.904] | 27.5% |
| 350/50 | 173 | −47.351 | [−1358.976,+1264.274] | 15.0% |

Every CI includes zero, so neither stack-specific profitability nor superiority was established. Starting bankroll differences directly affect bust rates, preventing their use as a standalone measure of policy strength. The contemporaneous Windows integration result of **690 passed, 1 xfailed, 1 warning** is likewise a historical regression-test record.

### P17-Retrain — Monitored GPU PPO: Cumulative Active Time and an Update Ceiling

```math
t_{active}=\sum_jt_{invocation,j},\quad t_{target}=18,000s,
\quad U_{attempt}=U_{accepted}+U_{rejected}.
```

This Phase17 is **separate from the evaluation above**. `FiveHourConfig`, `run_five_hour`, and checkpoint/ledger/log mirrors in [src/holdem_ai/experiments/ppo_five_hour.py](technical_sources/PYTHON_SOURCE_15.md#py-170), together with [scripts/phase17_gpu_retrain.py](technical_sources/PYTHON_SOURCE_04.md#py-047), manage a cumulative five-hour target. They reuse P16 stress's original PPO/centralized Q and post-KL diagnostics. Training starts from Phase14 **Update104**, retaining the source Actor and Actor Adam state while resetting Critic Adam.

Unlike the per-invocation durations in P11/P15, this phase counts **cumulative active execution time** across multiple resumes. Human downtime is excluded, while initial/intermediate evaluation and checkpoint I/O are included. State is saved at completed-update boundaries, and the final evaluation may slightly overshoot the target. Resume verifies source/manifest SHAs and all scientific settings.

```python
# src/holdem_ai/experiments/ppo_five_hour.py
if manifest.get("settings") != _settings(cfg):
    raise ValueError("scientific settings differ from saved run")
if manifest.get("source_sha256") != _sha(source):
    raise ValueError("original Phase14 source checkpoint changed")
```

At post KL>0.10, A0 preserves the already executed update and triggers an emergency halt; the presence of `hard_stop.json` prevents automatic resuming. A3 is an optional experiment that performs full rollback above 0.02. `latest.pt` stores the Actor, Q, both Adam states, CPU/CUDA RNGs, update accounting, time, manifest hash, metrics, and validation ledger. `best.pt` is an exploratory checkpoint selected for a higher mean repeated paired change against two frozen opponents.

**User `summary.json` preserved in the earlier integrated report dated 2026-10-13**:

| Field | Historical record |
|---|---:|
| Variant / protocol | A0 / `phase17_five_hour_ppo_v1` |
| Attempted / accepted / rejected | 5,000 / 5,000 / 0 |
| Global update | 5,104 |
| Training hands | 640,000 |
| Active seconds / target seconds | 12,743.274 / 18,000 |
| Validation count | 200 |
| Best global update | 2,104 |
| Best validation selection score | 1,450.4687 bb/100 |
| Stop / complete | `max_updates_ceiling` / `false` |

```math
5,000\times128=640,000\ \mathrm{hands},\quad
12,743.274s\approx3h32m23s,\quad
\frac{12,743.274}{18,000}\approx70.8\%.
```

The run reached its 5,000-update ceiling but did not complete the five-hour active-time budget. `best.pt` is Update2104, whereas `latest.pt` is Update5104; they serve different purposes. The highest validation score was selected through repeated use of the same deal bank and is not independent test EV. Collecting more hands alone cannot establish that P15's preflop all-in/fold bias or Critic EV≈0 problem has been resolved. Follow-up work requires an independent test distinct from the existing sealed-final evaluation, calibration/coverage checks across multiple snapshots, and comparison across multiple training seeds.

### Reproducing the HUNL Agent Track and Conditions for Further Development

The complete Python appendices contain validation sources from `tests/test_phase5_*` through [tests/test_phase17_gpu_retrain.py](technical_sources/PYTHON_SOURCE_25B.md#py-296). Key checks cover legality/masking, privacy, zero-sum rewards, network shapes/gradients, β0 equivalence and analytic correction, on-policy collection, seed-block statistics, source immutability, interrupted full-state recovery, and forced rollback of unsafe steps. This documentation edit is not a GPU study that reran those tests.

When continuing development, first distinguish **Base MC PPO from Expand GAE**, **independent resets from continuous stacks**, **Actor-only exports from full training checkpoints**, **validation-best checkpoints from latest resume checkpoints**, and **attempted from accepted updates**. The stage in which a human GUI plays a frozen Actor and saves history for display is distinct from a closed loop that collects human trajectories as versioned on-policy experience, retrains, independently evaluates, and promotes a model. Automatic PPO learning from human play is not currently complete.
<a name="section-73301c2cf7bd"></a>

## Expand Project — E1–E7

```math
\boxed{\text{E1 Table}\to\text{E2 InformationState}\to
\{\text{E3 PPO},\text{E4 Blueprint},\text{E5 Search}\}
\xrightarrow[\text{E6 public evidence}]{\text{E7 routing}}\text{legal table action}}
```

Expand extends Base's heads-up training path to **six physical seats, two to six participants, and variable stacks**. E3 provides the neural reinforcement-learning path; E4/E5 provide the MCCFR and search path. These paths use different models, checkpoints, and information-set contracts. E7 connects them through a common interface, but integration does not automatically enable every path.

The code blocks in this section are excerpts from the specified original `.py` files. Complete source files, tests, execution scripts, and exact per-file hashes are available under the corresponding paths in the Python source appendix and source manifest. Experimental figures below are **historical records** from earlier reports, rather than results of new training or reruns performed during this documentation consolidation.

| Phase | Development objective | Practical benefit | Deliverables required by the next stage |
|---|---|---|---|
| E1 | Manage a dynamic cash table outside the hand engine | Separate admission, departure, rebuy, stack carryover, and chip accounting | `DynamicTable`, `HandHandle`, `Settlement` |
| E2 | Define six-seat player observations and translate legal actions | Fix private-card boundaries and neural-network inputs | `InformationState`, `EncodedObservation`, `LegalActionSpec` |
| E3 | Train PPO for two- to six-player cash games | Provide a trainable multiplayer neural baseline | `expand-e03-v1` checkpoint |
| E4 | Implement external-sampling linear MCCFR | Validate regret logic in small games | Blueprint, abstraction fingerprint, exact infoset key |
| E5 | Run bounded search for the current decision | Sample worlds consistent with public information and compute a local policy | `SearchResult`, time/node/memory estimates |
| E6 | Model opponents from public actions and constrain policy changes | Track style changes and retain the baseline strategy when evidence is insufficient | `ModelEvidence`, `AdaptationResult` |
| E7 | Integrate policy paths, single-table tournaments, and evaluation | Track policy provenance, fallbacks, and evaluation units | `RoutedDecision`, tournament report, release manifest |

### E1 — Dynamic 6-Max Cash Table

```math
C_{\mathrm{live}}(t)=B_{\mathrm{cum}}(t)-W_{\mathrm{cum}}(t),\qquad
C_{\mathrm{live}}(t)=\sum_{i\in\mathrm{hand}}(s_i+c_i)+
\sum_{i\notin\mathrm{hand}}s_i.
```

```math
\sum_i S_{i,h}^{\mathrm{final}}=\sum_i S_{i,h}^{\mathrm{start}},\qquad
\sum_i r_{i,h}^{\mathrm{chips}}=0,\qquad
r_{i,h}^{\mathrm{BB}}=\frac{r_{i,h}^{\mathrm{chips}}}{BB}.
```

$`B_{\mathrm{cum}}`$ is cumulative buy-in, $`W_{\mathrm{cum}}`$ is cumulative cash-out, $`s_i`$ is the remaining stack, and $`c_i`$ is the player's cumulative contribution in the current hand. During a hand, the conservation equation includes chips already committed to the pot. After settlement, it checks the final stacks.

**Objective and benefit.** `MultiPlayerHandEngine` handles betting and settlement within one hand. E1's `DynamicTable` manages participants and external funding across multiple hands. Join, leave, rebuy, sitout, and resume requests enter a queue and take effect only at hand boundaries. Keeping the roster unchanged during a hand prevents contributions and side pots belonging to previous participants from being confused with newly admitted players. A normal entrant remains in `waiting_bb` until the hand in which that player posts the big blind; a bootstrap exception permits the initial two-player lineup to start the table.

| `.py` path — `src/holdem_ai/expand/environment/` | Responsibility |
|---|---|
| `table_state.py` | `TableConfig`, `Action`, and DTOs for seats, handles, events, and settlement |
| `cash_rules.py` | Validate buy-in/rebuy amounts and cash-game rules |
| `multiway_adapter.py` | Construct the Base engine and advance the button/blinds using surviving seats |
| `table_manager.py` | Request queue, hand start/actions/settlement, conservation checks, and `replay_journal` |
| `event_ledger.py` | Write and read ordered JSON events |
| `tournament_rules.py` | Explicitly reject tournament rules that E1 does not implement |

Actual code — `table_manager.py`, `apply_boundary_events()`:

```python
    def apply_boundary_events(self) -> None:
        if self._open_hand():
            raise RuntimeError('cannot apply pending changes during a hand')
        self.ledger.append('boundary_call')
        self._apply_pending()
```

**Input → processing → output.** `request_join(player_id, chips, seat)` reserves a seat and returns an event ID. `start_hand(seed)` applies pending requests → selects the roster/button → constructs and resets the Base engine → returns a `HandHandle`. `act(player_id, Action)` verifies the actual acting player → calls `step`/`step_raise_to` → checks conservation → returns a `StepEvent`. After terminal state, `settle_hand()` produces a `Settlement`, writes final stacks back to the table, and applies the next boundary requests. `replay_journal()` re-executes external commands and compares their resulting effects. A journal containing reproducibility seeds is privileged experimental material and is not exposed to Actor observations.

**Validation and limitations.** [scripts/expand_phase01_smoke.py](technical_sources/PYTHON_SOURCE_01.md#py-012), [tests/expand/test_e01_table_unit.py](technical_sources/PYTHON_SOURCE_19.md#py-217), `test_e01_smoke_unit.py`, and `test_e01_real_engine.py` validate boundary application, seat/fund conservation, and real-engine integration. The integrated report's `164 passed, 7 skipped` includes E1, but some cases were skipped because the real engine was unavailable; this cannot be interpreted as independent certification of real six-player poker rules. E1 supports cash games only and requires a single owner to serialize requests.

### E2 — Information State, Seat Encoding, Action Translation

```math
I_i=(H_i,B,\mathrm{public\ seats},\mathrm{public\ history},\mathcal A_i^{\mathrm{legal}}),\qquad
H_{j\ne i}\notin I_i.
```

```math
x_i=\mathrm{Encode}(I_i):\quad
X_{\mathrm{seat}}\in\mathbb R^{6\times19},\quad
x_{\mathrm{global}}\in\mathbb R^{114},\quad
X_{\mathrm{history}}\in\mathbb R^{128\times21},\quad
m_{\mathrm{action}}\in\{0,1\}^{7}.
```

```math
g(x)=\mathrm{sgn}(x)\log(1+|x|),\qquad
P'=P+c,\quad T(k)=b_i+c+\lfloor kP'\rfloor,\quad k\in\{\tfrac12,1,2\}.
```

$`H_i`$ denotes the player's two hole cards, $`B`$ the public board, $`P`$ the current pot, $`c`$ the amount required to call, and $`b_i`$ the amount already contributed on the current street. $`T(k)`$ is **the final total street wager, rather than the additional amount paid now**. An ordinary raise must satisfy $`T(k)>b_i+c`$ and $`T_{\min}\le T(k)\le b_i+s_i`$, and the engine must also permit raising. A short all-in call is represented by the single `call` action; duplicate raises with the same target amount are removed.

**Difference between the original report and current source.** The equations in the historical `EXPAND_PHASE02_INFORMATION_AND_ENCODING.md` state a global dimension of 110, although that same document's component diagram gives `52+52+4+6`. Current `encoder.py` declares `GLOBAL_DIM = 52 + 52 + 4 + 6` and checks the actual output length, so **114 is the execution contract**. The original report remains unchanged in the appendix; the main text uses the value verified in source. The 110-dimensional description must not be used as the model input specification.

**Objective and benefit.** The observation retains six physical-seat inputs regardless of participant count. It specifies access boundaries for public seat state and private cards, and prevents confusion between chip/BB units or between action-head indices and engine IDs. The declared `TrainingOnlyCriticState` is not an allowed Actor-encoder input type, and E3's value network currently uses only `EncodedObservation` as well.

| `.py` path — `src/holdem_ai/expand/information/` | Responsibility |
|---|---|
| `schema.py`, `builder.py` | Allowed observation DTOs and conversion of actor-safe engine observations |
| `encoder.py` | Card one-hot encoding, public seats, up to 128 actions, and padding masks |
| `actions.py` | Seven-action mask, integer raise-to calculation, and `to_table_action` |
| `adapter.py` | E1 connection and exact street recording for accepted actions |
| `policy_v2.py` | Seat Transformer pooling, action-history GRU, and seven-action head |

Actual code — raise-candidate calculation in `actions.py`, `legal_action_spec()`:

```python
    for index, factor, engine_id in ((3, 1, 2), (4, 2, 3), (5, 4, None)):
        # exact integer math: floor(k * (pot + to_call)) where k is 1/2/2.
        post_call = pot + to_call
        amount = (post_call * factor) // 2
        target = b + to_call + amount
        permitted = (4 in ids and target > high and minimum <= target <= maximum)
        if engine_id is not None:
            permitted = permitted and engine_id in ids
        # Distinct legal actions, so no duplicate wager-size decisions.
        if permitted and target not in targets:
            allowed[index], targets[index] = True, target
```

```math
z=f_\theta(\mathrm{Pool}(\mathrm{Transformer}(X_{\mathrm{seat}})),
\mathrm{GRU}(X_{\mathrm{history}}),x_{\mathrm{global}}),\qquad
\pi_\theta(a\mid I)=\frac{m_a e^{z_a}}{\sum_bm_b e^{z_b}}.
```

**Execution sequence.** Attach `Phase2TableAdapter(table)` before starting the hand. `observe()` checks the E1 snapshot and current actor observation, then calls `make_information_state()`. `encode_observation()` produces tensor-ready data for log-scaled BB amounts, cards, and history. `PolicyV2.forward()` excludes empty seats through Transformer padding, processes only the valid history length in the GRU, and masks illegal logits. The selected action name becomes an E1 `Action` through `to_table_action()`. A `PublicAction` is appended only after the underlying table accepts the action.

**Validation and limitations.** `expand_phase02_smoke.py`, `test_e02_state_and_actions.py`, `test_e02_policy.py`, `test_e02_adapter_fake.py`, and `test_e02_real_engine.py` cover duplicate cards, information boundaries, masks, exact wagers, and padding. E2 does not train a policy. Its public history preserves semantics but generally lacks a one-to-one correspondence with the **original engine action-ID trace** required by E4. Automatic conversion of E2 observations into E4 infosets therefore remains unverified.

### E3 — Dynamic 2–6-Player Multi-Agent PPO

```math
r_{i,T_i}^{\mathrm{BB}}=\frac{S_{i,h}^{\mathrm{final}}-S_{i,h}^{\mathrm{start}}}{BB},\quad
r_{i,t<T_i}=0,\quad d_{i,T_i}=1.
```

```math
\delta_{i,t}=r_{i,t}+\gamma(1-d_{i,t})V(x_{i,t+1})-V(x_{i,t}),\quad
\widehat A_{i,t}=\delta_{i,t}+\gamma\lambda(1-d_{i,t})\widehat A_{i,t+1}.
```

```math
\rho_t=\frac{\pi_\theta(a_t\mid x_t)}{\pi_{\mathrm{old}}(a_t\mid x_t)},\quad
L=-\mathbb E[\min(\rho_t\widehat A_t,
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\widehat A_t)]
+c_V\mathbb E[(V_\theta-\widehat R)^2]-c_H\mathbb E[\mathcal H(\pi_\theta)].
```

$`T_i`$ is that player's last decision position. Another player's turn is not appended as the next state of the player's own trajectory. Defaults are $`\gamma=1,\lambda=0.95,\epsilon=0.2,c_V=0.5,c_H=0.01`$.

**Objective and benefit.** This phase produces a trainable baseline policy for variable player counts and public six-seat observations. The collector holds weights fixed while collecting on-policy samples for an update. It mixes shared self-play policies, historical snapshots, and scripted opponents, and separates training seeds from evaluation seeds. E1's big-blind admission warmup is settled normally but excluded from the training-hand count. Stacks carry over within the table, while **each hand is terminal for GAE**.

| `.py` path — `src/holdem_ai/expand/rl/` | Responsibility |
|---|---|
| `model.py` | `ActorCritic` combining E2 `PolicyV2` and actor-visible `ValueV2` |
| `collector.py`, `population.py` | Two- to six-player E1/E2 rollouts and learner/snapshot/scripted roles |
| `returns.py` | Trajectory grouping by session, hand, and seat; GAE |
| `ppo.py` | Clipped objective, gradient clipping, rejection of nonfinite values, and KL stopping |
| `trainer.py` | Exclusive run lock, atomic saves, validation, resume, and trusted Actor loading |
| [scripts/expand_phase03_train.py](technical_sources/PYTHON_SOURCE_02.md#py-015), [scripts/expand_phase03_evaluate.py](technical_sources/PYTHON_SOURCE_02.md#py-014) | Training and read-only evaluation entry points |

Actual code — trajectory partitioning in `returns.py`, `prepare_gae()`:

```python
    groups: dict[tuple[str, str, int], list[int]] = {}
    for k, t in enumerate(transitions):
        key = t.session_id, t.hand_id, t.seat_id
        groups.setdefault(key, []).append(k)
```

Actual code — objective in `ppo.py`, `optimize_ppo()`:

```python
            ratio=log_ratio.exp()
            adv=advantages[indices]
            loss_pi=-torch.minimum(ratio*adv, ratio.clamp(1-config.clip_epsilon,1+config.clip_epsilon)*adv).mean()
            loss_v=torch.square(v-returns[indices]).mean()
            entropy=dist.entropy().mean()
            loss=loss_pi+config.value_coef*loss_v-config.entropy_coef*entropy
```

**Execution sequence.** `train()` → acquire the run lock and restore configuration/checkpoint → `MultiAgentCollector.collect()` → record terminal payoff on each seat's last sample → `prepare_gae()` → `optimize_ppo()` → evaluate against baselines using separate seeds → save `latest.pt`/`best.pt` and metrics. `load_trained_actor()` reads with `weights_only=True`, verifies the `expand-e03-v1`/`expand-e03-snapshot-v1` format, constructs the model from its configuration, and loads the weights.

**Validation and limitations.** The initial isolated E1–E3 record is `45 passed, 3 skipped`; real-engine and Windows CUDA acceptance remained pending. `test_e03_returns_and_policy.py`, `test_e03_trainer.py`, `test_e03_cli.py`, and `test_e03_real_engine.py` cover the relevant contracts. The E3 collector enforces rake-free **cash games only**. This is not tournament PPO training, and the model does not optimize payouts/ICM, elimination, or blind schedules. Loading it in Tournament ENV 1.4 is cash-to-knockout inference transfer. Base HUNL200 Actor checkpoints and E3 checkpoints are not interchangeable.

### E4 — External-Sampling Linear MCCFR Blueprint

```math
\sigma_i^t(a\mid I)=
\begin{cases}
\dfrac{[R_i^{t-1}(I,a)]_+}{\sum_b[R_i^{t-1}(I,b)]_+},&\sum_b[R_i^{t-1}(I,b)]_+>0,\\
1/|\mathcal A(I)|,&\text{otherwise},
\end{cases}
```

```math
\widetilde r_i^t(I,a)=\widetilde v_i^t(I,a)-
\sum_b\sigma_i^t(b\mid I)\widetilde v_i^t(I,b),\quad
R_i^t=R_i^{t-1}+w_t\widetilde r_i^t,
\quad w_t=\left\lfloor\frac{t}{N}\right\rfloor+1.
```

```math
S_i^t(I,a)=S_i^{t-1}(I,a)+w_t\pi_i^t(I)\sigma_i^t(a\mid I),\qquad
\bar\sigma_i^t(a\mid I)=\frac{S_i^t(I,a)}{\sum_b S_i^t(I,b)}.
```

$`N`$ is the participant count, $`t`$ is the code's zero-based completed-iteration index, and $`\pi_i^t(I)`$ is the traverser's own reach probability. Chance and non-traverser actions are sampled; traverser actions branch. Opponent/chance reach is therefore represented through sample selection, while average-strategy accumulation **explicitly multiplies by own reach**. As in the actual source, linear weighting applies to both regret and strategy sums.

**Objective and benefit.** Validate regret matching, own-reach averaging, iteration atomicity, and resume in small games, then connect them to a reduced Hold'em tree. This is an alternative training option to E3 PPO; it does not reproduce Pluribus's large-scale training volume or performance.

| `.py` path — `src/holdem_ai/expand/pluribus/` | Responsibility |
|---|---|
| `blueprint/mccfr.py` | `InfosetStore`, `MCCFRSolver`, budgets, and external-sampling updates |
| `blueprint/toy_games.py`, `blueprint/kuhn_reference.py` | Kuhn/Leduc and exact best response/NashConv for two-player Kuhn |
| `blueprint/holdem.py` | Base-engine state copying and a restricted legal-action tree |
| `abstraction/cards.py` | Exact preflop cards, temporary postflop hash buckets, and public-history keys |
| `blueprint/checkpoint.py`, `blueprint/evaluate.py` | Checkpoint save/restore and evaluation |
| [scripts/expand_phase04_mccfr.py](technical_sources/PYTHON_SOURCE_02.md#py-017), [scripts/expand_phase04_evaluate.py](technical_sources/PYTHON_SOURCE_02.md#py-016) | CLI training and evaluation |

Actual code — traverser update in `blueprint/mccfr.py`, `MCCFRSolver.iterate()`:

```python
            avg = vector_for(average_deltas,key,len(actions))
            for a in range(len(actions)):
                avg[a] += weight * own_reach * policy[a]
            vals = [visit(state.next_state(action),own_reach*policy[j],depth+1)
                    for j,action in enumerate(actions)]
            mean = sum(p*v for p,v in zip(policy,vals))
            reg = vector_for(regret_deltas,key,len(actions))
            for j in range(len(actions)):
                reg[j] += weight*(vals[j]-mean)
```

**Execution sequence.** Deal through `game.start(rng)` → select traverser and linear weight from the iteration index → construct an actor-safe infoset in `visit()` → sample one opponent action and recursively evaluate all legal traverser actions → finish local regret/average deltas → `store.update()` → increment the completed counter. If the node or depth budget is exceeded, partial traversal deltas are not applied to the store. For Hold'em, `next_state()` uses `deepcopy(engine)` and advances with a legal engine ID.

```math
\mathrm{NashConv}(\bar\sigma)=\sum_i
\left[\max_{\sigma_i'}u_i(\sigma_i',\bar\sigma_{-i})-u_i(\bar\sigma)\right].
```

| Historical two-player Kuhn iterations | Exact NashConv |
|---:|---:|
| 0 | 0.916667 |
| 3,000 | 0.050732 |
| 10,000 | 0.013073 |
| 30,000 | 0.015137 |

These are **Kuhn convergence diagnostics**, rather than 6-Max Hold'em exploitability measurements. Sampling results need not decrease monotonically. The actual postflop bucket is the temporary `SHA256(cards) mod K` bucket, rather than equity/EMD clustering. The default action tree offers fold, check/call, pot, and all-in, with optional half-pot; it differs from E2's full seven-action set. Negative-regret pruning, a large-scale 100BB Blueprint, and six-player equilibrium properties remain unverified. Related tests are `test_e04_toy_mccfr.py`, `test_e04_holdem_mock.py`, `test_e04_checkpoint.py`, `test_e04_cli.py`, and `test_e04_real_engine.py`.

### E5 — Belief-Conditioned Bounded Search

```math
q(\omega\mid H_i,B,h_{\mathrm{pub}})\propto
\mathbf1[\omega\text{ has distinct cards and matches visible facts}]
\prod_{(j,a)\in h_{\mathrm{pub}},\,j\ne i}\ell_j(a\mid I_j(\omega)).
```

```math
\pi_{\mathrm{traverse}}=(1-\alpha)\pi_{\mathrm{local}}+\alpha\pi_{\mathrm{blueprint}},
\quad\alpha=0.15,\qquad
\widehat v_{\mathrm{leaf}}=u(\text{one continuation rollout}).
```

$`\omega`$ is a complete world **newly sampled** to match the observation, rather than a world constructed by reading actual opponents' hole cards. Without an opponent-likelihood function, sampling is conditioned on public facts and card constraints; an action-conditioned posterior is not learned automatically. When such a function is provided, rejection sampling uses the product of likelihoods during public-action replay.

**Objective and benefit.** Compute a more detailed current decision than the Blueprint within a bounded CPU budget. Distinguish insufficient search or timeout from a completed solve so that the higher-level policy can handle each outcome. Default budgets are 500ms, 2,500 search nodes, 80 iterations, 1,000 infosets, depth 2, a 64MB **table-size estimate**, and at most 200 rollout steps. The 64MB estimate is not a strict cap on actual process RSS.

| `.py` path — `src/holdem_ai/expand/pluribus/search/` | Responsibility |
|---|---|
| `belief.py` | Generate toy/real-deck worlds consistent with hero/public facts and replay public actions |
| `continuations.py` | Four continuation variants: Blueprint and fold/call/raise biases |
| `solver.py` | Current-root, depth-limited MCCFR, budgets, and `SearchResult` |
| `nested.py` | Public trace, already executed hero commitments, and off-tree counts |
| [scripts/expand_phase05_search.py](technical_sources/PYTHON_SOURCE_02.md#py-019), [scripts/expand_phase05_evaluate.py](technical_sources/PYTHON_SOURCE_02.md#py-018) | Single search and toy A/B evaluation |

Actual code — belief-root validation in `solver.py`, `SubgameSolver.search()`:

```python
                state=belief.sample(rng)
                if state.current_player!=belief.hero or state.is_terminal():
                    raise ValueError('belief sampled an invalid hero root')
                if tuple(state.legal_actions())!=root_actions or state.infoset(belief.hero)!=root_key:
                    raise ValueError('belief root information mismatch; no safe solve possible')
```

**Execution sequence.** `HoldemObservationBelief(raw_obs, initial_stacks, blinds)` places the known hero cards and public board at the engine's deal/burn indices, then shuffles the remaining deck without collisions. `sample()` replays public action IDs/raise-to amounts from the initial state and checks agreement in current actor, stacks, pot, street, cards, and history. `search()` repeatedly samples worlds sharing the same hero infoset → performs local regret search to the depth limit → runs continuation rollouts at leaves → commits iteration deltas → returns the root average policy and status. A continuation variant is selected per `(player, infoset)`, preventing hidden-world-specific oracle choices within the same infoset.

**Validation and limitations.** The historical 40-iteration Kuhn record reports `solved`, 180 nodes, 12.78ms, `P(check)=0.609318`, and `P(bet)=0.390682`. A separate 16-game toy A/B evaluation reports mean Search−Blueprint payoff of `−0.125` chip/hand with a 95% normal-approximation CI of `[-0.2924,+0.0424]`; this did not establish improvement. Latency values reflect the execution environment used at the time. `test_e05_beliefs.py`, `test_e05_holdem_deck.py`, `test_e05_continuations.py`, `test_e05_solver.py`, `test_e05_nested.py`, `test_e05_evaluate.py`, `test_e05_cli.py`, and `test_e05_real_engine.py` cover the respective boundaries.

`NestedSearchSession` maintains a **factual history for searching the current decision again**. It does not implement original Pluribus nested re-solving that safely re-solves from the beginning of the betting round, a joint leaf-continuation subgame, or an exploitability guarantee. E5 can return results from completed iterations together with a timeout status, but **the E7 bridge accepts only `status='solved'`**.

### E6 — Public Opponent Model and Bounded Adaptation

```math
C_{t,c,a}^{L}=\lambda_L C_{t-1,c,a}^{L}+\mathbf1[a_t=a],\quad
C_{t,c,a}^{R}=\lambda_R C_{t-1,c,a}^{R}+\mathbf1[a_t=a],\quad
\widehat p_{c}(a)=\frac{C_{c,a}+\alpha}{\sum_bC_{c,b}+\alpha|\mathcal A_c|}.
```

```math
\pi_\eta=(1-\eta)\pi_{\mathrm{base}}+\eta\pi_{\mathrm{proposal}},\quad
0\le\eta\le\eta_{\max}\frac{n_{\mathrm{eff}}}{n_{\mathrm{eff}}+\kappa},\quad
D_{KL}(\pi_\eta\Vert\pi_{\mathrm{base}})\le0.025,\quad
D_{TV}(\pi_\eta,\pi_{\mathrm{base}})\le0.075.
```

$`c`$ is a context comprising street, position, SPR bucket, and the opponent's legal-action tuple. Default decays are $`\lambda_L=0.995,\lambda_R=0.88`$; the Dirichlet prior is $`\alpha=0.5`$, maximum mixture weight is $`\eta_{\max}=0.3`$, and the confidence pseudocount is $`\kappa=20`$. Model identity is `(player_id, player_session_id)`, so a previous participant's tendencies do not remain attached to the physical seat.

**Objective and benefit.** Estimate action distributions and style shifts from accepted public actions. Mix a separately provided hero-expert policy to a limited extent according to evidence. E6 does not itself learn a profitable counter-strategy. The opponent's legal-action set and the current hero's legal-action set belong to different observations, and their agreement checks must not be confused.

| `.py` path — `src/holdem_ai/expand/opponent/` | Responsibility |
|---|---|
| `schema.py`, `model.py` | Public events/contexts, dual decay, Dirichlet posterior, shift detection, and identity registry |
| `bridge.py`, `persistence.py` | Collect accepted E1/E2 events and save/restore the registry |
| `router.py`, `adaptive.py` | Select style experts, gate on evidence, and constrain mixing by KL/TV |
| [scripts/expand_phase06_benchmark.py](technical_sources/PYTHON_SOURCE_02.md#py-020) | Synthetic forecasting experiment separate from Hold'em |

Actual code — trust-region search in `adaptive.py`, `SafeAdaptivePolicy.adjust()`:

```python
        def blend(w):return {a:(1-w)*p[a]+w*q[a] for a in legal}
        def allowed(w):
            cand=blend(w)
            return kl_divergence(cand,p)<=self.guard.kl_budget+1e-14 and tv_distance(cand,p)<=self.guard.tv_budget+1e-14
        if not allowed(max_w):
            lo,hi=0.,max_w
            for _ in range(60):
                mid=(lo+hi)/2
                if allowed(mid):lo=mid
                else:hi=mid
            max_w=lo
```

**Execution sequence.** Bind player tenure through `sync_seats()` after the boundary → capture `before=adapter.observe()` → obtain actual table acceptance → `record_accepted_action(before,...)` → update evidence through `OpponentRegistry.observe()` → call `StyleExpertRouter.propose()` at the hero decision → check action support, sample count, change points, and KL/TV in `SafeAdaptivePolicy.adjust()` → return the policy. Rejected actions are not used as evidence. No new probability mass is assigned to actions whose baseline probability is zero.

The actual shift detector matches prior-to-data ratios with $`\alpha_R=\alpha_L N_R/N_L`$ so that differing recent/long accumulated counts do not create a false shift. With sufficient samples, consecutive exceedances of the KL threshold activate `shift_hold` and restart long counts in the recent regime. The baseline policy remains in use until evidence accumulates again.

| Historical synthetic 16-block evaluation | Value |
|---|---:|
| Uniform NLL / online model NLL | 1.098612 / 0.766122 |
| NLL improvement and 95% CI | +0.332491 / [+0.310901,+0.354081] |
| Shift detection | 16/16 blocks |
| Maximum observed policy KL / TV | 0.012583 / 0.071326 |

These are synthetic public-action forecasting results for a Tight→Aggressive transition. They are not measurements of actual poker EV, bb/100, or exploitability. A separate contract is also needed to translate unrestricted off-tree raise-to actions into E2's seven-action evidence representation. Related tests under `tests/expand/` are `test_e06_models.py`, `test_e06_router.py`, `test_e06_adaptive.py`, `test_e06_bridge.py`, `test_e06_persistence.py`, `test_e06_state_bridge.py`, `test_e06_cli.py`, and `test_e06_real_engine.py`.

### E7 — Integration, Single-Table Tournament, Scientific Benchmark

```math
\pi_{\mathrm{selected}}=
\mathrm{FirstValid}(\pi_{\mathrm{search}},\pi_{\mathrm{blueprint}},\pi_{\mathrm{neural}}),
\qquad
a\sim\pi_{\mathrm{selected}},\quad A_{\mathrm{table}}=\mathrm{Translate}(a).
```

```math
\ell(h)=\min\!\left(\left\lfloor h/H_{\mathrm{level}}\right\rfloor,L-1\right),\quad
\sum_{i\in\mathrm{alive}}S_i=C_{\mathrm{registered}},\quad
\widehat\mu\pm1.96\frac{s_{\mathrm{block}}}{\sqrt{K}}.
```

$`h`$ is the number of completed hands, and $`H_{\mathrm{level}}`$ is the number of hands per blind level. $`C_{\mathrm{registered}}`$ is the total number of chips introduced through admission and permitted re-entry. $`K`$ is the number of independent table/tournament blocks; all actions within one block are not counted as independent samples.

**Objective and benefit.** Connect play and evaluation while recording policy-call contracts, compatibility, and failure causes. Separate from E1's cash table, E7's `SingleTableTournament` manages entry/re-entry, stack carryover, blind schedules, elimination, rankings for simultaneous eliminations, and payouts. Chip payoff and final prize/rank are different objectives; implementing a tournament runner does not create a tournament learner.

| `.py` path — `src/holdem_ai/expand/integration/` | Responsibility |
|---|---|
| `agent.py` | `IntegratedPokerAgent`, source routing, legal-distribution normalization, and `RoutedDecision` |
| `source_bridges.py` | Opt-in adapters for exact E4 infosets and verified E5 beliefs |
| `tournament.py` | Boundary-only registration/re-entry, blind levels, survivor button, and rank/prize ledger |
| `benchmark.py` | Frozen cash/tournament strata, paired comparisons, and block CIs |
| `manifest.py` | Track configuration/seeds/runtime/checkpoint SHA-256 and evaluation artifacts |
| [scripts/expand_phase07_benchmark.py](technical_sources/PYTHON_SOURCE_02.md#py-021) | Reproducible evaluation CLI |

Actual code — candidate order in `agent.py`:

```python
        candidates = (
            ('search', self.search, self.config.use_search, self.config.verified_search),
            ('blueprint', self.blueprint, self.config.use_blueprint, self.config.verified_blueprint),
            ('neural', self.neural, True, True),
        )
```

Actual code — handling incomplete E5 solves in `source_bridges.py`:

```python
    def distribution(state:InformationState):
        result=solver.search(verified_root_belief(state))
        if result.status!='solved':
            raise TimeoutError(f'search incomplete: {result.status}')
        legal=tuple(a for a,yes in zip(ACTION_NAMES,state.legal_actions.mask) if yes)
        if set(result.action_probabilities)!=set(legal):
            raise ValueError('search root action abstraction incompatible')
        return dict(result.action_probabilities)
```

**Actual fallback contract.** E4 checks the caller-provided `exact_infoset_key` and the stored strategy's exact action set. An unseen infoset raises an error rather than being presented as a learned uniform strategy. E5 requires a verified belief; statuses other than `solved` and action mismatches pass control to the next source. The router attempts search→Blueprint→neural while collecting reasons for unverified sources, missing sources, exhausted budgets, invalid distributions, and exceptions. If every source fails, it explicitly declares `uniform_legal_emergency`. Exhausting the optional budget still leaves an opportunity to invoke the neural fallback. After source selection, E6 can mix an external expert proposal within its constraints.

**Input and output.** `act(InformationState,rng_seed,evidence)` → strict legal-set normalization → selection/adaptation → seeded sampling → `to_table_action()` → a `RoutedDecision` containing the action name, actual `Action`, probabilities, selected source, and fallback reasons. The cash benchmark evaluates one scored hand per independent table with matched player count, stack, style, hero seat, and deck seed; admission warmup is counted separately. The tournament benchmark aggregates wins, rank, and prize per completed tournament and does not assign an arbitrary champion when the maximum-hand limit stops play.

**Validation and limitations.** Historical isolated E7 tests report 23 passed; the E1–E7 combined record is `164 passed, 7 skipped`. Coverage includes `test_e07_agent.py`, `test_e07_bridges_and_cli.py`, `test_e07_cash_bridge.py`, `test_e07_benchmark.py`, `test_e07_manifest.py`, `test_e07_tournament.py`, `test_e07_real_engine.py`, and a fake tournament fixture. Exact E2→E4 key equality, actual Windows engine/CUDA acceptance, and real-game opponent-adaptation EV remain pending. Antes, a multi-table director, an ICM solver, and original Pluribus complete nested search are outside the implemented scope.

<a name="section-299f12b6cbd0"></a>

## HUNL GUI Project — G1.0–G1.6

```math
\boxed{\mathrm{Browser}\xrightarrow{(a,r)}\mathrm{FastAPI}
\xrightarrow{\mathrm{Session}}\mathrm{InteractiveHUNLAdapter}
\xrightarrow{}\mathrm{MultiPlayerHandEngine}}
```

```math
I_i=O_i(S),\quad a_i\sim\pi_\theta(\cdot\mid I_i),\quad
\mathrm{GameView}=\mathrm{Project}_{\mathrm{human}}(S),\quad
\mathrm{Accept}(a,r)\iff r=r_{\mathrm{server}}\land a\text{ is legal at the current human turn}.
```

The GUI is an **interface through which a human and a frozen Agent play on the same engine and inspect results**, rather than a learning algorithm. Policy input $`I_i`$ and the GameView sent to the human are different projections. The browser receives no deck, seed, privileged critic state, or unrevealed opponent cards. This section reads current Python backend source alongside historical phase reports, distinguishing the continuous default introduced in G1.6 from the historical independent-hand descriptions in G1.0–G1.5. Frontend files and behavior described below follow the established contracts in the original reports; actual code excerpts in this section come from the reviewed Python files.

| Phase | Objective | Benefit |
|---|---|---|
| G1.0 | Transport-independent session and human-safe observation contract | Separate rules, policy, and projection |
| G1.1 | Local FastAPI requests/responses | Fix schemas, revisions, and HTTP failure handling |
| G1.2 | Read-only React table | Visually inspect server state |
| G1.3 | Actual human betting and unrestricted raise-to | Allow a seven-head Agent and integer human wagers to coexist |
| G1.4 | Local trained-checkpoint inference | Model selection, compatibility checks, and explicit load failures |
| G1.5 | Completed-hand records and performance dashboard | Track cumulative profit/loss, variance, and history |
| G1.6 | Continuous stack play | Carry settled stacks forward, end on bankruptcy, and settle hands with no decisions |

### G1.0 — Human/Agent Session Contract

```math
V_{t+1}=\mathrm{Project}_{0}(T(S_t,a_t)),\qquad
\mathrm{opponent\_cards}=
\begin{cases}H_1,&\mathrm{terminal}\land\mathrm{genuine\ showdown},\\
\mathrm{null},&\text{otherwise}.
\end{cases}
```

The human occupies seat 0 and the bot seat 1. In the current implementation, a genuine showdown is identified by `not any(result.folded)` and `len(result.board)==5`. Opponent cards remain hidden when a hand terminates by folding.

**Objective and implemented files.** `HUNLGameSession`/`HUNLSessionRegistry` in `game_service/session_manager.py` handle multi-hand sessions, independent RNGs, `RLock`, and revisions. `action_adapter.py` validates seven-head human actions and constructs `ActionOffer` objects with actual amounts. `bot_adapter.py` supplies the policy with only that bot's `InformationState`. `state_projection.py` serializes only explicitly listed fields to JSON. The common prefix for the full original paths is `src/holdem_ai/`.

Actual code — result-card disclosure fields in current `state_projection.py`:

```python
            "showdown": is_showdown,
            # Revealing the opponent is allowed only after a genuine showdown.
            "opponent_cards": list(engine.hole_cards[opponent]) if is_showdown else None,
```

**Execution sequence.** `start_hand()` starts the engine and automatically advances bot actions until a human turn or terminal state. `play_human_action(index,expected_revision)` checks revision → active hand → human turn → legal index, then calls `_take_action()` and `_run_bots()` and returns the latest view. `view()` does not advance the hand. Indices 0–6 are the **policy heads** `[fold,check,call,half-pot,pot,double-pot,all-in]`; they differ from the Base engine's five-ID contract.

**Validation and limitations.** `test_gui10_hunl_contract.py` and [scripts/gui10_smoke.py](technical_sources/PYTHON_SOURCE_02.md#py-022) cover privacy, legal actions, revisions, and zero-sum behavior. Historically, G1.0 initialized every new hand at 200BB; this remains the Python session's `independent` default after G1.6. The session registry is memory-only.

### G1.1 — FastAPI Local Server

```math
\mathrm{Response}=\{\mathrm{session\_id},\mathrm{state}:V\},\qquad
\mathrm{stale\ revision}\mapsto409,\quad
\mathrm{invalid\ action/schema}\mapsto422,\quad
\mathrm{unknown\ session}\mapsto404.
```

**Objective and files.** [backend/main.py](technical_sources/PYTHON_SOURCE_01.md#py-003) constructs the app, CORS configuration, and registry, and runs on loopback. [backend/schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-004) defines strict request/response DTOs; [backend/game_router.py](technical_sources/PYTHON_SOURCE_01.md#py-002) handles routing under `/api/v1/hunl`. `start_next_hand(expected_revision)` in `session_manager.py` atomically rejects duplicate Next Hand clicks. The router does not reimplement poker rules.

Actual code — action HTTP handler in [backend/game_router.py](technical_sources/PYTHON_SOURCE_01.md#py-002):

```python
        state = run_game_operation(
            lambda: session.play_human_action(
                request.action_index, expected_revision=request.expected_revision
            )
        )
```

| HTTP operation — under `/api/v1/hunl` | Input and output |
|---|---|
| `POST /sessions` | Bot/seed/blinds/stack mode → new ID and GameView |
| `GET /sessions/{id}` | Latest view, without advancing the game |
| `POST /sessions/{id}/actions` | Action index + revision → post-processing view |
| `POST /sessions/{id}/next-hand` | Revision → next hand or 409 |
| `DELETE /sessions/{id}` | Remove the session |
| `GET /health` | Service status |

**Operation and validation.** Pydantic input validation → registry lookup → session lock/revision checks → engine advancement → response projection/schema serialization. [tests/test_gui11_http_api.py](technical_sources/PYTHON_SOURCE_22.md#py-258) and [scripts/gui11_smoke.py](technical_sources/PYTHON_SOURCE_02.md#py-023) cover HTTP success, 404/409/422, projection, and CORS. This is a localhost research server with a process-local registry and no authentication; the phase does not implement multiple workers, remote deployment, or persistence.

### G1.2 — Read-Only React Poker Table

```math
\mathrm{UI}=\mathrm{Render}(V_{\mathrm{server}}),\qquad
T_{\mathrm{backend}}(\mathrm{PreviewSelect})=\mathrm{identity}.
```

**Objective and benefit.** Display human cards, the public board, opponent card backs, BTN/SB/BB positions, stacks, street bets, pot, and action history from the same server projection. Preview fixtures are design samples; they do not advance the game or contribute to performance statistics.

**Implemented files and actual Python boundary.** A React table, card-rendering helpers, and an API client were added under `frontend/`. [tests/test_gui12_readonly_api.py](technical_sources/PYTHON_SOURCE_22.md#py-259) validates backend JSON, CORS, and privacy contracts. The Python service reuses G1.0/1.1. The current human-observation check in `state_projection.py` is:

```python
        if obs.player_id != hero_seat:
            raise AssertionError("human GUI cannot receive bot observations")
```

**Execution sequence.** Create/GET a session → receive `GameResponse.state` → render public fields → show Next Hand according to terminal status. Opponent cards are displayed only after a terminal showdown. This phase added no betting clicks, so a live hand stops at the human's first decision; G1.3 adds action submission.

**Validation and limitations.** The original report explicitly deferred `npm install/test/build` and actual browser verification because Node dependency access failed. The UI implementation description must not be interpreted as a passed production build.

### G1.3 — Interactive Betting and Exact Human Raise-to

```math
c_i=\max(0,b_{\max}-b_i),\quad
T_{\min}=b_{\max}+\Delta_{\mathrm{last\ full\ raise}},\quad
T_{\max}=b_i+s_i,\quad
\Delta\mathrm{chips\ paid}=\mathrm{raise\_to}-b_i.
```

**Objective and benefit.** In addition to the seven-action offers presented by the server, a human can submit arbitrary **integer-chip** raise-to amounts. The bot retains the existing seven-head contract. A short raise below the minimum full raise may still be permitted when it equals the exact all-in maximum, subject to engine checks of the rules and betting reopening; frontend range checks alone therefore do not determine final legality.

**Implemented and modified files.** `game_service/interactive_adapter.py` adds a human-specific subclass of the HUNL adapter; `session_manager.py` adds `play_human_raise_to()`; `state_projection.py` adds `can_raise_to` and bounds; [backend/game_router.py](technical_sources/PYTHON_SOURCE_01.md#py-002) and `schemas.py` add the `/raise-to` request. React `BettingPanel.tsx` and `lib/betting.ts` provide presets, a slider, and integer input.

Actual code — `InteractiveHUNLAdapter.step_human_raise_to()`:

```python
        engine.step_raise_to(amount)
        self.last_engine_action = EngineBetAction("raise_to", amount)
        self._history.append(f"{street}|seat={actor}|raise_to|raise_to={amount}")
        return self._receive()
```

**Input and execution.** `POST .../raise-to {raise_to:350,expected_revision:17}` → integer/schema checks → session lock/revision/turn checks → engine legality checks → record exact-amount history → automatic bot actions → latest GameView. If the engine rejects the action, session revision and history are not updated. On 409, the UI must fetch the latest state again.

**Validation and limitations.** Historical compatible-sandbox records report `130 passed,3 skipped`, five new API/custom-raise tests, and transpilation syntax checks for 14 TypeScript files. These figures do not establish a complete current-repository regression, an npm production build, or browser E2E success. Unrestricted raise-to can produce inputs outside the training action abstraction, and policy quality for those inputs remains unverified. [tests/test_gui13_interactive_betting.py](technical_sources/PYTHON_SOURCE_22.md#py-260) covers the relevant Python contracts.

### G1.4 — Frozen Trained-Agent Inference

```math
\theta_{\mathrm{session}}=\theta_{\mathrm{checkpoint}},\quad\Delta\theta=0,\qquad
a\sim\pi_\theta(\cdot\mid I_{\mathrm{bot}})\;(\mathrm{sampled}),\quad
a=\arg\max_a\pi_\theta(a\mid I_{\mathrm{bot}})\;(\mathrm{greedy}).
```

**Objective and files.** `game_service/checkpoint_catalog.py` manages PPO/QBoost checkpoint slots configured by server environment variables, file-existence checks, and caching. It reuses `load_actor()` from `evaluation/checkpoint_loader.py` and `NeuralActorPolicy` from `agents/neural_policy.py`. [backend/game_router.py](technical_sources/PYTHON_SOURCE_01.md#py-002) and `schemas.py` add `/bots` and trained-session configuration. The browser selects a bot ID rather than specifying an input path.

Actual code — `CheckpointCatalog.build_policy()`:

```python
                actor = load_actor(path, device=self.device)
                if actor.config.dropout != 0:
                    raise ValueError("evaluation model must have dropout=0")
                policy = NeuralActorPolicy(actor, device=self.device,
                                           big_blind=100, greedy=self.mode == "greedy")
```

**Execution sequence.** `/bots` reports whether configured files exist → strict loading at session creation → eval-only policy cache → input of the bot's private InformationState → masked seven-head inference → sampling with a per-session RNG. Changes in file path, mtime, or size invalidate the cache. Missing files or incorrect format, weights, or device produce an explicit 422 response rather than substitution with a random bot. Trained HUNL sessions require initial SB/BB=50/100 and 200BB stacks.

**Validation and limitations.** `test_gui14_trained_agents.py`, [scripts/gui14_smoke.py](technical_sources/PYTHON_SOURCE_02.md#py-024), and the original report record 34 backend tests and a compatible full-suite result of `137 passed,3 skipped`. Fixtures include a mocked checkpoint loader, so this is not evidence of completed actual Phase 15 weight validation, Windows CUDA acceptance, or React E2E tests. The `ppo`/`qboost` IDs are server configuration labels; the GUI does not independently certify their training algorithms. E3 multi-seat weights belong to a different family from these HUNL weights.

### G1.5 — Completed-Hand Journal and Performance Dashboard

```math
R_h=\frac{S_h^{\mathrm{final}}-S_h^{\mathrm{start}}}{BB_h},\quad
C_n=\sum_{h=1}^n R_h,\quad\widehat{bb/100}=100C_n/n,
\quad\widehat p_{\mathrm{win}}=N_{\mathrm{win}}/n.
```

```math
\delta_n=R_n-\bar R_{n-1},\quad
\bar R_n=\bar R_{n-1}+\delta_n/n,\quad
M_{2,n}=M_{2,n-1}+\delta_n(R_n-\bar R_n),\quad
s_n=\sqrt{M_{2,n}/(n-1)}.
```

```math
\mathrm{CI}_{95}\approx100\bar R_n\pm1.96\frac{100s_n}{\sqrt n},\qquad
\mathrm{MaxDrawdown}=\max_n\bigl(\max_{k\le n}C_k-C_n\bigr).
```

**Objective and benefit.** Save one human-safe record per completed hand so that history and statistics cannot count the same hand twice. `game_service/session_analytics.py` manages Welford variance, cumulative chips/BB, wins/losses/even results, showdowns, and maximum drawdown. Detailed history is limited to the latest 500 entries, while full cumulative statistics are retained. Connections include the terminal callback in `session_manager.py`, `/stats` and `/history` in [backend/schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-004)/`game_router.py`, and React `SessionAnalyticsPanel.tsx`/`lib/analytics.ts`.

Actual code — `SessionAnalytics.append_completed()`:

```python
        difference = bb - self.mean_bb
        self.mean_bb += difference / self.count
        self.m2_bb += difference * (bb - self.mean_bb)
        self.peak_bb = max(self.peak_bb, self.total_bb)
        self.max_drawdown_bb = max(self.max_drawdown_bb, self.peak_bb - self.total_bb)
```

**Execution sequence.** Engine terminal → `_finish_terminal()` → `project_human_view()` → terminal/privacy/unit/consecutive-hand-number checks in `append_completed()` → save cumulative statistics and a safe snapshot → GET stats/history → render the dashboard. History is newest-first, and the page limit is 1–100. Active hands and preview fixtures do not contribute to completed-hand performance.

**Validation and limitations.** [tests/test_gui15_session_analytics.py](technical_sources/PYTHON_SOURCE_22.md#py-262) and [scripts/gui15_smoke.py](technical_sources/PYTHON_SOURCE_03.md#py-025) cover terminal-only recording, duplicates/numbering, privacy, retention, pagination, and session isolation. The CI is descriptive and assumes independent hands and a normal approximation. Human adaptation and seed/seat imbalances may be present; skill certification requires seed-blocked independent evaluation. In-memory records disappear when the server restarts, and some historical Call/All-in entries lack exact amounts.

### G1.6 — Continuous Stack Play

```math
S_{i,h+1}^{\mathrm{start}}=S_{i,h}^{\mathrm{final}},\quad
R_{i,h}^{\mathrm{chips}}=S_{i,h}^{\mathrm{final}}-S_{i,h}^{\mathrm{start}},\quad
\sum_{i=0}^1S_{i,h}=2\cdot200\cdot100=40{,}000,
\quad BTN_h=(h-1)\bmod2.
```

```math
\mathrm{match\_over}=\mathbf1[\min_iS_{i,h}^{\mathrm{final}}=0].
```

**Objective and benefit.** Browser Play carries settled stacks from the previous hand into the next. The default `stack_mode` for GUI HTTP requests is `continuous`; the default for Python `HUNLGameSession()` and Base training/evaluation is `independent`. Both players start the first hand at 200BB with fixed blinds of 50/100, but subsequent stacks may be asymmetric. There is no rake, rebuy, or top-up.

**Implemented and modified files.** `InteractiveHUNLAdapter.reset_with_stacks()` creates a new hand using the existing Base engine's stacks argument. `session_manager.py` handles carryover, bankruptcy, and terminal hands with no decisions. `state_projection.py`, [backend/schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-004), and `game_router.py` expose `starting_stacks`, `stack_mode`, `match_over`, and `winner_seat`. Terminal seats display final settled stacks rather than pre-award balances.

Actual code — `HUNLGameSession.start_hand()`:

```python
            if self.stack_mode == "continuous" and self._adapter is not None:
                final = self._adapter.engine.get_result().final_stacks
                initial_stacks = (int(final[0]), int(final[1]))
                if min(initial_stacks) == 0:
                    raise RuntimeError("match over: a player is bankrupt; start a new session")
                if sum(initial_stacks) != 2 * self.stack_bb * self.big_blind:
                    raise AssertionError("chips changed across consecutive hands")
```

**Execution sequence.** Existing reset for the first hand → terminal record → Next Hand revision check → obtain the preceding final stacks → bankruptcy/conservation checks → `reset_with_stacks()` → alternate BTN → automatic bot advancement or immediate terminal state. A short stack unable to post the full blind may begin forced all-in and settle without a decision; its completed-hand record, profit/loss, and history are still recorded. Next Hand returns 409 after match-over, and a new session is required. `game_type='HUNL200'` identifies the API/policy family; it does not imply that every subsequent hand has 200BB stacks.

**Validation and limitations.** [tests/test_gui16_continuous_stack.py](technical_sources/PYTHON_SOURCE_22.md#py-263) and [scripts/gui16_smoke.py](technical_sources/PYTHON_SOURCE_03.md#py-026) validate carryover, BTN alternation, chip conservation, independent-mode compatibility, all-in bankruptcy, revisions, and short-blind terminal states. The historical smoke report records 20 consecutive hands and an always-fold bankruptcy finish after 267 hands at `[0,40000]`, but this was isolated validation using stand-ins for some Base imports. It is separate from a passed full-repository Windows/browser build. Performance of an Actor trained on independent 200BB hands is not guaranteed for asymmetric, short, or deep stacks; a variable-stack curriculum is a separate research task.


<a name="section-c5edcf32d2ad"></a>

## Verifying source integration and removing superseded reports

```math
\mathrm{GitBlobSHA}(b)=\mathrm{SHA1}
\left(\mathtt{blob\ }\Vert\mathrm{decimal}(|b|)\Vert\mathtt{NUL}\Vert b\right)
```

$`b`$ is the original Git blob's actual byte sequence, and $`|b|`$ is its byte count. The single LF byte used to separate a display fence from the source is excluded from the original text. Verification compared file paths, original lengths, and Git blob SHAs, and confirmed that each path appeared exactly once.

```math
\mathrm{paths}(A_{PY})=\mathrm{paths}(S_{PY}),\quad
\mathrm{paths}(A_{MD})=\mathrm{paths}(S_{MD})
```

```math
\forall p\in S_{PY}\cup S_{MD}:\quad
\mathrm{count}_A(p)=1,\quad |A[p]|=|S[p]|,\quad
\mathrm{GitBlobSHA}(A[p])=\mathrm{GitBlobSHA}(S[p]).
```

$`S`$ is the original snapshot, and $`A`$ is the final report's source appendices. Lossless preservation requires matching path sets, counts, source sizes, and SHAs. Completion was not inferred from screenshot progress or filenames alone. When the work resumed, the Python appendix contained 300 files; the 27 missing test files were added.

| Verification performed for the integration | Result |
|---|---:|
| Complete original Python sources | **327/327** · 1,400,837 bytes |
| Complete original Markdown sources | **76/76** · 1,049,564 bytes |
| Missing, duplicate, or extra original paths, or SHA mismatches | **0** |
| Syntax checks of complete original Python sources | **327 passed**; separate from game/training runtime tests |
| Python appendices | 28 logical volumes, 29 files; volume 25 split into A/B |
| Markdown appendices | 7 logical volumes, 8 files; volume 6 split into A/B |
| Verification of the source checker's error detection | Detected 8 fault classes: source tampering, duplication, omission, header SHA, marker SHA, anchors, extra sources, and extra appendices |
| Removal of superseded standalone reports | **73 deleted**; report body, README, TECH REPORT, and constituent appendices retained |

The deletion set is defined as follows.

```math
D=S_{MD}\setminus\{\mathtt{README.md},\ \mathtt{docs/TECH\_REPORT.md},\ 
\mathtt{docs/POKER\_AI\_FINAL\_INTEGRATED\_TECHNICAL\_REPORT.md}\}.
```

These 73 documents include the two reference reports, individual phase reports, earlier patch READMEs at the repository root, the earlier short code reference, and the preceding partial source-integration files. Their contents are preserved exactly in the Markdown appendices. Python, TypeScript, test and training code, and configuration were outside the deletion/change scope.

The manifest's SHAs are Git blob SHAs from the original-source reference commit. They are not claimed to match the new SHAs of the final report body, README, or TECH REPORT. The earlier original text of all three files is also included in the appendices, allowing comparison before and after rewriting.

### Python code to reproduce lossless verification

Save the code below as `verify-lossless.py` and run `python verify-lossless.py <repository-root>`. It uses only Python's standard library and does not modify files. It checks original-source SHAs, byte counts, markers, fences, anchors, and the path set, returning exit code 1 if errors are found.

```python
#!/usr/bin/env python3
"""Read-only verification of fenced Git blobs against a preservation manifest.

Usage: python verify-lossless.py /path/to/repository
Only Python's standard library is required. No files are created or modified.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


TOKEN = re.compile(rb'<!-- (BEGIN|END)_(PY|MD)_SOURCE\b[^>]*-->')
BEGIN = re.compile(rb'<!-- BEGIN_(PY|MD)_SOURCE path="([^"]+)" sha="([^"]+)" -->')
END = re.compile(rb'<!-- END_(PY|MD)_SOURCE path="([^"]+)" -->')
OPEN = re.compile(rb'^`{8}(python|markdown)\r?\n', re.M)
CLOSE = re.compile(rb'^`{8}\r?$', re.M)
HEADER = re.compile(rb'^(?=[^\r\n]*bytes)(?:\*\*Blob SHA:\*\*|Original (?:Git blob(?: SHA)?|[Bb]lob(?: SHA)?|SHA)|[Bb]lob|Git Blob)[^\r\n]*$', re.M)
SIZE = re.compile(rb'(?:\*\*Original bytes:\*\*\s*(\d+)|(\d+)\s+bytes)')
KINDS = {'py': 'PY', 'python': 'PY', 'md': 'MD', 'markdown': 'MD'}


def blob_sha(payload):
    return hashlib.sha1(b'blob ' + str(len(payload)).encode('ascii') + b'\0' + payload).hexdigest()


def relative_file(root, value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('path must be a nonempty POSIX-relative string')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('path must remain inside the repository')
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('resolved path escapes the repository')
    return resolved


def verify(root, manifest_path):
    issues = []
    def fail(message, **details):
        issues.append(dict(error=message, **details))
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, ValueError) as exc:
        return {'ok': False, 'errors': [{'error': 'cannot read manifest', 'detail': str(exc)}]}
    if not isinstance(manifest, dict) or not isinstance(manifest.get('entries'), list) or not manifest['entries']:
        return {'ok': False, 'errors': [{'error': 'manifest.entries must be a nonempty array'}]}
    for key in ('source_commit', 'main_base_commit'):
        if not isinstance(manifest.get(key), str) or not re.fullmatch('[0-9a-f]{40}', manifest[key]):
            fail('invalid manifest commit', field=key)
    expected, archives, anchors = {}, set(), set()
    for index, item in enumerate(manifest['entries']):
        if not isinstance(item, dict):
            fail('manifest entry must be an object', index=index)
            continue
        try:
            path = item['path']
            relative_file(root, path)
            archive = item['archive_path']
            relative_file(root, archive)
            kind = KINDS.get(str(item['kind']).lower())
            if kind is None:
                raise ValueError('kind must be py/python or md/markdown')
            if not path.endswith('.py' if kind == 'PY' else '.md'):
                raise ValueError('path extension disagrees with kind')
            if not isinstance(item['sha'], str) or not re.fullmatch('[0-9a-f]{40}', item['sha']):
                raise ValueError('sha must be a 40-character Git blob SHA')
            if type(item['size']) is not int or item['size'] < 0:
                raise ValueError('size must be a nonnegative integer')
            if not isinstance(item.get('anchor'), str) or not item['anchor']:
                raise ValueError('anchor must be a nonempty string')
            if not re.fullmatch(r'[A-Za-z0-9_-]+', item['anchor']):
                raise ValueError('anchor must be a plain HTML id without a leading #')
            if item['anchor'] in anchors:
                raise ValueError('duplicate manifest anchor')
            if path in expected:
                fail('duplicate manifest source path', path=path, index=index)
                continue
            expected[path] = dict(item, marker_kind=kind)
            anchors.add(item['anchor'])
            archives.add(archive)
        except (KeyError, TypeError, ValueError) as exc:
            fail('invalid manifest entry', index=index, detail=str(exc))
    # Inspect sibling archive volumes too, so unlisted extra source blocks fail.
    for archive in list(archives):
        parent = relative_file(root, archive).parent
        for sibling in parent.glob('*.md'):
            archives.add(sibling.relative_to(root).as_posix())
    seen = Counter()
    verified = Counter()
    for archive in sorted(archives):
        try:
            data = relative_file(root, archive).read_bytes()
        except (OSError, ValueError) as exc:
            fail('cannot read archive', archive=archive, detail=str(exc))
            continue
        tokens = list(TOKEN.finditer(data))
        literal_count = sum(data.count(x) for x in (
            b'<!-- BEGIN_PY_SOURCE', b'<!-- END_PY_SOURCE',
            b'<!-- BEGIN_MD_SOURCE', b'<!-- END_MD_SOURCE'))
        if literal_count != len(tokens):
            fail('malformed source delimiter', archive=archive)
        previous_end = 0
        for pos, token in enumerate(tokens):
            if token.group(1) == b'END':
                if pos == 0 or tokens[pos - 1].group(1) != b'BEGIN':
                    fail('unpaired end marker', archive=archive, offset=token.start())
                continue
            begin = BEGIN.fullmatch(token.group())
            if begin is None:
                fail('malformed begin marker', archive=archive, offset=token.start())
                continue
            try:
                path = begin.group(2).decode('utf-8')
                sha = begin.group(3).decode('ascii')
                kind = begin.group(1).decode('ascii')
            except UnicodeDecodeError:
                fail('invalid marker encoding', archive=archive)
                continue
            seen[path] += 1
            meta = expected.get(path)
            before_errors = len(issues)
            if meta is None:
                fail('source block absent from manifest', archive=archive, path=path)
            else:
                for actual, wanted, field in ((archive, meta['archive_path'], 'archive_path'),
                                             (kind, meta['marker_kind'], 'kind'),
                                             (sha, meta['sha'], 'marker SHA')):
                    if actual != wanted:
                        fail('source metadata mismatch', archive=archive, path=path, field=field)
            if pos + 1 >= len(tokens) or tokens[pos + 1].group(1) != b'END':
                fail('missing end marker or nested begin marker', archive=archive, path=path)
                continue
            ending = tokens[pos + 1]
            end = END.fullmatch(ending.group())
            if end is None or end.group(1) != begin.group(1) or end.group(2) != begin.group(2):
                fail('end marker does not match begin marker', archive=archive, path=path)
            header = data[previous_end:token.start()]
            previous_end = ending.end()
            source_anchors = re.findall(rb'<a (?:name|id)="([^"]+)"></a>', header)
            if meta and source_anchors != [meta['anchor'].encode('ascii')]:
                fail('missing, duplicate or wrong HTML source anchor', archive=archive, path=path)
            headers = list(HEADER.finditer(header))
            if len(headers) != 1:
                fail('missing or ambiguous source byte/SHA header', archive=archive, path=path)
            else:
                shas = re.findall(rb'`([0-9a-fA-F]{40})`', headers[0].group())
                sizes = list(SIZE.finditer(headers[0].group()))
                if len(shas) != 1 or len(sizes) != 1:
                    fail('malformed source byte/SHA header', archive=archive, path=path)
                elif meta:
                    declared = int(next(g for g in sizes[0].groups() if g is not None))
                    if shas[0].decode('ascii') != meta['sha'] or declared != meta['size']:
                        fail('source byte/SHA header disagrees with manifest', archive=archive, path=path)
            wrapped = data[token.end():ending.start()]
            openings, closings = list(OPEN.finditer(wrapped)), list(CLOSE.finditer(wrapped))
            if len(openings) != 1 or len(closings) != 1 or openings[0].end() > closings[0].start():
                fail('missing, duplicate or malformed eight-backtick fence', archive=archive, path=path)
                continue
            opener, closer = openings[0], closings[0]
            language = b'python' if kind == 'PY' else b'markdown'
            if opener.group(1) != language or wrapped[:opener.start()].strip() or wrapped[closer.end():].strip():
                fail('fence language or surrounding content is invalid', archive=archive, path=path)
            raw = wrapped[opener.end():closer.start()]
            candidates = [raw]
            if raw.endswith(b'\n'):
                candidates.append(raw[:-1])
            if raw.endswith(b'\r\n'):
                candidates.append(raw[:-2])
            if meta and not any(len(p) == meta['size'] and blob_sha(p) == meta['sha'] for p in candidates):
                fail('source bytes do not match manifest Git blob', archive=archive, path=path,
                     candidate_sizes=[len(p) for p in candidates])
            if meta and len(issues) == before_errors:
                verified[kind] += 1
    for path in sorted(set(expected) - set(seen)):
        fail('missing source block', path=path)
    for path, count in sorted(seen.items()):
        if count != 1:
            fail('duplicate source block', path=path, count=count)
    return {
        'ok': not issues,
        'source_commit': manifest.get('source_commit'),
        'main_base_commit': manifest.get('main_base_commit'),
        'manifest_entries': len(manifest['entries']),
        'unique_source_blocks': len(seen),
        'verified_python': verified['PY'],
        'verified_markdown': verified['MD'],
        'errors': issues,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', default='.', help='repository root (default: current directory)')
    parser.add_argument('--manifest', default='docs/lossless_source_manifest.json', help='manifest path relative to root')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    try:
        result = verify(root, relative_file(root, args.manifest))
    except (ValueError, OSError) as exc:
        result = {'ok': False, 'errors': [{'error': str(exc)}]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())

```

<a name="section-a0fe31ec5c73"></a>

## Appendix A — Complete Source Index for All 327 Python Files

Each link opens the **complete original source file**. The role column is a short index description derived from its module docstring or actual top-level definitions and test names. Execution flow is explained in the corresponding Project and Phase above. Package initializers, CLI scripts, helpers, regression tests, and empty placeholders are all included.

| Complete original source | Role / entry points | Original bytes |
|---|---|---:|
| [backend/__init__.py](technical_sources/PYTHON_SOURCE_01.md#py-001) | Local-only HTTP transport for the GUI-1.0 HUNL game service. | 67 |
| [backend/game_router.py](technical_sources/PYTHON_SOURCE_01.md#py-002) | HTTP adapter for GUI-1.0 sessions: no poker rules implemented here | 6,122 |
| [backend/main.py](technical_sources/PYTHON_SOURCE_01.md#py-003) | Local FastAPI app for Heads-Up GUI and Tournament ENV Phase 1.1 | 2,189 |
| [backend/schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-004) | Strict API request and intentionally whitelisted response models | 4,556 |
| [backend/tournament_router.py](technical_sources/PYTHON_SOURCE_01.md#py-005) | HTTP routes for Tournament ENV 1.1; separate from the HUNL200 routes. | 6,543 |
| [backend/tournament_schemas.py](technical_sources/PYTHON_SOURCE_01.md#py-006) | Strict, whitelisted API contract for Tournament ENV Phase 1.1. | 4,253 |
| [main.py](technical_sources/PYTHON_SOURCE_01.md#py-007) | Play independent heads-up hands with corrected positions by default. | 1,480 |
| [scripts/audit_corrected.py](technical_sources/PYTHON_SOURCE_01.md#py-008) | uv run python scripts/audit_corrected.py --hands 100 --seed 42 | 1,903 |
| [scripts/audit_multiplayer.py](technical_sources/PYTHON_SOURCE_01.md#py-009) | Quick random-benchmark smoke audit | 1,888 |
| [scripts/audit_rlcard.py](technical_sources/PYTHON_SOURCE_01.md#py-010) | Run against installed RLCard: uv run python scripts/audit_rlcard.py --hands 100 | 1,978 |
| [scripts/audit_tournament.py](technical_sources/PYTHON_SOURCE_01.md#py-011) | Single-table tournament smoke audit; run `uv run python scripts/audit_tournament.py`. | 2,399 |
| [scripts/expand_phase01_smoke.py](technical_sources/PYTHON_SOURCE_01.md#py-012) | Real-engine Expand Phase 1 acceptance smoke: each table target 2..6 seats | 5,137 |
| [scripts/expand_phase02_smoke.py](technical_sources/PYTHON_SOURCE_02.md#py-013) | Expand E2: real-engine 2..6 person acting observation & action smoke | 4,124 |
| [scripts/expand_phase03_evaluate.py](technical_sources/PYTHON_SOURCE_02.md#py-014) | Independent optional E3 heldout benchmark; never changes weights. | 1,497 |
| [scripts/expand_phase03_train.py](technical_sources/PYTHON_SOURCE_02.md#py-015) | Expand Phase 3: 2-6 player PPO self-play on the existing cash game engine | 4,063 |
| [scripts/expand_phase04_evaluate.py](technical_sources/PYTHON_SOURCE_02.md#py-016) | Read-only benchmark of a Phase4 blueprint against a uniform legal agent. | 1,280 |
| [scripts/expand_phase04_mccfr.py](technical_sources/PYTHON_SOURCE_02.md#py-017) | Expand Phase4 reference MCCFR | 5,062 |
| [scripts/expand_phase05_evaluate.py](technical_sources/PYTHON_SOURCE_02.md#py-018) | Paired-deal toy benchmark: search vs the SAME immutable Phase4 blueprint | 4,121 |
| [scripts/expand_phase05_search.py](technical_sources/PYTHON_SOURCE_02.md#py-019) | Phase5 read-only toy / Hold'em search benchmark | 3,918 |
| [scripts/expand_phase06_benchmark.py](technical_sources/PYTHON_SOURCE_02.md#py-020) | Seed-blocked synthetic change-point diagnostic: NOT poker win-rate testing | 6,571 |
| [scripts/expand_phase07_benchmark.py](technical_sources/PYTHON_SOURCE_02.md#py-021) | E7 no-training cash/tournament benchmark; never overwrites reports. | 5,102 |
| [scripts/gui10_smoke.py](technical_sources/PYTHON_SOURCE_02.md#py-022) | Transport-free GUI-1.0 HUNL smoke run; human plays check/call by default | 1,820 |
| [scripts/gui11_smoke.py](technical_sources/PYTHON_SOURCE_02.md#py-023) | HTTP smoke exercise without starting an external server | 2,129 |
| [scripts/gui14_smoke.py](technical_sources/PYTHON_SOURCE_02.md#py-024) | Smoke a trusted local Phase 15/16 neural checkpoint through GUI FastAPI | 2,785 |
| [scripts/gui15_smoke.py](technical_sources/PYTHON_SOURCE_03.md#py-025) | Exercise GUI-1.5 HTTP end-to-end without launching React/browser | 3,671 |
| [scripts/gui16_smoke.py](technical_sources/PYTHON_SOURCE_03.md#py-026) | Phase GUI-1.6 local HTTP multi-hand carryover smoke | 2,543 |
| [scripts/phase10_train.py](technical_sources/PYTHON_SOURCE_03.md#py-027) | Phase 10 experimental Q-Boost PPO; optional matched-start PPO comparator | 8,433 |
| [scripts/phase11_benchmark.py](technical_sources/PYTHON_SOURCE_03.md#py-028) | Measure inference batching gains without PPO optimizer steps. | 2,925 |
| [scripts/phase11_compare.py](technical_sources/PYTHON_SOURCE_03.md#py-029) | Compare separate PPO and QBoost JSONL logs at matched update counts | 4,163 |
| [scripts/phase11_train.py](technical_sources/PYTHON_SOURCE_03.md#py-030) | Phase 11 training CLI with real-time progress, time-left and ETA | 4,449 |
| [scripts/phase12_evaluate.py](technical_sources/PYTHON_SOURCE_03.md#py-031) | Phase12 independent paired-deck evaluator for Phase9/10/11 local checkpoints | 5,139 |
| [scripts/phase13_benchmark.py](technical_sources/PYTHON_SOURCE_03.md#py-032) | Measure ACTUAL training update throughput for each hands/update + env batch pair | 4,229 |
| [scripts/phase13_compare.py](technical_sources/PYTHON_SOURCE_03.md#py-033) | Compare Phase13 PPO and QBoost training diagnostics without claiming GTO | 4,027 |
| [scripts/phase13_train.py](technical_sources/PYTHON_SOURCE_03.md#py-034) | Phase 13: PPO/QBoost batch scaling, checkpoint lineage, live tqdm ETA | 4,257 |
| [scripts/phase14_study.py](technical_sources/PYTHON_SOURCE_03.md#py-035) | Multi-seed controlled PPO/QBoost HUNL experiment | 3,502 |
| [scripts/phase15_critic_audit.py](technical_sources/PYTHON_SOURCE_03.md#py-036) | Run a read-only diagnostic for two trusted Phase-15 HUNL PPO checkpoints | 3,142 |
| [scripts/phase15_finalize.py](technical_sources/PYTHON_SOURCE_04.md#py-037) | Phase 15 independent held-out evaluation, run ONCE after training is finished. | 639 |
| [scripts/phase15_initial_policy_comparison.py](technical_sources/PYTHON_SOURCE_04.md#py-038) | Read-only baseline action comparison for HUNL self-play, Phase 15 | 3,227 |
| [scripts/phase15_policy_audit.py](technical_sources/PYTHON_SOURCE_04.md#py-039) | CLI: fixed-state policy audit and existing PPO metrics review | 2,491 |
| [scripts/phase15_preflop_sequence_audit.py](technical_sources/PYTHON_SOURCE_04.md#py-040) | Read-only HUNL Phase15 preflop action-sequence audit | 2,636 |
| [scripts/phase15_terminal_reward_audit.py](technical_sources/PYTHON_SOURCE_04.md#py-041) | Read-only HUNL PPO per-hand reward tails and preflop exit audit | 2,694 |
| [scripts/phase15_train.py](technical_sources/PYTHON_SOURCE_04.md#py-042) | Phase 15 HUNL long-run training, from a Phase14 best.pt (or trusted PPO/QBoost checkpoint). | 2,089 |
| [scripts/phase16_evaluate.py](technical_sources/PYTHON_SOURCE_04.md#py-043) | No-training Phase16 benchmark for Phase14 and future Phase15 best.pt actors. | 6,132 |
| [scripts/phase16_ppo_ablation.py](technical_sources/PYTHON_SOURCE_04.md#py-044) | Phase16 isolated A0/A1/A2/A3 PPO ablation (NEW RUN ONLY). | 3,897 |
| [scripts/phase16_ppo_stress.py](technical_sources/PYTHON_SOURCE_04.md#py-045) | Phase16 long PPO A0/A3 stress, full-state resume, and paired comparison | 6,453 |
| [scripts/phase17_evaluate.py](technical_sources/PYTHON_SOURCE_04.md#py-046) | Phase 17 variable-stack continuous-match evaluation (no training) | 10,446 |
| [scripts/phase17_gpu_retrain.py](technical_sources/PYTHON_SOURCE_04.md#py-047) | Five-hour Phase17 GPU retraining: new run by default; resume explicitly. | 3,573 |
| [scripts/phase5_rlcard_smoke.py](technical_sources/PYTHON_SOURCE_04.md#py-048) | Run from repository root: uv run python scripts/phase5_rlcard_smoke.py | 880 |
| [scripts/phase6_benchmark.py](technical_sources/PYTHON_SOURCE_05.md#py-049) | CPU-only Phase 6 HUNL200 benchmark; no PyTorch/CUDA required | 3,422 |
| [scripts/phase6_hardware_probe.py](technical_sources/PYTHON_SOURCE_05.md#py-050) | Safe, tiny PyTorch CUDA compute probe for Pascal-class GPUs (optional). | 1,664 |
| [scripts/phase7_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-051) | Run from repo root: uv run --extra gpu python -m scripts.phase7_smoke --hands 100 --device cuda No policy optimization occurs in Phase  | 4,254 |
| [scripts/phase8_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-052) | Phase-8 integration smoke: real Phase-6 rollout -> Phase-7 encoders -> trainable nets | 6,235 |
| [scripts/phase9_train.py](technical_sources/PYTHON_SOURCE_05.md#py-053) | CLI: Phase-9 actual PPO updates, checkpoint/resume, random baseline evaluation | 4,240 |
| [scripts/run_tournament.py](technical_sources/PYTHON_SOURCE_05.md#py-054) | Run one 2-6 seat tournament (random-agent baseline) and show eliminations. | 1,534 |
| [scripts/tournament_env131_diagnose.py](technical_sources/PYTHON_SOURCE_05.md#py-055) | Show conditional preflop heuristic raise mix (NOT actual in-game raise frequency) | 1,557 |
| [scripts/tournament_env13_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-056) | Reproducible six-seat Tournament ENV 1.3 bot lineup smoke, no HTTP server. | 1,499 |
| [scripts/tournament_env14_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-057) | Real trusted Expand Phase3 checkpoint smoke for Tournament ENV 1.4 | 1,760 |
| [scripts/tournament_env17_release_audit.py](technical_sources/PYTHON_SOURCE_05.md#py-058) | Offline Git safety audit for Tournament ENV 1.7 | 5,498 |
| [scripts/tournament_env1_smoke.py](technical_sources/PYTHON_SOURCE_05.md#py-059) | API smoke: uv run --extra dev --extra gui python -m scripts.tournament_env1_smoke | 1,891 |
| [src/holdem_ai/__init__.py](technical_sources/PYTHON_SOURCE_05.md#py-060) | No-Limit Texas Hold'em AI platform. | 65 |
| [src/holdem_ai/agents/__init__.py](technical_sources/PYTHON_SOURCE_06.md#py-061) | Policy and agent implementations. | 40 |
| [src/holdem_ai/agents/base_agent.py](technical_sources/PYTHON_SOURCE_06.md#py-062) | Common interface for poker agents | 460 |
| [src/holdem_ai/agents/cfr_agent.py](technical_sources/PYTHON_SOURCE_06.md#py-063) | Empty file; no executable definitions. | 0 |
| [src/holdem_ai/agents/diagnostic_opponents.py](technical_sources/PYTHON_SOURCE_06.md#py-064) | Deterministic benchmark opponents; all actions are legal by construction | 2,418 |
| [src/holdem_ai/agents/neural_policy.py](technical_sources/PYTHON_SOURCE_06.md#py-065) | Inference-only adapter implementing the Phase-5 AgentPolicy interface. | 1,680 |
| [src/holdem_ai/agents/policies.py](technical_sources/PYTHON_SOURCE_06.md#py-066) | Mask-correct policies usable by random, CFR and neural learners. | 3,504 |
| [src/holdem_ai/agents/random_agent.py](technical_sources/PYTHON_SOURCE_06.md#py-067) | Random legal-action baseline, for testing the environment/agent boundary. | 518 |
| [src/holdem_ai/agents/rl_agent.py](technical_sources/PYTHON_SOURCE_06.md#py-068) | Empty file; no executable definitions. | 0 |
| [src/holdem_ai/core/__init__.py](technical_sources/PYTHON_SOURCE_06.md#py-069) | Consolidated functional package. | 39 |
| [src/holdem_ai/core/schema.py](technical_sources/PYTHON_SOURCE_06.md#py-070) | Framework-neutral, seat-relative agent decision contract (Python >= 3.11). | 3,547 |
| [src/holdem_ai/encoding/__init__.py](technical_sources/PYTHON_SOURCE_06.md#py-071) | Consolidated functional package. | 39 |
| [src/holdem_ai/encoding/actions.py](technical_sources/PYTHON_SOURCE_06.md#py-072) | Phase-5 compatible fixed 7-head action encoding and masked distributions | 1,632 |
| [src/holdem_ai/encoding/observation.py](technical_sources/PYTHON_SOURCE_07.md#py-073) | HUNL200 VRPO-Lite v1 actor/centralized-critic encoders | 8,671 |
| [src/holdem_ai/encoding/tensors.py](technical_sources/PYTHON_SOURCE_07.md#py-074) | Optional PyTorch adapters; importable without torch until batching is requested. | 2,352 |
| [src/holdem_ai/encoding/vocabulary.py](technical_sources/PYTHON_SOURCE_07.md#py-075) | Stable VRPO-Lite v1 token vocabulary (NOT the original VRPO paper encoding). | 1,074 |
| [src/holdem_ai/environment/__init__.py](technical_sources/PYTHON_SOURCE_07.md#py-076) | Poker hand and tournament environments. | 46 |
| [src/holdem_ai/environment/actions.py](technical_sources/PYTHON_SOURCE_07.md#py-077) | Independent no-limit betting constraints (amounts are integer chips) | 3,395 |
| [src/holdem_ai/environment/bet_abstraction.py](technical_sources/PYTHON_SOURCE_07.md#py-078) | Optional integer-chip action abstraction for engines with arbitrary raise-to amounts | 5,821 |
| [src/holdem_ai/environment/blind_manager.py](technical_sources/PYTHON_SOURCE_07.md#py-079) | Deterministic hand-count blind schedule for single-table tournaments | 2,482 |
| [src/holdem_ai/environment/corrected_game.py](technical_sources/PYTHON_SOURCE_07.md#py-080) | RLCard 1.2.0 game subclass fixing heads-up seats and betting round | 5,690 |
| [src/holdem_ai/environment/corrected_round.py](technical_sources/PYTHON_SOURCE_07.md#py-081) | Two-player no-limit betting round for the RLCard 1.2.0 action abstraction | 5,593 |
| [src/holdem_ai/environment/game_state.py](technical_sources/PYTHON_SOURCE_07.md#py-082) | Empty file; no executable definitions. | 0 |
| [src/holdem_ai/environment/hand_engine.py](technical_sources/PYTHON_SOURCE_07.md#py-083) | One-hand HUNL interface | 4,869 |
| [src/holdem_ai/environment/hand_evaluator.py](technical_sources/PYTHON_SOURCE_07.md#py-084) | Independent exact seven-card Hold'em evaluator for showdown verification | 3,274 |
| [src/holdem_ai/environment/hunl200_adapter.py](technical_sources/PYTHON_SOURCE_08.md#py-085) | HUNL200 adapter: real Phase-3 hand engine -> Phase-5 turn contract | 8,695 |
| [src/holdem_ai/environment/multi_player_engine.py](technical_sources/PYTHON_SOURCE_08.md#py-086) | Standalone 2-6 player no-limit Hold'em *hand* environment | 17,555 |
| [src/holdem_ai/environment/multi_player_seats.py](technical_sources/PYTHON_SOURCE_08.md#py-087) | Moving-button seat and action order for 2-6 seat Texas Hold'em hands | 2,113 |
| [src/holdem_ai/environment/observation.py](technical_sources/PYTHON_SOURCE_08.md#py-088) | Empty file; no executable definitions. | 0 |
| [src/holdem_ai/environment/player_manager.py](technical_sources/PYTHON_SOURCE_08.md#py-089) | Persistent chip stacks and elimination ranks across tournament hands | 2,666 |
| [src/holdem_ai/environment/pot_manager.py](technical_sources/PYTHON_SOURCE_08.md#py-090) | Reference side-pot construction and settlement with chip conservation | 3,297 |
| [src/holdem_ai/environment/rewards.py](technical_sources/PYTHON_SOURCE_08.md#py-091) | Hand-level chip rewards and terminal tournament placement rewards. | 1,789 |
| [src/holdem_ai/environment/rlcard_adapter.py](technical_sources/PYTHON_SOURCE_08.md#py-092) | Conservative RLCard Env adapter | 7,021 |
| [src/holdem_ai/environment/rules_audit.py](technical_sources/PYTHON_SOURCE_08.md#py-093) | Observational, non-mutating audit of installed RLCard HUNL baseline | 1,782 |
| [src/holdem_ai/environment/seat_manager.py](technical_sources/PYTHON_SOURCE_08.md#py-094) | Rule reference for heads-up seating and betting order | 1,643 |
| [src/holdem_ai/environment/tournament_env.py](technical_sources/PYTHON_SOURCE_08.md#py-095) | Single-table 2-6 seat knockout tournament over the Phase 3 hand engine | 10,191 |
| [src/holdem_ai/evaluation/__init__.py](technical_sources/PYTHON_SOURCE_08.md#py-096) | Game and tournament benchmarks. | 38 |
| [src/holdem_ai/evaluation/action_diagnostics.py](technical_sources/PYTHON_SOURCE_09.md#py-097) | Action-rate and legal-opportunity diagnostics on an on-policy trajectory | 4,003 |
| [src/holdem_ai/evaluation/balanced.py](technical_sources/PYTHON_SOURCE_09.md#py-098) | Seat/button-balanced held-out tests vs Random and a frozen anchor Actor. | 3,092 |
| [src/holdem_ai/evaluation/benchmark.py](technical_sources/PYTHON_SOURCE_09.md#py-099) | Empty file; no executable definitions. | 0 |
| [src/holdem_ai/evaluation/checkpoint_loader.py](technical_sources/PYTHON_SOURCE_09.md#py-100) | Read trusted Phase-9/10/11 checkpoint or Phase-11 anchor, actor ONLY. | 950 |
| [src/holdem_ai/evaluation/continuous_stack.py](technical_sources/PYTHON_SOURCE_09.md#py-101) | Phase 17: matched-seed, variable-stack HUNL continuous-match evaluation | 14,714 |
| [src/holdem_ai/evaluation/critic_diagnostics.py](technical_sources/PYTHON_SOURCE_09.md#py-102) | Read-only HUNL PPO critic calibration diagnostics | 11,696 |
| [src/holdem_ai/evaluation/fixed_state_policy_audit.py](technical_sources/PYTHON_SOURCE_09.md#py-103) | Evaluate frozen HUNL actor checkpoints on exactly the same actor-visible states | 6,773 |
| [src/holdem_ai/evaluation/initial_policy_comparison.py](technical_sources/PYTHON_SOURCE_09.md#py-104) | Read-only Agent comparison: uniform legal, random-init, trained Phase14 | 9,620 |
| [src/holdem_ai/evaluation/opponent_pool.py](technical_sources/PYTHON_SOURCE_09.md#py-105) | Phase16 diagnostic opponent-pool benchmark, independent of Phase15 training | 6,852 |
| [src/holdem_ai/evaluation/ppo_metric_audit.py](technical_sources/PYTHON_SOURCE_09.md#py-106) | Read-only Phase 15 PPO optimizer diagnostics from existing JSONL logs. | 1,906 |
| [src/holdem_ai/evaluation/preflop_sequence_audit.py](technical_sources/PYTHON_SOURCE_09.md#py-107) | Read-only HUNL agent preflop decision-sequence diagnostics | 11,220 |
| [src/holdem_ai/evaluation/reward_diagnostics.py](technical_sources/PYTHON_SOURCE_09.md#py-108) | Summarize on-policy HUNL terminal rewards and termination actions per update | 1,717 |
| [src/holdem_ai/evaluation/seed_blocked.py](technical_sources/PYTHON_SOURCE_10.md#py-109) | Independent seed-blocked, four-layout HUNL200 evaluation (Phase 12) | 10,961 |
| [src/holdem_ai/evaluation/seed_statistics.py](technical_sources/PYTHON_SOURCE_10.md#py-110) | Training-seed-level statistical estimates for fixed independent test-deck banks | 1,889 |
| [src/holdem_ai/evaluation/strategy_diagnostics.py](technical_sources/PYTHON_SOURCE_10.md#py-111) | Actor-observation-only behavioral diagnostics for evaluated HUNL200 hands. | 5,641 |
| [src/holdem_ai/evaluation/strategy_report.py](technical_sources/PYTHON_SOURCE_10.md#py-112) | Obsidian-compatible Phase16 benchmark report from JSON-safe results. | 4,153 |
| [src/holdem_ai/evaluation/terminal_reward_audit.py](technical_sources/PYTHON_SOURCE_10.md#py-113) | Read-only per-HAND payoff and preflop-exit audit for HUNL PPO | 12,130 |
| [src/holdem_ai/evaluation/tournament_eval.py](technical_sources/PYTHON_SOURCE_10.md#py-114) | Empty file; no executable definitions. | 0 |
| [src/holdem_ai/expand/__init__.py](technical_sources/PYTHON_SOURCE_10.md#py-115) | 6-Max expansion modules | 77 |
| [src/holdem_ai/expand/environment/__init__.py](technical_sources/PYTHON_SOURCE_10.md#py-116) | Expand Phase 1 public table API. | 561 |
| [src/holdem_ai/expand/environment/cash_rules.py](technical_sources/PYTHON_SOURCE_10.md#py-117) | E1 cash table admission rules: fixed blinds, no rake, wait for BB. | 1,073 |
| [src/holdem_ai/expand/environment/event_ledger.py](technical_sources/PYTHON_SOURCE_10.md#py-118) | Append-only deterministic journal; privileged seed log, not an actor observation. | 1,623 |
| [src/holdem_ai/expand/environment/multiway_adapter.py](technical_sources/PYTHON_SOURCE_10.md#py-119) | Late-bound adapter to the unchanged Base Phase 3 6-seat hand engine. | 943 |
| [src/holdem_ai/expand/environment/table_manager.py](technical_sources/PYTHON_SOURCE_10.md#py-120) | Expand Phase 1: dynamic 6-seat cash sessions on the unchanged Phase 3 hand engine | 22,864 |
| [src/holdem_ai/expand/environment/table_state.py](technical_sources/PYTHON_SOURCE_11.md#py-121) | Public, immutable table metadata; does not contain hole cards or hidden deck. | 3,204 |
| [src/holdem_ai/expand/environment/tournament_rules.py](technical_sources/PYTHON_SOURCE_11.md#py-122) | Protocol boundary for E7; E1 deliberately does not simulate tournaments. | 589 |
| [src/holdem_ai/expand/information/__init__.py](technical_sources/PYTHON_SOURCE_11.md#py-123) | Expand Phase 2 acting-player information, action translation and encoding. | 1,010 |
| [src/holdem_ai/expand/information/actions.py](technical_sources/PYTHON_SOURCE_11.md#py-124) | Exact legality-preserving discrete action translation, with no silent clamp. | 3,375 |
| [src/holdem_ai/expand/information/adapter.py](technical_sources/PYTHON_SOURCE_11.md#py-125) | Phase2 interface for Phase1 DynamicTable; records public action street exactly | 3,795 |
| [src/holdem_ai/expand/information/builder.py](technical_sources/PYTHON_SOURCE_11.md#py-126) | Construct an InformationState only from acting-player public observation | 6,109 |
| [src/holdem_ai/expand/information/encoder.py](technical_sources/PYTHON_SOURCE_11.md#py-127) | Stable, deterministic 6-seat tensor-ready encoding (Python only). | 3,985 |
| [src/holdem_ai/expand/information/policy_v2.py](technical_sources/PYTHON_SOURCE_11.md#py-128) | Small, trainable multi-seat neural policy | 4,370 |
| [src/holdem_ai/expand/information/schema.py](technical_sources/PYTHON_SOURCE_11.md#py-129) | Expand Phase 2: allowlisted, immutable acting-player-only information | 2,902 |
| [src/holdem_ai/expand/integration/__init__.py](technical_sources/PYTHON_SOURCE_11.md#py-130) | Package initialization and public exports. | 703 |
| [src/holdem_ai/expand/integration/agent.py](technical_sources/PYTHON_SOURCE_11.md#py-131) | E7 routing: strictly actor-visible InformationState -> one legal Phase1 Action | 7,108 |
| [src/holdem_ai/expand/integration/benchmark.py](technical_sources/PYTHON_SOURCE_11.md#py-132) | E7 reproducible *independent-session* cash and tournament benchmarks | 11,947 |
| [src/holdem_ai/expand/integration/manifest.py](technical_sources/PYTHON_SOURCE_12.md#py-133) | Reproducible experiment metadata; no checkpoint modification. | 2,055 |
| [src/holdem_ai/expand/integration/source_bridges.py](technical_sources/PYTHON_SOURCE_12.md#py-134) | Opt-in E4/E5 source bridges with explicit infoset/belief contract | 2,477 |
| [src/holdem_ai/expand/integration/tournament.py](technical_sources/PYTHON_SOURCE_12.md#py-135) | Single-table 2-6 seat tournament adapter over Base Phase3 Hand Engine | 12,012 |
| [src/holdem_ai/expand/opponent/__init__.py](technical_sources/PYTHON_SOURCE_12.md#py-136) | Expand Phase 6: public-only per-tenure opponents and bounded adaptation. | 948 |
| [src/holdem_ai/expand/opponent/adaptive.py](technical_sources/PYTHON_SOURCE_12.md#py-137) | Trust-region gate around an already valid public-information policy | 5,634 |
| [src/holdem_ai/expand/opponent/bridge.py](technical_sources/PYTHON_SOURCE_12.md#py-138) | Phase1/Phase2 public event bridge, called ONLY after a legal action commits | 4,284 |
| [src/holdem_ai/expand/opponent/model.py](technical_sources/PYTHON_SOURCE_12.md#py-139) | Per-player Bayesian-smoothed decayed action model; only public observations | 9,149 |
| [src/holdem_ai/expand/opponent/persistence.py](technical_sources/PYTHON_SOURCE_12.md#py-140) | Deterministic allowlisted JSON checkpoint, not Python pickle | 5,976 |
| [src/holdem_ai/expand/opponent/router.py](technical_sources/PYTHON_SOURCE_12.md#py-141) | Optional model-conditioned routing among PRE-EXISTING hero policies | 2,247 |
| [src/holdem_ai/expand/opponent/schema.py](technical_sources/PYTHON_SOURCE_12.md#py-142) | Allowlisted public-only events and fixed-cardinality legal-action context | 4,004 |
| [src/holdem_ai/expand/pluribus/__init__.py](technical_sources/PYTHON_SOURCE_12.md#py-143) | Pluribus-inspired research branch; NOT Pluribus original software. | 73 |
| [src/holdem_ai/expand/pluribus/abstraction/__init__.py](technical_sources/PYTHON_SOURCE_12.md#py-144) | Package initialization and public exports. | 111 |
| [src/holdem_ai/expand/pluribus/abstraction/cards.py](technical_sources/PYTHON_SOURCE_13.md#py-145) | Reproducible (deliberately *untrained*) toy 6-max hold'em abstraction | 2,718 |
| [src/holdem_ai/expand/pluribus/blueprint/__init__.py](technical_sources/PYTHON_SOURCE_13.md#py-146) | Package initialization and public exports. | 380 |
| [src/holdem_ai/expand/pluribus/blueprint/checkpoint.py](technical_sources/PYTHON_SOURCE_13.md#py-147) | Atomic gzipped JSON MCCFR checkpoints; no pickle or torch.load. | 3,419 |
| [src/holdem_ai/expand/pluribus/blueprint/evaluate.py](technical_sources/PYTHON_SOURCE_13.md#py-148) | Separate deterministic evaluation seeds; no trainer updates in evaluation. | 2,020 |
| [src/holdem_ai/expand/pluribus/blueprint/holdem.py](technical_sources/PYTHON_SOURCE_13.md#py-149) | Small-resource 2--6 seat Hold'em MCCFR adapter using the Phase3 engine | 4,709 |
| [src/holdem_ai/expand/pluribus/blueprint/kuhn_reference.py](technical_sources/PYTHON_SOURCE_13.md#py-150) | Exact 2-player Kuhn exploitability (unabstracted 3-card game only) | 2,042 |
| [src/holdem_ai/expand/pluribus/blueprint/mccfr.py](technical_sources/PYTHON_SOURCE_13.md#py-151) | Multi-player external-sampling MCCFR with linear weights and atomic iterations | 10,626 |
| [src/holdem_ai/expand/pluribus/blueprint/toy_games.py](technical_sources/PYTHON_SOURCE_13.md#py-152) | Tiny 2/3-player toy poker games for reference testing MCCFR | 6,048 |
| [src/holdem_ai/expand/pluribus/search/__init__.py](technical_sources/PYTHON_SOURCE_13.md#py-153) | Expand Phase5 bounded imperfect-information search prototype. | 558 |
| [src/holdem_ai/expand/pluribus/search/belief.py](technical_sources/PYTHON_SOURCE_13.md#py-154) | Public-knowledge-only roots for toy poker and collision-free Hold'em sampling. | 8,095 |
| [src/holdem_ai/expand/pluribus/search/continuations.py](technical_sources/PYTHON_SOURCE_13.md#py-155) | Four Pluribus-inspired continuation policy families | 1,934 |
| [src/holdem_ai/expand/pluribus/search/nested.py](technical_sources/PYTHON_SOURCE_13.md#py-156) | Public-trace session for re-searching at each decision / off-tree action | 1,862 |
| [src/holdem_ai/expand/pluribus/search/solver.py](technical_sources/PYTHON_SOURCE_14.md#py-157) | Bounded belief-conditioned external-sampling MCCFR with rollout leaf values | 9,865 |
| [src/holdem_ai/expand/rl/__init__.py](technical_sources/PYTHON_SOURCE_14.md#py-158) | Expand Phase 3: multiagent PPO on a public-information six-seat game. | 782 |
| [src/holdem_ai/expand/rl/collector.py](technical_sources/PYTHON_SOURCE_14.md#py-159) | Real-hand multiagent on-policy self-play on E1 table + E2 allowlisted adapter | 10,450 |
| [src/holdem_ai/expand/rl/model.py](technical_sources/PYTHON_SOURCE_14.md#py-160) | Expand 3: strictly actor-visible shared multi-seat Actor + Value networks. | 2,440 |
| [src/holdem_ai/expand/rl/population.py](technical_sources/PYTHON_SOURCE_14.md#py-161) | Training-time opponent selection, separate from final evaluation opponents. | 2,015 |
| [src/holdem_ai/expand/rl/ppo.py](technical_sources/PYTHON_SOURCE_14.md#py-162) | Clipped PPO on frozen collected 2-6 player trajectories. | 3,780 |
| [src/holdem_ai/expand/rl/returns.py](technical_sources/PYTHON_SOURCE_14.md#py-163) | Per-hand, per-seat GAE: decision streams are never conflated across players. | 2,513 |
| [src/holdem_ai/expand/rl/trainer.py](technical_sources/PYTHON_SOURCE_14.md#py-164) | Reproducible E3 PPO training, snapshots, safe resume and validation gates | 16,291 |
| [src/holdem_ai/experiments/__init__.py](technical_sources/PYTHON_SOURCE_14.md#py-165) | Consolidated functional package. | 39 |
| [src/holdem_ai/experiments/batch_scaling.py](technical_sources/PYTHON_SOURCE_14.md#py-166) | Phase 13 training-batch experiments, preserving Phase 9/10 checkpoints | 17,194 |
| [src/holdem_ai/experiments/controlled_study.py](technical_sources/PYTHON_SOURCE_14.md#py-167) | Matched-start PPO/QBoost study with independent held-out test evaluation | 21,714 |
| [src/holdem_ai/experiments/longrun.py](technical_sources/PYTHON_SOURCE_14.md#py-168) | Phase 15: safe long-run HUNL training, validation and sealed final audit | 26,085 |
| [src/holdem_ai/experiments/ppo_ablation.py](technical_sources/PYTHON_SOURCE_15.md#py-169) | Phase16 isolated PPO optimizer/critic ablations (AGENT ONLY) | 25,669 |
| [src/holdem_ai/experiments/ppo_five_hour.py](technical_sources/PYTHON_SOURCE_15.md#py-170) | Phase17: resumable 5-hour PPO retraining (AGENT ONLY, new run) | 24,123 |
| [src/holdem_ai/experiments/ppo_safety.py](technical_sources/PYTHON_SOURCE_15.md#py-171) | Phase 15 PPO stability safety policy | 944 |
| [src/holdem_ai/experiments/ppo_stress.py](technical_sources/PYTHON_SOURCE_15.md#py-172) | Phase16 long A0/A3 PPO stress test with transactional full-state resume | 30,107 |
| [src/holdem_ai/experiments/time_budget.py](technical_sources/PYTHON_SOURCE_15.md#py-173) | Safe wall-clock bounded Phase11 training with crash-recoverable observability | 15,737 |
| [src/holdem_ai/game_service/__init__.py](technical_sources/PYTHON_SOURCE_15.md#py-174) | Server-side HUNL human-vs-agent integration (GUI-1.0). | 182 |
| [src/holdem_ai/game_service/action_adapter.py](technical_sources/PYTHON_SOURCE_15.md#py-175) | Adapt the existing seven-action HUNL200 policy head to GUI action offers | 2,708 |
| [src/holdem_ai/game_service/bot_adapter.py](technical_sources/PYTHON_SOURCE_15.md#py-176) | Apply any Phase-5 AgentPolicy to a bot turn, including NeuralActorPolicy | 787 |
| [src/holdem_ai/game_service/checkpoint_catalog.py](technical_sources/PYTHON_SOURCE_15.md#py-177) | Trusted local checkpoint selection for GUI-1.4 HUNL evaluation | 4,986 |
| [src/holdem_ai/game_service/interactive_adapter.py](technical_sources/PYTHON_SOURCE_15.md#py-178) | Human-only exact bet sizing over the trained HUNL200 seven-action adapter | 2,755 |
| [src/holdem_ai/game_service/session_analytics.py](technical_sources/PYTHON_SOURCE_15.md#py-179) | GUI-1.5: bounded, hero-visible completed-hand ledger and online metrics | 5,496 |
| [src/holdem_ai/game_service/session_manager.py](technical_sources/PYTHON_SOURCE_15.md#py-180) | Local HUNL200 human-vs-agent game sessions; transport agnostic | 11,214 |
| [src/holdem_ai/game_service/state_projection.py](technical_sources/PYTHON_SOURCE_16.md#py-181) | Explicit whitelist of game data safe to serialize to the human browser | 4,653 |
| [src/holdem_ai/models/__init__.py](technical_sources/PYTHON_SOURCE_16.md#py-182) | Consolidated functional package. | 39 |
| [src/holdem_ai/models/transformer.py](technical_sources/PYTHON_SOURCE_16.md#py-183) | VRPO-Lite v1 trainable networks for the existing Phase-7 batch contracts | 8,612 |
| [src/holdem_ai/tournament_service/__init__.py](technical_sources/PYTHON_SOURCE_16.md#py-184) | Local-only GUI environment integration for the Phase 4 tournament engine. | 217 |
| [src/holdem_ai/tournament_service/baseline.py](technical_sources/PYTHON_SOURCE_16.md#py-185) | Temporary, engine-legal six-seat baseline policies | 1,345 |
| [src/holdem_ai/tournament_service/improved_bot.py](technical_sources/PYTHON_SOURCE_16.md#py-186) | Tournament ENV 1.3: four reproducible, observation-only heuristic opponents | 5,676 |
| [src/holdem_ai/tournament_service/journal.py](technical_sources/PYTHON_SOURCE_16.md#py-187) | Tournament ENV 1.6: append-only, local SQLite event journal | 10,830 |
| [src/holdem_ai/tournament_service/session.py](technical_sources/PYTHON_SOURCE_16.md#py-188) | Phase 1.1: six-seat, human-vs-baselines TournamentEnv session adapter | 20,805 |
| [src/holdem_ai/tournament_service/trained_actor.py](technical_sources/PYTHON_SOURCE_16.md#py-189) | Tournament ENV 1.4: frozen Expand Phase3 multi-seat PPO inference bridge | 9,173 |
| [src/holdem_ai/training/__init__.py](technical_sources/PYTHON_SOURCE_16.md#py-190) | Self-play and training pipelines. | 40 |
| [src/holdem_ai/training/batched_collector.py](technical_sources/PYTHON_SOURCE_16.md#py-191) | Batched (multi-hand, single-process) on-policy GPU inference for HUNL200 | 5,632 |
| [src/holdem_ai/training/episode_runner.py](technical_sources/PYTHON_SOURCE_16.md#py-192) | Turn-based multiagent self-play, one transition per decision by each player. | 2,963 |
| [src/holdem_ai/training/global_rollout.py](technical_sources/PYTHON_SOURCE_17.md#py-193) | Global-decision trajectories for future CTDE / VRPO learners (no training yet) | 3,127 |
| [src/holdem_ai/training/ppo.py](technical_sources/PYTHON_SOURCE_17.md#py-194) | On-policy PPO baseline for Phase-8 HUNL200 networks (NOT VRPO/Q-Boosting) | 15,577 |
| [src/holdem_ai/training/ppo_session.py](technical_sources/PYTHON_SOURCE_17.md#py-195) | Phase-9 on-policy PPO orchestration, safe checkpoints, and resume. | 5,241 |
| [src/holdem_ai/training/qboost.py](technical_sources/PYTHON_SOURCE_17.md#py-196) | Phase 10: doubly-corrected Q-control-variate PPO research baseline | 11,179 |
| [src/holdem_ai/training/qboost_session.py](technical_sources/PYTHON_SOURCE_17.md#py-197) | Phase-10 trainer, safe checkpoints and Phase-9 warm start | 5,741 |
| [src/holdem_ai/training/reward_models.py](technical_sources/PYTHON_SOURCE_17.md#py-198) | Terminal reward definitions; preserve distinction between chip and tournament utility. | 1,937 |
| [src/holdem_ai/training/self_play.py](technical_sources/PYTHON_SOURCE_17.md#py-199) | Empty file; no executable definitions. | 0 |
| [src/holdem_ai/training/trainer.py](technical_sources/PYTHON_SOURCE_17.md#py-200) | Empty file; no executable definitions. | 0 |
| [src/holdem_ai/utils/__init__.py](technical_sources/PYTHON_SOURCE_17.md#py-201) | Consolidated functional package. | 39 |
| [src/holdem_ai/utils/progress.py](technical_sources/PYTHON_SOURCE_17.md#py-202) | Live, optional-tqdm terminal progress for Phase11 training | 9,411 |
| [src/holdem_ai/utils/snapshot.py](technical_sources/PYTHON_SOURCE_17.md#py-203) | Safe, explicit experimental snapshot facility (not a CFR correctness guarantee). | 776 |
| [tests/__init__.py](technical_sources/PYTHON_SOURCE_17.md#py-204) | Empty file; no executable definitions. | 0 |
| [tests/agents/__init__.py](technical_sources/PYTHON_SOURCE_18.md#py-205) | Empty file; no executable definitions. | 0 |
| [tests/environment/__init__.py](technical_sources/PYTHON_SOURCE_18.md#py-206) | Empty file; no executable definitions. | 0 |
| [tests/environment/test_betting.py](technical_sources/PYTHON_SOURCE_18.md#py-207) | Regression checks: test_bb_min_raise_preflop, test_postflop_bet_minimum_and_stack_cap, test_short_all_in_is_allowed_but_is_not_a_full_raise, tes | 1,888 |
| [tests/environment/test_elimination.py](technical_sources/PYTHON_SOURCE_18.md#py-208) | Empty file; no executable definitions. | 0 |
| [tests/environment/test_hand_evaluator.py](technical_sources/PYTHON_SOURCE_18.md#py-209) | Regression checks: test_hierarchy_of_poker_hands, test_wheel_straight_loses_to_six_high, test_two_triplets_form_highest_full_house, test_board_p | 1,708 |
| [tests/environment/test_position.py](technical_sources/PYTHON_SOURCE_18.md#py-210) | Regression checks: test_heads_up_button_is_sb_and_first_preflop, test_blinds_are_charged_to_correct_seats, test_invalid_button_is_rejected | 942 |
| [tests/environment/test_side_pot.py](technical_sources/PYTHON_SOURCE_18.md#py-211) | Regression checks: test_three_way_all_in_has_correct_main_and_side_pots, test_folded_player_contribution_stays_in_pot, test_unmatched_wager_retu | 1,850 |
| [tests/expand/conftest.py](technical_sources/PYTHON_SOURCE_18.md#py-212) | Isolated stub for table-management tests; real-engine tests are separate. | 2,814 |
| [tests/expand/e03_fake_engine.py](technical_sources/PYTHON_SOURCE_18.md#py-213) | Deterministic test-only 2-6 player game | 3,340 |
| [tests/expand/e07_fake_tournament.py](technical_sources/PYTHON_SOURCE_18.md#py-214) | A deterministic side-effect-free test double, NEVER a poker engine. | 2,004 |
| [tests/expand/test_e01_real_engine.py](technical_sources/PYTHON_SOURCE_18.md#py-215) | These require the user's real Base Phase 3 modules (skipped in standalone patch). | 3,209 |
| [tests/expand/test_e01_smoke_unit.py](technical_sources/PYTHON_SOURCE_18.md#py-216) | CLI contract tested with a fake engine; the real smoke is a Windows gate. | 1,308 |
| [tests/expand/test_e01_table_unit.py](technical_sources/PYTHON_SOURCE_19.md#py-217) | Regression checks: test_config_rejects_unsupported_modes_and_amounts, test_initial_bootstrap_fixed_seats_and_cash_conservation, test_join_during | 9,209 |
| [tests/expand/test_e02_adapter_fake.py](technical_sources/PYTHON_SOURCE_19.md#py-218) | Regression checks: test_e02_tracker_carries_street_across_actions, test_e02_rejected_action_does_not_corrupt_tracker, test_e02_external_action_d | 3,925 |
| [tests/expand/test_e02_policy.py](technical_sources/PYTHON_SOURCE_19.md#py-219) | Regression checks: test_e02_policy_logits_gradients_and_illegal_zero, test_e02_policy_seed_and_legal_sampling, test_e02_seat_permutation_equivar | 2,999 |
| [tests/expand/test_e02_real_engine.py](technical_sources/PYTHON_SOURCE_19.md#py-220) | Only run where Base Phase3 is installed (Windows user's repository). | 2,882 |
| [tests/expand/test_e02_state_and_actions.py](technical_sources/PYTHON_SOURCE_19.md#py-221) | Regression checks: test_e02_only_acting_player_and_public_fields, test_e02_invariance_to_opponent_hole_and_future_deck, test_e02_street_hole_and | 7,263 |
| [tests/expand/test_e03_cli.py](technical_sources/PYTHON_SOURCE_19.md#py-222) | Regression checks: test_e03_cli_has_safe_defaults_and_restricted_resume, test_e03_custom_curriculum | 883 |
| [tests/expand/test_e03_real_engine.py](technical_sources/PYTHON_SOURCE_19.md#py-223) | Real 2-6 seat engine integration: deferred until the Windows full repo. | 999 |
| [tests/expand/test_e03_returns_and_policy.py](technical_sources/PYTHON_SOURCE_19.md#py-224) | Regression checks: test_population_probabilities_and_guards, test_rollout_all_2_to_6_and_reward_assignment, test_missing_terminal_rejected, test | 3,915 |
| [tests/expand/test_e03_trainer.py](technical_sources/PYTHON_SOURCE_19.md#py-225) | Regression checks: test_e03_checkpoint_resume_snapshot_validation, test_e03_resume_exact_reproducible, test_e03_checkpoint_tamper_rejected | 3,203 |
| [tests/expand/test_e04_checkpoint.py](technical_sources/PYTHON_SOURCE_19.md#py-226) | Regression checks: test_resume_exact_weights_and_round, test_checkpoint_roundtrip_leduc, test_checksum_detects_tampering, test_checkpoint_incomp | 2,163 |
| [tests/expand/test_e04_cli.py](technical_sources/PYTHON_SOURCE_19.md#py-227) | Regression checks: test_cli_kuhn_smoke_resume, test_cli_leduc_smoke, test_cli_budget_limit_saves_empty_model | 1,169 |
| [tests/expand/test_e04_holdem_mock.py](technical_sources/PYTHON_SOURCE_19.md#py-228) | Regression checks: test_real_interface_on_fast_deterministic_engine, test_private_rival_cards_do_not_affect_hero_infoset, test_postflop_card_buc | 2,673 |
| [tests/expand/test_e04_real_engine.py](technical_sources/PYTHON_SOURCE_20.md#py-229) | Testable on Windows after installing actual Base Phase 3 engine. | 1,393 |
| [tests/expand/test_e04_toy_mccfr.py](technical_sources/PYTHON_SOURCE_20.md#py-230) | Regression checks: test_toy_terminal_and_reward_zero_sum, test_mccfr_2_and_3_player_updates, test_infosets_hide_other_hole_and_future_community, | 4,100 |
| [tests/expand/test_e05_beliefs.py](technical_sources/PYTHON_SOURCE_20.md#py-231) | Regression checks: test_toy_belief_fix_hero_and_exclude_collisions, test_replay_history_only_public_actions, test_reject_impossible_trace | 1,406 |
| [tests/expand/test_e05_cli.py](technical_sources/PYTHON_SOURCE_20.md#py-232) | Regression checks: test_cli_toy_json_and_no_overwrite | 595 |
| [tests/expand/test_e05_continuations.py](technical_sources/PYTHON_SOURCE_20.md#py-233) | Regression checks: test_all_continuations_have_valid_legal_distribution, test_fivefold_weighting_is_over_original_mass, test_raise_bias_affects_ | 1,417 |
| [tests/expand/test_e05_evaluate.py](technical_sources/PYTHON_SOURCE_20.md#py-234) | Regression checks: test_evaluation_not_modifying_checkpoint | 757 |
| [tests/expand/test_e05_holdem_deck.py](technical_sources/PYTHON_SOURCE_20.md#py-235) | Regression checks: test_card_layout_exact_hero_visible_board_and_no_collision, test_card_collisions_and_bad_hole_rejected | 1,611 |
| [tests/expand/test_e05_nested.py](technical_sources/PYTHON_SOURCE_20.md#py-236) | Regression checks: test_off_tree_re_root_and_committed_own_actions, test_reject_bad_off_tree_size | 885 |
| [tests/expand/test_e05_real_engine.py](technical_sources/PYTHON_SOURCE_20.md#py-237) | Runs only on user's real base Phase3 repository install. | 2,180 |
| [tests/expand/test_e05_solver.py](technical_sources/PYTHON_SOURCE_20.md#py-238) | Regression checks: test_valid_probabilities_and_blueprint_unchanged, test_reproducible_with_seed_and_large_budget, test_node_cap_fallback_to_blu | 2,866 |
| [tests/expand/test_e06_adaptive.py](technical_sources/PYTHON_SOURCE_20.md#py-239) | Regression checks: test_no_evidence_preserves_base, test_low_samples_no_adapt, test_trust_kl_tv_bounds_and_probability_simplex, test_zero_suppor | 3,226 |
| [tests/expand/test_e06_bridge.py](technical_sources/PYTHON_SOURCE_20.md#py-240) | Regression checks: test_positions_six_and_four, test_prior_raises_counts_exact_raise_to, test_registry_tracks_join_tenures_from_snapshot, test_a | 2,084 |
| [tests/expand/test_e06_cli.py](technical_sources/PYTHON_SOURCE_21.md#py-241) | Regression checks: test_synthetic_blocked_benchmark, test_determinism | 991 |
| [tests/expand/test_e06_models.py](technical_sources/PYTHON_SOURCE_21.md#py-242) | Regression checks: test_spr_buckets_and_validation, test_session_isolation_across_same_seat, test_no_seat_lease_overlap, test_legal_context_and_ | 4,350 |
| [tests/expand/test_e06_persistence.py](technical_sources/PYTHON_SOURCE_21.md#py-243) | Regression checks: test_checkpoint_roundtrip_resume, test_checkpoint_tamper_detection, test_checkpoint_contains_no_card_or_deck | 1,702 |
| [tests/expand/test_e06_real_engine.py](technical_sources/PYTHON_SOURCE_21.md#py-244) | Runs only once Base Phase 3 engine is present in user's real repository. | 1,922 |
| [tests/expand/test_e06_router.py](technical_sources/PYTHON_SOURCE_21.md#py-245) | Regression checks: test_router_uses_different_experts_for_styles, test_router_uncertain_fallback | 1,797 |
| [tests/expand/test_e06_state_bridge.py](technical_sources/PYTHON_SOURCE_21.md#py-246) | Public-information bridge test: Phase2 private hero hole must never pass to Phase6. | 1,643 |
| [tests/expand/test_e07_agent.py](technical_sources/PYTHON_SOURCE_21.md#py-247) | Regression checks: test_e07_uniform_reproducible_and_legal, test_e07_priority_and_fallback, test_e07_unverified_abstraction_never_executed, test | 3,183 |
| [tests/expand/test_e07_benchmark.py](technical_sources/PYTHON_SOURCE_21.md#py-248) | Regression checks: test_e07_cash_scenarios_and_independent_clustering, test_e07_cash_paired_rejects_different_seed, test_e07_tournament_sessions | 1,777 |
| [tests/expand/test_e07_bridges_and_cli.py](technical_sources/PYTHON_SOURCE_21.md#py-249) | Regression checks: test_e07_search_source_rejects_timeout_without_mask_override, test_e07_search_source_rejects_nonmatching_action_support, test | 2,195 |
| [tests/expand/test_e07_cash_bridge.py](technical_sources/PYTHON_SOURCE_21.md#py-250) | Regression checks: test_e07_e1_e2_e6_accepts_public_actions | 584 |
| [tests/expand/test_e07_manifest.py](technical_sources/PYTHON_SOURCE_21.md#py-251) | Regression checks: test_e07_manifest_checkpoint_hash_and_no_overwrite | 847 |
| [tests/expand/test_e07_real_engine.py](technical_sources/PYTHON_SOURCE_21.md#py-252) | Gated: run only when Base Phase3 engine is installed in user's repository. | 1,008 |
| [tests/expand/test_e07_tournament.py](technical_sources/PYTHON_SOURCE_22.md#py-253) | Regression checks: test_e07_tournament_knockouts_chip_ledger_and_prizes, test_e07_reentry_new_id_late_registration, test_e07_tournament_guard_ba | 2,716 |
| [tests/test_blind_manager.py](technical_sources/PYTHON_SOURCE_22.md#py-254) | Regression checks: test_blind_levels_repeat_last_level, test_invalid_blinds, test_blind_hand_number_validation | 1,089 |
| [tests/test_corrected_rlcard_integration.py](technical_sources/PYTHON_SOURCE_22.md#py-255) | End-to-end corrected engine tests; run with RLCard 1.2.0 installed. | 2,786 |
| [tests/test_corrected_round.py](technical_sources/PYTHON_SOURCE_22.md#py-256) | Independent betting scenarios requiring no RLCard installation. | 3,360 |
| [tests/test_gui10_hunl_contract.py](technical_sources/PYTHON_SOURCE_22.md#py-257) | GUI-1.0 behavior, privacy and action compatibility with the real HUNL200 API. | 7,544 |
| [tests/test_gui11_http_api.py](technical_sources/PYTHON_SOURCE_22.md#py-258) | GUI-1.1 HTTP contract tests (requires GUI-1.0 + FastAPI/httpx). | 9,066 |
| [tests/test_gui12_readonly_api.py](technical_sources/PYTHON_SOURCE_22.md#py-259) | GUI-1.2 read-only display contract against the real GUI-1.1 API. | 2,754 |
| [tests/test_gui13_interactive_betting.py](technical_sources/PYTHON_SOURCE_22.md#py-260) | GUI-1.3: actual engine transitions through browser-facing HTTP contracts. | 4,882 |
| [tests/test_gui14_trained_agents.py](technical_sources/PYTHON_SOURCE_22.md#py-261) | GUI-1.4 local trained-agent selection, loading, privacy, and HTTP regression. | 7,689 |
| [tests/test_gui15_session_analytics.py](technical_sources/PYTHON_SOURCE_22.md#py-262) | GUI-1.5: completed-hand ledger, summary, privacy, pagination, API isolation. | 8,219 |
| [tests/test_gui16_continuous_stack.py](technical_sources/PYTHON_SOURCE_22.md#py-263) | GUI-1.6 end-to-end stack carryover, cash accounting, bust and mode isolation | 6,973 |
| [tests/test_hand_engine.py](technical_sources/PYTHON_SOURCE_22.md#py-264) | Offline tests for the public adapter API with a stand-in RLCard Env. | 2,191 |
| [tests/test_multi_player_engine.py](technical_sources/PYTHON_SOURCE_23.md#py-265) | Regression checks: test_preflop_and_postflop_order_three_players, test_six_players_betting_order, test_headsup_compatible_and_private_hole_cards | 11,045 |
| [tests/test_multi_player_seats.py](technical_sources/PYTHON_SOURCE_23.md#py-266) | Regression checks: test_heads_up_seat_rule_and_sparse_empty_seats, test_three_and_six_players, test_rotate_button_skips_empty_seats, test_invali | 1,202 |
| [tests/test_phase10_cli.py](technical_sources/PYTHON_SOURCE_23.md#py-267) | End-to-end Phase 10 CLI on CPU with mock Phase3 integration fixture. | 2,073 |
| [tests/test_phase10_gpu.py](technical_sources/PYTHON_SOURCE_23.md#py-268) | GTX1070 smoke; skipped in environments without PyTorch CUDA. | 907 |
| [tests/test_phase10_qboost.py](technical_sources/PYTHON_SOURCE_23.md#py-269) | Phase 10: algebra, information boundary, policy-gradient and integration. | 11,558 |
| [tests/test_phase11_batched.py](technical_sources/PYTHON_SOURCE_23.md#py-270) | Regression checks: test_batched_rows_are_legal_and_seat_specific, test_batch_one_matches_legacy_sequential, test_batched_reproducible_and_change | 3,130 |
| [tests/test_phase11_gpu.py](technical_sources/PYTHON_SOURCE_23.md#py-271) | CUDA-only batched rollout integration, runs on the user's GTX 1070. | 863 |
| [tests/test_phase11_progress.py](technical_sources/PYTHON_SOURCE_23.md#py-272) | Progress UI is an observer: no dependency on the poker engine or CUDA. | 5,397 |
| [tests/test_phase11_runner.py](technical_sources/PYTHON_SOURCE_23.md#py-273) | Regression checks: test_small_real_update_and_resume, test_compare_reports_after_two_run_types, test_budget_and_config_validation, test_tiny_dur | 4,381 |
| [tests/test_phase12_cli.py](technical_sources/PYTHON_SOURCE_23.md#py-274) | Regression checks: test_local_checkpoint_and_json_csv, test_phase11_anchor_and_paired_control, test_reject_same_input_file, test_reject_bad_chec | 2,673 |
| [tests/test_phase12_evaluation.py](technical_sources/PYTHON_SOURCE_23.md#py-275) | Phase12 robust evaluation: same-deck, opposite-seat, unit tests. | 5,558 |
| [tests/test_phase12_gpu.py](technical_sources/PYTHON_SOURCE_23.md#py-276) | GPU smoke on user's GTX1070; deliberately skipped on CPU-only runners. | 667 |
| [tests/test_phase13_gpu.py](technical_sources/PYTHON_SOURCE_24.md#py-277) | Actual CUDA run, skipped in environments without a CUDA GPU. | 800 |
| [tests/test_phase13_scaling.py](technical_sources/PYTHON_SOURCE_24.md#py-278) | Phase 13 backward-compatible workflow tests (real model and simulator contract). | 9,177 |
| [tests/test_phase14_gpu.py](technical_sources/PYTHON_SOURCE_24.md#py-279) | A tiny live CUDA end-to-end check for the user's Windows GPU. | 1,124 |
| [tests/test_phase14_study.py](technical_sources/PYTHON_SOURCE_24.md#py-280) | Phase14 correctness, matched-start and held-out test protection. | 5,973 |
| [tests/test_phase15_critic_diagnostics.py](technical_sources/PYTHON_SOURCE_24.md#py-281) | Tests for Phase-15 read-only critic calibration and numerical edge cases. | 6,084 |
| [tests/test_phase15_gpu.py](technical_sources/PYTHON_SOURCE_24.md#py-282) | Real CUDA invocation: only runs on CUDA-enabled hosts. | 1,352 |
| [tests/test_phase15_initial_policy_comparison.py](technical_sources/PYTHON_SOURCE_24.md#py-283) | Agent baseline all-in comparison regression tests (no environment edits). | 6,925 |
| [tests/test_phase15_longrun.py](technical_sources/PYTHON_SOURCE_24.md#py-284) | Phase15 safety, resume, validation selection, held-out sealing and CLI. | 9,764 |
| [tests/test_phase15_policy_audit.py](technical_sources/PYTHON_SOURCE_24.md#py-285) | Regression tests for Phase 15 *read-only* policy/optimizer diagnostics. | 3,422 |
| [tests/test_phase15_preflop_sequence_audit.py](technical_sources/PYTHON_SOURCE_24.md#py-286) | Phase15 agent-only preflop sequence audit checks. | 5,009 |
| [tests/test_phase15_safety_phase3.py](technical_sources/PYTHON_SOURCE_24.md#py-287) | Phase 15 phase-3 safety and reward diagnostics regression tests. | 2,072 |
| [tests/test_phase15_terminal_reward_audit.py](technical_sources/PYTHON_SOURCE_24.md#py-288) | Regression tests: hand-weighted payoff tails and preflop termination types. | 5,242 |
| [tests/test_phase16_cli.py](technical_sources/PYTHON_SOURCE_25A.md#py-289) | Regression checks: test_cli_smoke_creates_json_csv_without_training, test_cli_named_checkpoint_and_invalid_spec | 2,192 |
| [tests/test_phase16_diagnostics.py](technical_sources/PYTHON_SOURCE_25A.md#py-290) | Regression checks: test_effective_spr_public_observable, test_conditional_rate_not_raw_rate_and_strata, test_illegal_selection_rejected, test_en | 3,119 |
| [tests/test_phase16_evaluation.py](technical_sources/PYTHON_SOURCE_25A.md#py-291) | Regression checks: test_multiplayer_pool_reproducible_and_no_mutation, test_empty_and_duplicate_opponents_rejected, test_invalid_config, test_ne | 3,714 |
| [tests/test_phase16_opponents.py](technical_sources/PYTHON_SOURCE_25A.md#py-292) | Regression checks: test_opponents_choose_legal_actions, test_behavior_semantics, test_invalid_builtins, test_checkpoint_name_allowed, test_facto | 1,570 |
| [tests/test_phase16_ppo_ablation.py](technical_sources/PYTHON_SOURCE_25A.md#py-293) | Phase16 isolated PPO ablations: objectives, KL, checkpoints and safety. | 7,515 |
| [tests/test_phase16_ppo_stress.py](technical_sources/PYTHON_SOURCE_25A.md#py-294) | Phase16 A0/A3 long-stress regression and real-HUNL mini integration tests. | 9,100 |
| [tests/test_phase17_continuous_stack.py](technical_sources/PYTHON_SOURCE_25B.md#py-295) | Phase17 integration regression against the real GUI-1.6 hand adapter. | 7,905 |
| [tests/test_phase17_gpu_retrain.py](technical_sources/PYTHON_SOURCE_25B.md#py-296) | Phase17 5-hour PPO runner: immutable source, full resume and KL safeguards. | 7,873 |
| [tests/test_phase5_betting.py](technical_sources/PYTHON_SOURCE_25B.md#py-297) | Regression checks: test_street_raise_targets_are_integer_post_call_pot, test_raise_rounds_up_to_min_and_deduplicates, test_short_all_in_allowed_ | 3,059 |
| [tests/test_phase5_numpy_json.py](technical_sources/PYTHON_SOURCE_25B.md#py-298) | Regression tests: RLCard public state may include nested NumPy scalars. | 2,125 |
| [tests/test_phase5_rewards.py](technical_sources/PYTHON_SOURCE_25B.md#py-299) | Regression checks: test_cash_zero_sum_in_bb_units, test_chip_deltas_six_seat_and_bust, test_chip_delta_chips_not_conserved, test_bad_scale_and_s | 1,406 |
| [tests/test_phase5_rlcard_adapter.py](technical_sources/PYTHON_SOURCE_25B.md#py-300) | Regression checks: test_observation_never_exposes_opponents_hidden_cards_or_engine_data, test_rlcard_adapter_seed_masks_and_selfplay, test_rlcar | 5,089 |
| [tests/test_phase5_schema_policy.py](technical_sources/PYTHON_SOURCE_26.md#py-301) | Regression checks: test_infoset_key_stable_and_private_distinguished, test_legal_mask_rejects_illegal_and_type, test_nonterminal_must_have_legal | 3,631 |
| [tests/test_phase5_self_play.py](technical_sources/PYTHON_SOURCE_26.md#py-302) | Regression checks: test_selfplay_6_max_trajectories_link_next_same_seat, test_deterministic_selfplay_actions_seed, test_selfplay_missing_policy, | 3,544 |
| [tests/test_phase6_benchmark.py](technical_sources/PYTHON_SOURCE_26.md#py-303) | Small speed-test smoke; does not assert hardware-dependent throughput. | 640 |
| [tests/test_phase6_hunl_adapter.py](technical_sources/PYTHON_SOURCE_26.md#py-304) | Integration tests run against the actual Phase 3 engine in the repository. | 6,125 |
| [tests/test_phase7_encoding.py](technical_sources/PYTHON_SOURCE_26.md#py-305) | Phase 7 contracts: actor privacy, history, stable cards, critic feature schema. | 6,764 |
| [tests/test_phase7_rollout.py](technical_sources/PYTHON_SOURCE_26.md#py-306) | End-to-end HUNL200 -> global episodes -> actor/critic tensors | 1,772 |
| [tests/test_phase7_tensors_actions.py](technical_sources/PYTHON_SOURCE_26.md#py-307) | Action legality, padding masks, numeric tensors and gradients (GPU optional). | 3,709 |
| [tests/test_phase8_gpu.py](technical_sources/PYTHON_SOURCE_26.md#py-308) | Runs a minimal real Phase-7 batch on CUDA when available (e.g | 1,530 |
| [tests/test_phase8_networks.py](technical_sources/PYTHON_SOURCE_26.md#py-309) | Phase-8 independent unit tests using real Phase-7 batch structures. | 9,837 |
| [tests/test_phase9_cli.py](technical_sources/PYTHON_SOURCE_27.md#py-310) | Phase-9 executable CLI and checkpoint interface, CPU-only fast smoke. | 1,112 |
| [tests/test_phase9_gpu.py](technical_sources/PYTHON_SOURCE_27.md#py-311) | Run on user's GTX1070; skip on the CPU-only packaging machine. | 1,296 |
| [tests/test_phase9_ppo.py](technical_sources/PYTHON_SOURCE_27.md#py-312) | Phase-9 unit/integration tests against real HUNL200 engine in the user's repo. | 8,613 |
| [tests/test_player_manager.py](technical_sources/PYTHON_SOURCE_27.md#py-313) | Regression checks: test_elimination_and_pre_hand_stack_tiebreak, test_different_prehand_stacks_same_hand_bust, test_equal_start_busts_joint_rank | 1,675 |
| [tests/test_rewards_tournament.py](technical_sources/PYTHON_SOURCE_27.md#py-314) | Regression checks: test_chip_rewards_and_winner_take_all, test_joint_placing_split_prize_slots, test_invalid_rank_sequences, test_invalid_payout | 836 |
| [tests/test_rlcard_integration.py](technical_sources/PYTHON_SOURCE_27.md#py-315) | Runs only where RLCard is installed; verifies a real hand end-to-end. | 543 |
| [tests/test_rlcard_rules_integration.py](technical_sources/PYTHON_SOURCE_27.md#py-316) | Checks the installed RLCard (1.2.0) rather than synthetic game objects. | 1,036 |
| [tests/test_rules_audit.py](technical_sources/PYTHON_SOURCE_27.md#py-317) | Regression checks: test_audit_accepts_a_correct_heads_up_implementation, test_audit_detects_a_reversed_blind_and_first_actor | 881 |
| [tests/test_tournament_env.py](technical_sources/PYTHON_SOURCE_27.md#py-318) | Regression checks: test_reproducible_two_player_complete_tournament, test_tournament_random_end_to_end, test_incremental_step_observation_no_pri | 8,761 |
| [tests/test_tournament_env131_aggression.py](technical_sources/PYTHON_SOURCE_28.md#py-319) | ENV 1.3.1 regression tests for multiway bot aggression and action safety. | 2,720 |
| [tests/test_tournament_env13_improved_bots.py](technical_sources/PYTHON_SOURCE_28.md#py-320) | Tournament ENV 1.3: explicit six-seat lineup with four heuristic policies. | 5,331 |
| [tests/test_tournament_env14_trained_actor.py](technical_sources/PYTHON_SOURCE_28.md#py-321) | ENV 1.4: checkpoint gating, action contract and no silent baseline fallback. | 7,407 |
| [tests/test_tournament_env15_progress.py](technical_sources/PYTHON_SOURCE_28.md#py-322) | Tournament ENV 1.5 public, read-only standings and blind level contract. | 4,064 |
| [tests/test_tournament_env161_showdown.py](technical_sources/PYTHON_SOURCE_28.md#py-323) | Phase 1.6.1: show only legitimate showdown hands, and only after settlement. | 7,741 |
| [tests/test_tournament_env16_persistence.py](technical_sources/PYTHON_SOURCE_28.md#py-324) | Phase 1.6: durable actor-visible transitions and public replay. | 6,901 |
| [tests/test_tournament_env17_integration.py](technical_sources/PYTHON_SOURCE_28.md#py-325) | Tournament ENV 1.7: release contract across both GUIs and durable replay | 4,731 |
| [tests/test_tournament_env17_release_audit.py](technical_sources/PYTHON_SOURCE_28.md#py-326) | Unit tests for the privacy/staging release gate, no project game imports. | 2,253 |
| [tests/test_tournament_env1_api.py](technical_sources/PYTHON_SOURCE_28.md#py-327) | Tournament ENV 1.1 -- API/engine integration and privacy boundary. | 10,183 |

<a name="section-27594225cc64"></a>

## Appendix B — Complete Original Text Index for All 76 Markdown Files

These appendices reproduce the original frontmatter, prose, equations, Python/PowerShell code, experiment logs, and links without alteration. Historical links and paths retain their original context; this index provides the current navigation. Both initial reference reports and every detailed report are included.

| Complete original document | Original bytes | Consolidation result |
|---|---:|---|
| [README.md](report_sources/ORIGINAL_REPORTS_07.md#md-001) | 6,673 | Current document rewritten; original text preserved in the appendix |
| [README_TOURNAMENT_ENV_1_3_1.md](report_sources/ORIGINAL_REPORTS_06B.md#md-002) | 399 | Standalone document removed; complete text incorporated into the final report appendix |
| [README_TOURNAMENT_ENV_PHASE_1_2_1.md](report_sources/ORIGINAL_REPORTS_07.md#md-003) | 1,058 | Standalone document removed; complete text incorporated into the final report appendix |
| [README_TOURNAMENT_ENV_PHASE_1_3.md](report_sources/ORIGINAL_REPORTS_07.md#md-004) | 610 | Standalone document removed; complete text incorporated into the final report appendix |
| [README_TOURNAMENT_ENV_PHASE_1_4.md](report_sources/ORIGINAL_REPORTS_07.md#md-005) | 1,193 | Standalone document removed; complete text incorporated into the final report appendix |
| [README_TOURNAMENT_ENV_PHASE_1_5.md](report_sources/ORIGINAL_REPORTS_07.md#md-006) | 1,320 | Standalone document removed; complete text incorporated into the final report appendix |
| [README_TOURNAMENT_ENV_PHASE_1_6.md](report_sources/ORIGINAL_REPORTS_07.md#md-007) | 1,074 | Standalone document removed; complete text incorporated into the final report appendix |
| [README_TOURNAMENT_ENV_PHASE_1_6_1.md](report_sources/ORIGINAL_REPORTS_07.md#md-008) | 1,310 | Standalone document removed; complete text incorporated into the final report appendix |
| [TOURNAMENT_ENV_PHASE_1_1_README.md](report_sources/ORIGINAL_REPORTS_07.md#md-009) | 1,469 | Standalone document removed; complete text incorporated into the final report appendix |
| [TOURNAMENT_ENV_PHASE_1_2_README.md](report_sources/ORIGINAL_REPORTS_07.md#md-010) | 2,642 | Standalone document removed; complete text incorporated into the final report appendix |
| [TOURNAMENT_ENV_PHASE_1_7_README.md](report_sources/ORIGINAL_REPORTS_07.md#md-011) | 1,091 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/AGENT_DEVELOPMENT_TECHNICAL_REPORT_2026-10-13.md](report_sources/ORIGINAL_REPORTS_01.md#md-012) | 93,299 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/EXPAND_MASTER_DESIGN_6MAX_v1_0.md](report_sources/ORIGINAL_REPORTS_01.md#md-013) | 50,610 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/EXPAND_PHASE01_DESIGN.md](report_sources/ORIGINAL_REPORTS_01.md#md-014) | 12,011 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/EXPAND_PHASE01_TO_07_INTEGRATED_TECHNICAL_REPORT.md](report_sources/ORIGINAL_REPORTS_01.md#md-015) | 31,992 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/EXPAND_PHASE02_INFORMATION_AND_ENCODING.md](report_sources/ORIGINAL_REPORTS_01.md#md-016) | 15,623 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/EXPAND_PHASE03_MULTIAGENT_PPO.md](report_sources/ORIGINAL_REPORTS_01.md#md-017) | 15,621 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/EXPAND_PHASE04_MCCFR_BLUEPRINT.md](report_sources/ORIGINAL_REPORTS_01.md#md-018) | 14,494 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/EXPAND_PHASE05_SUBGAME_SEARCH.md](report_sources/ORIGINAL_REPORTS_01.md#md-019) | 15,379 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/EXPAND_PHASE06_OPPONENT_MODELING.md](report_sources/ORIGINAL_REPORTS_01.md#md-020) | 16,222 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/EXPAND_PHASE07_INTEGRATION_TOURNAMENT_BENCHMARK.md](report_sources/ORIGINAL_REPORTS_02.md#md-021) | 20,388 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/GUI_1_0_HUNL_CONTRACT.md](report_sources/ORIGINAL_REPORTS_02.md#md-022) | 7,076 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/GUI_1_1_FASTAPI.md](report_sources/ORIGINAL_REPORTS_02.md#md-023) | 6,246 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/GUI_1_2_REACT_TABLE.md](report_sources/ORIGINAL_REPORTS_02.md#md-024) | 4,148 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/GUI_1_3_INTERACTIVE_BETTING.md](report_sources/ORIGINAL_REPORTS_02.md#md-025) | 6,003 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/GUI_1_4_TRAINED_AGENT.md](report_sources/ORIGINAL_REPORTS_02.md#md-026) | 6,531 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/GUI_1_5_PERFORMANCE_DASHBOARD.md](report_sources/ORIGINAL_REPORTS_02.md#md-027) | 6,040 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/GUI_1_6_CONTINUOUS_STACK.md](report_sources/ORIGINAL_REPORTS_02.md#md-028) | 7,436 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/GUI_TRAINED_CHECKPOINT_SETUP.md](report_sources/ORIGINAL_REPORTS_02.md#md-029) | 5,914 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE10_QBOOST_PPO.md](report_sources/ORIGINAL_REPORTS_03.md#md-030) | 9,382 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE11_FIVE_HOUR_TRAINING.md](report_sources/ORIGINAL_REPORTS_03.md#md-031) | 11,162 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE11_PROGRESS.md](report_sources/ORIGINAL_REPORTS_03.md#md-032) | 3,865 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE12_EVALUATION.md](report_sources/ORIGINAL_REPORTS_03.md#md-033) | 8,509 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE13_BATCH_SCALING.md](report_sources/ORIGINAL_REPORTS_03.md#md-034) | 9,451 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE14_CONTROLLED_STUDY.md](report_sources/ORIGINAL_REPORTS_03.md#md-035) | 9,315 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE15_CRITIC_DIAGNOSIS.md](report_sources/ORIGINAL_REPORTS_03.md#md-036) | 4,914 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE15_EXPERIMENT_REPORT_2026-10-13.md](report_sources/ORIGINAL_REPORTS_03.md#md-037) | 16,335 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE15_LONG_RUN_TRAINING.md](report_sources/ORIGINAL_REPORTS_03.md#md-038) | 11,248 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE15_PHASE3_SAFETY_TELEMETRY.md](report_sources/ORIGINAL_REPORTS_03.md#md-039) | 3,626 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE15_PREFLOP_SEQUENCE_AUDIT.md](report_sources/ORIGINAL_REPORTS_03.md#md-040) | 3,345 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE15_TERMINAL_REWARD_AUDIT.md](report_sources/ORIGINAL_REPORTS_04.md#md-041) | 4,206 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE15_THREE_POLICY_COMPARISON.md](report_sources/ORIGINAL_REPORTS_04.md#md-042) | 3,940 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE16_LONG_STRESS_TEST.md](report_sources/ORIGINAL_REPORTS_04.md#md-043) | 9,005 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE16_OPPONENT_POOL_AND_STRATEGY.md](report_sources/ORIGINAL_REPORTS_04.md#md-044) | 8,643 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE16_PPO_ABLATION.md](report_sources/ORIGINAL_REPORTS_04.md#md-045) | 8,324 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE17_CONTINUOUS_STACK_EVALUATION.md](report_sources/ORIGINAL_REPORTS_04.md#md-046) | 8,630 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE17_FIVE_HOUR_GPU_RETRAIN.md](report_sources/ORIGINAL_REPORTS_04.md#md-047) | 7,944 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE5_AGENT_INTERFACE.md](report_sources/ORIGINAL_REPORTS_05.md#md-048) | 11,848 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE6_HUNL200_INTEGRATION.md](report_sources/ORIGINAL_REPORTS_05.md#md-049) | 6,229 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE7_VRPO_LITE_ENCODING.md](report_sources/ORIGINAL_REPORTS_05.md#md-050) | 7,323 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE8_VRPO_LITE_NETWORKS.md](report_sources/ORIGINAL_REPORTS_05.md#md-051) | 6,923 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE9_PPO_BASELINE.md](report_sources/ORIGINAL_REPORTS_05.md#md-052) | 8,515 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PHASE_VERIFIED_PYTHON_CODE_REFERENCE.md](report_sources/ORIGINAL_REPORTS_02.md#md-053) | 14,753 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md](report_sources/ORIGINAL_REPORTS_05.md#md-054) | 24,744 | Current document rewritten; original text preserved in the appendix |
| [docs/PROJECT_PROGRESS_GUI16_PHASE17.md](report_sources/ORIGINAL_REPORTS_05.md#md-055) | 16,132 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/PROJECT_REPORT.md](report_sources/ORIGINAL_REPORTS_05.md#md-056) | 28,236 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/ROADMAP_PHASE12_TO_16.md](report_sources/ORIGINAL_REPORTS_05.md#md-057) | 3,492 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/TECH_REPORT.md](report_sources/ORIGINAL_REPORTS_05.md#md-058) | 2,369 | Current document rewritten; original text preserved in the appendix |
| [docs/TOURNAMENT_ENV_PHASE1_FULL_TECHNICAL_REPORT.md](report_sources/ORIGINAL_REPORTS_06B.md#md-059) | 52,111 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/TOURNAMENT_ENV_PHASE_1_1.md](report_sources/ORIGINAL_REPORTS_06A.md#md-060) | 6,122 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/TOURNAMENT_ENV_PHASE_1_1_KO.md](report_sources/ORIGINAL_REPORTS_05.md#md-061) | 4,487 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/TOURNAMENT_ENV_PHASE_1_2.md](report_sources/ORIGINAL_REPORTS_06A.md#md-062) | 4,703 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/TOURNAMENT_ENV_PHASE_1_3.md](report_sources/ORIGINAL_REPORTS_06A.md#md-063) | 5,192 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/TOURNAMENT_ENV_PHASE_1_3_1_AGGRESSION_FIX.md](report_sources/ORIGINAL_REPORTS_06A.md#md-064) | 3,715 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/TOURNAMENT_ENV_PHASE_1_4.md](report_sources/ORIGINAL_REPORTS_06A.md#md-065) | 6,534 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/TOURNAMENT_ENV_PHASE_1_5.md](report_sources/ORIGINAL_REPORTS_06B.md#md-066) | 7,204 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/TOURNAMENT_ENV_PHASE_1_6.md](report_sources/ORIGINAL_REPORTS_06B.md#md-067) | 6,756 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/TOURNAMENT_ENV_PHASE_1_6_1_SHOWDOWN.md](report_sources/ORIGINAL_REPORTS_06B.md#md-068) | 4,890 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/TOURNAMENT_ENV_PHASE_1_7_FINAL.md](report_sources/ORIGINAL_REPORTS_06B.md#md-069) | 10,078 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/architecture.md](report_sources/ORIGINAL_REPORTS_01.md#md-070) | 601 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/archive/INTEGRATED_SOURCE_SNAPSHOT_01.md](report_sources/ORIGINAL_REPORTS_01.md#md-071) | 298,011 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/phase1_rlcard.md](report_sources/ORIGINAL_REPORTS_02.md#md-072) | 1,393 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/phase2_1_correction.md](report_sources/ORIGINAL_REPORTS_04.md#md-073) | 2,474 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/phase2_rule_validation.md](report_sources/ORIGINAL_REPORTS_04.md#md-074) | 3,369 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/phase3_multiplayer.md](report_sources/ORIGINAL_REPORTS_04.md#md-075) | 3,006 | Standalone document removed; complete text incorporated into the final report appendix |
| [docs/phase4_tournament.md](report_sources/ORIGINAL_REPORTS_04.md#md-076) | 2,638 | Standalone document removed; complete text incorporated into the final report appendix |

[Manifest: per-file SHA, byte length, appendix location, and consolidation list](lossless_source_manifest.json)
