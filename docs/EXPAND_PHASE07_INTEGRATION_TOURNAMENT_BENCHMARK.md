---
title: "Expand Phase 7 — Integration, Tournament Extension & Scientific Benchmark"
project: HUNL to Dynamic 6-Max NLHE
phase: "Expand 7"
version: "1.0-prototype"
status: "Local isolated tests passed; Windows Base Phase 3 engine acceptance pending"
date: 2026-10-12
tags:
  - poker/6max
  - reinforcement-learning
  - mccfr
  - tournament
  - evaluation
---

# Expand Phase 7 — Integrated 6-Max Poker Agent

> **Release classification:** An E1–E6 integration/evaluation **prototype**, not a certified production Poker Agent or reproduction of Pluribus. An E3 checkpoint remains necessary for learned play; if omitted, the example CLI runs an explicitly labelled **uniform legal baseline**. Pluribus-style E4/E5 requires an externally verified exact-infoset adapter, and is **not automatically enabled** by this CLI.
>
> **Testing:** E1–E7 source combination: `164 passed, 7 skipped` in an isolated mock-engine environment. Real Base Phase 3 engine + Windows/CUDA acceptance remains pending. No GitHub Push.

## 1. Objective and boundaries

The Phase 7 output is an **operational integration interface**, single-table tournament runner, and reproducible **evaluation** pipeline, without modifying Base Phase 1–16 or Expand Phase 1–6.

```text
                                    Dynamic 6-seat Table (E1)
                                                |
                                       InformationState (E2)
                                                |
                  +----------------- Policy Routing (E7) -----------------+
                  |                      |                                |
              E3 Neural            E4 Blueprint                      E5 Search
           trusted checkpoint     only verified key              only verified root
                  |                      |                                |
                  +----------------------+--------------------------------+
                                         |
                                E6 Optional Adaptation
                           candidate expert + KL/TV guard
                                         |
                               Legal ActionTranslator (E2)
                                         |
                                Base Phase 3 Hand Engine
                                         |
                              Independent-session Benchmark
                                         |
                                E7 Release Manifest
```

### 1.1 What is implemented?

| Capability | State | Notes |
|---|---|---|
| E1 Cash + E2 InformationState + E6 accepted public action registry | Implemented | E6 evidence updated **only after action acceptance** |
| E3 PPO checkpoint inference | Implemented | `from_e03_checkpoint` demands trusted E3 format |
| Neural/Blueprint/Search fallback router | Implemented | Search > Blueprint > Neural; source must be explicitly marked compatible |
| E4 Blueprint adapter | Opt-in adapter | Caller must provide **exact** infoset key. Rejects unknown/mismatched actions; missing key is NOT a trained strategy |
| E5 Search adapter | Opt-in adapter | Caller provides verified root belief; incomplete/timeout results fallback |
| E6 StyleExpertRouter + SafeAdaptivePolicy | Implemented / optional | Requires separately trained **hero policy** experts and enough public evidence |
| Cash benchmarks 2–6, 20/100/200BB, opponent pool | Implemented | Each scored hand is from an **independent table session** |
| Single-table Tournament with rising blinds | Implemented | Reuses Base Phase 3 Hand Engine; 0-chip elimination after settlement |
| Late registration and re-entry | Implemented | **Between hands only**; new entry ID; configurable registration cutoff and re-entry cap |
| Payout EV and terminal ranking | Implemented | per-entry finish rank/prize; same-hand equal-before-stack elimination tied |
| Repeatable A/B cash comparison | Implemented | identical scenario seeds, paired per-session differences |
| CLI JSON + SHA256 manifest | Implemented | checkpoint hashes, evaluation script hash, git SHA when available |
| Antes / multi-table tournament / ICM | **Not implemented** | Unsupported ante setting fails explicitly |
| Pluribus-equivalent 100BB Blueprint and full nested re-solving | **Not implemented** | E4 hash buckets, E5 approximate search not production solvers |

## 2. Integrated policy API

```python
from holdem_ai.expand.integration import (
    IntegratedPokerAgent, RoutingConfig, from_e03_checkpoint,
)

neural = from_e03_checkpoint("runs/expand_phase03_pilot/best.pt", device="cuda")
agent = IntegratedPokerAgent(neural, config=RoutingConfig())
choice = agent.act(phase2_information_state, rng_seed=12345)
table_adapter.act(choice.action)
```

`act()` inputs **only** `InformationState` plus RNG seed and optional Phase 6 public opponent evidence. It does **not** take engine instances, hidden deck, opponent hole cards, or CriticState. Source exceptions/mismatches cause labelled fallback.

Let allowed action set be $\mathcal A(I)$ and configured sources $S,E,B,N$ for Search/Blueprint/Neural. Source selection:

$$
\pi_{\rm route}(\cdot\mid I)=
\operatorname{FirstVerifiedValid}(\pi_{S},\pi_{B},\pi_{N},\pi_{U}),
\qquad \pi_U(a\mid I)=\frac{1}{|\mathcal A(I)|}
$$

The uniform emergency source is **not a claim of strategic correctness**. Every selected distribution is checked against precisely the acting hero's legal action support:

$$\mathcal A_{\pi}=\mathcal A(I), \quad
\pi(a)\ge 0,\quad \sum_{a\in\mathcal A(I)}\pi(a)=1.$$

The resulting choice is passed through E2 `to_table_action` so a Raise-to **always** means total committed on the current street, in **integer chips**. No silent bet clamping.

### 2.1 Critical E4 / E5 limitation

The E4 infoset-key generator requires the exact history of engine action IDs (and precise raise-to values). E2's `PublicAction` preserves *public action semantics* but does not in general preserve an unambiguous original engine-ID trace. Consequently, **a generic E2 → E4 exact Infoset map is currently unproven**. This is why Phase 7 does **not** automatically activate an E4 or E5 checkpoint.

```python
from holdem_ai.expand.integration import from_e04_checkpoint

# DO NOT deploy until exact_infoset_key is independently verified:
source, abstraction_id = from_e04_checkpoint(
    "runs/expand_phase04_holdem6/blueprint.json.gz",
    exact_infoset_key=verified_exact_infoset_key,
)
```

E5's opt-in adapter similarly requires an externally validated hero/public belief root:

```python
from holdem_ai.expand.integration import from_e05_solver
source = from_e05_solver(solver, verified_root_belief=verified_root_belief)
```

The E5 solver is still the *Phase 5 current-decision depth-limited approximation*, not the original Pluribus nested search.

### 2.2 E6 optional adaptation

Expert policies must be previously trained on hero-visible information. Phase 6 does **not** train them or guarantee profitability. For a trusted evidence structure $e$ and proposal $q$:

$$
\pi_{\rm adapt}=(1-\eta)\pi_{\rm base}+\eta q,
\qquad
D_{\rm KL}(\pi_{\rm adapt}\|\pi_{\rm base})\le\epsilon_{\rm KL},
\quad
D_{\rm TV}(\pi_{\rm adapt},\pi_{\rm base})\le\epsilon_{\rm TV}.
$$

Adaptation is withheld for insufficient evidence, context mismatch or strategy-shift detection. The Cash benchmark records public action samples through `PublicOpponentTracker`, identified by `(Player ID, join-tenure ID)`. **The opponent decision's legal-action set is never assumed equal to the hero's.**

## 3. Single-table Tournament

Source: `src/holdem_ai/expand/integration/tournament.py`.

Unlike E1's cash-only `DynamicTable`, E7 `SingleTableTournament` holds a roster and reuses the unchanged Base Phase 3 Hand Engine **each hand**, persisting stacks and rotating the button among surviving players.

```python
from holdem_ai.expand.integration import SingleTableTournament, TournamentConfig

cfg = TournamentConfig(
    starting_stack=2000,
    blind_levels=((10,20),(20,40),(40,80),(80,160),(160,320)),
    hands_per_level=4,
    registration_until_hand=3,
    max_reentries_per_player=1,
    payouts=(0.70, 0.30),
)
tour = SingleTableTournament(cfg)
tour.register("Alice", seat=0)
tour.register("Bob", seat=1)
# start_hand(seed) -> observe_current() -> act(E1 Action) -> settle_hand()
```

**Semantics:**

- Each registration creates a distinct `entry_id`. Re-entry does not revive a previously busted entry.
- All registrations/re-entries are permitted only **between hands** and before the cutoff. New entrants receive configured starting chips; a complete cash-equivalent ledger of entry fees is **not** implemented.
- All-in players with `stack==0` are **not eliminated mid-hand**. A bust is declared only after terminal settlement and side pots.
- Head-up SB/BTN uses the existing engine's moving-button model. No tournament dead-button model or missed-blind policy certification.
- One live entry remaining **while late registration is still open** is *not declared champion* until the window closes. The arena may wait for another entrant. `close_registration()` is provided.
- The ante setting is explicitly rejected (`NotImplementedError`) because the current Base Phase 3 hand engine does not support a tournament ante contract. **Do not add ante to rewards without implementing chip movement and pot eligibility.**
- Re-entry cannot occur before the prior entry has a confirmed hand-boundary elimination and is limited per player.
- The tournament ends only when one entry is left **and registration is closed**. `max_hands` raises rather than fabricating a winner.
- Payout ranks are based on elimination time. Simultaneous busts order by pre-hand stack; equal pre-hand stacks tie and split occupied prize slots. A re-entry is a new participant entry for this prototype's prize accounting.

### 3.1 Chip invariant

For a given hand $h$, with absent seats recorded as zero:

$$
\sum_{i=0}^{5}S_{i,h}^{\rm before}=
\sum_{i=0}^{5}S_{i,h}^{\rm after},\qquad
\sum_{i=0}^{5}R_{i,h}^{\rm chips}=0.
$$

For externally registered entries, tournament **total chips can increase between hands** when a new entry is admitted, so only per-hand invariants apply. Tournament chip gains **must not** be confused with final monetary prize utility.

### 3.2 Reward separation

Cash:

$$R^{\rm cash}_{i,h}=\frac{S^{\rm after}_{i,h}-S^{\rm before}_{i,h}}{BB_h},
\qquad \widehat{bb/100}=100\cdot\overline{R^{\rm cash}_{i,h}}.$$

Tournament:

$$R^{\rm tournament}_{i}=p_{\operatorname{rank}(i)},
\qquad \widehat J_{\rm prize}=\frac1K\sum_{k=1}^K R^{\rm tournament}_{i,k}.$$

A player who re-enters may have multiple entries. The Phase 7 report is **per-entry**, and any player-level economic EV would additionally require entry fees and rebuy cost accounting, which is out of scope.

## 4. Scientific Benchmark

Source: `src/holdem_ai/expand/integration/benchmark.py`.

### 4.1 Cash design

Each benchmark block constructs a **fresh table**, admits 2–6 players, plays possible BB-admission warm-up hands, and records **one** target hand after the entire roster is seated. Warm-up hands are counted separately and **excluded from the EV sample**. Important subtlety: warm-up dynamics condition the sampled stacks/seat states; this is a benchmark estimand, not a guaranteed uniform 100BB independent hand distribution. Position varies by `hero_seat = block_index % N`.

Benchmark axes include:

$$N\in\{2,3,4,5,6\},\qquad S_0\in\{20,100,200\}\,BB,
\qquad\mathcal O=\{\text{random, calling, tight, pot pressure}\}.$$

For independent session blocks $X_b$:

$$
\widehat{\mu}_{bb/100}=100\cdot\frac1K\sum_{b=1}^K X_b,
\quad
SE_{bb/100}=100\frac{s_X}{\sqrt K},
\quad CI_{95\%}^{\rm normal}\approx\widehat\mu\pm1.96 SE.
$$

**Caution:** The reported interval is a normal approximation. With a small number of blocks or rare large pots it can be severely inaccurate. More samples, heavy-tail diagnostics, and multiple independent model-training seeds are needed before reporting superiority.

### 4.2 Tournament design

One completed tournament is one cluster. Tournament metrics are payout EV, win rate, average number of hands, and score distributions. Cash `bb/100` is **not** reported as the tournament's primary reward. Single-table only; no multi-table merging/ICM validation.

### 4.3 Candidate-vs-baseline paired comparison

The same player count/stack/opponent/seat/deck seed is used for both frozen E3 checkpoint candidates (run separately):

$$D_b=X_b^{\rm candidate}-X_b^{\rm baseline}.$$

$$\widehat\Delta_{bb/100}=100\overline D,
\qquad SE(\widehat\Delta)=100\frac{s_D}{\sqrt K}.$$

The code rejects mismatched benchmark configs. **Matching initial RNG seeds does not guarantee identical paths once two policies take different actions.** The conditional paired confidence interval does not represent uncertainty across unseen training seeds.

### 4.4 Release Manifest

Generated JSON includes:

- Git commit SHA when the runtime is a Git checkout (otherwise null).
- E3 checkpoint exact SHA-256 and paths (plus baseline/expert files when supplied).
- Benchmark config, seed, abstraction ID.
- SHA-256 of the benchmark CLI.
- No raw hole cards or private deck in the JSON result.

Output files are protected by no-overwrite exclusive creation. Never upload training checkpoints (`*.pt`) or `runs/` wholesale by accident.

## 5. Install (Windows PowerShell)

**Prerequisite:** Expand Phase 1–6 installed in the original GitHub project's root.

```powershell
Expand-Archive .\holdem-ai-expand-phase07.zip -DestinationPath .\holdem-ai-expand-phase07 -Force

Copy-Item .\holdem-ai-expand-phase07\src\holdem_ai\expand\integration `
  .\src\holdem_ai\expand\ -Recurse -Force

Copy-Item .\holdem-ai-expand-phase07\scripts\expand_phase07_benchmark.py `
  .\scripts\ -Force

Copy-Item .\holdem-ai-expand-phase07\tests\expand\test_e07_*.py `
  .\tests\expand\ -Force

Copy-Item .\holdem-ai-expand-phase07\tests\expand\e07_fake_tournament.py `
  .\tests\expand\ -Force

Copy-Item .\holdem-ai-expand-phase07\docs\EXPAND_PHASE07_INTEGRATION_TOURNAMENT_BENCHMARK.md `
  .\docs\ -Force
```

Tests:

```powershell
uv run --extra dev --extra gpu python -m pytest -q -k e07
uv run --extra dev --extra gpu python -m pytest -q tests/expand
uv run --extra dev --extra gpu python -m pytest -q
```

### 5.1 Cash smoke (no E3 checkpoint required)

```powershell
uv run --extra gpu python -m scripts.expand_phase07_benchmark `
  --mode cash --groups 2 --players 2,3 --stacks-bb 100 `
  --opponents calling_station --seed 7070001 `
  --output-json runs/expand_phase07/cash_smoke_uniform.json
```

This is a uniform-legal **infrastructure smoke**, not a trained AI evaluation.

### 5.2 Cash full experiment with trained E3 checkpoint

Only after actual E3 training has produced the file:

```powershell
uv run --extra gpu python -m scripts.expand_phase07_benchmark `
  --mode cash --groups 300 --seed 7071000 `
  --checkpoint runs/expand_phase03_pilot/best.pt `
  --device cuda `
  --players 2,3,4,5,6 --stacks-bb 20,100,200 `
  --opponents random,calling_station,tight,pot_pressure `
  --output-json runs/expand_phase07/cash_ppo_300.json
```

Total target hands: $5\times3\times4\times300=18,000$, PLUS admission warm-ups. Start small before attempting 300 groups, especially on GTX 1070.

### 5.3 Optional paired PPO model comparison

```powershell
uv run --extra gpu python -m scripts.expand_phase07_benchmark `
  --mode cash --groups 100 --players 2,3,4,5,6 `
  --stacks-bb 100 --opponents random,calling_station,tight,pot_pressure `
  --checkpoint runs/expand_phase03_main/best.pt `
  --baseline-checkpoint runs/expand_phase03_pilot/best.pt `
  --device cuda --seed 7072000 `
  --output-json runs/expand_phase07/ppo_paired_100.json
```

The paths are examples. The script does not assume those checkpoints already exist.

### 5.4 Optional E6 adaptive experts

```powershell
uv run --extra gpu python -m scripts.expand_phase07_benchmark `
  --mode cash --groups 100 --players 2,3 --stacks-bb 100 `
  --opponents tight,pot_pressure --seed 7073000 `
  --checkpoint runs/expand_phase03_main/best.pt `
  --adaptive `
  --expert-checkpoint fold_heavy=runs/experts/fold.pt `
  --expert-checkpoint aggressive=runs/experts/aggressive.pt `
  --device cuda `
  --output-json runs/expand_phase07/adaptive_100.json
```

These expert files are **not** part of any delivered Phase 1–7 patch. They would need separate training and verification. With insufficient public evidence E6 will remain on the base policy.

### 5.5 Tournament smoke and benchmark

```powershell
uv run --extra gpu python -m scripts.expand_phase07_benchmark `
  --mode tournament --groups 2 --players 2,3 `
  --opponents calling_station --seed 7074000 `
  --output-json runs/expand_phase07/tournament_smoke_uniform.json
```

For real E3 candidate:

```powershell
uv run --extra gpu python -m scripts.expand_phase07_benchmark `
  --mode tournament --groups 100 --players 2,3,4,5,6 `
  --opponents random,calling_station,tight,pot_pressure `
  --checkpoint runs/expand_phase03_main/best.pt `
  --device cuda --seed 7075000 `
  --output-json runs/expand_phase07/tournament_ppo_100.json
```

Note: Tournament CLI evaluates the default **registration-closed, single-entry** configuration; late-registration/re-entry is tested via the `SingleTableTournament` API and is not an independent benchmark-axis CLI option yet.

## 6. Release acceptance and explicit gates

| Gate | Evidence | State |
|---|---|---|
| G1 Python source compiled | compileall | pass in sandbox |
| G2 E7 isolated tests | 23 passed | pass in sandbox |
| G3 E1–E7 combined isolated regression | 164 passed, 7 skipped | pass in sandbox |
| G4 Real Base Phase 3 hand engine tests | `test_e07_real_engine.py` | **pending** |
| G5 Windows GTX 1070 actual CUDA E3 inference | GPU pilot | **pending** |
| G6 Cash 2–6 / Side Pot / Join / Leave / Bust | full 6-Max evaluation | **pending** |
| G7 Tournament blind-level / registration / reentry real-engine traces | real engine | **pending** |
| G8 Robust E4 exact infoset mapping | old E2 raw ID preservation gap | **blocked** |
| G9 E5 full Pluribus nested search | prototype scope | **not implemented** |
| G10 Push to GitHub | only after user's own verification | **pending** |

## 7. Known constraints / planned remediation

1. Preserve **exact observed engine action IDs + raise-to amounts** in an actor-allowlisted public history without leaking hidden cards; then validate E4 key equality with game engine under controlled histories.
2. Replace Phase 4's hash-bucket prototype with learned/equity-based card abstraction, and re-evaluate large action trees and resource budgets.
3. Add full E5 Pluribus-style betting-round replay, safe search subgames, and more accurate continuation value computation.
4. Explicit ante support needs Base Hand Engine contract changes; do **not** silently charge ante outside the engine.
5. Add tournament fee/re-entry cost ledger for real player-level monetary EV; do not treat payout-only per-entry reward as net profit.
6. Benchmark with multiple independent training seeds and difficult learned opponents (not just fixed scripted profiles).
7. Compare Adaptive vs non-adaptive using predeclared held-out scenario blocks **after** actual exploit experts are trained.
8. Cash-game benchmark presently scores one hand *after* warm-up seating (conditional stack distribution). Use direct full-table seeded reset or stratification for truly fixed-stack experiments.
9. No support for cross-table migration, hand-for-hand, 7+ players, rake, or payout accounting in actual money.

## 8. Source attribution

- Base Phase 3: `src/holdem_ai/environment/multi_player_engine.py`, `multi_player_seats.py`, `pot_manager.py`.
- Expand E1: dynamic cash table & event ledger.
- Expand E2: acting-player InformationState and action translator.
- Expand E3: trusted PPO checkpoint inference only.
- Expand E4/E5: optional CFR/search sources with **explicit compatibility evidence**.
- Expand E6: public-only opponent registry, change detection, KL/TV guard.
- Pluribus: Brown & Sandholm, *Superhuman AI for Multiplayer Poker* (2019), https://noambrown.github.io/papers/19-Science-Superhuman.pdf ; Supplement https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf . The E7 design is **our own extension** and is not the Pluribus source code.
