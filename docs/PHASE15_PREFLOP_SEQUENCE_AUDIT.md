# Phase 15 — Preflop Sequence Diagnosis

Read-only Agent audit; existing engine/GUI, training/optimizer, previous runs and checkpoints are untouched.

## Why

The previous 512-hand-per-model comparison found:

| Metric | Uniform Legal | Random Initialization | Trained Phase14 |
| --- | ---: | ---: | ---: |
| Preflop decisions | 1397 | 1335 | 2267 |
| Explicit Preflop All-in | 218 | 218 | 438 |
| All-in legal opportunities | 1179 | 1117 | 1829 |
| Preflop terminal folds | 318 | 353 | 53 |
| Preflop terminal calls | 97 | 84 | 433 |
| Pot raises | 193 | 176 | 523 |
| Double-pot raises | 175 | 224 | 545 |

In Phase14:

\[
P(\mathrm{Preflop\ Allin\ per\ hand})=438/512\approx0.8555
\]

\[
P(\mathrm{Allin\ selected}\mid\mathrm{legal\ decision})
=438/1829\approx0.2395
\]

The denominators differ. The new audit reconstructs every preflop sampled action, for each hand.

## New diagnostic outputs

- Most common **complete preflop action-head sequences** (each hand counts once).
- Most common **action-head prefixes ending at the first explicit preflop All-in**.
- First All-in decision index histogram and prior action distribution.
- Number of actions prior to the hand's first All-in; number of actions for hands ending in a Preflop Fold.
- Selected action counts and legality-conditioned action rates by preflop decision index.
- All seven action choices conditional on public outstanding bet (to-call > 0 versus zero), and button/BB position.
- Counts of hands whose last Actor decision was Preflop and whose final action was Fold or Call.

Uses only actor-visible preflop observations from the existing frozen Agent rollout collector. No private cards, CriticState or hidden deck enter the Actor.

## Interpret carefully

- A 7-head policy action index 6 means an **explicit** all-in; an ordinary CALL can also exhaust the stack.
- A sampled Raise head sequence is NOT automatically a formal 3-bet/4-bet classification or an exact wager amount.
- A preflop last action may be followed by a full-board automatic runout before showdown.
- Each policy plays both seats against itself. Same deck seeds do not force identical action histories.
- Random Initialization is a *synthetic fresh initialization*, not necessarily the historical training-start checkpoint.

## PowerShell

First:

~~~powershell
git pull --ff-only origin agent/phase15-collapse-diagnostics
uv run --extra gpu pytest tests/test_phase15_preflop_sequence_audit.py -q
uv run --extra gpu pytest -q
~~~

If tests pass:

~~~powershell
uv run --extra gpu python -m scripts.phase15_preflop_sequence_audit --trained runs/phase14_main/seed_52/ppo/best.pt --init-seed 52 --device cuda --hands 512 --seed 600000000 --output runs/phase15_audit/phase14_preflop_sequences.json
~~~

New report output is exclusive. No training or saved-checkpoint modification.

Inspect:

~~~powershell
$r = Get-Content .\runs\phase15_audit\phase14_preflop_sequences.json -Raw | ConvertFrom-Json
$r.models.trained_phase14.first_allin_decision_number_counts
$r.models.trained_phase14.immediately_before_first_allin
$r.models.trained_phase14.most_common_prefixes_through_first_allin | Format-Table -AutoSize
$r.models.uniform_legal.most_common_prefixes_through_first_allin | Format-Table -AutoSize
~~~

Only after observing the dominant sequences should we design a controlled PPO/self-play ablation.
