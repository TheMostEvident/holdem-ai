---
title: Poker AI — Technical Report Index
date: 2026-10-13
status: lossless-source-verified
---

# TECH REPORT — Reading paths by objective

**English** | [한국어](TECH_REPORT.ko.md)

GitHub equations use protected display and inline delimiters to preserve TeX escapes. For Obsidian, replace each opening `math` fence and its matching closing fence with `$$`; remove the protecting backticks from inline equations. Keep the TeX body unchanged.

```math
\boxed{\text{Game rules }T\ \longrightarrow\ \text{Observation }I\ \longrightarrow\ \text{Policy }\pi_\theta\ \longrightarrow\ \text{Action }a\ \longrightarrow\ \text{Settlement }u}
```

Game rules create states; observations restrict each player's information; policies select legal actions; and settlement supplies rewards for training and evaluation. This document is the entry point to the [Final Integrated Technical Report](POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Detailed explanations of every phase and all original sources are linked through the final report and its appendices.

## Projects and phases by objective

| What you want to understand | Project and phase to read | Key checks |
|---|---|---|
| Game rules and state transitions | Base Environment P0–P4 | Cards, action order, raises, side pots, showdown, and chip conservation |
| Heads-up actor and critic implementation | HUNL Agent P5–P8 | InformationState, 7-action mask, token encoder, Transformer, and information boundaries |
| Differences between PPO and QBoost | HUNL Agent P9–P10 | Terminal MC return, old policy, clipped loss, and control variate |
| Training efficiency and evaluation design | HUNL Agent P11–P14 | Batched collector, time budget, checkpoints, seed blocks, and held-out data |
| Policy collapse and critic problems | HUNL Agent P15–P16 | Action denominators, rewards, KL, A0–A3, stress tests, and rollback |
| Continuous stacks and retraining | HUNL Agent P17a/P17b | Separate the evaluation interface from the GPU training experiment |
| Cash tables with 2–6 players | Expand E1–E3 | Hand-boundary events, changing seats, multiplayer observations, and PPO/GAE |
| CFR, search, and opponent adaptation | Expand E4–E7 | External sampling, belief, budgets, policy routing, and actual-engine validation |
| Human–HUNL agent GUI | HUNL GUI G1.0–G1.6 | Frozen actor, API revision, statistics, and stack carry |
| Six-player tournament GUI | Tournament ENV T1.1–T1.7 | Bots, actor bridge, elimination ranks, blind levels, SQLite, replay, and card disclosure |
| Complete original code or documents | Source indexes in the final report | Open each of the 327 Python files and 76 Markdown documents individually |

## How to read each phase

```math
\boxed{\text{Equations}\rightarrow\text{Variables}\rightarrow\text{Purpose and value}\rightarrow\text{Actual code}\rightarrow\text{Execution flow}\rightarrow\text{Validation and limitations}}
```

Equations define the implementation contract. Variable definitions identify the data needed. The purpose states the problem that the phase solves. Actual code and call order show how inputs become results. Validation and limitations distinguish implemented behavior, historical test results, and practical playing strength.

Base P9/P10 use terminal Monte Carlo returns for HUNL training, unlike Expand E3's GAE training. P17a covers continuous-stack evaluation; P17b covers GPU retraining. Connecting Expand E3 to Tournament performs frozen inference from a cash-game checkpoint.

## Shared mathematical contracts

```math
s_{t+1}=T(s_t,a_t,\omega_t),\quad a_t\in A_{\mathrm{legal}}(s_t)
```

The environment transitions states through legal actions and separately handles card-dealing chance events $`\omega_t`$.

```math
\pi_\theta(a\mid I)=\frac{\mathbf1[a\in A(I)]e^{\ell_\theta(I,a)}}{\sum_{b\in A(I)}e^{\ell_\theta(I,b)}}
```

The actor forms probabilities from its own information set $`I`$ and the legal-action mask. Private information available to the training-only critic must not enter the actor's inference input.

```math
\sum_i S_{i,t}^{\mathrm{behind}}+P_t=C_h,\quad
R_{i,h}^{BB}=\frac{S_{i,h}^{\mathrm{final}}-S_{i,h}^{\mathrm{start}}}{BB_h},\quad
\widehat{bb/100}=\frac{100}{H}\sum_h R_{i,h}^{BB}
```

Chips are conserved when there is no rake or external transfer of funds during a hand. Distinguish returns normalized by $`BB`$ from placement rewards, and align the independent evaluation unit with the experiment's seed blocks.

## Complete sources and lossless verification

```math
\mathrm{GitBlobSHA}(b)=\mathrm{SHA1}(\mathtt{blob\ }\Vert\mathrm{decimal}(|b|)\Vert\mathtt{NUL}\Vert b)
```

$`b`$ denotes the original file's UTF-8 bytes. Calculate length and SHA excluding the LF used to separate the source from its closing fence. Each source path must occur exactly once, and its byte count and SHA must match the original tree.

| Verification item | Result |
|---|---:|
| Original Python sources | 327/327 files · 1,400,837 bytes |
| Original Markdown sources | 76/76 files · 1,049,564 bytes |
| Omissions, duplicates, or source SHA mismatches | 0 |
| Superseded standalone reports | 73 deleted; original text included in the final report's appendices |
| Python appendices | 28 logical volumes / 29 files; volume 25 split into A/B |
| Markdown appendices | 7 logical volumes / 8 files; volume 6 split into A/B |

The source reference is `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. The [manifest](lossless_source_manifest.json) provides all original paths, SHAs, byte counts, appendix paths, anchors, and the deletion list. The final report includes Python code to reproduce the same verification.

This documentation work checked original-source integrity, syntax, links, and change scope. Pytest, frontend, and training figures in the earlier documents are historical records; they do not represent reruns of that training or the complete runtime test suite for this documentation work.
