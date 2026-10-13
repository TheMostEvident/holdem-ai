---
title: Poker AI — Technical Report
date: 2026-10-13
status: canonical-index
---

# Poker AI — Technical Report

This is the **technical report entry point** for the HUNL and Tournament AI project. Read the sources in this order:

1. [Final integrated research and engineering report](POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md) — phase objectives, mathematical contracts, implementation map, reward/strategy derivations, experiments, risk and acceptance boundaries.
2. [Verified Python source excerpts](PHASE_VERIFIED_PYTHON_CODE_REFERENCE.md) — **actual repository source**, selected from core phases and explained by execution role.
3. [Original Agent report](AGENT_DEVELOPMENT_TECHNICAL_REPORT_2026-10-13.md) — the detailed Base P0–17 and Expand E1–E7 research record.
4. [Original Tournament Environment report](TOURNAMENT_ENV_PHASE1_FULL_TECHNICAL_REPORT.md) — detailed ENV 1.1–1.7 history, API, bot fix, SQLite, replay and GUI QA.
5. [Original Expand design](EXPAND_MASTER_DESIGN_6MAX_v1_0.md) — detailed specification and proposed contracts.

## Mathematical nucleus

$$
s_{t+1}=T(s_t,a_t,\omega_t),\quad a_t\in\mathcal A_{\rm legal}(s_t)
$$

$$
\pi_\theta(a\mid I)=\frac{\exp(\ell_\theta(I,a))\,\mathbf1[a\in\mathcal A(I)]}
{\sum_{b\in\mathcal A(I)}\exp(\ell_\theta(I,b))}
$$

$$
\sum_i S_{i,t}^{behind}+Pot_t=C_t,\quad R_{i,h}^{BB}=\frac{S_{i,h}^{final}-S_{i,h}^{initial}}{BB_h}
$$

$$
\widehat{bb/100}=\frac{100}{H}\sum_{h=1}^{H}R_{i,h}^{BB}
$$

The equations define **correctness contracts**, not a claim of a solved game or a specific strategy's out-of-sample strength. Treat HUNL Agent, Expand multi-seat Agent, HUNL GUI, and Tournament Environment as distinct development tracks. Detailed phase explanation and verified code are linked above.

## Report preservation policy

All existing granular \`docs/\` reports are retained until **lossless text coverage and source provenance** are checked. The current integrated report is a substantive synthesis but is not yet a complete exact copy of every previous document or every tracked Python file. No historical report should be deleted merely because this summary exists.

For commands and phase-specific test matrices refer to the linked source reports. Historical test counts are attributed to their original phase; this update does not claim newly executed end-to-end tests.
