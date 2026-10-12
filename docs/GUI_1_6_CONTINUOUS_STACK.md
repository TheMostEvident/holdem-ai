# GUI-1.6 — Continuous Stack Play Mode

> Project: `holdem-ai` / HUNL 2-player / Phase 0–16 compatible GUI patch
>
> Scope: Play sessions and frontend only. **Do not modify** the Phase-6 `HUNL200Adapter.reset()` or Phase 9–16 RL training/evaluation defaults.

## 1. Modes and invariants

| Area | Default | Beginning of hand $h$ | Interface |
|---|---|---|---|
| Local browser Play | `continuous` | Previous hand's settled final stacks | React + FastAPI |
| Python `HUNLGameSession()` / legacy independent research use | `independent` | `(200BB, 200BB)` | Python |
| Phase 6–16 training/evaluation | independent, unchanged | `(200BB, 200BB)` | Python CLI |

First Play hand with `stack_bb=200`, `BB=100` starts with 20,000 chips per player. For player $i\in\{0,1\}$ and hand $h$:

$$
S_{i,h+1}^{\mathrm{start}} = S_{i,h}^{\mathrm{final}}
$$

$$
R_{i,h}^{\mathrm{chips}}=S_{i,h}^{\mathrm{final}}-S_{i,h}^{\mathrm{start}}
$$

$$
\sum_{i=0}^1 S_{i,h}^{\mathrm{start}} = \sum_{i=0}^1 S_{i,h}^{\mathrm{final}} = 2 \times 200 \times 100=40{,}000.
$$

The button rotates each hand: $BTN_h=(h-1)\bmod2$. Blinds stay 50/100. There is no rake, rebuy, or cash top-up in GUI-1.6.

### Match terminal condition

$$
\mathrm{match\_over}=\mathbb{1}\left[\min_i(S_{i,h}^{\mathrm{final}})=0\right].
$$

When match_over is true, `/next-hand` rejects with HTTP 409. The UI hides the **Deal next hand** button and displays the winning seat. Start a **new session** to play again.

**Short stacks:** If a player's stack is smaller than a forced blind, the Phase-3 engine handles the short blind as an all-in. A hand can be dealt and settled *without any player decision*. Such a hand still advances the session `completed_hands`, cumulative chips/BB, and history.

## 2. Architecture

```text
frontend/src/App.tsx
  ├─ Play Mode (continuous HUNL; selectable bot)
  └─ Training / Research (CLI guidance; no training browser)
       |
       | POST /api/v1/hunl/sessions  {"stack_mode":"continuous", ...}
       v
backend/game_router.py + backend/schemas.py
       |
       v
HUNLGameSession(stack_mode="continuous")
       |
       ├─ Hand 1: InteractiveHUNLAdapter.reset() (original 200BB)
       |
       └─ Hand 2+: InteractiveHUNLAdapter.reset_with_stacks(
                       preceding_hand.result.final_stacks)
                             |
                             v
                 MultiPlayerHandEngine(stacks=(hero, bot), ...)
```

The GUI adapter's `reset_with_stacks()` *only* modifies the interactive GUI path and calls the Phase-3 engine's existing `stacks` argument. The Phase-6 training adapter's independent reset path remains intact. Trained bots still use the existing legal-action abstraction (7 action indices). Exact human `raise_to` passes through the existing engine validation.

## 3. API additions (backward compatible request)

`POST /api/v1/hunl/sessions` request body:

```json
{
  "bot": "calling_station",
  "seed": 42,
  "small_blind": 50,
  "big_blind": 100,
  "stack_bb": 200,
  "stack_mode": "continuous"
}
```

For compatibility with existing HTTP clients, absent `stack_mode` **defaults to `continuous` on the GUI server**. Research scripts that instantiate `HUNLGameSession` directly without specifying a mode keep independent default. The server accepts explicit `"stack_mode":"independent"` for legacy local tests/diagnostics, but the browser's Play mode *always sends `continuous`*.

Additional public fields in `GameView`:

```json
{
  "stack_mode": "continuous",
  "starting_stacks": [20000, 20000],
  "match_over": false,
  "winner_seat": null
}
```

`HandResultView.final_stacks` gives **settled** stacks, including the pot award. On a terminal view `seats[].stack` now also displays these settled stacks; earlier versions could display the remaining in-hand balance before awards. Each history item includes the corresponding `starting_stacks` for auditability.

On a zero-stack bust `match_over=true` and `winner_seat` becomes `0` (human) or `1` (bot). No opponent hole-card data is revealed on a fold; opponent hole cards are only disclosed after a genuine showdown.

**Compatibility:** The `game_type` response field remains `HUNL200` for the existing API/head-up seven-action policy family. Use `stack_mode` to distinguish the opening-stack dynamics; `HUNL200` in this field is not a promise that every subsequent hand has exactly 200BB.

## 4. Trained Agent limitations

A PPO / Q-Boost checkpoint trained under **two 200BB starting stacks** may encounter off-distribution situations in Continuous Play (unequal stacks, short stacks, deep stacks). The engine stays rule-correct, but **the policy is not guaranteed to remain strong or calibrated** outside the training distribution.

$$
\pi_\theta(a\mid I_h,\mathcal A_h^{\mathrm{legal}})
$$

The network, training script and optimization algorithm are unchanged in this patch. Variable-stack curriculum/self-play would be a separate research change.

## 5. Install (Windows PowerShell)

**Existing GUI-1.5 users**: apply only the GUI-1.6 incremental ZIP over the **root** `holdem-ai/` folder.

```powershell
cd C:\Users\yiydg\Desktop\workspace\holdem-ai-starter\holdem-ai
Expand-Archive -Path "$HOME\Downloads\holdem-ai-gui-1-6.zip" -DestinationPath . -Force

uv run --extra dev --extra gui --extra gpu python -m pytest -q
uv run --extra dev --extra gui --extra gpu python -m scripts.gui16_smoke --hands 20 --seed 42
```

**Full GUI 1.0–1.6 install**: apply only the combined `holdem-ai-gui-1-0-to-1-6.zip` to an existing Phase 0–16 root, **not** to an empty directory. The full ZIP includes the corrected GUI-1.5 TS patch. Dependencies are set up as in GUI-1.5 (`uv add --optional gui ...`, `uv sync --extra dev --extra gui --extra gpu`).

Launch in two terminals:

```powershell
# Terminal A - root
uv run --extra gui --extra gpu python -m backend.main

# Terminal B - root/frontend
cd frontend
npm.cmd run typecheck
npm.cmd run build
npm.cmd run test
npm.cmd run dev
```

Navigate to `http://127.0.0.1:5173`. Select **Play Mode** (default), choose a bot, and start a session. **Training / Research** selection shows Python CLI guidance; it does not start GPU training from a browser.

## 6. Test matrix and limitations

- GUI-1.6 test cases: default continuous, ten sequential hands, exact carryover, button alternation, chip conservation, legacy independent mode, all-in bust, revision-safe rejection, unchanged training adapter reset, short-stack zero-decision terminal.
- HTTP smoke: 20 consecutive hands with monotone revision and exact `starting_stacks == previous.final_stacks`.
- Long-run edge smoke: the sample seed and always-fold strategy runs until bankroll exhaustion. A verification run settled 267 hands, including a short blind auto-showdown, and ended with `[0, 40000]` chips.
- Tests in the assistant's container used historical Phase-2/3 engine files and **lightweight stand-ins for unavailable Phase-5/6 imports**. They do not constitute a verified build of the full latest GitHub repository on Windows.
- Full production TypeScript build was not run in the assistant's container because npm dependencies were not available. The user's Windows machine must run `npm.cmd run typecheck`, `npm.cmd run build`, and `npm.cmd run test`.

**Session/analytics persistence:** still in-memory and lost when local FastAPI server stops. This patch does not add on-disk bankroll saving or cross-session rebuys.
