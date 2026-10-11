# Phase 7 — VRPO-Lite v1 Observation & Action Encoding

> Scope: **tensor input pipeline only**. This patch does **not** implement the VRPO Transformer, Q-Critic, Q-Boosting, PPO, or training. It uses the **Phase 6 fixed 7-action abstraction**, not the paper's base-16 multi-step bet representation.

## 1. Contracts and privacy boundary

Phase 6 provides `InformationState` (player-visible) and `CriticState` (training-only). These are intentionally separate functions:

- `encode_actor(observation: InformationState) -> EncodedActor`
- `encode_critic(state: CriticState) -> EncodedCritic`

The actor has no dependency on, argument for, or direct reference to `CriticState` or the mutable engine. The caller must obey this separation when building Phase 8 networks. A privileged `CriticState` is obtainable only through `HUNL200Adapter(expose_training_state=True)` for training collectors.

$$
I_t^i = O_i(s_t, h_t^{\mathrm{public}})
$$

$$
\mathrm{ActorInput}_t^i = E_A(I_t^i)
\qquad
\mathrm{CriticInput}_t = E_C(s_t)
$$

This separation is a software boundary, not a formal proof of information-set perfection: Phase 6 currently stores coarse action history, and **all-in/call amounts are not fully recorded in every history entry**. Before claiming perfect recall, extend the source action history with exact public contributions and test all betting branches.

## 2. Vocabulary (`vocabulary.py`)

The stable 73-token **project-specific vocabulary** is not claimed to match the original VRPO paper's encoding.

| IDs | Meaning |
|---|---|
| 0 | PAD |
| 1–11 | sequence/role/seat/control markers |
| 12–63 | exactly 52 canonical cards (`s/d/c/h` × `23456789TJQKA`) |
| 64–67 | preflop, flop, turn, river |
| 68–72 | fold, check, call, raise_to, all_in |

Action history is parsed exactly from Phase 6:

`preflop|seat=0|raise_to|raise_to=450`

This becomes `[EVENT, STREET_PREFLOP, SEAT_0, ACTION_RAISE_TO, AMOUNT]`, with the final token's numeric `token_value` equal to `450/100 = 4.5` BB. Other token values are zero. An invalid history record raises `ValueError` rather than being silently ignored.

`max_tokens=256` by default. History is **never silently truncated**; exceeding capacity raises `ValueError`. Increase `max_tokens` for longer hands, and remember that learned positional embedding size must accommodate it in Phase 8.

## 3. Actor shape

`encode_actor(obs, big_blind=100, max_tokens=256)` returns:

| Field | Type / shape | Description |
|---|---|---|
| `token_ids` | tuple[int, T] | own two hole cards, visible board, street, public action sequence |
| `token_values` | tuple[float, T] | public raise-to target normalized by BB; zero on other tokens |
| `features` | tuple[float, 20] | 12 public Phase 6 features plus 2×4 status one-hot |
| `legal_action_mask` | tuple[bool, 7] | Phase 5/6's fixed discrete action head |

The first 12 scalar features, in their existing Phase 6 order:

`[acting_seat, is_button, street_index, pot_bb, to_call_bb, own_street_bet_bb, own_stack_bb, opponent_stack_bb, opponent_street_bet_bb, own_contribution_bb, opponent_contribution_bb, min_raise_to_bb]`.

Remaining eight features: statuses for seats 0,1, each one-hot in order `[active, folded, allin, out]`. Actor encodings are not permitted to include the opponent's hole cards, burned cards, or undealt deck.

## 4. Critic shape

`encode_critic(privileged_state, big_blind=100, max_tokens=256)` returns:

| Field | Type / shape | Description |
|---|---|---|
| `token_ids` | tuple[int, T'] | **both players'** hole cards, board, history, optional terminal marker |
| `token_values` | tuple[float, T'] | public raise-to amounts / BB |
| `features` | tuple[float, 24] | centralized game data normalized in BB |

Critic feature positions:

- `[0:2]`: one-hot button seat (2)
- `[2:4]`: one-hot acting seat, both zero at terminal (2)
- `[4:8]`: one-hot street (4)
- `[8:10]`: stacks / BB (2)
- `[10:12]`: total contributions / BB (2)
- `[12:14]`: current street bets / BB (2)
- `[14]`: pot / BB (1)
- `[15:23]`: 2×4 status one-hots (8)
- `[23]`: terminal flag (1)

$$
2+2+4+2+2+2+1+8+1 = 24
$$

No future deck order is included, even in the Critic state. The Critic input is centralized by design and **must not** be passed to the player-facing Actor.

## 5. Batching and legal actions

`batch_actors([encoded...], device='cuda')` yields:

- `token_ids`: `torch.int64[B,T_max]`
- `token_values`: `torch.float32[B,T_max]`
- `attention_mask`: `torch.bool[B,T_max]`, true for real tokens
- `features`: `torch.float32[B,20]`
- `legal_action_mask`: `torch.bool[B,7]`

`batch_critics(...)` uses the same first three tensors, with `features[B,24]`.

The `T_max` dimension is batch-local: shorter examples are right-padded, and only pad tokens receive `attention_mask=False`.

`encode_action(action_id, legal_mask)` returns a 7-dimensional one-hot tuple, refusing all illegal indices. `masked_categorical(logits, legal_mask)` builds a valid distribution over only legal actions:

$$
\pi_\theta(a\mid I)=\frac{m_a\exp(z_\theta(I,a))}{\sum_b m_b\exp(z_\theta(I,b))}
$$

where `m_a` is the zero/one legal-action mask. Illegal choices get probability zero and zero gradient through the masked logit.

## 6. Tests and commands (Windows / uv)

From the actual repository root (`pyproject.toml` is present):

```powershell
uv run --extra dev --extra gpu python -m pytest -q -k phase7
uv run --extra dev --extra gpu python -m pytest -q
uv run --extra gpu python -m scripts.phase7_smoke --hands 100 --device cuda
```

The Smoke Test performs **real Phase 6 HUNL200 random-policy episodes**, actor/critic encoding, target-state encoding, batched CUDA tensors, mask-correct sampling, and a single forward/backward through a tiny **synthetic head**. It is not VRPO model training. `--device cpu` is available.

## 7. Acceptance criteria

1. All Phase 7 tests pass on the user's **real existing HUNL engine**.
2. No existing Phase 1–6 regression.
3. Actor tokens/features never contain opponent hidden hole cards.
4. Critic tensors include both hole cards; no future deck.
5. Phase 6 history ordering and supplied exact `raise_to` amounts are preserved; no implicit truncation.
6. Action mask never selects illegal action, including synthetic gradient test.
7. GTX 1070 8GB can allocate tensor batches and complete synthetic head backward.

## 8. Known limitations and next steps

- Existing Phase 6 `history` sets `raise_to='-'` for `all_in`; the action is identifiable but its **exact public wager amount is missing** from the Phase 6 string. Standard `call` history also does not carry the paid chip amount. Fix this in a dedicated history-schema change before perfect-recall claims or research-grade action-conditioned Q inputs.
- The discrete action space is 7 heads; the paper's base-16 Multi-Step Action Encoding is intentionally postponed.
- Full historical `max_tokens` above 256 raises an error; the Phase 8 Transformer must support configurable positions/relative positions.
- These are *v1 data encoders*: Phase 8 builds/benchmarks real Actor and Critic; Phase 9 begins PPO training; Phase 10 introduces VRPO Q-Boosting.
- In the assistant's isolated environment tests were executed with a substitute hand engine. **Only the user's Windows full suite and real-engine smoke can close the integration validation.**
