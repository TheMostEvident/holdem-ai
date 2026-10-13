# Complete Python source volume 16/28

[Integrated technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Pinned source commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. All code literal and complete.

## src/holdem_ai/game_service/state_projection.py

Original blob `89bd50d93587ac4dc2831a388a52a4fd77315d2f` (4653 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/game_service/state_projection.py" sha="89bd50d93587ac4dc2831a388a52a4fd77315d2f" -->

````````python
"""Explicit whitelist of game data safe to serialize to the human browser.

This is a TRUSTED SERVER boundary: we can read both seats' stacks and the
human's private cards, but we must never serialize the future deck, burned
cards, training critic state or live opponent hole cards. The bot separately
receives its own InformationState directly from HUNL200Adapter.
"""
from __future__ import annotations

from typing import Any

from holdem_ai.core.schema import Terminal, Turn

from .action_adapter import legal_gui_actions


def project_human_view(
    adapter: Any, transition: Turn | Terminal,
    *, hero_seat: int, hand_number: int, revision: int,
    action_log: list[dict[str, Any]], completed_hands: int,
    cumulative_bb: float, stack_mode: str = "independent",
) -> dict[str, Any]:
    """Return a fresh JSON-serializable object with an intentional field list."""
    if hero_seat not in (0, 1):
        raise ValueError("HUNL has exactly two physical seats")
    engine = adapter.engine  # trusted server code only; never return this object
    over = isinstance(transition, Terminal)
    opponent = 1 - hero_seat
    street = engine.street
    result = engine.get_result() if over else None
    total_chips = 2 * adapter.stack_bb * adapter.big_blind
    if sum(engine.initial_stacks) != total_chips:
        raise AssertionError("invalid match chip sum")
    match_over = bool(stack_mode == "continuous" and over and min(result.final_stacks) == 0)
    winner_seat = (result.final_stacks.index(max(result.final_stacks)) if match_over else None)
    # The human always sees their own cards, even while waiting or after folding.
    view: dict[str, Any] = {
        "hand_number": hand_number,
        "revision": revision,
        "phase": "complete" if over else "playing",
        "game_type": "HUNL200",
        "stack_mode": stack_mode,
        "starting_stacks": list(engine.initial_stacks),
        "match_over": match_over,
        "winner_seat": winner_seat,
        "hero_seat": hero_seat,
        "button": engine.button,
        "small_blind": engine.small_blind,
        "big_blind": engine.big_blind,
        "street": street,
        "board": list(engine.board),
        "hero_cards": list(engine.hole_cards[hero_seat]),
        "seats": [
            {"seat": i, "role": "human" if i == hero_seat else "bot",
             "stack": int(result.final_stacks[i] if over else engine.stacks[i]), "street_bet": int(engine.street_bets[i]),
             "total_contribution": int(engine.contributions[i]),
             "status": str(engine.status[i])}
            for i in (0, 1)
        ],
        "pot": int(sum(engine.contributions)),
        "to_act": None if over else int(transition.player_id),
        "your_turn": not over and transition.player_id == hero_seat,
        "actions": [],
        "to_call": None,
        "min_raise_to": None,
        "max_raise_to": None,
        "can_raise_to": False,
        "action_log": [dict(event) for event in action_log],
        "completed_hands": completed_hands,
        "cumulative_bb": cumulative_bb,
        "bb_per_100": 100.0 * cumulative_bb / completed_hands if completed_hands else 0.0,
        "hand_result": None,
    }
    if not over and transition.player_id == hero_seat:
        obs = transition.observation
        # Verify that acting-player observation is the human's and not the bot's.
        if obs.player_id != hero_seat:
            raise AssertionError("human GUI cannot receive bot observations")
        raw = engine.observe(hero_seat)["raw_obs"]
        view["to_call"] = int(raw["to_call"])
        can_raise = 4 in engine.legal_actions
        view["can_raise_to"] = can_raise
        if can_raise:
            view["min_raise_to"] = int(raw["min_raise_to"])
            view["max_raise_to"] = int(raw["street_bets"][hero_seat]) + int(raw["stacks"][hero_seat])
        view["actions"] = legal_gui_actions(obs, engine)
    if over:
        assert result is not None
        if sum(result.payoffs_chips) != 0 or sum(result.final_stacks) != total_chips:
            raise AssertionError("chip conservation violated")
        is_showdown = not any(result.folded) and len(result.board) == 5
        view["hand_result"] = {
            "payoffs_chips": list(result.payoffs_chips),
            "payoffs_bb": list(result.payoffs_bb),
            "final_stacks": list(result.final_stacks),
            "awards": list(result.awards),
            "showdown": is_showdown,
            # Revealing the opponent is allowed only after a genuine showdown.
            "opponent_cards": list(engine.hole_cards[opponent]) if is_showdown else None,
        }
    return view

````````

<!-- END_PY_SOURCE path="src/holdem_ai/game_service/state_projection.py" -->

## src/holdem_ai/models/__init__.py

Original blob `c45dbcdd525a3fadb2abb368e5264f09bb51e773` (39 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/models/__init__.py" sha="c45dbcdd525a3fadb2abb368e5264f09bb51e773" -->

````````python
"""Consolidated functional package."""

````````

<!-- END_PY_SOURCE path="src/holdem_ai/models/__init__.py" -->

## src/holdem_ai/models/transformer.py

Original blob `9d16728ef844cd0b3e5c59ed9855237ccdbfbafe` (8612 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/models/transformer.py" sha="9d16728ef844cd0b3e5c59ed9855237ccdbfbafe" -->

````````python
"""VRPO-Lite v1 trainable networks for the existing Phase-7 batch contracts.

This is a lightweight research baseline, not a reproduction of the full
2026 VRPO paper. Actor inputs NEVER contain privileged CriticState.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from holdem_ai.encoding.actions import masked_categorical, masked_logits
from holdem_ai.encoding.observation import ACTOR_FEATURE_DIM, CRITIC_FEATURE_DIM, NUM_ACTIONS
from holdem_ai.encoding.tensors import ActorBatch, CriticBatch
from holdem_ai.encoding.vocabulary import PAD, VOCAB_SIZE


@dataclass(frozen=True)
class NetworkConfig:
    embed_dim: int = 64
    attention_heads: int = 2
    transformer_layers: int = 2
    ff_dim: int = 128
    hidden_dim: int = 128
    max_tokens: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for key in ('embed_dim', 'attention_heads', 'transformer_layers', 'ff_dim', 'hidden_dim', 'max_tokens'):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.embed_dim % self.attention_heads:
            raise ValueError('embed_dim must be divisible by attention_heads')
        if not isinstance(self.dropout, (int, float)) or not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')


class CausalTokenEncoder(nn.Module):
    """Use the *last valid* token after a causal Transformer.

    The attention mask is True for real tokens and False for right padding.
    Token amounts enter as a scalar embedding, separate from token identity.
    No history is silently truncated; oversize batches raise ValueError.
    """

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.embed_dim
        self.token_embed = nn.Embedding(VOCAB_SIZE, dim, padding_idx=PAD)
        self.position_embed = nn.Embedding(config.max_tokens, dim)
        self.value_embed = nn.Linear(1, dim, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.attention_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.transformer_layers, enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, token_ids: Tensor, token_values: Tensor, attention_mask: Tensor) -> Tensor:
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError('token_ids must be int64 [batch, time]')
        bsz, steps = token_ids.shape
        if bsz < 1 or steps < 1 or steps > self.config.max_tokens:
            raise ValueError('invalid time dimension or max_tokens exceeded')
        if token_values.shape != token_ids.shape or token_values.dtype != torch.float32:
            raise ValueError('token_values must be float32 and match token_ids')
        if attention_mask.shape != token_ids.shape or attention_mask.dtype != torch.bool:
            raise ValueError('attention_mask must be bool and match token_ids')
        if not bool(attention_mask[:, 0].all().item()):
            raise ValueError('every sample needs at least the first valid token')
        if not bool((attention_mask[:, 1:] <= attention_mask[:, :-1]).all().item()):
            raise ValueError('only contiguous right padding is supported')
        if not bool(((token_ids != PAD) == attention_mask).all().item()):
            raise ValueError('token ids must be PAD exactly at padded positions')
        if not bool(torch.isfinite(token_values).all().item()):
            raise ValueError('nonfinite token values')
        positions = torch.arange(steps, device=token_ids.device)
        x = (self.token_embed(token_ids)
             + self.position_embed(positions)[None, :, :]
             + self.value_embed(torch.asinh(token_values).unsqueeze(-1)))
        # In a PyTorch bool attention mask, True means blocked.
        causal = torch.ones((steps, steps), device=token_ids.device, dtype=torch.bool).triu(1)
        x = self.transformer(x, mask=causal, src_key_padding_mask=~attention_mask)
        last = attention_mask.long().sum(dim=1) - 1
        return self.output_norm(x[torch.arange(bsz, device=x.device), last])


class TransformerActor(nn.Module):
    """Decentralized policy: ActorBatch -> masked action logits [B,7]."""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        self.tokens = CausalTokenEncoder(self.config)
        self.policy_head = nn.Sequential(
            nn.Linear(self.config.embed_dim + ACTOR_FEATURE_DIM, self.config.hidden_dim),
            nn.GELU(), nn.Linear(self.config.hidden_dim, NUM_ACTIONS),
        )

    def forward(self, batch: ActorBatch) -> Tensor:
        if not isinstance(batch, ActorBatch):
            raise TypeError('Actor requires ActorBatch only (no privileged CriticBatch)')
        if batch.features.ndim != 2 or batch.features.shape[-1] != ACTOR_FEATURE_DIM:
            raise ValueError('Actor features must be [B,20]')
        if not bool(torch.isfinite(batch.features).all().item()):
            raise ValueError('nonfinite Actor features')
        state = self.tokens(batch.token_ids, batch.token_values, batch.attention_mask)
        logits = self.policy_head(torch.cat((state, torch.asinh(batch.features)), dim=-1))
        return masked_logits(logits, batch.legal_action_mask)

    def distribution(self, batch: ActorBatch) -> torch.distributions.Categorical:
        return masked_categorical(self.forward(batch), batch.legal_action_mask)


class CentralizedQCritic(nn.Module):
    """Privileged 2-player zero-sum Q-values in big-blind reward units.

    Output shape: [batch, seat=2, action=7].  Both players' returns
    are opposite for each full-game state/action.  At terminal states,
    bootstrap Q(s_terminal, a) is identically zero for both players.

    The Q head does not select poker actions; the Actor always determines
    legal-action probabilities using player-visible InformationState.
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        self.tokens = CausalTokenEncoder(self.config)
        self.q_head = nn.Sequential(
            nn.Linear(self.config.embed_dim + CRITIC_FEATURE_DIM, self.config.hidden_dim),
            nn.GELU(), nn.Linear(self.config.hidden_dim, NUM_ACTIONS),
        )

    def forward(self, batch: CriticBatch) -> Tensor:
        if not isinstance(batch, CriticBatch):
            raise TypeError('Critic requires training-only CriticBatch')
        if batch.features.ndim != 2 or batch.features.shape[-1] != CRITIC_FEATURE_DIM:
            raise ValueError('Critic features must be [B,24]')
        if not bool(torch.isfinite(batch.features).all().item()):
            raise ValueError('nonfinite Critic features')
        state = self.tokens(batch.token_ids, batch.token_values, batch.attention_mask)
        q0 = self.q_head(torch.cat((state, torch.asinh(batch.features)), dim=-1))
        # The final Phase-7 Critic feature is the terminal-state indicator.
        # Terminal states have no legal actions, so Q is used only as V=0.
        q0 = q0.masked_fill(batch.features[:, -1:].to(dtype=torch.bool), 0.0)
        return torch.stack((q0, -q0), dim=1)


def chosen_q(q_values: Tensor, acting_seats: Tensor, actions: Tensor) -> Tensor:
    """Gather player-conditioned Q for actual actions. All tensors batched."""
    if q_values.ndim != 3 or q_values.shape[1:] != (2, NUM_ACTIONS):
        raise ValueError('q_values must be [B,2,7]')
    if acting_seats.shape != (q_values.shape[0],) or acting_seats.dtype != torch.long:
        raise ValueError('acting_seats must be int64 [B]')
    if actions.shape != acting_seats.shape or actions.dtype != torch.long:
        raise ValueError('actions must be int64 [B]')
    if not bool(((acting_seats >= 0) & (acting_seats < 2)).all().item()):
        raise ValueError('invalid player index')
    if not bool(((actions >= 0) & (actions < NUM_ACTIONS)).all().item()):
        raise ValueError('invalid action index')
    rows = torch.arange(q_values.shape[0], device=q_values.device)
    return q_values[rows, acting_seats, actions]

````````

<!-- END_PY_SOURCE path="src/holdem_ai/models/transformer.py" -->

## src/holdem_ai/tournament_service/__init__.py

Original blob `8f32b7a86dae0351f40a7e7b4ecdd9741795e0bb` (217 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/tournament_service/__init__.py" sha="8f32b7a86dae0351f40a7e7b4ecdd9741795e0bb" -->

````````python
"""Local-only GUI environment integration for the Phase 4 tournament engine."""

from .session import TournamentGameSession, TournamentSessionRegistry

__all__ = ["TournamentGameSession", "TournamentSessionRegistry"]

````````

<!-- END_PY_SOURCE path="src/holdem_ai/tournament_service/__init__.py" -->

## src/holdem_ai/tournament_service/baseline.py

Original blob `c97ba3b8a729c1636a00d6153f82d99443ea7a84` (1345 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/tournament_service/baseline.py" sha="c97ba3b8a729c1636a00d6153f82d99443ea7a84" -->

````````python
"""Temporary, engine-legal six-seat baseline policies.

These are NOT the Tournament Agent or the proposed Improved Bots.  The
Tournament Policy Adapter milestone must replace/inject those policies after
an observation/action compatibility audit.
"""
from __future__ import annotations

import random
from typing import Any, Mapping


BASELINES = ("random", "calling_station", "tight", "pot_pressure")


class TournamentBaseline:
    def __init__(self, name: str = "calling_station", *, seed: int | None = None):
        if name not in BASELINES:
            raise ValueError("unknown tournament baseline policy")
        self.name = name
        self.rng = random.Random(seed)

    def select_action(self, observation: Mapping[str, Any]) -> int:
        legal = tuple(int(i) for i in observation["legal_actions"])
        if not legal:
            raise RuntimeError("no legal tournament actions")
        if self.name == "random":
            return self.rng.choice(legal)
        if self.name == "calling_station":
            priorities = (1, 0, 2, 3, 4)
        elif self.name == "tight":
            to_call = int(observation["raw_obs"]["to_call"])
            priorities = (1, 0, 2, 3, 4) if to_call == 0 else (0, 1, 2, 3, 4)
        else:
            priorities = (3, 2, 4, 1, 0)
        return next(i for i in priorities if i in legal)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/tournament_service/baseline.py" -->

## src/holdem_ai/tournament_service/improved_bot.py

Original blob `1eff74765a1e295c8e164de964494ee105c6d11d` (5676 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/tournament_service/improved_bot.py" sha="1eff74765a1e295c8e164de964494ee105c6d11d" -->

````````python
"""Tournament ENV 1.3: four reproducible, observation-only heuristic opponents.

No neural weights, RL training, opponent hole cards, Monte Carlo, or claims of
improved win-rate. ``improved`` names the user-facing development slot; actual
performance must be benchmarked separately against frozen reference bots.

Only Phase-3 public observe(actor) plus the actor's own private hand is read.
Action IDs are the native five-action tournament abstraction (0..4).
"""
from __future__ import annotations

from itertools import combinations
import random
from typing import Any, Mapping

from holdem_ai.environment.hand_evaluator import parse_card, rank_five

PROFILES: tuple[str, ...] = (
    "tight_aggressive", "balanced_value", "pot_odds", "controlled_pressure",
)


def _preflop_strength(hole: tuple[str, str]) -> float:
    (_, x), (_, y) = (parse_card(card) for card in hole)
    high, low = max(x, y), min(x, y)
    if x == y:
        return min(0.99, 0.54 + (high - 2) / 12.0 * 0.43)
    suited = hole[0][0].lower() == hole[1][0].lower()
    gap = high - low
    high_value = (high - 2) / 12.0
    low_value = (low - 2) / 12.0
    score = 0.13 + 0.40 * high_value + 0.24 * low_value
    if suited:
        score += 0.07
    if gap <= 2:
        score += 0.07 if gap == 1 else 0.035
    if high == 14:
        score += 0.04
    return max(0.0, min(0.94, score))


def _postflop_strength(hole: tuple[str, str], board: tuple[str, ...]) -> float:
    cards = (*hole, *board)
    if len(cards) < 5:
        raise ValueError("flop must contain at least three board cards")
    best = max(rank_five(combo) for combo in combinations(cards, 5))
    category = best[0]
    # Coarse made-hand heuristic, not showdown equity against multiple rivals.
    base = (0.18, 0.49, 0.70, 0.80, 0.88, 0.92, 0.96, 0.985, 0.997)[category]
    if category == 0:
        base += 0.13 * ((best[1] - 2) / 12.0)
    elif category == 1:
        base += 0.09 * ((best[1] - 2) / 12.0)
    if len(board) == 5:
        # Common board-only straights, flushes, etc are not exclusive strength.
        board_rank = rank_five(board)
        if best == board_rank:
            base = min(base, 0.49)
    # Transparent, cheap heuristic draws; no hidden/deck-peeking equity.
    suits = [parse_card(c)[0] for c in cards]
    if len(board) < 5 and max(suits.count(s) for s in set(suits)) >= 4:
        base += 0.07
    return min(0.999, base)


class ImprovedTournamentBot:
    """Native legal-action rule policy; four distinguishable playing styles."""

    def __init__(self, profile: str, *, seed: int | None = None):
        if profile not in PROFILES:
            raise ValueError(f"unknown improved bot profile: {profile}")
        self.profile = profile
        self._rng = random.Random(seed)

    def select_action(self, observation: Mapping[str, Any]) -> int:
        legal = tuple(int(a) for a in observation["legal_actions"])
        if not legal or any(a not in range(5) for a in legal):
            raise ValueError("tournament observation has invalid legal action indices")
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
        live_rivals = sum(
            1 for i, status in enumerate(raw["status"])
            if i != raw["player_id"] and status not in ("out", "folded")
        )
        # More multiway opponents -> stronger hands needed for value betting.
        multiway = min(0.12, max(0, live_rivals - 1) * 0.022)
        # 1.3.1: The 1.3 thresholds over-folded/over-called against 5 live
        # rivals. These are *heuristic* activation thresholds, NOT equity or
        # empirical win-rate estimates. Tune via separate seeded tournaments.
        cfg = {
            "tight_aggressive": (0.675, 0.49, 0.12),
            "balanced_value": (0.565, 0.40, 0.16),
            "pot_odds": (0.635, 0.34, 0.06),
            "controlled_pressure": (0.495, 0.395, 0.22),
        }
        raise_threshold, continue_threshold, aggression = cfg[self.profile]
        jitter = (self._rng.random() - 0.5) * 0.035
        value = strength + jitter - multiway
        # Short effective stacks permit a value-oriented shove, but never a
        # random automatic 200BB all-in simply because action 4 is legal.
        if 4 in legal and stack <= max(1, pot) and value >= 0.87:
            return 4
        # Important: half-pot can be *illegal* while a pot raise is legal
        # (e.g. minimum legal raise > half-pot). Never gate pot-sized raises
        # on the availability of action 2.
        if value >= raise_threshold and (2 in legal or 3 in legal):
            if 3 in legal and (
                2 not in legal or value >= raise_threshold + 0.14 or
                (self.profile == "controlled_pressure" and self._rng.random() < aggression)
            ):
                return 3
            if 2 in legal:
                return 2
        if 1 in legal and (call == 0 or value >= continue_threshold or
                           (self.profile == "pot_odds" and value + 0.08 >= pot_odds)):
            return 1
        if 0 in legal:
            return 0
        if 1 in legal:
            return 1
        # Engine-only fallback: every returned action must belong to legal.
        return legal[0]

````````

<!-- END_PY_SOURCE path="src/holdem_ai/tournament_service/improved_bot.py" -->

## src/holdem_ai/tournament_service/journal.py

Original blob `f04f1242f476e94fe10ab32a4d8736b84c3d598c` (10830 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/tournament_service/journal.py" sha="f04f1242f476e94fe10ab32a4d8736b84c3d598c" -->

````````python
"""Tournament ENV 1.6: append-only, local SQLite event journal.

Persistence is independent of game rules and trained-policy inference. The API
replays recorded actor-visible states; it does not pretend to resume a running
TournamentEnv from a serialized snapshot. No deck or other seats' hole cards
are stored in an action record.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class TournamentJournal:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("HOLDEM_AI_TOURNAMENT_JOURNAL", ".local/tournament_history.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS tournament_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_utc TEXT NOT NULL,
                    updated_utc TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('playing','complete','closed')),
                    options_json TEXT NOT NULL,
                    latest_hand INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 0,
                    winner_seat INTEGER
                );
                CREATE TABLE IF NOT EXISTS tournament_actions (
                    session_id TEXT NOT NULL,
                    hand_number INTEGER NOT NULL,
                    action_no INTEGER NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, hand_number, action_no),
                    FOREIGN KEY (session_id) REFERENCES tournament_sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS tournament_hands (
                    session_id TEXT NOT NULL,
                    hand_number INTEGER NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, hand_number),
                    FOREIGN KEY (session_id) REFERENCES tournament_sessions(session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_tournament_sessions_updated
                    ON tournament_sessions(updated_utc DESC);
            """)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=15)
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA busy_timeout = 15000")
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("PRAGMA synchronous = FULL")
        return db

    def begin(self, session_id: str, options: Mapping[str, Any]) -> None:
        safe = {k: v for k, v in options.items() if k in (
            "seed", "starting_stack", "button", "hands_per_level", "blind_levels", "max_hands", "lineup"
        )}
        with self._lock, self._connect() as db:
            instant = _now()
            db.execute("INSERT INTO tournament_sessions (session_id,created_utc,updated_utc,status,options_json) VALUES (?,?,?,?,?)",
                       (session_id, instant, instant, "playing", _json(safe)))

    def discard_failed_create(self, session_id: str) -> None:
        with self._lock, self._connect() as db:
            db.execute("DELETE FROM tournament_actions WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM tournament_hands WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM tournament_sessions WHERE session_id=?", (session_id,))

    def record_step(self, session_id: str, action: Mapping[str, Any],
                    completed_hands: Iterable[Mapping[str, Any]], *, revision: int,
                    winner_seat: int | None) -> None:
        completed = list(completed_hands)
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO tournament_actions (session_id,hand_number,action_no,data_json) VALUES (?,?,?,?)",
                       (session_id, action["hand_number"], action["action_no"], _json(action)))
            for hand in completed:
                db.execute("INSERT INTO tournament_hands (session_id,hand_number,data_json) VALUES (?,?,?)",
                           (session_id, hand["hand_number"], _json(hand)))
            db.execute("UPDATE tournament_sessions SET updated_utc=?, revision=?, latest_hand=MAX(latest_hand,?), status=?, winner_seat=? WHERE session_id=?",
                       (_now(), revision, max([action["hand_number"]] + [h["hand_number"] for h in completed]),
                        "complete" if winner_seat is not None else "playing", winner_seat, session_id))

    def record_initial_hands(self, session_id: str, hands: Iterable[Mapping[str, Any]], *,
                             revision: int, winner_seat: int | None) -> None:
        """Only for hands completed immediately on reset without any action."""
        hand_list = list(hands)
        if not hand_list:
            return
        with self._lock, self._connect() as db:
            for hand in hand_list:
                db.execute("INSERT OR IGNORE INTO tournament_hands (session_id,hand_number,data_json) VALUES (?,?,?)",
                           (session_id, hand["hand_number"], _json(hand)))
            db.execute("UPDATE tournament_sessions SET updated_utc=?, revision=?, latest_hand=MAX(latest_hand,?), status=?, winner_seat=? WHERE session_id=?",
                       (_now(), revision, hand_list[-1]["hand_number"], "complete" if winner_seat is not None else "playing",
                        winner_seat, session_id))

    def close(self, session_id: str) -> None:
        """Release a live session, retaining immutable history for replay."""
        with self._lock, self._connect() as db:
            db.execute("UPDATE tournament_sessions SET status=CASE WHEN status='complete' THEN 'complete' ELSE 'closed' END, updated_utc=? WHERE session_id=?",
                       (_now(), session_id))

    def list_sessions(self, *, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            total = db.execute("SELECT COUNT(*) FROM tournament_sessions").fetchone()[0]
            rows = db.execute("SELECT session_id,created_utc,updated_utc,status,options_json,latest_hand,revision,winner_seat FROM tournament_sessions ORDER BY updated_utc DESC, session_id DESC LIMIT ? OFFSET ?",
                              (limit, offset)).fetchall()
        return {"total": total, "offset": offset, "limit": limit,
                "sessions": [{"session_id": r[0], "created_utc": r[1], "updated_utc": r[2], "status": r[3],
                              "options": json.loads(r[4]), "latest_hand": r[5], "revision": r[6], "winner_seat": r[7]}
                             for r in rows]}

    def _require_session(self, db: sqlite3.Connection, session_id: str) -> None:
        if db.execute("SELECT 1 FROM tournament_sessions WHERE session_id=?", (session_id,)).fetchone() is None:
            raise KeyError(session_id)

    def hands(self, session_id: str, *, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            self._require_session(db, session_id)
            total = db.execute("SELECT COUNT(*) FROM tournament_hands WHERE session_id=?", (session_id,)).fetchone()[0]
            rows = db.execute("SELECT data_json FROM tournament_hands WHERE session_id=? ORDER BY hand_number DESC LIMIT ? OFFSET ?",
                              (session_id, limit, offset)).fetchall()
        return {"total": total, "offset": offset, "limit": limit,
                "hands": [json.loads(row[0]) for row in rows]}

    @staticmethod
    def _public_replay(action: Mapping[str, Any]) -> dict[str, Any]:
        # Return hero hole only for the human's own turn. Never expose AI holes.
        result = {k: v for k, v in action.items() if k != "actor_hole"}
        result["hero_hole"] = action.get("actor_hole", []) if action["seat"] == 0 else []
        return result

    def replay(self, session_id: str, hand_number: int) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            self._require_session(db, session_id)
            rows = db.execute("SELECT data_json FROM tournament_actions WHERE session_id=? AND hand_number=? ORDER BY action_no",
                              (session_id, hand_number)).fetchall()
            outcome = db.execute("SELECT data_json FROM tournament_hands WHERE session_id=? AND hand_number=?",
                                 (session_id, hand_number)).fetchone()
        if not rows and outcome is None:
            raise KeyError("hand")
        return {"session_id": session_id, "hand_number": hand_number,
                "actions": [self._public_replay(json.loads(r[0])) for r in rows],
                "settlement": json.loads(outcome[0]) if outcome is not None else None,
                "complete": outcome is not None}

    def actor_export_rows(self, session_id: str) -> list[dict[str, Any]]:
        """Each sample contains solely its acting player's own hole, public state,
        legal action IDs and post-hand chip reward (never opponents' hole/deck).
        Intended as an interchange contract, not trained PPO rollouts.
        """
        with self._lock, self._connect() as db:
            self._require_session(db, session_id)
            rows = db.execute("SELECT a.hand_number,a.action_no,a.data_json,h.data_json FROM tournament_actions a LEFT JOIN tournament_hands h ON a.session_id=h.session_id AND a.hand_number=h.hand_number WHERE a.session_id=? ORDER BY a.hand_number,a.action_no",
                              (session_id,)).fetchall()
        out: list[dict[str, Any]] = []
        for hand_no, action_no, raw_action, raw_hand in rows:
            action = json.loads(raw_action)
            hand = json.loads(raw_hand) if raw_hand is not None else None
            out.append({
                "schema": "tournament.actor_transition.v1", "session_id": session_id,
                "hand_number": hand_no, "action_no": action_no,
                "acting_seat": action["seat"],
                "observation": {**action["public_before"], "own_hole_cards": action["actor_hole"]},
                "action": {"index": action["index"], "kind": action["kind"],
                           "raise_to": action["raise_to"]},
                "hand_settled": hand is not None,
                "payoff_chips": hand["payoffs_chips"][action["seat"]] if hand is not None else None,
            })
        return out

````````

<!-- END_PY_SOURCE path="src/holdem_ai/tournament_service/journal.py" -->

## src/holdem_ai/tournament_service/session.py

Original blob `57b906cad8bfe2a3724d6da7c5a880035fbf03ab` (20805 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/tournament_service/session.py" sha="57b906cad8bfe2a3724d6da7c5a880035fbf03ab" -->

````````python
"""Phase 1.1: six-seat, human-vs-baselines TournamentEnv session adapter.

The Phase 4 engine owns rules, blinds, stack conservation and placement.
This module owns GUI projections, revision control and automatic baseline
turns.  No RL training, checkpoint mutation or privileged observations.
"""
from __future__ import annotations

import random
import secrets
import threading
from typing import Any, Mapping, Protocol, Sequence

from holdem_ai.environment.tournament_env import TournamentEnv, TournamentHandRecord

from .baseline import TournamentBaseline
from .improved_bot import ImprovedTournamentBot, PROFILES
from .trained_actor import TrainedTournamentActor
from .journal import TournamentJournal


class TournamentPolicy(Protocol):
    """Phase-4 native action contract: indices 0..4, NOT HUNL200 0..6."""

    def select_action(self, observation: Mapping[str, Any]) -> int: ...


_ACTION_NAMES = {0: "fold", 1: "check_call", 2: "raise_half_pot", 3: "raise_pot", 4: "all_in", 5: "raise_to"}


class TournamentGameSession:
    def __init__(
        self, *, seed: int | None = None, starting_stack: int = 1000,
        button: int = 0, hands_per_level: int = 10,
        blind_levels: Sequence[tuple[int, int]] | None = None,
        max_hands: int = 10_000, max_bot_actions: int = 100_000,
        policies: Mapping[int, TournamentPolicy] | None = None,
        lineup: str = "baseline_5",
        journal: TournamentJournal | None = None, journal_session_id: str | None = None,
    ):
        if type(max_bot_actions) is not int or max_bot_actions <= 0:
            raise ValueError("max_bot_actions must be positive")
        if lineup not in ("baseline_5", "improved_4", "trained_1_improved_4"):
            raise ValueError("unknown tournament lineup")
        if policies is not None and any(type(seat) is not int or seat not in range(1, 6)
                                        for seat in policies):
            raise ValueError("bot policies must target physical seats 1..5")
        if (journal is None) != (journal_session_id is None):
            raise ValueError("journal and journal_session_id must be supplied together")
        self._journal = journal
        self._journal_session_id = journal_session_id
        self._action_numbers: dict[int, int] = {}
        self.env = TournamentEnv(
            num_players=6, starting_stack=starting_stack, button=button,
            hands_per_level=hands_per_level, blind_levels=blind_levels,
            max_hands=max_hands, seed=seed,
        )
        self._rng = random.Random(seed)
        self.lineup = lineup
        self._policies: dict[int, TournamentPolicy] = {
            seat: TournamentBaseline("calling_station", seed=self._rng.getrandbits(64))
            for seat in range(1, 6)
        }
        if lineup in ("improved_4", "trained_1_improved_4"):
            if lineup == "trained_1_improved_4":
                # Fail closed: never silently replace a missing trained model.
                self._policies[1] = TrainedTournamentActor.from_environment(
                    seed=self._rng.getrandbits(64))
            else:
                self._policies[1] = ImprovedTournamentBot(
                    "controlled_pressure", seed=self._rng.getrandbits(64))
            for seat, profile in enumerate(PROFILES, start=2):
                self._policies[seat] = ImprovedTournamentBot(
                    profile, seed=self._rng.getrandbits(64))
        if policies is not None:
            self._policies.update(policies)
        self._lock = threading.RLock()
        self.max_bot_actions = max_bot_actions
        self.revision = 1
        self._events: list[dict[str, Any]] = []
        self._policy_history: list[dict[str, Any]] = []
        self._last_finished: dict[str, Any] | None = None
        self.env.reset()
        if self._journal is not None:
            self._journal.record_initial_hands(
                self._journal_session_id, (self._hand_summary(h) for h in self.env.hand_history),
                revision=self.revision, winner_seat=self.env.players.winner)
        self._run_bots()

    def _check_revision(self, revision: int) -> None:
        if type(revision) is not int or revision != self.revision:
            raise ValueError("stale revision; refresh tournament state")

    def _capture_completed(self, before: int) -> None:
        """Handle zero-decision hands that may finish during a single step."""
        new_records = self.env.hand_history[before:]
        for record in new_records:
            self._last_finished = self._hand_summary(record)
        if new_records:
            # The engine automatically starts the next hand.  We do not
            # expose a partially completed old-hand action list as live data.
            self._events = []
            self._policy_history = []

    @staticmethod
    def _hand_summary(rec: TournamentHandRecord) -> dict[str, Any]:
        result = rec.hand_result
        return {
            "hand_number": rec.hand_number,
            "button": rec.button,
            "level_index": rec.level_index,
            "small_blind": rec.blinds.small_blind,
            "big_blind": rec.blinds.big_blind,
            "stacks_before": list(rec.stacks_before),
            "stacks_after": list(rec.stacks_after),
            "eliminated": list(rec.eliminated),
            "board": list(result.board),
            "payoffs_chips": list(result.payoffs_chips),
            "awards": list(result.awards),
            # Only AFTER settlement, and only seats that truly reached showdown.
            # Never expose folded player hole cards or the hidden deck.
            "showdown_cards": [
                {"seat": i, "cards": list(cards)}
                for i, cards in enumerate(rec.showdown_hole_cards) if len(cards) == 2
            ],
            "action_history": [
                {"seat": int(seat), "index": int(action),
                 "kind": _ACTION_NAMES[action], "raise_to": amount}
                for (seat, action), amount in zip(result.action_history, result.action_raise_to)
            ],
            # Pre-settlement opponent cards/deck/training state stay private.
        }

    def _step(self, action_index: int | None = None, raise_to: int | None = None) -> None:
        if self.env.is_over():
            raise RuntimeError("tournament is complete")
        hand = self.env._hand
        assert hand is not None
        actor = hand.current_player
        street = hand.street
        hand_number = self.env._hand_number
        completed_before = len(self.env.hand_history)
        # Capture ONLY public information at the decision instant; required
        # for a faithful actor-visible history, never saved to the HTTP view.
        pre = hand.observe(actor)["raw_obs"]
        before_call = int(pre["to_call"])
        before_pot = int(pre["pot"])
        # Capture the actor-visible information BEFORE engine mutation. A replay
        # is a series of immutable decisions plus the authoritative settlement.
        # No full deck and no other player's private hand enters this record.
        public_before = {
            "street": street, "board": list(hand.board),
            "button": int(hand.button), "small_blind": int(self.env._level.small_blind),
            "big_blind": int(self.env._level.big_blind),
            "stacks": list(hand.stacks), "street_bets": list(hand.street_bets),
            "contributions": list(hand.contributions), "statuses": list(hand.status),
            "pot": before_pot, "to_call": before_call,
            "legal_actions": [int(x) for x in hand.legal_actions],
        }
        actor_hole = list(hand.hole_cards[actor])
        action_no = self._action_numbers.get(hand_number, 0) + 1
        # Exact street-total raise target for actor-visible export and replay.
        resolved_target = raise_to
        if action_index == 2:
            resolved_target = int(pre["street_bets"][actor]) + before_call + (before_pot + before_call) // 2
        elif action_index == 3:
            resolved_target = int(pre["street_bets"][actor]) + before_call + before_pot + before_call
        elif action_index == 4:
            resolved_target = int(pre["street_bets"][actor]) + int(pre["stacks"][actor])
        if raise_to is not None:
            self.env.step_raise_to(raise_to)
            index = 5
        else:
            assert action_index is not None
            self.env.step(action_index)
            index = action_index
        self.revision += 1
        self._action_numbers[hand_number] = action_no
        completed = self.env.hand_history[completed_before:]
        if self._journal is not None:
            self._journal.record_step(
                self._journal_session_id,
                {"hand_number": hand_number, "action_no": action_no, "seat": actor,
                 "index": index, "kind": _ACTION_NAMES[index], "raise_to": resolved_target,
                 "public_before": public_before, "actor_hole": actor_hole},
                (self._hand_summary(rec) for rec in completed),
                revision=self.revision, winner_seat=self.env.players.winner)
        if completed:
            self._capture_completed(completed_before)
        else:
            self._events.append({
                "hand_number": hand_number, "seat": actor, "street": street,
                "index": index, "kind": _ACTION_NAMES[index], "raise_to": raise_to,
            })
            target = raise_to
            if index == 2:
                target = int(pre["street_bets"][actor]) + before_call + (before_pot + before_call) // 2
            elif index == 3:
                target = int(pre["street_bets"][actor]) + before_call + before_pot + before_call
            elif index == 4:
                target = int(pre["street_bets"][actor]) + int(pre["stacks"][actor])
            self._policy_history.append({
                "seat": actor, "street": street, "index": index,
                "raise_to": target, "to_call_before": before_call, "pot_before": before_pot,
            })

    def _run_bots(self) -> None:
        actions = 0
        while not self.env.is_over() and self.env.current_player != 0:
            actions += 1
            if actions > self.max_bot_actions:
                raise RuntimeError("automatic bot action limit reached")
            actor = self.env.current_player
            if actor not in self._policies:
                raise RuntimeError(f"tournament seat {actor} has no policy")
            observation = self.env.observe(actor)
            # Session-owned public history uses exact street and pre-action
            # values. No deck, hidden hands or critic state is added here.
            if isinstance(self._policies[actor], TrainedTournamentActor):
                observation = dict(observation)
                observation["public_action_log"] = tuple(self._policy_history)
            chosen = self._policies[actor].select_action(observation)
            if type(chosen) is not int or chosen not in observation["legal_actions"]:
                raise RuntimeError(f"bot at seat {actor} returned illegal tournament action")
            self._step(action_index=chosen)

    def play_action(self, action_index: int, *, expected_revision: int) -> dict[str, Any]:
        with self._lock:
            self._check_revision(expected_revision)
            if self.env.is_over() or self.env.current_player != 0:
                raise RuntimeError("not the human's turn")
            if type(action_index) is not int or action_index not in self.env._hand.legal_actions:
                raise ValueError("illegal human action")
            self._step(action_index=action_index)
            self._run_bots()
            return self._view_unlocked()

    def play_raise_to(self, amount: int, *, expected_revision: int) -> dict[str, Any]:
        with self._lock:
            self._check_revision(expected_revision)
            if self.env.is_over() or self.env.current_player != 0:
                raise RuntimeError("not the human's turn")
            if type(amount) is not int:
                raise ValueError("raise_to must be integer chips")
            # Engine.step_raise_to performs atomic legality checks before state mutation.
            self._step(raise_to=amount)
            self._run_bots()
            return self._view_unlocked()

    def history(self, *, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("invalid history pagination")
        with self._lock:
            records = self.env.hand_history
            return {
                "total": len(records), "offset": offset, "limit": limit,
                "hands": [self._hand_summary(record) for record in records[offset:offset+limit]],
            }

    def view(self) -> dict[str, Any]:
        with self._lock:
            return self._view_unlocked()

    def _view_unlocked(self) -> dict[str, Any]:
        env = self.env
        done = env.is_over()
        hand = env._hand
        ranks = list(env.players.ranks)
        if hand is None and not done:
            raise AssertionError("nonterminal tournament missing current hand")
        if done:
            stacks = list(env.players.stacks)
            board = list(self._last_finished["board"]) if self._last_finished else []
            hero_cards: list[str] = []
            hand_number = len(env.hand_history)
            button = env.button
            level_index, level = env.blinds.at_hand(max(1, hand_number))
            seat_states = ["winner" if x > 0 else "out" for x in stacks]
            street_bets = [0] * 6
            contributions = [0] * 6
            to_act: int | None = None
            pot = 0
            min_raise_to: int | None = None
            max_raise_to: int | None = None
            to_call: int | None = None
            offers: list[dict[str, Any]] = []
            starting = list(self._last_finished["stacks_before"]) if self._last_finished else stacks
            small_seat = None
            big_seat = None
        else:
            assert hand is not None
            stacks = list(hand.stacks)
            board = list(hand.board)
            hero_cards = list(hand.hole_cards[0]) if env.players.stacks[0] > 0 else []
            hand_number = env._hand_number
            button = hand.button
            level_index = env._level_index
            level = env._level
            seat_states = list(hand.status)
            street_bets = list(hand.street_bets)
            contributions = list(hand.contributions)
            to_act = hand.current_player
            pot = sum(hand.contributions)
            starting = list(hand.initial_stacks)
            small_seat = hand.seats.small_blind
            big_seat = hand.seats.big_blind
            min_raise_to = None
            max_raise_to = None
            to_call = None
            offers = []
            if to_act == 0:
                raw = hand.observe(0)["raw_obs"]
                to_call = int(raw["to_call"])
                legal = hand.legal_actions
                for index in legal:
                    offers.append({"index": index, "kind": _ACTION_NAMES[index]})
                if 4 in legal:
                    min_raise_to = int(hand.min_raise_to())
                    max_raise_to = int(hand.street_bets[0] + hand.stacks[0])
        # Include chips already committed in the pot.  A short all-in hand may
        # have 0 in the sitting stack but has NOT busted until settlement.
        if not done and sum(stacks) + pot != env.players.total_chips:
            raise AssertionError("active tournament chip conservation violated")
        if done and sum(stacks) != env.players.total_chips:
            raise AssertionError("completed tournament chip conservation violated")
        # Tournament-wide stacks are the last SETTLED chip balances; during a
        # live hand, the current `stacks` are *behind* the betting line.
        # Keeping these separate prevents short all-ins from appearing busted.
        level_count = len(env.blinds.levels)
        next_level = env.blinds.levels[level_index + 1] if not done and level_index + 1 < level_count else None
        level_hand = ((hand_number - 1) % env.blinds.hands_per_level) + 1 if not done else None
        return {
            "game_type": "TOURNAMENT_6MAX", "revision": self.revision,
            "lineup": self.lineup,
            "phase": "complete" if done else "playing",
            "tournament_over": done,
            "winner_seat": env.players.winner,
            "human_eliminated": env.players.stacks[0] == 0,
            "hand_number": hand_number,
            "hands_completed": len(env.hand_history),
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
            "level_index": level_index,
            "small_blind": level.small_blind, "big_blind": level.big_blind,
            "button": button,
            "small_blind_seat": small_seat, "big_blind_seat": big_seat,
            "street": None if done else hand.street,
            "board": board,
            "hero_cards": hero_cards,
            "seats": [
                {"seat": seat,
                 "role": ("human" if seat == 0 else
                          "baseline_bot" if self.lineup == "baseline_5" else
                          ("tournament_agent" if self.lineup == "trained_1_improved_4" else
                           "tournament_agent_placeholder") if seat == 1 else "improved_bot"),
                 "policy_name": ("human" if seat == 0 else
                                 str(getattr(self._policies[seat], "profile",
                                             getattr(self._policies[seat], "name",
                                                     type(self._policies[seat]).__name__)))),
                 "stack": stacks[seat], "status": seat_states[seat],
                 "street_bet": street_bets[seat], "total_contribution": contributions[seat],
                 "rank": ranks[seat]}
                for seat in range(6)
            ],
            "total_chips": env.players.total_chips,
            "pot": pot,
            "to_act": to_act,
            "your_turn": not done and to_act == 0,
            "to_call": to_call,
            "min_raise_to": min_raise_to,
            "max_raise_to": max_raise_to,
            "can_raise_to": min_raise_to is not None,
            "actions": offers,
            "action_log": [dict(e) for e in self._events],
            "last_hand": self._last_finished,
            "starting_stacks": starting,
        }


class TournamentSessionRegistry:
    def __init__(self, *, max_sessions: int = 128, journal: TournamentJournal | None = None):
        if type(max_sessions) is not int or max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self.max_sessions = max_sessions
        self.journal = journal
        self._lock = threading.RLock()
        self._sessions: dict[str, TournamentGameSession] = {}

    def create(self, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError("tournament session capacity reached")
            session_id = secrets.token_urlsafe(24)
            if self.journal is not None:
                self.journal.begin(session_id, kwargs)
            try:
                session = TournamentGameSession(
                    **kwargs, journal=self.journal,
                    journal_session_id=session_id if self.journal is not None else None)
            except Exception:
                if self.journal is not None:
                    self.journal.discard_failed_create(session_id)
                raise
            self._sessions[session_id] = session
            return session_id, session.view()

    def get(self, session_id: str) -> TournamentGameSession:
        with self._lock:
            return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        with self._lock:
            del self._sessions[session_id]
            if self.journal is not None:
                self.journal.close(session_id)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/tournament_service/session.py" -->

## src/holdem_ai/tournament_service/trained_actor.py

Original blob `910b29ab575076918fefc4ae11fec020524de22a` (9173 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/tournament_service/trained_actor.py" sha="910b29ab575076918fefc4ae11fec020524de22a" -->

````````python
"""Tournament ENV 1.4: frozen Expand Phase3 multi-seat PPO inference bridge.

The checkpoint was trained on a 2-6 player *cash* table, not the knockout
TournamentEnv. Tournament transfer is experimental. No training or mutation.
Never load HUNL200 2-player weights here and never pass privileged state.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


CHECKPOINT_ENV = "HOLDEM_AI_TOURNAMENT_CHECKPOINT"
MODE_ENV = "HOLDEM_AI_TOURNAMENT_POLICY_MODE"
DEVICE_ENV = "HOLDEM_AI_TOURNAMENT_DEVICE"
FORMAT = "expand-e03-v1"

# The current Phase3 hand engine has 5 legal actions; Expand Phase3 has 7.
# Native: 0 fold, 1 check_call, 2 half pot, 3 pot, 4 all-in.
# Expand: 0 fold, 1 check, 2 call, 3 half pot, 4 pot, 5 double pot, 6 all-in.
EXPAND_TO_NATIVE: tuple[int | None, ...] = (0, 1, 1, 2, 3, None, 4)


def status() -> dict[str, object]:
    """Describe local configuration; actual model structure verified at create."""
    config = os.environ.get(CHECKPOINT_ENV, "").strip()
    selected = Path(config).expanduser() if config else None
    return {
        "configured": bool(config),
        "file_found": bool(selected and selected.is_file()),
        "checkpoint_name": selected.name if selected else None,
        "format": FORMAT,
        "model_family": "Expand Phase3 multi-seat PPO (cash-game transfer)",
        "device": os.environ.get(DEVICE_ENV, "cpu"),
        "mode": os.environ.get(MODE_ENV, "sampled"),
        "validated": False,  # a file's presence does not prove compatibility
    }


def _sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_public_state(observation: Mapping[str, Any]):
    """Construct the Phase3 actor state using only the acting seat's observe().

    The separate public_action_log comes from the GUI session, where street and
    pre-action pot/to_call are captured *before* the engine mutates state.
    """
    from holdem_ai.expand.information.schema import (
        InformationState, LegalActionSpec, PublicAction, PublicSeatFeatures,
    )

    raw = observation["raw_obs"]
    tournament = observation["tournament"]
    actor = int(raw["player_id"])
    button = int(raw["button"])
    bb = int(tournament["big_blind"])
    if bb <= 0 or actor not in range(6):
        raise ValueError("invalid tournament actor or big blind")
    legal_native = {int(a) for a in observation["legal_actions"]}
    if not legal_native or not legal_native.issubset(set(range(5))):
        raise ValueError("invalid native tournament legal actions")
    raw_call = int(raw["to_call"])
    stacks = tuple(int(x) for x in raw["stacks"])
    statuses = tuple(str(x) for x in raw["status"])
    investments = tuple(int(x) for x in raw["total_contributions"])
    bets = tuple(int(x) for x in raw["street_bets"])
    if not all(len(x) == 6 for x in (stacks, statuses, investments, bets)):
        raise ValueError("expected six physical seat observations")
    max_raise = bets[actor] + stacks[actor]
    min_raise = int(raw["min_raise_to"])
    mask = (
        0 in legal_native,
        1 in legal_native and raw_call == 0,
        1 in legal_native and raw_call > 0,
        2 in legal_native,
        3 in legal_native,
        False,  # double-pot is NOT a native action: do not silently remap
        4 in legal_native,
    )
    if not any(mask):
        raise ValueError("no legal 7-action representation of tournament turn")
    amount = [None] * 7
    # Exact engine chip arithmetic, not floating pot fractions.
    amount[3] = bets[actor] + raw_call + (int(raw["pot"]) + raw_call) // 2 if mask[3] else None
    amount[4] = bets[actor] + raw_call + (int(raw["pot"]) + raw_call) if mask[4] else None
    amount[6] = max_raise if mask[6] else None
    legal = LegalActionSpec(
        mask=mask, raise_to_chips=tuple(amount),
        min_raise_to_chips=min_raise, max_raise_to_chips=max_raise,
        to_call_chips=raw_call,
    )
    seats = tuple(
        PublicSeatFeatures(
            occupied=st != "out", dealt_in=st != "out",
            folded=st == "folded", all_in=st == "allin",
            seat_status=st, table_status="seated" if st != "out" else "sitting_out",
            stack_bb=stacks[i] / bb, invested_bb=investments[i] / bb,
            street_bet_bb=bets[i] / bb,
            relative_position=(i - button) % 6,
            is_hero=i == actor,
        ) for i, st in enumerate(statuses)
    )
    public_events = observation.get("public_action_log", ())
    names = {0: "fold", 1: "check", 2: "raise_half_pot", 3: "raise_pot",
             4: "all_in", 5: "raise_to"}
    history = []
    for event in public_events:
        index = int(event["index"])
        kind = names[index]
        if index == 1 and int(event["to_call_before"]) > 0:
            kind = "call"
        target = event.get("raise_to")
        history.append(PublicAction(
            actor_seat=int(event["seat"]), street=str(event["street"]), kind=kind,
            raise_to_bb=None if target is None else int(target) / bb,
            to_call_bb=int(event["to_call_before"]) / bb,
            pot_bb=int(event["pot_before"]) / bb,
        ))
    return InformationState(
        hero_seat=actor, hero_hole=tuple(raw["hand"]),
        board=tuple(raw["public_cards"]), street=str(raw["street"]),
        button_seat=button, seats=seats,
        action_history=tuple(history), legal_actions=legal,
        pot_bb=int(raw["pot"]) / bb, to_call_bb=raw_call / bb,
        big_blind_chips=bb, num_dealt_in=sum(s != "out" for s in statuses),
    )


class TrainedTournamentActor:
    """Readonly trained Expand PPO agent, physical seat assigned by session."""

    def __init__(self, model: Any, *, fingerprint: str, seed: int | None = None,
                 mode: str = "sampled"):
        if mode not in ("sampled", "greedy"):
            raise ValueError("Tournament policy mode must be sampled or greedy")
        self.model = model
        self.mode = mode
        self._rng = random.Random(seed)
        self.fingerprint = fingerprint
        self.name = f"Expand Phase3 PPO [{fingerprint[:12]}]"

    @classmethod
    def from_environment(cls, *, seed: int | None = None) -> "TrainedTournamentActor":
        value = os.environ.get(CHECKPOINT_ENV, "").strip()
        if not value:
            raise ValueError(f"trained tournament mode requires {CHECKPOINT_ENV}")
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError(f"configured Tournament checkpoint is missing: {path.name}")
        device = os.environ.get(DEVICE_ENV, "cpu")
        mode = os.environ.get(MODE_ENV, "sampled")
        if device not in ("cpu", "cuda") or mode not in ("sampled", "greedy"):
            raise ValueError("Tournament device must be cpu/cuda and policy mode sampled/greedy")
        # Import lazily; the default 5-baseline/improved modes do not require torch.
        from holdem_ai.expand.rl.trainer import load_trained_actor
        try:
            model = load_trained_actor(path, device=device)
            fingerprint = _sha256(path)
        except (KeyError, ValueError, TypeError, RuntimeError, OSError) as exc:
            raise ValueError("checkpoint rejected: expected a compatible, trusted Expand Phase3 PPO checkpoint; HUNL 2-seat weights are not supported") from exc
        return cls(model, fingerprint=fingerprint, seed=seed, mode=mode)

    def select_action(self, observation: Mapping[str, Any]) -> int:
        from holdem_ai.expand.information.encoder import encode_observation
        state = _prepare_public_state(observation)
        # Using the actor head ONLY; the model critic is not queried.
        probabilities = tuple(float(p) for p in self.model.actor.action_probabilities(
            encode_observation(state)))
        if len(probabilities) != 7 or any(not math.isfinite(p) or p < 0 for p in probabilities):
            raise RuntimeError("invalid trained policy distribution")
        mask = state.legal_actions.mask
        weights = [p if allowed else 0.0 for p, allowed in zip(probabilities, mask)]
        total = math.fsum(weights)
        if total <= 0:
            raise RuntimeError("no probability assigned to a legal tournament action")
        if self.mode == "greedy":
            selected = max((i for i, ok in enumerate(mask) if ok), key=lambda i: weights[i])
        else:
            point = self._rng.random() * total
            cumulative = 0.0
            selected = next(i for i, p in enumerate(weights) if p > 0)  # numeric fallback
            for i, value in enumerate(weights):
                cumulative += value
                if point < cumulative and value > 0:
                    selected = i
                    break
        native = EXPAND_TO_NATIVE[selected]
        if native is None or native not in observation["legal_actions"]:
            raise RuntimeError("trained agent selected an invalid native tournament action")
        return native

````````

<!-- END_PY_SOURCE path="src/holdem_ai/tournament_service/trained_actor.py" -->

## src/holdem_ai/training/__init__.py

Original blob `4eac3fc82dded49e40f2471913810949e63ee45d` (40 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/training/__init__.py" sha="4eac3fc82dded49e40f2471913810949e63ee45d" -->

````````python
"""Self-play and training pipelines."""

````````

<!-- END_PY_SOURCE path="src/holdem_ai/training/__init__.py" -->

## src/holdem_ai/training/batched_collector.py

Original blob `0e1e2241022b55e549130eb5edd19076a47bb08a` (5632 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/training/batched_collector.py" sha="0e1e2241022b55e549130eb5edd19076a47bb08a" -->

````````python
"""Batched (multi-hand, single-process) on-policy GPU inference for HUNL200.

Every hand remains an independent engine instance; a single frozen Actor
processes the *next* observation of each unfinished hand as a batch.
No opponent cards enter ActorBatch. The same policy version acts on both seats.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch

from holdem_ai.core.schema import Terminal, Turn
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
from holdem_ai.encoding.tensors import batch_actors
from holdem_ai.encoding.observation import encode_actor, encode_critic
from holdem_ai.models.transformer import TransformerActor
from holdem_ai.training.ppo import PPOConfig, TrajectoryRows


@dataclass
class _Hand:
    index: int
    env: HUNL200Adapter
    rng: random.Random
    outcome: Turn | Terminal
    actors: list = field(default_factory=list)
    critics: list = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    seats: list[int] = field(default_factory=list)
    rewards: tuple[float, float] | None = None


def collect_batched_self_play(
    actor: TransformerActor, config: PPOConfig, *, iteration: int = 0,
    device: str = 'cpu', env_batch_size: int = 8,
) -> TrajectoryRows:
    """Batch inference across independently evolving hands; preserve on-policy sampling.

    At most ``env_batch_size`` hands live in memory simultaneously.  Every
    unfinished hand advances exactly ONE engine decision per forward batch.
    Returns are assigned only after its terminal state is observed. ``index``
    and shuffle seed match Phase9's deterministic per-hand scheduling.
    """
    if type(iteration) is not int or iteration < 0:
        raise ValueError('iteration must be a nonnegative integer')
    if type(env_batch_size) is not int or env_batch_size < 1:
        raise ValueError('env_batch_size must be a positive integer')
    if actor.config.dropout != 0:
        raise ValueError('on-policy collector requires deterministic dropout=0')
    actor = actor.to(device).eval()
    all_actors, all_critics, all_actions, all_seats, all_returns = [], [], [], [], []
    all_hand_rewards = []
    terminal_actions = []
    decisions_per_hand = []
    for offset in range(0, config.hands_per_update, env_batch_size):
        hands: list[_Hand] = []
        for k in range(offset, min(offset + env_batch_size, config.hands_per_update)):
            index = iteration * config.hands_per_update + k
            seed = config.seed + index
            env = HUNL200Adapter(
                button=index % 2, big_blind=config.big_blind,
                expose_training_state=True,
            )
            outcome = env.reset(seed=seed)
            hands.append(_Hand(index, env, random.Random(seed), outcome))
        while True:
            active = [h for h in hands if isinstance(h.outcome, Turn)]
            if not active:
                break
            actor_rows = [
                encode_actor(h.outcome.observation, big_blind=config.big_blind,
                             max_tokens=actor.config.max_tokens)
                for h in active
            ]
            # Critic states may contain opponent's hole cards: never part of batch_actors.
            critic_rows = [
                encode_critic(h.env.training_state(), big_blind=config.big_blind,
                              max_tokens=actor.config.max_tokens)
                for h in active
            ]
            with torch.inference_mode():
                batch = batch_actors(actor_rows, device=device)
                probabilities = actor.distribution(batch).probs.detach().cpu().tolist()
            for hand, encoded_actor, encoded_critic, probs in zip(
                active, actor_rows, critic_rows, probabilities
            ):
                assert isinstance(hand.outcome, Turn)
                current = hand.outcome.observation
                action = hand.rng.choices(range(7), weights=probs, k=1)[0]
                current.assert_legal(action)
                if len(hand.actions) >= config.max_decisions:
                    raise RuntimeError('max_decisions exceeded in batched rollout')
                hand.actors.append(encoded_actor)
                hand.critics.append(encoded_critic)
                hand.seats.append(current.player_id)
                hand.actions.append(action)
                hand.outcome = hand.env.step(action)
                if isinstance(hand.outcome, Terminal):
                    rewards = tuple(float(x) for x in hand.outcome.rewards)
                    if len(rewards) != 2 or not math.isclose(sum(rewards), 0.0, abs_tol=1e-8):
                        raise AssertionError('non-zero-sum terminal reward')
                    hand.rewards = rewards
        for hand in hands:
            if hand.rewards is None or not hand.actions:
                raise AssertionError('incomplete hand')
            all_actors.extend(hand.actors)
            all_critics.extend(hand.critics)
            all_actions.extend(hand.actions)
            all_seats.extend(hand.seats)
            all_returns.extend(hand.rewards[seat] for seat in hand.seats)
            all_hand_rewards.append(hand.rewards)
            terminal_actions.append(hand.actions[-1])
            decisions_per_hand.append(len(hand.actions))
    return TrajectoryRows(
        tuple(all_actors), tuple(all_critics), tuple(all_actions),
        tuple(all_seats), tuple(all_returns), config.hands_per_update,
        tuple(all_hand_rewards),
        tuple(terminal_actions), tuple(decisions_per_hand),
    )

````````

<!-- END_PY_SOURCE path="src/holdem_ai/training/batched_collector.py" -->

## src/holdem_ai/training/episode_runner.py

Original blob `6f08af4094349bcaf0d3a6b5c87b73ded774dcf8` (2963 bytes).

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/training/episode_runner.py" sha="6f08af4094349bcaf0d3a6b5c87b73ded774dcf8" -->

````````python
"""Turn-based multiagent self-play, one transition per decision by each player."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping

from holdem_ai.agents.policies import AgentPolicy
from holdem_ai.core.schema import InformationState, Terminal, Turn, TurnBasedEnvironment


@dataclass(frozen=True, slots=True)
class Transition:
    decision_index: int
    seat: int
    observation: InformationState
    action_index: int
    next_observation: InformationState | None
    reward: float
    done: bool


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    rewards: tuple[float, ...]
    transitions: tuple[Transition, ...]
    num_decisions: int

    def for_seat(self, seat: int) -> tuple[Transition, ...]:
        return tuple(t for t in self.transitions if t.seat == seat)


def run_self_play(
    environment: TurnBasedEnvironment,
    policies: Mapping[int, AgentPolicy],
    *,
    seed: int | None = None,
    max_decisions: int = 100_000,
) -> EpisodeResult:
    """Collect trajectories; next_obs is the NEXT turn of the SAME seat.

    Intermediate reward is zero by contract; only the pending last transition
    per seat receives its terminal utility. Seat IDs must be stable throughout
    the episode, including after tournament elimination.
    """
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    rng = random.Random(seed)
    outcome = environment.reset(seed=seed)
    pending: dict[int, tuple[int, InformationState, int]] = {}
    completed: list[Transition] = []
    decisions = 0
    while isinstance(outcome, Turn):
        obs = outcome.observation
        if obs.player_id not in policies:
            raise KeyError(f"no policy for seat {obs.player_id}")
        if obs.player_id in pending:
            old_index, old_obs, old_action = pending.pop(obs.player_id)
            completed.append(Transition(old_index, obs.player_id, old_obs, old_action, obs, 0.0, False))
        if decisions >= max_decisions:
            raise RuntimeError("max_decisions exhausted; possible nonterminating game")
        action = policies[obs.player_id].act(obs, rng)
        obs.assert_legal(action)
        pending[obs.player_id] = (decisions, obs, action)
        outcome = environment.step(action)
        decisions += 1
    if not isinstance(outcome, Terminal):
        raise TypeError("environment must return Turn or Terminal")
    if any(seat >= len(outcome.rewards) for seat in pending):
        raise ValueError("terminal rewards missing a seat")
    for seat, (index, obs, action) in pending.items():
        completed.append(Transition(index, seat, obs, action, None, outcome.rewards[seat], True))
    if any(not math.isfinite(r) for r in outcome.rewards):
        raise ValueError("nonfinite terminal reward")
    return EpisodeResult(outcome.rewards, tuple(sorted(completed, key=lambda t: t.decision_index)), decisions)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/training/episode_runner.py" -->
