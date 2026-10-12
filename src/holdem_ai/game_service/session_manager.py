"""Local HUNL200 human-vs-agent game sessions; transport agnostic.

No server or React dependency. Training remains independent 200BB; local
GUI sessions may carry stacks between hands. Synchronous revision-checked play.
"""
from __future__ import annotations

import random
import secrets
import threading
from typing import Any

from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.core.schema import Terminal, Turn
from .interactive_adapter import InteractiveHUNLAdapter

from .action_adapter import validate_human_action
from .bot_adapter import BotActionAdapter, GUIBotPolicy
from .state_projection import project_human_view
from .session_analytics import SessionAnalytics


class HUNLGameSession:
    """An in-memory multi-hand heads-up session: human=seat 0, bot=seat 1.

    A bot policy can be RandomLegalPolicy, a diagnostic agent or a compatible
    NeuralActorPolicy. A trusted server must load checkpoints separately.
    """

    def __init__(
        self, *, bot_policy: GUIBotPolicy | None = None,
        big_blind: int = 100, small_blind: int = 50,
        stack_bb: int = 200, hero_seat: int = 0,
        seed: int | None = None, max_bot_actions: int = 1000,
        stack_mode: str = "independent",
    ) -> None:
        if hero_seat != 0:
            raise ValueError("GUI-1.0 fixes human seat to 0")
        if type(max_bot_actions) is not int or max_bot_actions < 1:
            raise ValueError("max_bot_actions must be positive integer")
        if stack_mode not in ("independent", "continuous"):
            raise ValueError("stack_mode must be independent or continuous")
        self.stack_mode = stack_mode
        # Validate configuration BEFORE accepting a game session.
        InteractiveHUNLAdapter(big_blind=big_blind, small_blind=small_blind, stack_bb=stack_bb)
        self.big_blind = big_blind
        self.small_blind = small_blind
        self.stack_bb = stack_bb
        self.hero_seat = hero_seat
        self.max_bot_actions = max_bot_actions
        self._seed_rng = random.Random(seed)
        self.bot = BotActionAdapter(bot_policy or RandomLegalPolicy(), random.Random(self._seed_rng.getrandbits(64)))
        self._lock = threading.RLock()
        self._adapter: InteractiveHUNLAdapter | None = None
        self._transition: Turn | Terminal | None = None
        self._action_log: list[dict[str, Any]] = []
        self.hand_number = 0
        self.revision = 0
        self.completed_hands = 0
        self.cumulative_bb = 0.0
        self.analytics = SessionAnalytics()

    def start_hand(self) -> dict[str, Any]:
        with self._lock:
            if self._transition is not None and not isinstance(self._transition, Terminal):
                raise RuntimeError("finish the current hand before starting another")
            initial_stacks: tuple[int, int] | None = None
            if self.stack_mode == "continuous" and self._adapter is not None:
                final = self._adapter.engine.get_result().final_stacks
                initial_stacks = (int(final[0]), int(final[1]))
                if min(initial_stacks) == 0:
                    raise RuntimeError("match over: a player is bankrupt; start a new session")
                if sum(initial_stacks) != 2 * self.stack_bb * self.big_blind:
                    raise AssertionError("chips changed across consecutive hands")
            next_hand_number = self.hand_number + 1
            adapter = InteractiveHUNLAdapter(
                big_blind=self.big_blind, small_blind=self.small_blind,
                stack_bb=self.stack_bb, button=(next_hand_number - 1) % 2,
                expose_training_state=False,
            )
            deck_seed = self._seed_rng.getrandbits(64)
            next_transition = (adapter.reset_with_stacks(initial_stacks, seed=deck_seed)
                               if initial_stacks is not None else adapter.reset(seed=deck_seed))
            self.hand_number = next_hand_number
            self._adapter = adapter
            self._action_log = []
            self._transition = next_transition
            self.revision += 1
            if isinstance(next_transition, Terminal):
                # A short blind may force a zero-decision, immediate showdown.
                # It is still a completed hand with a real payoff.
                self._finish_terminal(next_transition)
            else:
                self._run_bots()
            return self._view_unlocked()

    def start_next_hand(self, *, expected_revision: int) -> dict[str, Any]:
        """Start the next hand atomically, rejecting stale/double clicks.

        GUI-1.1 extension. Unlike the legacy start_hand() entry point, this
        is safe to call from an HTTP handler with optimistic concurrency.
        """
        with self._lock:
            if type(expected_revision) is not int or expected_revision != self.revision:
                raise ValueError("stale revision: refresh game state before acting")
            if not isinstance(self._transition, Terminal):
                raise RuntimeError("finish the current hand before starting another")
            return self.start_hand()

    def play_human_action(self, action_index: int, *, expected_revision: int) -> dict[str, Any]:
        with self._lock:
            if type(expected_revision) is not int or expected_revision != self.revision:
                raise ValueError("stale revision: refresh game state before acting")
            if self._adapter is None or not isinstance(self._transition, Turn):
                raise RuntimeError("start a live hand before submitting an action")
            if self._transition.player_id != self.hero_seat:
                raise RuntimeError("it is not the human's turn")
            action_index = validate_human_action(self._transition.observation, action_index)
            self._take_action(action_index)
            self._run_bots()
            return self._view_unlocked()

    def play_human_raise_to(self, amount: int, *, expected_revision: int) -> dict[str, Any]:
        """Human-only arbitrary legal raise-to; the bot remains 7-action.

        An invalid amount or stale revision leaves the hand and revision intact.
        """
        with self._lock:
            if type(expected_revision) is not int or expected_revision != self.revision:
                raise ValueError("stale revision: refresh game state before acting")
            if self._adapter is None or not isinstance(self._transition, Turn):
                raise RuntimeError("start a live hand before submitting an action")
            if self._transition.player_id != self.hero_seat:
                raise RuntimeError("it is not the human's turn")
            actor = self._transition.player_id
            street = self._transition.observation.stage
            # Engine validates BEFORE changing state; no revision or event on error.
            next_state = self._adapter.step_human_raise_to(amount)
            self._record_transition(next_state, actor, street, -1,
                                    kind="raise_to", raise_to=amount)
            self._run_bots()
            return self._view_unlocked()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return self.analytics.summary()

    def history(self, *, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        with self._lock:
            return self.analytics.history(offset=offset, limit=limit)

    def view(self) -> dict[str, Any]:
        with self._lock:
            return self._view_unlocked()

    def _take_action(self, action_index: int) -> None:
        assert self._adapter is not None and isinstance(self._transition, Turn)
        actor = self._transition.player_id
        street = self._transition.observation.stage
        # If illegal, step raises and no action/revision/event is committed.
        next_state = self._adapter.step(action_index)
        resolved = self._adapter.last_engine_action
        assert resolved is not None
        self._record_transition(next_state, actor, street, action_index,
                                kind=resolved.kind, raise_to=resolved.raise_to)

    def _record_transition(self, next_state: Turn | Terminal, actor: int,
                           street: str, action_index: int, *, kind: str,
                           raise_to: int | None) -> None:
        self._transition = next_state
        self._action_log.append({
            "seat": actor, "street": street, "index": action_index,
            "kind": kind, "raise_to": raise_to,
        })
        self.revision += 1
        if isinstance(next_state, Terminal):
            self._finish_terminal(next_state)

    def _finish_terminal(self, transition: Terminal) -> None:
        self.completed_hands += 1
        self.cumulative_bb += transition.rewards[self.hero_seat]
        self.analytics.append_completed(self._view_unlocked())

    def _run_bots(self) -> None:
        count = 0
        while isinstance(self._transition, Turn) and self._transition.player_id != self.hero_seat:
            count += 1
            if count > self.max_bot_actions:
                raise RuntimeError("bot action limit reached")
            action_index = self.bot.choose(self._transition.observation)
            self._take_action(action_index)

    def _view_unlocked(self) -> dict[str, Any]:
        if self._adapter is None or self._transition is None:
            raise RuntimeError("start_hand must be called before view")
        return project_human_view(
            self._adapter, self._transition,
            hero_seat=self.hero_seat, hand_number=self.hand_number,
            revision=self.revision, action_log=self._action_log,
            stack_mode=self.stack_mode,
            completed_hands=self.completed_hands, cumulative_bb=self.cumulative_bb,
        )


class HUNLSessionRegistry:
    """Opaque session ID lookup for a future local-only FastAPI router.

    Store in memory only. GUI-1.1 must add API-level request validation and
    restrict CORS and binding to localhost; no auth or remote hosting yet.
    """

    def __init__(self, *, max_sessions: int = 128) -> None:
        if type(max_sessions) is not int or max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self.max_sessions = max_sessions
        self._sessions: dict[str, HUNLGameSession] = {}
        self._lock = threading.RLock()

    def create(self, **session_options: Any) -> tuple[str, dict[str, Any]]:
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError("session capacity reached")
            session = HUNLGameSession(**session_options)
            first_view = session.start_hand()
            session_id = secrets.token_urlsafe(24)
            self._sessions[session_id] = session
            return session_id, first_view

    def get(self, session_id: str) -> HUNLGameSession:
        with self._lock:
            if not isinstance(session_id, str):
                raise KeyError("unknown session")
            return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        with self._lock:
            del self._sessions[session_id]
