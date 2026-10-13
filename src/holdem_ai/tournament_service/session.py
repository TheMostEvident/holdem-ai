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
