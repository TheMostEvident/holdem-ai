"""GUI-1.0 behavior, privacy and action compatibility with the real HUNL200 API."""
from __future__ import annotations

import json
import random

import pytest

from holdem_ai.agents.diagnostic_opponents import CallingStation
from holdem_ai.game_service import HUNLGameSession, HUNLSessionRegistry


def _legal(view, *labels):
    candidates = [action for action in view["actions"] if action["label"] in labels]
    if not candidates:
        raise AssertionError(f"missing action {labels}, got {view['actions']}")
    return candidates[0]["index"]


def test_hunl200_human_first_and_legal_actions():
    session = HUNLGameSession(bot_policy=CallingStation(), seed=13)
    view = session.start_hand()
    assert view["game_type"] == "HUNL200"
    assert view["hand_number"] == 1
    assert view["hero_seat"] == view["button"] == 0
    assert view["your_turn"] is True
    assert view["to_act"] == 0
    assert len(view["hero_cards"]) == 2
    assert len(view["board"]) == 0
    assert view["to_call"] == 50
    assert view["pot"] == 150
    assert set(view["actions"][0]) == {"index", "label", "kind", "raise_to"}
    assert {a["label"] for a in view["actions"]}.issuperset({"fold", "call"})
    assert all(0 <= item["index"] < 7 for item in view["actions"])
    assert all(a["raise_to"] is None or isinstance(a["raise_to"], int) for a in view["actions"])


def test_serialized_view_does_not_leak_hidden_opponent_cards_or_future_deck():
    session = HUNLGameSession(bot_policy=CallingStation(), seed=42)
    state = session.start_hand()
    # The test is trusted and intentionally inspects engine state to check privacy.
    opponent_cards = session._adapter.engine.hole_cards[1]
    wire = json.dumps(state)
    assert all(card not in wire for card in opponent_cards)
    assert "hole_cards" not in state
    assert "deck" not in state
    assert "burned" not in state
    assert "critic" not in wire.lower()
    assert "private_cards" not in wire
    assert "training_state" not in wire
    with pytest.raises(PermissionError):
        session._adapter.training_state()


def test_fold_terminates_and_records_bb_return_but_hides_unshown_cards():
    session = HUNLGameSession(bot_policy=CallingStation(), seed=9)
    initial = session.start_hand()
    final = session.play_human_action(_legal(initial, "fold"), expected_revision=initial["revision"])
    assert final["phase"] == "complete"
    assert final["your_turn"] is False
    assert final["actions"] == []
    assert final["to_act"] is None
    assert final["hand_result"]["showdown"] is False
    assert final["hand_result"]["opponent_cards"] is None
    assert final["hand_result"]["payoffs_chips"] == [-50, 50]
    assert final["completed_hands"] == 1
    assert final["cumulative_bb"] == -0.5
    assert final["bb_per_100"] == -50.0
    assert sum(final["hand_result"]["final_stacks"]) == 40000
    next_hand = session.start_hand()
    assert next_hand["hand_number"] == 2
    assert next_hand["button"] == 1
    assert next_hand["revision"] > final["revision"]


def test_cannot_start_a_new_hand_while_one_is_live():
    session = HUNLGameSession(seed=9)
    session.start_hand()
    with pytest.raises(RuntimeError, match="finish"):
        session.start_hand()


def test_bad_human_action_and_stale_revision_are_rejected_without_mutation():
    session = HUNLGameSession(bot_policy=CallingStation(), seed=11)
    state = session.start_hand()
    with pytest.raises(ValueError, match="stale revision"):
        session.play_human_action(0, expected_revision=state["revision"] + 1)
    with pytest.raises(ValueError):
        session.play_human_action(999, expected_revision=state["revision"])
    with pytest.raises(TypeError):
        session.play_human_action(True, expected_revision=state["revision"])
    assert session.view() == state


def test_calling_station_can_play_full_showdown_human_only_check_call():
    session = HUNLGameSession(bot_policy=CallingStation(), seed=10)
    state = session.start_hand()
    for _ in range(30):
        if state["phase"] == "complete":
            break
        assert state["your_turn"] is True
        idx = _legal(state, "check", "call")
        state = session.play_human_action(idx, expected_revision=state["revision"])
    assert state["phase"] == "complete"
    assert state["street"] == "river"
    assert len(state["board"]) == 5
    assert state["hand_result"]["showdown"] is True
    assert len(state["hand_result"]["opponent_cards"]) == 2
    assert not set(state["hand_result"]["opponent_cards"]) & set(state["hero_cards"] + state["board"])
    assert sum(state["hand_result"]["payoffs_chips"]) == 0


def test_bot_receives_its_information_state_only():
    class SpyPolicy:
        def __init__(self):
            self.seen = []

        def act(self, observation, rng):
            assert observation.player_id == 1
            assert len(observation.private_cards) == 2
            assert len(observation.legal_action_mask) == 7
            self.seen.append(observation)
            for idx in (1, 2, 0):  # check, call, fold
                if idx in observation.legal_actions:
                    return idx
            return observation.legal_actions[0]

    spy = SpyPolicy()
    session = HUNLGameSession(bot_policy=spy, seed=22)
    start = session.start_hand()
    next_state = session.play_human_action(_legal(start, "call"), expected_revision=start["revision"])
    assert spy.seen
    hero_cards = set(start["hero_cards"])
    assert all(not (set(obs.private_cards) & hero_cards) for obs in spy.seen)
    assert next_state["phase"] in ("complete", "playing")


def test_illegal_bot_policy_is_rejected():
    class IllegalPolicy:
        def act(self, observation, rng):
            return 999

    session = HUNLGameSession(bot_policy=IllegalPolicy(), seed=22)
    start = session.start_hand()
    with pytest.raises(ValueError):
        session.play_human_action(_legal(start, "call"), expected_revision=start["revision"])


def test_registry_create_lookup_and_delete():
    registry = HUNLSessionRegistry(max_sessions=1)
    token, state = registry.create(bot_policy=CallingStation(), seed=20)
    assert len(token) > 20
    assert state == registry.get(token).view()
    with pytest.raises(RuntimeError, match="capacity"):
        registry.create()
    registry.delete(token)
    with pytest.raises(KeyError):
        registry.get(token)


def test_reproducibility_and_chip_conservation_multiple_hands():
    a = HUNLGameSession(bot_policy=CallingStation(), seed=2026)
    b = HUNLGameSession(bot_policy=CallingStation(), seed=2026)
    for _ in range(12):
        state_a = a.start_hand()
        state_b = b.start_hand()
        assert state_a == state_b
        for _ in range(40):
            if state_a["phase"] == "complete":
                break
            index = _legal(state_a, "check", "call")
            state_a = a.play_human_action(index, expected_revision=state_a["revision"])
            state_b = b.play_human_action(index, expected_revision=state_b["revision"])
            assert state_a == state_b
        assert state_a["phase"] == "complete"
        assert sum(state_a["hand_result"]["payoffs_chips"]) == 0
        assert sum(state_a["hand_result"]["final_stacks"]) == 40000
    assert a.view()["completed_hands"] == 12


def test_config_validation():
    with pytest.raises(ValueError):
        HUNLGameSession(hero_seat=1)
    with pytest.raises(ValueError):
        HUNLGameSession(stack_bb=0)
    with pytest.raises(ValueError):
        HUNLSessionRegistry(max_sessions=0)
