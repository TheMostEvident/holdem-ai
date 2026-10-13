# Complete Python source volume 22/28

[Integrated technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Pinned source commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. All code literal and complete.

## tests/expand/test_e07_tournament.py

Original blob `a710c3749c15b9752fdbf52ab2776c9b15b78f1d` (2716 bytes).

<a name="py-253"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e07_tournament.py" sha="a710c3749c15b9752fdbf52ab2776c9b15b78f1d" -->

````````python
import pytest
from holdem_ai.expand.integration import SingleTableTournament,TournamentConfig
from holdem_ai.expand.environment.table_state import Action
from e07_fake_tournament import KnockoutEngine


def mk(**kw):return SingleTableTournament(TournamentConfig(**kw),engine_factory=KnockoutEngine)


def drain(t):
    h=t.start_hand(100+t.hands_played)
    assert h is not None
    while t.current_player_id is not None:t.act(Action('check'))
    return t.settle_hand()


def test_e07_tournament_knockouts_chip_ledger_and_prizes():
    t=mk(payouts=(.7,.3,0.))
    for i in range(3):t.register(f'p{i}',seat=i)
    t.close_registration()
    first=drain(t)
    assert first.hand_number==1
    assert len(first.eliminated_entries)==1
    assert sum(first.payoffs_chips)==0
    while not t.finished:drain(t)
    report=t.result()
    assert report.hands_played==2
    assert len(report.entries)==3
    assert {x[2] for x in report.entries}=={1,2,3}
    assert sum(x[3] for x in report.entries)==pytest.approx(1.)
    assert t.start_hand(333) is None


def test_e07_reentry_new_id_late_registration():
    t=mk(registration_until_hand=3,max_reentries_per_player=1)
    t.register('a',seat=0);t.register('b',seat=1);t.register('c',seat=2)
    with pytest.raises(ValueError):t.register('a',seat=3,reentry=True)
    first=drain(t)
    assert 'a#entry1' in first.eliminated_entries
    second=t.register('a',seat=0,reentry=True)
    assert second=='a#entry2'
    with pytest.raises(ValueError):t.register('a',seat=3,reentry=True)
    drain(t)
    assert len(t.entries)==4
    assert t.entries[0].eliminated_hand==1
    assert t.entries[-1].entry_id=='a#entry2'
    t.close_registration()
    while not t.finished:drain(t)
    assert len(t.result().entries)==4


def test_e07_tournament_guard_bad_blinds_ante_registration():
    with pytest.raises(NotImplementedError):TournamentConfig(ante_chips=1)
    with pytest.raises(ValueError):TournamentConfig(blind_levels=((2,1),))
    t=mk(registration_until_hand=0)
    t.register('a');t.register('b')
    drain(t)
    with pytest.raises(RuntimeError):t.register('late',seat=2)


def test_e07_no_midhand_mutation_and_no_forged_completion():
    t=mk()
    t.register('a');t.register('b')
    h=t.start_hand(1)
    with pytest.raises(RuntimeError):t.register('c',seat=2)
    with pytest.raises(RuntimeError):t.close_registration()
    with pytest.raises(RuntimeError):t.result()
    while t.current_player_id is not None:t.act(Action('check'))
    t.settle_hand()
    assert t.finished


def test_e07_raise_mapping_short_call():
    t=mk()
    t.register('a');t.register('b')
    t.start_hand(3)
    with pytest.raises(ValueError):t.act(Action('call'))
    t.act(Action('check'))

````````

<!-- END_PY_SOURCE path="tests/expand/test_e07_tournament.py" -->

## tests/test_blind_manager.py

Original blob `880eab0d7c27e9f91be99f6c4a1933c05d8d5932` (1089 bytes).

<a name="py-254"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_blind_manager.py" sha="880eab0d7c27e9f91be99f6c4a1933c05d8d5932" -->

````````python
import pytest

from holdem_ai.environment.blind_manager import BlindLevel, BlindManager


def test_blind_levels_repeat_last_level():
    b = BlindManager([(5, 10), (10, 20), (20, 40)], hands_per_level=3)
    assert [b.at_hand(i) for i in range(1, 12)] == [
        (0, BlindLevel(5, 10)), (0, BlindLevel(5, 10)), (0, BlindLevel(5, 10)),
        (1, BlindLevel(10, 20)), (1, BlindLevel(10, 20)), (1, BlindLevel(10, 20)),
        (2, BlindLevel(20, 40)), (2, BlindLevel(20, 40)), (2, BlindLevel(20, 40)),
        (2, BlindLevel(20, 40)), (2, BlindLevel(20, 40)),
    ]


@pytest.mark.parametrize('levels,hands', [([], 10), ([(2, 2)], 1), ([(0, 10)], 1),
                                          ([(5, 10), (4, 20)], 2), ([(5, 10)], 0),
                                          ([(5, 10)], True)])
def test_invalid_blinds(levels, hands):
    with pytest.raises(ValueError):
        BlindManager(levels, hands)


def test_blind_hand_number_validation():
    b = BlindManager([(1, 2)])
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            b.at_hand(value)

````````

<!-- END_PY_SOURCE path="tests/test_blind_manager.py" -->

## tests/test_corrected_rlcard_integration.py

Original blob `0c26823cc52601fe3bbf81ea2b8ea18644fd9190` (2786 bytes).

<a name="py-255"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_corrected_rlcard_integration.py" sha="0c26823cc52601fe3bbf81ea2b8ea18644fd9190" -->

````````python
"""End-to-end corrected engine tests; run with RLCard 1.2.0 installed."""
import pytest

pytest.importorskip('rlcard')

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.hand_engine import HandEngine
from holdem_ai.environment.rules_audit import audit_preflop


@pytest.mark.parametrize('button', [0, 1])
def test_corrected_heads_up_blinds_and_first_action(button):
    engine = HandEngine(starting_stack=200, seed=123, button=button)
    state, player = engine.reset()
    checks = audit_preflop(engine.env.game)
    assert all(c.passed for c in checks), checks
    assert player == button
    assert state['legal_actions']


@pytest.mark.parametrize('button', [0, 1])
def test_bb_acts_first_on_flop_after_preflop_call_and_check(button):
    engine = HandEngine(starting_stack=200, seed=12, button=button)
    _, player = engine.reset()
    assert player == button
    # BTN calls posted BB, BB checks its option.
    engine.step(1)
    assert engine.current_player == 1 - button
    engine.step(1)
    assert engine.current_player == 1 - button
    assert engine.env.game.stage.name == 'FLOP'
    assert len(engine.env.game.public_cards) == 3


def test_real_corrected_100_random_hands():
    engine = HandEngine(starting_stack=200, seed=42)
    agents = (RandomAgent(seed=42), RandomAgent(seed=43))
    for _ in range(100):
        result = engine.play_hand(agents)
        assert sum(result.payoffs_chips) == 0
        assert result.action_history


def test_short_blind_stack_goes_to_showdown_without_stalling():
    engine = HandEngine(starting_stack=2, seed=33)
    # With equal stacks of 2 chips, BB is all-in from posting the blind.
    state, _ = engine.reset()
    assert engine.env.game.players[1].remained_chips == 0
    for _ in range(20):
        if engine.is_over():
            break
        if 1 in state['legal_actions']:
            state, _ = engine.step(1)
        else:
            raise AssertionError('No CHECK_CALL action in nonterminal game')
    assert engine.is_over()
    assert sum(engine.get_result().payoffs_chips) == 0


def test_fold_ends_hand_without_dealing_board_cards():
    engine = HandEngine(starting_stack=200, seed=47)
    engine.reset()
    engine.step(0)
    assert engine.is_over()
    assert engine.env.game.public_cards == []
    result = engine.get_result()
    assert sum(result.payoffs_chips) == 0
    assert sorted(result.payoffs_chips) == [-1.0, 1.0]


def test_step_back_after_a_fold():
    engine = HandEngine(starting_stack=200, seed=47, allow_step_back=True)
    engine.reset()
    engine.step(0)
    assert engine.is_over()
    previous = engine.env.step_back()
    assert previous is not False
    assert not engine.env.is_over()
    assert all(x.passed for x in audit_preflop(engine.env.game))

````````

<!-- END_PY_SOURCE path="tests/test_corrected_rlcard_integration.py" -->

## tests/test_corrected_round.py

Original blob `d97b74c499c3f4c15d1a7dea69e29b581bb1f5d2` (3360 bytes).

<a name="py-256"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_corrected_round.py" sha="d97b74c499c3f4c15d1a7dea69e29b581bb1f5d2" -->

````````python
"""Independent betting scenarios requiring no RLCard installation."""
from enum import IntEnum
from types import SimpleNamespace

import pytest

from holdem_ai.environment.corrected_round import ActionId, HeadsUpBettingRound


class Status(IntEnum):
    ALIVE = 0
    ALLIN = 1
    FOLDED = 2


class Player:
    def __init__(self, stack, posted=0):
        self.in_chips = posted
        self.remained_chips = stack - posted
        self.status = Status.ALIVE

    def bet(self, chips):
        amount = min(chips, self.remained_chips)
        self.in_chips += amount
        self.remained_chips -= amount


def make_round(stacks=(200, 200)):
    players = [Player(stacks[0], 1), Player(stacks[1], 2)]
    r = HeadsUpBettingRound(players, 2, SimpleNamespace(pot=3), ActionId,
                            Status.ALIVE, Status.ALLIN, Status.FOLDED)
    r.start_new_round(0, raised=[1, 2])
    return r, players


def test_heads_up_preflop_blinds_call_then_big_blind_option():
    r, p = make_round()
    assert r.game_pointer == 0
    assert 1 in r.get_nolimit_legal_actions()
    r.proceed_round(p, ActionId.CHECK_CALL)
    assert p[0].in_chips == 2
    assert not r.is_over()
    assert r.game_pointer == 1
    assert ActionId.RAISE_POT in r.get_nolimit_legal_actions()
    r.proceed_round(p, ActionId.CHECK_CALL)
    assert r.is_over()


def test_min_raise_preflop_four_chips():
    r, players = make_round()
    assert r._raise_to(0, ActionId.RAISE_HALF_POT) == 4
    assert r._raise_to(0, ActionId.RAISE_POT) == 6
    r.proceed_round(players, ActionId.RAISE_HALF_POT)
    assert players[0].in_chips == 4
    assert r.last_full_raise == 2
    assert r.game_pointer == 1
    assert r._to_call(1) == 2


def test_short_all_in_does_not_reopen_betting():
    r, players = make_round(stacks=(200, 5))
    # BTN first full-raises to 4; BB shoves to 5, a short raise of 1.
    r.proceed_round(players, ActionId.RAISE_HALF_POT)
    assert ActionId.ALL_IN in r.get_nolimit_legal_actions()
    r.proceed_round(players, ActionId.ALL_IN)
    assert r.game_pointer == 0
    assert r._to_call(0) == 1
    assert r.get_nolimit_legal_actions() == [ActionId.FOLD, ActionId.CHECK_CALL]
    r.proceed_round(players, ActionId.CHECK_CALL)
    assert r.is_over()


def test_short_all_in_call_by_smaller_stack():
    r, players = make_round(stacks=(2, 200))
    assert ActionId.ALL_IN not in r.get_nolimit_legal_actions()
    r.proceed_round(players, ActionId.CHECK_CALL)
    assert players[0].in_chips == 2
    assert players[0].status == Status.ALLIN
    assert not r.is_over()  # BB still has the option to check or raise.
    assert r.game_pointer == 1
    r.proceed_round(players, ActionId.CHECK_CALL)
    assert r.is_over()


def test_fold_finishes_round():
    r, players = make_round()
    r.proceed_round(players, ActionId.FOLD)
    assert players[0].status == Status.FOLDED
    assert r.is_over()


def test_illegal_action_rejected():
    r, players = make_round(stacks=(2, 200))
    with pytest.raises(ValueError):
        r.proceed_round(players, ActionId.RAISE_POT)


def test_cannot_raise_once_opponent_is_all_in():
    r, players = make_round(stacks=(200, 200))
    r.proceed_round(players, ActionId.ALL_IN)
    assert players[0].status == Status.ALLIN
    assert r.game_pointer == 1
    assert r.get_nolimit_legal_actions() == [ActionId.FOLD, ActionId.CHECK_CALL]

````````

<!-- END_PY_SOURCE path="tests/test_corrected_round.py" -->

## tests/test_gui10_hunl_contract.py

Original blob `29d751b3311c6677f5338f8a2f455e3486249b44` (7544 bytes).

<a name="py-257"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_gui10_hunl_contract.py" sha="29d751b3311c6677f5338f8a2f455e3486249b44" -->

````````python
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

````````

<!-- END_PY_SOURCE path="tests/test_gui10_hunl_contract.py" -->

## tests/test_gui11_http_api.py

Original blob `79f70fa36d6529161acca76feb49fff80806da32` (9066 bytes).

<a name="py-258"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_gui11_http_api.py" sha="79f70fa36d6529161acca76feb49fff80806da32" -->

````````python
"""GUI-1.1 HTTP contract tests (requires GUI-1.0 + FastAPI/httpx)."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app


@pytest.fixture
def client():
    with TestClient(create_app(max_sessions=16)) as c:
        yield c


def new_game(client, **kwargs):
    res = client.post("/api/v1/hunl/sessions", json={"bot": "calling_station", "seed": 42, **kwargs})
    assert res.status_code == 201, res.text
    assert res.headers["cache-control"] == "no-store"
    return res.json()


def act(client, sid, state, kind=None):
    assert state["phase"] == "playing" and state["your_turn"]
    offers = state["actions"]
    if kind is None:
        offer = next((a for a in offers if a["kind"] in ("check", "call")), offers[0])
    else:
        offer = next(a for a in offers if a["kind"] == kind)
    response = client.post(
        f"/api/v1/hunl/sessions/{sid}/actions",
        json={"action_index": offer["index"], "expected_revision": state["revision"]},
    )
    assert response.status_code == 200, response.text
    return response.json()["state"]


def finish_hand(client, sid, state):
    for _ in range(100):
        if state["phase"] == "complete":
            return state
        state = act(client, sid, state)
    pytest.fail("hand did not terminate after 100 human actions")


def test_health_and_openapi(client):
    r = client.get("/api/v1/hunl/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "hunl-gui-local"}
    assert client.get("/openapi.json").status_code == 200


def test_create_get_state_no_private_leaks(client):
    item = new_game(client)
    sid, state = item["session_id"], item["state"]
    assert state["hero_seat"] == 0
    assert state["game_type"] == "HUNL200"
    assert state["your_turn"] and state["to_act"] == 0
    assert len(state["hero_cards"]) == 2
    assert sum(s["stack"] + s["total_contribution"] for s in state["seats"]) == 40_000
    assert client.get(f"/api/v1/hunl/sessions/{sid}").json() == item
    # The trusted engine has both hands; the browser must not have opponent's.
    session = client.app.state.gui11_registry.get(sid)
    secret = session._adapter.engine.hole_cards[1]  # only test accesses privileged private member
    visible = json.dumps(item)
    for card in secret:
        assert card not in state["hero_cards"]
        assert '"' + card + '"' not in visible
    assert "hole_cards" not in visible
    assert "critic" not in visible.lower()
    assert "_deck" not in visible
    assert "training_state" not in visible
    assert state["hand_result"] is None


def test_human_action_rev_increments_and_stale_request_fails(client):
    item = new_game(client)
    sid, before = item["session_id"], item["state"]
    after = act(client, sid, before)
    assert after["revision"] > before["revision"]
    stale = client.post(f"/api/v1/hunl/sessions/{sid}/actions", json={
        "action_index": before["actions"][0]["index"], "expected_revision": before["revision"]
    })
    assert stale.status_code == 409
    assert client.get(f"/api/v1/hunl/sessions/{sid}").json()["state"] == after


def test_invalid_action_rejected_without_mutation(client):
    item = new_game(client)
    sid, state = item["session_id"], item["state"]
    illegal = next(i for i in range(7) if i not in {a["index"] for a in state["actions"]})
    response = client.post(f"/api/v1/hunl/sessions/{sid}/actions", json={
        "action_index": illegal, "expected_revision": state["revision"]
    })
    assert response.status_code == 422
    assert client.get(f"/api/v1/hunl/sessions/{sid}").json()["state"] == state


def test_strict_http_payloads(client):
    for invalid in (
        {"bot": "unknown"}, {"bot": "random", "seed": "42"},
        {"big_blind": 10, "small_blind": 10}, {"surprise": 3},
        {"stack_bb": 1}, {"big_blind": True},
    ):
        assert client.post("/api/v1/hunl/sessions", json=invalid).status_code == 422
    item = new_game(client)
    sid, state = item["session_id"], item["state"]
    for invalid in (
        {"action_index": "2", "expected_revision": state["revision"]},
        {"action_index": True, "expected_revision": state["revision"]},
        {"action_index": 10, "expected_revision": state["revision"]},
        {"action_index": 2, "expected_revision": 0},
        {"action_index": 2, "expected_revision": state["revision"], "unexpected": 1},
    ):
        assert client.post(f"/api/v1/hunl/sessions/{sid}/actions", json=invalid).status_code == 422
    assert client.get(f"/api/v1/hunl/sessions/{sid}").json()["state"] == state


def test_complete_hand_showdown_and_next_hand_revision(client):
    item = new_game(client)
    sid, state = item["session_id"], item["state"]
    assert client.post(f"/api/v1/hunl/sessions/{sid}/next-hand", json={
        "expected_revision": state["revision"]
    }).status_code == 409
    finished = finish_hand(client, sid, state)
    assert finished["hand_result"]["showdown"] is True
    assert len(finished["hand_result"]["opponent_cards"]) == 2
    assert finished["completed_hands"] == 1
    assert finished["hand_result"]["payoffs_chips"][0] == -finished["hand_result"]["payoffs_chips"][1]
    assert finished["hand_result"]["payoffs_bb"][0] == finished["cumulative_bb"]
    res = client.post(f"/api/v1/hunl/sessions/{sid}/next-hand", json={
        "expected_revision": finished["revision"]
    })
    assert res.status_code == 200, res.text
    next_state = res.json()["state"]
    assert next_state["hand_number"] == 2
    assert next_state["button"] == 1
    assert next_state["completed_hands"] == 1
    assert next_state["revision"] > finished["revision"]
    assert next_state["hero_seat"] == 0
    assert next_state["hand_result"] is None
    assert len(next_state["hero_cards"]) == 2
    assert sum(s["stack"] + s["total_contribution"] for s in next_state["seats"]) == 40_000
    again = client.post(f"/api/v1/hunl/sessions/{sid}/next-hand", json={
        "expected_revision": finished["revision"]
    })
    assert again.status_code == 409


def test_multiple_hands_stats_and_fold_hidden(client):
    game = new_game(client)
    sid, state = game["session_id"], game["state"]
    fold = act(client, sid, state, kind="fold")
    assert fold["phase"] == "complete"
    assert fold["hand_result"]["showdown"] is False
    assert fold["hand_result"]["opponent_cards"] is None
    assert fold["completed_hands"] == 1
    assert fold["bb_per_100"] == pytest.approx(100 * fold["cumulative_bb"])
    next_res = client.post(f"/api/v1/hunl/sessions/{sid}/next-hand", json={
        "expected_revision": fold["revision"]
    })
    assert next_res.status_code == 200
    second = finish_hand(client, sid, next_res.json()["state"])
    assert second["completed_hands"] == 2
    assert second["bb_per_100"] == pytest.approx(50 * second["cumulative_bb"])


def test_missing_and_deleted_session(client):
    assert client.get("/api/v1/hunl/sessions/not-real").status_code == 404
    item = new_game(client)
    sid = item["session_id"]
    r = client.delete(f"/api/v1/hunl/sessions/{sid}")
    assert r.status_code == 204
    assert client.get(f"/api/v1/hunl/sessions/{sid}").status_code == 404
    assert client.delete(f"/api/v1/hunl/sessions/{sid}").status_code == 404


def test_sessions_isolated_and_capacity():
    with TestClient(create_app(max_sessions=1)) as c:
        first = new_game(c)
        assert c.post("/api/v1/hunl/sessions", json={}).status_code == 503
        assert c.delete(f"/api/v1/hunl/sessions/{first['session_id']}").status_code == 204
        assert c.post("/api/v1/hunl/sessions", json={}).status_code == 201
    with TestClient(create_app()) as c:
        a = new_game(c, seed=11)
        b = new_game(c, seed=12)
        assert a["session_id"] != b["session_id"]
        state_a = act(c, a["session_id"], a["state"])
        assert c.get(f"/api/v1/hunl/sessions/{b['session_id']}").json()["state"] == b["state"]
        assert state_a["revision"] > a["state"]["revision"]


def test_cors_is_only_local_frontend(client):
    allow = client.options(
        "/api/v1/hunl/sessions", headers={"Origin": "http://localhost:5173",
                                           "Access-Control-Request-Method": "POST"}
    )
    assert allow.status_code == 200
    assert allow.headers["access-control-allow-origin"] == "http://localhost:5173"
    denied = client.options(
        "/api/v1/hunl/sessions", headers={"Origin": "https://untrusted.example",
                                           "Access-Control-Request-Method": "POST"}
    )
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers


def test_builtin_bot_variants(client):
    for bot in ("random", "calling_station", "tight", "pot_pressure"):
        res = client.post("/api/v1/hunl/sessions", json={"bot": bot, "seed": 1})
        assert res.status_code == 201, (bot, res.text)
        s = res.json()["state"]
        assert s["your_turn"] or s["phase"] == "complete"

````````

<!-- END_PY_SOURCE path="tests/test_gui11_http_api.py" -->

## tests/test_gui12_readonly_api.py

Original blob `adb3185d531a768577d1426a78859964d4b536e3` (2754 bytes).

<a name="py-259"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_gui12_readonly_api.py" sha="adb3185d531a768577d1426a78859964d4b536e3" -->

````````python
"""GUI-1.2 read-only display contract against the real GUI-1.1 API."""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import create_app


def test_frontend_bootstrap_contract_and_cache_headers():
    with TestClient(create_app()) as client:
        response = client.post('/api/v1/hunl/sessions', json={
            'bot': 'random', 'seed': 42, 'big_blind': 100, 'small_blind': 50, 'stack_bb': 200,
        })
        assert response.status_code == 201
        body = response.json()
        state = body['state']
        assert response.headers['cache-control'] == 'no-store'
        assert isinstance(body['session_id'], str)
        assert state['game_type'] == 'HUNL200'
        assert len(state['hero_cards']) == 2
        assert state['phase'] == 'playing'
        assert len(state['seats']) == 2
        assert {s['role'] for s in state['seats']} == {'human', 'bot'}
        assert all(set(a) == {'index', 'label', 'kind', 'raise_to'} for a in state['actions'])
        assert state['hand_result'] is None
        assert 'hole_cards' not in str(state)
        assert 'opponent_cards' not in str(state)
        assert client.get(f"/api/v1/hunl/sessions/{body['session_id']}").json() == body
        assert client.delete(f"/api/v1/hunl/sessions/{body['session_id']}").status_code == 204


def test_browser_cors_is_restricted_to_local_development_origins():
    with TestClient(create_app()) as client:
        legit = client.options('/api/v1/hunl/sessions', headers={
            'Origin': 'http://127.0.0.1:5173',
            'Access-Control-Request-Method': 'POST',
        })
        assert legit.status_code == 200
        assert legit.headers['access-control-allow-origin'] == 'http://127.0.0.1:5173'
        remote = client.options('/api/v1/hunl/sessions', headers={
            'Origin': 'https://example.com',
            'Access-Control-Request-Method': 'POST',
        })
        assert 'access-control-allow-origin' not in remote.headers


def test_contract_hides_bot_cards_on_fold_finish():
    with TestClient(create_app()) as client:
        game = client.post('/api/v1/hunl/sessions', json={'bot': 'calling_station', 'seed': 1}).json()
        state = game['state']
        fold_index = next(a['index'] for a in state['actions'] if a['kind'] == 'fold')
        response = client.post(f"/api/v1/hunl/sessions/{game['session_id']}/actions", json={
            'action_index': fold_index, 'expected_revision': state['revision'],
        })
        assert response.status_code == 200
        state = response.json()['state']
        assert state['phase'] == 'complete'
        assert state['hand_result']['showdown'] is False
        assert state['hand_result']['opponent_cards'] is None

````````

<!-- END_PY_SOURCE path="tests/test_gui12_readonly_api.py" -->

## tests/test_gui13_interactive_betting.py

Original blob `b41a06a22c2563a883923c4d7d2b2f81c69b64ed` (4882 bytes).

<a name="py-260"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_gui13_interactive_betting.py" sha="b41a06a22c2563a883923c4d7d2b2f81c69b64ed" -->

````````python
"""GUI-1.3: actual engine transitions through browser-facing HTTP contracts."""
import pytest
from fastapi.testclient import TestClient
from backend.main import create_app

PREFIX = '/api/v1/hunl/sessions'

@pytest.fixture
def client():
    with TestClient(create_app(max_sessions=16)) as api:
        yield api


def create(client, seed=42, bot='calling_station'):
    response = client.post(PREFIX, json={'seed': seed, 'bot': bot})
    assert response.status_code == 201, response.text
    return response.json()


def test_human_exact_raise_and_bot_response(client):
    game = create(client)
    sid, state = game['session_id'], game['state']
    assert state['your_turn'] and state['can_raise_to']
    target = state['min_raise_to'] + 1
    assert target < state['max_raise_to']
    r = client.post(f'{PREFIX}/{sid}/raise-to', json={
        'raise_to': target, 'expected_revision': state['revision'],
    })
    assert r.status_code == 200, r.text
    nxt = r.json()['state']
    assert nxt['revision'] > state['revision']
    assert nxt['action_log'][0]['index'] == -1
    assert nxt['action_log'][0]['kind'] == 'raise_to'
    assert nxt['action_log'][0]['raise_to'] == target
    assert any(event['seat'] == 1 for event in nxt['action_log'])
    assert not any(k in str(nxt).lower() for k in ('_deck', 'criticstate', 'burned'))
    assert nxt['hand_result'] is None or nxt['hand_result']['opponent_cards'] is None or nxt['hand_result']['showdown']


def test_invalid_raise_is_atomic_and_revision_safe(client):
    game = create(client)
    sid, state = game['session_id'], game['state']
    endpoint = f'{PREFIX}/{sid}/raise-to'
    for target in (0, state['min_raise_to'] - 1, state['max_raise_to'] + 1):
        response = client.post(endpoint, json={'raise_to': target, 'expected_revision': state['revision']})
        assert response.status_code == 422, response.text
        unchanged = client.get(f'{PREFIX}/{sid}').json()['state']
        assert unchanged == state
    first = client.post(endpoint, json={'raise_to': state['min_raise_to'], 'expected_revision': state['revision']})
    assert first.status_code == 200, first.text
    stale = client.post(endpoint, json={'raise_to': state['min_raise_to'], 'expected_revision': state['revision']})
    assert stale.status_code == 409
    assert 'stale' in stale.json()['detail']


def test_browser_regular_action_and_next_hand(client):
    game = create(client, seed=7)
    sid, state = game['session_id'], game['state']
    assert state['your_turn']
    fold = next(a for a in state['actions'] if a['kind'] == 'fold')
    action = client.post(f'{PREFIX}/{sid}/actions', json={
        'action_index': fold['index'], 'expected_revision': state['revision']})
    assert action.status_code == 200, action.text
    finished = action.json()['state']
    assert finished['phase'] == 'complete'
    assert finished['hand_result']['opponent_cards'] is None
    following = client.post(f'{PREFIX}/{sid}/next-hand', json={'expected_revision': finished['revision']})
    assert following.status_code == 200, following.text
    follow_state = following.json()['state']
    assert follow_state['hand_number'] == 2
    assert follow_state['button'] == 1
    assert follow_state['completed_hands'] == 1
    assert follow_state['phase'] == 'playing'


def test_reject_bool_fraction_unknown_json_and_nonhuman_turn(client):
    game = create(client)
    sid, state = game['session_id'], game['state']
    url = f'{PREFIX}/{sid}/raise-to'
    for target in (True, 250.5, '250', None):
        response = client.post(url, json={'raise_to': target, 'expected_revision': state['revision']})
        assert response.status_code == 422, response.text
    extra = client.post(url, json={'raise_to': 300, 'expected_revision': state['revision'], 'opponent_cards': ['sA', 'hA']})
    assert extra.status_code == 422
    assert client.get(f'{PREFIX}/{sid}').json()['state'] == state


def test_fold_check_call_loop_and_chip_conservation(client):
    game = create(client, seed=19)
    sid, state = game['session_id'], game['state']
    for _ in range(80):
        if state['phase'] == 'complete':
            break
        offer = next((o for o in state['actions'] if o['kind'] in ('check', 'call')), None)
        if offer is None:
            offer = state['actions'][0]
        response = client.post(f'{PREFIX}/{sid}/actions', json={
            'action_index': offer['index'], 'expected_revision': state['revision']})
        assert response.status_code == 200, response.text
        state = response.json()['state']
    else:
        pytest.fail('hand did not terminate')
    assert state['hand_result'] is not None
    assert sum(state['hand_result']['final_stacks']) == 40000
    assert sum(state['hand_result']['payoffs_chips']) == 0
    if state['hand_result']['showdown']:
        assert len(state['hand_result']['opponent_cards']) == 2

````````

<!-- END_PY_SOURCE path="tests/test_gui13_interactive_betting.py" -->

## tests/test_gui14_trained_agents.py

Original blob `99322ca9844b54d2c9f4336f0f9b605931253bb9` (7689 bytes).

<a name="py-261"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_gui14_trained_agents.py" sha="99322ca9844b54d2c9f4336f0f9b605931253bb9" -->

````````python
"""GUI-1.4 local trained-agent selection, loading, privacy, and HTTP regression."""
from __future__ import annotations

import json
import sys
import types

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from holdem_ai.game_service.checkpoint_catalog import CheckpointCatalog

API = '/api/v1/hunl'


def _client(environment=None):
    return TestClient(create_app(catalog=CheckpointCatalog(environment=environment or {})))


def test_catalog_no_models_does_not_load_torch():
    catalog = CheckpointCatalog(environment={})
    opts = catalog.options()
    assert [o['id'] for o in opts] == [
        'random', 'calling_station', 'tight', 'pot_pressure', 'ppo', 'qboost']
    assert all(x['available'] for x in opts[:4])
    assert not any(x['available'] for x in opts[4:])
    with _client() as client:
        res = client.get(f'{API}/bots')
        assert res.status_code == 200
        assert res.headers['cache-control'] == 'no-store'
        assert res.json()['device'] == 'cpu'
        assert res.json()['policy_mode'] == 'sampled'
        assert res.json()['bots'] == opts
        assert client.post(f'{API}/sessions', json={'bot': 'random'}).status_code == 201


def test_unconfigured_checkpoint_never_falls_back_to_random():
    with _client() as client:
        for bot in ('ppo', 'qboost'):
            result = client.post(f'{API}/sessions', json={'bot': bot})
            assert result.status_code == 422
            assert 'checkpoint' in result.json()['detail'].lower()
            assert not client.app.state.gui11_registry._sessions


def test_invalid_neural_game_configuration_rejected():
    with _client() as client:
        for bad in ({'big_blind': 20, 'small_blind': 10},
                    {'stack_bb': 100}, {'big_blind': 100, 'small_blind': 25}):
            result = client.post(f'{API}/sessions', json={'bot': 'ppo', **bad})
            assert result.status_code == 422
            assert 'HUNL200' in result.json()['detail']


def test_catalog_config_validation():
    with pytest.raises(ValueError, match='DEVICE'):
        CheckpointCatalog(environment={'HOLDEM_AI_GUI_DEVICE': 'x'})
    with pytest.raises(ValueError, match='POLICY_MODE'):
        CheckpointCatalog(environment={'HOLDEM_AI_GUI_POLICY_MODE': 'test'})
    assert CheckpointCatalog(environment={'HOLDEM_AI_GUI_DEVICE': 'cuda',
                                          'HOLDEM_AI_GUI_POLICY_MODE': 'greedy'}).mode == 'greedy'


def _install_loader_stubs(monkeypatch, calls, seen, failure=False):
    """Stub Phase-16 loader in minimal Phase 1-4 test fixture.

    In the FULL Phase 0-16 repo the real weights-only Phase-16 loader is used.
    """
    fake_pkg = types.ModuleType('holdem_ai.evaluation')
    fake_pkg.__path__ = []
    fake_loader = types.ModuleType('holdem_ai.evaluation.checkpoint_loader')

    class Actor:
        config = types.SimpleNamespace(dropout=0.0)

    def load_actor(path, *, device):
        calls.append((str(path), device))
        if failure:
            raise ValueError('corrupt file / VERY SECRET path')
        return Actor()

    class Policy:
        def __init__(self, actor, *, device, big_blind, greedy):
            assert isinstance(actor, Actor)
            assert big_blind == 100
            seen.append(('constructed', device, greedy))

        def act(self, observation, rng):
            seen.append(observation)
            assert len(observation.private_cards) == 2
            # Bot must see ONLY its own information state, never both hands.
            return next((a for a in (1, 2, 0, 6, 3, 4, 5) if a in observation.legal_actions))

    fake_loader.load_actor = load_actor
    fake_policy = types.ModuleType('holdem_ai.agents.neural_policy')
    fake_policy.NeuralActorPolicy = Policy
    monkeypatch.setitem(sys.modules, 'holdem_ai.evaluation', fake_pkg)
    monkeypatch.setitem(sys.modules, 'holdem_ai.evaluation.checkpoint_loader', fake_loader)
    monkeypatch.setitem(sys.modules, 'holdem_ai.agents.neural_policy', fake_policy)


def test_trained_http_play_and_private_cards(tmp_path, monkeypatch):
    path = tmp_path / 'trusted.pt'
    path.write_bytes(b'fake-test-fixture-only')
    calls, seen = [], []
    _install_loader_stubs(monkeypatch, calls, seen)
    env = {'HOLDEM_AI_PPO_CHECKPOINT': str(path),
           'HOLDEM_AI_GUI_POLICY_MODE': 'greedy'}
    with _client(env) as client:
        info = client.get(f'{API}/bots').json()
        assert next(x for x in info['bots'] if x['id'] == 'ppo')['available']
        assert next(x for x in info['bots'] if x['id'] == 'qboost')['available'] is False
        assert str(path) not in json.dumps(info)
        game = client.post(f'{API}/sessions', json={'bot': 'ppo', 'seed': 42})
        assert game.status_code == 201, game.text
        sid = game.json()['session_id']
        state = game.json()['state']
        assert calls == [(str(path), 'cpu')]
        assert seen[0] == ('constructed', 'cpu', True)
        secret = client.app.state.gui11_registry.get(sid)._adapter.engine.hole_cards[1]
        for card in secret:
            assert '"' + card + '"' not in json.dumps(game.json())
        for _ in range(100):
            if state['phase'] == 'complete':
                break
            offer = next((a for a in state['actions'] if a['kind'] in ('check','call')), state['actions'][0])
            result = client.post(f'{API}/sessions/{sid}/actions', json={
                'action_index': offer['index'], 'expected_revision': state['revision']})
            assert result.status_code == 200, result.text
            state = result.json()['state']
        seen_bot = [o for o in seen if getattr(o, 'player_id', None) == 1]
        assert seen_bot
        assert all(set(o.private_cards) == set(secret) for o in seen_bot)
        assert all(not (set(o.private_cards) & set(game.json()['state']['hero_cards'])) for o in seen_bot)
        assert all('hole_cards' not in dict(o.public_state) for o in seen_bot)
        assert state['phase'] == 'complete'
        assert sum(state['hand_result']['payoffs_chips']) == 0
        next_res = client.post(f'{API}/sessions/{sid}/next-hand', json={
            'expected_revision': state['revision']})
        assert next_res.status_code == 200
        other = client.post(f'{API}/sessions', json={'bot': 'ppo', 'seed': 99})
        assert other.status_code == 201
        assert len(calls) == 1  # Frozen Actor cached; independent sessions have different RNGs.


def test_corrupt_trained_checkpoint_reports_safe_error(tmp_path, monkeypatch):
    path = tmp_path / 'secret_local_weights.pt'
    path.write_bytes(b'corrupt')
    calls, seen = [], []
    _install_loader_stubs(monkeypatch, calls, seen, failure=True)
    with _client({'HOLDEM_AI_QBOOST_CHECKPOINT': str(path)}) as client:
        result = client.post(f'{API}/sessions', json={'bot': 'qboost'})
        assert result.status_code == 422
        assert 'Could not load' in result.json()['detail']
        assert 'secret_local_weights' not in result.text
        assert 'VERY SECRET' not in result.text
        assert not client.app.state.gui11_registry._sessions


def test_checkpoint_reloaded_when_file_changes(tmp_path, monkeypatch):
    path = tmp_path / 'trusted.pt'
    path.write_bytes(b'aaa')
    calls, seen = [], []
    _install_loader_stubs(monkeypatch, calls, seen)
    catalog = CheckpointCatalog(environment={'HOLDEM_AI_PPO_CHECKPOINT': str(path)})
    first = catalog.build_policy('ppo')
    assert catalog.build_policy('ppo') is first
    path.write_bytes(b'aaaaaa')  # different size invalidates cache
    second = catalog.build_policy('ppo')
    assert second is not first
    assert len(calls) == 2

````````

<!-- END_PY_SOURCE path="tests/test_gui14_trained_agents.py" -->

## tests/test_gui15_session_analytics.py

Original blob `eb5a5b2b170d24d501a283e71d8892b309637595` (8219 bytes).

<a name="py-262"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_gui15_session_analytics.py" sha="eb5a5b2b170d24d501a283e71d8892b309637595" -->

````````python
"""GUI-1.5: completed-hand ledger, summary, privacy, pagination, API isolation."""
from __future__ import annotations

import json
import math

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from holdem_ai.game_service.session_analytics import SessionAnalytics

API = '/api/v1/hunl'


def _finish(client, sid, state, *, fold=False):
    for _ in range(200):
        if state['phase'] == 'complete':
            return state
        action = next((a for a in state['actions'] if a['kind'] == ('fold' if fold else 'check')), None)
        if action is None:
            action = next((a for a in state['actions'] if a['kind'] == ('call' if not fold else 'fold')), None)
        if action is None:
            action = state['actions'][0]
        resp = client.post(f'{API}/sessions/{sid}/actions', json={
            'action_index': action['index'], 'expected_revision': state['revision']})
        assert resp.status_code == 200, resp.text
        state = resp.json()['state']
    pytest.fail('did not finish hand within 200 human decisions')


def _new(client, seed=42):
    r = client.post(f'{API}/sessions', json={'bot': 'calling_station', 'seed': seed})
    assert r.status_code == 201, r.text
    return r.json()['session_id'], r.json()['state']


def test_empty_stats_and_history_are_clean_and_cached_nowhere():
    with TestClient(create_app()) as c:
        sid, state = _new(c)
        for endpoint in ('stats', 'history'):
            r = c.get(f'{API}/sessions/{sid}/{endpoint}')
            assert r.status_code == 200
            assert r.headers['cache-control'] == 'no-store'
        summary = c.get(f'{API}/sessions/{sid}/stats').json()
        assert summary['completed_hands'] == 0
        assert summary['ci95_lower_bb100'] is None
        assert summary['net_bb'] == 0
        history = c.get(f'{API}/sessions/{sid}/history').json()
        assert history['items'] == [] and history['has_more'] is False


def test_folded_hands_never_reveal_opponent_cards_in_completed_journal():
    with TestClient(create_app()) as c:
        sid, state = _new(c, seed=42)
        secret = c.app.state.gui11_registry.get(sid)._adapter.engine.hole_cards[1]
        state = _finish(c, sid, state, fold=True)
        assert state['hand_result']['showdown'] is False
        history = c.get(f'{API}/sessions/{sid}/history').json()
        assert history['total'] == 1
        record = history['items'][0]
        assert record['hand_number'] == 1
        assert record['result']['opponent_cards'] is None
        assert record['result']['showdown'] is False
        for card in secret:
            assert '"' + card + '"' not in json.dumps(history)
        assert record['hero_cards'] == state['hero_cards']
        assert record['action_log'] == state['action_log']
        assert record['hero_net_chips'] == state['hand_result']['payoffs_chips'][0]
        assert math.isclose(record['hero_net_bb'], state['hand_result']['payoffs_bb'][0])


def test_two_hand_ledger_and_stats_match_session_and_payoffs():
    with TestClient(create_app()) as c:
        sid, state = _new(c, seed=44)
        results = []
        for index in range(3):
            state = _finish(c, sid, state)
            results.append(state['hand_result']['payoffs_bb'][0])
            assert c.get(f'{API}/sessions/{sid}/history').json()['completed_hands'] == len(results)
            if index < 2:
                r = c.post(f'{API}/sessions/{sid}/next-hand', json={'expected_revision': state['revision']})
                assert r.status_code == 200
                state = r.json()['state']
        stats = c.get(f'{API}/sessions/{sid}/stats').json()
        ledger = c.get(f'{API}/sessions/{sid}/history').json()['items']
        assert [r['hand_number'] for r in ledger] == [3, 2, 1]
        assert stats['completed_hands'] == 3
        assert stats['wins'] + stats['losses'] + stats['break_even'] == 3
        assert stats['retained_hand_records'] == 3
        assert math.isclose(stats['net_bb'], sum(results), abs_tol=1e-8)
        assert math.isclose(stats['bb_per_100'], 100 * sum(results)/3, abs_tol=1e-8)
        assert math.isclose(stats['net_bb'], state['cumulative_bb'], abs_tol=1e-8)
        assert stats['ci95_lower_bb100'] is not None
        assert stats['ci95_lower_bb100'] <= stats['ci95_upper_bb100']
        assert math.isclose(ledger[0]['cumulative_bb'], stats['net_bb'], abs_tol=1e-8)
        assert ledger[-1]['button'] == 0
        assert ledger[1]['button'] == 1
        assert not any('critic_state' in r or 'future_deck' in r for r in ledger)


def test_history_offset_limit_and_input_validation():
    with TestClient(create_app()) as c:
        sid, state = _new(c, seed=9)
        for _ in range(4):
            state = _finish(c, sid, state, fold=True)
            state = c.post(f'{API}/sessions/{sid}/next-hand',
                           json={'expected_revision': state['revision']}).json()['state']
        page1 = c.get(f'{API}/sessions/{sid}/history?offset=0&limit=2').json()
        page2 = c.get(f'{API}/sessions/{sid}/history?offset=2&limit=2').json()
        assert [h['hand_number'] for h in page1['items']] == [4, 3]
        assert [h['hand_number'] for h in page2['items']] == [2, 1]
        assert page1['has_more'] and not page2['has_more']
        assert c.get(f'{API}/sessions/{sid}/history?offset=-1').status_code == 422
        assert c.get(f'{API}/sessions/{sid}/history?limit=0').status_code == 422
        assert c.get(f'{API}/sessions/{sid}/history?limit=101').status_code == 422
        assert c.get(f'{API}/sessions/{sid}/history?offset=999').json()['items'] == []


def test_aggregate_survives_record_retention_and_history_is_defensive_copy():
    store = SessionAnalytics(history_limit=2)
    for n, chips in enumerate([100, -50, 75], 1):
        store.append_completed({
            'phase': 'complete', 'hand_number': n, 'hero_seat': 0,
            'button': (n-1) % 2, 'street': 'river',
            'big_blind': 100, 'small_blind': 50, 'hero_cards': ['sA', 'hK'],
            'board': [], 'action_log': [],
            'hand_result': {'payoffs_chips': [chips, -chips],
                            'payoffs_bb': [chips/100, -chips/100],
                            'final_stacks': [20000+chips, 20000-chips],
                            'awards': [0, 0], 'showdown': False,
                            'opponent_cards': None},
        })
    assert store.summary()['completed_hands'] == 3
    assert store.summary()['retained_hand_records'] == 2
    assert store.summary()['records_truncated']
    assert store.summary()['net_chips'] == 125
    page = store.history()
    assert [h['hand_number'] for h in page['items']] == [3, 2]
    page['items'][0]['hero_cards'][0] = 'INVALID'
    assert store.history()['items'][0]['hero_cards'][0] == 'sA'
    with pytest.raises(ValueError):
        store.history(limit=101)
    with pytest.raises(ValueError):
        SessionAnalytics(history_limit=0)


def test_session_isolation_and_deletion():
    with TestClient(create_app()) as c:
        one, first = _new(c, seed=42)
        two, other = _new(c, seed=43)
        _finish(c, one, first, fold=True)
        assert c.get(f'{API}/sessions/{one}/stats').json()['completed_hands'] == 1
        assert c.get(f'{API}/sessions/{two}/stats').json()['completed_hands'] == 0
        assert c.delete(f'{API}/sessions/{one}').status_code == 204
        assert c.get(f'{API}/sessions/{one}/stats').status_code == 404
        assert c.get(f'{API}/sessions/{one}/history').status_code == 404
        assert c.get(f'{API}/sessions/{two}/history').json()['total'] == 0


def test_statistics_do_not_update_mid_hand_or_on_invalid_action():
    with TestClient(create_app()) as c:
        sid, state = _new(c)
        before = c.get(f'{API}/sessions/{sid}/stats').json()
        bad = c.post(f'{API}/sessions/{sid}/actions', json={
            'action_index': 6 if 6 not in [x['index'] for x in state['actions']] else 99,
            'expected_revision': state['revision']})
        assert bad.status_code == 422
        assert c.get(f'{API}/sessions/{sid}/stats').json() == before
        assert c.get(f'{API}/sessions/{sid}/history').json()['items'] == []

````````

<!-- END_PY_SOURCE path="tests/test_gui15_session_analytics.py" -->

## tests/test_gui16_continuous_stack.py

Original blob `2d3084bd922382e0307bb8f12e7f3be32416bc5f` (6973 bytes).

<a name="py-263"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_gui16_continuous_stack.py" sha="2d3084bd922382e0307bb8f12e7f3be32416bc5f" -->

````````python
"""GUI-1.6 end-to-end stack carryover, cash accounting, bust and mode isolation.

Runs on the full repository after GUI-1.0~1.6 is installed.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient
import pytest

from backend.main import create_app
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
from holdem_ai.game_service.session_manager import HUNLGameSession
from holdem_ai.agents.diagnostic_opponents import CallingStation

API = '/api/v1/hunl'


def _create(client, **options):
    result = client.post(f'{API}/sessions', json={'bot': 'calling_station', 'seed': 42, **options})
    assert result.status_code == 201, result.text
    return result.json()['session_id'], result.json()['state']


def _action(client, sid, state, action_index):
    response = client.post(f'{API}/sessions/{sid}/actions', json={
        'action_index': action_index, 'expected_revision': state['revision']})
    assert response.status_code == 200, response.text
    return response.json()['state']


def _complete_by_folding(client, sid, state):
    for _ in range(80):
        if state['phase'] == 'complete':
            return state
        assert state['your_turn']
        fold = next(a for a in state['actions'] if a['kind'] == 'fold')
        state = _action(client, sid, state, fold['index'])
    pytest.fail('more than 80 fold decisions in one hand')


def _next(client, sid, state):
    response = client.post(f'{API}/sessions/{sid}/next-hand', json={
        'expected_revision': state['revision']})
    assert response.status_code == 200, response.text
    return response.json()['state']


def test_browser_default_is_continuous_and_chips_carried_exactly():
    with TestClient(create_app()) as client:
        sid, state = _create(client)
        assert state['stack_mode'] == 'continuous'
        assert state['starting_stacks'] == [20000, 20000]
        for h in range(1, 11):
            opening = tuple(state['starting_stacks'])
            assert tuple(s['stack'] + s['total_contribution'] for s in state['seats']) == opening
            assert state['button'] == (h - 1) % 2
            assert state['hand_number'] == h
            final = _complete_by_folding(client, sid, state)
            result = final['hand_result']
            assert sum(result['final_stacks']) == 40000
            assert tuple(x['stack'] for x in final['seats']) == tuple(result['final_stacks'])
            assert result['payoffs_chips'] == [result['final_stacks'][i] - opening[i] for i in (0, 1)]
            assert not final['match_over']
            # Never expose opponent hole cards when a player folds.
            assert result['opponent_cards'] is None
            assert 'hole_cards' not in json.dumps(final)
            journal = client.get(f'{API}/sessions/{sid}/history').json()
            assert journal['items'][0]['starting_stacks'] == list(opening)
            if h < 10:
                state = _next(client, sid, final)
                assert tuple(state['starting_stacks']) == tuple(result['final_stacks'])
                assert state['revision'] > final['revision']


def test_browser_explicit_independent_mode_keeps_original_behavior():
    with TestClient(create_app()) as client:
        sid, state = _create(client, stack_mode='independent')
        assert state['stack_mode'] == 'independent'
        ended = _complete_by_folding(client, sid, state)
        new_state = _next(client, sid, ended)
        assert new_state['starting_stacks'] == [20000, 20000]
        assert ended['hand_result']['final_stacks'] != [20000, 20000]


def test_continuous_all_in_bust_finishes_match_and_blocks_next_hand():
    with TestClient(create_app()) as client:
        sid, state = _create(client, seed=123)
        allin = next(a for a in state['actions'] if a['kind'] == 'all_in')
        ended = _action(client, sid, state, allin['index'])
        # CallingStation calls the all-in: all five board cards are run out.
        assert ended['phase'] == 'complete'
        assert ended['hand_result']['showdown']
        if ended['hand_result']['final_stacks'] == [20000, 20000]:
            pytest.skip('Extremely rare exact split with same seed; retry another fixed seed')
        assert ended['match_over']
        assert sum(ended['hand_result']['final_stacks']) == 40000
        assert min(ended['hand_result']['final_stacks']) == 0
        assert ended['winner_seat'] == ended['hand_result']['final_stacks'].index(40000)
        rejected = client.post(f'{API}/sessions/{sid}/next-hand', json={
            'expected_revision': ended['revision']})
        assert rejected.status_code == 409
        reread = client.get(f'{API}/sessions/{sid}').json()['state']
        assert reread == ended  # no revision/hand counter mutation on rejection
        stale = client.post(f'{API}/sessions/{sid}/next-hand', json={
            'expected_revision': ended['revision'] - 1})
        assert stale.status_code == 409


def test_training_reset_unchanged_and_direct_session_default_independent():
    adapter = HUNL200Adapter(big_blind=100, small_blind=50, stack_bb=200)
    adapter.reset(seed=90)
    assert tuple(adapter.engine.initial_stacks) == (20000, 20000)
    adapter.engine.stacks[0] -= 10  # simulate mutation of game before independent reset
    adapter.reset(seed=91)
    assert tuple(adapter.engine.initial_stacks) == (20000, 20000)
    session = HUNLGameSession(bot_policy=CallingStation(), seed=10)
    assert session.stack_mode == 'independent'
    assert session.start_hand()['stack_mode'] == 'independent'


def test_bad_stack_modes_rejected_safely():
    with pytest.raises(ValueError, match='stack_mode'):
        HUNLGameSession(stack_mode='unexpected')
    with TestClient(create_app()) as client:
        assert client.post(f'{API}/sessions', json={'stack_mode': 'x'}).status_code == 422


def test_automatic_short_blind_terminal_counts_and_finishes_match():
    """Regression: a hand may finish during reset without a human or bot action."""
    with TestClient(create_app()) as client:
        sid, state = _create(client, seed=42)
        auto_settlements = 0
        for _ in range(290):
            if state['phase'] == 'playing':
                state = _complete_by_folding(client, sid, state)
            if state['action_log'] == []:
                auto_settlements += 1
            assert state['completed_hands'] == state['hand_number']
            assert sum(state['hand_result']['final_stacks']) == 40000
            assert tuple(x['stack'] for x in state['seats']) == tuple(state['hand_result']['final_stacks'])
            if state['match_over']:
                break
            state = _next(client, sid, state)
        assert state['match_over']
        assert state['completed_hands'] == 267
        assert auto_settlements >= 1
        assert state['hand_result']['final_stacks'] == [0, 40000]
        assert client.get(f'{API}/sessions/{sid}/stats').json()['completed_hands'] == 267

````````

<!-- END_PY_SOURCE path="tests/test_gui16_continuous_stack.py" -->

## tests/test_hand_engine.py

Original blob `d2a4ee6842bd0b51b9cee49db68317266b97de6f` (2191 bytes).

<a name="py-264"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_hand_engine.py" sha="d2a4ee6842bd0b51b9cee49db68317266b97de6f" -->

````````python
"""Offline tests for the public adapter API with a stand-in RLCard Env."""
from types import SimpleNamespace

import pytest

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.hand_engine import HandEngine


class FakeEnv:
    def __init__(self):
        self.game = SimpleNamespace(big_blind=2, dealer_id=0)
        self.actions = []
        self.counter = 0

    def reset(self):
        self.counter = 0
        self.actions = []
        return {'legal_actions': {1: None, 2: None}}, 0

    def is_over(self):
        return self.counter == 2

    def step(self, action):
        self.actions.append(action)
        self.counter += 1
        if self.counter == 1:
            return {'legal_actions': {0: None, 1: None}}, 1
        return {'legal_actions': {}}, 0

    def get_payoffs(self):
        return (4, -4)


@pytest.fixture
def engine(monkeypatch):
    from holdem_ai.environment import corrected_game
    monkeypatch.setattr(corrected_game, 'make_corrected_rlcard_env', lambda **kwargs: FakeEnv())
    return HandEngine(starting_stack=200, seed=123)


def test_hand_engine_play_hand_is_zero_sum(engine):
    result = engine.play_hand((RandomAgent(1), RandomAgent(2)))
    assert result.payoffs_chips == (4.0, -4.0)
    assert result.payoffs_bb == (2.0, -2.0)
    assert len(result.action_history) == 2
    assert [actor for actor, _ in result.action_history] == [0, 1]


def test_reject_illegal_action(engine):
    engine.reset()
    with pytest.raises(ValueError, match='Illegal action'):
        engine.step(999)


def test_random_agent_never_leaves_legal_actions():
    agent = RandomAgent(seed=7)
    state = {'legal_actions': {0: None, 3: None, 4: None}}
    for _ in range(100):
        assert agent.select_action(state) in state['legal_actions']


def test_cannot_read_result_before_hand_finished(engine):
    engine.reset()
    with pytest.raises(RuntimeError, match='not over'):
        engine.get_result()


def test_button_rotates_between_hands(engine):
    engine.reset()
    assert engine.env.game.dealer_id == 0
    engine.reset()
    assert engine.env.game.dealer_id == 1
    engine.reset()
    assert engine.env.game.dealer_id == 0

````````

<!-- END_PY_SOURCE path="tests/test_hand_engine.py" -->
