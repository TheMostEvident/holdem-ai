import pytest

from holdem_ai.environment.rlcard_adapter import RLCardTurnDriver, observe_rlcard_state
from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.training.episode_runner import run_self_play
from holdem_ai.utils.snapshot import snapshot


class FakeRawState:
    def __init__(self, seat, turn):
        self.seat = seat
        self.turn = turn
        self.state = {
            "obs": [float(turn), [2.0, 3.0]],
            "legal_actions": {0: None, 2: None, 5: None},
            "raw_obs": {
                "hand": [f"private_{seat}_1", f"private_{seat}_2"],
                "public_cards": ["SA", "DK", "CQ"],
                "stage": "FLOP",
                "all_chips": [100, 100],
                "pot": 25,
                "opponent_hand": ["LEAK", "LEAK"],
                "perfect_information": {"hole_cards_all": [["LEAK"]]},
                "deck": ["LEAK"],
            },
            "all_private_hands": [["LEAK"]],
        }


class FakeRLCard:
    def __init__(self):
        self.counter = 0
        self.actions = []
        self.used_seed = None

    def seed(self, seed):
        self.used_seed = seed

    def reset(self):
        self.counter = 0
        self.actions = []
        return FakeRawState(0, self.counter).state, 0

    def step(self, action):
        self.actions.append(action)
        self.counter += 1
        return FakeRawState(self.counter % 2, self.counter).state, self.counter % 2

    def is_over(self):
        return self.counter >= 4

    def get_payoffs(self):
        return [10, -10]


def test_observation_never_exposes_opponents_hidden_cards_or_engine_data():
    s = FakeRawState(0, 2).state
    o = observe_rlcard_state(s, player_id=0, num_actions=6)
    assert o.private_cards == ("private_0_1", "private_0_2")
    assert o.public_cards == ("SA", "DK", "CQ")
    assert o.legal_action_mask == (True, False, True, False, False, True)
    assert o.features == (2, 2, 3)
    assert "LEAK" not in o.infoset_key()
    assert "private_1" not in o.infoset_key()
    assert "deck" not in o.infoset_key()
    assert o.public_state == (("all_chips", "[100,100]"), ("pot", "25"))


def test_rlcard_adapter_seed_masks_and_selfplay():
    raw = FakeRLCard()
    driver = RLCardTurnDriver(raw, ("fold", "check", "call", "raise_half", "raise_pot", "allin"))
    result = run_self_play(driver, {0: RandomLegalPolicy(), 1: RandomLegalPolicy()}, seed=101)
    assert raw.used_seed == 101
    assert result.rewards == (10.0, -10.0)
    assert result.num_decisions == 4
    assert all(a in (0, 2, 5) for a in raw.actions)
    assert len(result.for_seat(0)) == 2
    assert len(result.for_seat(1)) == 2
    assert result.for_seat(0)[-1].reward == 10
    assert result.for_seat(1)[-1].reward == -10


def test_rlcard_adapter_snapshot_keeps_position_and_deep_state():
    driver = RLCardTurnDriver(FakeRLCard(), tuple(str(i) for i in range(6)))
    driver.reset(22)
    twin = snapshot(driver)
    original_state = driver.step(0)
    cloned_state = twin.step(0)
    assert original_state == cloned_state
    assert twin.env is not driver.env
    twin.env.counter = 999
    assert driver.env.counter == 1


def test_illegal_rlcard_action_rejected_before_engine():
    env = FakeRLCard()
    driver = RLCardTurnDriver(env, tuple(str(i) for i in range(6)))
    driver.reset()
    with pytest.raises(ValueError):
        driver.step(1)
    assert env.actions == []


@pytest.mark.parametrize("legal", [{-1: None}, {6: None}, {"fold": None}])
def test_out_of_range_and_wrong_action_ids_rejected(legal):
    s = FakeRawState(0, 0).state
    s["legal_actions"] = legal
    with pytest.raises((ValueError, TypeError)):
        observe_rlcard_state(s, 0, 6)


def test_rlcard_hidden_raw_history_does_not_leak():
    raw = FakeRawState(0, 0).state
    raw["raw_obs"]["history"] = [("opponent_hole_cards", "LEAK")]
    obs = observe_rlcard_state(raw, 0, 6)
    assert "LEAK" not in obs.infoset_key()


def test_terminal_replay_rejected_and_reward_override():
    env = FakeRLCard()
    driver = RLCardTurnDriver(env, tuple(str(i) for i in range(6)), terminal_rewards_fn=lambda e: [2.0, -2.0])
    driver.reset()
    for _ in range(4):
        result = driver.step(0)
    assert result.rewards == (2.0, -2.0)
    with pytest.raises(RuntimeError):
        driver.step(0)


def test_observation_requires_legal_actions_mapping():
    s = FakeRawState(0, 0).state
    s["legal_actions"] = [0, 2]
    with pytest.raises(TypeError):
        observe_rlcard_state(s, 0, 6)


def test_direct_rlcard_integration_if_library_installed():
    rlcard = pytest.importorskip("rlcard")
    env = rlcard.make("no-limit-holdem", config={"seed": 33})
    labels = tuple(f"action_{i}" for i in range(env.num_actions))
    driver = RLCardTurnDriver(env, labels)
    result = run_self_play(driver, {i: RandomLegalPolicy() for i in range(env.num_players)}, seed=33, max_decisions=500)
    assert len(result.rewards) == env.num_players
    assert result.num_decisions > 0
    assert all(t.action_index in t.observation.legal_actions for t in result.transitions)
