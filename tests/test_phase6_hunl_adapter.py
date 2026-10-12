"""Integration tests run against the actual Phase 3 engine in the repository."""
from __future__ import annotations

import json

import pytest

from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.core.schema import Terminal, Turn
from holdem_ai.training.episode_runner import run_self_play
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
from holdem_ai.training.global_rollout import collect_global_episode


def test_hunl200_blinds_and_seat_order_and_actor_privacy():
    env = HUNL200Adapter(button=1)
    turn = env.reset(seed=23)
    assert isinstance(turn, Turn)
    assert turn.player_id == 1  # Heads-up BTN/SB first preflop
    assert env.engine.contributions == [100, 50]
    assert env.engine.initial_stacks == (20000, 20000)
    obs = turn.observation
    assert len(obs.legal_action_mask) == 7
    assert len(obs.features) == 12
    assert obs.private_cards == tuple(env.engine.hole_cards[1])
    assert obs.private_cards != tuple(env.engine.hole_cards[0])
    serialized = obs.infoset_key()
    assert "hole_cards" not in serialized and "_deck" not in serialized
    assert str(env.engine.hole_cards[0]) not in serialized
    assert all(k not in ("hand", "hole_cards", "deck") for k, _ in obs.public_state)
    assert len(obs.history) == 0
    with pytest.raises(PermissionError):
        env.training_state()


def test_check_call_and_history_persists_through_streets():
    env = HUNL200Adapter(button=0)
    turn = env.reset(seed=7)
    assert isinstance(turn, Turn)
    # SB can call, then BB can check; flop is next and BB acts first.
    assert turn.observation.legal_action_mask[2]
    turn = env.step(2)
    assert isinstance(turn, Turn) and turn.player_id == 1
    assert turn.observation.legal_action_mask[1]
    turn = env.step(1)
    assert isinstance(turn, Turn) and turn.player_id == 1
    assert turn.observation.stage == "flop"
    assert turn.observation.history == (
        "preflop|seat=0|call|raise_to=-",
        "preflop|seat=1|check|raise_to=-",
    )
    assert len(turn.observation.public_cards) == 3


def test_raise_mapping_invariants_and_illegal_action_does_not_mutate():
    env = HUNL200Adapter(button=0)
    turn = env.reset(seed=11)
    assert isinstance(turn, Turn)
    initial = (tuple(env.engine.contributions), tuple(env.engine.action_history))
    with pytest.raises(ValueError):
        env.step(1)  # check is impossible when SB still owes chips
    assert initial == (tuple(env.engine.contributions), tuple(env.engine.action_history))
    # Phase 5 raise_half_pot -> exact street raise_to in Phase 3 engine.
    assert turn.observation.legal_action_mask[3]
    turn = env.step(3)
    assert isinstance(turn, Turn)
    assert env.last_engine_action.kind == "raise_to"
    assert env.engine.action_history[-1] == (0, 5)
    assert env.engine.action_raise_to[-1] == env.last_engine_action.raise_to
    assert env.engine.action_raise_to[-1] >= 200  # legal minimum preflop raise


def test_fold_terminal_chips_bb_and_reset_reproducible():
    env = HUNL200Adapter(button=0)
    env.reset(seed=31)
    result = env.step(0)
    assert isinstance(result, Terminal)
    assert result.rewards == (-0.5, 0.5)
    assert env.engine.get_result().payoffs_chips == (-50, 50)
    with pytest.raises(RuntimeError):
        env.step(0)
    a = env.reset(seed=31).observation
    b = HUNL200Adapter(button=0).reset(seed=31).observation
    assert a == b


def test_real_engine_random_200_hands_and_phase5_runner():
    for seed in range(200):
        env = HUNL200Adapter(button=seed % 2)
        episode = run_self_play(
            env, {0: RandomLegalPolicy(), 1: RandomLegalPolicy()},
            seed=seed, max_decisions=200,
        )
        assert abs(sum(episode.rewards)) < 1e-9
        assert episode.num_decisions == len(episode.transitions)
        assert all(tr.action_index in tr.observation.legal_actions for tr in episode.transitions)
        assert env.engine.is_over()
        assert sum(env.engine.get_result().final_stacks) == 40000
        assert tuple(x / 100 for x in env.engine.get_result().payoffs_chips) == episode.rewards


def test_global_collector_private_data_split_and_terminal_rewards():
    env = HUNL200Adapter(button=0, expose_training_state=True)
    episode = collect_global_episode(
        env, {0: RandomLegalPolicy(), 1: RandomLegalPolicy()},
        seed=99, privileged=True,
    )
    assert episode.decisions
    assert episode.decisions[-1].done
    assert episode.decisions[-1].reward_vector == episode.terminal_rewards
    assert all(d.reward_vector == (0.0, 0.0) for d in episode.decisions[:-1])
    for i, d in enumerate(episode.decisions):
        assert d.time_index == i
        assert d.critic_state is not None and d.next_critic_state is not None
        assert len(d.critic_state.hole_cards) == 2
        assert d.observation.private_cards == d.critic_state.hole_cards[d.player_id]
        assert d.observation.private_cards != d.critic_state.hole_cards[1-d.player_id]
        assert d.observation.stage == d.critic_state.street
        if i + 1 < len(episode.decisions):
            assert d.next_player_id == episode.decisions[i+1].player_id
        else:
            assert d.next_player_id is None
        assert len(d.observation.history) == i
    assert abs(sum(episode.terminal_rewards)) < 1e-9


def test_global_collector_no_privileged_data_by_default():
    env = HUNL200Adapter()
    result = collect_global_episode(
        env, {0: RandomLegalPolicy(), 1: RandomLegalPolicy()}, seed=3,
    )
    assert all(d.critic_state is None and d.next_critic_state is None for d in result.decisions)
    with pytest.raises(PermissionError):
        collect_global_episode(env, {0: RandomLegalPolicy(), 1: RandomLegalPolicy()}, privileged=True)


def test_observation_serialization_is_public():
    env = HUNL200Adapter()
    turn = env.reset(seed=41)
    assert isinstance(turn, Turn)
    state = dict(turn.observation.public_state)
    stacks = json.loads(state["stacks"])
    assert stacks == [19950, 19900]
    assert not any(key in state for key in ("hand", "hole_cards", "deck", "burned"))
