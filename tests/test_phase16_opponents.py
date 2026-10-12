from __future__ import annotations

import random

import pytest

from holdem_ai.agents.diagnostic_opponents import BUILTIN_OPPONENTS, OpponentSpec, build_builtin
from holdem_ai.core.schema import InformationState


def obs(legal):
    return InformationState(player_id=0, features=(0., 1., 0., 3., 0., 0., 190., 190.),
                            legal_action_mask=tuple(i in legal for i in range(7)),
                            stage='preflop')


@pytest.mark.parametrize('name', BUILTIN_OPPONENTS)
@pytest.mark.parametrize('legal', [(1,), (0, 2), (3, 4, 5, 6), (0, 1, 2, 3, 4, 5, 6)])
def test_opponents_choose_legal_actions(name, legal):
    policy = build_builtin(name)
    observation = obs(legal)
    for i in range(3):
        assert policy.act(observation, random.Random(i)) in legal


def test_behavior_semantics():
    assert build_builtin('calling_station').act(obs((0, 2, 6)), random.Random()) == 2
    assert build_builtin('tight').act(obs((0, 2, 6)), random.Random()) == 0
    assert build_builtin('pot_pressure').act(obs((0, 2, 4, 6)), random.Random()) == 4
    assert build_builtin('calling_station').act(obs((0, 1)), random.Random()) == 1


@pytest.mark.parametrize('name', ['bad name', '', 'random/one', 'unknown', 'a=b'])
def test_invalid_builtins(name):
    with pytest.raises(ValueError):
        OpponentSpec(name)


def test_checkpoint_name_allowed():
    assert OpponentSpec('phase14-ppo', 'checkpoint.pt').checkpoint == 'checkpoint.pt'


def test_factory_rejects_unknown():
    with pytest.raises(ValueError):
        build_builtin('missing')
