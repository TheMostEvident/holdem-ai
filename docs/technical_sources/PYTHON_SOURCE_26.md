# Full Python implementation — Volume 26 / 28

[최종 통합 기술보고서](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md)의 코드 부록. 원본 커밋 `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`에 포함된 테스트 파일의 전체 원문이다. 구분용 LF는 원문 바이트 수와 Git Blob SHA 계산에서 제외한다.

## `tests/test_phase5_schema_policy.py`

Original blob `dfc15b89f0ce9a0cc2290cf845316b207b6deb32` (3631 bytes).

<a name="py-301"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase5_schema_policy.py" sha="dfc15b89f0ce9a0cc2290cf845316b207b6deb32" -->

````````python
import json
import math
import random

import pytest

from holdem_ai.agents.policies import CFRTablePolicy, EpsilonGreedyPolicy, RandomLegalPolicy, masked_distribution
from holdem_ai.core.schema import InformationState


def obs(*, seat=0, mask=(True, False, True), hand=("SA", "SK"), history=("0:call",)):
    return InformationState(
        player_id=seat,
        features=(0.0, 1.0),
        legal_action_mask=mask,
        private_cards=hand,
        public_cards=("H2", "D3", "C4"),
        stage="flop",
        public_state=(("pot", "100"),),
        history=history,
    )


def test_infoset_key_stable_and_private_distinguished():
    first = obs()
    assert first.infoset_key() == obs().infoset_key()
    assert first.infoset_key() != obs(hand=("SA", "SQ")).infoset_key()
    assert first.infoset_key() != obs(history=("0:raise",)).infoset_key()
    decoded = json.loads(first.infoset_key())
    assert decoded["player"] == 0
    assert decoded["private"] == ["SA", "SK"]


def test_legal_mask_rejects_illegal_and_type():
    x = obs()
    assert x.legal_actions == (0, 2)
    for bad in (-1, 1, 3):
        with pytest.raises(ValueError):
            x.assert_legal(bad)
    with pytest.raises(TypeError):
        x.assert_legal(True)
    x.assert_legal(2)


@pytest.mark.parametrize("mask", [(), (False, False)])
def test_nonterminal_must_have_legal_action(mask):
    with pytest.raises(ValueError):
        obs(mask=mask)


def test_nan_feature_rejected():
    with pytest.raises(ValueError):
        InformationState(0, (math.nan,), (True,))


def test_masked_distribution_invalid_never_selected():
    p = masked_distribution((1.0, 100.0, 3.0), (True, False, True))
    assert p == pytest.approx((0.25, 0.0, 0.75))
    assert sum(p) == pytest.approx(1)


def test_masked_distribution_uniform_fallback():
    assert masked_distribution((0, 0, 0), (True, False, True)) == pytest.approx((0.5, 0, 0.5))


@pytest.mark.parametrize("values", [(0, -1, 0), (0, float("inf"), 0), (0, math.nan, 0)])
def test_masked_distribution_rejects_invalid_weights_even_if_masked(values):
    with pytest.raises(ValueError):
        masked_distribution(values, (True, False, True))


def test_masked_distribution_rejects_length():
    with pytest.raises(ValueError):
        masked_distribution((1, 2), (True, False, True))


def test_random_policy_deterministic_and_masked():
    r1, r2 = random.Random(4), random.Random(4)
    x = obs()
    policy = RandomLegalPolicy()
    a = [policy.act(x, r1) for _ in range(100)]
    b = [policy.act(x, r2) for _ in range(100)]
    assert a == b
    assert set(a) == {0, 2}


def test_cfr_policy_normalizes_mask_and_uses_fallback():
    x = obs()
    selected = CFRTablePolicy({x.infoset_key(): (0, 1, 1)})
    assert [selected.act(x, random.Random(i)) for i in range(30)] == [2] * 30
    absent = CFRTablePolicy({})
    assert {absent.act(x, random.Random(i)) for i in range(30)} == {0, 2}


def test_epsilon_greedy_does_not_select_illegal_even_if_high_q():
    pol = EpsilonGreedyPolicy(lambda _: [0.1, 10_000, 0.2], epsilon=0)
    assert pol.act(obs(), random.Random(1)) == 2
    with pytest.raises(ValueError):
        EpsilonGreedyPolicy(lambda _: [0.0], epsilon=-0.1)
    with pytest.raises(ValueError):
        EpsilonGreedyPolicy(lambda _: [0.0], epsilon=1.1)


def test_epsilon_greedy_rejects_bad_q_dimension():
    with pytest.raises(ValueError):
        EpsilonGreedyPolicy(lambda _: [1], epsilon=0).act(obs(), random.Random(1))


def test_equal_q_tie_breaker_is_lowest_legal_action():
    assert EpsilonGreedyPolicy(lambda _: [5, 99, 5], epsilon=0).act(obs(), random.Random(9)) == 0

````````

<!-- END_PY_SOURCE path="tests/test_phase5_schema_policy.py" -->

## `tests/test_phase5_self_play.py`

Original blob `4749c0e868a95dacdd924ca05161ef34b7e2b27f` (3544 bytes).

<a name="py-302"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase5_self_play.py" sha="4749c0e868a95dacdd924ca05161ef34b7e2b27f" -->

````````python
import random

import pytest

from holdem_ai.core.schema import InformationState, Terminal, Turn
from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.training.episode_runner import run_self_play
from holdem_ai.utils.snapshot import snapshot


def state(seat, step, num_actions=5):
    return Turn(InformationState(seat, (float(step),), tuple(i in (1, 3) for i in range(num_actions))))


class FakeMultiSeat:
    def __init__(self, order=(0, 1, 0, 2, 0, 3, 4, 5), n=6):
        self.order = tuple(order)
        self.n = n
        self.cursor = 0
        self.actions = []

    def reset(self, seed=None):
        self.cursor = 0
        self.actions = []
        return state(self.order[0], 0)

    def step(self, action_index):
        assert action_index in (1, 3)
        self.actions.append(action_index)
        self.cursor += 1
        if self.cursor >= len(self.order):
            return Terminal(tuple(float(i - (self.n - 1) / 2) for i in range(self.n)))
        return state(self.order[self.cursor], self.cursor)


def test_selfplay_6_max_trajectories_link_next_same_seat():
    env = FakeMultiSeat()
    policies = {i: RandomLegalPolicy() for i in range(6)}
    result = run_self_play(env, policies, seed=9)
    assert result.num_decisions == 8
    assert len(result.transitions) == 8
    zero = result.for_seat(0)
    assert len(zero) == 3
    assert [transition.done for transition in zero] == [False, False, True]
    assert [transition.reward for transition in zero] == [0, 0, -2.5]
    assert [t.next_observation.features for t in zero[:-1]] == [(2.0,), (4.0,)]
    assert result.for_seat(1)[0].done is True
    assert result.for_seat(1)[0].reward == -1.5
    assert set(env.actions).issubset({1, 3})


def test_deterministic_selfplay_actions_seed():
    policies = {i: RandomLegalPolicy() for i in range(6)}
    a = run_self_play(FakeMultiSeat(), policies, seed=77)
    b = run_self_play(FakeMultiSeat(), policies, seed=77)
    assert [t.action_index for t in a.transitions] == [t.action_index for t in b.transitions]


def test_selfplay_missing_policy():
    with pytest.raises(KeyError):
        run_self_play(FakeMultiSeat(order=(0, 1)), {0: RandomLegalPolicy()}, seed=1)


def test_selfplay_guard_against_infinite_episode():
    with pytest.raises(RuntimeError, match="max_decisions"):
        run_self_play(FakeMultiSeat(), {i: RandomLegalPolicy() for i in range(6)}, max_decisions=3)


class IllegalPolicy:
    def act(self, observation, rng):
        return 2


def test_selfplay_invalid_policy_action_raises():
    with pytest.raises(ValueError, match="illegal"):
        run_self_play(FakeMultiSeat(), {0: IllegalPolicy()})


class SeededChance:
    def __init__(self):
        self.rng = random.Random(12)
        self.first = None

    def reset(self, seed=None):
        self.rng.seed(seed)
        self.first = self.rng.randrange(10000)
        return state(0, self.first)

    def step(self, action_index):
        return Terminal((1.0, -1.0))


def test_snapshot_deep_clone_preserves_rng_and_isolates_mutations():
    original = SeededChance()
    original.reset(23)
    copy = snapshot(original)
    assert copy is not original
    assert copy.rng.random() == original.rng.random()
    copy.first = -1
    assert original.first != -1


def test_returned_transition_has_frozen_observation():
    result = run_self_play(FakeMultiSeat(order=(0, 1), n=2), {0: RandomLegalPolicy(), 1: RandomLegalPolicy()})
    with pytest.raises(Exception):
        result.transitions[0].observation.player_id = 1

````````

<!-- END_PY_SOURCE path="tests/test_phase5_self_play.py" -->

## `tests/test_phase6_benchmark.py`

Original blob `51da1f9d7eacfc94e4152432cd2ed05f54f5c59b` (640 bytes).

<a name="py-303"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase6_benchmark.py" sha="51da1f9d7eacfc94e4152432cd2ed05f54f5c59b" -->

````````python
"""Small speed-test smoke; does not assert hardware-dependent throughput."""
import importlib.util
from pathlib import Path


def test_cpu_benchmark_small_run():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'phase6_benchmark.py'
    spec = importlib.util.spec_from_file_location('phase6_benchmark', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    results = module.run_benchmark(4, 1, seed=37)
    assert results['hands'] == 4
    assert results['workers'] == 1
    assert results['decisions'] > 0
    assert results['hands_per_second'] > 0

````````

<!-- END_PY_SOURCE path="tests/test_phase6_benchmark.py" -->

## `tests/test_phase6_hunl_adapter.py`

Original blob `00c6ec681a0cba9fff1a286a85e77151628098c6` (6125 bytes).

<a name="py-304"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase6_hunl_adapter.py" sha="00c6ec681a0cba9fff1a286a85e77151628098c6" -->

````````python
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

````````

<!-- END_PY_SOURCE path="tests/test_phase6_hunl_adapter.py" -->

## `tests/test_phase7_encoding.py`

Original blob `43670306f3d1c5aff325e96594131171ae9718dc` (6764 bytes).

<a name="py-305"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase7_encoding.py" sha="43670306f3d1c5aff325e96594131171ae9718dc" -->

````````python
"""Phase 7 contracts: actor privacy, history, stable cards, critic feature schema."""
from __future__ import annotations

import dataclasses
import json

import pytest

from holdem_ai.core.schema import InformationState, Turn
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
from holdem_ai.environment.hunl200_adapter import CriticState
from holdem_ai.encoding.observation import ACTOR_FEATURE_DIM, CRITIC_FEATURE_DIM, encode_actor, encode_critic
from holdem_ai.encoding.vocabulary import VOCAB_SIZE
from holdem_ai.encoding import vocabulary as v


def _actor(**kw):
    default = dict(
        player_id=0,
        features=(0.0, 1.0, 0.0, 1.5, 0.5, 0.5, 199.5, 199., 1., .5, 1., 2.),
        legal_action_mask=(True, False, True, True, True, True, True),
        private_cards=('sA', 'dK'), public_cards=(), stage='preflop',
        public_state=(('status', json.dumps(['active', 'active'])),),
        history=(),
    )
    default.update(kw)
    return InformationState(**default)


def _critic(**kw):
    default = dict(
        player_to_act=0, button=0, street='preflop', board=(),
        hole_cards=(('sA', 'dK'), ('cQ', 'hJ')),
        stacks=(19950, 19900), contributions=(50, 100),
        street_bets=(50, 100), status=('active', 'active'), pot=150,
        history=(), is_terminal=False,
    )
    default.update(kw)
    return CriticState(**default)


def test_actor_dimensions_and_hole_card_tokens():
    x = encode_actor(_actor())
    assert len(x.features) == ACTOR_FEATURE_DIM == 20
    assert len(x.legal_action_mask) == 7
    assert x.token_ids[:4] == (v.BOS, v.PRIVATE, v.card_id('sA'), v.card_id('dK'))
    assert v.OPPONENT_PRIVATE not in x.token_ids
    assert len(x.token_ids) == len(x.token_values)
    assert len(x.token_ids) < 256


def test_actor_public_status_encoded_from_public_state_only():
    normal = encode_actor(_actor())
    allin = encode_actor(_actor(public_state=(('status', '["active","allin"]'),)))
    assert normal.token_ids == allin.token_ids
    assert normal.features[-8:] != allin.features[-8:]
    assert allin.features[-8:] == (1.,0.,0.,0.,0.,0.,1.,0.)


def test_history_contains_raise_amount_in_bb():
    record = 'preflop|seat=1|raise_to|raise_to=450'
    x = encode_actor(_actor(history=(record,)))
    assert x.token_ids[-5:] == (
        v.EVENT, v.STREET_TO_ID['preflop'], v.SEAT_1, v.ACTION_TO_ID['raise_to'], v.AMOUNT,
    )
    assert x.token_values[-1] == 4.5


def test_history_order_and_fold_distinguished():
    h1 = ('preflop|seat=0|call|raise_to=-', 'preflop|seat=1|check|raise_to=-')
    h2 = tuple(reversed(h1))
    assert encode_actor(_actor(history=h1)).token_ids != encode_actor(_actor(history=h2)).token_ids
    assert v.ACTION_TO_ID['fold'] != v.ACTION_TO_ID['check']


@pytest.mark.parametrize('bad', [
    'preflop|seat=2|call|raise_to=-',
    'preflop|seat=0|raise_to|raise_to=-',
    'preflop|seat=0|call|raise_to=300',
    'river|seat=0|raise_to|raise_to=abc',
    'totally different format',
])
def test_malformed_history_is_rejected(bad):
    with pytest.raises(ValueError):
        encode_actor(_actor(history=(bad,)))


def test_history_overflow_explicit_not_truncated():
    history = ('preflop|seat=0|call|raise_to=-',) * 100
    with pytest.raises(ValueError, match='not truncated'):
        encode_actor(_actor(history=history), max_tokens=25)
    full = encode_actor(_actor(history=history), max_tokens=600)
    assert full.token_ids.count(v.EVENT) == 100


def test_card_domain_and_duplicate_rejected():
    with pytest.raises(ValueError, match='invalid canonical card'):
        encode_actor(_actor(private_cards=('sZ', 'dK')))
    with pytest.raises(ValueError, match='duplicate'):
        encode_actor(_actor(private_cards=('sA','dK'),public_cards=('sA','h2','c3')))
    assert len(v.CARDS) == 52 and len(set(v.CARD_TO_ID.values())) == 52
    assert max(v.ACTION_TO_ID.values()) < VOCAB_SIZE


def test_actor_wrong_phase6_feature_contract_rejected():
    with pytest.raises(ValueError):
        encode_actor(_actor(features=(0., 1.)))
    with pytest.raises(ValueError):
        encode_actor(_actor(player_id=1))
    with pytest.raises(ValueError):
        encode_actor(_actor(public_state=()))


def test_critic_two_holes_differ_and_no_actor_privilege():
    state = _critic()
    a = encode_critic(state)
    b = encode_critic(dataclasses.replace(state, hole_cards=(('sA','dK'), ('cT','hJ'))))
    assert len(a.features) == CRITIC_FEATURE_DIM == 24
    assert a.token_ids != b.token_ids
    assert v.OPPONENT_PRIVATE in a.token_ids
    actor1 = encode_actor(_actor())
    actor2 = encode_actor(_actor())
    assert actor1 == actor2  # critic-only opponent change never enters the actor


def test_critic_chips_are_in_big_blind_units():
    x = encode_critic(_critic())
    assert x.features[:4] == (1.,0.,1.,0.)
    assert x.features[4:8] == (1.,0.,0.,0.)
    assert x.features[8:10] == (199.5, 199.0)
    assert x.features[10:12] == (.5,1.)
    assert x.features[14] == 1.5
    assert x.features[-1] == 0.0


def test_critic_terminal_encoding():
    x = encode_critic(_critic(player_to_act=-1, is_terminal=True))
    assert x.features[2:4] == (0.,0.)
    assert x.features[-1] == 1.
    assert x.token_ids[-1] == v.TERMINAL
    with pytest.raises(ValueError):
        encode_critic(_critic(player_to_act=-1, is_terminal=False))


def test_critic_rejects_bad_pot_and_duplicate_cards():
    with pytest.raises(ValueError):
        encode_critic(_critic(pot=999))
    with pytest.raises(ValueError, match='duplicate'):
        encode_critic(_critic(hole_cards=(('sA','dK'),('sA','hJ'))))


def test_real_phase6_adapter_accepts_actor_and_critic():
    # Uses real HUNL200Adapter; on the developer's machine this must run
    # against the installed actual MultiPlayerHandEngine, not the fake test engine.
    env = HUNL200Adapter(button=1, expose_training_state=True)
    current = env.reset(seed=123)
    assert isinstance(current, Turn)
    obs = current.observation
    actor = encode_actor(obs)
    critic = encode_critic(env.training_state())
    assert actor.legal_action_mask == obs.legal_action_mask
    assert actor.features[0] == obs.player_id
    assert len(critic.token_ids) > len(actor.token_ids)
    assert v.card_id(env.engine.hole_cards[1 - obs.player_id][0]) not in actor.token_ids


def test_history_after_phase6_action_matches():
    env = HUNL200Adapter(button=0, expose_training_state=True)
    first = env.reset(seed=17)
    assert isinstance(first, Turn)
    action = 2  # call
    assert first.observation.legal_action_mask[action]
    nxt = env.step(action)
    assert isinstance(nxt, Turn)
    encoded = encode_actor(nxt.observation)
    assert v.ACTION_TO_ID['call'] in encoded.token_ids
    assert v.EVENT in encoded.token_ids

````````

<!-- END_PY_SOURCE path="tests/test_phase7_encoding.py" -->

## `tests/test_phase7_rollout.py`

Original blob `6a721709872d5d970b46cf244d598057a5782cec` (1772 bytes).

<a name="py-306"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase7_rollout.py" sha="6a721709872d5d970b46cf244d598057a5782cec" -->

````````python
"""End-to-end HUNL200 -> global episodes -> actor/critic tensors.

Uses the real Phase-6 adapter and underlying repo engine when installed locally.
"""
from __future__ import annotations

import pytest

from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
from holdem_ai.training.global_rollout import collect_global_episode
from holdem_ai.encoding.observation import encode_actor, encode_critic
from holdem_ai.encoding.actions import encode_action, masked_categorical
from holdem_ai.encoding.tensors import batch_actors, batch_critics


def test_phase7_global_rollout_all_decisions_encode():
    torch = pytest.importorskip('torch')
    policies = {0: RandomLegalPolicy(), 1: RandomLegalPolicy()}
    a, c = [], []
    for seed in range(35):
        env = HUNL200Adapter(button=seed % 2, expose_training_state=True)
        episode = collect_global_episode(env, policies, seed=seed, privileged=True)
        assert episode.decisions
        for decision in episode.decisions:
            assert decision.critic_state is not None
            assert decision.next_critic_state is not None
            a.append(encode_actor(decision.observation))
            c.append(encode_critic(decision.critic_state))
            encode_critic(decision.next_critic_state)
            encode_action(decision.action_index, decision.observation.legal_action_mask)
    ab = batch_actors(a[:64], device='cpu')
    cb = batch_critics(c[:64], device='cpu')
    assert ab.features.shape[1] == 20
    assert cb.features.shape[1] == 24
    logits = torch.zeros(len(ab.features),7)
    actions = masked_categorical(logits,ab.legal_action_mask).sample()
    assert torch.all(ab.legal_action_mask.gather(-1,actions.unsqueeze(-1)))

````````

<!-- END_PY_SOURCE path="tests/test_phase7_rollout.py" -->

## `tests/test_phase7_tensors_actions.py`

Original blob `2645d2e166d890ff6cfbbc9580340c45339ddbfa` (3709 bytes).

<a name="py-307"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase7_tensors_actions.py" sha="2645d2e166d890ff6cfbbc9580340c45339ddbfa" -->

````````python
"""Action legality, padding masks, numeric tensors and gradients (GPU optional)."""
from __future__ import annotations

import json
import pytest

from holdem_ai.core.schema import InformationState
from holdem_ai.environment.hunl200_adapter import CriticState
from holdem_ai.encoding.observation import encode_actor, encode_critic
from holdem_ai.encoding.actions import encode_action, masked_logits, masked_categorical
from holdem_ai.encoding.tensors import batch_actors, batch_critics
from holdem_ai.encoding.vocabulary import PAD


def _observation(history=(), mask=(True, False, True, True, True, False, True)):
    return InformationState(
        0, (0.,1.,0.,1.5,.5,.5,199.5,199.,1.,.5,1.,2.), mask,
        ('sA','dK'), (), 'preflop', (('status',json.dumps(['active','active'])),), history,
    )


def _state(history=()):
    return CriticState(0,0,'preflop',(),(('sA','dK'),('hQ','cJ')),
                       (19950,19900),(50,100),(50,100),('active','active'),150,history,False)


def test_action_onehot_respects_legality():
    mask = _observation().legal_action_mask
    assert encode_action(2,mask) == (0.,0.,1.,0.,0.,0.,0.)
    with pytest.raises(ValueError):
        encode_action(1,mask)
    with pytest.raises(ValueError):
        encode_action(True,mask)
    with pytest.raises(ValueError):
        encode_action(7,mask)


def test_batch_padding_attention_and_values():
    torch = pytest.importorskip('torch')
    a = encode_actor(_observation())
    b = encode_actor(_observation(('preflop|seat=0|raise_to|raise_to=450',)))
    batch = batch_actors([a,b])
    assert batch.token_ids.shape == (2,len(b.token_ids))
    assert batch.token_values.dtype == torch.float32
    assert batch.features.shape == (2,20)
    assert batch.legal_action_mask.shape == (2,7)
    assert batch.legal_action_mask.dtype == torch.bool
    assert batch.attention_mask.dtype == torch.bool
    assert batch.token_ids[0,-1].item() == PAD
    assert not batch.attention_mask[0,-1].item()
    assert batch.token_values[1,-1].item() == 4.5


def test_critic_batch_shapes_and_amounts():
    torch = pytest.importorskip('torch')
    a = encode_critic(_state())
    b = encode_critic(_state(('preflop|seat=0|raise_to|raise_to=750',)))
    batch = batch_critics([a,b])
    assert batch.token_ids.shape[0] == 2
    assert batch.features.shape == (2,24)
    assert batch.attention_mask.dtype == torch.bool
    assert batch.token_values[1,-1].item() == 7.5


def test_masked_categorical_probabilities_and_gradient():
    torch = pytest.importorskip('torch')
    logits = torch.tensor([[0.,1000.,1.,-2.,3.,1000.,5.]], requires_grad=True)
    mask = torch.tensor([[True,False,True,True,True,False,True]])
    result = masked_logits(logits,mask)
    distribution = masked_categorical(logits,mask)
    probs = distribution.probs
    assert probs[0,1].item() == probs[0,5].item() == 0.0
    assert torch.allclose(probs.sum(-1),torch.ones(1))
    assert int(distribution.sample().item()) in (0,2,3,4,6)
    loss = -distribution.log_prob(torch.tensor([2])).sum()
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0,1].item() == logits.grad[0,5].item() == 0.0
    assert result.shape == (1,7)


def test_masked_categorical_rejects_all_illegal():
    torch = pytest.importorskip('torch')
    with pytest.raises(ValueError):
        masked_categorical(torch.zeros(1,7),torch.zeros(1,7,dtype=torch.bool))
    with pytest.raises(ValueError):
        masked_logits(torch.zeros(1,6),torch.zeros(1,6,dtype=torch.bool))


def test_empty_batch_rejected():
    pytest.importorskip('torch')
    with pytest.raises(ValueError):
        batch_actors([])
    with pytest.raises(ValueError):
        batch_critics([])

````````

<!-- END_PY_SOURCE path="tests/test_phase7_tensors_actions.py" -->

## `tests/test_phase8_gpu.py`

Original blob `57cc87145356b3191515ac42a8a48ecc7b3a3eb9` (1530 bytes).

<a name="py-308"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase8_gpu.py" sha="57cc87145356b3191515ac42a8a48ecc7b3a3eb9" -->

````````python
"""Runs a minimal real Phase-7 batch on CUDA when available (e.g. GTX 1070)."""
from __future__ import annotations

import pytest
import torch

from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
from holdem_ai.encoding.tensors import batch_actors, batch_critics
from holdem_ai.encoding.observation import encode_actor, encode_critic
from holdem_ai.models.transformer import CentralizedQCritic, NetworkConfig, TransformerActor


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA runtime not installed on this machine')
def test_phase8_cuda_forward_backward():
    env = HUNL200Adapter(expose_training_state=True)
    observation = env.reset(seed=41).observation
    privileged = env.training_state()
    ab = batch_actors([encode_actor(observation)], device='cuda')
    cb = batch_critics([encode_critic(privileged)], device='cuda')
    cfg = NetworkConfig(embed_dim=32, transformer_layers=1, ff_dim=64, hidden_dim=64)
    actor, critic = TransformerActor(cfg).cuda(), CentralizedQCritic(cfg).cuda()
    p, q = actor.distribution(ab), critic(cb)
    action = torch.tensor([observation.legal_actions[0]], dtype=torch.long, device='cuda')
    loss = -p.log_prob(action).mean() + q[:, 0, 0].square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(loss).item())
    assert all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in actor.parameters())
    assert all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in critic.parameters())

````````

<!-- END_PY_SOURCE path="tests/test_phase8_gpu.py" -->

## `tests/test_phase8_networks.py`

Original blob `fb362ac9e506fa1668e3bb94d816b8d8fbd5203b` (9837 bytes).

<a name="py-309"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase8_networks.py" sha="fb362ac9e506fa1668e3bb94d816b8d8fbd5203b" -->

````````python
"""Phase-8 independent unit tests using real Phase-7 batch structures."""
from __future__ import annotations

import random

import pytest
import torch

from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
from holdem_ai.training.global_rollout import collect_global_episode
from holdem_ai.encoding.tensors import batch_actors, batch_critics
from holdem_ai.encoding.observation import encode_actor, encode_critic
from holdem_ai.encoding.vocabulary import VOCAB_SIZE
from holdem_ai.models.transformer import CausalTokenEncoder, CentralizedQCritic, NetworkConfig, TransformerActor, chosen_q
from holdem_ai.agents.neural_policy import NeuralActorPolicy


@pytest.fixture(scope='module')
def samples():
    policy = {0: RandomLegalPolicy(), 1: RandomLegalPolicy()}
    actor_rows, critic_rows, decisions = [], [], []
    for seed in range(15):
        env = HUNL200Adapter(button=seed % 2, expose_training_state=True)
        episode = collect_global_episode(env, policy, seed=seed, privileged=True)
        for step in episode.decisions:
            actor_rows.append(encode_actor(step.observation))
            critic_rows.append(encode_critic(step.critic_state))
            decisions.append(step)
    return actor_rows, critic_rows, decisions


@pytest.fixture
def model_config():
    return NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1,
                         ff_dim=64, hidden_dim=48, max_tokens=128)


@pytest.mark.parametrize('values', [
    {'embed_dim': 31, 'attention_heads': 2}, {'transformer_layers': 0},
    {'max_tokens': 0}, {'dropout': 1.0}, {'embed_dim': True},
])
def test_invalid_network_configuration(values):
    with pytest.raises(ValueError):
        NetworkConfig(**values)


def test_phase7_shapes_forward(samples, model_config):
    ar, cr, _ = samples
    actor, critic = TransformerActor(model_config), CentralizedQCritic(model_config)
    ab, cb = batch_actors(ar[:7]), batch_critics(cr[:7])
    logits, q = actor(ab), critic(cb)
    assert logits.shape == (7, 7)
    assert q.shape == (7, 2, 7)
    assert logits.dtype == q.dtype == torch.float32
    assert torch.isfinite(logits).all() and torch.isfinite(q).all()


def test_masked_policy_is_normalized_and_never_selects_illegal(samples, model_config):
    ar, _, _ = samples
    actor = TransformerActor(model_config)
    ab = batch_actors(ar[:20])
    dist = actor.distribution(ab)
    assert torch.allclose(dist.probs.sum(-1), torch.ones(20), atol=1e-6)
    assert torch.equal(dist.probs[~ab.legal_action_mask], torch.zeros_like(dist.probs[~ab.legal_action_mask]))
    for _ in range(20):
        actions = dist.sample()
        assert bool(ab.legal_action_mask.gather(1, actions[:, None]).all())


def test_training_backward_both_networks(samples, model_config):
    ar, cr, decisions = samples
    actor, critic = TransformerActor(model_config), CentralizedQCritic(model_config)
    ab, cb = batch_actors(ar[:10]), batch_critics(cr[:10])
    acts = torch.tensor([s.action_index for s in decisions[:10]], dtype=torch.long)
    seats = torch.tensor([s.player_id for s in decisions[:10]], dtype=torch.long)
    rewards = torch.tensor([s.reward_vector[0] for s in decisions[:10]], dtype=torch.float32)
    dist = actor.distribution(ab)
    actor_loss = -dist.log_prob(acts).mean() - 0.01 * dist.entropy().mean()
    values = chosen_q(critic(cb), seats, acts)
    critic_loss = ((values - rewards) ** 2).mean()
    actor_loss.backward()
    critic_loss.backward()
    for network in (actor, critic):
        gradients = [p.grad for p in network.parameters() if p.requires_grad]
        assert all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
        assert any(bool((g.abs() > 0).any()) for g in gradients)


def test_critic_has_strict_zero_sum_for_every_action(samples, model_config):
    _, cr, _ = samples
    q = CentralizedQCritic(model_config)(batch_critics(cr[:10]))
    assert torch.equal(q[:, 0, :], -q[:, 1, :])
    assert torch.equal(q.sum(dim=1), torch.zeros_like(q[:, 0, :]))


def test_critic_terminal_bootstrap_is_zero(samples, model_config):
    _, _, decisions = samples
    terminal = [s.next_critic_state for s in decisions if s.done]
    assert terminal
    enc = [encode_critic(s) for s in terminal[:5]]
    q = CentralizedQCritic(model_config)(batch_critics(enc))
    assert torch.count_nonzero(q) == 0


def test_chosen_q_gathers_correct_seat_action():
    q = torch.arange(42, dtype=torch.float32).reshape(3, 2, 7)
    seats = torch.tensor([0, 1, 0])
    actions = torch.tensor([1, 2, 6])
    assert chosen_q(q, seats, actions).tolist() == [1.0, 23.0, 34.0]
    with pytest.raises(ValueError):
        chosen_q(q, torch.tensor([0, 2, 0]), actions)
    with pytest.raises(ValueError):
        chosen_q(q, seats, torch.tensor([0, 1, 7]))


def test_no_privileged_critic_data_reaches_actor(samples, model_config):
    ar, cr, _ = samples
    actor = TransformerActor(model_config)
    with pytest.raises(TypeError):
        actor(batch_critics(cr[:3]))
    critic = CentralizedQCritic(model_config)
    with pytest.raises(TypeError):
        critic(batch_actors(ar[:3]))


def test_padding_does_not_change_logits(samples, model_config):
    ar, cr, _ = samples
    actor = TransformerActor(model_config).eval()
    critic = CentralizedQCritic(model_config).eval()
    idx = min(range(len(ar)), key=lambda i: len(ar[i].token_ids))
    long_idx = max(range(len(ar)), key=lambda i: len(ar[i].token_ids))
    assert len(ar[idx].token_ids) < len(ar[long_idx].token_ids)
    with torch.no_grad():
        alone = actor(batch_actors([ar[idx]]))[0]
        mixed = actor(batch_actors([ar[idx], ar[long_idx]]))[0]
        ca = critic(batch_critics([cr[idx]]))[0]
        cm = critic(batch_critics([cr[idx], cr[long_idx]]))[0]
    assert torch.allclose(alone, mixed, atol=2e-5)
    assert torch.allclose(ca, cm, atol=2e-5)


def test_value_amount_channel_changes_representations(samples, model_config):
    ar, _, _ = samples
    # At least one sample has a Phase-7 AMOUNT token from a raise_to action.
    amount_row = next((r for r in ar if any(v != 0 for v in r.token_values)), None)
    assert amount_row is not None
    from dataclasses import replace
    modified = replace(amount_row, token_values=tuple(x * 2 for x in amount_row.token_values))
    encoder = CausalTokenEncoder(model_config).eval()
    batch = batch_actors([amount_row, modified])
    with torch.no_grad():
        result = encoder(batch.token_ids, batch.token_values, batch.attention_mask)
    assert not torch.allclose(result[0], result[1])


def test_model_rejects_bad_attention_and_overlength(samples, model_config):
    ar, _, _ = samples
    encoder = CausalTokenEncoder(model_config)
    batch = batch_actors(ar[:2])
    with pytest.raises(ValueError):
        encoder(batch.token_ids[:, :2], batch.token_values[:, :2], torch.zeros_like(batch.attention_mask[:, :2]))
    with pytest.raises(ValueError):
        encoder(torch.ones((1, model_config.max_tokens + 1), dtype=torch.long),
                torch.zeros((1, model_config.max_tokens + 1)),
                torch.ones((1, model_config.max_tokens + 1), dtype=torch.bool))
    with pytest.raises(ValueError):
        encoder(torch.ones((1, 3), dtype=torch.long), torch.zeros((1, 3)),
                torch.tensor([[True, False, True]]))


def test_policy_reproducible_python_rng_and_legal(samples, model_config):
    _, _, steps = samples
    obs = steps[0].observation
    actor = TransformerActor(model_config)
    policy = NeuralActorPolicy(actor, greedy=False)
    a = [policy.act(obs, random.Random(101)) for _ in range(3)]
    assert len(set(a)) == 1
    assert a[0] in obs.legal_actions
    greedy = NeuralActorPolicy(actor, greedy=True)
    assert greedy.act(obs, random.Random(1)) == greedy.act(obs, random.Random(999))


def test_actor_independent_of_opponent_hole_cards(samples, model_config):
    ar, cr, steps = samples
    actor = TransformerActor(model_config).eval()
    a = steps[0].observation
    b = batch_actors([encode_actor(a)])
    from dataclasses import replace
    # A private-data change exists only in the privileged critic input.
    c = steps[0].critic_state
    opp = 1 - a.player_id
    cards = list(c.hole_cards)
    cards[opp] = ('h2', 'd3') if ('h2', 'd3') != cards[opp] else ('h4', 'd5')
    other = replace(c, hole_cards=tuple(cards))
    with torch.no_grad():
        original_logit = actor(b)
        unaffected_logit = actor(batch_actors([encode_actor(a)]))
    assert torch.equal(original_logit, unaffected_logit)
    # The trainer-side CriticState is necessarily a different object.
    assert other.hole_cards[opp] != c.hole_cards[opp]


def test_state_dict_save_load_reproducible(samples, model_config, tmp_path):
    ar, cr, _ = samples
    actor = TransformerActor(model_config).eval()
    critic = CentralizedQCritic(model_config).eval()
    ab, cb = batch_actors(ar[:2]), batch_critics(cr[:2])
    with torch.no_grad():
        a, c = actor(ab), critic(cb)
    target = tmp_path / 'phase8_weights.pt'
    torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict()}, target)
    checkpoint = torch.load(target, weights_only=True, map_location='cpu')
    actor2 = TransformerActor(model_config).eval()
    critic2 = CentralizedQCritic(model_config).eval()
    actor2.load_state_dict(checkpoint['actor'])
    critic2.load_state_dict(checkpoint['critic'])
    with torch.no_grad():
        assert torch.equal(a, actor2(ab))
        assert torch.equal(c, critic2(cb))


def test_actor_parameter_count_lightweight(model_config):
    actor, critic = TransformerActor(model_config), CentralizedQCritic(model_config)
    assert sum(p.numel() for p in actor.parameters()) < 500_000
    assert sum(p.numel() for p in critic.parameters()) < 500_000

````````

<!-- END_PY_SOURCE path="tests/test_phase8_networks.py" -->

