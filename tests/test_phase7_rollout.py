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
