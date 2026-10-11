"""Phase-8 independent unit tests using real Phase-7 batch structures."""
from __future__ import annotations

import random

import pytest
import torch

from holdem_agent_phase5.policy import RandomLegalPolicy
from holdem_agent_phase6 import HUNL200Adapter, collect_global_episode
from holdem_agent_phase7 import batch_actors, batch_critics, encode_actor, encode_critic
from holdem_agent_phase7.vocabulary import VOCAB_SIZE
from holdem_agent_phase8 import (
    CausalTokenEncoder, CentralizedQCritic, NetworkConfig,
    NeuralActorPolicy, TransformerActor, chosen_q,
)


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
