# Python source volume 25A/28

[Canonical technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Immutable commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`.

## tests/test_phase16_cli.py

Git Blob `d049328372ddbfa85eaa10dd45ff1e76e8d643df` (2192 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_phase16_cli.py" sha="d049328372ddbfa85eaa10dd45ff1e76e8d643df" -->

````````python
from __future__ import annotations

import json

import pytest
import torch

from holdem_ai.models.transformer import NetworkConfig, TransformerActor
from scripts.phase16_evaluate import parser, run, _specs


def checkpoint(path):
    network = NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1,
                            ff_dim=64, hidden_dim=48, max_tokens=256)
    model = TransformerActor(network)
    from dataclasses import asdict
    torch.save({'network_config': asdict(network), 'actor_state': model.state_dict()}, path)


def test_cli_smoke_creates_json_csv_without_training(tmp_path):
    model_path = tmp_path / 'phase14_best.pt'
    checkpoint(model_path)
    json_path, csv_path, md_path = tmp_path / 'out.json', tmp_path / 'out.csv', tmp_path / 'out.md'
    args = parser().parse_args(['--candidate', str(model_path), '--groups', '2',
                                 '--seed', '987664', '--opponents', 'random,tight',
                                 '--no-progress', '--output-json', str(json_path),
                                 '--output-csv', str(csv_path), '--output-md', str(md_path)])
    report = run(args)
    assert report['phase'] == 16
    assert report['pool_summary']['total_hands'] == 16
    assert json.loads(json_path.read_text(encoding='utf-8'))['phase'] == 16
    assert csv_path.read_text(encoding='utf-8').count('\n') == 3
    assert 'Phase 16' in md_path.read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):
        run(args)


def test_cli_named_checkpoint_and_invalid_spec(tmp_path):
    model_path = tmp_path / 'source.pt'
    checkpoint(model_path)
    args = parser().parse_args(['--candidate', str(model_path), '--opponents', '',
                                '--opponent-checkpoint', f'previous={model_path}'])
    assert _specs(args)[0].name == 'previous'
    args = parser().parse_args(['--candidate', str(model_path), '--opponents', '',
                                '--opponent-checkpoint', 'broken'])
    with pytest.raises(ValueError):
        _specs(args)
    args = parser().parse_args(['--candidate', str(model_path), '--opponents', ''])
    with pytest.raises(ValueError):
        _specs(args)

````````

<!-- END_PY_SOURCE path="tests/test_phase16_cli.py" -->

## tests/test_phase16_diagnostics.py

Git Blob `e857adaef64d7fee0104717cc5fbb554c7608fd5` (3119 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_phase16_diagnostics.py" sha="e857adaef64d7fee0104717cc5fbb554c7608fd5" -->

````````python
from __future__ import annotations

import pytest
import torch

from holdem_ai.core.schema import InformationState
from holdem_ai.models.transformer import NetworkConfig, TransformerActor
from holdem_ai.evaluation.strategy_diagnostics import ActionProfile, spr_bucket


def obs(*, mask=(True, False, True, True, True, True, True), street='preflop',
        button=1.0, pot=10.0, stack=20.0, opponent=30.0):
    return InformationState(player_id=0,
                            features=(0., button, 0., pot, 0., 0., stack, opponent, 0., 0., 0., 2.),
                            legal_action_mask=mask, private_cards=('sA', 'dK'),
                            stage=street)


def test_effective_spr_public_observable():
    assert spr_bucket(obs(pot=100., stack=10.)) == 'spr_lt_1'
    assert spr_bucket(obs(pot=10., stack=20.)) == 'spr_1_to_4'
    assert spr_bucket(obs(pot=5., stack=20.)) == 'spr_4_to_10'
    assert spr_bucket(obs(pot=1., stack=20.)) == 'spr_ge_10'


def test_conditional_rate_not_raw_rate_and_strata():
    profile = ActionProfile()
    profile.record(obs(mask=(True, False, True, True, True, True, True)), 0)
    profile.record(obs(mask=(False, True, False, True, True, True, True),
                       street='river', button=0.0, pot=60.), 1)
    out = profile.summary()
    assert out['decisions'] == 2
    assert out['actions']['fold'] == 1
    assert out['legal_opportunities']['fold'] == 1
    assert out['conditional_action_rate']['fold'] == 1.0
    assert out['by_street']['river']['actions']['check'] == 1
    assert out['by_position']['button']['actions']['fold'] == 1
    assert out['by_position']['big_blind']['actions']['check'] == 1
    assert out['by_spr']['spr_lt_1']['actions']['check'] == 1
    assert out['conditional_action_rate']['check'] == 1.0
    assert out['conditional_action_rate']['call'] == 0.0
    assert out['mean_policy_entropy'] is None
    assert 'private_cards' not in str(out)


def test_illegal_selection_rejected():
    profile = ActionProfile()
    with pytest.raises(ValueError):
        profile.record(obs(), 1)
    with pytest.raises(ValueError):
        profile.record(obs(street='unknown'), 0)
    with pytest.raises(ValueError):
        spr_bucket(obs(pot=-1))
    with pytest.raises(ValueError):
        ActionProfile().summary(entropy_batch_size=0)


def test_entropy_uses_only_actor_information():
    torch.manual_seed(22)
    model = TransformerActor(NetworkConfig(embed_dim=32, attention_heads=2,
                            transformer_layers=1, ff_dim=64, hidden_dim=48, max_tokens=256))
    profile = ActionProfile()
    from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
    real_obs = HUNL200Adapter().reset(seed=16).observation
    profile.record(real_obs, real_obs.legal_actions[0])
    profile.record(real_obs, real_obs.legal_actions[-1])
    model.train()
    result = profile.summary(model, device='cpu', entropy_batch_size=1)
    assert result['mean_policy_entropy'] >= 0
    assert -0.000001 <= result['mean_normalized_entropy'] <= 1.000001
    assert model.training is True
    assert 'private_cards' not in str(result)

````````

<!-- END_PY_SOURCE path="tests/test_phase16_diagnostics.py" -->

## tests/test_phase16_evaluation.py

Git Blob `e916b432c2f4525936f81f264f61768055d76233` (3714 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_phase16_evaluation.py" sha="e916b432c2f4525936f81f264f61768055d76233" -->

````````python
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from holdem_ai.models.transformer import NetworkConfig, TransformerActor
from holdem_ai.agents.diagnostic_opponents import OpponentSpec
from holdem_ai.evaluation.opponent_pool import PoolEvaluationConfig, evaluate_opponent_pool
from holdem_ai.evaluation.seed_blocked import model_fingerprint


@pytest.fixture(scope='module')
def net():
    torch.manual_seed(16)
    return TransformerActor(NetworkConfig(embed_dim=32, attention_heads=2,
                            transformer_layers=1, ff_dim=64, hidden_dim=48,
                            max_tokens=256))


def config():
    return PoolEvaluationConfig(groups=2, seed=16001111, mode='greedy')


def test_multiplayer_pool_reproducible_and_no_mutation(net):
    before = model_fingerprint(net)
    net.train(True)
    opponents = (OpponentSpec('random'), OpponentSpec('calling_station'),
                 OpponentSpec('pot_pressure'), OpponentSpec('tight'))
    a = evaluate_opponent_pool(net, opponents, config=config())
    b = evaluate_opponent_pool(net, opponents, config=config())
    assert a == b
    assert model_fingerprint(net) == before
    assert net.training is True
    assert a['pool_summary']['total_hands'] == 32
    assert a['pool_summary']['opponent_count'] == 4
    assert len(a['opponents']) == 4
    for row in a['opponents']:
        assert row['hands'] == 8
        assert row['groups'] == 2
        assert row['ci95_lower_bb100'] <= row['mean_bb100'] <= row['ci95_upper_bb100']
        p = row['candidate_action_profile']
        assert p['decisions'] == sum(p['actions'].values())
        assert all(p['actions'][x] <= p['legal_opportunities'][x] for x in p['actions'])
        assert sum(x['decisions'] for x in p['by_street'].values()) == p['decisions']
        assert sum(x['decisions'] for x in p['by_spr'].values()) == p['decisions']
    assert 'private_cards' not in str(a)
    assert 'CriticState' not in str(a)


def test_empty_and_duplicate_opponents_rejected(net):
    with pytest.raises(ValueError):
        evaluate_opponent_pool(net, (), config=config())
    with pytest.raises(ValueError):
        evaluate_opponent_pool(net, (OpponentSpec('random'), OpponentSpec('random')),
                               config=config())


@pytest.mark.parametrize('changes', [
    {'groups': 1}, {'groups': True}, {'seed': -1}, {'mode': 'wrong'},
    {'device': 'gpu'}, {'max_decisions': 0}, {'entropy_batch_size': 0},
    {'big_blind': 1},
])
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        replace(config(), **changes)


def test_neural_checkpoint_mirror_and_profile(net, tmp_path):
    source = tmp_path / 'anchor.pt'
    torch.save({'network_config': vars(net.config),
                'actor_state': net.state_dict()}, source)
    out = evaluate_opponent_pool(net, (OpponentSpec('neural_copy', str(source)),),
                                 config=config())
    assert out['opponents'][0]['mean_bb100'] == 0.0
    assert out['opponents'][0]['opponent_id'] == model_fingerprint(net)


def test_on_group_callback(net):
    seen = []
    evaluate_opponent_pool(net, (OpponentSpec('random'),),
                           config=config(), on_group=seen.append)
    assert seen == ['random', 'random']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_cuda_only_when_available(net):
    actor = TransformerActor(net.config).cuda()
    actor.load_state_dict(net.state_dict())
    out = evaluate_opponent_pool(actor, (OpponentSpec('random'),),
                                 config=replace(config(), device='cuda'))
    assert out['pool_summary']['total_hands'] == 8

````````

<!-- END_PY_SOURCE path="tests/test_phase16_evaluation.py" -->

## tests/test_phase16_opponents.py

Git Blob `7263b79951c3edd90c5b1fa2dcca4ed90da22e50` (1570 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_phase16_opponents.py" sha="7263b79951c3edd90c5b1fa2dcca4ed90da22e50" -->

````````python
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

````````

<!-- END_PY_SOURCE path="tests/test_phase16_opponents.py" -->

## tests/test_phase16_ppo_ablation.py

Git Blob `84ac169a69b5ed53bffdd74e2d0c1edee63c0d67` (7515 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_phase16_ppo_ablation.py" sha="84ac169a69b5ed53bffdd74e2d0c1edee63c0d67" -->

````````python
"""Phase16 isolated PPO ablations: objectives, KL, checkpoints and safety."""
from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from holdem_ai.encoding.observation import EncodedActor, EncodedCritic
from holdem_ai.experiments import ppo_ablation as a
from holdem_ai.models.transformer import CentralizedQCritic, NetworkConfig, TransformerActor
from holdem_ai.training.ppo import PPOConfig, TrajectoryRows
from holdem_ai.training.ppo_session import create_session, save_checkpoint


def _net():
    return NetworkConfig(
        embed_dim=8, attention_heads=2, transformer_layers=1,
        ff_dim=16, hidden_dim=12, max_tokens=256,
    )


def _rows():
    states = []
    critics = []
    for street in (0., 0., 1., 1.):
        f = [0.] * 20
        f[1] = 1.
        f[2] = street
        states.append(EncodedActor(
            (1,), (0.,), tuple(f),
            (True, False, True, True, True, False, True),
        ))
        c = [0.] * 24
        c[0] = 1.
        c[2] = 1.
        critics.append(EncodedCritic((1,), (0.,), tuple(c)))
    return TrajectoryRows(
        actors=tuple(states), critics=tuple(critics),
        actions=(3, 6, 2, 0), acting_seats=(0, 1, 0, 1),
        returns_bb=(200., -200., -2., 2.),
        hands=2, hand_rewards=((200., -200.), (-2., 2.)),
        terminal_actions=(6, 0), decisions_per_hand=(2, 2),
    )


def test_direct_v_zero_sum_warm_start_and_gradient():
    q = CentralizedQCritic(_net())
    model = a.DirectVCritic(q)
    cb = a.batch_critics(_rows().critics)
    with torch.no_grad():
        predictions = model(cb)
    assert predictions.shape == (4, 2)
    assert torch.allclose(predictions[:, 0], -predictions[:, 1], atol=1e-6)
    loss = predictions[:, 0].sum()
    # Forward is used with gradients after the no_grad correctness probe.
    model(cb)[:, 0].sum().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_scale_q_head_reparameterizes_all_action_outputs():
    model = CentralizedQCritic(_net())
    cb = a.batch_critics(_rows().critics)
    with torch.no_grad():
        original = model(cb)
    a.scale_q_head_(model, 200.)
    with torch.no_grad():
        scaled = model(cb)
    assert torch.allclose(scaled * 200., original, atol=5e-5)
    with pytest.raises(ValueError):
        a.scale_q_head_(model, -1)


def test_exact_post_update_kl_is_zero_then_positive():
    actor = TransformerActor(_net())
    rows = _rows()
    original = a.snapshot_probs(actor, rows, device="cpu")
    assert a.exact_policy_kl(actor, rows, old_probs=original, device="cpu") == pytest.approx(0., abs=1e-6)
    with torch.no_grad():
        actor.policy_head[-1].bias[6].add_(3.)
    changed = a.exact_policy_kl(actor, rows, old_probs=original, device="cpu")
    assert changed > .01


@pytest.mark.parametrize("variant", ["a1", "a2"])
def test_freeze_and_train_new_variants_cpu(variant):
    actor = TransformerActor(_net())
    q = CentralizedQCritic(_net())
    if variant == "a1":
        a.scale_q_head_(q, 200.)
        critic = q
    else:
        critic = a.DirectVCritic(q)
    rows = _rows()
    frozen = a.freeze_variant(
        actor, critic, rows, variant=variant, target_scale=200.,
        device="cpu", chunk_size=2,
    )
    assert frozen.old_log_probs.shape == (4,)
    assert frozen.advantages.std(unbiased=False).item() == pytest.approx(1., abs=1e-5)
    assert torch.isfinite(frozen.advantages).all()
    cfg = PPOConfig(hands_per_update=2, epochs=1, minibatch_size=4, seed=20)
    ao = torch.optim.Adam(actor.parameters(), lr=3e-4)
    co = torch.optim.Adam(critic.parameters(), lr=3e-4)
    stats = a.optimize_variant(
        actor, critic, ao, co, frozen, cfg, variant=variant,
        target_scale=200., device="cpu", shuffle_seed=123,
    )
    assert stats["minibatches"] == 1
    assert stats["decisions"] == 4
    assert math.isfinite(stats["critic_loss"])


@pytest.mark.parametrize("config", [
    {"variants": ("a0", "a0")},
    {"variants": ("broken",)},
    {"updates": 0},
    {"eval_groups": 1},
    {"env_batch_size": 1000},
    {"max_post_kl": -1.},
    {"target_scale": 0.},
    {"validation_seed": 660000000},
])
def test_invalid_ablation_config(config):
    with pytest.raises(ValueError):
        a.AblationConfig(source_checkpoint="x", root="y", **config)


def test_exclusive_run_path_rejected_prior_to_load(tmp_path):
    source = tmp_path / "dummy.pt"
    source.write_bytes(b"trusted placeholder")
    used = tmp_path / "already_used"
    used.mkdir()
    (used / "report.json").write_text("{}")
    with pytest.raises(FileExistsError):
        a.run_ablation(a.AblationConfig(
            source_checkpoint=str(source), root=str(used),
            variants=("a0",), device="cpu", updates=1,
        ))


def test_pilot_one_update_real_env_and_evaluation_cpu(tmp_path):
    """Small end-to-end new-run smoke with two-hand rollouts, no existing-run edits."""
    source = tmp_path / "source.pt"
    session = create_session(
        seed=77, network_config=_net(),
        ppo_config=PPOConfig(hands_per_update=2, epochs=1, minibatch_size=2),
    )
    save_checkpoint(session, source)
    original = source.read_bytes()
    root = tmp_path / "phase16_new"
    cfg = a.AblationConfig(
        source_checkpoint=str(source), root=str(root),
        variants=("a1",), device="cpu", hands_per_update=2,
        updates=1, minibatch_size=2, env_batch_size=2,
        eval_groups=2, critic_eval_hands=2,
    )
    result = a.run_ablation(cfg)
    assert source.read_bytes() == original
    assert result["variants"]["a1"]["critic_holdout"]["decisions"] >= 2
    assert (root / "run.json").is_file()
    assert (root / "a1" / "result.json").is_file()
    assert (root / "a1" / "final_actor.pt").is_file()
    with pytest.raises(FileExistsError):
        a.run_ablation(cfg)



def test_a3_rejected_update_restores_actor_and_keeps_source(tmp_path, monkeypatch):
    """A3 must rollback a deliberately mutated Actor when post-update KL fails."""
    from holdem_ai.evaluation.checkpoint_loader import load_actor

    path = tmp_path / "source.pt"
    session = create_session(
        seed=77, network_config=_net(),
        ppo_config=PPOConfig(hands_per_update=2, epochs=1, minibatch_size=2),
    )
    save_checkpoint(session, path)
    original_bytes = path.read_bytes()
    saved_actor = load_actor(path, device="cpu")

    def fake_optim(actor, critic, ao, co, frozen, config, *, device, shuffle_seed):
        with torch.no_grad():
            actor.policy_head[-1].bias[6].add_(25.)
            critic.q_head[-1].bias[6].add_(25.)
        return {"critic_loss": 1.0, "minibatches": 1,
                "decisions": len(frozen.rows.actions)}

    monkeypatch.setattr(a, "optimize_ppo", fake_optim)
    monkeypatch.setattr(a, "exact_policy_kl",
                        lambda actor, rows, *, old_probs, device: 10.)
    root = tmp_path / "a3_reject"
    cfg = a.AblationConfig(
        source_checkpoint=str(path), root=str(root),
        variants=("a3",), hands_per_update=2, updates=1,
        minibatch_size=2, env_batch_size=2, eval_groups=2,
        critic_eval_hands=2,
    )
    report = a.run_ablation(cfg)
    after = load_actor(root / "a3" / "final_actor.pt", device="cpu")
    assert report["variants"]["a3"]["rejected_updates"] == 1
    assert all(torch.equal(saved_actor.state_dict()[key], after.state_dict()[key])
               for key in saved_actor.state_dict())
    assert path.read_bytes() == original_bytes

````````

<!-- END_PY_SOURCE path="tests/test_phase16_ppo_ablation.py" -->

## tests/test_phase16_ppo_stress.py

Git Blob `c0108d4ac755da9211472d966214c93b386efa5b` (9100 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_phase16_ppo_stress.py" sha="c0108d4ac755da9211472d966214c93b386efa5b" -->

````````python
"""Phase16 A0/A3 long-stress regression and real-HUNL mini integration tests."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from holdem_ai.experiments import ppo_stress as stress
from holdem_ai.evaluation.checkpoint_loader import load_actor
from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import PPOConfig
from holdem_ai.training.ppo_session import create_session, save_checkpoint


def _source(tmp_path: Path) -> Path:
    net = NetworkConfig(
        embed_dim=8, attention_heads=2,
        transformer_layers=1, ff_dim=16, hidden_dim=12,
        max_tokens=256,
    )
    session = create_session(
        seed=41, network_config=net,
        ppo_config=PPOConfig(seed=41, hands_per_update=2, epochs=1,
                             minibatch_size=2),
    )
    source = tmp_path / "source.pt"
    save_checkpoint(session, source)
    return source


def _cfg(source: Path, root: Path, variant: str = "a0", **kwargs) -> stress.StressConfig:
    values = dict(
        source_checkpoint=str(source),
        root=str(root),
        variant=variant,
        device="cpu",
        updates=2,
        hands_per_update=2,
        minibatch_size=2,
        epochs=1,
        env_batch_size=2,
        eval_every=2,
        eval_groups=2,
        critic_eval_hands=2,
        save_every=1,
        rollout_seed=680000000,
        validation_seed=78000000,
        critic_seed=690000000,
    )
    values.update(kwargs)
    return stress.StressConfig(**values)


@pytest.mark.parametrize("bad", [
    {"variant": "a2"},
    {"updates": 0},
    {"hands_per_update": 0},
    {"eval_groups": 1},
    {"env_batch_size": 3},
    {"max_post_kl": 0},
    {"hard_stop_kl": .01},
    {"rollout_seed": -1},
    {"critic_seed": 680000001},
    {"max_updates_this_call": 0},
])
def test_reject_invalid_stress_settings(bad):
    values = dict(
        source_checkpoint="model.pt", root="new_run", variant="a0",
        hands_per_update=2, env_batch_size=2,
    )
    values.update(bad)
    with pytest.raises(ValueError):
        stress.StressConfig(**values)


def test_full_checkpoint_resume_no_duplicate_rollouts(tmp_path):
    source = _source(tmp_path)
    source_bytes = source.read_bytes()
    base = _cfg(source, tmp_path / "continued", max_updates_this_call=1)
    first = stress.run_stress(base)
    assert first["attempted"] == 1 and not first["completed"]
    checkpoint = tmp_path / "continued" / "latest.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["stress"]["attempted_updates"] == 1
    assert payload["completed_updates"] == 1
    assert "actor_optimizer" in payload and "critic_optimizer" in payload
    assert payload["stress"]["records"][0]["global_update"] == 1
    assert (tmp_path / "continued" / "snapshots" / "attempt_00001.pt").exists()
    assert source.read_bytes() == source_bytes
    # The checkpoint is a true PPO resume checkpoint, not an actor-only export.
    resumed_session = stress.load_checkpoint(checkpoint, device="cpu")
    assert resumed_session.completed_updates == 1

    second = stress.run_stress(
        replace(base, resume=True, max_updates_this_call=None))
    assert second["attempted"] == 2 and second["completed"]
    assert second["training_hands"] == 4
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert [r["attempted"] for r in state["stress"]["records"]] == [1, 2]
    assert [v["attempted"] for v in state["stress"]["validations"]] == [2]
    assert (tmp_path / "continued" / "snapshots" / "attempt_00002.pt").exists()
    assert source.read_bytes() == source_bytes
    assert stress.run_stress(
        replace(base, resume=True, max_updates_this_call=1))["attempted"] == 2
    with pytest.raises(FileExistsError):
        stress.run_stress(base)


def test_resumed_model_identical_to_uninterrupted(tmp_path):
    source = _source(tmp_path)
    base = _cfg(source, tmp_path / "whole")
    whole = stress.run_stress(base)
    assert whole["completed"]
    cut = _cfg(source, tmp_path / "cut", max_updates_this_call=1)
    stress.run_stress(cut)
    stress.run_stress(replace(cut, resume=True, max_updates_this_call=None))
    p = torch.load(tmp_path / "whole" / "latest.pt", map_location="cpu", weights_only=True)
    q = torch.load(tmp_path / "cut" / "latest.pt", map_location="cpu", weights_only=True)
    for name in ("actor_state", "critic_state"):
        assert all(torch.equal(p[name][k], q[name][k]) for k in p[name])
    for name in ("actor_optimizer", "critic_optimizer"):
        # Parameter-index dictionaries and Adam moments must be restored.
        assert p[name]["param_groups"] == q[name]["param_groups"]
        assert p[name]["state"].keys() == q[name]["state"].keys()
        for k in p[name]["state"]:
            for field in p[name]["state"][k]:
                left, right = p[name]["state"][k][field], q[name]["state"][k][field]
                if torch.is_tensor(left):
                    assert torch.equal(left, right)
                else:
                    assert left == right
    assert p["stress"]["records"] == q["stress"]["records"] or [
        {k: v for k, v in x.items() if k != "elapsed_seconds"}
        for x in p["stress"]["records"]
    ] == [
        {k: v for k, v in x.items() if k != "elapsed_seconds"}
        for x in q["stress"]["records"]
    ]


def test_a3_forced_reject_restores_full_actor_critic(tmp_path, monkeypatch):
    source = _source(tmp_path)
    saved = stress.load_checkpoint(source, device="cpu")
    # Force the KL gate to reject the update, regardless of its normal size.
    monkeypatch.setattr(
        stress, "exact_policy_kl",
        lambda actor, rows, *, old_probs, device: 1.0,
    )
    cfg = _cfg(source, tmp_path / "rejected", variant="a3",
               updates=1, eval_every=1)
    report = stress.run_stress(cfg)
    assert report["attempted"] == 1
    assert report["accepted"] == 0
    assert report["rejected"] == 1
    saved_actor = saved.actor.state_dict()
    saved_critic = saved.critic.state_dict()
    updated = stress.load_checkpoint(tmp_path / "rejected" / "latest.pt", device="cpu")
    assert all(torch.equal(updated.actor.state_dict()[k], value)
               for k, value in saved_actor.items())
    assert all(torch.equal(updated.critic.state_dict()[k], value)
               for k, value in saved_critic.items())
    assert updated.completed_updates == 1
    payload = torch.load(tmp_path / "rejected" / "latest.pt",
                         map_location="cpu", weights_only=True)
    assert payload["stress"]["records"][0]["rejected"] is True


def test_paired_a0_a3_report_same_validation_decks(tmp_path):
    source = _source(tmp_path)
    cfg0 = _cfg(source, tmp_path / "a0", variant="a0", updates=1, eval_every=1)
    cfg3 = _cfg(source, tmp_path / "a3", variant="a3", updates=1, eval_every=1)
    stress.run_stress(cfg0)
    stress.run_stress(cfg3)
    paired = stress.compare_stress_runs(tmp_path / "a0", tmp_path / "a3")
    assert paired["a0_attempted"] == paired["a3_attempted"] == 1
    assert "1" in paired["paired_validations"]
    for label in ("random_legal", "frozen_phase14"):
        assert paired["paired_validations"]["1"][label]["a3_minus_a0_delta_bb100"] == 0.
    with pytest.raises(ValueError, match="cannot pair"):
        # Modify only the CONFIG record on disk in a copy? The function should
        # reject a changed seed even if a maliciously edited manifest mismatches.
        different = _cfg(
            source, tmp_path / "nonmatching", variant="a3",
            updates=1, eval_every=1, validation_seed=78000001)
        stress.run_stress(different)
        stress.compare_stress_runs(tmp_path / "a0", tmp_path / "nonmatching")


def test_completed_source_checkpoint_is_never_overwritten(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(ValueError):
        stress.run_stress(_cfg(source, source.parent, variant="a0"))
    assert source.exists()


def test_recover_committed_checkpoint_when_metrics_mirror_is_missing(tmp_path):
    source = _source(tmp_path)
    cfg = _cfg(source, tmp_path / "rebuild",
               max_updates_this_call=1)
    real_mirror = stress._mirrors
    observed = {"calls": 0}

    def interrupt_mirror(*args, **kwargs):
        observed["calls"] += 1
        if observed["calls"] == 2:
            raise RuntimeError("simulated power cut after atomic checkpoint")
        return real_mirror(*args, **kwargs)

    from unittest.mock import patch
    with patch.object(stress, "_mirrors", interrupt_mirror):
        with pytest.raises(RuntimeError, match="simulated power cut"):
            stress.run_stress(cfg)
    full = torch.load(tmp_path / "rebuild" / "latest.pt",
                      map_location="cpu", weights_only=True)
    assert full["stress"]["attempted_updates"] == 1
    resumed = stress.run_stress(replace(cfg, resume=True, max_updates_this_call=None))
    assert resumed["attempted"] == 2
    assert (tmp_path / "rebuild" / "metrics.jsonl").exists()

````````

<!-- END_PY_SOURCE path="tests/test_phase16_ppo_stress.py" -->
