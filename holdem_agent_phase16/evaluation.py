"""Phase16 diagnostic opponent-pool benchmark, independent of Phase15 training.

For every shuffle seed, the four layouts are replayed for every opponent.
One seed group (NOT one hand) is the statistical replicate.
The fixed opponent pool is a benchmark, not GTO exploitability.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import torch

from holdem_agent_phase5.schema import InformationState
from holdem_agent_phase6 import HUNL200Adapter, collect_global_episode
from holdem_agent_phase8 import NeuralActorPolicy, TransformerActor
from holdem_agent_phase12 import model_fingerprint, summarize_block_scores
from holdem_agent_phase12.checkpoint import load_actor

from .diagnostics import ActionProfile
from .opponents import OpponentSpec, build_builtin


@dataclass(frozen=True)
class PoolEvaluationConfig:
    groups: int = 40
    seed: int = 16_000_000
    mode: str = 'sampled'
    device: str = 'cpu'
    big_blind: int = 100
    max_decisions: int = 200
    entropy_batch_size: int = 64

    def __post_init__(self):
        if type(self.groups) is not int or self.groups < 2:
            raise ValueError('groups must be >=2')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be nonnegative int')
        if self.mode not in ('sampled', 'greedy'):
            raise ValueError('mode must be sampled or greedy')
        if self.device != 'cpu' and not self.device.startswith('cuda'):
            raise ValueError('invalid device')
        for name in ('big_blind', 'max_decisions', 'entropy_batch_size'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive int')
        if self.big_blind < 2:
            raise ValueError('HUNL200 requires big_blind >= 2 to set a positive smaller small blind')


class _TrackedCandidate:
    def __init__(self, neural_policy: NeuralActorPolicy, profile: ActionProfile):
        self.policy = neural_policy
        self.profile = profile

    def act(self, obs: InformationState, rng) -> int:
        action = self.policy.act(obs, rng)
        self.profile.record(obs, action)
        return action


def _opponent_policy(spec: OpponentSpec, config: PoolEvaluationConfig):
    if spec.checkpoint is None:
        return build_builtin(spec.name), 'builtin:' + spec.name, None
    model = load_actor(spec.checkpoint, device=config.device)
    if model.config.dropout:
        raise ValueError('opponent dropout must be zero during evaluation')
    policy = NeuralActorPolicy(model, device=config.device,
                               big_blind=config.big_blind, greedy=config.mode == 'greedy')
    return policy, model_fingerprint(model), model


def evaluate_opponent_pool(candidate: TransformerActor, opponents: tuple[OpponentSpec, ...],
                           *, config: PoolEvaluationConfig | None = None,
                           on_group: Callable[[str], None] | None = None) -> dict:
    """Returns JSON-safe summary without any hole cards, opponent cards or samples."""
    cfg = config or PoolEvaluationConfig()
    if not opponents or len({x.name for x in opponents}) != len(opponents):
        raise ValueError('provide a nonempty pool with unique opponent names')
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    if candidate.config.dropout:
        raise ValueError('candidate dropout must be zero during evaluation')
    identity = model_fingerprint(candidate)
    initial_mode = candidate.training
    results = []
    try:
        actor_policy = NeuralActorPolicy(candidate, device=cfg.device,
                                        big_blind=cfg.big_blind, greedy=cfg.mode == 'greedy')
        for spec in opponents:
            other, other_id, other_model = _opponent_policy(spec, cfg)
            action_profile = ActionProfile()
            wrapper = _TrackedCandidate(actor_policy, action_profile)
            blocks: list[float] = []
            for group in range(cfg.groups):
                seed = cfg.seed + group
                payoffs = []
                for button in (0, 1):
                    for candidate_seat in (0, 1):
                        adapter = HUNL200Adapter(button=button, big_blind=cfg.big_blind,
                                                 small_blind=max(1, cfg.big_blind // 2))
                        episode = collect_global_episode(
                            adapter, {candidate_seat: wrapper, 1-candidate_seat: other},
                            seed=seed, privileged=False, max_decisions=cfg.max_decisions,
                        )
                        if len(episode.terminal_rewards) != 2 or not math.isclose(
                                math.fsum(episode.terminal_rewards), 0.0, abs_tol=1e-7):
                            raise AssertionError('nonzero-sum HUNL game payoff')
                        payoffs.append(float(episode.terminal_rewards[candidate_seat]))
                blocks.append(math.fsum(payoffs) / 4)
                if on_group is not None:
                    on_group(spec.name)
            result = {
                'opponent': spec.name, 'opponent_id': other_id,
                **summarize_block_scores(tuple(blocks)),
                'groups': cfg.groups, 'hands': cfg.groups * 4,
                'candidate_action_profile': action_profile.summary(
                    candidate, device=cfg.device, big_blind=cfg.big_blind,
                    entropy_batch_size=cfg.entropy_batch_size,
                ),
            }
            results.append(result)
            del other_model
        if model_fingerprint(candidate) != identity:
            raise AssertionError('evaluation unexpectedly mutated candidate weights')
    finally:
        candidate.train(initial_mode)
    scores = [float(x['mean_bb100']) for x in results]
    return {
        'phase': 16,
        'protocol': 'phase16_fixed_opponent_pool_four_layout',
        'candidate_id': identity,
        'config': asdict(cfg),
        'opponents': results,
        'pool_summary': {
            'opponent_count': len(results),
            'total_hands': len(results) * cfg.groups * 4,
            'equal_weight_mean_bb100': round(math.fsum(scores) / len(scores), 4),
            'worst_opponent': results[min(range(len(scores)), key=scores.__getitem__)]['opponent'],
            'worst_observed_mean_bb100': round(min(scores), 4),
        },
        'interpretation': 'Fixed-pool empirical benchmark, not GTO exploitability or robust game-theoretic guarantee.',
    }
