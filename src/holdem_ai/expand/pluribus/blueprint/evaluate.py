"""Separate deterministic evaluation seeds; no trainer updates in evaluation."""
from __future__ import annotations
import hashlib
import math
import random
from .mccfr import MCCFRSolver


def evaluate_vs_uniform(solver: MCCFRSolver, *, hands: int = 100, seed: int = 440_000_000,
                        mode: str = 'average') -> dict:
    if type(hands) is not int or hands < 2:
        raise ValueError('at least two evaluation hands required')
    if mode not in ('average','current'):
        raise ValueError('policy mode must be average/current')
    scores = []
    rng = random.Random(seed)
    for _ in range(hands):
        hand_seed = rng.getrandbits(120)
        state = solver.game.start(random.Random(hand_seed))
        ids = getattr(state,'player_ids',tuple(range(solver.game.num_players)))
        hero = ids[rng.randrange(len(ids))]
        actor_rng = random.Random(rng.getrandbits(120))
        steps = 0
        while not state.is_terminal():
            steps += 1
            if steps > 256:
                raise RuntimeError('game not terminating; check game engine')
            actions = state.legal_actions()
            if state.current_player == hero:
                key=state.infoset(hero)
                p=solver.store.policy(key,actions,average=(mode=='average'))
                action=actor_rng.choices(actions,weights=p,k=1)[0]
            else:
                action=actor_rng.choice(actions)
            state=state.next_state(action)
        scores.append(float(state.terminal_payoffs()[hero]))
    mean=sum(scores)/len(scores)
    var=sum((x-mean)**2 for x in scores)/(len(scores)-1)
    return {'mode':mode,'hands':hands,'seed':seed,'mean_reward':round(mean,6),
            'ci95_lower':round(mean-1.96*math.sqrt(var/len(scores)),6),
            'ci95_upper':round(mean+1.96*math.sqrt(var/len(scores)),6),
            'unit':'BB/hand' if solver.game.name=='holdem' else 'toy_chips/hand',
            'note':'uniform-opponent exploratory evaluation, NOT Nash exploitability'}
