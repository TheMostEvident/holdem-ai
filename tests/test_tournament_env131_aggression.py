"""ENV 1.3.1 regression tests for multiway bot aggression and action safety."""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import pytest

from holdem_ai.tournament_service.improved_bot import ImprovedTournamentBot, PROFILES
from holdem_ai.tournament_service.session import TournamentGameSession


def observation(hole, *, legal=(0, 1, 2, 3, 4), board=(), call=10, pot=45, stack=20000):
    return {
        'legal_actions': {int(action): None for action in legal},
        'raw_obs': {
            'hand': list(hole), 'public_cards': list(board),
            'player_id': 2, 'to_call': call, 'pot': pot,
            'stacks': [stack] * 6, 'status': ['active'] * 6,
        },
    }


@pytest.mark.parametrize('profile', PROFILES)
def test_pot_raise_remains_possible_when_half_pot_is_illegal(profile):
    # Under old code the guard `and 2 in legal` wrongly downgraded
    # a legal pot raise to a call for even AA.
    bot = ImprovedTournamentBot(profile, seed=42)
    for _ in range(5):
        assert bot.select_action(observation(('sA','hA'), legal=(0,1,3,4))) == 3


@pytest.mark.parametrize('profile,min_raises', [
    ('tight_aggressive', 45),
    ('balanced_value', 150),
    ('pot_odds', 75),
    ('controlled_pressure', 240),
])
def test_policies_reach_value_raise_on_preflop_hand_distribution(profile, min_raises):
    """Synthetic 1,326 two-card combos, not an estimate of real match raise %.

    Fixed pot/stack/legal actions deliberately isolate the policy from engine
    and strategy effects. Real-game action frequencies require match logging.
    """
    deck = [f'{s}{r}' for s in 'shdc' for r in '23456789TJQKA']
    bot = ImprovedTournamentBot(profile, seed=2026)
    counts = Counter(bot.select_action(observation(cards)) for cards in combinations(deck, 2))
    assert sum(counts.values()) == 1326
    assert counts[2] + counts[3] >= min_raises, counts
    assert counts[4] == 0  # deep stack: no blind automatic all-in
    assert set(counts) <= {0,1,2,3,4}


def test_agent_slot_is_honestly_marked_as_untrained_heuristic():
    game = TournamentGameSession(seed=31, starting_stack=400,
                                 hands_per_level=2, lineup='improved_4')
    seat = game.view()['seats'][1]
    assert seat['role'] == 'tournament_agent_placeholder'
    assert seat['policy_name'] == 'controlled_pressure'
    assert 'PPO' not in str(seat) and 'trained' not in str(seat)


def test_no_raise_if_engine_does_not_offer_any_raise_action():
    for profile in PROFILES:
        bot = ImprovedTournamentBot(profile, seed=20)
        for _ in range(10):
            assert bot.select_action(observation(('sA', 'hA'), legal=(0,1))) in (0, 1)
