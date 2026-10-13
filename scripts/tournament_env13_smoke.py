"""Reproducible six-seat Tournament ENV 1.3 bot lineup smoke, no HTTP server."""
from __future__ import annotations

import json
from holdem_ai.tournament_service.improved_bot import PROFILES
from holdem_ai.tournament_service.session import TournamentGameSession


def main() -> None:
    game = TournamentGameSession(
        seed=42, starting_stack=100, hands_per_level=2,
        blind_levels=[(5, 10), (10, 20), (20, 40), (40, 80), (80, 160)],
        lineup="improved_4",
    )
    actions = 0
    while True:
        view = game.view()
        assert [x['role'] for x in view['seats']] == [
            'human', 'tournament_agent_placeholder', *(['improved_bot'] * 4),
        ]
        assert [x['policy_name'] for x in view['seats'][2:]] == list(PROFILES)
        assert sum(x['stack'] for x in view['seats']) + view['pot'] == 600
        if view['tournament_over']:
            break
        assert view['your_turn']
        game.play_action(1, expected_revision=view['revision'])
        actions += 1
        if actions > 1500:
            raise AssertionError('tournament exceeded smoke action budget')
    print(json.dumps(dict(phase='Tournament ENV 1.3', result='passed',
                          human_actions=actions, hands=view['hands_completed'],
                          winner_seat=view['winner_seat'],
                          lineup=view['lineup'], total_chips=view['total_chips']),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
