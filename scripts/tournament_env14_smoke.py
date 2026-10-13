"""Real trusted Expand Phase3 checkpoint smoke for Tournament ENV 1.4.

Requires HOLDEM_AI_TOURNAMENT_CHECKPOINT environment variable pointing to
an EXISTING compatible 6-seat Expand E03 checkpoint. Never mutates weights.
"""
from __future__ import annotations

import json

from holdem_ai.tournament_service.session import TournamentGameSession
from holdem_ai.tournament_service.trained_actor import status


def main() -> None:
    if not status()['file_found']:
        raise SystemExit('Set HOLDEM_AI_TOURNAMENT_CHECKPOINT to runs/expand_phase03_pilot/best.pt before smoke')
    game = TournamentGameSession(seed=42, starting_stack=500, hands_per_level=5,
                                 lineup='trained_1_improved_4')
    moves=0
    while moves<50 and not game.view()['tournament_over']:
        view=game.view()
        assert view['seats'][1]['role']=='tournament_agent'
        assert sum(s['stack'] for s in view['seats'])+view['pot']==view['total_chips']
        if not view['your_turn']:
            raise AssertionError('session must auto-run all trained/improved bots until human turn')
        legal=[int(a['index']) for a in view['actions']]
        action=1 if 1 in legal else legal[0]
        game.play_action(action,expected_revision=view['revision'])
        moves+=1
    view=game.view()
    print(json.dumps({'phase':'Tournament ENV 1.4','checkpoint':status()['checkpoint_name'],
                      'agent':view['seats'][1]['policy_name'],'human_actions':moves,
                      'hands_completed':view['hands_completed'],
                      'chip_conservation':sum(s['stack'] for s in view['seats'])+view['pot']==view['total_chips'],
                      'result':'passed'},indent=2))


if __name__ == '__main__':
    main()
