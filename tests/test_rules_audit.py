from types import SimpleNamespace

from holdem_ai.environment.rules_audit import audit_preflop, summarize


def test_audit_accepts_a_correct_heads_up_implementation():
    game = SimpleNamespace(
        dealer_id=0, game_pointer=0,
        small_blind=1, big_blind=2,
        init_chips=[100, 100],
        players=[SimpleNamespace(in_chips=1), SimpleNamespace(in_chips=2)],
    )
    checks = audit_preflop(game)
    assert len(checks) == 3
    assert all(c.passed for c in checks)


def test_audit_detects_a_reversed_blind_and_first_actor():
    game = SimpleNamespace(
        dealer_id=0, game_pointer=1,
        small_blind=1, big_blind=2,
        init_chips=[100, 100],
        players=[SimpleNamespace(in_chips=2), SimpleNamespace(in_chips=1)],
    )
    checks = audit_preflop(game)
    assert sum(not c.passed for c in checks) == 3
    assert 'FAIL' in summarize(checks)
