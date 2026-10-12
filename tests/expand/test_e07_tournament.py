import pytest
from holdem_ai.expand.integration import SingleTableTournament,TournamentConfig
from holdem_ai.expand.environment.table_state import Action
from e07_fake_tournament import KnockoutEngine


def mk(**kw):return SingleTableTournament(TournamentConfig(**kw),engine_factory=KnockoutEngine)


def drain(t):
    h=t.start_hand(100+t.hands_played)
    assert h is not None
    while t.current_player_id is not None:t.act(Action('check'))
    return t.settle_hand()


def test_e07_tournament_knockouts_chip_ledger_and_prizes():
    t=mk(payouts=(.7,.3,0.))
    for i in range(3):t.register(f'p{i}',seat=i)
    t.close_registration()
    first=drain(t)
    assert first.hand_number==1
    assert len(first.eliminated_entries)==1
    assert sum(first.payoffs_chips)==0
    while not t.finished:drain(t)
    report=t.result()
    assert report.hands_played==2
    assert len(report.entries)==3
    assert {x[2] for x in report.entries}=={1,2,3}
    assert sum(x[3] for x in report.entries)==pytest.approx(1.)
    assert t.start_hand(333) is None


def test_e07_reentry_new_id_late_registration():
    t=mk(registration_until_hand=3,max_reentries_per_player=1)
    t.register('a',seat=0);t.register('b',seat=1);t.register('c',seat=2)
    with pytest.raises(ValueError):t.register('a',seat=3,reentry=True)
    first=drain(t)
    assert 'a#entry1' in first.eliminated_entries
    second=t.register('a',seat=0,reentry=True)
    assert second=='a#entry2'
    with pytest.raises(ValueError):t.register('a',seat=3,reentry=True)
    drain(t)
    assert len(t.entries)==4
    assert t.entries[0].eliminated_hand==1
    assert t.entries[-1].entry_id=='a#entry2'
    t.close_registration()
    while not t.finished:drain(t)
    assert len(t.result().entries)==4


def test_e07_tournament_guard_bad_blinds_ante_registration():
    with pytest.raises(NotImplementedError):TournamentConfig(ante_chips=1)
    with pytest.raises(ValueError):TournamentConfig(blind_levels=((2,1),))
    t=mk(registration_until_hand=0)
    t.register('a');t.register('b')
    drain(t)
    with pytest.raises(RuntimeError):t.register('late',seat=2)


def test_e07_no_midhand_mutation_and_no_forged_completion():
    t=mk()
    t.register('a');t.register('b')
    h=t.start_hand(1)
    with pytest.raises(RuntimeError):t.register('c',seat=2)
    with pytest.raises(RuntimeError):t.close_registration()
    with pytest.raises(RuntimeError):t.result()
    while t.current_player_id is not None:t.act(Action('check'))
    t.settle_hand()
    assert t.finished


def test_e07_raise_mapping_short_call():
    t=mk()
    t.register('a');t.register('b')
    t.start_hand(3)
    with pytest.raises(ValueError):t.act(Action('call'))
    t.act(Action('check'))
