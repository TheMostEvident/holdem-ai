"""CLI contract tested with a fake engine; the real smoke is a Windows gate."""
import json
import pytest
from conftest import FakeHandEngine
from holdem_ai.expand.environment import DynamicTable
import holdem_ai.expand.environment.table_manager as tm
from scripts import expand_phase01_smoke as cli


def test_e01_smoke_cli_counts_exact_2_to_6_players(tmp_path, monkeypatch):
    def blinds(button, participants):
        ordered = [p for p in ((button + k) % 6 for k in range(1, 7)) if p in participants]
        return (button, ordered[0]) if len(participants) == 2 else (ordered[0], ordered[1])
    monkeypatch.setattr(tm, 'blind_seats', blinds)
    monkeypatch.setattr(cli, 'DynamicTable', lambda *args, **kwargs: DynamicTable(
        *args, engine_factory=FakeHandEngine, **kwargs))
    output = tmp_path / 'smoke.json'
    result = cli.run(hands_per_n=3, seed=101, output=str(output))
    assert result['target_hands'] == 15
    assert [x['hands'] for x in result['results']] == [3, 3, 3, 3, 3]
    assert result['total_hands'] >= 15
    assert all(x['final_table_chips'] == x['buyins_minus_cashouts'] for x in result['results'])
    assert json.loads(output.read_text(encoding='utf-8')) == result
    with pytest.raises(FileExistsError):
        cli.run(hands_per_n=3, seed=101, output=str(output))
