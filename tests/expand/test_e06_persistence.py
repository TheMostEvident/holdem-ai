import json
import pytest
from holdem_ai.expand.opponent import OpponentRegistry,PublicDecision,save_registry,load_registry


def test_checkpoint_roundtrip_resume(tmp_path):
    r=OpponentRegistry();r.bind_seat(3,'bob','join-123')
    for i in range(40):
        r.observe(PublicDecision('bob','join-123',f'h{i}',i,3,'flop','middle','ge10',('fold','call'),'call'))
    file=tmp_path/'stats.json'
    save_registry(file,r)
    restored=load_registry(file)
    assert restored.evidence(3,('flop','middle','ge10',('fold','call'))).posterior==pytest.approx(
        r.evidence(3,('flop','middle','ge10',('fold','call'))).posterior)
    assert restored.active_by_seat==r.active_by_seat
    with pytest.raises(ValueError):restored.observe(PublicDecision('bob','join-123','h2',2,3,'flop','middle','ge10',('fold','call'),'call'))
    restored.observe(PublicDecision('bob','join-123','new',41,3,'flop','middle','ge10',('fold','call'),'fold'))
    assert restored.evidence(3,('flop','middle','ge10',('fold','call'))).raw_events==41
    with pytest.raises(FileExistsError):save_registry(file,r)

def test_checkpoint_tamper_detection(tmp_path):
    r=OpponentRegistry();r.bind_seat(1,'id','tenure')
    target=tmp_path/'registry.json';save_registry(target,r)
    text=target.read_text();target.write_text(text.replace('"player_id": "id"','"player_id": "evil"'))
    with pytest.raises(ValueError):load_registry(target)

def test_checkpoint_contains_no_card_or_deck(tmp_path):
    r=OpponentRegistry();r.bind_seat(1,'id','tenure')
    target=tmp_path/'registry.json';save_registry(target,r)
    payload=json.loads(target.read_text())['payload']
    assert set(payload)=={'version','config','active_seats','models'}
