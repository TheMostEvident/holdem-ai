import { useEffect, useState } from 'react';
import { formatChips } from '../lib/poker';
import { tournamentActorExportUrl, tournamentApi } from './api';
import { TournamentShowdown } from './TournamentShowdown';
import type { TournamentArchiveSession, TournamentHand, TournamentReplay } from './types';

const seatName = (seat: number) => seat === 0 ? 'Human' : `Seat ${seat + 1}`;

/** Phase 1.6 read-only archive; never invokes the poker action endpoints. */
export function TournamentArchive({ refreshToken }: { refreshToken: number }) {
  const [list, setList] = useState<TournamentArchiveSession[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [hands, setHands] = useState<TournamentHand[]>([]);
  const [replay, setReplay] = useState<TournamentReplay | null>(null);
  const [step, setStep] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    let alive = true;
    tournamentApi.archive(0, 50).then(result => { if (alive) { setList(result.sessions); setError(null); } })
      .catch((err: unknown) => { if (alive) setError(err instanceof Error ? err.message : 'Archive unavailable'); });
    return () => { alive = false; };
  }, [refreshToken, refresh]);

  const chooseSession = async (id: string) => {
    setBusy(true); setError(null); setSelected(id); setReplay(null); setHands([]);
    try { const data = await tournamentApi.archivedHands(id, 0, 100); setHands(data.hands); }
    catch (e) { setError(e instanceof Error ? e.message : 'Cannot load recorded hands'); }
    finally { setBusy(false); }
  };
  const chooseHand = async (hand: number) => {
    if (!selected) return;
    setBusy(true); setError(null);
    try { const data = await tournamentApi.replay(selected, hand); setReplay(data); setStep(0); }
    catch (e) { setError(e instanceof Error ? e.message : 'Cannot replay hand'); }
    finally { setBusy(false); }
  };

  const archiveEntry = list.find(item => item.session_id === selected);
  const lastFinished = hands.length ? Math.max(...hands.map(h => h.hand_number)) : 0;
  const partialHand = archiveEntry && archiveEntry.latest_hand > lastFinished ? archiveEntry.latest_hand : null;
  const maxStep = replay?.actions.length ?? 0;
  const frame = replay && step < replay.actions.length ? replay.actions[step] : null;
  const final = replay?.settlement ?? null;
  return <section className="t-archive panel">
    <div className="t-side-heading"><div><span className="eyebrow">TOURNAMENT ENV / 1.6</span><h2>Saved sessions &amp; hand replay</h2></div>
      <button type="button" disabled={busy} onClick={() => setRefresh(x => x + 1)}>↻ Refresh archive</button></div>
    <p className="t-muted">Saved on the local computer. Archives survive Backend restart; an interrupted live tournament cannot be resumed. Before settlement, AI hole cards remain hidden; only actual showdown hands are opened after the final action.</p>
    {error && <p className="t-error-banner" role="alert">{error}</p>}
    <div className="t-archive-grid">
      <div><span className="section-label">SESSIONS · LATEST 50</span>
        {list.length === 0 && <p className="t-muted">No tournament archive yet. Play a hand, then refresh.</p>}
        {list.map(s => <button key={s.session_id} type="button" className={`t-archive-entry ${selected === s.session_id ? 't-archive-selected' : ''}`}
          disabled={busy} onClick={() => void chooseSession(s.session_id)}>
          <strong>{new Date(s.created_utc).toLocaleString()}</strong><small>{s.options.lineup ?? 'Unknown lineup'} · {s.status} · hand {s.latest_hand}</small>
        </button>)}
        {selected && <a className="t-archive-export" href={tournamentActorExportUrl(selected)} download>↓ Export actor-visible JSONL</a>}
      </div>
      <div><span className="section-label">RECORDED HANDS · LATEST 100</span>
        {selected && hands.length === 0 && <p className="t-muted">No completed hands yet. Partial hands appear in the export after an action.</p>}
        {partialHand != null && <button type="button" disabled={busy} onClick={() => void chooseHand(partialHand)}>
          Hand #{partialHand} · PARTIAL RECORD (no confirmed settlement)
        </button>}
        <div className="t-archive-hands">{hands.map(h => <button key={h.hand_number} type="button" disabled={busy}
          onClick={() => void chooseHand(h.hand_number)}>Hand #{h.hand_number} · {h.small_blind}/{h.big_blind} · Human {h.payoffs_chips[0] >= 0 ? '+' : ''}{h.payoffs_chips[0]}</button>)}</div>
      </div>
    </div>
    {replay && <div className="t-replay" aria-label="Tournament hand replay">
      <div className="t-side-heading"><strong>HAND #{replay.hand_number} · REPLAY</strong><small>Frame {step + 1} / {maxStep + 1}</small></div>
      <div className="t-replay-controls">
        <button type="button" disabled={step === 0} onClick={() => setStep(s => Math.max(0, s - 1))}>← Previous</button>
        <input aria-label="Replay frame" type="range" min="0" max={maxStep} value={step} onChange={e => setStep(Number(e.target.value))}/>
        <button type="button" disabled={step >= maxStep} onClick={() => setStep(s => Math.min(maxStep, s + 1))}>Next →</button>
      </div>
      {frame ? <div className="t-replay-data">
        <p><strong>Before action #{frame.action_no}: {seatName(frame.seat)} · {frame.kind}</strong> {frame.raise_to != null ? `to ${formatChips(frame.raise_to)}` : ''}</p>
        <p>{frame.public_before.street} · Board: {frame.public_before.board.join(' ') || '(preflop)'} · Pot {formatChips(frame.public_before.pot)} · Call {formatChips(frame.public_before.to_call)}</p>
        <p>Hero cards: {frame.hero_hole.length ? frame.hero_hole.join(' ') : 'Hidden / not the human turn'}</p>
        <div className="t-replay-seats">{frame.public_before.stacks.map((chips, i) => <div key={i}><strong>{seatName(i)}</strong><span>{formatChips(chips)} chips</span><small>{frame.public_before.statuses[i]}</small></div>)}</div>
      </div> : final ? <div className="t-replay-data"><p><strong>Showdown / settled result</strong> · Board: {final.board.join(' ') || '(no board)'}</p>
        <div className="t-replay-seats">{final.stacks_after.map((chips,i) => <div key={i}><strong>{seatName(i)}</strong><span>{formatChips(chips)} chips</span><small>Δ {final.payoffs_chips[i] >= 0 ? '+' : ''}{final.payoffs_chips[i]}</small></div>)}</div>
        <TournamentShowdown hand={final} />
        <p>Eliminated: {final.eliminated.length ? final.eliminated.map(seatName).join(', ') : 'None'}</p></div>
        : <p className="t-muted">This is a partial hand. No settlement record exists yet.</p>}
    </div>}
  </section>;
}
