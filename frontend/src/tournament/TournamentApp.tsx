import { useCallback, useEffect, useRef, useState } from 'react';
import { formatChips, formatSigned } from '../lib/poker';
import { TournamentApiError, tournamentApi } from './api';
import { TournamentBetting } from './TournamentBetting';
import { TournamentTable } from './TournamentTable';
import { TournamentProgressPanel } from './TournamentProgressPanel';
import { TournamentArchive } from './TournamentArchive';
import { TournamentShowdown } from './TournamentShowdown';
import { chipsAccounting, eventText, seatLabel } from './rules';
import type { TournamentAgentStatus, TournamentHand, TournamentResponse } from './types';

const parseInteger = (input: string, name: string, min: number, max: number): number => {
  const trimmed = input.trim();
  if (!/^-?\d+$/.test(trimmed)) throw new Error(`${name}: enter a whole number.`);
  const num = Number(trimmed);
  if (!Number.isSafeInteger(num) || num < min || num > max) throw new Error(`${name}: allowed range ${min}–${max}.`);
  return num;
};

export function TournamentApp() {
  const [seed, setSeed] = useState('42');
  const [agentStatus, setAgentStatus] = useState<TournamentAgentStatus | null>(null);
  const [agentLineup, setAgentLineup] = useState<'improved_4' | 'trained_1_improved_4'>('improved_4');
  const [startingStack, setStartingStack] = useState('1000');
  const [handsPerLevel, setHandsPerLevel] = useState('10');
  const [button, setButton] = useState('0');
  const [session, setSession] = useState<TournamentResponse | null>(null);
  const [history, setHistory] = useState<TournamentHand[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [openedHand, setOpenedHand] = useState<number | null>(null);
  const [archiveRefresh, setArchiveRefresh] = useState(0);
  const [online, setOnline] = useState<boolean | null>(null);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState('Tournament ENV 1.6.1: showdown hands visible after settlement.');
  const flight = useRef(false);

  useEffect(() => {
    let active = true;
    tournamentApi.health().then(() => { if (active) setOnline(true); })
      .catch(() => { if (active) setOnline(false); });
    tournamentApi.agentStatus().then(result => {
      if (!active) return;
      setAgentStatus(result);
      if (result.configured && result.file_found) setAgentLineup('trained_1_improved_4');
    }).catch(() => { if (active) setAgentStatus(null); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!session || session.state.hands_completed === 0) { setHistory([]); setHistoryError(null); return; }
    let active = true;
    const offset = Math.max(0, session.state.hands_completed - 25);
    tournamentApi.history(session.session_id, offset, 25)
      .then(result => { if (active) { setHistory(result.hands.slice().reverse()); setHistoryError(null); } })
      .catch((cause: unknown) => { if (active) setHistoryError(cause instanceof Error ? cause.message : 'Cannot load tournament history'); });
    return () => { active = false; };
  }, [session?.session_id, session?.state.hands_completed]);

  const run = useCallback(async (work: () => Promise<void>) => {
    if (flight.current) return;
    flight.current = true;
    setWorking(true);
    setError(null);
    try { await work(); setOnline(true); }
    catch (cause) {
      const message = cause instanceof Error ? cause.message : 'Unknown tournament error';
      setError(message);
      if (cause instanceof TournamentApiError && cause.status === 0) setOnline(false);
    } finally { flight.current = false; setWorking(false); }
  }, []);

  const start = () => run(async () => {
    const payload = {
      seed: seed.trim() === '' ? null : parseInteger(seed, 'Seed', -2147483647, 2147483647),
      starting_stack: parseInteger(startingStack, 'Starting stack', 1, 10_000_000),
      button: parseInteger(button, 'Dealer seat', 0, 5),
      hands_per_level: parseInteger(handsPerLevel, 'Hands per blind level', 1, 100_000),
      max_hands: 10_000,
      lineup: agentLineup,
    };
    // Create first. An invalid new session must NOT erase the current match.
    const fresh = await tournamentApi.create(payload);
    if (session) { try { await tournamentApi.delete(session.session_id); } catch { /* new session already exists */ } }
    setSession(fresh);
    setHistory([]);
    setOpenedHand(null);
    setArchiveRefresh(x => x + 1);
    setNotice(fresh.state.tournament_over ? 'Tournament completed during automatic play.' : `Hand ${fresh.state.hand_number} · ${agentLineup === 'trained_1_improved_4' ? 'frozen 6-seat PPO (cash transfer)' : 'untrained placeholder'} + four heuristic bots. ${fresh.state.your_turn ? 'Your turn.' : 'Bots playing.'}`);
  });

  const refresh = () => {
    if (!session) return;
    return run(async () => {
      const latest = await tournamentApi.get(session.session_id);
      setSession(latest);
      setNotice(`Tournament refreshed. Revision ${latest.state.revision}.`);
    });
  };

  const doAction = (action: number) => {
    if (!session || !session.state.your_turn || session.state.tournament_over) return;
    const { session_id, state } = session;
    return run(async () => {
      try {
        const next = await tournamentApi.action(session_id, action, state.revision);
        setSession(next);
        setNotice(next.state.tournament_over ? 'Tournament finished. See champion and final placements.' : `Action accepted · hand ${next.state.hand_number} · revision ${next.state.revision}`);
      } catch (cause) {
        if (cause instanceof TournamentApiError && cause.status === 409) setSession(await tournamentApi.get(session_id));
        throw cause;
      }
    });
  };

  const doRaise = (amount: number) => {
    if (!session || !session.state.your_turn || session.state.tournament_over) return;
    const { session_id, state } = session;
    return run(async () => {
      try {
        const next = await tournamentApi.raiseTo(session_id, amount, state.revision);
        setSession(next);
        setNotice(`Raised to ${formatChips(amount)} chips · hand ${next.state.hand_number}`);
      } catch (cause) {
        if (cause instanceof TournamentApiError && cause.status === 409) setSession(await tournamentApi.get(session_id));
        throw cause;
      }
    });
  };

  const end = () => {
    if (!session) return;
    return run(async () => {
      await tournamentApi.delete(session.session_id);
      setSession(null);
      setHistory([]);
      setOpenedHand(null);
      setNotice('Tournament session closed. Its recorded history remains in the local SQLite archive.');
      setArchiveRefresh(x => x + 1);
    });
  };

  const state = session?.state ?? null;
  const accounting = state ? chipsAccounting(state) : null;
  const last = state?.last_hand ?? null;
  return <div className="t-app app-shell">
    <header className="global-header"><div className="global-logo"><span className="logo-hex">◇</span><span>HOLD<span className="logo-accent">'EM</span><small>TOURNAMENT ENVIRONMENT</small></span></div>
      <div className="header-route"><span>6-MAX ENVIRONMENT</span><span className="route-separator">/</span><strong>TOURNAMENT TABLE</strong></div>
      <div className="header-right"><span className="header-build">TOURNAMENT ENV / 1.6.1</span><span className={`server-indicator ${online ? 'server-online' : online === false ? 'server-offline' : ''}`}><span className="status-dot"/>{online ? 'API ONLINE' : online === false ? 'API OFFLINE' : 'CHECKING API'}</span></div>
    </header>
    <div className="t-workspace">
      <aside className="t-settings panel"><div className="panel-heading"><div><span className="eyebrow">01 / TOURNAMENT SETUP</span><h2>Control room</h2></div><span className="panel-heading-icon">⌘</span></div>
        <div className="t-settings-fields">
          <span className="section-label">SEATING / PHASE 1.6.1</span>
          <div className="t-lineup"><strong>SEAT 1 · HUMAN</strong><small>YOU · 1 AI SLOT · 4 IMPROVED BOTS</small></div>
          <p className="t-muted">Seat 2: choose Trained PPO with other Improved Bots or the untrained heuristic placeholder. The trained PPO was developed for 2–6-seat cash games; tournament transfer is experimental. Seats 3–6 remain four improved heuristic bots.</p>
          <label className="form-label" htmlFor="t-agent-lineup">SEAT 2 / TOURNAMENT AGENT</label>
          <select className="seed-input" id="t-agent-lineup" value={agentLineup} onChange={e => setAgentLineup(e.target.value as 'improved_4' | 'trained_1_improved_4')} disabled={working}>
            <option value="improved_4">Untrained heuristic placeholder</option>
            <option value="trained_1_improved_4">Trained PPO with other Improved Bots</option>
          </select>
          <p className="t-muted" aria-live="polite">{agentStatus?.file_found
            ? `Checkpoint configured: ${agentStatus.checkpoint_name} · ${agentStatus.mode} · ${agentStatus.device}. Format validated when session starts.`
            : 'Trained PPO can be selected, but playing requires a valid local checkpoint. Set HOLDEM_AI_TOURNAMENT_CHECKPOINT and restart FastAPI. A missing or incompatible checkpoint causes an explicit error; no Random Bot fallback.'}</p>
          <label className="form-label" htmlFor="t-seed">DECK SEED <span>OPTIONAL</span></label>
          <input className="seed-input" id="t-seed" value={seed} onChange={e => setSeed(e.target.value)} disabled={working} placeholder="Random" />
          <label className="form-label" htmlFor="t-stack">STARTING CHIPS / SEAT</label>
          <input className="seed-input" id="t-stack" type="number" value={startingStack} onChange={e => setStartingStack(e.target.value)} disabled={working} min="1" max="10000000" />
          <label className="form-label" htmlFor="t-button">INITIAL BUTTON SEAT</label>
          <select className="seed-input" id="t-button" value={button} onChange={e => setButton(e.target.value)} disabled={working}>{Array.from({length:6},(_,i)=><option key={i} value={i}>Seat {i + 1}{i===0?' · You':''}</option>)}</select>
          <label className="form-label" htmlFor="t-level">HANDS PER BLIND LEVEL</label>
          <input className="seed-input" id="t-level" type="number" min="1" max="100000" value={handsPerLevel} onChange={e => setHandsPerLevel(e.target.value)} disabled={working} />
          <button type="button" className="primary-button" disabled={working} onClick={start}>{working ? 'WORKING…' : session ? 'New tournament' : 'Start tournament'} <span aria-hidden="true">↗</span></button>
          {session && <div className="inline-actions"><button type="button" onClick={refresh} disabled={working}>↻ Refresh</button><button type="button" onClick={end} disabled={working}>End session</button></div>}
          <div className="section-divider" />
          <span className="section-label">LIVE STATUS</span>
          <div className="t-sidebar-stat"><span>Current hand</span><strong>#{state?.hand_number ?? '—'}</strong></div>
          <div className="t-sidebar-stat"><span>Completed hands</span><strong>{state?.hands_completed ?? 0}</strong></div>
          <div className="t-sidebar-stat"><span>Current level</span><strong>{state ? state.level_index + 1 : '—'}</strong></div>
          <div className="t-sidebar-stat"><span>Your stack</span><strong>{state ? formatChips(state.seats.find(s => s.seat === 0)?.stack ?? 0) : '—'}</strong></div>
          <div className="t-sidebar-stat"><span>Current pot</span><strong>{state ? formatChips(state.pot) : '—'}</strong></div>
          <TournamentProgressPanel state={state} history={history}/>
        </div>
        <div className="settings-footer"><span>ENGINE</span><strong>PHASE 4 · 6-SEAT KNOCKOUT</strong><small>Single-table · no rebuy · local only</small></div>
      </aside>
      <div className="t-center"><TournamentTable state={state}/>
        <div className={`session-notice ${error ? 't-error' : ''}`} role="status" aria-live="polite"><span className="notice-symbol">{error ? '!' : 'i'}</span><span>{error ?? notice}</span></div>
        {accounting && !accounting.valid && <p className="t-error-banner" role="alert">Chip conservation mismatch. Do not continue this session: {accounting.accounted} vs {accounting.total}.</p>}
        {last && <section className="t-last-hand panel"><div className="t-side-heading"><span className="eyebrow">MOST RECENT SETTLEMENT</span><strong>HAND #{last.hand_number}</strong></div>
          <div className="t-last-result"><span>Hero payoff <strong className={last.payoffs_chips[0]>=0?'t-profit':'t-danger'}>{formatSigned(last.payoffs_chips[0] ?? 0,0)} chips</strong></span><span>Eliminated <strong>{last.eliminated.length ? last.eliminated.map(s=>`Seat ${s+1}`).join(', ') : 'None'}</strong></span></div>
          <TournamentShowdown hand={last} />
          <p className="t-muted">Only players who reached a contested showdown reveal their cards. Folded hands are never exposed. The next hand deals automatically.</p>
        </section>}
        <section className="t-history panel"><div className="t-side-heading"><div><span className="eyebrow">TOURNAMENT JOURNAL</span><h2>Completed hands</h2></div><span className="t-muted">{state?.hands_completed ?? 0} total · latest 25</span></div>
          {historyError && <p className="t-error-banner">{historyError}</p>}
          {history.length === 0 && <p className="t-muted">Completed hands appear here automatically. Hand settlements are saved in the local SQLite archive.</p>}
          <div className="t-history-items">{history.map(h => <div className="t-history-entry" key={h.hand_number}>
            <button type="button" aria-expanded={openedHand===h.hand_number} onClick={()=>setOpenedHand(n=>n===h.hand_number?null:h.hand_number)}><span>HAND #{h.hand_number}</span><strong className={h.payoffs_chips[0]>=0?'t-profit':'t-danger'}>{formatSigned(h.payoffs_chips[0]??0,0)} chips</strong><small>{h.eliminated.length?'ELIMINATION':'SETTLED'}</small><span aria-hidden="true">{openedHand===h.hand_number?'−':'+'}</span></button>
            {openedHand===h.hand_number && <div className="t-history-detail"><p>Button: Seat {h.button+1} · Blinds {h.small_blind}/{h.big_blind}</p><p>Before: {h.stacks_before.map(formatChips).join(' · ')}</p><p>After: {h.stacks_after.map(formatChips).join(' · ')}</p>
              <TournamentShowdown hand={h} />
              <ol>{h.action_history.map((a,i)=><li key={i}>{seatLabel(a.seat, state)} · {eventText(a)}</li>)}</ol></div>}
          </div>)}</div>
        </section>
        <TournamentArchive refreshToken={archiveRefresh + (state?.hands_completed ?? 0)} />
      </div>
      <aside className="t-inspector panel"><div className="panel-heading"><div><span className="eyebrow">03 / ACTIONS & DATA</span><h2>Game inspector</h2></div><span className="panel-heading-icon">◎</span></div>
        <div className="t-side-section"><span className="section-label">CURRENT HAND</span>
          <div className="t-sidebar-stat"><span>Street</span><strong>{state?.street ?? '—'}</strong></div><div className="t-sidebar-stat"><span>Pot</span><strong>{state ? formatChips(state.pot) : '—'}</strong></div><div className="t-sidebar-stat"><span>To call</span><strong>{state?.to_call != null ? formatChips(state.to_call) : '—'}</strong></div><div className="t-sidebar-stat"><span>To act</span><strong>{state?.to_act != null ? seatLabel(state.to_act, state) : state?.tournament_over ? 'WINNER DETERMINED' : '—'}</strong></div>
          <span className="section-label t-activity-title">LIVE ACTIONS</span><div className="t-live-actions">{state?.action_log.length ? state.action_log.slice(-10).map((a,i)=><div key={`${state.revision}-${i}`}><strong>{seatLabel(a.seat, state)}</strong> {eventText(a)}</div>) : <p className="t-muted">No live actions this hand (completed hands are in the journal).</p>}</div>
        </div>
        <TournamentBetting state={state} working={working} onAction={doAction} onRaise={doRaise}/>
      </aside>
    </div>
    <footer className="global-footer"><span>HOLD'EM AI ◇ TOURNAMENT ENV</span><span>ENVIRONMENT ONLY · FROZEN ACTOR · NO RL TRAINING · 1.6.1</span></footer>
  </div>;
}
