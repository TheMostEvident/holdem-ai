import { useCallback, useEffect, useRef, useState } from 'react';
import type { BotCatalog, BotName, GameResponse } from './api/types';
import { GameApiError, hunlApi } from './api/client';
import { previewStates, type PreviewKey } from './fixtures/preview';
import { Inspector } from './components/Inspector';
import { PokerTable } from './components/PokerTable';
import { SessionAnalyticsPanel } from './components/SessionAnalyticsPanel';
import { formatSigned } from './lib/poker';

const BOT_LABELS: Record<BotName, string> = {
  random: 'Random policy',
  calling_station: 'Calling station',
  tight: 'Tight policy',
  pot_pressure: 'Pot pressure',
  ppo: 'Trained PPO',
  qboost: 'Trained Q-Boost PPO',
};
const PREVIEW_LABELS: { key: PreviewKey; text: string }[] = [
  { key: 'preflop', text: 'Pre-flop' },
  { key: 'flop', text: 'Flop' },
  { key: 'turn', text: 'Turn' },
  { key: 'river', text: 'River' },
  { key: 'showdown', text: 'Showdown' },
  { key: 'fold', text: 'Fold finish' },
];

export default function App() {
  const [bot, setBot] = useState<BotName>('random');
  const [workspaceMode, setWorkspaceMode] = useState<'play' | 'training'>('play');
  const [activeBot, setActiveBot] = useState<BotName>('random');
  const [catalog, setCatalog] = useState<BotCatalog | null>(null);
  const [seed, setSeed] = useState('42');
  const [session, setSession] = useState<GameResponse | null>(null);
  const [preview, setPreview] = useState(true);
  const [previewKey, setPreviewKey] = useState<PreviewKey>('flop');
  const [working, setWorking] = useState(false);
  const flight = useRef(false);
  const [online, setOnline] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState('Preview mode. Start a local session to use the real HUNL engine.');

  useEffect(() => {
    let active = true;
    Promise.all([hunlApi.health(), hunlApi.bots()])
      .then(([, info]) => { if (active) { setOnline(true); setCatalog(info); } })
      .catch(() => { if (active) setOnline(false); });
    return () => { active = false; };
  }, []);

  const run = useCallback(async (work: () => Promise<void>) => {
    if (flight.current) return;
    flight.current = true;
    setWorking(true); setError(null);
    try { await work(); setOnline(true); }
    catch (cause) {
      const text = cause instanceof Error ? cause.message : 'Unknown connection error';
      setError(text);
      if (cause instanceof GameApiError && cause.status === 0) setOnline(false);
    } finally { setWorking(false); flight.current = false; }
  }, []);

  const startSession = () => run(async () => {
    if (workspaceMode !== 'play') throw new Error('Training mode is CLI-only; switch to Play to launch a table.');
    const trimmed = seed.trim();
    const parsedSeed = trimmed === '' ? null : Number(trimmed);
    if (parsedSeed !== null && (!Number.isSafeInteger(parsedSeed) || Math.abs(parsedSeed) > 2147483647)) {
      throw new Error('Seed must be a safe integer between -2147483647 and 2147483647.');
    }
    // Only retire the old game AFTER the new checkpoint/session is valid.
    // Failed model load must not discard the current hand.
    const result = await hunlApi.create({ bot, seed: parsedSeed, big_blind: 100, small_blind: 50, stack_bb: 200, stack_mode: 'continuous' });
    if (session) { try { await hunlApi.delete(session.session_id); } catch { /* new game is still usable */ } }
    setSession(result); setActiveBot(bot); setPreview(false);
    setNotice(`Hand ${result.state.hand_number} dealt. ${result.state.your_turn ? 'It is your turn.' : 'AI turn.'}`);
  });

  const refresh = () => {
    if (!session) return;
    return run(async () => {
      const response = await hunlApi.get(session.session_id);
      setSession(response);
      setNotice(`Session refreshed · revision ${response.state.revision}`);
    });
  };

  const submitAction = (actionIndex: number) => {
    if (!session || preview || !session.state.your_turn) return;
    const expected = session.state.revision;
    return run(async () => {
      try {
        const response = await hunlApi.action(session.session_id, actionIndex, expected);
        setSession(response);
        setNotice(response.state.phase === 'complete'
          ? response.state.match_over
            ? `Match over. ${response.state.winner_seat === response.state.hero_seat ? 'You won!' : 'Bot won.'} Start a new session.`
            : `Hand ${response.state.hand_number} complete. Deal the next hand.`
          : `Action accepted. Hand ${response.state.hand_number} · ${response.state.street}.`);
      } catch (cause) {
        if (cause instanceof GameApiError && cause.status === 409) {
          const latest = await hunlApi.get(session.session_id);
          setSession(latest);
        }
        throw cause;
      }
    });
  };

  const submitRaiseTo = (amount: number) => {
    if (!session || preview || !session.state.your_turn) return;
    const expected = session.state.revision;
    return run(async () => {
      try {
        const response = await hunlApi.raiseTo(session.session_id, amount, expected);
        setSession(response);
        setNotice(response.state.phase === 'complete'
          ? response.state.match_over
            ? `Match over. ${response.state.winner_seat === response.state.hero_seat ? 'You won!' : 'Bot won.'} Start a new session.`
            : `Hand ${response.state.hand_number} complete. Deal the next hand.`
          : `Raised to ${amount} chips. ${response.state.street} in progress.`);
      } catch (cause) {
        if (cause instanceof GameApiError && cause.status === 409) {
          const latest = await hunlApi.get(session.session_id);
          setSession(latest);
        }
        throw cause;
      }
    });
  };

  const nextHand = () => {
    if (!session || session.state.phase !== 'complete' || session.state.match_over || preview) return;
    return run(async () => {
      const response = await hunlApi.nextHand(session.session_id, session.state.revision);
      setSession(response);
      setNotice(response.state.match_over
        ? `Match over. ${response.state.winner_seat === response.state.hero_seat ? 'You won!' : 'Bot won.'} Start a new session.`
        : response.state.phase === 'complete'
          ? `Hand ${response.state.hand_number} automatically settled. Deal the next hand.`
          : `Hand ${response.state.hand_number} dealt.`);
    });
  };

  const endSession = () => {
    if (!session) return;
    return run(async () => {
      await hunlApi.delete(session.session_id);
      setSession(null); setPreview(true);
      setNotice('Session ended. Previewing a sample table.');
    });
  };

  const shown = preview ? previewStates[previewKey] : session?.state ?? null;
  const botName = preview ? 'AI · SAMPLE' : activeBot === 'random' ? 'RANDOM BOT' : BOT_LABELS[activeBot].toUpperCase();
  const stats = session?.state;

  return <div className="app-shell">
    <header className="global-header">
      <div className="global-logo" aria-label="holdem-ai home"><span className="logo-hex">◇</span><span>HOLD<span className="logo-accent">'EM</span><small>AI RESEARCH LAB</small></span></div>
      <div className="header-route"><span>RESEARCH ENVIRONMENT</span><span className="route-separator">/</span><strong>HUNL TABLE</strong></div>
      <div className="header-right"><span className="header-build">GUI / BUILD 1.6</span><span className={`server-indicator ${online === true ? 'server-online' : online === false ? 'server-offline' : ''}`}><span className="status-dot" />{online === true ? 'API ONLINE' : online === false ? 'API OFFLINE' : 'CHECKING API'}</span></div>
    </header>

    <div className="workspace">
      <aside className="settings-panel panel">
        <div className="panel-heading"><div><span className="eyebrow">01 / CONTROL ROOM</span><h2>Game setup</h2></div><span className="panel-heading-icon">⌘</span></div>
        <div className="settings-content">
          <div className="section-label">GAME FORMAT</div>
          <div className="entry-modes" role="group" aria-label="Choose application mode">
            <button type="button" className={`mode-choice ${workspaceMode === 'play' ? 'selected' : ''}`} aria-pressed={workspaceMode === 'play'} onClick={() => setWorkspaceMode('play')}><span className="mode-choice-icon">P</span><span><strong>Play Mode</strong><small>Continuous stacks · Heads-Up NLHE</small></span>{workspaceMode === 'play' && <span className="mode-check">✓</span>}</button>
            <button type="button" className={`mode-choice ${workspaceMode === 'training' ? 'selected' : ''}`} aria-pressed={workspaceMode === 'training'} onClick={() => setWorkspaceMode('training')}><span className="mode-choice-icon">T</span><span><strong>Training / Research</strong><small>Independent 200BB · Python CLI</small></span>{workspaceMode === 'training' && <span className="mode-check">✓</span>}</button>
            </div>
          {workspaceMode === 'training' && <p className="training-cli-note" role="status">Training remains in the Python CLI, not this browser game. Use the Phase 15/16 scripts from the repository root. Select Play Mode to start the continuous-stack table.</p>}
          <div className="settings-grid"><div><span>{workspaceMode === 'play' ? 'Initial stack' : 'Hand stack'}</span><strong>200 BB</strong></div><div><span>Blinds</span><strong>50 / 100</strong></div></div>
          <p className="field-description">{workspaceMode === 'play' ? 'Stack balances carry over after every hand. A bust ends the match.' : 'Training hands each start with 200BB; the RL pipeline is unchanged.'}</p>
          <div className="section-divider" />
          <label className="form-label" htmlFor="bot-select">OPPONENT POLICY</label>
          <div className="select-wrap"><select id="bot-select" value={bot} disabled={working} onChange={(e) => setBot(e.target.value as BotName)}>{Object.entries(BOT_LABELS).map(([key, value]) => <option key={key} value={key}>{value}{catalog?.bots.find(x => x.id === key)?.available === false ? ' — Checkpoint required' : ''}</option>)}</select><span>⌄</span></div>
          <p className="field-description" aria-live="polite">{catalog?.bots.find(x => x.id === bot)?.message ?? 'PPO / Q-Boost require trusted local checkpoint configuration.'} {catalog?.bots.find(x => x.id === bot)?.available === false ? `To play with this model, configure ${bot === 'ppo' ? 'HOLDEM_AI_PPO_CHECKPOINT' : 'HOLDEM_AI_QBOOST_CHECKPOINT'} on the FastAPI server and restart it. The selection is allowed, but creating a neural game without a valid checkpoint will fail explicitly.` : ''} {catalog ? `Inference: ${catalog.device.toUpperCase()} · ${catalog.policy_mode}` : ''}</p>
          <label className="form-label seed-label" htmlFor="seed-input">DECK SEED <span>OPTIONAL</span></label>
          <input id="seed-input" className="seed-input" value={seed} disabled={working} placeholder="Random" inputMode="numeric" onChange={(e) => setSeed(e.target.value)} />
          <p className="field-description">Use the same seed to reproduce a card shuffle. Clear for a fresh random seed.</p>
          <button className="primary-button" disabled={working || workspaceMode !== 'play'} onClick={startSession}>{working ? 'WORKING…' : session ? 'Start new session' : 'Deal a new game'} <span aria-hidden="true">↗</span></button>
          {session && <div className="inline-actions"><button disabled={working} onClick={refresh}>↻ Refresh</button><button disabled={working} onClick={endSession}>End session</button></div>}
          <div className="section-divider" />
          <div className="preview-heading"><span className="section-label">VIEW MODE</span><span className="preview-help">UI showcase</span></div>
          <div className="view-switch"><button className={!preview ? 'on' : ''} disabled={!session || working} onClick={() => setPreview(false)}>Live table</button><button className={preview ? 'on' : ''} disabled={working} onClick={() => setPreview(true)}>Preview</button></div>
          {preview && <div className="preview-streets" aria-label="Design preview street selector">{PREVIEW_LABELS.map(item => <button className={previewKey === item.key ? 'active' : ''} key={item.key} onClick={() => setPreviewKey(item.key)}>{item.text}</button>)}</div>}
          <div className="preview-disclaimer">{preview ? 'Sample fixture only. Cards, chip totals, and outcomes are illustrative — not a real game.' : 'Live state comes from the local FastAPI engine. Actions are validated by the engine.'}</div>
        </div>
        <div className="settings-footer"><span>INFERENCE</span><strong>LOCAL / INFERENCE READY</strong><small>Frozen Actor checkpoints · 7-action neural policy</small></div>
      </aside>

      <div className="center-content">
        <PokerTable state={shown} preview={preview} botName={botName} />
        <section className="session-notice" aria-live="polite">
          <span className={`notice-symbol ${error ? 'notice-error' : ''}`}>{error ? '!' : 'i'}</span>
          <span>{error ?? notice}</span>
          {preview && <strong>PREVIEW · NOT LIVE</strong>}
        </section>
        <div className="stats-row">
          <div className="stat-tile"><small>COMPLETED HANDS</small><strong>{stats?.completed_hands ?? 0}</strong><span>Since session start</span></div>
          <div className="stat-tile"><small>NET RESULT</small><strong className={(stats?.cumulative_bb ?? 0) >= 0 ? 'stat-positive' : 'stat-negative'}>{formatSigned(stats?.cumulative_bb ?? 0)} <em>BB</em></strong><span>{stats?.stack_mode === 'continuous' ? `Stacks carry over · Your stack ${stats.seats[stats.hero_seat]?.stack.toLocaleString() ?? '—'}` : 'Training hands reset every hand'}</span></div>
          <div className="stat-tile"><small>PERFORMANCE</small><strong>{formatSigned(stats?.bb_per_100 ?? 0)} <em>bb/100</em></strong><span>Sample size matters</span></div>
        </div>
        <SessionAnalyticsPanel key={session?.session_id ?? "inactive"} sessionId={!preview ? session?.session_id ?? null : null} completedHands={session?.state.completed_hands ?? 0} />
      </div>
      <Inspector state={shown} preview={preview} working={working} onNext={nextHand} onAction={submitAction} onRaiseTo={submitRaiseTo} />
    </div>
    <footer className="global-footer"><span>HOLD'EM AI <b>◇</b> RESEARCH INTERFACE</span><span>LOCAL DEVELOPMENT ONLY <span className="footer-dot">·</span> V1.6</span></footer>
  </div>;
}
