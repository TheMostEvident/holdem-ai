import { useEffect, useState } from 'react';
import type { CompletedHand, SessionStats } from '../api/types';
import { hunlApi } from '../api/client';
import { chartPath, winRate } from '../lib/analytics';
import { actionDescription, formatSigned, streetName } from '../lib/poker';
import { PlayingCard } from './PlayingCard';

const PAGE_SIZE = 30;

type Props = { sessionId: string | null; completedHands: number };

function HandDetail({ record }: { record: CompletedHand }) {
  return <div className="ledger-detail" aria-label={`Hand ${record.hand_number} details`}>
    <div className="ledger-cards-row">
      <div><small>YOUR HAND · Started {record.starting_stacks[0].toLocaleString()} chips</small><div className="ledger-cards">{record.hero_cards.map((card, i) => <PlayingCard key={i} card={card} compact />)}</div></div>
      <div><small>BOARD</small><div className="ledger-cards">{record.board.length
        ? record.board.map((card, i) => <PlayingCard key={i} card={card} compact />)
        : <span className="ledger-muted">Not dealt</span>}</div></div>
    </div>
    <div className="ledger-cards-row">
      <div><small>AI HAND · Started {record.starting_stacks[1].toLocaleString()} chips</small><div className="ledger-cards">{record.result.showdown && record.result.opponent_cards
        ? record.result.opponent_cards.map((card, i) => <PlayingCard key={i} card={card} compact />)
        : <span className="ledger-muted">Hidden · no showdown</span>}</div></div>
      <div className="ledger-payoff"><small>HERO NET</small><strong>{formatSigned(record.hero_net_chips, 0)} chips</strong><span>{formatSigned(record.hero_net_bb)} BB</span></div>
    </div>
    <div className="ledger-timeline">
      <small>PUBLIC ACTION SEQUENCE</small>
      {record.action_log.length
        ? <ol>{record.action_log.map((item, i) => <li key={i}><span>{streetName(item.street)}</span><strong>{item.seat === 0 ? 'You' : 'AI'}</strong> {actionDescription(item)}</li>)}</ol>
        : <p className="ledger-muted">No player decisions (automatic terminal hand).</p>}
    </div>
  </div>;
}

export function SessionAnalyticsPanel({ sessionId, completedHands }: Props) {
  const [stats, setStats] = useState<SessionStats | null>(null);
  const [history, setHistory] = useState<CompletedHand[]>([]);
  const [total, setTotal] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [olderLoading, setOlderLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) { setStats(null); setHistory([]); setTotal(0); setHasMore(false); setSelected(null); return; }
    let cancelled = false;
    setLoading(true); setError(null);
    Promise.all([hunlApi.stats(sessionId), hunlApi.history(sessionId, 0, PAGE_SIZE)])
      .then(([summary, result]) => {
        if (cancelled) return;
        setStats(summary);
        setHistory(result.items);
        setTotal(result.total);
        setHasMore(result.has_more);
        setSelected(current => current !== null && result.items.some(h => h.hand_number === current) ? current : result.items[0]?.hand_number ?? null);
      })
      .catch(reason => { if (!cancelled) setError(reason instanceof Error ? reason.message : 'Analytics unavailable'); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [sessionId, completedHands]);

  const loadOlder = async () => {
    if (!sessionId || olderLoading || loading || !hasMore) return;
    setOlderLoading(true); setError(null);
    try {
      const result = await hunlApi.history(sessionId, history.length, PAGE_SIZE);
      setHistory(current => {
        const existing = new Set(current.map(h => h.hand_number));
        return [...current, ...result.items.filter(h => !existing.has(h.hand_number))];
      });
      setHasMore(result.has_more);
      setTotal(result.total);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Unable to load older hands');
    } finally { setOlderLoading(false); }
  };

  if (!sessionId) return <section className="analytics-panel panel" aria-label="Session analytics">
    <div className="analytics-heading"><span className="eyebrow">03 / PERFORMANCE</span><h2>Session laboratory</h2></div>
    <p className="analytics-placeholder">Start a live game to track decisions, hand outcomes, and cumulative BB results. Preview data is excluded.</p>
  </section>;

  const detail = history.find(row => row.hand_number === selected);
  const path = chartPath(history);
  return <section className="analytics-panel panel" aria-label="Session analytics">
    <div className="analytics-heading"><div><span className="eyebrow">03 / PERFORMANCE</span><h2>Session laboratory</h2></div><span className="analytics-count">{stats?.completed_hands ?? completedHands} completed hands</span></div>
    {error && <p className="analytics-error" role="alert">{error}</p>}
    {loading && !stats && <p className="analytics-placeholder" role="status">Loading local session analytics…</p>}
    <div className="analytics-kpis">
      <div><small>W / L / EVEN</small><strong>{stats ? `${stats.wins} / ${stats.losses} / ${stats.break_even}` : '—'}</strong></div>
      <div><small>WIN RATE</small><strong>{stats ? `${winRate(stats).toFixed(1)}%` : '—'}</strong></div>
      <div><small>SHOWDOWNS</small><strong>{stats?.showdowns ?? '—'}</strong></div>
      <div><small>MAX DRAWDOWN</small><strong>{stats ? `${stats.max_drawdown_bb.toFixed(2)} BB` : '—'}</strong></div>
    </div>
    <div className="analytics-chart-block">
      <div className="analytics-chart-title"><div><strong>Cumulative result</strong><span>Net BB by completed hand · last {history.length} displayed</span></div><b>{stats ? `${formatSigned(stats.net_bb)} BB` : '—'}</b></div>
      {path ? <svg className="analytics-chart" viewBox="0 0 600 152" preserveAspectRatio="none" role="img" aria-label="Line chart of cumulative net big blinds by hand">
        <line x1="16" y1="134" x2="584" y2="134" stroke="#385244" strokeWidth="1" />
        <path d={path} fill="none" stroke="#c3daa0" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg> : <p className="analytics-placeholder">The chart starts after your first completed hand.</p>}
      <div className="analytics-chart-caption"><span>Observed bb/100: <b>{stats ? formatSigned(stats.bb_per_100) : '—'}</b></span><span>95% hand-level CI: {stats?.ci95_lower_bb100 != null && stats.ci95_upper_bb100 != null ? `[${stats.ci95_lower_bb100.toFixed(1)}, ${stats.ci95_upper_bb100.toFixed(1)}]` : 'N/A (requires ≥2 hands)'}</span></div>
      <p className="analytics-disclaimer">Descriptive session data only. The simple CI assumes independent hands; it does not establish exploitability, GTO strength, or performance against other players.</p>
    </div>
    <div className="analytics-ledger-head"><div><span className="eyebrow">04 / GAME JOURNAL</span><h3>Completed hands</h3></div><span>{history.length} of {total} retained records</span></div>
    {!history.length && <p className="analytics-placeholder">Finish a hand to add it to the session journal.</p>}
    <div className="analytics-history-list" role="list" aria-label="Completed hands history">
      {history.map(item => <div key={item.hand_number} className="ledger-item" role="listitem">
        <button className={`ledger-item-toggle ${selected === item.hand_number ? 'selected' : ''}`} onClick={() => setSelected(selected === item.hand_number ? null : item.hand_number)} aria-expanded={selected === item.hand_number}>
          <span className="ledger-item-num">#{String(item.hand_number).padStart(3, '0')}</span>
          <span className="ledger-item-meta">{streetName(item.street)} · {item.result.showdown ? 'Showdown' : 'No showdown'}</span>
          <strong className={item.hero_net_chips >= 0 ? 'ledger-win' : 'ledger-loss'}>{formatSigned(item.hero_net_bb)} BB</strong>
          <span aria-hidden="true">{selected === item.hand_number ? '−' : '+'}</span>
        </button>
        {selected === item.hand_number && detail && <HandDetail record={detail} />}
      </div>)}
    </div>
    {hasMore && <button className="analytics-more" onClick={loadOlder} disabled={olderLoading || loading}>{olderLoading ? 'Loading…' : 'Load earlier hands'}</button>}
    {stats?.records_truncated && <p className="analytics-disclaimer">Older detailed records were pruned to bound local memory use; aggregate statistics still include all completed hands.</p>}
  </section>;
}
