import { formatChips } from '../lib/poker';
import { blindProgressText, recentKnockouts, tournamentStandings } from './standings';
import { seatLabel } from './rules';
import type { TournamentHand, TournamentState } from './types';

type Props = { state: TournamentState | null; history: TournamentHand[] };

export function TournamentProgressPanel({ state, history }: Props) {
  if (!state) return <section className="t-progress-panel" aria-label="Tournament standings"><span className="section-label">TOURNAMENT PROGRESS</span><p className="t-muted">Start a tournament to view the live standings.</p></section>;
  const standings = tournamentStandings(state);
  const knockouts = recentKnockouts(history);
  const p = state.blind_progress;
  const leader = standings.find(s => s.alive);
  return <section className="t-progress-panel" aria-label="Tournament standings">
    <div className="t-progress-head"><span className="section-label">TOURNAMENT STANDINGS</span><strong>{state.alive_seats.length} / 6 REMAINING</strong></div>
    <p className="t-muted">Chip ranks use the last settled hand, not chips temporarily committed to the pot. Placements become official on elimination.</p>
    <div className="t-standings" role="list" aria-label="Current rankings">
      {standings.map(row => <div className={`t-standing-row ${row.seat === 0 ? 't-standing-human' : ''} ${!row.alive ? 't-standing-out' : ''}`} role="listitem" key={row.seat}>
        <span className={`t-standing-position ${row.champion ? 't-champion-position' : ''}`}>{row.champion ? '★' : row.officialRank != null ? `#${row.officialRank}` : `~${row.chipRank}`}</span>
        <div className="t-standing-identity"><strong>{seatLabel(row.seat,state)}</strong><small>{row.champion ? 'CHAMPION · OFFICIAL #1' : row.alive ? 'PROVISIONAL CHIP POSITION' : 'ELIMINATED · FINAL PLACE'}</small></div>
        <strong className="t-standing-stack">{formatChips(row.stack)}</strong>
      </div>)}
    </div>
    <div className="t-blind-panel" aria-label="Blind level progress">
      <div className="t-blind-heading"><span className="section-label">BLIND STRUCTURE</span><strong>LEVEL {state.level_index + 1}</strong></div>
      <strong className="t-blind-current">{formatChips(state.small_blind)} / {formatChips(state.big_blind)} <small>SB / BB</small></strong>
      <p className="t-blind-next">{blindProgressText(state)}</p>
      {!state.tournament_over && !p.final_level && p.hand_in_level != null && <>
        <div className="t-level-track" role="progressbar" aria-label="Current blind level hand progress" aria-valuemin={0} aria-valuemax={p.hands_per_level} aria-valuenow={p.hand_in_level}><span style={{width:`${100*p.hand_in_level/p.hands_per_level}%`}}/></div>
        <span className="t-level-count">Hand {p.hand_in_level} of {p.hands_per_level} at this level</span>
      </>}
    </div>
    {leader && <p className="t-leading-info">{state.tournament_over ? 'WINNER' : 'CHIP LEADER'} · {seatLabel(leader.seat,state)} · {formatChips(leader.stack)} chips</p>}
    {knockouts.length > 0 && <div className="t-knockout-list"><span className="section-label">RECENT ELIMINATIONS</span>{knockouts.map(k => <div key={`${k.hand}-${k.seat}`}>Hand {k.hand} · <strong>{seatLabel(k.seat,state)}</strong> eliminated</div>)}</div>}
  </section>;
}
