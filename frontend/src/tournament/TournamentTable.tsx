import { PlayingCard } from '../components/PlayingCard';
import { formatChips } from '../lib/poker';
import { activeCount, chipsAccounting, policyDescription, seatLabel, SEATS } from './rules';
import type { TournamentSeat, TournamentState } from './types';

type Props = { state: TournamentState | null };

function Seat({ seat, state }: { seat: TournamentSeat | undefined; state: TournamentState | null }) {
  if (!seat) return null;
  const isHero = seat.seat === 0;
  // Zero sitting chips during an all-in do not imply elimination before settlement.
  const out = !!state && !state.alive_seats.includes(seat.seat);
  const badges = [
    state?.button === seat.seat ? 'D' : null,
    state?.small_blind_seat === seat.seat ? 'SB' : null,
    state?.big_blind_seat === seat.seat ? 'BB' : null,
  ].filter(Boolean);
  const showCards = !!state && !state.tournament_over && !out;
  return <div className={`t-seat t-seat-${seat.seat} ${isHero ? 't-seat-human' : ''} ${out ? 't-seat-out' : ''} ${state?.to_act === seat.seat ? 't-seat-acting' : ''}`}
    aria-label={`Seat ${seat.seat + 1} ${seatLabel(seat.seat, state)} chips ${seat.stack}`}>
    <div className="t-seat-head"><span className="t-seat-avatar">{isHero ? 'H' : seat.seat}</span><div className="t-seat-title"><strong>{seatLabel(seat.seat, state)}</strong><small>{out ? 'ELIMINATED' : state?.to_act === seat.seat ? 'TO ACT' : seat.status.toUpperCase()}</small></div></div>
    <span className="t-seat-policy" title={policyDescription(seat)}>{policyDescription(seat)}</span>
    <strong className="t-seat-chips">{formatChips(seat.stack)} <small>CHIPS</small></strong>
    <div className="t-seat-badges">
      {badges.map(b => <span className="t-seat-position" key={b}>{b}</span>)}
      {seat.rank != null && <span className="t-seat-position t-seat-rank">#{seat.rank}</span>}
      {seat.street_bet > 0 && <span className="t-seat-contribution">BET {formatChips(seat.street_bet)}</span>}
    </div>
    {showCards && <div className="t-seat-cards">
      {isHero && state.hero_cards.length === 2 ? state.hero_cards.map((card, index) => <PlayingCard key={index} card={card} compact />) : <><PlayingCard faceDown compact /><PlayingCard faceDown compact /></>}
    </div>}
  </div>;
}

export function TournamentTable({ state }: Props) {
  const accounting = state ? chipsAccounting(state) : null;
  return <section className="t-table-wrapper">
    <div className="table-header t-heading"><div><span className="eyebrow">THE ARENA / 6 SEATS</span><h1>6-Max <em>Tournament</em></h1><p>Single-table knockout · continuous stacks · no rebuy</p></div>
      <span className={`phase-pill ${state ? 'phase-live' : ''}`}><span className="status-dot" />{state ? state.tournament_over ? 'TOURNAMENT OVER' : 'LIVE / LOCAL' : 'WAITING FOR SESSION'}</span></div>
    <div className="t-arena" aria-label="Six player tournament table">
      <div className="t-felt"><div className="t-felt-inner">
        <span className="t-felt-watermark">◇ 6-MAX HOLD'EM</span>
        <span className="t-street">{state?.street?.toUpperCase() ?? 'TOURNAMENT'}</span>
        <div className="t-pot"><span className="t-chip-icon">◉</span><small>LIVE POT</small><strong>{formatChips(state?.pot ?? 0)}</strong><small>CHIPS</small></div>
        <div className="t-board" aria-label="Community cards">{SEATS.slice(0, 5).map(index => <PlayingCard key={index} card={state?.board[index]} placeholder={!state?.board[index]}/>)}</div>
      </div></div>
      {state ? SEATS.map(i => <Seat key={i} seat={state.seats.find(s => s.seat === i)} state={state} />) : <div className="t-empty">Your 6-player table is ready.<span>Create a tournament to deal the first hand.</span></div>}
    </div>
    <div className="t-table-stats">
      <div><small>HAND</small><strong>#{state ? state.hand_number : '—'}</strong></div>
      <div><small>BLINDS · LEVEL {state ? state.level_index + 1 : '—'}</small><strong>{state ? `${formatChips(state.small_blind)} / ${formatChips(state.big_blind)}` : '—'}</strong></div>
      <div><small>REMAINING / ALL-IN</small><strong>{state ? activeCount(state) : '—'} / 6</strong></div>
      <div><small>CHIP ACCOUNTING</small><strong className={accounting && !accounting.valid ? 't-danger' : ''}>{accounting ? `${formatChips(accounting.accounted)} / ${formatChips(accounting.total)}` : '—'}</strong></div>
    </div>
    {state?.tournament_over && <div className="t-finish-banner t-champion-banner" role="status">
      <span className="t-champion-trophy" aria-hidden="true">★</span>
      <div><small>TOURNAMENT COMPLETE · OFFICIAL 1ST PLACE</small><strong>{state.winner_seat === 0 ? 'YOU ARE THE CHAMPION' : `CHAMPION · ${seatLabel(state.winner_seat ?? 0, state)}`}</strong><span>All placements are final. Start a new tournament to play again.</span></div>
    </div>}
    {state?.human_eliminated && !state.tournament_over && <div className="t-finish-banner" role="status">YOU WERE ELIMINATED · The remaining heuristic and placeholder bots continued automatically to the tournament result.</div>}
  </section>;
}
