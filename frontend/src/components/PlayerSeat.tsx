import type { SeatView } from '../api/types';
import { formatChips } from '../lib/poker';
import { PlayingCard } from './PlayingCard';

type Props = {
  seat: SeatView;
  label: string;
  isButton: boolean;
  isActing: boolean;
  holeCards: string[] | null;
  isHero: boolean;
  showCards: boolean;
};
export function PlayerSeat({ seat, label, isButton, isActing, holeCards, isHero, showCards }: Props) {
  return <section className={`seat ${isHero ? 'seat-hero' : 'seat-bot'} ${isActing ? 'seat-active' : ''} ${seat.status === 'folded' ? 'seat-folded' : ''}`} aria-label={`${label} player seat`}>
    <div className="seat-identity">
      <div className={`avatar ${isHero ? 'human-avatar' : 'bot-avatar'}`}>{isHero ? 'YOU' : 'AI'}</div>
      <div className="seat-meta">
        <div className="seat-name-row"><strong>{label}</strong>{isButton && <span className="dealer-mark" title="Dealer button">D</span>}</div>
        <div className="seat-stack"><span className="stack-indicator" /> {formatChips(seat.stack)} <small>chips</small></div>
      </div>
      {isActing && <div className="thinking-indicator">ACTING</div>}
    </div>
    {showCards && <div className="seat-hole-cards">
      {(holeCards ?? []).length === 2
        ? holeCards!.map((c, i) => <PlayingCard compact key={`${c}-${i}`} card={c} />)
        : [0, 1].map((i) => <PlayingCard compact key={i} faceDown label="Opponent hole card hidden" />)}
    </div>}
    {seat.street_bet > 0 && <div className="seat-bet"><span className="tiny-chip" /> Bet {formatChips(seat.street_bet)}</div>}
    {seat.status !== 'active' && <span className="seat-status">{seat.status.toUpperCase()}</span>}
  </section>;
}
