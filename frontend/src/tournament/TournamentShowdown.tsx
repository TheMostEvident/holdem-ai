import { PlayingCard } from '../components/PlayingCard';
import type { TournamentHand } from './types';

/** Public only after completed hand settlement. Never show folded seats' holes. */
export function TournamentShowdown({ hand }: { hand: TournamentHand }) {
  const contenders = hand.showdown_cards ?? [];
  return <div className="t-showdown" aria-label={`Showdown cards for hand ${hand.hand_number}`}>
    <div className="t-showdown-heading">{contenders.length >= 2 ? 'SHOWDOWN · REVEALED HANDS' : 'NO CARD REVEAL'}</div>
    {contenders.length >= 2 ? <>
      <div className="t-showdown-board"><span>FINAL BOARD</span><div className="t-showdown-cards">{hand.board.map((card, i) => <PlayingCard key={i} card={card} compact />)}</div></div>
      <div className="t-showdown-players">
        {contenders.map(({seat, cards}) => <div className="t-showdown-player" key={seat}>
          <div className="t-showdown-label"><strong>{seat === 0 ? "HUMAN" : `SEAT ${seat + 1}`}</strong><small>{hand.awards[seat] > 0 ? `Award +${hand.awards[seat]}` : 'Showed cards'}</small></div>
          <div className="t-showdown-cards">{cards.map((card,i) => <PlayingCard key={i} card={card} compact />)}</div>
        </div>)}
      </div>
    </> : <p className="t-muted">The pot was won without a contested showdown, or these cards were not recorded in an earlier archive. Folded hands remain hidden.</p>}
  </div>;
}
