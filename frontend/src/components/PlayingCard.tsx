import { decodeCard } from '../lib/poker';

type Props = { card?: string | null; faceDown?: boolean; placeholder?: boolean; compact?: boolean; label?: string };
export function PlayingCard({ card, faceDown = false, placeholder = false, compact = false, label }: Props) {
  const parsed = card ? decodeCard(card) : null;
  if (placeholder || (!card && !faceDown)) {
    return <div className={`playing-card card-placeholder ${compact ? 'compact' : ''}`} aria-label={label ?? 'Empty card slot'}>
      <span>✧</span>
    </div>;
  }
  if (faceDown) {
    return <div className={`playing-card card-back ${compact ? 'compact' : ''}`} aria-label={label ?? 'Opponent card hidden'}>
      <div className="card-back-pattern"><span>♠</span></div>
    </div>;
  }
  return <div className={`playing-card card-face ${parsed?.red ? 'card-red' : 'card-black'} ${compact ? 'compact' : ''}`} aria-label={parsed ? `${parsed.rank} of ${parsed.label}` : 'Invalid card'}>
    <span className="card-corner"><strong>{parsed?.rank ?? '?'}</strong><small>{parsed?.glyph ?? '?'}</small></span>
    <span className="card-center">{parsed?.glyph ?? '?'}</span>
    <span className="card-corner bottom-corner"><strong>{parsed?.rank ?? '?'}</strong><small>{parsed?.glyph ?? '?'}</small></span>
  </div>;
}
