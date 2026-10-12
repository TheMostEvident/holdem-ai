import type { GameView } from '../api/types';
import { formatChips, opponentCardsToRender, streetName } from '../lib/poker';
import { PlayingCard } from './PlayingCard';
import { PlayerSeat } from './PlayerSeat';

type Props = { state: GameView | null; preview: boolean; botName: string };
export function PokerTable({ state, preview, botName }: Props) {
  const hero = state?.seats.find((s) => s.seat === state.hero_seat);
  const villain = state?.seats.find((s) => s.seat !== state.hero_seat);
  const opponentCards = state ? opponentCardsToRender(state) : null;
  return <main className="table-column">
    <div className="table-header">
      <div>
        <span className="eyebrow">THE ARENA <span className="eyebrow-line" /></span>
        <h1>Heads-Up <em>No-Limit</em></h1>
        <p>A private table. A live decision. One opponent.</p>
      </div>
      <span className={`phase-pill ${preview ? 'phase-preview' : state ? 'phase-live' : ''}`}>
        <span className="status-dot" /> {preview ? 'DESIGN PREVIEW' : state ? 'LOCAL SESSION' : 'AWAITING SESSION'}
      </span>
    </div>
    <div className="table-scene">
      <div className="scene-ambient scene-ambient-left" />
      <div className="scene-ambient scene-ambient-right" />
      <div className="felt-rim"><div className="felt-inner">
        <div className="felt-sheen" />
        <div className="felt-brand"><span className="brand-diamond">◆</span> HUNL <span className="felt-brand-sub">{state?.stack_mode === 'continuous' ? 'CONTINUOUS STACKS' : 'RESEARCH TABLE / 200 BB'}</span></div>
        <div className="felt-stage">{state ? streetName(state.street).toUpperCase() : 'READY TO DEAL'}</div>
        <div className="felt-pot">
          <div className="pot-chip-stack"><span /><span /><span /></div>
          <div className="pot-copy"><small>TOTAL POT</small><strong>{formatChips(state?.pot ?? 0)}</strong><span>CHIPS</span></div>
        </div>
        <div className="community-cards" aria-label="Community cards">
          {Array.from({ length: 5 }, (_, i) => state?.board[i] ? <PlayingCard key={i} card={state.board[i]} /> : <PlayingCard key={i} placeholder />)}
        </div>
        <div className="felt-center-decoration" />
      </div></div>
      {villain && <PlayerSeat seat={villain} label={botName} isButton={state!.button === villain.seat}
        isActing={state!.to_act === villain.seat} holeCards={opponentCards} isHero={false} showCards />}
      {hero && <PlayerSeat seat={hero} label="You" isButton={state!.button === hero.seat}
        isActing={state!.to_act === hero.seat} holeCards={state!.hero_cards} isHero showCards />}
      {!state && <div className="table-blank-message"><strong>Your table is waiting.</strong><span>Launch a session to deal your first hand.</span></div>}
      <div className="scene-footer-left">{state?.stack_mode === 'continuous' ? 'HUNL PLAY' : 'HUNL 200'} / {state ? `HAND ${String(state.hand_number).padStart(3, '0')}` : 'SESSION OFFLINE'}</div>
      <div className="scene-footer-right">NO RAKE · LOCAL SIMULATION</div>
    </div>
    <div className="table-bottom-strip">
      <div className="strip-cell"><small>BLINDS</small><strong>{state ? `${formatChips(state.small_blind)} / ${formatChips(state.big_blind)}` : '50 / 100'}</strong></div>
      <div className="strip-cell"><small>BUTTON</small><strong>{state ? (state.button === state.hero_seat ? 'YOU' : 'AI') : '—'}</strong></div>
      <div className="strip-cell"><small>CURRENT TURN</small><strong>{state ? state.match_over ? 'MATCH OVER' : state.phase === 'complete' ? 'HAND COMPLETE' : state.your_turn ? 'YOUR ACTION' : 'AI THINKING' : 'NOT STARTED'}</strong></div>
      <div className="strip-cell"><small>TABLE TYPE</small><strong>{state?.stack_mode === 'continuous' ? '2 PLAYER · CONTINUOUS' : '2 PLAYER · 200 BB'}</strong></div>
    </div>
  </main>;
}
