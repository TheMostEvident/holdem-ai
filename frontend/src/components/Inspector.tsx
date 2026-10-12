import type { GameView } from '../api/types';
import { actionDescription, formatChips, formatSigned, streetName } from '../lib/poker';
import { BettingPanel } from './BettingPanel';

type Props = {
  state: GameView | null;
  preview: boolean;
  working: boolean;
  onNext: () => void;
  onAction: (index: number) => void;
  onRaiseTo: (amount: number) => void;
};
export function Inspector({ state, preview, working, onNext, onAction, onRaiseTo }: Props) {
  const result = state?.hand_result;
  const hero = state?.hero_seat ?? 0;
  return <aside className="inspector panel">
    <div className="panel-heading"><div><span className="eyebrow">02 / GAME DATA</span><h2>Hand inspector</h2></div><span className="panel-heading-icon">◎</span></div>
    <div className="inspector-scroll">
      <div className="data-section-title"><span>CURRENT HAND</span><span>{state ? `#${String(state.hand_number).padStart(3, '0')}` : '—'}</span></div>
      <div className="inspector-data">
        <div><span>Street</span><strong>{state ? streetName(state.street) : '—'}</strong></div>
        <div><span>Current pot</span><strong>{state ? formatChips(state.pot) : '—'} <small>CHP</small></strong></div>
        <div><span>To call</span><strong>{state?.to_call != null ? formatChips(state.to_call) : '—'}</strong></div>
        <div><span>Min raise-to</span><strong>{state?.min_raise_to != null ? formatChips(state.min_raise_to) : '—'}</strong></div>
      </div>
      {result && <div className={`result-box ${result.payoffs_chips[hero] >= 0 ? 'result-win' : 'result-loss'}`}>
        <span className="eyebrow">HAND RESULT</span>
        <strong>{result.payoffs_chips[hero] >= 0 ? 'Pot awarded' : 'Hand finished'}</strong>
        <div>{formatSigned(result.payoffs_chips[hero], 0)} chips <span>({formatSigned(result.payoffs_bb[hero])} BB)</span></div>
        <small>{result.showdown ? 'Showdown — opponent cards revealed' : 'No showdown — opponent cards hidden'}</small>
      </div>}
      {state && !preview && <div className="stack-status"><small>{state.phase === 'complete' ? 'FINAL' : 'CURRENT'} STACKS · HAND {state.hand_number}</small><strong>{formatChips(state.seats[hero].stack)} / {formatChips(state.seats[1 - hero].stack)} chips</strong><span>{state.stack_mode === 'continuous' ? 'Carried into the next hand' : 'Reset to 200BB each hand'}</span></div>}
      <div className="data-section-title history-header"><span>ACTIVITY FEED</span><span>{state?.action_log.length ?? 0} EVENTS</span></div>
      <div className="activity-feed" aria-live="polite">
        {state?.action_log.length ? state.action_log.slice().reverse().map((a, revIndex) => <div className="feed-event" key={`${state.revision}-${revIndex}`}>
          <div className={`feed-bullet ${a.seat === hero ? 'is-hero' : ''}`} />
          <div><strong>{a.seat === hero ? 'You' : 'AI'}</strong> {actionDescription(a)}<small>{streetName(a.street)} · action {state.action_log.length - revIndex}</small></div>
        </div>) : <div className="feed-empty">No action history yet.<span>Events appear here as the hand progresses.</span></div>}
      </div>
    </div>
    <BettingPanel state={state} preview={preview} working={working} onAction={onAction} onRaiseTo={onRaiseTo} onNext={onNext}/>
  </aside>;
}
