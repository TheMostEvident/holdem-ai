import { useEffect, useState } from 'react';
import { formatChips } from '../lib/poker';
import { isLegal, raiseToValidation, TOURNEY_ACTIONS } from './rules';
import type { TournamentState } from './types';

type Props = {
  state: TournamentState | null;
  working: boolean;
  onAction: (index: number) => void;
  onRaise: (amount: number) => void;
};

export function TournamentBetting({ state, working, onAction, onRaise }: Props) {
  const [text, setText] = useState('');
  const canAct = !!state?.your_turn && !state.tournament_over && !working;
  const amount = Number(text);
  const checkCall = isLegal(state, TOURNEY_ACTIONS.CHECK_CALL);
  const raiseInvalid = raiseToValidation(state, text.trim() ? amount : NaN);
  const min = state?.min_raise_to ?? 0;
  const max = state?.max_raise_to ?? 0;
  const valueForSlider = Number.isSafeInteger(amount) ? Math.min(max, Math.max(min, amount)) : min;
  useEffect(() => {
    setText(state?.can_raise_to && state.min_raise_to != null ? String(state.min_raise_to) : '');
  }, [state?.revision, state?.hand_number, state?.can_raise_to, state?.min_raise_to]);
  return <section className="t-betting" aria-label="Tournament betting panel">
    <div className="t-side-heading"><span className="eyebrow">02 / HUMAN ACTION</span><strong className={canAct ? 't-turn-on' : 't-turn-off'}>{canAct ? 'YOUR TURN' : state?.tournament_over ? 'FINISHED' : 'WAITING'}</strong></div>
    <p className="t-hint">{!state ? 'Start a tournament to play.' : state.human_eliminated ? 'You are out of this tournament.' : state.your_turn ? `To call: ${formatChips(state.to_call ?? 0)} chips. All wagers are checked by the server.` : 'Bot decisions advance automatically between your turns.'}</p>
    <div className="t-action-grid">
      <button className="bet-button bet-fold" type="button" disabled={!canAct || !isLegal(state, TOURNEY_ACTIONS.FOLD)} onClick={() => onAction(TOURNEY_ACTIONS.FOLD)}>Fold</button>
      <button className="bet-button bet-call" type="button" disabled={!canAct || !checkCall} onClick={() => onAction(TOURNEY_ACTIONS.CHECK_CALL)}>Check / Call</button>
      <button className="bet-button bet-allin" type="button" disabled={!canAct || !isLegal(state, TOURNEY_ACTIONS.ALL_IN)} onClick={() => onAction(TOURNEY_ACTIONS.ALL_IN)}>All-In</button>
    </div>
    <div className="t-action-grid t-action-grid-two">
      <button className="bet-button" type="button" disabled={!canAct || !isLegal(state, TOURNEY_ACTIONS.HALF_POT)} onClick={() => onAction(TOURNEY_ACTIONS.HALF_POT)}>Raise ½ Pot</button>
      <button className="bet-button" type="button" disabled={!canAct || !isLegal(state, TOURNEY_ACTIONS.POT)} onClick={() => onAction(TOURNEY_ACTIONS.POT)}>Raise Pot</button>
    </div>
    <div className="t-raise-area"><div className="t-raise-head"><label htmlFor="tournament-raise-to">RAISE TO · STREET TOTAL</label><small>{state?.can_raise_to ? `${formatChips(min)} – ${formatChips(max)} chips` : 'Not available'}</small></div>
      <div className="t-raise-controls"><input id="tournament-raise-to" aria-label="Tournament raise-to total" type="number" step="1" min={min} max={max} value={text} disabled={!canAct || !state?.can_raise_to} onChange={e => setText(e.target.value)}/><button type="button" disabled={!canAct || !!raiseInvalid} onClick={() => onRaise(amount)}>Raise to</button></div>
      {state?.can_raise_to && max >= min && <><input aria-label="Tournament raise slider" type="range" className="bet-slider" min={min} max={max} step="1" value={valueForSlider} disabled={!canAct || max === min} onChange={e => setText(e.target.value)} /><div className="t-raise-presets"><button disabled={!canAct} onClick={() => setText(String(min))}>Minimum</button><button disabled={!canAct} onClick={() => setText(String(max))}>Maximum</button></div></>}
      {canAct && raiseInvalid && state?.can_raise_to && <span className="t-validation" role="status">{raiseInvalid}</span>}
      <p className="betting-tip">Tournament action indices are 0–4 (not HUNL200's 0–6). Arbitrary Raise-to uses a separate endpoint.</p>
    </div>
  </section>;
}
