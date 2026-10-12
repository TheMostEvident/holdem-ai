import { useEffect, useState } from 'react';
import type { GameView } from '../api/types';
import { defaultRaiseTo, offerFor, raiseLimits, raiseToError } from '../lib/betting';
import { formatChips } from '../lib/poker';

type Props = {
  state: GameView | null;
  preview: boolean;
  working: boolean;
  onAction: (index: number) => void;
  onRaiseTo: (target: number) => void;
  onNext: () => void;
};

export function BettingPanel({ state, preview, working, onAction, onRaiseTo, onNext }: Props) {
  const [amountText, setAmountText] = useState('');
  const turn = !!state && !preview && state.phase === 'playing' && state.your_turn;
  const limits = state && !preview ? raiseLimits(state) : null;
  const fold = state ? offerFor(state, 'fold') : undefined;
  const check = state ? offerFor(state, 'check') : undefined;
  const call = state ? offerFor(state, 'call') : undefined;
  const checkOrCall = check ?? call;
  const allIn = state ? offerFor(state, 'all_in') : undefined;
  const offers = state?.actions.filter(a => a.kind === 'raise_to') ?? [];
  const amount = amountText.trim() === '' ? NaN : Number(amountText);
  const invalid = state && turn ? raiseToError(state, amount) : null;

  useEffect(() => {
    setAmountText(state && !preview ? String(defaultRaiseTo(state) ?? '') : '');
  }, [state?.revision, state?.hand_number, preview]);

  const quickBets = offers.map((offer) => ({
    ...offer, description: offer.index === 3 ? '½ Pot' : offer.index === 4 ? 'Pot' : offer.index === 5 ? '2× Pot' : offer.label,
  }));

  return <div className="betting-panel" aria-label="Human betting controls">
    <div className="betting-panel-title">
      <span className="eyebrow">YOUR DECISION</span>
      {turn ? <strong className="decision-indicator">YOUR TURN</strong> : <span className="decision-idle">{preview ? 'PREVIEW ONLY' : state?.match_over ? 'MATCH OVER' : state?.phase === 'complete' ? 'HAND COMPLETE' : 'WAITING'}</span>}
    </div>
    {turn ? <div className="betting-callout">To call <b>{formatChips(state.to_call ?? 0)}</b> chips · Pot <b>{formatChips(state.pot)}</b></div> : <p className="betting-tip">{preview ? 'Start a live session to use betting controls.' : state?.match_over ? 'Match finished. A player has no chips remaining. Start a new session.' : state?.phase === 'complete' ? 'This hand is complete. Deal the next hand.' : 'Start a game; the server will deal and advance the bot automatically.'}</p>}

    <div className="betting-main-actions">
      <button type="button" className="bet-button bet-fold" disabled={!turn || working || !fold} onClick={() => fold && onAction(fold.index)}>Fold</button>
      <button type="button" className="bet-button bet-call" disabled={!turn || working || !checkOrCall} onClick={() => { if (checkOrCall) onAction(checkOrCall.index); }}>{check ? 'Check' : call ? `Call ${formatChips(state?.to_call ?? 0)}` : 'Check / Call'}</button>
      <button type="button" className="bet-button bet-allin" disabled={!turn || working || !allIn} onClick={() => allIn && onAction(allIn.index)}>All-In</button>
    </div>

    {limits && <div className="bet-sizing">
      <div className="bet-sizing-head"><label htmlFor="raise-to-input">RAISE TO (CHIPS)</label><small>{limits.shortAllInOnly ? 'Short all-in only' : `Min ${formatChips(limits.min)} · Max ${formatChips(limits.max)}`}</small></div>
      <div className="bet-sizing-controls">
        <input id="raise-to-input" aria-label="Raise-to total current street bet" type="number" inputMode="numeric" step="1" min={limits.min} max={limits.max} value={amountText} disabled={!turn || working} onChange={event => setAmountText(event.target.value)}/>
        <button type="button" className="raise-submit" disabled={!turn || working || invalid != null} onClick={() => onRaiseTo(amount)}>Raise to {Number.isSafeInteger(amount) ? formatChips(amount) : '—'}</button>
      </div>
      <input aria-label="Raise-to slider" className="bet-slider" type="range" min={limits.min} max={limits.max} step={1} value={Number.isSafeInteger(amount) ? Math.min(limits.max, Math.max(limits.min, amount)) : limits.min} disabled={!turn || working || limits.shortAllInOnly} onChange={event => setAmountText(event.target.value)}/>
      <div className="bet-quick-actions">
        <button type="button" disabled={!turn || working} onClick={() => setAmountText(String(limits.min))}>Min</button>
        {quickBets.map(offer => <button type="button" key={offer.index} title={`Raise-to ${formatChips(offer.raise_to!)}`} disabled={!turn || working} onClick={() => offer.raise_to != null && setAmountText(String(offer.raise_to))}>{offer.description}</button>)}
        <button type="button" disabled={!turn || working} onClick={() => setAmountText(String(limits.max))}>Max</button>
      </div>
      {invalid && turn && <p className="bet-validation" role="status">{invalid}</p>}
      <p className="betting-tip">Raise-to is the total amount you have bet this street. Server validates every wager.</p>
    </div>}
    {state?.phase === 'complete' && !state.match_over && !preview && <button className="primary-button next-hand-button" disabled={working} onClick={onNext}>Deal next hand <span aria-hidden="true">→</span></button>}
    {state?.match_over && !preview && <p className="match-over-banner" role="status">Match finished · {state.winner_seat === state.hero_seat ? 'You won!' : 'Bot won!'} Start a new session to play again.</p>}
  </div>;
}
