import type { ActionEvent, GameView, Street } from '../api/types';

const SUITS: Record<string, { label: string; glyph: string; red: boolean }> = {
  s: { label: 'Spades', glyph: '♠', red: false },
  h: { label: 'Hearts', glyph: '♥', red: true },
  d: { label: 'Diamonds', glyph: '♦', red: true },
  c: { label: 'Clubs', glyph: '♣', red: false },
};
export function decodeCard(card: string) {
  if (!/^[shdc][2-9TJQKA]$/.test(card)) return null;
  const suit = SUITS[card[0]];
  return { rank: card[1] === 'T' ? '10' : card[1], ...suit };
}
export function formatChips(value: number): string {
  return Number.isFinite(value) ? Math.round(value).toLocaleString('en-US') : '—';
}
export function formatSigned(value: number, decimals = 2): string {
  const prefix = value > 0 ? '+' : '';
  return `${prefix}${value.toFixed(decimals)}`;
}
export function streetName(street: Street): string {
  return street === 'preflop' ? 'Pre-flop' : street.charAt(0).toUpperCase() + street.slice(1);
}
export function actionDescription(event: ActionEvent): string {
  switch (event.kind) {
    case 'fold': return 'folds';
    case 'check': return 'checks';
    case 'call': return 'calls';
    case 'all_in': return 'moves all-in';
    case 'raise_to': return `raises to ${formatChips(event.raise_to ?? 0)}`;
  }
}
export function canStartNextHand(state: GameView): boolean {
  return state.phase === 'complete' && state.hand_result !== null;
}
export function opponentCardsToRender(state: GameView): string[] | null {
  return state.phase === 'complete' && state.hand_result?.showdown === true
    ? state.hand_result.opponent_cards : null;
}
export function isHeroActing(state: GameView): boolean {
  return state.phase === 'playing' && state.your_turn && state.to_act === state.hero_seat;
}
