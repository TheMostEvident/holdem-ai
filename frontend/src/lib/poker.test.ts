import { describe, expect, it } from 'vitest';
import { actionDescription, canStartNextHand, decodeCard, formatChips, isHeroActing, opponentCardsToRender } from './poker';
import { previewStates } from '../fixtures/preview';

describe('GUI-1.2 poker presentation logic', () => {
  it('decodes canonical suit first notation, including ten', () => {
    expect(decodeCard('hT')).toMatchObject({ rank: '10', glyph: '♥', red: true });
    expect(decodeCard('sA')).toMatchObject({ rank: 'A', red: false });
    expect(decodeCard('Ah')).toBeNull();
    expect(decodeCard('xx')).toBeNull();
  });
  it('formats chips and action history', () => {
    expect(formatChips(20000)).toBe('20,000');
    expect(actionDescription({ seat: 1, street: 'flop', index: 4, kind: 'raise_to', raise_to: 375 })).toBe('raises to 375');
  });
  it('keeps opponent cards hidden until actual showdown', () => {
    expect(opponentCardsToRender(previewStates.preflop)).toBeNull();
    expect(opponentCardsToRender(previewStates.flop)).toBeNull();
    expect(opponentCardsToRender(previewStates.fold)).toBeNull();
    expect(opponentCardsToRender(previewStates.showdown)).toEqual(['dQ', 'cQ']);
  });
  it('allows next hand only after completion', () => {
    expect(canStartNextHand(previewStates.flop)).toBe(false);
    expect(canStartNextHand(previewStates.fold)).toBe(true);
    expect(isHeroActing(previewStates.preflop)).toBe(true);
    expect(isHeroActing(previewStates.showdown)).toBe(false);
  });
});
