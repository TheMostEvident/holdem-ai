import { describe, expect, it } from 'vitest';
import { defaultRaiseTo, offerFor, raiseLimits, raiseToError } from './betting';
import { previewStates } from '../fixtures/preview';

describe('GUI-1.3 human bet sizing contract', () => {
  it('recognizes a legal abstract action and an exact raise-to target', () => {
    const state = previewStates.preflop;
    expect(offerFor(state, 'call')?.index).toBe(2);
    expect(offerFor(state, 'check')).toBeUndefined();
    expect(raiseLimits(state)).toEqual({ min: 200, max: 20000, shortAllInOnly: false });
    expect(defaultRaiseTo(state)).toBe(200);
    expect(raiseToError(state, 250)).toBeNull();
  });
  it('rejects invalid, too small and too large wagers', () => {
    const state = previewStates.preflop;
    expect(raiseToError(state, 199)).toContain('between');
    expect(raiseToError(state, 20001)).toContain('between');
    expect(raiseToError(state, 202.5)).toContain('integer');
    expect(raiseToError(state, NaN)).toContain('integer');
  });
  it('allows an engine-validated short all-in below normal min raise', () => {
    const state = { ...previewStates.preflop, min_raise_to: 250, max_raise_to: 200 };
    expect(raiseLimits(state)).toEqual({ min: 200, max: 200, shortAllInOnly: true });
    expect(raiseToError(state, 200)).toBeNull();
    expect(raiseToError(state, 199)).not.toBeNull();
  });
  it('disables raising in previews and completed states', () => {
    expect(raiseLimits(previewStates.showdown)).toBeNull();
    expect(raiseToError(previewStates.fold, 100)).not.toBeNull();
  });
});
