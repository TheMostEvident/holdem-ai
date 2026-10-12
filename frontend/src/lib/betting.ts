import type { ActionOffer, GameView } from '../api/types';

/** Human actions are exact engine offers; all-in calling is represented as CALL. */
export function offerFor(state: GameView, kind: ActionOffer['kind']): ActionOffer | undefined {
  return state.actions.find(action => action.kind === kind);
}

/** A short all-in can be legal even if it is below the ordinary full-raise minimum. */
export function raiseLimits(state: GameView): { min: number; max: number; shortAllInOnly: boolean } | null {
  if (!state.your_turn || !state.can_raise_to || state.min_raise_to == null || state.max_raise_to == null) return null;
  if (!Number.isSafeInteger(state.min_raise_to) || !Number.isSafeInteger(state.max_raise_to) || state.max_raise_to < 1) return null;
  return {
    min: Math.min(state.min_raise_to, state.max_raise_to),
    max: state.max_raise_to,
    shortAllInOnly: state.min_raise_to > state.max_raise_to,
  };
}

export function raiseToError(state: GameView, target: number): string | null {
  const limits = raiseLimits(state);
  if (!limits) return 'Raising is not available on this turn.';
  if (!Number.isSafeInteger(target)) return 'Enter an integer chip amount.';
  if (target < limits.min || target > limits.max) return `Choose a raise-to amount between ${limits.min} and ${limits.max}.`;
  if (limits.shortAllInOnly && target !== limits.max) return 'Only a short all-in raise is available.';
  return null;
}

/** Chosen amount is TOTAL invested on the current street, not additional chips. */
export function defaultRaiseTo(state: GameView): number | null {
  const limits = raiseLimits(state);
  if (!limits) return null;
  return limits.min;
}
