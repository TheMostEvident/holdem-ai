import type { TournamentActionEvent, TournamentSeat, TournamentState } from './types';

export const SEATS = [0, 1, 2, 3, 4, 5] as const;
export const TOURNEY_ACTIONS = { FOLD: 0, CHECK_CALL: 1, HALF_POT: 2, POT: 3, ALL_IN: 4 } as const;
export function isLegal(state: TournamentState | null, index: number): boolean {
  return !!state?.your_turn && !state.tournament_over && state.actions.some(a => a.index === index);
}
export function chipsAccounting(state: TournamentState): { accounted: number; total: number; valid: boolean } {
  const accounted = state.seats.reduce((sum: number, s: TournamentSeat) => sum + s.stack, 0) + state.pot;
  return { accounted, total: state.total_chips, valid: accounted === state.total_chips };
}
export function raiseToValidation(state: TournamentState | null, value: number): string | null {
  if (!state || !state.your_turn || !state.can_raise_to) return 'Raise is not currently available.';
  if (!Number.isSafeInteger(value)) return 'Enter an integer chip amount.';
  if (state.min_raise_to == null || state.max_raise_to == null) return 'Server did not provide legal raise limits.';
  if (value < state.min_raise_to || value > state.max_raise_to) {
    return `Enter a raise-to amount between ${state.min_raise_to} and ${state.max_raise_to}.`;
  }
  return null;
}
export function eventText(event: TournamentActionEvent): string {
  switch (event.kind) {
    case 'fold': return 'folds';
    case 'check_call': return 'checks / calls';
    case 'raise_half_pot': return 'raises 1/2 pot';
    case 'raise_pot': return 'raises pot';
    case 'all_in': return 'moves all-in';
    case 'raise_to': return `raises to ${event.raise_to ?? '?'} chips`;
    default: return event.kind;
  }
}
export function seatLabel(seat: number, state?: TournamentState | null): string {
  if (seat === 0) return 'YOU';
  const role = state?.seats.find(s => s.seat === seat)?.role;
  if (role === 'tournament_agent') return 'TRAINED TOURNAMENT AGENT';
  if (role === 'tournament_agent_placeholder') return 'AGENT PLACEHOLDER';
  if (role === 'improved_bot') return `IMPROVED BOT ${seat - 1}`;
  return `BASELINE BOT ${seat}`;
}
export function policyDescription(seat: TournamentSeat): string {
  return seat.role === 'tournament_agent' ? `Frozen multi-seat PPO: ${seat.policy_name} (cash-game transfer)` :
    seat.role === 'tournament_agent_placeholder' ? 'Untrained heuristic fallback - Agent slot reserved' :
    seat.role === 'improved_bot' ? `Heuristic: ${seat.policy_name.replace(/_/g, ' ')}` :
    seat.role === 'human' ? 'Human' : `Baseline: ${seat.policy_name}`;
}
export function activeCount(state: TournamentState): number {
  return state.alive_seats.length;
}
