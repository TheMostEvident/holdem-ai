import type { TournamentHand, TournamentState } from './types';

export type StandingsRow = {
  seat: number;
  stack: number;
  alive: boolean;
  officialRank: number | null;
  chipRank: number | null;
  champion: boolean;
};

/** Last SETTLED stacks: current-hand pot commitments must not change standings. */
export function tournamentStandings(state: TournamentState): StandingsRow[] {
  const alive = new Set(state.alive_seats);
  const live = state.seats.filter(s => alive.has(s.seat))
    .sort((a, b) => state.tournament_stacks[b.seat] - state.tournament_stacks[a.seat] || a.seat - b.seat);
  const eliminated = state.seats.filter(s => !alive.has(s.seat))
    .sort((a, b) => (a.rank ?? 7) - (b.rank ?? 7) || a.seat - b.seat);
  let previous: number | undefined;
  let chipRank = 0;
  return [
    ...live.map((seat, index) => {
      const stack = state.tournament_stacks[seat.seat];
      if (previous !== stack) chipRank = index + 1;
      previous = stack;
      return {seat:seat.seat, stack, alive:true, officialRank:seat.rank,
        chipRank:state.tournament_over ? null : chipRank, champion: state.winner_seat === seat.seat};
    }),
    ...eliminated.map(seat => ({seat:seat.seat, stack:0, alive:false, officialRank:seat.rank,
      chipRank:null, champion:false})),
  ];
}

export function recentKnockouts(history: TournamentHand[], maxCount = 4): {seat:number; hand:number}[] {
  return history.flatMap(h => h.eliminated.map(seat => ({seat, hand:h.hand_number})))
    .sort((a,b)=>b.hand-a.hand || a.seat-b.seat).slice(0,maxCount);
}

export function blindProgressText(state: TournamentState): string {
  const p = state.blind_progress;
  if (state.tournament_over) return 'Tournament finished';
  if (p.final_level) return 'Final blind level · no further increases';
  if (p.hands_to_next == null || p.next_small_blind == null || p.next_big_blind == null) return 'Blind schedule unavailable';
  return `Next ${p.next_small_blind} / ${p.next_big_blind} in ${p.hands_to_next} hand${p.hands_to_next === 1 ? '' : 's'} (including current)`;
}
