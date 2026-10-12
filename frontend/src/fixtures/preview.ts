/** Presentation-only sample data. NEVER sent to or used by the real game engine. */
import type { GameView } from '../api/types';

const baseline: GameView = {
  hand_number: 12, revision: 36, phase: 'playing', game_type: 'HUNL200',
  stack_mode: 'independent', starting_stacks: [20000, 20000], match_over: false, winner_seat: null,
  hero_seat: 0, button: 0, small_blind: 50, big_blind: 100,
  street: 'preflop', board: [], hero_cards: ['hA', 'sK'],
  seats: [
    { seat: 0, role: 'human', stack: 19950, street_bet: 50, total_contribution: 50, status: 'active' },
    { seat: 1, role: 'bot', stack: 19900, street_bet: 100, total_contribution: 100, status: 'active' },
  ],
  pot: 150, to_act: 0, your_turn: true,
  actions: [
    { index: 0, kind: 'fold', label: 'Fold', raise_to: null },
    { index: 2, kind: 'call', label: 'Call', raise_to: null },
    { index: 3, kind: 'raise_to', label: 'Half-pot raise', raise_to: 250 },
    { index: 4, kind: 'raise_to', label: 'Pot raise', raise_to: 350 },
    { index: 5, kind: 'raise_to', label: 'Double-pot raise', raise_to: 550 },
    { index: 6, kind: 'all_in', label: 'All-in', raise_to: null },
  ],
  to_call: 50, min_raise_to: 200, max_raise_to: 20000, can_raise_to: true,
  action_log: [], completed_hands: 11, cumulative_bb: 7.25, bb_per_100: 65.91,
  hand_result: null,
};

const flop: GameView = {
  ...baseline, revision: 42, street: 'flop', board: ['hQ', 'dJ', 'c2'], pot: 600,
  seats: [
    { seat: 0, role: 'human', stack: 19700, street_bet: 0, total_contribution: 300, status: 'active' },
    { seat: 1, role: 'bot', stack: 19700, street_bet: 0, total_contribution: 300, status: 'active' },
  ], to_call: 0, min_raise_to: 100, max_raise_to: 19700,
  can_raise_to: true,
  actions: [
    { index: 0, kind: 'fold', label: 'Fold', raise_to: null },
    { index: 1, kind: 'check', label: 'Check', raise_to: null },
    { index: 3, kind: 'raise_to', label: 'Half-pot raise', raise_to: 300 },
    { index: 4, kind: 'raise_to', label: 'Pot raise', raise_to: 600 },
    { index: 5, kind: 'raise_to', label: 'Double-pot raise', raise_to: 1200 },
    { index: 6, kind: 'all_in', label: 'All-in', raise_to: null },
  ],
  action_log: [
    { seat: 0, street: 'preflop', index: 4, kind: 'raise_to', raise_to: 300 },
    { seat: 1, street: 'preflop', index: 2, kind: 'call', raise_to: null },
  ],
};

const turn: GameView = {
  ...flop, revision: 46, street: 'turn', board: [...flop.board, 's8'], pot: 1200,
  seats: [
    { seat: 0, role: 'human', stack: 19400, street_bet: 0, total_contribution: 600, status: 'active' },
    { seat: 1, role: 'bot', stack: 19400, street_bet: 0, total_contribution: 600, status: 'active' },
  ],
  action_log: [
    ...flop.action_log,
    { seat: 1, street: 'flop', index: 4, kind: 'raise_to', raise_to: 300 },
    { seat: 0, street: 'flop', index: 2, kind: 'call', raise_to: null },
  ],
};

const river: GameView = {
  ...turn, revision: 49, street: 'river', board: [...turn.board, 'h3'], pot: 2000,
  seats: [
    { seat: 0, role: 'human', stack: 19000, street_bet: 0, total_contribution: 1000, status: 'active' },
    { seat: 1, role: 'bot', stack: 19000, street_bet: 0, total_contribution: 1000, status: 'active' },
  ],
  action_log: [
    ...turn.action_log,
    { seat: 1, street: 'turn', index: 4, kind: 'raise_to', raise_to: 400 },
    { seat: 0, street: 'turn', index: 2, kind: 'call', raise_to: null },
  ],
};

const showdown: GameView = {
  ...river, revision: 51, phase: 'complete', to_act: null, your_turn: false,
  actions: [], to_call: null, min_raise_to: null, max_raise_to: null, can_raise_to: false,
  completed_hands: 12, cumulative_bb: 17.25, bb_per_100: 143.75,
  seats: [
    { seat: 0, role: 'human', stack: 21000, street_bet: 0, total_contribution: 1000, status: 'active' },
    { seat: 1, role: 'bot', stack: 19000, street_bet: 0, total_contribution: 1000, status: 'active' },
  ],
  hand_result: {
    payoffs_chips: [1000, -1000], payoffs_bb: [10, -10],
    final_stacks: [21000, 19000], awards: [2000, 0], showdown: true,
    opponent_cards: ['dQ', 'cQ'],
  },
  action_log: [
    ...river.action_log,
    { seat: 1, street: 'river', index: 1, kind: 'check', raise_to: null },
    { seat: 0, street: 'river', index: 1, kind: 'check', raise_to: null },
  ],
};

const fold: GameView = {
  ...flop, revision: 45, phase: 'complete', to_act: null, your_turn: false,
  actions: [], to_call: null, min_raise_to: null, max_raise_to: null, can_raise_to: false,
  completed_hands: 12, cumulative_bb: 4.25, bb_per_100: 35.42,
  seats: [
    { seat: 0, role: 'human', stack: 19700, street_bet: 0, total_contribution: 300, status: 'folded' },
    { seat: 1, role: 'bot', stack: 20300, street_bet: 0, total_contribution: 300, status: 'active' },
  ],
  hand_result: {
    payoffs_chips: [-300, 300], payoffs_bb: [-3, 3],
    final_stacks: [19700, 20300], awards: [0, 600], showdown: false,
    opponent_cards: null,
  },
  action_log: [...flop.action_log, { seat: 0, street: 'flop', index: 0, kind: 'fold', raise_to: null }],
};

export const previewStates = { preflop: baseline, flop, turn, river, showdown, fold } as const;
export type PreviewKey = keyof typeof previewStates;
