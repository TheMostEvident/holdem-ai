import { describe, it, expect } from 'vitest';
import { chartPath, chartPoints, winRate } from './analytics';
import type { CompletedHand, SessionStats } from '../api/types';

const fake = (hand_number: number, cumulative_bb: number): CompletedHand => ({
  hand_number, cumulative_bb, starting_stacks: [20000, 20000], button: 0, street: 'river', big_blind: 100,
  small_blind: 50, hero_cards: ['sA', 'dK'], board: [], action_log: [],
  hero_net_chips: 0, hero_net_bb: 0, outcome: 'even',
  result: { payoffs_chips: [0, 0], payoffs_bb: [0, 0], final_stacks: [20000, 20000],
    awards: [0, 0], showdown: false, opponent_cards: null },
});

describe('GUI-1.5 analytics helpers', () => {
  it('maps newest-first history to oldest-first chart order without mutating input', () => {
    const source = [fake(3, 2), fake(2, -1), fake(1, 1)];
    expect(chartPoints(source)).toEqual([
      { hand: 1, cumulativeBB: 1 },
      { hand: 2, cumulativeBB: -1 },
      { hand: 3, cumulativeBB: 2 },
    ]);
    expect(source[0].hand_number).toBe(3);
    expect(chartPath(source)).toMatch(/^M /);
  });
  it('gracefully handles empty history and one flat result', () => {
    expect(chartPath([])).toBe('');
    expect(chartPath([fake(1, 0)])).toMatch(/^M /);
  });
  it('uses the hand count rather than total decisions for win percentage', () => {
    expect(winRate({ completed_hands: 0, wins: 0 } as SessionStats)).toBe(0);
    expect(winRate({ completed_hands: 8, wins: 3 } as SessionStats)).toBe(37.5);
  });
});
