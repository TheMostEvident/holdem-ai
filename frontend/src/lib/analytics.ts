import type { CompletedHand, SessionStats } from '../api/types';

/** Transform newest-first records into chronological chart points. */
export function chartPoints(history: CompletedHand[]): { hand: number; cumulativeBB: number }[] {
  return history.slice().reverse().map(hand => ({
    hand: hand.hand_number, cumulativeBB: hand.cumulative_bb,
  }));
}

export function winRate(stats: SessionStats): number {
  return stats.completed_hands ? (100 * stats.wins / stats.completed_hands) : 0;
}

/** A small, unopinionated line-chart projection. Handles flat lines and negatives. */
export function chartPath(history: CompletedHand[], width = 600, height = 152): string {
  const points = chartPoints(history);
  if (!points.length) return '';
  const values = points.map(p => p.cumulativeBB);
  const low = Math.min(...values, 0);
  const high = Math.max(...values, 0);
  const spread = Math.max(1, high - low);
  const x0 = 16, x1 = width - 16, y0 = 12, y1 = height - 18;
  return points.map((p, i) => {
    const x = x0 + ((points.length === 1 ? 0.5 : i / (points.length - 1)) * (x1 - x0));
    const y = y1 - ((p.cumulativeBB - low) / spread) * (y1 - y0);
    return `${i ? 'L' : 'M'} ${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(' ');
}
