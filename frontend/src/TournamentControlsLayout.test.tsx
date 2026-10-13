import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { TournamentApp } from './tournament/TournamentApp';

describe('Tournament action panel visibility', () => {
  it('places a single actionable betting panel before the arena and history', () => {
    const html = renderToStaticMarkup(<TournamentApp />);
    const control = html.indexOf('aria-label="Tournament betting panel"');
    const arena = html.indexOf('aria-label="Six player tournament table"');
    const history = html.indexOf('TOURNAMENT JOURNAL');
    expect(control).toBeGreaterThan(-1);
    expect(arena).toBeGreaterThan(-1);
    expect(history).toBeGreaterThan(-1);
    expect(control).toBeLessThan(arena);
    expect(arena).toBeLessThan(history);
    expect(html.match(/aria-label="Tournament betting panel"/g)).toHaveLength(1);
    expect(html).toContain('Fold');
    expect(html).toContain('Check / Call');
    expect(html).toContain('Raise to');
  });
});
