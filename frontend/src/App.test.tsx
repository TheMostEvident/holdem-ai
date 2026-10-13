import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import App from './App';

describe('GUI environment landing', () => {
  it('starts with two environment options instead of mounting a game', () => {
    const html = renderToStaticMarkup(<App />);
    expect(html).toContain('Choose your table.');
    expect(html).toContain('HUNL');
    expect(html).toContain('Tournament');
    expect(html).toContain('environment-landing');
    expect(html).not.toContain('environment-switcher');
  });
});
