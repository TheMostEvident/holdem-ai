import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import HUNLApp from './HUNLApp';
import { TournamentApp } from './tournament/TournamentApp';

describe('trained policy setup menus', () => {
  it('renders selectable HUNL PPO choices, without a checkpoint precondition on the HTML options', () => {
    const html = renderToStaticMarkup(<HUNLApp />);
    expect(html).toMatch(/<option value="ppo">Trained PPO<\/option>/);
    expect(html).toMatch(/<option value="qboost">Trained Q-Boost PPO<\/option>/);
  });

  it('uses the requested Tournament display label without disabling its option', () => {
    const html = renderToStaticMarkup(<TournamentApp />);
    expect(html).toMatch(/<option value="trained_1_improved_4">Trained PPO with other Improved Bots<\/option>/);
    expect(html).toContain('HOLDEM_AI_TOURNAMENT_CHECKPOINT');
  });
});
