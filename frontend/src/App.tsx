import { useState } from 'react';
import HUNLApp from './HUNLApp';
import { TournamentApp } from './tournament/TournamentApp';
import './environment-landing.css';

type Environment = 'hunl' | 'tournament';

/**
 * A fresh browser load always starts at the environment selector.
 * Hidden environments stay mounted after the first visit to preserve
 * in-progress local sessions while navigating back to the selector.
 */
export default function App() {
  const [environment, setEnvironment] = useState<Environment | null>(null);
  const [visited, setVisited] = useState<Record<Environment, boolean>>({
    hunl: false,
    tournament: false,
  });

  const choose = (selected: Environment) => {
    setVisited(previous => ({ ...previous, [selected]: true }));
    setEnvironment(selected);
  };

  return <>
    {environment === null ? (
      <main className="environment-landing" aria-labelledby="environment-title">
        <header className="environment-landing-head">
          <span className="eyebrow">HOLD'EM AI / ENVIRONMENT SELECTOR</span>
          <h1 id="environment-title">Choose your table.</h1>
          <p>Select the poker environment you want to play. A new page load always returns here; no mode is automatically selected.</p>
        </header>
        <section className="environment-landing-options" aria-label="Available poker environments">
          <button type="button" className="environment-option" onClick={() => choose('hunl')}>
            <span className="option-icon" aria-hidden="true">♠</span>
            <strong>HUNL</strong>
            <small>Heads-Up No-Limit Hold'em. One human versus one opponent, with optional locally configured trained agent.</small>
            <span className="option-detail">2 PLAYERS · 200 BB BASELINE →</span>
          </button>
          <button type="button" className="environment-option" onClick={() => choose('tournament')}>
            <span className="option-icon" aria-hidden="true">♣</span>
            <strong>Tournament</strong>
            <small>Six-seat single-table knockout. Progressive blind levels, elimination, recorded hand history and replay.</small>
            <span className="option-detail">6 PLAYERS · NO REBUY →</span>
          </button>
        </section>
        <p className="environment-landing-footnote">Local FastAPI server required for live play. Trained policies need valid local checkpoints. Entering a mode does not launch training.</p>
      </main>
    ) : (
      <nav className="environment-switcher" aria-label="Choose poker environment">
        <button type="button" onClick={() => setEnvironment(null)}>← Environments</button>
        <span className="environment-switch-label">ENVIRONMENT</span>
        <button type="button" className={environment === 'hunl' ? 'environment-selected' : ''} aria-pressed={environment === 'hunl'} onClick={() => choose('hunl')}>Heads-Up · HUNL</button>
        <button type="button" className={environment === 'tournament' ? 'environment-selected' : ''} aria-pressed={environment === 'tournament'} onClick={() => choose('tournament')}>6-Max · Tournament</button>
      </nav>
    )}
    {visited.hunl && <div hidden={environment !== 'hunl'}><HUNLApp /></div>}
    {visited.tournament && <div hidden={environment !== 'tournament'}><TournamentApp /></div>}
  </>;
}
