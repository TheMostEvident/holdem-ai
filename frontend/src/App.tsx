import { useState } from 'react';
import HUNLApp from './HUNLApp';
import { TournamentApp } from './tournament/TournamentApp';

/** Two independent mounted environments. Switching tables preserves local session state. */
export default function App() {
  const [environment, setEnvironment] = useState<'hunl' | 'tournament'>('hunl');
  return <>
    <nav className="environment-switcher" aria-label="Choose poker environment">
      <span className="environment-switch-label">ENVIRONMENT</span>
      <button type="button" className={environment==='hunl'?'environment-selected':''} aria-pressed={environment==='hunl'} onClick={()=>setEnvironment('hunl')}>Heads-Up · HUNL</button>
      <button type="button" className={environment==='tournament'?'environment-selected':''} aria-pressed={environment==='tournament'} onClick={()=>setEnvironment('tournament')}>6-Max · Tournament</button>
    </nav>
    <div style={{display:environment==='hunl'?'block':'none'}}><HUNLApp/></div>
    <div style={{display:environment==='tournament'?'block':'none'}}><TournamentApp/></div>
  </>;
}
