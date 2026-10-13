import type { TournamentAgentStatus, TournamentCreate, TournamentHistory, TournamentResponse, TournamentArchiveList, TournamentReplay } from './types';

const ROOT = '/api/v1/tournament';

export class TournamentApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = 'TournamentApiError';
  }
}

async function send<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(ROOT + path, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init.headers },
      cache: 'no-store',
    });
  } catch {
    throw new TournamentApiError('Cannot reach Tournament API. Start FastAPI at 127.0.0.1:8000.', 0);
  }
  if (!response.ok) {
    let message = `Tournament API returned HTTP ${response.status}`;
    try {
      const json: { detail?: string } = await response.json();
      if (typeof json.detail === 'string') message = json.detail;
    } catch { /* retain the HTTP error */ }
    throw new TournamentApiError(message, response.status);
  }
  return response.status === 204 ? undefined as T : response.json() as Promise<T>;
}

function idPath(id: string): string { return `/sessions/${encodeURIComponent(id)}`; }

export const tournamentApi = {
  archive: (offset = 0, limit = 50) => send<TournamentArchiveList>(`/archive?offset=${offset}&limit=${limit}`),
  archivedHands: (id: string, offset = 0, limit = 50) => send<TournamentHistory>(`/archive/${encodeURIComponent(id)}/hands?offset=${offset}&limit=${limit}`),
  replay: (id: string, hand: number) => send<TournamentReplay>(`/archive/${encodeURIComponent(id)}/hands/${hand}/replay`),
  health: () => send<{ status: string; service: string; phase: string }>('/health'),
  agentStatus: () => send<TournamentAgentStatus>('/agent-status'),
  create: (options: TournamentCreate) => send<TournamentResponse>('/sessions', { method: 'POST', body: JSON.stringify(options) }),
  get: (id: string) => send<TournamentResponse>(idPath(id)),
  history: (id: string, offset = 0, limit = 25) => send<TournamentHistory>(`${idPath(id)}/history?offset=${offset}&limit=${limit}`),
  action: (id: string, action_index: number, expected_revision: number) => send<TournamentResponse>(`${idPath(id)}/actions`, {
    method: 'POST', body: JSON.stringify({ action_index, expected_revision }),
  }),
  raiseTo: (id: string, raise_to: number, expected_revision: number) => send<TournamentResponse>(`${idPath(id)}/raise-to`, {
    method: 'POST', body: JSON.stringify({ raise_to, expected_revision }),
  }),
  delete: (id: string) => send<void>(idPath(id), { method: 'DELETE' }),
};

/** URL served by the local FastAPI backend via the existing Vite proxy. */
export function tournamentActorExportUrl(id: string): string {
  return `${ROOT}/archive/${encodeURIComponent(id)}/export`;
}
