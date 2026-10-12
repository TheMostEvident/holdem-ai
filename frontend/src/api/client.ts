import type { BotCatalog, CreateSessionRequest, GameResponse, HistoryResponse, SessionStats } from './types';

const ROOT = '/api/v1/hunl';

export class GameApiError extends Error {
  constructor(message: string, public readonly status: number) {
    super(message);
    this.name = 'GameApiError';
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(ROOT + path, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init.headers },
      cache: 'no-store',
    });
  } catch {
    throw new GameApiError('Cannot reach local HUNL API. Start FastAPI on 127.0.0.1:8000.', 0);
  }
  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const body: { detail?: string | unknown[] } = await response.json();
      if (typeof body.detail === 'string') message = body.detail;
    } catch { /* non-JSON errors still show HTTP status */ }
    throw new GameApiError(message, response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const hunlApi = {
  bots: () => request<BotCatalog>('/bots'),
  stats: (sessionId: string) =>
    request<SessionStats>(`/sessions/${encodeURIComponent(sessionId)}/stats`),
  history: (sessionId: string, offset = 0, limit = 30) =>
    request<HistoryResponse>(`/sessions/${encodeURIComponent(sessionId)}/history?offset=${offset}&limit=${limit}`),
  create: (options: CreateSessionRequest) => request<GameResponse>('/sessions', {
    method: 'POST', body: JSON.stringify(options),
  }),
  get: (sessionId: string) => request<GameResponse>(`/sessions/${encodeURIComponent(sessionId)}`),
  action: (sessionId: string, actionIndex: number, expectedRevision: number) =>
    request<GameResponse>(`/sessions/${encodeURIComponent(sessionId)}/actions`, {
      method: 'POST', body: JSON.stringify({ action_index: actionIndex, expected_revision: expectedRevision }),
    }),
  raiseTo: (sessionId: string, amount: number, expectedRevision: number) =>
    request<GameResponse>(`/sessions/${encodeURIComponent(sessionId)}/raise-to`, {
      method: 'POST', body: JSON.stringify({ raise_to: amount, expected_revision: expectedRevision }),
    }),
  nextHand: (sessionId: string, expectedRevision: number) =>
    request<GameResponse>(`/sessions/${encodeURIComponent(sessionId)}/next-hand`, {
      method: 'POST', body: JSON.stringify({ expected_revision: expectedRevision }),
    }),
  delete: (sessionId: string) => request<void>(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  }),
  health: () => request<{ status: string; service: string }>('/health'),
};
