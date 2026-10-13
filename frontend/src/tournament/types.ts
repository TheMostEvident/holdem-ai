/** Exact read-only projection from Tournament ENV Phase 1.1 FastAPI contract. */
export type TournamentSeat = {
  seat: number;
  role: 'human' | 'baseline_bot' | 'tournament_agent_placeholder' | 'tournament_agent' | 'improved_bot';
  policy_name: string;
  stack: number;
  status: string;
  street_bet: number;
  total_contribution: number;
  rank: number | null;
};
export type TournamentActionOffer = { index: number; kind: string };
export type TournamentActionEvent = {
  seat: number;
  index: number;
  kind: string;
  raise_to: number | null;
  hand_number?: number | null;
  street?: string | null;
};
export type TournamentShowdownCards = { seat: number; cards: [string, string] };
export type TournamentHand = {
  hand_number: number;
  button: number;
  level_index: number;
  small_blind: number;
  big_blind: number;
  stacks_before: number[];
  stacks_after: number[];
  eliminated: number[];
  board: string[];
  payoffs_chips: number[];
  awards: number[];
  /** Absent in archives created before Phase 1.6.1. Only real showdown seats. */
  showdown_cards?: TournamentShowdownCards[];
  action_history: TournamentActionEvent[];
};
export type TournamentBlindProgress = {
  hands_per_level: number;
  hand_in_level: number | null;
  hands_to_next: number | null;
  next_small_blind: number | null;
  next_big_blind: number | null;
  final_level: boolean;
};
export type TournamentState = {
  game_type: 'TOURNAMENT_6MAX';
  lineup: 'baseline_5' | 'improved_4' | 'trained_1_improved_4';
  revision: number;
  phase: 'playing' | 'complete';
  tournament_over: boolean;
  winner_seat: number | null;
  human_eliminated: boolean;
  hand_number: number;
  hands_completed: number;
  alive_seats: number[];
  tournament_stacks: number[];
  blind_progress: TournamentBlindProgress;
  level_index: number;
  small_blind: number;
  big_blind: number;
  button: number;
  small_blind_seat: number | null;
  big_blind_seat: number | null;
  street: string | null;
  board: string[];
  hero_cards: string[];
  seats: TournamentSeat[];
  total_chips: number;
  pot: number;
  to_act: number | null;
  your_turn: boolean;
  to_call: number | null;
  min_raise_to: number | null;
  max_raise_to: number | null;
  can_raise_to: boolean;
  actions: TournamentActionOffer[];
  action_log: TournamentActionEvent[];
  last_hand: TournamentHand | null;
  starting_stacks: number[];
};
export type TournamentResponse = { session_id: string; state: TournamentState };
export type TournamentHistory = { total: number; offset: number; limit: number; hands: TournamentHand[] };
export type TournamentCreate = {
  seed: number | null;
  lineup?: 'baseline_5' | 'improved_4' | 'trained_1_improved_4';
  starting_stack: number;
  button: number;
  hands_per_level: number;
  max_hands: number;
};

export type TournamentAgentStatus = {
  configured: boolean;
  file_found: boolean;
  checkpoint_name: string | null;
  format: string;
  model_family: string;
  device: string;
  mode: string;
  validated: boolean;
};

/** Phase 1.6 immutable persisted public replay + actor-visible data export. */
export type TournamentArchiveSession = {
  session_id: string;
  created_utc: string;
  updated_utc: string;
  status: 'playing' | 'complete' | 'closed';
  options: { lineup?: string; seed?: number | null; starting_stack?: number };
  latest_hand: number;
  revision: number;
  winner_seat: number | null;
};
export type TournamentArchiveList = {
  total: number; offset: number; limit: number; sessions: TournamentArchiveSession[];
};
export type TournamentRecordedAction = {
  hand_number: number; action_no: number; seat: number; index: number;
  kind: string; raise_to: number | null; hero_hole: string[];
  public_before: {
    street: string; board: string[]; button: number; small_blind: number; big_blind: number;
    stacks: number[]; street_bets: number[]; contributions: number[];
    statuses: string[]; pot: number; to_call: number; legal_actions: number[];
  };
};
export type TournamentReplay = {
  session_id: string; hand_number: number; actions: TournamentRecordedAction[];
  settlement: TournamentHand | null; complete: boolean;
};
