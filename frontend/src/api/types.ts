/** Strictly matches backend.schemas.GameView / GameResponse (GUI-1.1). */
export type Street = 'preflop' | 'flop' | 'turn' | 'river';
export type HandPhase = 'playing' | 'complete';
export type ActionKind = 'fold' | 'check' | 'call' | 'raise_to' | 'all_in';
export type BotName = 'random' | 'calling_station' | 'tight' | 'pot_pressure' | 'ppo' | 'qboost';
export interface BotOption { id: BotName; label: string; trained: boolean; available: boolean; message: string; }
export interface BotCatalog { bots: BotOption[]; device: 'cpu' | 'cuda'; policy_mode: 'sampled' | 'greedy'; }

export interface SeatView {
  seat: number;
  role: 'human' | 'bot';
  stack: number;
  street_bet: number;
  total_contribution: number;
  status: string;
}
export interface ActionOffer {
  index: number;
  label: string;
  kind: ActionKind;
  raise_to: number | null;
}
export interface ActionEvent {
  seat: number;
  street: Street;
  index: number;
  kind: ActionKind;
  raise_to: number | null;
}
export interface HandResult {
  payoffs_chips: number[];
  payoffs_bb: number[];
  final_stacks: number[];
  awards: number[];
  showdown: boolean;
  opponent_cards: string[] | null;
}
export interface GameView {
  hand_number: number;
  revision: number;
  phase: HandPhase;
  game_type: 'HUNL200';
  stack_mode: 'continuous' | 'independent';
  starting_stacks: number[];
  match_over: boolean;
  winner_seat: number | null;
  hero_seat: number;
  button: number;
  small_blind: number;
  big_blind: number;
  street: Street;
  board: string[];
  hero_cards: string[];
  seats: SeatView[];
  pot: number;
  to_act: number | null;
  your_turn: boolean;
  actions: ActionOffer[];
  to_call: number | null;
  min_raise_to: number | null;
  max_raise_to: number | null;
  can_raise_to: boolean;
  action_log: ActionEvent[];
  completed_hands: number;
  cumulative_bb: number;
  bb_per_100: number;
  hand_result: HandResult | null;
}
export interface GameResponse {
  session_id: string;
  state: GameView;
}
export interface CreateSessionRequest {
  bot: BotName;
  seed: number | null;
  big_blind: number;
  small_blind: number;
  stack_bb: number;
  stack_mode: 'continuous';
}

/** GUI-1.5 sanitized, completed-hand records. Never contains live villain cards. */
export interface CompletedHand {
  hand_number: number;
  starting_stacks: number[];
  button: number;
  street: Street;
  big_blind: number;
  small_blind: number;
  hero_cards: string[];
  board: string[];
  action_log: ActionEvent[];
  hero_net_chips: number;
  hero_net_bb: number;
  cumulative_bb: number;
  outcome: 'win' | 'loss' | 'even';
  result: HandResult;
}
export interface HistoryResponse {
  items: CompletedHand[];
  offset: number;
  limit: number;
  total: number;
  completed_hands: number;
  has_more: boolean;
  records_truncated: boolean;
}
export interface SessionStats {
  completed_hands: number;
  wins: number;
  losses: number;
  break_even: number;
  showdowns: number;
  net_chips: number;
  net_bb: number;
  bb_per_100: number;
  mean_bb_per_hand: number;
  std_bb_per_hand: number | null;
  ci95_lower_bb100: number | null;
  ci95_upper_bb100: number | null;
  max_drawdown_bb: number;
  retained_hand_records: number;
  records_truncated: boolean;
  note: string;
}
