import { describe, expect, it } from 'vitest';
import { activeCount, chipsAccounting, eventText, isLegal, policyDescription, seatLabel, raiseToValidation, TOURNEY_ACTIONS } from './rules';
import type { TournamentState } from './types';

const fixture = (): TournamentState => ({
  game_type:'TOURNAMENT_6MAX', lineup:'baseline_5', revision:5, phase:'playing', tournament_over:false,
  winner_seat:null, human_eliminated:false, hand_number:2, hands_completed:1,
  alive_seats:[0,1,2,3,4,5], tournament_stacks:[1000,1000,1000,1000,1000,1000],
  blind_progress:{hands_per_level:10,hand_in_level:2,hands_to_next:9,next_small_blind:10,next_big_blind:20,final_level:false},
  level_index:0, small_blind:5, big_blind:10, button:1, small_blind_seat:2,
  big_blind_seat:3, street:'preflop', board:[], hero_cards:['sA','dK'],
  seats:Array.from({length:6},(_,i)=>({ seat:i, role:i===0?'human' as const:'baseline_bot' as const, policy_name: i===0?'human':'calling_station',
    stack:i===0?990:1000, status:'active', street_bet:0, total_contribution:i===0?10:0, rank:null })),
  total_chips:6000, pot:10, to_act:0, your_turn:true, to_call:10,
  min_raise_to:20, max_raise_to:990, can_raise_to:true,
  actions:[{index:0,kind:'fold'},{index:1,kind:'check_call'},{index:4,kind:'all_in'}],
  action_log:[], last_hand:null, starting_stacks:[1000,1000,1000,1000,1000,1000],
});

describe('Tournament ENV 1.2 UI rules',()=>{
  it('renders real lineup roles and identifies the placeholder unambiguously',()=>{
    const f=fixture(); f.lineup='improved_4';
    f.seats[1].role='tournament_agent_placeholder';
    for(let i=2;i<6;i++) { f.seats[i].role='improved_bot'; f.seats[i].policy_name='balanced_value'; }
    expect(seatLabel(1,f)).toContain('PLACEHOLDER');
    expect(seatLabel(2,f)).toBe('IMPROVED BOT 1');
    expect(policyDescription(f.seats[1])).toContain('reserved');
    expect(policyDescription(f.seats[3])).toContain('Heuristic');
  });
  it('uses five-action tournament indices, not seven-action HUNL indices',()=>{
    const f=fixture();
    expect(TOURNEY_ACTIONS.ALL_IN).toBe(4);
    expect(isLegal(f,TOURNEY_ACTIONS.ALL_IN)).toBe(true);
    expect(isLegal(f,TOURNEY_ACTIONS.POT)).toBe(false);
    expect(isLegal(f,5)).toBe(false);
  });
  it('blocks actions after tournament finishes or human has no turn',()=>{
    const f=fixture(); f.your_turn=false;
    expect(isLegal(f,0)).toBe(false);
    f.your_turn=true; f.tournament_over=true;
    expect(isLegal(f,1)).toBe(false);
  });
  it('counts committed chips in pot for total-chip invariant',()=>{
    const f=fixture();
    expect(chipsAccounting(f)).toEqual({accounted:6000,total:6000,valid:true});
    f.pot+=1;
    expect(chipsAccounting(f).valid).toBe(false);
  });
  it('validates integer absolute raise-to limits sent by server',()=>{
    const f=fixture();
    expect(raiseToValidation(f,20)).toBeNull();
    expect(raiseToValidation(f,990)).toBeNull();
    expect(raiseToValidation(f,19)).toMatch(/between/);
    expect(raiseToValidation(f,991)).toMatch(/between/);
    expect(raiseToValidation(f,20.5)).toMatch(/integer/);
  });
  it('disallows raise-to when backend does not permit it',()=>{
    const f=fixture(); f.can_raise_to=false;
    expect(raiseToValidation(f,20)).toMatch(/not currently available/);
  });
  it('handles shorter than big blind and all-in seat states in active count',()=>{
    const f=fixture();
    f.seats[1].stack=0; f.seats[1].total_contribution=100; f.seats[1].status='all_in';
    f.seats[2].stack=0; f.seats[2].status='out'; f.alive_seats=[0,1,3,4,5];
    expect(activeCount(f)).toBe(5);
  });
  it('uses action event kind check_call without inventing check/call distinction',()=>{
    expect(eventText({seat:3,index:1,kind:'check_call',raise_to:null})).toBe('checks / calls');
    expect(eventText({seat:0,index:5,kind:'raise_to',raise_to:250})).toContain('250');
  });
});
