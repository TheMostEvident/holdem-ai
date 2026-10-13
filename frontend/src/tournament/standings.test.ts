import { describe, expect, it } from 'vitest';
import type { TournamentState, TournamentHand } from './types';
import { blindProgressText, recentKnockouts, tournamentStandings } from './standings';

function fixture(): TournamentState {
  return {
    game_type:'TOURNAMENT_6MAX', lineup:'improved_4', phase:'playing', revision:1,
    tournament_over:false, winner_seat:null, human_eliminated:false,
    hand_number:3, hands_completed:2, level_index:1, small_blind:10,big_blind:20,
    alive_seats:[0,1,2,3,4,5], tournament_stacks:[900,1400,1000,900,800,1000],
    blind_progress:{hands_per_level:2,hand_in_level:1,hands_to_next:2,next_small_blind:20,next_big_blind:40,final_level:false},
    button:2,small_blind_seat:3,big_blind_seat:4,street:'flop',board:['sQ','h4','c2'],hero_cards:['dA','dK'],
    seats:Array.from({length:6},(_,i)=>({seat:i,role:i===0?'human' as const:'improved_bot' as const,policy_name:'example',
      stack:[0,1400,1000,900,800,1000][i],status:i===0?'allin':'active',street_bet:0,total_contribution:i===0?900:0,rank:null})),
    total_chips:6000,pot:900,to_act:1,your_turn:false,to_call:null,min_raise_to:null,max_raise_to:null,can_raise_to:false,
    actions:[],action_log:[],last_hand:null,starting_stacks:[900,1400,1000,900,800,1000],
  };
}

describe('Tournament ENV 1.5 standings and blinds',()=>{
  it('all-in zero behind is NOT an elimination and uses settled stack',()=>{
    const rows=tournamentStandings(fixture());
    const hero=rows.find(r=>r.seat===0)!;
    expect(hero.alive).toBe(true);
    expect(hero.stack).toBe(900);
    expect(hero.officialRank).toBeNull();
    expect(hero.chipRank).toBe(4);
  });
  it('chip rankings use competition ties, not arbitrary ordinal ranks',()=>{
    const rows=tournamentStandings(fixture());
    expect(rows.slice(0,6).map(r=>r.chipRank)).toEqual([1,2,2,4,4,6]);
  });
  it('final placements are official ranks; eliminated seats are not ranked by their zero stacks',()=>{
    const state=fixture();
    state.alive_seats=[1,3];
    state.tournament_stacks=[0,3500,0,2500,0,0];
    state.seats[0].rank=3; state.seats[2].rank=5; state.seats[4].rank=6; state.seats[5].rank=4;
    const rows=tournamentStandings(state);
    expect(rows.map(r=>r.seat)).toEqual([1,3,0,5,2,4]);
    expect(rows.find(r=>r.seat===0)?.officialRank).toBe(3);
    expect(rows.find(r=>r.seat===0)?.alive).toBe(false);
  });
  it('completed tournament marks champion as official rank 1',()=>{
    const state=fixture();
    state.tournament_over=true;state.phase='complete';state.winner_seat=3;
    state.alive_seats=[3]; state.tournament_stacks=[0,0,0,6000,0,0];
    state.seats[3].rank=1;
    const rows=tournamentStandings(state);
    expect(rows[0]).toMatchObject({seat:3,officialRank:1,chipRank:null,champion:true});
    expect(blindProgressText(state)).toBe('Tournament finished');
  });
  it('blind countdown includes current hand and respects last level',()=>{
    const state=fixture();
    expect(blindProgressText(state)).toContain('in 2 hands');
    state.blind_progress={...state.blind_progress,hand_in_level:2,hands_to_next:1};
    expect(blindProgressText(state)).toContain('in 1 hand (including current)');
    state.blind_progress={...state.blind_progress,final_level:true,next_small_blind:null,next_big_blind:null,hands_to_next:null};
    expect(blindProgressText(state)).toContain('Final blind level');
  });
  it('knockout feed is sorted by hand and supports multiple simultaneous busts',()=>{
    const hand=(n:number,eliminated:number[]):TournamentHand=>({hand_number:n,button:0,level_index:0,
      small_blind:5,big_blind:10,stacks_before:[1,1,1,1,1,1],stacks_after:[1,1,1,1,1,1],
      eliminated,board:[],payoffs_chips:[0,0,0,0,0,0],awards:[0,0,0,0,0,0],action_history:[]});
    expect(recentKnockouts([hand(2,[1]),hand(5,[4,2]),hand(3,[3])],3)).toEqual([
      {seat:2,hand:5},{seat:4,hand:5},{seat:3,hand:3},
    ]);
  });
});
