"""Exact legality-preserving discrete action translation, with no silent clamp."""
from __future__ import annotations
from .schema import ACTION_NAMES, LegalActionSpec
from holdem_ai.expand.environment.table_state import Action


def legal_action_spec(raw: dict, legal_ids: tuple[int, ...], bb: int) -> LegalActionSpec:
    """Translate Phase 3 legal IDs and exact public raise-to bounds.

    The Phase3 ALL_IN (4) legality is the *raise* capability marker. A short
    stack can call all-in via ID 1; we intentionally represent that as CALL to
    avoid duplicated identical actions in the policy categorical distribution.
    """
    if type(bb) is not int or bb <= 0:
        raise ValueError('positive big blind required')
    for key in ('pot', 'to_call', 'min_raise_to', 'street_bets', 'stacks', 'player_id'):
        if key not in raw:
            raise ValueError(f'missing engine observation key: {key}')
    player = raw['player_id']
    if type(player) is not int or not 0 <= player < 6:
        raise ValueError('bad player_id')
    bets, stacks = raw['street_bets'], raw['stacks']
    if len(bets) != 6 or len(stacks) != 6:
        raise ValueError('six stack/bet entries required')
    if any(type(x) is not int or x < 0 for x in (*bets, *stacks)):
        raise ValueError('invalid public chip values')
    b, s = bets[player], stacks[player]
    pot, to_call, minimum = (raw[x] for x in ('pot', 'to_call', 'min_raise_to'))
    if any(type(v) is not int or v < 0 for v in (pot, to_call, minimum)):
        raise ValueError('invalid pot/call/minimum')
    maximum = b + s
    high = b + to_call
    ids = set(legal_ids)
    if any(type(i) is not int or i not in range(5) for i in ids) or not (ids & {0, 1}):
        raise ValueError('unsupported engine legal ID set')
    allowed = [False] * len(ACTION_NAMES)
    targets: list[int | None] = [None] * len(ACTION_NAMES)
    allowed[0] = 0 in ids
    allowed[1] = (1 in ids and to_call == 0)
    allowed[2] = (1 in ids and to_call > 0)
    for index, factor, engine_id in ((3, 1, 2), (4, 2, 3), (5, 4, None)):
        # exact integer math: floor(k * (pot + to_call)) where k is 1/2/2.
        post_call = pot + to_call
        amount = (post_call * factor) // 2
        target = b + to_call + amount
        permitted = (4 in ids and target > high and minimum <= target <= maximum)
        if engine_id is not None:
            permitted = permitted and engine_id in ids
        # Distinct legal actions, so no duplicate wager-size decisions.
        if permitted and target not in targets:
            allowed[index], targets[index] = True, target
    # ID 4 is raise-only. All-in CALL is canonicalized as CALL (ID 1).
    if 4 in ids and maximum > high and maximum not in targets:
        allowed[6], targets[6] = True, maximum
    return LegalActionSpec(tuple(allowed), tuple(targets), minimum, maximum, to_call)


def to_table_action(spec: LegalActionSpec, name: str) -> Action:
    ix = spec.index(name)
    if not spec.mask[ix]:
        raise ValueError(f'{name} illegal for current observation')
    if name in ('fold', 'check', 'call'):
        return Action(name)
    if name == 'all_in':
        return Action('all_in')
    # Always use exact raise_to target; no clamp, no rounding in the engine.
    target = spec.raise_to_chips[ix]
    assert target is not None
    return Action('raise_to', target)
