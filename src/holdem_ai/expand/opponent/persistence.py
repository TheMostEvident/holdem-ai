"""Deterministic allowlisted JSON checkpoint, not Python pickle. Atomic writes.

The snapshot is a research artifact containing public action histories only.
Never accept untrusted checkpoint bytes as an authoritative model without
validating checksum and schema. Player IDs may nevertheless be personal data:
keep the resulting files private and/or use pseudonymous IDs.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import os
from .model import ContextCounts, OpponentConfig, OpponentModel, OpponentRegistry
from .schema import PublicDecision

FORMAT=1


def _row(c:tuple[str,str,str,tuple[str,...]],v:ContextCounts)->dict:
    return {'context':[c[0],c[1],c[2],list(c[3])], 'actions':list(v.actions),
            'long':v.long,'recent':v.recent,'raw_events':v.raw_events,
            'over_streak':v.over_streak,'shift_hold':v.shift_hold,'shift_events':v.shift_events}


def _payload(reg:OpponentRegistry)->dict:
    return {'version':FORMAT,'config':asdict(reg.config),
            'active_seats':[[k,*v] for k,v in sorted(reg.active_by_seat.items())],
            'models':[{'player_id':m.player_id,'player_session_id':m.player_session_id,
                       'contexts':[_row(c,v) for c,v in sorted(m.contexts.items())],
                       'seen_events':[list(e) for e in sorted(m.seen_events)],
                       'hands':m.hands,'finished_hands':m.finished_hands,
                       'vpip_hands':m.vpip_hands,'pfr_hands':m.pfr_hands,
                       'threebet_hands':m.threebet_hands}
                      for _,m in sorted(reg.models.items())]}


def save_registry(path:str|Path,registry:OpponentRegistry)->None:
    file=Path(path)
    if file.exists():raise FileExistsError('do not overwrite analysis checkpoints')
    file.parent.mkdir(parents=True,exist_ok=True)
    payload=_payload(registry)
    body=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    doc={'sha256':hashlib.sha256(body.encode()).hexdigest(),'payload':payload}
    temp=file.with_suffix(file.suffix+'.tmp')
    try:
        with temp.open('x',encoding='utf-8') as out:
            json.dump(doc,out,ensure_ascii=False,allow_nan=False,indent=2)
            out.write('\n');out.flush();os.fsync(out.fileno())
        os.replace(temp,file)
    finally:
        if temp.exists():temp.unlink()


def load_registry(path:str|Path)->OpponentRegistry:
    doc=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(doc,dict) or set(doc)!= {'sha256','payload'}:raise ValueError('invalid registry wrapper')
    payload=doc['payload']
    body=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    if doc['sha256']!=hashlib.sha256(body.encode()).hexdigest():raise ValueError('registry checksum mismatch')
    if payload.get('version')!=FORMAT:raise ValueError('unknown registry format')
    reg=OpponentRegistry(OpponentConfig(**payload['config']))
    for item in payload['models']:
        identity=(item['player_id'],item['player_session_id'])
        if identity in reg.models:raise ValueError('duplicate identity')
        m=OpponentModel(*identity,reg.config)
        for row in item['contexts']:
            st,pos,spr,acts=row['context']
            c=(st,pos,spr,tuple(acts))
            # Use public action schema for context validation without storing
            # private information; the observed action is a context action.
            PublicDecision(identity[0],identity[1],'validation',0,0,st,pos,spr,tuple(acts),acts[0])
            if tuple(row['actions'])!=c[-1] or c in m.contexts:raise ValueError('invalid context')
            counters=ContextCounts(c[-1],list(row['long']),list(row['recent']),
                                    row['raw_events'],row['over_streak'],row['shift_hold'],row['shift_events'])
            if (any(type(n) not in (float,int) or not math.isfinite(n) or n<0 for n in (*counters.long,*counters.recent))
                    or len(counters.long)!=len(c[-1]) or len(counters.recent)!=len(c[-1])):
                raise ValueError('invalid persisted counts')
            if any(type(getattr(counters,key)) is not int or getattr(counters,key)<0 for key in
                   ('raw_events','over_streak','shift_hold','shift_events')):raise ValueError('invalid persisted counters')
            m.contexts[c]=counters
        for event in item['seen_events']:
            if not isinstance(event,list) or len(event)!=4:raise ValueError('bad event ID')
            eid=tuple(event)
            if eid[:2]!=identity or type(eid[2]) is not str or type(eid[3]) is not int or eid[3]<0:raise ValueError('bad event identity')
            m.seen_events.add(eid)
        if len(m.seen_events)!=len(item['seen_events']) or len(m.seen_events)>reg.config.max_event_ids:
            raise ValueError('duplicate/oversized events')
        if len(m.contexts)>reg.config.max_contexts_per_player:raise ValueError('too many contexts')
        m.hands=item['hands']
        if not isinstance(m.hands,dict) or any(not isinstance(v,dict) or set(v)!= {'vpip','pfr','threebet'} or
                any(type(a) is not bool for a in v.values()) for v in m.hands.values()):
            raise ValueError('invalid hand statistics')
        for key in ('finished_hands','vpip_hands','pfr_hands','threebet_hands'):
            n=item[key]
            if type(n) is not int or n<0:raise ValueError('invalid stat')
            setattr(m,key,n)
        if max(m.vpip_hands,m.pfr_hands,m.threebet_hands)>m.finished_hands:
            raise ValueError('invalid hand stat count')
        reg.models[identity]=m
    for x in payload['active_seats']:
        seat,pid,sid=x
        if type(seat) is not int or not 0<=seat<6 or seat in reg.active_by_seat or (pid,sid) not in reg.models:
            raise ValueError('invalid active seats')
        reg.bind_seat(seat,pid,sid)
    return reg
