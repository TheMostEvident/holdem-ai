"""Tournament ENV 1.6: append-only, local SQLite event journal.

Persistence is independent of game rules and trained-policy inference. The API
replays recorded actor-visible states; it does not pretend to resume a running
TournamentEnv from a serialized snapshot. No deck or other seats' hole cards
are stored in an action record.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class TournamentJournal:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("HOLDEM_AI_TOURNAMENT_JOURNAL", ".local/tournament_history.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS tournament_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_utc TEXT NOT NULL,
                    updated_utc TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('playing','complete','closed')),
                    options_json TEXT NOT NULL,
                    latest_hand INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 0,
                    winner_seat INTEGER
                );
                CREATE TABLE IF NOT EXISTS tournament_actions (
                    session_id TEXT NOT NULL,
                    hand_number INTEGER NOT NULL,
                    action_no INTEGER NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, hand_number, action_no),
                    FOREIGN KEY (session_id) REFERENCES tournament_sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS tournament_hands (
                    session_id TEXT NOT NULL,
                    hand_number INTEGER NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, hand_number),
                    FOREIGN KEY (session_id) REFERENCES tournament_sessions(session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_tournament_sessions_updated
                    ON tournament_sessions(updated_utc DESC);
            """)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=15)
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA busy_timeout = 15000")
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("PRAGMA synchronous = FULL")
        return db

    def begin(self, session_id: str, options: Mapping[str, Any]) -> None:
        safe = {k: v for k, v in options.items() if k in (
            "seed", "starting_stack", "button", "hands_per_level", "blind_levels", "max_hands", "lineup"
        )}
        with self._lock, self._connect() as db:
            instant = _now()
            db.execute("INSERT INTO tournament_sessions (session_id,created_utc,updated_utc,status,options_json) VALUES (?,?,?,?,?)",
                       (session_id, instant, instant, "playing", _json(safe)))

    def discard_failed_create(self, session_id: str) -> None:
        with self._lock, self._connect() as db:
            db.execute("DELETE FROM tournament_actions WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM tournament_hands WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM tournament_sessions WHERE session_id=?", (session_id,))

    def record_step(self, session_id: str, action: Mapping[str, Any],
                    completed_hands: Iterable[Mapping[str, Any]], *, revision: int,
                    winner_seat: int | None) -> None:
        completed = list(completed_hands)
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO tournament_actions (session_id,hand_number,action_no,data_json) VALUES (?,?,?,?)",
                       (session_id, action["hand_number"], action["action_no"], _json(action)))
            for hand in completed:
                db.execute("INSERT INTO tournament_hands (session_id,hand_number,data_json) VALUES (?,?,?)",
                           (session_id, hand["hand_number"], _json(hand)))
            db.execute("UPDATE tournament_sessions SET updated_utc=?, revision=?, latest_hand=MAX(latest_hand,?), status=?, winner_seat=? WHERE session_id=?",
                       (_now(), revision, max([action["hand_number"]] + [h["hand_number"] for h in completed]),
                        "complete" if winner_seat is not None else "playing", winner_seat, session_id))

    def record_initial_hands(self, session_id: str, hands: Iterable[Mapping[str, Any]], *,
                             revision: int, winner_seat: int | None) -> None:
        """Only for hands completed immediately on reset without any action."""
        hand_list = list(hands)
        if not hand_list:
            return
        with self._lock, self._connect() as db:
            for hand in hand_list:
                db.execute("INSERT OR IGNORE INTO tournament_hands (session_id,hand_number,data_json) VALUES (?,?,?)",
                           (session_id, hand["hand_number"], _json(hand)))
            db.execute("UPDATE tournament_sessions SET updated_utc=?, revision=?, latest_hand=MAX(latest_hand,?), status=?, winner_seat=? WHERE session_id=?",
                       (_now(), revision, hand_list[-1]["hand_number"], "complete" if winner_seat is not None else "playing",
                        winner_seat, session_id))

    def close(self, session_id: str) -> None:
        """Release a live session, retaining immutable history for replay."""
        with self._lock, self._connect() as db:
            db.execute("UPDATE tournament_sessions SET status=CASE WHEN status='complete' THEN 'complete' ELSE 'closed' END, updated_utc=? WHERE session_id=?",
                       (_now(), session_id))

    def list_sessions(self, *, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            total = db.execute("SELECT COUNT(*) FROM tournament_sessions").fetchone()[0]
            rows = db.execute("SELECT session_id,created_utc,updated_utc,status,options_json,latest_hand,revision,winner_seat FROM tournament_sessions ORDER BY updated_utc DESC, session_id DESC LIMIT ? OFFSET ?",
                              (limit, offset)).fetchall()
        return {"total": total, "offset": offset, "limit": limit,
                "sessions": [{"session_id": r[0], "created_utc": r[1], "updated_utc": r[2], "status": r[3],
                              "options": json.loads(r[4]), "latest_hand": r[5], "revision": r[6], "winner_seat": r[7]}
                             for r in rows]}

    def _require_session(self, db: sqlite3.Connection, session_id: str) -> None:
        if db.execute("SELECT 1 FROM tournament_sessions WHERE session_id=?", (session_id,)).fetchone() is None:
            raise KeyError(session_id)

    def hands(self, session_id: str, *, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            self._require_session(db, session_id)
            total = db.execute("SELECT COUNT(*) FROM tournament_hands WHERE session_id=?", (session_id,)).fetchone()[0]
            rows = db.execute("SELECT data_json FROM tournament_hands WHERE session_id=? ORDER BY hand_number DESC LIMIT ? OFFSET ?",
                              (session_id, limit, offset)).fetchall()
        return {"total": total, "offset": offset, "limit": limit,
                "hands": [json.loads(row[0]) for row in rows]}

    @staticmethod
    def _public_replay(action: Mapping[str, Any]) -> dict[str, Any]:
        # Return hero hole only for the human's own turn. Never expose AI holes.
        result = {k: v for k, v in action.items() if k != "actor_hole"}
        result["hero_hole"] = action.get("actor_hole", []) if action["seat"] == 0 else []
        return result

    def replay(self, session_id: str, hand_number: int) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            self._require_session(db, session_id)
            rows = db.execute("SELECT data_json FROM tournament_actions WHERE session_id=? AND hand_number=? ORDER BY action_no",
                              (session_id, hand_number)).fetchall()
            outcome = db.execute("SELECT data_json FROM tournament_hands WHERE session_id=? AND hand_number=?",
                                 (session_id, hand_number)).fetchone()
        if not rows and outcome is None:
            raise KeyError("hand")
        return {"session_id": session_id, "hand_number": hand_number,
                "actions": [self._public_replay(json.loads(r[0])) for r in rows],
                "settlement": json.loads(outcome[0]) if outcome is not None else None,
                "complete": outcome is not None}

    def actor_export_rows(self, session_id: str) -> list[dict[str, Any]]:
        """Each sample contains solely its acting player's own hole, public state,
        legal action IDs and post-hand chip reward (never opponents' hole/deck).
        Intended as an interchange contract, not trained PPO rollouts.
        """
        with self._lock, self._connect() as db:
            self._require_session(db, session_id)
            rows = db.execute("SELECT a.hand_number,a.action_no,a.data_json,h.data_json FROM tournament_actions a LEFT JOIN tournament_hands h ON a.session_id=h.session_id AND a.hand_number=h.hand_number WHERE a.session_id=? ORDER BY a.hand_number,a.action_no",
                              (session_id,)).fetchall()
        out: list[dict[str, Any]] = []
        for hand_no, action_no, raw_action, raw_hand in rows:
            action = json.loads(raw_action)
            hand = json.loads(raw_hand) if raw_hand is not None else None
            out.append({
                "schema": "tournament.actor_transition.v1", "session_id": session_id,
                "hand_number": hand_no, "action_no": action_no,
                "acting_seat": action["seat"],
                "observation": {**action["public_before"], "own_hole_cards": action["actor_hole"]},
                "action": {"index": action["index"], "kind": action["kind"],
                           "raise_to": action["raise_to"]},
                "hand_settled": hand is not None,
                "payoff_chips": hand["payoffs_chips"][action["seat"]] if hand is not None else None,
            })
        return out
