"""Live, optional-tqdm terminal progress for Phase11 training.

Observes JSON event callbacks, never touches the Actor, optimizers, poker RNG,
checkpoints, or the metrics.jsonl event stream. A heartbeat reports remaining
wall-clock budget even while an update/evaluation is taking a long time.
"""
from __future__ import annotations

import math
import sys
import threading
import time
from typing import Callable, TextIO


def format_duration(seconds: float | None) -> str:
    """Display a human-readable conservative duration, or -- if unknown."""
    if seconds is None or not math.isfinite(seconds):
        return '--:--:--'
    sec = max(0, math.ceil(seconds))
    hours, rem = divmod(sec, 3600)
    minutes, sec = divmod(rem, 60)
    return f'{hours:02d}:{minutes:02d}:{sec:02d}'


class TrainingProgress:
    """Live display fed by the *existing* run_experiment emit callback.

    Progress is measured by additional updates if max_updates is supplied,
    otherwise by the wall-clock budget. `time_left` always means remaining
    invocation budget, NOT guaranteed time until the job ends (evaluations and
    stopping at boundaries can cause overruns). Update ETA is an estimate.
    """

    def __init__(self, *, duration_seconds: float, max_updates: int | None,
                 interval: float = 1.0, stream: TextIO | None = None,
                 clock: Callable[[], float] | None = None,
                 threaded: bool = True, prefer_tqdm: bool = True) -> None:
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError('duration_seconds must be finite and positive')
        if max_updates is not None and (type(max_updates) is not int or max_updates <= 0):
            raise ValueError('max_updates must be None or positive integer')
        if not math.isfinite(interval) or interval < 0.1:
            raise ValueError('interval must be >= 0.1 second')
        self.duration_seconds = float(duration_seconds)
        self.max_updates = max_updates
        self.interval = float(interval)
        self.stream = stream if stream is not None else sys.stderr
        self.clock = clock or time.monotonic
        self.threaded = threaded
        self.prefer_tqdm = prefer_tqdm
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._start: float | None = None
        self._initial_updates = 0
        self._updates = 0
        self._total_updates = 0
        self._hands_per_second: float | None = None
        self._last_update_seconds: float | None = None
        self._status = 'starting'
        self._done = False
        self._bar = None
        self._last_plain_chars = 0

    def __enter__(self) -> 'TrainingProgress':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def on_event(self, event: dict) -> None:
        """Consume a runner event, without changing its persisted JSON."""
        kind = event.get('kind')
        with self._lock:
            if kind in ('start', 'resume'):
                if self._start is not None:
                    raise RuntimeError('progress received a second start event')
                self._start = self.clock()
                self._initial_updates = int(event['completed_updates'])
                self._total_updates = self._initial_updates
                self._status = 'running'
                self._open_display()
                if self.threaded:
                    self._worker = threading.Thread(target=self._heartbeat, daemon=True,
                                                    name='phase11-progress')
                    self._worker.start()
            elif kind == 'update' and self._start is not None:
                self._total_updates = int(event['update'])
                self._updates = max(0, self._total_updates - self._initial_updates)
                rate = event.get('hands_per_second_training')
                if isinstance(rate, (float, int)) and math.isfinite(float(rate)):
                    self._hands_per_second = float(rate)
                update_seconds = event.get('elapsed_seconds')
                if isinstance(update_seconds, (float, int)) and not isinstance(update_seconds, bool) and math.isfinite(float(update_seconds)) and update_seconds > 0:
                    self._last_update_seconds = float(update_seconds)
                self._status = 'training'
            elif kind == 'evaluation':
                self._status = 'evaluated'
            elif kind == 'interrupt':
                self._status = 'interrupted'
            elif kind == 'end':
                self._status = str(event.get('stop_reason', 'finished'))
                self._done = True
            self._refresh()

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = max(0.0, (self.clock() - self._start)) if self._start is not None else 0.0
            time_left = max(0.0, self.duration_seconds - elapsed)
            remaining_updates = (max(0, self.max_updates - self._updates)
                                 if self.max_updates is not None else None)
            eta_updates = (elapsed / self._updates * remaining_updates
                           if remaining_updates is not None and self._updates > 0 else None)
            estimated_remaining = (min(time_left, eta_updates)
                                   if eta_updates is not None else time_left)
            fraction = (min(1.0, self._updates / self.max_updates)
                        if self.max_updates is not None else min(1.0, elapsed / self.duration_seconds))
            return dict(elapsed=elapsed, time_left=time_left,
                        update_eta=eta_updates, estimated_remaining=estimated_remaining,
                        remaining_updates=remaining_updates, fraction=fraction,
                        updates=self._updates, total_updates=self._total_updates,
                        hands_per_second=self._hands_per_second,
                        last_update_seconds=self._last_update_seconds, status=self._status)

    def _open_display(self) -> None:
        if not self.prefer_tqdm:
            return
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return
        target = (self.max_updates if self.max_updates is not None
                  else self.duration_seconds)
        unit = 'upd' if self.max_updates is not None else 's'
        self._bar = tqdm(total=target, desc='Phase 11', unit=unit,
                         file=self.stream, dynamic_ncols=True,
                         mininterval=0, leave=True, disable=False)

    def _refresh(self) -> None:
        if self._start is None:
            return
        s = self.snapshot()
        left = format_duration(s['time_left'])
        eta = format_duration(s['estimated_remaining'])
        counter = (f"{s['total_updates']} (+{s['updates']}/{self.max_updates})"
                   if self.max_updates is not None else str(s['total_updates']))
        speed = ('--' if s['hands_per_second'] is None
                 else f"{s['hands_per_second']:.1f}")
        elapsed = format_duration(s['elapsed'])
        postfix = (f"upd={counter} hands/s={speed} "
                   f"elapsed={elapsed} time_left={left} ETA={eta} stage={s['status']}")
        if self._bar is not None:
            self._bar.n = (s['updates'] if self.max_updates is not None
                           else min(self.duration_seconds, s['elapsed']))
            self._bar.set_postfix_str(postfix, refresh=False)
            self._bar.refresh()
        else:
            width = 22
            filled = round(s['fraction'] * width)
            line = '[' + '#' * filled + '-' * (width - filled) + '] '
            line += f"{100*s['fraction']:5.1f}% {postfix}"
            self.stream.write('\r' + line + ' ' * max(0, self._last_plain_chars - len(line)))
            self.stream.flush()
            self._last_plain_chars = len(line)
            if self._done:
                self.stream.write('\n')
                self.stream.flush()
                self._last_plain_chars = 0

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                if self._done:
                    return
                self._refresh()

    def message(self, message: str) -> None:
        """Print a non-progress message without leaving a broken bar."""
        with self._lock:
            if self._bar is not None:
                self._bar.clear()
            elif self._last_plain_chars:
                self.stream.write('\r' + ' ' * self._last_plain_chars + '\r')
                self.stream.flush()
            print(message, flush=True)
            if self._start is not None and not self._done:
                self._refresh()

    def close(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=max(0.5, self.interval * 2))
            self._worker = None
        with self._lock:
            if self._start is not None:
                self._refresh()
            if self._bar is not None:
                self._bar.close()
                self._bar = None
            elif self._last_plain_chars:
                self.stream.write('\n')
                self.stream.flush()
                self._last_plain_chars = 0
