"""Alerts, dwell timers, CSV logging, and optional system beep."""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


ISSUE_LABELS = {
    "chin_rest": "托腮",
    "forward_head": "颈前倾",
    "head_tilt": "头侧倾",
    "too_close": "萤幕过近",
}

ISSUE_MESSAGES = {
    "chin_rest": "请放下托腮，双手离开脸颊",
    "forward_head": "请坐直一点，收下巴、放松颈部",
    "head_tilt": "请保持头部端正，避免侧歪",
    "too_close": "请拉远一点，保持舒适的萤幕距离",
}


@dataclass
class IssueTimer:
    """Track how long each issue has been continuously active."""

    started_at: Dict[str, float] = field(default_factory=dict)
    alerted: Set[str] = field(default_factory=set)
    today_seconds: Dict[str, float] = field(default_factory=dict)
    _day: date = field(default_factory=date.today)
    _last_tick: Optional[float] = None

    def update(self, active: Dict[str, bool], dwell_seconds: float) -> List[str]:
        """
        Update timers. Return list of issue keys that newly crossed dwell threshold.
        Also accumulates today's bad-posture seconds while active.
        """
        now = time.monotonic()
        self._roll_day()
        newly_alerted: List[str] = []

        dt = 0.0 if self._last_tick is None else max(0.0, now - self._last_tick)
        self._last_tick = now

        for key, is_on in active.items():
            if is_on:
                if key not in self.started_at:
                    self.started_at[key] = now
                    self.alerted.discard(key)
                self.today_seconds[key] = self.today_seconds.get(key, 0.0) + dt
                elapsed = now - self.started_at[key]
                if elapsed >= dwell_seconds and key not in self.alerted:
                    self.alerted.add(key)
                    newly_alerted.append(key)
            else:
                self.started_at.pop(key, None)
                self.alerted.discard(key)

        return newly_alerted

    def elapsed(self, key: str) -> float:
        if key not in self.started_at:
            return 0.0
        return time.monotonic() - self.started_at[key]

    def _roll_day(self) -> None:
        today = date.today()
        if today != self._day:
            self._day = today
            self.today_seconds = {}


class AlertLogger:
    """Append alert events to data/posture_log.csv."""

    HEADER = ["timestamp", "issue", "label", "duration_sec"]

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.HEADER)

    def log(self, issue: str, duration_sec: float) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    issue,
                    ISSUE_LABELS.get(issue, issue),
                    f"{duration_sec:.1f}",
                ]
            )


def play_beep(enabled: bool = True) -> None:
    """Best-effort local beep (macOS afplay / Linux beep / Windows winsound)."""
    if not enabled:
        return
    try:
        if sys.platform == "darwin":
            subprocess.Popen(
                ["afplay", "/System/Library/Sounds/Ping.aiff"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif sys.platform.startswith("linux"):
            subprocess.Popen(
                ["printf", "\a"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif sys.platform == "win32":
            import winsound

            winsound.MessageBeep()
    except OSError:
        pass
