"""
DrowSAFE — Event logger.

Writes drowsiness events to a timestamped CSV file in the logs/ directory.
A new file is created each time DrowSAFE starts.
"""

import csv
import os
import time
import logging
from datetime import datetime
from collections import Counter

from config.config import LOG_DIR, LOG_EVENTS
from src.state_machine import LEVEL_NAMES

log = logging.getLogger("drowsafe.logger")


class EventLogger:
    """
    Logs alert state transitions and periodic fatigue score snapshots
    to a CSV file for post-session analysis.

    CSV columns
    -----------
    timestamp, elapsed_s, alert_level, alert_name,
    fatigue_score, ear, mar, head_pitch
    """

    SNAPSHOT_INTERVAL = 5.0  # Write a row every N seconds regardless of state

    def __init__(self):
        self._file    = None
        self._writer  = None
        self._last_level = -1
        self._last_snap  = 0.0
        self._start_time = time.monotonic()
        self._start_wall_time = datetime.now()
        self._filepath = None
        self._summary_path = None
        self._samples = 0
        self._score_total = 0.0
        self._max_score = 0.0
        self._level_counts = Counter()
        self._reason_counts = Counter()

        if LOG_EVENTS:
            self._open_file()

    def _open_file(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(LOG_DIR, f"drowsafe_{ts}.csv")
        self._filepath = filepath
        self._summary_path = os.path.join(LOG_DIR, f"trip_summary_{ts}.txt")

        self._file   = open(filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "timestamp", "elapsed_s", "alert_level", "alert_name",
            "fatigue_score", "reason", "ear", "mar", "head_pitch",
        ])
        log.info("Event log: %s", filepath)

    def log(self, alert_level: int, score: float, features, reason: str = ""):
        """
        Write a row on state change or periodic snapshot.

        Parameters
        ----------
        alert_level : int
        score       : float
        features    : Features | None
        """
        now     = time.monotonic()
        elapsed = round(now - self._start_time, 2)
        self._samples += 1
        self._score_total += score
        self._max_score = max(self._max_score, score)
        self._level_counts[alert_level] += 1
        if alert_level > 0 and reason:
            self._reason_counts[reason] += 1

        if not LOG_EVENTS or self._writer is None:
            return

        state_changed = alert_level != self._last_level
        snapshot_due  = (now - self._last_snap) >= self.SNAPSHOT_INTERVAL

        if not (state_changed or snapshot_due):
            return

        ear   = round(features.ear,        3) if features else ""
        mar   = round(features.mar,        3) if features else ""
        pitch = round(features.head_pitch, 1) if features else ""

        self._writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            elapsed,
            alert_level,
            LEVEL_NAMES[alert_level],
            score,
            reason,
            ear, mar, pitch,
        ])
        self._file.flush()

        self._last_level = alert_level
        self._last_snap  = now

    def close(self):
        self._write_summary()
        if self._file:
            self._file.close()
            log.info("Event log closed.")

    def _write_summary(self):
        if not LOG_EVENTS or not self._summary_path or self._samples == 0:
            return

        duration = time.monotonic() - self._start_time
        avg_score = self._score_total / self._samples
        level_total = sum(self._level_counts.values()) or 1

        lines = [
            "DrowSAFE Trip Summary",
            "=" * 22,
            f"Started: {self._start_wall_time.isoformat(timespec='seconds')}",
            f"Ended: {datetime.now().isoformat(timespec='seconds')}",
            f"Duration: {duration:.1f} seconds",
            f"Samples: {self._samples}",
            f"Average fatigue score: {avg_score:.1f}",
            f"Maximum fatigue score: {self._max_score:.1f}",
            "",
            "Alert level distribution:",
        ]

        for level, name in LEVEL_NAMES.items():
            pct = 100.0 * self._level_counts[level] / level_total
            lines.append(f"- {name}: {pct:.1f}%")

        lines.extend(["", "Top alert reasons:"])
        if self._reason_counts:
            for reason, count in self._reason_counts.most_common(5):
                lines.append(f"- {reason}: {count} samples")
        else:
            lines.append("- No warning or critical alert reasons recorded")

        if self._filepath:
            lines.extend(["", f"Event log: {self._filepath}"])

        with open(self._summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        log.info("Trip summary: %s", self._summary_path)

    @property
    def is_enabled(self) -> bool:
        return LOG_EVENTS and self._writer is not None

    @property
    def log_path(self):
        return self._filepath

    @property
    def summary_path(self):
        return self._summary_path
