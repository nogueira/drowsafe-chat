"""
DrowSAFE guided calibration.

Collects short feature samples from a driver and recommends personalised
thresholds for EAR, MAR, and head pitch.
"""

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from statistics import median

from config.config import LOG_DIR


@dataclass(frozen=True)
class CalibrationStep:
    key: str
    title: str
    instruction: str
    duration_sec: float


class GuidedCalibrator:
    """Small state machine for an in-cabin guided calibration session."""

    STEPS = (
        CalibrationStep(
            "neutral",
            "Neutral face",
            "Look straight ahead with eyes open and mouth relaxed.",
            8.0,
        ),
        CalibrationStep(
            "eyes_closed",
            "Eye closure",
            "Close your eyes for a moment, then blink naturally.",
            6.0,
        ),
        CalibrationStep(
            "yawn",
            "Mouth opening",
            "Open your mouth wide as if yawning.",
            6.0,
        ),
        CalibrationStep(
            "head_nod",
            "Head nod",
            "Let your head tilt forward like a drowsy nod.",
            6.0,
        ),
    )

    def __init__(self):
        self._step_index = 0
        self._step_started = None
        self._samples = {step.key: [] for step in self.STEPS}
        self._recommendations = None
        self._saved_path = None

    @property
    def is_complete(self) -> bool:
        return self._step_index >= len(self.STEPS)

    @property
    def current_step(self):
        if self.is_complete:
            return None
        return self.STEPS[self._step_index]

    @property
    def progress(self) -> float:
        if self.is_complete or self._step_started is None:
            return 1.0 if self.is_complete else 0.0
        elapsed = time.monotonic() - self._step_started
        return max(0.0, min(1.0, elapsed / self.current_step.duration_sec))

    @property
    def recommendations(self):
        if self._recommendations is None:
            self._recommendations = self._compute_recommendations()
        return self._recommendations

    @property
    def saved_path(self):
        return self._saved_path

    def update(self, features):
        if self.is_complete:
            return

        now = time.monotonic()
        if self._step_started is None:
            self._step_started = now

        if features is not None:
            self._samples[self.current_step.key].append(features)

        if now - self._step_started >= self.current_step.duration_sec:
            self._step_index += 1
            self._step_started = now if not self.is_complete else None
            if self.is_complete:
                self._recommendations = self._compute_recommendations()

    def report_lines(self):
        rec = self.recommendations
        lines = [
            "Calibration complete",
            f"EAR threshold: {rec['EAR_THRESHOLD']:.3f}",
            f"MAR threshold: {rec['MAR_THRESHOLD']:.3f}",
            f"Head pitch threshold: {rec['HEAD_PITCH_THRESHOLD']:.1f}",
        ]
        if self._saved_path:
            lines.append(f"Saved to: {self._saved_path}")
        return lines

    def save(self, log_dir: str = LOG_DIR):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"calibration_{ts}.json")
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "recommendations": self.recommendations,
            "sample_counts": {key: len(value) for key, value in self._samples.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self._saved_path = path
        return path

    def _compute_recommendations(self):
        neutral = self._samples["neutral"]
        eyes_closed = self._samples["eyes_closed"]
        yawn = self._samples["yawn"]
        head_nod = self._samples["head_nod"]

        neutral_ear = _median_attr(neutral, "ear", fallback=0.17)
        closed_ear = _low_attr(eyes_closed, "ear", fallback=neutral_ear * 0.55)
        ear_threshold = _clamp((neutral_ear + closed_ear) / 2.0, 0.08, 0.28)

        neutral_mar = _median_attr(neutral, "mar", fallback=0.12)
        yawn_mar = _high_attr(yawn, "mar", fallback=max(0.45, neutral_mar * 3.0))
        mar_threshold = _clamp((neutral_mar + yawn_mar) / 2.0, 0.25, 0.75)

        neutral_pitch = _median_attr(neutral, "head_pitch", fallback=0.0)
        nod_pitch = _high_attr(head_nod, "head_pitch", fallback=neutral_pitch + 20.0)
        pitch_threshold = _clamp((neutral_pitch + nod_pitch) / 2.0, 10.0, 35.0)

        return {
            "EAR_THRESHOLD": round(ear_threshold, 3),
            "MAR_THRESHOLD": round(mar_threshold, 3),
            "HEAD_PITCH_THRESHOLD": round(pitch_threshold, 1),
            "baselines": {
                "neutral_ear": round(neutral_ear, 3),
                "closed_ear": round(closed_ear, 3),
                "neutral_mar": round(neutral_mar, 3),
                "yawn_mar": round(yawn_mar, 3),
                "neutral_pitch": round(neutral_pitch, 1),
                "nod_pitch": round(nod_pitch, 1),
            },
        }


def _median_attr(samples, attr, fallback):
    values = [getattr(sample, attr) for sample in samples if sample is not None]
    return median(values) if values else fallback


def _low_attr(samples, attr, fallback):
    values = sorted(getattr(sample, attr) for sample in samples if sample is not None)
    if not values:
        return fallback
    return values[max(0, int(len(values) * 0.1) - 1)]


def _high_attr(samples, attr, fallback):
    values = sorted(getattr(sample, attr) for sample in samples if sample is not None)
    if not values:
        return fallback
    return values[min(len(values) - 1, int(len(values) * 0.9))]


def _clamp(value, low, high):
    return max(low, min(high, value))
