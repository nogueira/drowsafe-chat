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


DEFAULT_FACE_CALIBRATION = {
    "EAR_THRESHOLD": 0.13,
    "MAR_THRESHOLD": 0.45,
    "HEAD_PITCH_THRESHOLD": 20.0,
}


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
            f"Confidence: {rec['confidence']}",
        ]
        if rec["warnings"]:
            lines.append(f"Warning: {rec['warnings'][0]}")
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
        return calibrate_face_parameters(self._samples)


def calibrate_face_parameters(samples_by_phase, min_samples: int = 15):
    """
    Compute personalised face thresholds from guided calibration samples.

    Expected phases:
      neutral      - eyes open, relaxed mouth, upright head
      eyes_closed  - deliberate eye closure / slow blinks
      yawn         - mouth open as if yawning
      head_nod     - head tilted forward

    Returns a JSON-serialisable dict with threshold recommendations,
    measured baselines, sample counts, quality warnings, and confidence.
    """
    neutral = list(samples_by_phase.get("neutral", []))
    eyes_closed = list(samples_by_phase.get("eyes_closed", []))
    yawn = list(samples_by_phase.get("yawn", []))
    head_nod = list(samples_by_phase.get("head_nod", []))
    warnings = []

    neutral_ear = _median_attr(neutral, "ear", fallback=0.17)
    open_ear_floor = _low_attr(neutral, "ear", fallback=neutral_ear)
    closed_ear = _low_attr(eyes_closed, "ear", fallback=neutral_ear * 0.55)
    if closed_ear >= open_ear_floor:
        warnings.append("Eye calibration samples are not well separated")
        ear_threshold = DEFAULT_FACE_CALIBRATION["EAR_THRESHOLD"]
    else:
        # Bias slightly toward the closed-eye side so normal open-eye variance
        # does not trigger false positives.
        ear_threshold = closed_ear + (open_ear_floor - closed_ear) * 0.45
    ear_threshold = _clamp(ear_threshold, 0.08, 0.28)

    neutral_mar = _median_attr(neutral, "mar", fallback=0.12)
    resting_mar_ceiling = _high_attr(neutral, "mar", fallback=neutral_mar)
    yawn_mar = _high_attr(yawn, "mar", fallback=max(0.45, neutral_mar * 3.0))
    if yawn_mar <= resting_mar_ceiling:
        warnings.append("Mouth calibration samples are not well separated")
        mar_threshold = DEFAULT_FACE_CALIBRATION["MAR_THRESHOLD"]
    else:
        mar_threshold = resting_mar_ceiling + (yawn_mar - resting_mar_ceiling) * 0.55
    mar_threshold = _clamp(mar_threshold, 0.25, 0.75)

    neutral_pitch = _median_attr(neutral, "head_pitch", fallback=0.0)
    neutral_pitch_ceiling = _high_attr(neutral, "head_pitch", fallback=neutral_pitch)
    nod_pitch = _high_attr(head_nod, "head_pitch", fallback=neutral_pitch + 20.0)
    if nod_pitch <= neutral_pitch_ceiling:
        warnings.append("Head nod calibration samples are not well separated")
        pitch_threshold = DEFAULT_FACE_CALIBRATION["HEAD_PITCH_THRESHOLD"]
    else:
        pitch_threshold = neutral_pitch_ceiling + (nod_pitch - neutral_pitch_ceiling) * 0.55
    pitch_threshold = _clamp(pitch_threshold, 10.0, 35.0)

    sample_counts = {
        "neutral": len(neutral),
        "eyes_closed": len(eyes_closed),
        "yawn": len(yawn),
        "head_nod": len(head_nod),
    }
    for phase, count in sample_counts.items():
        if count < min_samples:
            warnings.append(f"Low sample count for {phase}: {count}")

    confidence = _calibration_confidence(sample_counts, min_samples, warnings)
    return {
        "EAR_THRESHOLD": round(ear_threshold, 3),
        "MAR_THRESHOLD": round(mar_threshold, 3),
        "HEAD_PITCH_THRESHOLD": round(pitch_threshold, 1),
        "baselines": {
            "neutral_ear": round(neutral_ear, 3),
            "open_ear_floor": round(open_ear_floor, 3),
            "closed_ear": round(closed_ear, 3),
            "neutral_mar": round(neutral_mar, 3),
            "resting_mar_ceiling": round(resting_mar_ceiling, 3),
            "yawn_mar": round(yawn_mar, 3),
            "neutral_pitch": round(neutral_pitch, 1),
            "neutral_pitch_ceiling": round(neutral_pitch_ceiling, 1),
            "nod_pitch": round(nod_pitch, 1),
        },
        "sample_counts": sample_counts,
        "confidence": confidence,
        "warnings": tuple(warnings),
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


def _calibration_confidence(sample_counts, min_samples, warnings):
    if not sample_counts:
        return "low"
    sample_score = min(sample_counts.values()) / max(min_samples, 1)
    if warnings or sample_score < 0.75:
        return "low"
    if sample_score < 1.25:
        return "medium"
    return "high"
