"""
DrowSAFE — Fatigue scoring module.

Computes a composite fatigue score (0–100) from:
  - PERCLOS (primary signal)
  - Instantaneous EAR
  - Yawn frequency (MAR)
  - Head pose (pitch / nod)

The score drives the alert state machine.
"""

import time
import logging
from collections import deque
from dataclasses import dataclass
from config.config import (
    EAR_THRESHOLD,
    EAR_CONSEC_FRAMES,
    EAR_RECOVERY_FRAMES,
    MAR_THRESHOLD,
    MAR_CONSEC_FRAMES,
    HEAD_PITCH_THRESHOLD,
    HEAD_NOD_CONSEC_FRAMES,
    PERCLOS_WINDOW_SEC,
    PERCLOS_THRESHOLD,
    SCORE_WEIGHT_PERCLOS,
    SCORE_WEIGHT_EAR,
    SCORE_WEIGHT_MAR,
    SCORE_WEIGHT_HEAD_POSE,
    FACE_MISSING_WARNING_SEC,
    FACE_MISSING_CRITICAL_SEC,
    FACE_MISSING_WARNING_SCORE,
    FACE_MISSING_CRITICAL_SCORE,
)

log = logging.getLogger("drowsafe.scoring")


@dataclass
class ScoreDetails:
    """Human-readable explanation for the latest fatigue score."""
    perclos: float = 0.0
    eye_closed: bool = False
    yawning: bool = False
    head_nod: bool = False
    face_missing: bool = False
    missing_for: float = 0.0
    yawns_per_min: float = 0.0
    nods_per_min: float = 0.0
    reasons: tuple = ()

    @property
    def primary_reason(self) -> str:
        return self.reasons[0] if self.reasons else "No fatigue signs detected"


class FatigueScorer:
    """
    Maintains a rolling window of per-frame signals and produces a
    single fatigue score in the range [0, 100].

    Score interpretation
    --------------------
    0  – 39  : Alert
    40 – 69  : Warning
    70 – 100 : Critical
    """

    def __init__(self):
        # PERCLOS rolling window: stores (timestamp, eye_closed) tuples
        self._eye_history: deque = deque()

        # Yawn counter over the same window
        self._yawn_history: deque = deque()

        # Head nod counter
        self._nod_history: deque  = deque()

        self._last_score: float   = 0.0
        self._yawn_in_progress    = False
        self._nod_in_progress     = False
        self._ear_low_count       = 0    # consecutive low-EAR frames
        self._ear_high_count      = 0    # consecutive high-EAR frames (grace period)
        self._mar_high_count      = 0    # consecutive high-MAR frames
        self._nod_high_count      = 0    # consecutive head-down frames
        self._missing_since       = None # first timestamp of current no-face/no-frame run
        self._details             = ScoreDetails()

        log.info("FatigueScorer ready (PERCLOS window=%ds)", PERCLOS_WINDOW_SEC)

    def update(self, features) -> float:
        """
        Update rolling history with the latest frame's features and
        return the current fatigue score.

        Parameters
        ----------
        features : Features | None
            Extracted features from the current frame.
            If None (no face/frame visible), the scorer holds briefly and then
            escalates according to the configured visibility timeout.

        Returns
        -------
        float
            Fatigue score in [0, 100].
        """
        now = time.monotonic()

        if features is None or not getattr(features, "face_visible", True):
            # Face/camera missing: hold briefly, then escalate instead of
            # freezing a stale healthy score forever.
            if self._missing_since is None:
                self._missing_since = now
            missing_for = now - self._missing_since

            if missing_for >= FACE_MISSING_CRITICAL_SEC:
                self._last_score = max(self._last_score, FACE_MISSING_CRITICAL_SCORE)
            elif missing_for >= FACE_MISSING_WARNING_SEC:
                self._last_score = max(self._last_score, FACE_MISSING_WARNING_SCORE)

            self._details = ScoreDetails(
                perclos=self._compute_perclos(),
                face_missing=True,
                missing_for=missing_for,
                reasons=("Face or camera unavailable",),
            )
            return self._last_score

        self._missing_since = None

        # --- Eye closed? (blink filter) ---
        if features.ear < EAR_THRESHOLD:
            self._ear_low_count  += 1
            self._ear_high_count  = 0
        else:
            self._ear_high_count += 1
            if self._ear_high_count >= EAR_RECOVERY_FRAMES:
                self._ear_low_count  = 0
                self._ear_high_count = 0
        eye_closed = self._ear_low_count >= EAR_CONSEC_FRAMES
        self._eye_history.append((now, eye_closed))

        # --- Yawning? (sustained threshold + falling edge event) ---
        yawning = features.mar > MAR_THRESHOLD
        if yawning:
            self._mar_high_count += 1
            if self._mar_high_count >= MAR_CONSEC_FRAMES:
                self._yawn_in_progress = True
        else:
            if self._yawn_in_progress:
                self._yawn_history.append(now)  # Record completed sustained yawn
            self._yawn_in_progress = False
            self._mar_high_count = 0

        # --- Nodding? (sustained threshold + falling edge event) ---
        nodding = features.head_pitch > HEAD_PITCH_THRESHOLD
        if nodding:
            self._nod_high_count += 1
            if self._nod_high_count >= HEAD_NOD_CONSEC_FRAMES:
                self._nod_in_progress = True
        else:
            if self._nod_in_progress:
                self._nod_history.append(now)
            self._nod_in_progress = False
            self._nod_high_count = 0

        # --- Prune expired history entries ---
        cutoff = now - PERCLOS_WINDOW_SEC
        while self._eye_history and self._eye_history[0][0] < cutoff:
            self._eye_history.popleft()
        while self._yawn_history and self._yawn_history[0] < cutoff:
            self._yawn_history.popleft()
        while self._nod_history and self._nod_history[0] < cutoff:
            self._nod_history.popleft()

        # --- PERCLOS ---
        perclos = self._compute_perclos()

        # --- Normalised sub-scores (each 0–1) ---
        # EAR: invert and normalise so 0 = fully open, 1 = fully closed
        ear_norm      = max(0.0, min(1.0, 1.0 - (features.ear / EAR_THRESHOLD)))

        # PERCLOS: normalise relative to threshold (1.0 = at threshold, >1 = above)
        perclos_norm  = min(1.0, perclos / max(PERCLOS_THRESHOLD, 1e-6))

        # MAR / yawn frequency: yawns per minute, capped at 1.0 above 6/min
        yawns_per_min = len(self._yawn_history) / (PERCLOS_WINDOW_SEC / 60.0)
        mar_norm      = min(1.0, yawns_per_min / 6.0)

        # Head pose: nods per minute, capped at 1.0 above 10/min
        nods_per_min  = len(self._nod_history) / (PERCLOS_WINDOW_SEC / 60.0)
        pose_norm     = min(1.0, nods_per_min / 10.0)

        # --- Composite weighted score ---
        raw = (
            SCORE_WEIGHT_PERCLOS   * perclos_norm +
            SCORE_WEIGHT_EAR       * ear_norm     +
            SCORE_WEIGHT_MAR       * mar_norm     +
            SCORE_WEIGHT_HEAD_POSE * pose_norm
        )

        score = round(min(100.0, max(0.0, raw * 100.0)), 1)
        reasons = []
        if eye_closed:
            reasons.append("Eyes closed")
        if perclos >= PERCLOS_THRESHOLD:
            reasons.append("High eye-closure rate")
        if self._yawn_in_progress or features.mar > MAR_THRESHOLD:
            reasons.append("Yawning detected")
        if self._nod_in_progress or features.head_pitch > HEAD_PITCH_THRESHOLD:
            reasons.append("Head nod detected")
        if score >= 40 and not reasons:
            reasons.append("Elevated fatigue score")

        self._details = ScoreDetails(
            perclos=perclos,
            eye_closed=eye_closed,
            yawning=self._yawn_in_progress or features.mar > MAR_THRESHOLD,
            head_nod=self._nod_in_progress or features.head_pitch > HEAD_PITCH_THRESHOLD,
            yawns_per_min=yawns_per_min,
            nods_per_min=nods_per_min,
            reasons=tuple(reasons),
        )
        self._last_score = score
        return score

    def _compute_perclos(self) -> float:
        """
        Compute PERCLOS from the rolling eye closure history.

        Returns fraction of frames where eyes were closed, in [0, 1].
        """
        if not self._eye_history:
            return 0.0
        closed = sum(1 for _, c in self._eye_history if c)
        return closed / len(self._eye_history)

    @property
    def perclos(self) -> float:
        """Current PERCLOS value (read-only)."""
        return self._compute_perclos()

    @property
    def details(self) -> ScoreDetails:
        """Explanation for the latest score (read-only)."""
        return self._details

    def reset(self):
        """Clear all rolling history (e.g. driver change)."""
        self._eye_history.clear()
        self._yawn_history.clear()
        self._nod_history.clear()
        self._last_score = 0.0
        self._ear_low_count  = 0
        self._ear_high_count = 0
        self._mar_high_count = 0
        self._nod_high_count = 0
        self._yawn_in_progress = False
        self._nod_in_progress = False
        self._missing_since = None
        self._details = ScoreDetails()
        log.info("FatigueScorer reset.")
