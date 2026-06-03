"""Unit tests for face-parameter calibration recommendations."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.calibration import calibrate_face_parameters
from src.features import Features


def make_features(ear=0.30, mar=0.12, pitch=0.0):
    return Features(
        ear=ear,
        ear_left=ear,
        ear_right=ear,
        mar=mar,
        head_pitch=pitch,
        head_yaw=0.0,
        head_roll=0.0,
        face_visible=True,
    )


def test_calibrate_face_parameters_from_separated_samples():
    samples = {
        "neutral": [make_features(ear=0.30, mar=0.10, pitch=1.0) for _ in range(30)],
        "eyes_closed": [make_features(ear=0.07, mar=0.10, pitch=1.0) for _ in range(30)],
        "yawn": [make_features(ear=0.30, mar=0.72, pitch=1.0) for _ in range(30)],
        "head_nod": [make_features(ear=0.30, mar=0.10, pitch=26.0) for _ in range(30)],
    }

    rec = calibrate_face_parameters(samples)

    assert 0.10 < rec["EAR_THRESHOLD"] < 0.25
    assert 0.35 < rec["MAR_THRESHOLD"] < 0.65
    assert 10.0 <= rec["HEAD_PITCH_THRESHOLD"] < 25.0
    assert rec["confidence"] == "high"
    assert rec["warnings"] == ()


def test_calibrate_face_parameters_warns_when_samples_overlap():
    samples = {
        "neutral": [make_features(ear=0.20, mar=0.30, pitch=5.0) for _ in range(20)],
        "eyes_closed": [make_features(ear=0.21, mar=0.30, pitch=5.0) for _ in range(20)],
        "yawn": [make_features(ear=0.20, mar=0.28, pitch=5.0) for _ in range(20)],
        "head_nod": [make_features(ear=0.20, mar=0.30, pitch=4.0) for _ in range(20)],
    }

    rec = calibrate_face_parameters(samples)

    assert rec["confidence"] == "low"
    assert any("Eye calibration" in warning for warning in rec["warnings"])
    assert any("Mouth calibration" in warning for warning in rec["warnings"])
    assert any("Head nod calibration" in warning for warning in rec["warnings"])


def test_calibrate_face_parameters_reports_low_sample_count():
    samples = {
        "neutral": [make_features()],
        "eyes_closed": [make_features(ear=0.07)],
        "yawn": [make_features(mar=0.70)],
        "head_nod": [make_features(pitch=25.0)],
    }

    rec = calibrate_face_parameters(samples, min_samples=5)

    assert rec["confidence"] == "low"
    assert rec["sample_counts"]["neutral"] == 1
    assert any("Low sample count" in warning for warning in rec["warnings"])
