"""
DrowSAFE — main entry point.

Starts the full detection pipeline:
  Camera → Detector → Features → Scoring → State Machine → Dashboard + Alert

Run modes
---------
  Normal   : python src/main.py
  Simulate : python src/main.py --simulate
             Runs the full dashboard with synthetic fatigue data.
             No camera or MediaPipe required. Use while awaiting hardware.
"""

import sys
import os
import time
import math
import signal
import logging
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import (
    FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE,
    PROCESS_WIDTH, PROCESS_HEIGHT,
    NATIVE_THREAD_LIMIT,
    CAMERA_FLIP,
    DISPLAY_WIDTH, DISPLAY_HEIGHT, FULLSCREEN,
    SHOW_LANDMARKS, SHOW_FPS,
    DETECTOR_REFINE_LANDMARKS,
    DETECTOR_MIN_DETECTION_CONF,
    DETECTOR_MIN_TRACKING_CONF,
)

# Keep native libraries from oversubscribing the Pi 5 CPU. These environment
# variables must be set before importing modules that load NumPy/OpenCV.
os.environ.setdefault("OMP_NUM_THREADS", str(NATIVE_THREAD_LIMIT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(NATIVE_THREAD_LIMIT))
os.environ.setdefault("MKL_NUM_THREADS", str(NATIVE_THREAD_LIMIT))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(NATIVE_THREAD_LIMIT))

from src.camera        import Camera
from src.scoring       import FatigueScorer
from src.state_machine import AlertStateMachine
from src.alert         import AlertController
from src.dashboard     import Dashboard
from src.logger        import EventLogger
from src.features      import Features
from src.calibration   import GuidedCalibrator
from src.health        import run_startup_checks
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("drowsafe.main")


# ---------------------------------------------------------------------------
# Synthetic feature generator for simulation mode
# ---------------------------------------------------------------------------
def _simulate_features(t: float) -> Features:
    """
    Generate realistic-looking drowsiness features that slowly cycle
    through alert → warning → critical → recovery over ~90 seconds.
    Useful for demoing the full dashboard without a camera.
    """
    cycle = t % 90.0  # 90-second demo cycle

    if cycle < 30:
        # Phase 1: alert driver (0–30s)
        ear   = 0.30 + 0.04 * math.sin(t * 2)
        mar   = 0.20 + 0.05 * math.sin(t * 1.3)
        pitch = 2.0  * math.sin(t * 0.5)
    elif cycle < 55:
        # Phase 2: growing drowsiness (30–55s)
        p     = (cycle - 30) / 25.0
        ear   = 0.30 - p * 0.12 + 0.02 * math.sin(t * 3)
        mar   = 0.20 + p * 0.30 + 0.05 * math.sin(t * 0.8)
        pitch = p * 15.0 + 2.0 * math.sin(t * 0.4)
    elif cycle < 70:
        # Phase 3: critical drowsiness (55–70s)
        ear   = 0.14 + 0.04 * math.sin(t * 4)
        mar   = 0.65 + 0.10 * math.sin(t * 0.6)
        pitch = 22.0 + 5.0 * math.sin(t * 0.3)
    else:
        # Phase 4: recovery (70–90s)
        p     = (cycle - 70) / 20.0
        ear   = 0.14 + p * 0.16 + 0.02 * math.sin(t * 2)
        mar   = 0.65 - p * 0.45
        pitch = 22.0 - p * 20.0

    return Features(
        ear        = max(0.05, min(0.45, ear)),
        ear_left   = max(0.05, min(0.45, ear + 0.01)),
        ear_right  = max(0.05, min(0.45, ear - 0.01)),
        mar        = max(0.10, min(0.90, mar)),
        head_pitch = pitch,
        head_yaw   = 3.0 * math.sin(t * 0.2),
        head_roll  = 1.5 * math.sin(t * 0.3),
        face_visible = True,
    )


def _processing_frame(frame):
    """Return a resized RGB frame for MediaPipe plus its dimensions."""
    h, w = frame.shape[:2]
    target_w = PROCESS_WIDTH or w
    target_h = PROCESS_HEIGHT or h

    if target_w == w and target_h == h:
        return frame, w, h

    resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized, target_w, target_h


def _run_guided_calibration(camera, detector, extractor, dashboard, simulate: bool):
    """Run the guided calibration flow and save recommended thresholds."""
    calibrator = GuidedCalibrator()
    fps_timer = time.perf_counter()
    fps_count = 0
    fps_display = 0.0
    start_time = time.perf_counter()
    saved = False
    complete_started = None
    last_frame = None
    last_features = None

    log.info("Guided calibration started.")

    while dashboard.is_running():
        loop_start = time.perf_counter()
        frame = camera.read()
        t = loop_start - start_time

        if frame is None:
            feat = None
            display_frame = None
        elif simulate:
            feat = _simulate_features(t)
            display_frame = frame
        else:
            detection_frame, process_w, process_h = _processing_frame(frame)
            if extractor.frame_w != process_w or extractor.frame_h != process_h:
                extractor.update_frame_size(process_w, process_h)
            landmarks, annotated_frame = detector.process(
                detection_frame,
                draw=SHOW_LANDMARKS,
                annotation_frame=frame if SHOW_LANDMARKS else None,
            )
            display_frame = annotated_frame if SHOW_LANDMARKS else frame
            feat = extractor.extract(landmarks) if landmarks else None

        if not calibrator.is_complete:
            calibrator.update(feat)
            step = calibrator.current_step
            title = step.title if step else "Calibration complete"
            instruction = step.instruction if step else "Review the recommended thresholds."
            recommendation_lines = None
        else:
            if not saved:
                calibrator.save()
                log.info("Calibration recommendations saved: %s", calibrator.saved_path)
                saved = True
                complete_started = time.perf_counter()
            title = "Calibration complete"
            instruction = "Review the recommended thresholds."
            recommendation_lines = calibrator.report_lines()

        if frame is not None:
            last_frame = display_frame
            last_features = feat

        fps_count += 1
        elapsed = time.perf_counter() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count = 0
            fps_timer = time.perf_counter()

        dashboard.render_calibration(
            frame=display_frame if frame is not None else last_frame,
            features=feat if frame is not None else last_features,
            step_title=title,
            instruction=instruction,
            progress=calibrator.progress,
            recommendation_lines=recommendation_lines,
            fps=fps_display if SHOW_FPS else None,
        )

        if complete_started and time.perf_counter() - complete_started >= 10.0:
            break

    log.info("Guided calibration stopped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DrowSAFE driver drowsiness detection")
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run in simulation mode (no camera or MediaPipe required)"
    )
    parser.add_argument(
        "--camera-index", type=int, default=None,
        help="Camera index for OpenCV webcam capture (default from config)"
    )
    parser.add_argument(
        "--windowed", action="store_true",
        help="Run dashboard in a window instead of fullscreen"
    )
    parser.add_argument(
        "--no-flip", action="store_true",
        help="Disable 180-degree camera flip from config"
    )
    parser.add_argument(
        "--guided-calibration", action="store_true",
        help="Run the guided calibration flow and save threshold recommendations"
    )
    args = parser.parse_args()

    simulate = args.simulate
    camera_index = args.camera_index if args.camera_index is not None else None
    fullscreen = FULLSCREEN and not args.windowed
    camera_flip = CAMERA_FLIP and not args.no_flip

    log.info("DrowSAFE starting... (mode=%s)", "SIMULATE" if simulate else "LIVE")

    # ------------------------------------------------------------------
    # Initialise subsystems
    # ------------------------------------------------------------------
    camera_kwargs = {}
    if camera_index is not None:
        camera_kwargs["camera_index"] = camera_index
    camera        = Camera(
        FRAME_WIDTH,
        FRAME_HEIGHT,
        FRAME_RATE,
        simulate=simulate,
        flip=camera_flip,
        **camera_kwargs,
    )

    # In simulation mode, skip detector import (MediaPipe not needed)
    detector = None
    if not simulate:
        from src.detector import FaceDetector
        from src.features import FeatureExtractor
        detector  = FaceDetector(
            refine_landmarks=DETECTOR_REFINE_LANDMARKS,
            min_detection_confidence=DETECTOR_MIN_DETECTION_CONF,
            min_tracking_confidence=DETECTOR_MIN_TRACKING_CONF,
        )
        extractor = FeatureExtractor(PROCESS_WIDTH, PROCESS_HEIGHT)
    else:
        extractor = None
        log.info("Simulation mode: MediaPipe detector bypassed.")

    scorer        = FatigueScorer()
    state_machine = AlertStateMachine()
    alert         = AlertController()
    dashboard     = Dashboard(DISPLAY_WIDTH, DISPLAY_HEIGHT, fullscreen)
    event_logger  = EventLogger()

    health_checks = run_startup_checks(camera, detector, dashboard, alert, event_logger, simulate)
    for check in health_checks:
        log.info("Self-test: %s=%s (%s)", check.name, "OK" if check.ok else "WARN", check.detail)
    dashboard.render_startup(health_checks)

    if args.guided_calibration:
        _run_guided_calibration(camera, detector, extractor, dashboard, simulate)
        log.info("Shutting down...")
        alert.stop()
        camera.release()
        dashboard.quit()
        event_logger.close()
        if detector:
            detector.close()
        log.info("DrowSAFE stopped cleanly.")
        return

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------
    running = True

    def shutdown(sig, frame):
        nonlocal running
        log.info("Shutdown signal received.")
        running = False

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    log.info("Entering main loop. Press Ctrl-C or ESC to stop.")

    fps_timer   = time.perf_counter()
    fps_count   = 0
    fps_display = 0.0
    start_time  = time.perf_counter()

    while running:
        loop_start = time.perf_counter()
        t = loop_start - start_time

        # 1. Grab frame
        frame = camera.read()
        if frame is None:
            score       = scorer.update(None)
            alert_level = state_machine.update(score)
            alert.update(alert_level)
            reason = scorer.details.primary_reason
            event_logger.log(alert_level, score, None, reason)
            dashboard.render(
                frame       = None,
                score       = score,
                alert_level = alert_level,
                features    = None,
                fps         = fps_display if SHOW_FPS else None,
                simulated   = simulate,
                alert_reason= reason if alert_level > 0 else None,
            )
            time.sleep(1.0 / max(FRAME_RATE, 1))
            if not dashboard.is_running():
                running = False
            continue

        # 2. Extract features
        if simulate:
            feat           = _simulate_features(t)
            annotated_frame = frame
        else:
            detection_frame, process_w, process_h = _processing_frame(frame)
            if extractor.frame_w != process_w or extractor.frame_h != process_h:
                extractor.update_frame_size(process_w, process_h)

            landmarks, annotated_frame = detector.process(
                detection_frame,
                draw=SHOW_LANDMARKS,
                annotation_frame=frame if SHOW_LANDMARKS else None,
            )
            feat = extractor.extract(landmarks) if landmarks else None

        # 3. Score + state machine
        score       = scorer.update(feat)
        alert_level = state_machine.update(score)

        # 4. Physical alert
        alert.update(alert_level)

        # 5. Log
        reason = scorer.details.primary_reason
        event_logger.log(alert_level, score, feat, reason)

        # 6. FPS counter
        fps_count += 1
        elapsed = time.perf_counter() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count   = 0
            fps_timer   = time.perf_counter()

        # 7. Render
        display_frame = annotated_frame if (SHOW_LANDMARKS and not simulate) else frame
        dashboard.render(
            frame       = display_frame,
            score       = score,
            alert_level = alert_level,
            features    = feat,
            fps         = fps_display if SHOW_FPS else None,
            simulated   = simulate,
            alert_reason= reason if alert_level > 0 else None,
        )

        if not dashboard.is_running():
            running = False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    log.info("Shutting down...")
    alert.stop()
    camera.release()
    dashboard.quit()
    event_logger.close()
    if detector:
        detector.close()
    log.info("DrowSAFE stopped cleanly.")


if __name__ == "__main__":
    main()
