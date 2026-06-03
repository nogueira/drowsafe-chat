# =============================================================================
# DrowSAFE Configuration
# =============================================================================
# All system-wide parameters live here.
# Tune thresholds after running calibration (see docs/CALIBRATION.md).
# =============================================================================

# --- Camera ---
CAMERA_INDEX         = 0       # libcamera device index (0 = first camera)
CAMERA_FLIP          = True   # Flip vertically — set True if the camera image is upside down
FRAME_WIDTH          = 1280    # Capture resolution width  (pixels)
FRAME_HEIGHT         = 720     # Capture resolution height (pixels)
FRAME_RATE           = 30      # Target capture FPS
CAMERA_BUFFER_COUNT  = 2       # Lower buffer count reduces camera latency on Pi 5

# --- Raspberry Pi 5 performance ---
# FaceMesh is the most expensive stage. Capture can stay at 720p for display,
# while landmarks are inferred on a smaller frame with the same aspect ratio.
PROCESS_WIDTH        = 640     # MediaPipe inference width  (pixels)
PROCESS_HEIGHT       = 360     # MediaPipe inference height (pixels)
NATIVE_THREAD_LIMIT  = 2       # Limit native BLAS/OpenCV thread contention
OPENCV_NUM_THREADS   = 1       # OpenCV mostly does resize/colour conversion here

# --- Display ---
DISPLAY_WIDTH        = 800     # RPi Touch Display v1.1 native width
DISPLAY_HEIGHT       = 480     # RPi Touch Display v1.1 native height
FULLSCREEN           = True    # Run dashboard fullscreen on DSI display

# --- Face Detector ---
# refine_landmarks adds iris refinement cost; DrowSAFE uses eyelid/mouth/pose
# landmarks, so the lighter model is preferable on Raspberry Pi 5.
DETECTOR_REFINE_LANDMARKS      = False
DETECTOR_MIN_DETECTION_CONF    = 0.5
DETECTOR_MIN_TRACKING_CONF     = 0.5

# --- EAR  (Eye Aspect Ratio) ---
# EAR = (vertical eye span) / (horizontal eye span)
# Drops sharply on blink; stays low on sustained closure.
EAR_THRESHOLD        = 0.13   # Below this = eye closed. Set well below open-eye EAR (~0.17)
EAR_CONSEC_FRAMES    = 20     # Frames of sustained closure to trigger warning (~670ms @ 30fps)
EAR_RECOVERY_FRAMES  = 3      # Frames of open eyes needed to reset the closure counter
                                  # Normal blink = 8–12 frames, drowsy closure = 15+ frames

# --- MAR  (Mouth Aspect Ratio) ---
# Same geometry as EAR but applied to mouth landmarks.
MAR_THRESHOLD        = 0.45   # Above this → yawn detected (new direct ratio: rest~0.05, yawn~0.6)
MAR_CONSEC_FRAMES    = 15     # Consecutive frames above threshold to confirm yawn

# --- Head Pose ---
# Estimated via PnP solve on 6 canonical facial landmarks.
HEAD_PITCH_THRESHOLD   = 20   # Forward nod angle in degrees (positive = downward)
HEAD_NOD_CONSEC_FRAMES = 20   # Frames of nodding to contribute to score

# --- PERCLOS ---
# Percentage of Eye Closure: fraction of frames where eyes were closed
# over a rolling time window. NHTSA standard: >15% = drowsy.
PERCLOS_WINDOW_SEC   = 60     # Rolling window duration (seconds)
PERCLOS_THRESHOLD    = 0.15   # Fraction of closed frames → drowsy signal

# --- Face / Camera Visibility ---
# If the face disappears or the camera stops producing frames, escalate instead
# of freezing the previous score indefinitely.
FACE_MISSING_WARNING_SEC   = 2.0   # Seconds without face/frame before warning
FACE_MISSING_CRITICAL_SEC  = 5.0   # Seconds without face/frame before critical
FACE_MISSING_WARNING_SCORE = 45    # Score forced after warning timeout
FACE_MISSING_CRITICAL_SCORE= 80    # Score forced after critical timeout

# --- Fatigue Score ---
# Composite 0–100 score combining EAR, MAR, head pose, PERCLOS.
# Weights control how much each signal contributes.
SCORE_WEIGHT_PERCLOS   = 0.40  # 40% weight on PERCLOS
SCORE_WEIGHT_EAR       = 0.25  # 25% weight on instantaneous EAR
SCORE_WEIGHT_MAR       = 0.15  # 15% weight on yawn frequency
SCORE_WEIGHT_HEAD_POSE = 0.20  # 20% weight on head nod

# --- Alert State Machine ---
# Level 0 = Alert  (green  — no action)
# Level 1 = Warning (amber  — soft intermittent beep)
# Level 2 = Critical (red   — sustained alarm)
WARNING_SCORE        = 40     # Score threshold to enter Level 1
CRITICAL_SCORE       = 70     # Score threshold to enter Level 2

# Hysteresis: score must fall below these before downgrading alert level.
# Prevents flickering between states on noisy frames.
WARNING_HYSTERESIS   = 30     # Must drop below this to return to Level 0
CRITICAL_HYSTERESIS  = 55     # Must drop below this to return to Level 1

# --- GPIO ---
BUZZER_PIN           = 18     # BCM pin number for active piezo buzzer
BUZZER_WARNING_HZ    = 1      # Beep frequency for Level 1 (Hz)
BUZZER_CRITICAL_HZ   = 4      # Beep frequency for Level 2 (Hz)

# --- Logging ---
LOG_DIR              = "logs"
LOG_EVENTS           = True   # Write drowsiness events to timestamped CSV

# --- Debug ---
SHOW_LANDMARKS       = True   # Overlay MediaPipe mesh on camera feed (dev mode)
SHOW_FPS             = True   # Display FPS counter on dashboard
