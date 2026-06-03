"""
DrowSAFE — Dashboard UI.

Pygame-based fullscreen dashboard for the Raspberry Pi Touch Display v1.1.
Shows live camera feed, fatigue score, alert level, and key metrics.

Layout (800×480)
----------------
  ┌──────────────────────────────────────────┐
  │  CAMERA FEED (left 60%)   │  METRICS     │
  │                           │  (right 40%) │
  │                           │  Score gauge │
  │                           │  EAR / MAR   │
  │                           │  Head pose   │
  │                           │  PERCLOS     │
  │  ALERT BANNER (bottom)                   │
  └──────────────────────────────────────────┘
"""

import sys
import logging
import numpy as np
import time
import math
from collections import deque

log = logging.getLogger("drowsafe.dashboard")

try:
    from config.config import EAR_THRESHOLD, EAR_CONSEC_FRAMES, EAR_RECOVERY_FRAMES, MAR_THRESHOLD, HEAD_PITCH_THRESHOLD, PERCLOS_THRESHOLD
except ImportError:
    EAR_THRESHOLD        = 0.22
    EAR_CONSEC_FRAMES    = 20
    EAR_RECOVERY_FRAMES  = 3
    MAR_THRESHOLD        = 0.45
    HEAD_PITCH_THRESHOLD = 20
    PERCLOS_THRESHOLD    = 0.15

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False
    log.warning("Pygame not available — dashboard disabled.")

# Alert level colours (RGB)
COLOURS = {
    0: (46,  204, 113),   # Green
    1: (245, 176,  65),   # Amber
    2: (231,  76,  60),   # Red
}

LABEL_COLOURS = {
    0: "ALERT",
    1: "WARNING",
    2: "CRITICAL",
}

BG_COLOUR      = (16,  20,  24)
PANEL_COLOUR   = (23,  29,  34)
PANEL_BORDER   = (45,  55,  63)
TEXT_PRIMARY   = (242, 245, 247)
TEXT_SECONDARY = (154, 167, 178)
TEXT_MUTED     = (105, 118, 128)
ACCENT_BLUE    = (52,  152, 219)

CAMERA_UNAVAILABLE_TITLE = "Camera frame unavailable"
CAMERA_UNAVAILABLE_DETAIL = "Check the camera connection"
FACE_NOT_DETECTED_TITLE = "Face not detected"
FACE_NOT_DETECTED_DETAIL = "Look toward the camera and improve lighting"


class Dashboard:
    """Pygame fullscreen dashboard."""

    TOP_BAR_H = 34
    BANNER_H = 58
    SIDE_PANEL_W = 250
    GAP = 14

    def __init__(self, width: int = 800, height: int = 480, fullscreen: bool = True):
        self._width      = width
        self._height     = height
        self._running    = _PYGAME_AVAILABLE
        self._screen     = None
        self._clock      = None
        self._font_large = None
        self._font_score = None
        self._font_med   = None
        self._font_small = None
        self._font_tiny  = None

        if not _PYGAME_AVAILABLE:
            return

        pygame.init()
        flags = pygame.FULLSCREEN | pygame.NOFRAME if fullscreen else 0
        self._screen = pygame.display.set_mode((width, height), flags)
        pygame.display.set_caption("DrowSAFE")
        self._clock = pygame.time.Clock()

        # Fonts — uses system DejaVu Sans (installed via apt)
        self._font_large = pygame.font.SysFont("dejavusans", 52, bold=True)
        self._font_score = pygame.font.SysFont("dejavusans", 46, bold=True)
        self._font_med   = pygame.font.SysFont("dejavusans", 26, bold=True)
        self._font_small = pygame.font.SysFont("dejavusans", 18)
        self._font_tiny  = pygame.font.SysFont("dejavusans", 14)

        self._ear_low_frames  = 0   # consecutive frames EAR below threshold
        self._ear_high_frames = 0   # consecutive frames EAR above threshold (grace period)
        self._score_history = deque(maxlen=120)
        self._last_history_sample = 0.0

        log.info("Dashboard initialised (%dx%d, fullscreen=%s)", width, height, fullscreen)

    def render(
        self,
        frame,
        score: float,
        alert_level: int,
        features,
        fps: float = None,
        simulated: bool = False,
        alert_reason: str = None,
        perclos: float = None,
    ):
        """
        Render one dashboard frame.

        Parameters
        ----------
        frame       : numpy.ndarray (RGB) | None
        score       : float  — fatigue score 0–100
        alert_level : int    — 0 / 1 / 2
        features    : Features | None
        fps         : float | None
        simulated   : bool   — show simulation badge if True
        """
        if not _PYGAME_AVAILABLE or self._screen is None:
            return

        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False
                return

        colour = COLOURS.get(alert_level, COLOURS[0])
        self._screen.fill(BG_COLOUR)

        content_top = self.TOP_BAR_H + self.GAP
        content_h = self._height - self.TOP_BAR_H - self.BANNER_H - (self.GAP * 2)
        camera_rect = pygame.Rect(
            self.GAP,
            content_top,
            self._width - self.SIDE_PANEL_W - (self.GAP * 3),
            content_h,
        )
        panel_x = camera_rect.right + self.GAP

        # --- Camera feed ---
        if frame is not None:
            self._draw_camera(frame, camera_rect)
        else:
            self._draw_camera_message(
                camera_rect,
                CAMERA_UNAVAILABLE_TITLE,
                CAMERA_UNAVAILABLE_DETAIL,
            )

        if frame is not None and features is None:
            self._draw_camera_message(
                camera_rect,
                FACE_NOT_DETECTED_TITLE,
                FACE_NOT_DETECTED_DETAIL,
            )

        self._sample_score_history(score)
        self._draw_video_scrim(camera_rect)
        self._draw_status_bar(frame, features, fps, simulated, alert_level)
        self._draw_metric_chips(panel_x, content_top, features, fps, perclos)
        self._draw_score_gauge(panel_x, content_top + 152, score, alert_level, alert_reason)
        self._draw_score_history(panel_x, self._height - self.BANNER_H - self.GAP - 72, colour)

        # --- Alert banner (bottom) ---
        self._draw_alert_banner(alert_level, score, colour, alert_reason)

        pygame.display.flip()
        self._clock.tick(60)

    def render_calibration(
        self,
        frame,
        features,
        step_title: str,
        instruction: str,
        progress: float,
        recommendation_lines=None,
        fps: float = None,
    ):
        """Render the guided calibration screen."""
        if not _PYGAME_AVAILABLE or self._screen is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False
                return

        self._screen.fill(BG_COLOUR)
        cam_w = int(self._width * 0.60)
        cam_h = self._height - 60
        camera_rect = pygame.Rect(0, 0, cam_w, cam_h)

        if frame is not None:
            self._draw_camera(frame, camera_rect)
        else:
            self._draw_camera_message(
                camera_rect,
                CAMERA_UNAVAILABLE_TITLE,
                CAMERA_UNAVAILABLE_DETAIL,
            )

        if frame is not None and features is None and not recommendation_lines:
            self._draw_camera_message(
                camera_rect,
                FACE_NOT_DETECTED_TITLE,
                FACE_NOT_DETECTED_DETAIL,
            )

        metrics_x = cam_w + 10
        self._draw_calibration_metrics(metrics_x, features, recommendation_lines)
        self._draw_calibration_overlay(cam_w, cam_h, step_title, instruction, progress, recommendation_lines)

        if fps is not None:
            fps_surf = self._font_small.render(f"{fps:.1f} fps", True, TEXT_SECONDARY)
            self._screen.blit(fps_surf, (8, 8))

        pygame.draw.rect(self._screen, (52, 152, 219), pygame.Rect(0, self._height - 54, self._width, 54))
        text = "Guided calibration - press ESC to exit"
        s = self._fit_text(text, self._font_med, self._width - 24, (255, 255, 255))
        self._screen.blit(s, (self._width // 2 - s.get_width() // 2, self._height - 37))

        pygame.display.flip()
        self._clock.tick(60)

    def render_startup(self, checks, duration_sec: float = 2.5):
        """Show startup self-test results briefly."""
        if not _PYGAME_AVAILABLE or self._screen is None:
            return

        started = time.monotonic()
        while self._running and time.monotonic() - started < duration_sec:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._running = False
                    return

            self._screen.fill(BG_COLOUR)
            title = self._font_med.render("Startup self-test", True, TEXT_PRIMARY)
            self._screen.blit(title, (28, 28))

            y = 82
            for check in checks:
                colour = (39, 174, 96) if check.ok else (243, 156, 18)
                status = "OK" if check.ok else "WARN"
                line = f"{status}  {check.name}: {check.detail}"
                s = self._fit_text(line, self._font_small, self._width - 56, colour)
                self._screen.blit(s, (28, y))
                y += 30

            pygame.display.flip()
            self._clock.tick(30)

    def _draw_camera(self, frame, rect):
        """Scale and blit the camera frame into a target rectangle."""
        import cv2

        # Normalise to 3-channel
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # Scale to fit panel
        h, w  = frame.shape[:2]
        scale = min(rect.width / w, rect.height / h)
        nw, nh = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (nw, nh))

        # Convert to pygame Surface using numpy transpose (proven reliable method)
        # frame is RGB, transpose to (width, height, 3) for surfarray
        surface = pygame.surfarray.make_surface(
            np.ascontiguousarray(np.transpose(frame, (1, 0, 2)))
        )

        # Centre in panel
        ox = rect.x + (rect.width - nw) // 2
        oy = rect.y + (rect.height - nh) // 2
        self._screen.blit(surface, (ox, oy))

    def _draw_calibration_metrics(self, x: int, features, recommendation_lines):
        panel_w = self._width - x - 10
        y = 24

        title = self._font_small.render("CALIBRATION", True, TEXT_SECONDARY)
        self._screen.blit(title, (x, y))
        y += 34

        def row(label, value):
            nonlocal y
            self._screen.blit(self._font_small.render(label, True, TEXT_SECONDARY), (x, y))
            self._screen.blit(self._font_small.render(value, True, TEXT_PRIMARY), (x + 126, y))
            y += 28

        if features:
            row("EAR", f"{features.ear:.3f}")
            row("MAR", f"{features.mar:.3f}")
            row("Head pitch", f"{features.head_pitch:+.1f} deg")
        else:
            row("EAR", "--")
            row("MAR", "--")
            row("Head pitch", "--")

        if recommendation_lines:
            y += 18
            for line in recommendation_lines[1:]:
                s = self._fit_text(line, self._font_small, panel_w, TEXT_PRIMARY)
                self._screen.blit(s, (x, y))
                y += 28

    def _draw_calibration_overlay(
        self,
        cam_w: int,
        cam_h: int,
        step_title: str,
        instruction: str,
        progress: float,
        recommendation_lines,
    ):
        overlay_h = 116 if not recommendation_lines else 150
        overlay = pygame.Surface((cam_w, overlay_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 175))
        oy = cam_h - overlay_h - 12
        self._screen.blit(overlay, (0, oy))

        title = recommendation_lines[0] if recommendation_lines else step_title
        title_s = self._fit_text(title, self._font_med, cam_w - 32, (255, 255, 255))
        self._screen.blit(title_s, (16, oy + 14))

        body = "Review the recommended thresholds in the side panel." if recommendation_lines else instruction
        body_s = self._fit_text(body, self._font_small, cam_w - 32, TEXT_PRIMARY)
        self._screen.blit(body_s, (16, oy + 54))

        bar_w = cam_w - 32
        bar_y = oy + overlay_h - 30
        pygame.draw.rect(self._screen, (70, 70, 70), (16, bar_y, bar_w, 12), border_radius=6)
        fill_w = int(bar_w * max(0.0, min(1.0, progress)))
        if fill_w > 0:
            pygame.draw.rect(self._screen, (52, 152, 219), (16, bar_y, fill_w, 12), border_radius=6)

    def _draw_camera_message(self, rect, text: str, detail_text: str):
        """Draw a centered status message over the camera area."""
        overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 145))
        self._screen.blit(overlay, rect.topleft)

        message = self._font_med.render(text, True, (255, 255, 255))
        detail = self._font_small.render(detail_text, True, TEXT_SECONDARY)

        mx = rect.x + rect.width // 2 - message.get_width() // 2
        my = rect.y + rect.height // 2 - message.get_height()
        dx = rect.x + rect.width // 2 - detail.get_width() // 2
        dy = my + message.get_height() + 8

        self._screen.blit(message, (mx, my))
        self._screen.blit(detail, (dx, dy))

    def _sample_score_history(self, score: float):
        now = time.monotonic()
        if not self._score_history or now - self._last_history_sample >= 0.5:
            self._score_history.append(max(0.0, min(100.0, score)))
            self._last_history_sample = now

    def _draw_video_scrim(self, rect):
        top = pygame.Surface((rect.width, 96), pygame.SRCALPHA)
        top.fill((0, 0, 0, 80))
        self._screen.blit(top, rect.topleft)
        bottom = pygame.Surface((rect.width, 122), pygame.SRCALPHA)
        bottom.fill((0, 0, 0, 105))
        self._screen.blit(bottom, (rect.x, rect.bottom - bottom.get_height()))

    def _draw_status_bar(self, frame, features, fps, simulated: bool, alert_level: int):
        pygame.draw.rect(self._screen, BG_COLOUR, (0, 0, self._width, self.TOP_BAR_H))
        pygame.draw.line(self._screen, PANEL_BORDER, (0, self.TOP_BAR_H - 1), (self._width, self.TOP_BAR_H - 1))

        camera_text = "CAMERA OK" if frame is not None else "CAMERA LOST"
        face_text = "FACE TRACKED" if features is not None else "FACE LOST"
        fps_text = f"{fps:.1f} FPS" if fps is not None else "FPS --"
        mode_text = "SIMULATION" if simulated else "DRIVER MODE"
        status = f"{mode_text} | {camera_text} | {face_text} | {fps_text}"

        status_colour = COLOURS[alert_level] if alert_level else TEXT_SECONDARY
        dot_x = 14
        pygame.draw.circle(self._screen, status_colour, (dot_x, self.TOP_BAR_H // 2), 5)
        s = self._fit_text(status, self._font_tiny, self._width - 42, TEXT_PRIMARY)
        self._screen.blit(s, (28, self.TOP_BAR_H // 2 - s.get_height() // 2))

    def _draw_score_gauge(self, x: int, y: int, score: float, alert_level: int, alert_reason: str = None):
        colour = COLOURS.get(alert_level, COLOURS[0])
        rect = pygame.Rect(x, y, self.SIDE_PANEL_W, 118)
        self._draw_panel(rect)

        center = (rect.x + 70, rect.y + 58)
        radius = 42
        track = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
        pygame.draw.arc(self._screen, (56, 64, 72), track, math.radians(145), math.radians(395), 8)

        sweep = 250.0 * max(0.0, min(100.0, score)) / 100.0
        if sweep > 0:
            pygame.draw.arc(
                self._screen,
                colour,
                track,
                math.radians(145),
                math.radians(145 + sweep),
                8,
            )

        score_s = self._font_score.render(f"{int(score)}", True, TEXT_PRIMARY)
        self._screen.blit(score_s, (center[0] - score_s.get_width() // 2, center[1] - 34))

        label_s = self._font_tiny.render(LABEL_COLOURS[alert_level], True, colour)
        self._screen.blit(label_s, (center[0] - label_s.get_width() // 2, center[1] + 18))

        reason = alert_reason or "Monitoring active"
        heading = self._font_tiny.render("FATIGUE SCORE", True, TEXT_SECONDARY)
        reason_s = self._fit_text(reason, self._font_small, rect.width - 150, TEXT_PRIMARY)
        self._screen.blit(heading, (rect.x + 140, rect.y + 28))
        self._screen.blit(reason_s, (rect.x + 140, rect.y + 52))

    def _draw_metric_chips(self, x: int, y: int, features, fps, perclos):
        gap = 8
        w = (self.SIDE_PANEL_W - gap) // 2
        h = 42

        if features:
            ear_warn = self._update_eye_warning(features.ear)
            chips = [
                ("EAR", f"{features.ear:.3f}", ear_warn),
                ("MAR", f"{features.mar:.3f}", features.mar > MAR_THRESHOLD),
                ("HEAD", f"{features.head_pitch:+.1f} deg", abs(features.head_pitch) > HEAD_PITCH_THRESHOLD),
                ("PERCLOS", f"{(perclos or 0.0) * 100:.0f}%", (perclos or 0.0) >= PERCLOS_THRESHOLD),
            ]
        else:
            chips = [
                ("EAR", "--", True),
                ("MAR", "--", True),
                ("HEAD", "--", True),
                ("PERCLOS", f"{(perclos or 0.0) * 100:.0f}%", False),
            ]

        chips.append(("FACE", "TRACKED" if features else "LOST", features is None))
        if fps is not None:
            chips.append(("FPS", f"{fps:.1f}", fps < 18.0))

        for i, (label, value, warn) in enumerate(chips):
            col = i % 2
            row = i // 2
            rect = pygame.Rect(x + col * (w + gap), y + row * (h + gap), w, h)
            self._draw_chip(rect, label, value, warn)

    def _update_eye_warning(self, ear: float) -> bool:
        if ear < EAR_THRESHOLD:
            self._ear_low_frames += 1
            self._ear_high_frames = 0
        else:
            self._ear_high_frames += 1
            if self._ear_high_frames >= EAR_RECOVERY_FRAMES:
                self._ear_low_frames = 0
                self._ear_high_frames = 0
        return self._ear_low_frames >= EAR_CONSEC_FRAMES

    def _draw_chip(self, rect, label: str, value: str, warn: bool = False):
        colour = COLOURS[1] if warn else ACCENT_BLUE
        if warn and value in ("LOST", "--"):
            colour = COLOURS[2]
        self._draw_panel(rect, alpha=205)
        pygame.draw.rect(self._screen, colour, (rect.x, rect.y, 4, rect.height), border_radius=2)

        label_s = self._font_tiny.render(label, True, TEXT_SECONDARY)
        value_s = self._fit_text(value, self._font_small, rect.width - 22, TEXT_PRIMARY)
        self._screen.blit(label_s, (rect.x + 12, rect.y + 5))
        self._screen.blit(value_s, (rect.right - value_s.get_width() - 10, rect.y + 19))

    def _draw_score_history(self, x: int, y: int, colour):
        rect = pygame.Rect(x, y, self.SIDE_PANEL_W, 72)
        self._draw_panel(rect, alpha=190)
        title = self._font_tiny.render("60 SEC RISK TREND", True, TEXT_SECONDARY)
        self._screen.blit(title, (rect.x + 12, rect.y + 8))

        plot = pygame.Rect(rect.x + 12, rect.y + 30, rect.width - 24, 30)
        pygame.draw.line(self._screen, (58, 66, 74), (plot.x, plot.centery), (plot.right, plot.centery), 1)
        values = list(self._score_history)
        if len(values) < 2:
            return

        step = plot.width / max(1, len(values) - 1)
        points = []
        for i, value in enumerate(values):
            px = plot.x + int(i * step)
            py = plot.bottom - int((value / 100.0) * plot.height)
            points.append((px, py))
        if len(points) >= 2:
            pygame.draw.lines(self._screen, colour, False, points, 2)

    def _draw_panel(self, rect, alpha: int = 218):
        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        panel.fill((*PANEL_COLOUR, alpha))
        self._screen.blit(panel, rect.topleft)
        pygame.draw.rect(self._screen, PANEL_BORDER, rect, width=1, border_radius=8)

    def _fit_text(self, text: str, font, max_width: int, colour):
        """Render text, shortening the middle if it would overflow."""
        if font.size(text)[0] <= max_width:
            return font.render(text, True, colour)

        ellipsis = "..."
        while text and font.size(text + ellipsis)[0] > max_width:
            text = text[:-1]
        return font.render((text + ellipsis) if text else ellipsis, True, colour)

    def _draw_alert_banner(self, alert_level: int, score: float, colour, alert_reason: str = None):
        """Draw the bottom alert banner."""
        bh      = 54
        by      = self._height - bh
        banner  = pygame.Rect(0, by, self._width, bh)
        pygame.draw.rect(self._screen, colour, banner)

        if alert_level == 0:
            text = "Driver monitoring active"
            subtext = "Stay alert"
        elif alert_level == 1:
            text = "Drowsiness detected"
            subtext = alert_reason or "Please take a break"
        else:
            text = "WAKE UP"
            subtext = "Pull over immediately" if not alert_reason else f"Reason: {alert_reason}"

        title_font = self._font_med if alert_level < 2 else self._font_large
        s = self._fit_text(text, title_font, self._width - 36, (255, 255, 255))
        detail = self._fit_text(subtext, self._font_tiny, self._width - 36, (255, 255, 255))
        total_h = s.get_height() + detail.get_height() + 2
        self._screen.blit(s, (self._width // 2 - s.get_width() // 2, by + bh // 2 - total_h // 2))
        self._screen.blit(detail, (self._width // 2 - detail.get_width() // 2, by + bh // 2 - total_h // 2 + s.get_height() + 2))

    def is_running(self) -> bool:
        return self._running

    def quit(self):
        if _PYGAME_AVAILABLE:
            pygame.quit()
        log.info("Dashboard closed.")
