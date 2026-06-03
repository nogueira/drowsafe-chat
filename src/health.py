"""Startup self-test helpers for DrowSAFE."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HealthCheck:
    name: str
    ok: bool
    detail: str


def run_startup_checks(camera, detector, dashboard, alert, event_logger, simulate: bool):
    """Return concise startup health checks for display and logs."""
    checks = []

    camera_ok = camera is not None and getattr(camera, "_backend", None) is not None
    if camera_ok:
        detail = "Simulation camera active" if camera.is_simulated else "Camera backend active"
    else:
        detail = "Camera backend unavailable"
    checks.append(HealthCheck("Camera", camera_ok, detail))

    if simulate:
        checks.append(HealthCheck("Face detector", True, "Bypassed in simulation mode"))
    else:
        detector_ok = detector is not None
        detail = "MediaPipe FaceMesh ready" if detector_ok else "MediaPipe FaceMesh unavailable"
        checks.append(HealthCheck("Face detector", detector_ok, detail))

    dashboard_ok = dashboard is not None and dashboard.is_running()
    checks.append(HealthCheck("Dashboard", dashboard_ok, "Display surface ready" if dashboard_ok else "Dashboard disabled"))

    buzzer_ok = alert is not None and alert.is_enabled
    checks.append(HealthCheck("Buzzer", buzzer_ok, "GPIO buzzer enabled" if buzzer_ok else "GPIO buzzer unavailable"))

    log_ok = event_logger is not None and event_logger.is_enabled
    checks.append(HealthCheck("Event log", log_ok, "CSV logging enabled" if log_ok else "CSV logging disabled"))

    return checks
