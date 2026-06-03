"""Unit tests for the GPIO alert controller fallback path."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import alert as alert_mod
from src.alert import AlertController
from src.state_machine import WARNING


def test_disabled_buzzer_does_not_spawn_thread(monkeypatch):
    monkeypatch.setattr(alert_mod, "_open_gpio_chip", lambda: None)

    controller = AlertController()
    try:
        controller.update(WARNING)
        assert not controller.is_enabled
        assert controller._thread is None
    finally:
        controller.stop()
