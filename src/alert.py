"""
DrowSAFE — Alert controller.

Drives the GPIO buzzer based on the current alert level.
Uses lgpio (the correct GPIO library for Raspberry Pi 5 on Bookworm).
"""

import threading
import logging

log = logging.getLogger("drowsafe.alert")

lgpio = None

from config.config import BUZZER_PIN, BUZZER_WARNING_HZ, BUZZER_CRITICAL_HZ
from src.state_machine import ALERT, WARNING, CRITICAL


def _open_gpio_chip():
    """Import lgpio and open the GPIO chip only when the controller starts."""
    global lgpio
    if lgpio is None:
        try:
            import lgpio as _lgpio
            lgpio = _lgpio
        except Exception as e:
            log.warning("lgpio not available (%s) — buzzer disabled.", e)
            return None

    try:
        chip = lgpio.gpiochip_open(0)
        log.info("lgpio initialised successfully.")
        return chip
    except Exception as e:
        log.warning("Could not open GPIO chip (%s) — buzzer disabled.", e)
        return None


def _claim_pin(chip, pin):
    """Claim GPIO output pin, freeing it first if already busy."""
    try:
        lgpio.gpio_claim_output(chip, pin, 0)
        return True
    except Exception:
        pass
    # Pin busy — try to free it and reclaim
    try:
        lgpio.gpio_free(chip, pin)
        lgpio.gpio_claim_output(chip, pin, 0)
        log.info("GPIO pin %d reclaimed after busy state.", pin)
        return True
    except Exception as e:
        log.warning("Could not claim GPIO pin %d: %s — buzzer disabled.", pin, e)
        return False


class AlertController:
    def __init__(self):
        self._level    = ALERT
        self._running  = True
        self._lock     = threading.Lock()
        self._wake     = threading.Event()
        self._pin_ok   = False
        self._chip     = None
        self._thread   = None

        self._chip = _open_gpio_chip()
        if self._chip is not None:
            self._pin_ok = _claim_pin(self._chip, BUZZER_PIN)

        if self._pin_ok:
            self._thread = threading.Thread(
                target=self._buzzer_loop, daemon=True, name="buzzer"
            )
            self._thread.start()
        log.info("AlertController started (buzzer=%s).", "enabled" if self._pin_ok else "disabled")

    def update(self, alert_level: int):
        with self._lock:
            self._level = alert_level
        self._wake.set()

    def _set_buzzer(self, state: bool):
        if self._pin_ok and self._chip is not None:
            try:
                lgpio.gpio_write(self._chip, BUZZER_PIN, 1 if state else 0)
            except Exception:
                pass

    def _buzzer_loop(self):
        while self._running:
            with self._lock:
                level = self._level

            if level == ALERT:
                self._set_buzzer(False)
                self._sleep_or_wake(0.1)
            elif level == WARNING:
                period = 1.0 / BUZZER_WARNING_HZ
                self._set_buzzer(True)
                if self._sleep_or_wake(period / 2):
                    continue
                self._set_buzzer(False)
                if self._sleep_or_wake(period / 2):
                    continue
            elif level == CRITICAL:
                period = 1.0 / BUZZER_CRITICAL_HZ
                self._set_buzzer(True)
                if self._sleep_or_wake(period / 2):
                    continue
                self._set_buzzer(False)
                if self._sleep_or_wake(period / 2):
                    continue

        self._set_buzzer(False)

    def _sleep_or_wake(self, timeout: float):
        woke = self._wake.wait(timeout)
        self._wake.clear()
        return woke

    def stop(self):
        self._running = False
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._chip is not None:
            try:
                if self._pin_ok:
                    lgpio.gpio_write(self._chip, BUZZER_PIN, 0)
                    lgpio.gpio_free(self._chip, BUZZER_PIN)
                lgpio.gpiochip_close(self._chip)
            except Exception:
                pass
            self._chip = None
        log.info("AlertController stopped.")

    @property
    def is_enabled(self) -> bool:
        return self._pin_ok and self._chip is not None
