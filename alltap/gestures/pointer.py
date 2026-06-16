"""Tap and swipe detection from the index fingertip (touchscreen-style).

The single primitive is a **tap** — "finger down" — detected purely from the
fingertip's accelerate-then-decelerate motion, regardless of the wrist. What
happens in the short window after that contact decides the outcome:

- continued lateral travel past a threshold -> a ``SWIPE`` (poke-then-drag), or
- no significant travel within the decision window -> a ``TAP`` (click).

Working in normalized coordinates with real timestamps keeps this decoupled from
the camera frame rate and from calibration.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Optional, Tuple

from alltap.gestures.events import GestureType
from alltap.gestures.geometry import distance
from alltap.types import Point
from alltap.utils.config import Config, get_config

logger = logging.getLogger(__name__)


class _Mode(Enum):
    IDLE = auto()
    THRUSTING = auto()  # fingertip speed is above threshold, watching for decel
    DECIDING = auto()  # contact made; tap-vs-swipe decision window open
    COOLDOWN = auto()  # debounce after a fired gesture


class PointerStateMachine:
    """Detects taps and swipes from a stream of fingertip positions."""

    def __init__(self, config: Optional[Config] = None) -> None:
        cfg = config or get_config()
        self._speed_threshold = float(cfg.get("gestures.tap_speed_threshold"))
        self._decel_ratio = float(cfg.get("gestures.tap_decel_ratio"))
        self._tap_time = float(cfg.get("gestures.tap_time_ms")) / 1000.0
        self._cooldown = float(cfg.get("gestures.tap_cooldown_ms")) / 1000.0
        self._decision_time = float(cfg.get("gestures.swipe_decision_ms")) / 1000.0
        self._swipe_distance = float(cfg.get("gestures.swipe_distance_threshold"))
        self.reset()

    def reset(self) -> None:
        self._mode = _Mode.IDLE
        self._prev_pos: Optional[Point] = None
        self._prev_time: Optional[float] = None
        self._peak_speed = 0.0
        self._thrust_start = 0.0
        self._contact_pos: Optional[Point] = None
        self._contact_time = 0.0
        self._cooldown_until = 0.0

    def update(
        self, tip: Point, timestamp: float
    ) -> Optional[Tuple[GestureType, Optional[Point]]]:
        """Feed one fingertip sample; return a fired (gesture, position) or None."""
        speed = 0.0
        if self._prev_pos is not None and timestamp > self._prev_time:
            speed = distance(tip, self._prev_pos) / (timestamp - self._prev_time)
        self._prev_pos, self._prev_time = tip, timestamp

        if self._mode is _Mode.COOLDOWN:
            if timestamp >= self._cooldown_until:
                self._mode = _Mode.IDLE
            return None

        if self._mode is _Mode.IDLE:
            if speed >= self._speed_threshold:
                self._mode = _Mode.THRUSTING
                self._peak_speed = speed
                self._thrust_start = timestamp
            return None

        if self._mode is _Mode.THRUSTING:
            self._peak_speed = max(self._peak_speed, speed)
            if timestamp - self._thrust_start > self._tap_time:
                self._mode = _Mode.IDLE  # thrust dragged on without a clean stop
            elif speed <= self._decel_ratio * self._peak_speed:
                self._contact_pos = tip  # decel point = contact = sampled location
                self._contact_time = timestamp
                self._mode = _Mode.DECIDING
            return None

        if self._mode is _Mode.DECIDING:
            travel = distance(tip, self._contact_pos)
            if travel >= self._swipe_distance:
                gesture = self._swipe_direction(tip, self._contact_pos)
                logger.debug("Swipe %s (travel %.3f)", gesture.value, travel)
                self._enter_cooldown(timestamp)
                return gesture, None
            if timestamp - self._contact_time >= self._decision_time:
                logger.debug("Tap at (%.3f, %.3f)", self._contact_pos.x, self._contact_pos.y)
                self._enter_cooldown(timestamp)
                return GestureType.TAP, self._contact_pos
            return None

        return None

    def _enter_cooldown(self, timestamp: float) -> None:
        self._mode = _Mode.COOLDOWN
        self._cooldown_until = timestamp + self._cooldown

    @staticmethod
    def _swipe_direction(current: Point, contact: Point) -> GestureType:
        dx = current.x - contact.x
        dy = current.y - contact.y
        if abs(dx) >= abs(dy):
            return GestureType.SWIPE_RIGHT if dx > 0 else GestureType.SWIPE_LEFT
        return GestureType.SWIPE_DOWN if dy > 0 else GestureType.SWIPE_UP
