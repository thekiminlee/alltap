"""Tap, scroll, and swipe detection from the index fingertip (touchscreen-style).

Everything starts with a **poke** — the fingertip accelerating then decelerating
sharply, our stand-in for "touch down." What happens in the short window after
that contact decides the gesture:

- no drag                      -> ``TAP`` (click)
- drag with **one** finger     -> ``SCROLL`` (continuous, by vertical motion)
- drag with **two** fingers    -> ``SWIPE`` (directional, one-shot)

The poke is what separates a deliberate scroll/swipe from plain hovering (which
is also single-finger motion). Finger count is just "is the middle finger
extended" — we ignore ring/pinky so a relaxed hand still works. Working in
normalized coordinates with real timestamps keeps this independent of frame rate
and calibration.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import NamedTuple, Optional

from alltap.gestures.events import GestureType
from alltap.gestures.geometry import distance
from alltap.types import INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, WRIST, Hand, Point
from alltap.utils.config import Config, get_config

logger = logging.getLogger(__name__)

_MAX_SCROLL_TICKS = 5


class Fired(NamedTuple):
    """A gesture produced this frame."""

    type: GestureType
    position: Optional[Point] = None
    scroll_ticks: int = 0


class _Mode(Enum):
    IDLE = auto()
    THRUSTING = auto()  # fingertip speeding up; watching for the decel (contact)
    DECIDING = auto()  # contact made; tap-vs-drag decision window open
    SCROLLING = auto()  # one-finger drag in progress -> continuous scroll
    COOLDOWN = auto()  # debounce after a tap/swipe


def _middle_extended(hand: Hand) -> bool:
    """True when the middle finger is extended (tip farther from wrist than PIP)."""
    wrist = hand.landmark(WRIST)
    middle_pip = hand.landmarks[10]
    return distance(hand.landmark(MIDDLE_TIP), wrist) > distance(middle_pip, wrist)


class PointerStateMachine:
    """Detects taps, scrolls, and swipes from a stream of hand frames."""

    def __init__(self, config: Optional[Config] = None) -> None:
        cfg = config or get_config()
        self._speed_threshold = float(cfg.get("gestures.tap_speed_threshold"))
        self._decel_ratio = float(cfg.get("gestures.tap_decel_ratio"))
        self._tap_time = float(cfg.get("gestures.tap_time_ms")) / 1000.0
        self._cooldown = float(cfg.get("gestures.tap_cooldown_ms")) / 1000.0
        self._decision_time = float(cfg.get("gestures.drag_decision_ms")) / 1000.0
        self._drag_distance = float(cfg.get("gestures.drag_distance_threshold"))
        self._scroll_speed = float(cfg.get("gestures.scroll_speed_threshold"))
        self._scroll_multiplier = float(cfg.get("gestures.scroll_multiplier"))
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
        self._scroll_prev_y = 0.0

    def update(self, hand: Hand, timestamp: float) -> Optional[Fired]:
        """Feed one frame; return a fired gesture or None (caller treats None as hover)."""
        tip = hand.landmark(INDEX_TIP)
        dt = (timestamp - self._prev_time) if self._prev_time is not None else 0.0
        speed = distance(tip, self._prev_pos) / dt if dt > 0 else 0.0
        self._prev_pos, self._prev_time = tip, timestamp

        if self._mode is _Mode.COOLDOWN:
            if timestamp >= self._cooldown_until:
                self._mode = _Mode.IDLE
            return None

        if self._mode is _Mode.SCROLLING:
            return self._scroll(tip, speed, dt)

        if self._mode is _Mode.IDLE:
            if speed >= self._speed_threshold:
                self._mode = _Mode.THRUSTING
                self._peak_speed = speed
                self._thrust_start = timestamp
            return None

        if self._mode is _Mode.THRUSTING:
            self._peak_speed = max(self._peak_speed, speed)
            if timestamp - self._thrust_start > self._tap_time:
                self._mode = _Mode.IDLE  # dragged on without a clean stop
            elif speed <= self._decel_ratio * self._peak_speed:
                self._contact_pos = tip  # decel point = contact = sampled location
                self._contact_time = timestamp
                self._mode = _Mode.DECIDING
            return None

        if self._mode is _Mode.DECIDING:
            return self._decide(hand, tip, timestamp)

        return None

    def _decide(self, hand: Hand, tip: Point, timestamp: float) -> Optional[Fired]:
        if distance(tip, self._contact_pos) >= self._drag_distance:
            if _middle_extended(hand):  # two fingers -> swipe
                gesture = self._swipe_direction(tip, self._contact_pos)
                logger.debug("Swipe %s", gesture.value)
                self._enter_cooldown(timestamp)
                return Fired(gesture)
            # one finger -> begin a continuous scroll
            self._mode = _Mode.SCROLLING
            self._scroll_prev_y = tip.y
            return None
        if timestamp - self._contact_time >= self._decision_time:
            logger.debug("Tap at (%.3f, %.3f)", self._contact_pos.x, self._contact_pos.y)
            self._enter_cooldown(timestamp)
            return Fired(GestureType.TAP, position=self._contact_pos)
        return None

    def _scroll(self, tip: Point, speed: float, dt: float) -> Optional[Fired]:
        if speed < self._scroll_speed:  # finger stopped -> end the scroll
            self._mode = _Mode.IDLE
            return None
        y_speed = (tip.y - self._scroll_prev_y) / dt if dt > 0 else 0.0
        self._scroll_prev_y = tip.y
        if abs(y_speed) < self._scroll_speed:
            return None
        ticks = min(int(abs(y_speed) / self._scroll_speed * self._scroll_multiplier), _MAX_SCROLL_TICKS)
        ticks = max(ticks, 1)
        # Image y grows downward, so moving up (y decreasing) scrolls up.
        direction = GestureType.SCROLL_UP if y_speed < 0 else GestureType.SCROLL_DOWN
        return Fired(direction, scroll_ticks=ticks)

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
