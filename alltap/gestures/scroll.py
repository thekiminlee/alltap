"""Two-finger scroll detection.

When the index and middle fingers are extended (and ring + pinky are folded),
the average vertical motion of the two fingertips drives continuous scrolling,
with the tick rate proportional to speed.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from alltap.gestures.events import GestureType
from alltap.gestures.geometry import distance
from alltap.types import (
    INDEX_TIP,
    MIDDLE_TIP,
    PINKY_TIP,
    RING_TIP,
    WRIST,
    Hand,
)
from alltap.utils.config import Config, get_config

logger = logging.getLogger(__name__)

# Each fingertip's matching PIP joint (the second knuckle), used to tell whether
# the finger is extended: an extended finger's tip is farther from the wrist
# than its PIP; a folded finger's tip curls back closer than the PIP.
_PIP = {INDEX_TIP: 6, MIDDLE_TIP: 10, RING_TIP: 14, PINKY_TIP: 18}

_MAX_TICKS = 5


def _is_extended(hand: Hand, tip: int) -> bool:
    wrist = hand.landmark(WRIST)
    return distance(hand.landmark(tip), wrist) > distance(hand.landmark(_PIP[tip]), wrist)


class ScrollDetector:
    """Detects the two-finger scroll pose and emits scroll ticks."""

    def __init__(self, config: Optional[Config] = None) -> None:
        cfg = config or get_config()
        self._speed_threshold = float(cfg.get("gestures.scroll_speed_threshold"))
        self._multiplier = float(cfg.get("gestures.scroll_multiplier"))
        self.reset()

    def reset(self) -> None:
        self._prev_y: Optional[float] = None
        self._prev_time: Optional[float] = None

    @staticmethod
    def is_scroll_pose(hand: Hand) -> bool:
        """True when index + middle are extended and ring + pinky are folded."""
        return (
            _is_extended(hand, INDEX_TIP)
            and _is_extended(hand, MIDDLE_TIP)
            and not _is_extended(hand, RING_TIP)
            and not _is_extended(hand, PINKY_TIP)
        )

    def update(
        self, hand: Hand, timestamp: float
    ) -> Optional[Tuple[GestureType, int]]:
        """Feed one frame in the scroll pose; return (direction, ticks) or None."""
        avg_y = (hand.landmark(INDEX_TIP).y + hand.landmark(MIDDLE_TIP).y) / 2.0

        if self._prev_y is None or timestamp <= self._prev_time:
            self._prev_y, self._prev_time = avg_y, timestamp
            return None

        speed = (avg_y - self._prev_y) / (timestamp - self._prev_time)
        self._prev_y, self._prev_time = avg_y, timestamp

        if abs(speed) < self._speed_threshold:
            return None

        ticks = min(int(abs(speed) / self._speed_threshold * self._multiplier), _MAX_TICKS)
        ticks = max(ticks, 1)
        # Image y grows downward, so moving up (y decreasing) scrolls up.
        direction = GestureType.SCROLL_UP if speed < 0 else GestureType.SCROLL_DOWN
        logger.debug("Scroll %s x%d (speed %.3f)", direction.value, ticks, speed)
        return direction, ticks
