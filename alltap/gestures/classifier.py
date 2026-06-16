"""Gesture classifier: turn a stream of hand frames into discrete gestures.

This is the pure-logic heart of alltap. Each frame, :meth:`GestureClassifier.update`
takes the detected hands plus the frame timestamp and returns one
:class:`GestureEvent`. It does no I/O — no camera, no model, no OS — so it is
fully testable with synthetic landmark sequences.

Flow per frame:
  1. Pick the active hand: the first hand inside the activation zone (a tight
     "is the hand close to the screen" gate). No hand in zone -> ``NONE``.
  2. Two-finger pose -> scroll path (suppresses tap/swipe).
  3. Otherwise -> the tap/swipe pointer state machine.
  4. Nothing fired -> ``HOVER`` with the fingertip position (drives the cursor).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from alltap.gestures.events import GestureEvent, GestureType
from alltap.gestures.geometry import distance
from alltap.gestures.pointer import PointerStateMachine
from alltap.gestures.scroll import ScrollDetector
from alltap.types import INDEX_TIP, MIDDLE_TIP, WRIST, Hand
from alltap.utils.config import Config, get_config

logger = logging.getLogger(__name__)


class GestureClassifier:
    """Stateful classifier producing one :class:`GestureEvent` per frame."""

    def __init__(self, config: Optional[Config] = None) -> None:
        cfg = config or get_config()
        self._zone_threshold = float(cfg.get("gestures.activation_zone_threshold"))
        self._pointer = PointerStateMachine(cfg)
        self._scroll = ScrollDetector(cfg)
        self._in_zone = False

    def update(self, hands: List[Hand], timestamp: float) -> GestureEvent:
        hand = self._active_hand(hands)

        if hand is None:
            if self._in_zone:
                logger.debug("Hand left activation zone")
                self._in_zone = False
            self._pointer.reset()
            self._scroll.reset()
            return GestureEvent(type=GestureType.NONE)

        if not self._in_zone:
            logger.debug("Hand entered activation zone")
            self._in_zone = True

        # Scroll takes precedence over the pointer gestures.
        if ScrollDetector.is_scroll_pose(hand):
            self._pointer.reset()
            scrolled = self._scroll.update(hand, timestamp)
            if scrolled is not None:
                direction, ticks = scrolled
                return GestureEvent(type=direction, scroll_ticks=ticks)
            return GestureEvent(type=GestureType.HOVER, position=hand.landmark(INDEX_TIP))

        self._scroll.reset()
        fired = self._pointer.update(hand.landmark(INDEX_TIP), timestamp)
        if fired is not None:
            gesture, position = fired
            return GestureEvent(type=gesture, position=position)

        return GestureEvent(type=GestureType.HOVER, position=hand.landmark(INDEX_TIP))

    def _active_hand(self, hands: List[Hand]) -> Optional[Hand]:
        """First hand inside the activation zone, or None."""
        for hand in hands:
            if distance(hand.landmark(WRIST), hand.landmark(MIDDLE_TIP)) >= self._zone_threshold:
                return hand
        return None
