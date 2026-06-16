"""Gesture classifier: turn a stream of hand frames into discrete gestures.

This is the pure-logic heart of alltap. Each frame, :meth:`GestureClassifier.update`
takes the detected hands plus the frame timestamp and returns one
:class:`GestureEvent`. It does no I/O — no camera, no model, no OS — so it is
fully testable with synthetic landmark sequences.

Flow per frame:
  1. Pick the active hand: the first hand inside the activation zone (a tight,
     pose-invariant "is the hand close to the screen" gate using palm size).
     No hand in zone -> ``NONE``.
  2. Delegate to the pointer state machine (tap / scroll / swipe).
  3. Nothing fired -> ``HOVER`` with the fingertip position (drives the cursor).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from alltap.gestures.events import GestureEvent, GestureType
from alltap.gestures.geometry import distance
from alltap.gestures.pointer import PointerStateMachine
from alltap.types import INDEX_TIP, MIDDLE_MCP, WRIST, Hand
from alltap.utils.config import Config, get_config

logger = logging.getLogger(__name__)


class GestureClassifier:
    """Stateful classifier producing one :class:`GestureEvent` per frame."""

    def __init__(self, config: Optional[Config] = None) -> None:
        cfg = config or get_config()
        self._zone_threshold = float(cfg.get("gestures.activation_zone_threshold"))
        self._pointer = PointerStateMachine(cfg)
        self._in_zone = False

    def update(self, hands: List[Hand], timestamp: float) -> GestureEvent:
        hand = self._active_hand(hands)

        if hand is None:
            if self._in_zone:
                logger.debug("Hand left activation zone")
                self._in_zone = False
            self._pointer.reset()
            return GestureEvent(type=GestureType.NONE)

        if not self._in_zone:
            logger.debug("Hand entered activation zone")
            self._in_zone = True

        fired = self._pointer.update(hand, timestamp)
        if fired is not None:
            return GestureEvent(
                type=fired.type, position=fired.position, scroll_ticks=fired.scroll_ticks
            )

        return GestureEvent(type=GestureType.HOVER, position=hand.landmark(INDEX_TIP))

    def _active_hand(self, hands: List[Hand]) -> Optional[Hand]:
        """First hand inside the activation zone, or None.

        Closeness is gauged by palm size (wrist -> middle-finger MCP), which does
        not change when fingers fold — important now that single-finger gestures
        curl the middle finger.
        """
        for hand in hands:
            palm = distance(hand.landmark(WRIST), hand.landmark(MIDDLE_MCP))
            if palm >= self._zone_threshold:
                return hand
        return None
