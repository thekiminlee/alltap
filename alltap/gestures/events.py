"""Gesture event types emitted by the classifier.

A :class:`GestureEvent` is what one frame of classification produces. It stays
in normalized camera coordinates — the classifier never touches the screen; the
integration layer (Section 8) maps ``position`` to a pixel via calibration.

There is no ``DOUBLE_TAP``: two quick ``TAP`` clicks are coalesced into a
double-click by the OS, so we don't track double-taps ourselves.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel

from alltap.types import Point


class GestureType(str, Enum):
    NONE = "none"
    HOVER = "hover"
    TAP = "tap"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"


class GestureEvent(BaseModel):
    """One frame's classification result.

    Attributes:
        type: Which gesture (or ``NONE``/``HOVER``) this frame produced.
        position: Normalized fingertip location for ``HOVER`` and ``TAP``.
        scroll_ticks: Number of scroll ticks for ``SCROLL_*`` (0 otherwise).
    """

    type: GestureType
    position: Optional[Point] = None
    scroll_ticks: int = 0
