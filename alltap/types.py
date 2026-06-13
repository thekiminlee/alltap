"""Shared data primitives used across every alltap module.

Keeping these in one place keeps module boundaries clean: the camera produces
frames, the tracker turns frames into :class:`Hand` objects, the classifier
consumes :class:`Hand` streams, and calibration maps a normalized :class:`Point`
to a pixel :class:`ScreenPoint`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# MediaPipe hand-landmark indices.
#
# MediaPipe Hands returns 21 landmarks per hand in a fixed order. We only name
# the few we actually use; the full map is documented at
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

#: Total number of landmarks MediaPipe emits per detected hand.
NUM_LANDMARKS = 21


@dataclass(frozen=True)
class Point:
    """A normalized landmark coordinate.

    ``x`` and ``y`` are in the range ``[0, 1]`` relative to the frame width and
    height (MediaPipe's native output space). ``z`` is the relative depth
    MediaPipe reports, with smaller values closer to the camera; it is optional
    because not every consumer needs it.
    """

    x: float
    y: float
    z: Optional[float] = None


@dataclass(frozen=True)
class ScreenPoint:
    """An absolute screen coordinate in integer pixels."""

    x: int
    y: int


@dataclass
class Hand:
    """A single detected hand.

    Attributes:
        landmarks: 21 :class:`Point` landmarks in MediaPipe order. Index using
            the module-level constants (e.g. ``hand.landmarks[INDEX_TIP]``).
        handedness: ``"Left"`` or ``"Right"`` as reported by MediaPipe.
        confidence: Detection/handedness confidence in ``[0, 1]``.
    """

    landmarks: List[Point] = field(default_factory=list)
    handedness: str = ""
    confidence: float = 0.0

    def landmark(self, index: int) -> Point:
        """Return the landmark at ``index`` (use the named constants)."""
        return self.landmarks[index]
