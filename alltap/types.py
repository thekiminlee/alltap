"""Shared data primitives used across every alltap module.

Keeping these in one place keeps module boundaries clean: the camera produces
frames, the tracker turns frames into :class:`Hand` objects, the classifier
consumes :class:`Hand` streams, and calibration maps a normalized :class:`Point`
to a pixel :class:`ScreenPoint`.

These are :mod:`pydantic` models so we get validation, value-based equality, and
JSON (de)serialization for free — handy for persisting calibration data and for
logging/debugging hand frames. For the hot capture path (21 landmarks per hand,
up to ~60fps), construct trusted MediaPipe output with
:meth:`pydantic.BaseModel.model_construct`, which skips validation.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

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


class Point(BaseModel):
    """A normalized landmark coordinate.

    ``x`` and ``y`` are in the range ``[0, 1]`` relative to the frame width and
    height (MediaPipe's native output space). ``z`` is the relative depth
    MediaPipe reports, with smaller values closer to the camera; it is optional
    because not every consumer needs it.
    """

    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    z: Optional[float] = None


class ScreenPoint(BaseModel):
    """An absolute screen coordinate in integer pixels."""

    model_config = ConfigDict(frozen=True)

    x: int
    y: int


class Hand(BaseModel):
    """A single detected hand.

    Attributes:
        landmarks: 21 :class:`Point` landmarks in MediaPipe order. Index using
            the module-level constants (e.g. ``hand.landmarks[INDEX_TIP]``).
        handedness: ``"Left"`` or ``"Right"`` as reported by MediaPipe.
        confidence: Detection/handedness confidence in ``[0, 1]``.
    """

    landmarks: List[Point] = Field(default_factory=list)
    handedness: str = ""
    confidence: float = 0.0

    def landmark(self, index: int) -> Point:
        """Return the landmark at ``index`` (use the named constants)."""
        return self.landmarks[index]


class CapturedFrame(BaseModel):
    """One frame grabbed from the camera.

    ``image`` is a raw OpenCV BGR array; ``arbitrary_types_allowed`` lets it
    live on a pydantic model without per-pixel validation. ``timestamp`` is a
    :func:`time.perf_counter` reading taken at the moment of capture and is the
    anchor for the end-to-end latency budget. On the hot capture path build
    these with :meth:`pydantic.BaseModel.model_construct` to skip validation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: np.ndarray
    timestamp: float
    frame_index: int
