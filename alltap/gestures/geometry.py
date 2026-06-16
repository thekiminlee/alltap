"""Small geometry helpers shared by the gesture detectors."""

from __future__ import annotations

import math

from alltap.types import Point


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two normalized points (x/y only)."""
    return math.hypot(a.x - b.x, a.y - b.y)
