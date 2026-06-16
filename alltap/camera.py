"""Webcam capture: open a device and yield timestamped frames, resiliently.

This layer hides the messy realities of real hardware — webcams that lie about
their fps, a camera index that changes at runtime, and mid-session unplugs —
behind a small generator API. It never lets a raw OpenCV failure escape: an
unopenable device raises :class:`CameraError`, and a device that stops
delivering frames raises :class:`CameraDisconnectedError`, both of which the
tray layer (Section 8) turns into a user-facing notification.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, Iterator, Optional

import cv2

from alltap.types import CapturedFrame
from alltap.utils.config import Config, get_config

logger = logging.getLogger(__name__)

#: Consecutive failed reads before we declare the camera disconnected
#: (~0.3s at 30fps): long enough to ignore transient hiccups, short enough to
#: stay responsive.
DISCONNECT_THRESHOLD = 10

#: How many recent grab timestamps to keep for the measured-fps estimate.
_FPS_WINDOW = 30


class CameraError(Exception):
    """The camera could not be opened (not found, or already in use)."""


class CameraDisconnectedError(CameraError):
    """The camera stopped delivering frames mid-session (likely unplugged)."""


class Camera:
    """A resilient wrapper around :class:`cv2.VideoCapture`.

    All settings are read from config (``camera.*``), so the camera index can be
    changed at runtime and :meth:`frames` will re-open the new device without a
    restart.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or get_config()
        self._cap: Optional[cv2.VideoCapture] = None
        self._index: Optional[int] = None
        self._frame_index = 0
        self._reported_fps = 0.0
        self._timestamps: Deque[float] = deque(maxlen=_FPS_WINDOW)

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        """Open the camera at the configured index. Raises :class:`CameraError`."""
        index = int(self._config.get("camera.index", 0))
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"Could not open camera at index {index}. "
                "Is it connected and not in use by another app?"
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.get("camera.width"))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.get("camera.height"))
        self._reported_fps = self._negotiate_fps(cap)

        self._cap = cap
        self._index = index
        logger.info("Opened camera index %d (reported %.0ffps)", index, self._reported_fps)

    def _negotiate_fps(self, cap: cv2.VideoCapture) -> float:
        """Request the target fps, falling back to ``fallback_fps`` if unmet.

        ``CAP_PROP_FPS`` is advisory — many webcams ignore it — so the value
        returned here is only a hint. :attr:`measured_fps` is the real number.
        """
        target = float(self._config.get("camera.target_fps"))
        fallback = float(self._config.get("camera.fallback_fps"))

        cap.set(cv2.CAP_PROP_FPS, target)
        reported = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if reported and reported < target:
            logger.info(
                "Camera won't run at %.0ffps (reports %.0f); falling back to %.0ffps.",
                target,
                reported,
                fallback,
            )
            cap.set(cv2.CAP_PROP_FPS, fallback)
            reported = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return reported

    def release(self) -> None:
        """Release the underlying device (idempotent)."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # -- capture -----------------------------------------------------------
    def frames(self) -> Iterator[CapturedFrame]:
        """Yield :class:`CapturedFrame`s until the device is released or lost.

        Re-opens the device if ``camera.index`` changes at runtime. Raises
        :class:`CameraDisconnectedError` after ``DISCONNECT_THRESHOLD``
        consecutive failed reads.
        """
        if self._cap is None:
            self.open()

        consecutive_failures = 0
        try:
            while True:
                desired_index = int(self._config.get("camera.index", 0))
                if desired_index != self._index:
                    logger.info(
                        "Camera index changed %s -> %s; reopening.",
                        self._index,
                        desired_index,
                    )
                    self.release()
                    self.open()

                ok, image = self._cap.read()
                timestamp = time.perf_counter()

                if not ok:
                    consecutive_failures += 1
                    if consecutive_failures >= DISCONNECT_THRESHOLD:
                        raise CameraDisconnectedError(
                            f"Camera index {self._index} stopped delivering frames."
                        )
                    continue

                consecutive_failures = 0
                self._timestamps.append(timestamp)
                # Trusted source — skip pydantic validation on the hot path.
                frame = CapturedFrame.model_construct(
                    image=image, timestamp=timestamp, frame_index=self._frame_index
                )
                self._frame_index += 1
                yield frame
        finally:
            self.release()

    # -- introspection -----------------------------------------------------
    @property
    def measured_fps(self) -> float:
        """Actual frame rate measured over recent grabs (0 until enough data)."""
        if len(self._timestamps) < 2:
            return self._reported_fps
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / span
