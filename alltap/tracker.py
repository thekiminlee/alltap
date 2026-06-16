"""Hand tracking: wrap MediaPipe Hands and emit clean :class:`Hand` objects.

This is the bridge between raw pixels (the camera) and gesture logic: a
:class:`CapturedFrame` goes in, a list of :class:`Hand` objects — each 21
normalized landmarks plus handedness — comes out. We do no ML ourselves;
MediaPipe supplies the landmarks and we only reshape them into our own types.

We use the MediaPipe **Tasks** ``HandLandmarker`` API in VIDEO mode (the legacy
``solutions`` API is not shipped in current mediapipe builds). VIDEO mode needs
a monotonically increasing millisecond timestamp per frame, which is exactly
what :attr:`CapturedFrame.timestamp` provides. The model lives in the repo at
``assets/models/hand_landmarker.task`` and is bundled at packaging time.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from alltap.types import CapturedFrame, Hand, Point
from alltap.utils.config import Config, get_config

logger = logging.getLogger(__name__)

_OPPOSITE_HAND = {"Left": "Right", "Right": "Left"}
_MODEL_FILENAME = "hand_landmarker.task"


def _model_path() -> Path:
    """Resolve the bundled model, handling both source and frozen (PyInstaller) runs."""
    frozen_base = getattr(sys, "_MEIPASS", None)
    base = Path(frozen_base) if frozen_base else Path(__file__).resolve().parent.parent
    return base / "assets" / "models" / _MODEL_FILENAME


class HandTracker:
    """Detect hands in a frame and return them as :class:`Hand` objects.

    All settings come from the ``tracker.*`` config block. Use as a context
    manager (or call :meth:`close`) so the MediaPipe graph is released.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or get_config()
        self._mirrored = bool(self._config.get("tracker.mirrored", False))
        self._last_timestamp_ms = -1  # VIDEO mode needs strictly increasing stamps

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_model_path())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=int(self._config.get("tracker.max_num_hands", 2)),
            min_hand_detection_confidence=float(
                self._config.get("tracker.min_detection_confidence", 0.5)
            ),
            min_hand_presence_confidence=float(
                self._config.get("tracker.min_presence_confidence", 0.5)
            ),
            min_tracking_confidence=float(
                self._config.get("tracker.min_tracking_confidence", 0.5)
            ),
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame: CapturedFrame) -> List[Hand]:
        """Return the hands found in ``frame`` (``[]`` when none are present)."""
        # MediaPipe expects RGB; OpenCV frames are BGR.
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires a strictly increasing integer ms timestamp.
        timestamp_ms = max(int(frame.timestamp * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        results = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return self._to_hands(results)

    def _to_hands(self, results: Any) -> List[Hand]:
        """Map a MediaPipe results object onto our :class:`Hand` list.

        Kept separate from :meth:`detect` so the mapping is unit-testable with a
        stubbed results object — no real model required.
        """
        landmark_sets = getattr(results, "hand_landmarks", None)
        if not landmark_sets:
            return []

        handedness_sets = getattr(results, "handedness", None) or []

        hands: List[Hand] = []
        for i, landmarks in enumerate(landmark_sets):
            # Hot path (21 landmarks x up to 2 hands per frame): skip validation.
            points = [
                Point.model_construct(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks
            ]

            handedness, confidence = "", 0.0
            if i < len(handedness_sets):
                category = handedness_sets[i][0]
                handedness = category.category_name
                confidence = category.score
                if self._mirrored:
                    handedness = _OPPOSITE_HAND.get(handedness, handedness)

            logger.debug("Detected %s hand (confidence %.2f)", handedness or "?", confidence)
            hands.append(
                Hand.model_construct(
                    landmarks=points, handedness=handedness, confidence=confidence
                )
            )
        return hands

    def close(self) -> None:
        """Release the MediaPipe graph (idempotent)."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
