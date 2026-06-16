"""Manual tracker check — NOT run in CI (requires a real webcam).

Opens the configured camera, runs the hand tracker, and prints how many hands
are detected plus the index-fingertip position, so you can confirm Section 2
works on real hardware. Full visual confirmation arrives with the Section 4
debug overlay.

    uv run python scripts/check_tracker.py
"""

from __future__ import annotations

import itertools

from alltap.camera import Camera
from alltap.tracker import HandTracker
from alltap.types import INDEX_TIP
from alltap.utils.logger import setup_logging


def main() -> int:
    setup_logging("INFO")
    with Camera() as cam, HandTracker() as tracker:
        for frame in itertools.islice(cam.frames(), 300):
            hands = tracker.detect(frame)
            if frame.frame_index % 15 == 0:
                if hands:
                    tip = hands[0].landmark(INDEX_TIP)
                    print(
                        f"frame {frame.frame_index:>4}: {len(hands)} hand(s) "
                        f"[{hands[0].handedness}] index_tip=({tip.x:.2f}, {tip.y:.2f})"
                    )
                else:
                    print(f"frame {frame.frame_index:>4}: no hands")
    print("done — tracker works")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
