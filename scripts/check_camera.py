"""Manual camera check — NOT run in CI (requires a real webcam).

Opens the configured camera, grabs ~120 frames, and prints the frame shape and
measured fps so you can confirm Section 1 works on real hardware.

    uv run python scripts/check_camera.py
"""

from __future__ import annotations

import itertools

from alltap.camera import Camera
from alltap.utils.logger import setup_logging


def main() -> int:
    setup_logging("INFO")
    with Camera() as cam:
        for frame in itertools.islice(cam.frames(), 120):
            if frame.frame_index % 30 == 0:
                print(
                    f"frame {frame.frame_index:>4}: "
                    f"shape={frame.image.shape} "
                    f"measured_fps={cam.measured_fps:.1f}"
                )
    print("done — camera capture works")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
