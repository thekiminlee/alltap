# alltap

> Turn any non-touch monitor into a touch surface using a standard USB webcam.

People who use iPads instinctively reach out and tap regular monitors. **alltap**
makes that gesture actually work: clip a webcam to the top of your monitor and
use hand tracking to drive tap-to-click, swipe, and two-finger scroll — no touch
hardware required.

> **Status: early development (v0.1.0).** This is being built section by section.
> The scaffold (shared types, configuration, logging) and a resilient camera
> capture layer are in place; hand tracking and gestures are not yet wired up.
> See [Roadmap](#roadmap) for what works today.

---

## Roadmap

| Section | Area | Status |
| ------- | ---- | ------ |
| 0 | Project scaffold, config, logging, shared types | ✅ Done |
| 1 | Camera capture | ✅ Done |
| 2 | Hand tracker (MediaPipe) | ⬜ Planned |
| 3 | Gesture classification (tap / swipe / scroll) | ⬜ Planned |
| 4 | Debug overlay | ⬜ Planned |
| 5 | Calibration (4-corner homography) | ⬜ Planned |
| 6 | Configuration system (finalized) | ⬜ Planned |
| 7 | OS input dispatch | ⬜ Planned |
| 8 | System-tray app & lifecycle | ⬜ Planned |
| 9 | Packaging & distribution | ⬜ Planned |

## System requirements

- **OS:** macOS 12+ or Windows 10+ (target platforms for v0.1.0)
- **Camera:** any USB webcam capable of 30fps
- **Python:** 3.10–3.12
- **Tooling:** [`uv`](https://docs.astral.sh/uv/) for the virtualenv and dependencies

## Getting started (development)

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
# 1. Install uv (see https://docs.astral.sh/uv/getting-started/installation/)

# 2. Clone and sync — creates a .venv and installs all dependencies from uv.lock
git clone https://github.com/thekiminlee/alltap.git
cd alltap
uv sync

# 3. Run the app (scaffold only at this stage — initializes config + logging and exits)
uv run python -m alltap.main

# 4. Run the test suite
uv run pytest
```

On first run, alltap creates its state directory at `~/.alltap/`:

- `~/.alltap/config.json` — your configuration (written from defaults on first launch)
- `~/.alltap/logs/` — daily-rotating log files

## Configuration

All tunable parameters live in `~/.alltap/config.json`, created on first launch
from [`config/default_config.json`](config/default_config.json). Your file is
deep-merged over the defaults, so you only need to include the keys you want to
change; out-of-range values are clamped with a warning on load.

Sections cover `camera`, `calibration`, `gestures`, `cursor`, `keybindings`,
and `app`. (The full schema is finalized in Section 6.)

The `camera` block controls capture: `index` (which webcam — changeable at
runtime), `target_fps` / `fallback_fps`, and `width` / `height`.

## Camera capture

The camera layer yields timestamped frames and tolerates real-hardware quirks —
fps that can't be honored, a camera index changed at runtime, and mid-session
unplugs (raised as a typed `CameraDisconnectedError` rather than a crash).

```python
from alltap.camera import Camera

with Camera() as cam:
    for frame in cam.frames():
        # frame.image is a BGR numpy array; frame.timestamp anchors latency
        print(frame.frame_index, frame.image.shape, cam.measured_fps)
```

To verify capture on a real webcam:

```bash
uv run python scripts/check_camera.py
```

## Project layout

```
alltap/
  types.py            # shared Point / ScreenPoint / Hand pydantic models + landmark indices
  main.py             # entry point
  utils/
    config.py         # load/merge/validate config, dot-notation access
    logger.py         # rotating file + console logging
  camera.py           # resilient webcam capture (yields CapturedFrame)
  tracker.py          # (Section 2)
  gestures/           # (Section 3)
  calibration.py      # (Section 5)
  input/              # (Section 7)
  ui/                 # debug overlay + tray (Sections 4, 8)
config/default_config.json
scripts/check_camera.py  # manual camera check (real webcam)
tests/
```

## License

[MIT](LICENSE) © 2026 Kimin Lee
