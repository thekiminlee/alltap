# alltap

> Turn any non-touch monitor into a touch surface using a standard USB webcam.

People who use iPads instinctively reach out and tap regular monitors. **alltap**
makes that gesture actually work: clip a webcam to the top of your monitor and
use hand tracking to drive tap-to-click, swipe, and two-finger scroll — no touch
hardware required.

> **Status: early development (v0.1.0).** This is being built section by section.
> The scaffold, camera capture, MediaPipe hand tracking, and gesture
> classification (tap / swipe / two-finger scroll) are in place; calibration and
> OS input are not yet wired up. See [Roadmap](#roadmap) for what works today.

---

## Roadmap

| Section | Area | Status |
| ------- | ---- | ------ |
| 0 | Project scaffold, config, logging, shared types | ✅ Done |
| 1 | Camera capture | ✅ Done |
| 2 | Hand tracker (MediaPipe) | ✅ Done |
| 3 | Gesture classification (tap / swipe / scroll) | ✅ Done |
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
runtime), `target_fps` / `fallback_fps`, and `width` / `height`. The `tracker`
block controls hand detection: `max_num_hands`, the `min_*_confidence` gates,
and `mirrored` (set `true` if you run the webcam as a mirror, so handedness
labels match your real hands).

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

## Hand tracking

The tracker wraps MediaPipe's `HandLandmarker` (Tasks API, VIDEO mode) and turns
a frame into `Hand` objects — each 21 normalized landmarks plus handedness. We
do no ML ourselves; MediaPipe supplies the landmarks. The model ships in the
repo at [`assets/models/hand_landmarker.task`](assets/models/) and is bundled at
packaging time (no runtime download).

```python
from alltap.camera import Camera
from alltap.tracker import HandTracker
from alltap.types import INDEX_TIP

with Camera() as cam, HandTracker() as tracker:
    for frame in cam.frames():
        for hand in tracker.detect(frame):
            tip = hand.landmark(INDEX_TIP)  # index fingertip, normalized 0..1
            print(hand.handedness, tip.x, tip.y)
```

To verify tracking on a real webcam:

```bash
uv run python scripts/check_tracker.py
```

> On Linux, MediaPipe needs system GL libraries (e.g. `libgl1`, `libglesv2`) at
> runtime. macOS and Windows include these. (The unit tests don't require them.)

## Gesture classification

The classifier turns a *stream* of `Hand` frames into discrete gestures. It is
pure logic — no camera, model, or OS — and works in normalized coordinates, so a
single frame is never a gesture; intent emerges from motion over time. Feed it
the detected hands plus the frame timestamp:

```python
from alltap.gestures.classifier import GestureClassifier

classifier = GestureClassifier()
for frame in cam.frames():
    event = classifier.update(tracker.detect(frame), frame.timestamp)
    # event.type -> NONE / HOVER / TAP / SWIPE_* / SCROLL_*
    # event.position -> normalized fingertip (HOVER, TAP); mapped to a pixel later
```

It models touchscreen semantics:

- **Activation zone** — gestures only fire when the hand is close to the screen
  (a tight wrist-to-middle-finger distance gate), which keeps stray motion out.
- **Tap** — the fingertip thrusts in and decelerates sharply (a poke); the
  location is sampled at the contact point. Two quick taps become an OS
  double-click — we don't track double-taps ourselves.
- **Swipe** — a tap immediately followed by a lateral drag (poke-then-drag),
  decided within a short window after contact.
- **Scroll** — index + middle extended (ring + pinky folded); their vertical
  motion scrolls continuously, faster motion = more ticks.

Tap/swipe detection is based on fingertip kinematics (not the wrist), so it works
regardless of camera tilt. All thresholds live in the `gestures` config block.

## Project layout

```
alltap/
  types.py            # shared Point / ScreenPoint / Hand pydantic models + landmark indices
  main.py             # entry point
  utils/
    config.py         # load/merge/validate config, dot-notation access
    logger.py         # rotating file + console logging
  camera.py           # resilient webcam capture (yields CapturedFrame)
  tracker.py          # MediaPipe HandLandmarker wrapper (frame -> list[Hand])
  gestures/           # tap / swipe / scroll classification (pure logic)
    classifier.py     #   orchestrator: hands + timestamp -> GestureEvent
    pointer.py        #   tap + swipe state machine
    scroll.py         #   two-finger scroll
    events.py         #   GestureType / GestureEvent
  calibration.py      # (Section 5)
  input/              # (Section 7)
  ui/                 # debug overlay + tray (Sections 4, 8)
assets/models/hand_landmarker.task  # bundled MediaPipe model
config/default_config.json
scripts/check_camera.py   # manual camera check (real webcam)
scripts/check_tracker.py  # manual tracker check (real webcam)
tests/
```

## License

[MIT](LICENSE) © 2026 Kimin Lee
