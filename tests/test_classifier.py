"""Gesture classifier tests driven by synthetic landmark sequences.

No camera, no model, no calibration — we hand-build Hand frames with timestamps
and assert the classifier produces the right gestures. This is where the
product's correctness is really earned.
"""

import pytest

from alltap.gestures.classifier import GestureClassifier
from alltap.gestures.events import GestureType
from alltap.types import NUM_LANDMARKS, Hand, Point
from alltap.utils.config import Config

FPS_DT = 1 / 30  # seconds per frame at 30fps


@pytest.fixture
def cfg(tmp_path):
    return Config(path=tmp_path / "config.json")


# --- synthetic hand builder ------------------------------------------------
#
# Landmark layout we care about (wrist far from fingertips -> "in zone"):
#   0 wrist, 8 index tip, 12 middle tip, plus PIPs 6/10/14/18 and tips 16/20.
# Defaults put all fingers extended (so it is NOT a scroll pose).

def make_hand(index_tip=(0.5, 0.40), *, wrist=(0.5, 0.95), scroll_pose=False, in_zone=True):
    if not in_zone:
        # Put the wrist near the middle tip so the zone distance is tiny.
        wrist = (0.5, 0.46)

    pts = [[0.5, 0.5, 0.0] for _ in range(NUM_LANDMARKS)]
    pts[0] = [wrist[0], wrist[1], 0.0]
    pts[8] = [index_tip[0], index_tip[1], 0.0]   # index tip
    pts[6] = [0.50, 0.62, 0.0]                    # index pip
    pts[12] = [0.55, 0.40, 0.0]                   # middle tip (extended)
    pts[10] = [0.55, 0.62, 0.0]                   # middle pip
    # Ring + pinky: extended by default, folded for the scroll pose.
    ring_tip = (0.60, 0.85) if scroll_pose else (0.60, 0.40)
    pinky_tip = (0.65, 0.85) if scroll_pose else (0.65, 0.40)
    pts[16], pts[14] = [ring_tip[0], ring_tip[1], 0.0], [0.60, 0.62, 0.0]
    pts[20], pts[18] = [pinky_tip[0], pinky_tip[1], 0.0], [0.65, 0.62, 0.0]

    landmarks = [Point.model_construct(x=p[0], y=p[1], z=p[2]) for p in pts]
    return Hand.model_construct(landmarks=landmarks, handedness="Right", confidence=0.9)


def run(classifier, frames):
    """Feed (hand, timestamp) frames; return the list of GestureTypes produced."""
    return [classifier.update([hand], t).type for hand, t in frames]


def fired(types):
    """Gesture types that are actual actions (not NONE/HOVER)."""
    return [t for t in types if t not in (GestureType.NONE, GestureType.HOVER)]


# --- tests -----------------------------------------------------------------

def test_no_hand_is_none(cfg):
    c = GestureClassifier(config=cfg)
    assert c.update([], 0.0).type is GestureType.NONE


def test_hand_in_zone_hovers(cfg):
    c = GestureClassifier(config=cfg)
    ev = c.update([make_hand((0.5, 0.4))], 0.0)
    assert ev.type is GestureType.HOVER
    assert ev.position is not None


def test_out_of_zone_never_fires(cfg):
    c = GestureClassifier(config=cfg)
    # A fast thrust, but the hand is out of the activation zone the whole time.
    frames, t, y = [], 0.0, 0.40
    for _ in range(8):
        frames.append((make_hand((0.5, y), in_zone=False), t))
        t += FPS_DT
        y += 0.08
    types = run(c, frames)
    assert all(x is GestureType.NONE for x in types)


def test_valid_tap_fires(cfg):
    c = GestureClassifier(config=cfg)
    frames = [
        (make_hand((0.5, 0.40)), 0.000),  # rest
        (make_hand((0.5, 0.475)), 0.033),  # fast thrust
        (make_hand((0.5, 0.550)), 0.066),  # fast thrust (peak)
        (make_hand((0.5, 0.550)), 0.099),  # stop -> contact
        (make_hand((0.5, 0.550)), 0.150),  # holding...
        (make_hand((0.5, 0.550)), 0.200),  # holding...
        (make_hand((0.5, 0.550)), 0.260),  # > decision window -> TAP
    ]
    types = run(c, frames)
    assert fired(types) == [GestureType.TAP]


def test_tap_samples_contact_point(cfg):
    c = GestureClassifier(config=cfg)
    frames = [
        (make_hand((0.5, 0.40)), 0.000),
        (make_hand((0.5, 0.475)), 0.033),
        (make_hand((0.5, 0.550)), 0.066),
        (make_hand((0.5, 0.550)), 0.099),
        (make_hand((0.5, 0.550)), 0.260),
    ]
    events = [c.update([h], t) for h, t in frames]
    tap = next(e for e in events if e.type is GestureType.TAP)
    # Location is the contact (decel) point, not the pre-thrust rest position.
    assert tap.position.y == pytest.approx(0.550)


def test_slow_movement_does_not_tap(cfg):
    c = GestureClassifier(config=cfg)
    frames, t, y = [], 0.0, 0.40
    for _ in range(12):
        frames.append((make_hand((0.5, y)), t))
        t += FPS_DT
        y += 0.01  # slow drift, well under the speed threshold
    types = run(c, frames)
    assert fired(types) == []


def test_poke_then_drag_is_swipe(cfg):
    c = GestureClassifier(config=cfg)
    frames = [
        (make_hand((0.5, 0.40)), 0.000),
        (make_hand((0.5, 0.475)), 0.033),  # thrust
        (make_hand((0.5, 0.550)), 0.066),  # thrust (peak)
        (make_hand((0.5, 0.550)), 0.099),  # contact
        (make_hand((0.62, 0.550)), 0.130),  # drag right past threshold
    ]
    types = run(c, frames)
    assert fired(types) == [GestureType.SWIPE_RIGHT]


def test_two_finger_scroll_fires_continuously(cfg):
    c = GestureClassifier(config=cfg)
    frames, t, y = [], 0.0, 0.60
    for _ in range(5):
        frames.append((make_hand((0.5, y), scroll_pose=True), t))
        t += FPS_DT
        y -= 0.03  # both fingers move up
    types = run(c, frames)
    assert all(x is GestureType.SCROLL_UP for x in fired(types))
    assert len(fired(types)) >= 3  # continuous, not a single event


def test_scroll_pose_suppresses_tap(cfg):
    c = GestureClassifier(config=cfg)
    # Same fast vertical motion, but in the scroll pose -> never a TAP.
    frames = [
        (make_hand((0.5, 0.40), scroll_pose=True), 0.000),
        (make_hand((0.5, 0.475), scroll_pose=True), 0.033),
        (make_hand((0.5, 0.550), scroll_pose=True), 0.066),
        (make_hand((0.5, 0.550), scroll_pose=True), 0.260),
    ]
    types = run(c, frames)
    assert GestureType.TAP not in types
