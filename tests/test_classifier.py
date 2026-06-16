"""Gesture classifier tests driven by synthetic landmark sequences.

No camera, no model, no calibration — we hand-build Hand frames with timestamps
and assert the classifier produces the right gestures. This is where the
product's correctness is really earned.

Grammar under test (everything starts with a poke = "touch down"):
  poke, no drag            -> TAP
  poke, 1-finger drag      -> SCROLL (continuous)
  poke, 2-finger drag      -> SWIPE
  in-zone, no gesture      -> HOVER
"""

import pytest

from alltap.gestures.classifier import GestureClassifier
from alltap.gestures.events import GestureType
from alltap.types import NUM_LANDMARKS, Hand, Point


@pytest.fixture
def cfg(tmp_path):
    from alltap.utils.config import Config

    return Config(path=tmp_path / "config.json")


# --- synthetic hand builder ------------------------------------------------
#
# Zone uses wrist(0) -> middle-MCP(9) (palm size, pose-invariant). Finger count
# uses whether the middle finger (tip 12 vs pip 10) is extended; ring/pinky are
# ignored. `two_finger=True` extends the middle finger; otherwise it's folded.

def make_hand(index_tip=(0.5, 0.40), *, two_finger=False, in_zone=True):
    wrist = (0.5, 0.95) if in_zone else (0.5, 0.58)

    pts = [[0.5, 0.5, 0.0] for _ in range(NUM_LANDMARKS)]
    pts[0] = [wrist[0], wrist[1], 0.0]
    pts[9] = [0.50, 0.65, 0.0]                    # middle MCP (palm size for zone)
    pts[8] = [index_tip[0], index_tip[1], 0.0]    # index tip (pointer)
    pts[10] = [0.55, 0.63, 0.0]                   # middle PIP
    pts[12] = [0.55, 0.40, 0.0] if two_finger else [0.55, 0.80, 0.0]  # middle tip

    landmarks = [Point.model_construct(x=p[0], y=p[1], z=p[2]) for p in pts]
    return Hand.model_construct(landmarks=landmarks, handedness="Right", confidence=0.9)


def run(classifier, frames):
    return [classifier.update([hand], t).type for hand, t in frames]


def fired(types):
    return [t for t in types if t not in (GestureType.NONE, GestureType.HOVER)]


# A poke: rest -> fast thrust -> stop (contact). Times in seconds.
_POKE = [
    ((0.5, 0.40), 0.000),
    ((0.5, 0.475), 0.033),
    ((0.5, 0.550), 0.066),
    ((0.5, 0.550), 0.099),
]


def _poke_frames(two_finger=False):
    return [(make_hand(pos, two_finger=two_finger), t) for pos, t in _POKE]


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
    frames, t, y = [], 0.0, 0.40
    for _ in range(8):
        frames.append((make_hand((0.5, y), in_zone=False), t))
        t, y = t + 0.033, y + 0.08
    assert all(x is GestureType.NONE for x in run(c, frames))


def test_poke_no_drag_is_tap(cfg):
    c = GestureClassifier(config=cfg)
    frames = _poke_frames() + [
        (make_hand((0.5, 0.550)), 0.150),
        (make_hand((0.5, 0.550)), 0.260),  # past decision window -> TAP
    ]
    assert fired(run(c, frames)) == [GestureType.TAP]


def test_tap_samples_contact_point(cfg):
    c = GestureClassifier(config=cfg)
    frames = _poke_frames() + [(make_hand((0.5, 0.550)), 0.260)]
    events = [c.update([h], t) for h, t in frames]
    tap = next(e for e in events if e.type is GestureType.TAP)
    assert tap.position.y == pytest.approx(0.550)  # contact, not the rest pose


def test_slow_movement_does_not_fire(cfg):
    c = GestureClassifier(config=cfg)
    frames, t, y = [], 0.0, 0.40
    for _ in range(12):
        frames.append((make_hand((0.5, y)), t))
        t, y = t + 0.033, y + 0.01  # under the speed threshold
    assert fired(run(c, frames)) == []


def test_poke_then_one_finger_drag_is_scroll(cfg):
    c = GestureClassifier(config=cfg)
    frames = _poke_frames(two_finger=False) + [
        (make_hand((0.5, 0.64)), 0.130),  # 1-finger drag down past threshold
        (make_hand((0.5, 0.73)), 0.163),
        (make_hand((0.5, 0.82)), 0.196),
    ]
    got = fired(run(c, frames))
    assert got and all(x is GestureType.SCROLL_DOWN for x in got)


def test_poke_then_two_finger_drag_is_swipe(cfg):
    c = GestureClassifier(config=cfg)
    frames = _poke_frames(two_finger=True) + [
        (make_hand((0.62, 0.55), two_finger=True), 0.130),  # 2-finger drag right
    ]
    assert fired(run(c, frames)) == [GestureType.SWIPE_RIGHT]


def test_same_drag_differs_by_finger_count(cfg):
    """A vertical drag scrolls with one finger but swipes with two."""
    drag = [((0.5, 0.64), 0.130), ((0.5, 0.73), 0.163), ((0.5, 0.82), 0.196)]

    one = GestureClassifier(config=cfg)
    one_frames = _poke_frames(False) + [(make_hand(p, two_finger=False), t) for p, t in drag]
    assert GestureType.SCROLL_DOWN in fired(run(one, one_frames))

    two = GestureClassifier(config=cfg)
    two_frames = _poke_frames(True) + [(make_hand(p, two_finger=True), t) for p, t in drag]
    assert GestureType.SWIPE_DOWN in fired(run(two, two_frames))
