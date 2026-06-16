"""Tests for the hand tracker (headless / CI-safe).

The MediaPipe model is never loaded: we patch ``HandLandmarker.create_from_options``
with a fake whose ``detect_for_video()`` returns a stubbed Tasks results object,
so we exercise the BGR->RGB call and the landmark->Hand mapping without a model.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from alltap import tracker as tracker_mod
from alltap.tracker import HandTracker
from alltap.types import NUM_LANDMARKS, CapturedFrame
from alltap.utils.config import Config


class FakeLandmarker:
    """Stand-in for mediapipe Tasks HandLandmarker."""

    def __init__(self, results):
        self._results = results
        self.received = None
        self.closed = False

    def detect_for_video(self, mp_image, timestamp_ms):
        self.received = (mp_image, timestamp_ms)
        return self._results

    def close(self):
        self.closed = True


def _landmarks(n=NUM_LANDMARKS):
    return [SimpleNamespace(x=i / n, y=0.5, z=0.0) for i in range(n)]


def _handedness(label, score=0.9):
    # Tasks: results.handedness[i] is a list of categories; [0] is the top one.
    return [SimpleNamespace(category_name=label, score=score, index=0)]


def _results(hand_landmarks, handedness):
    return SimpleNamespace(hand_landmarks=hand_landmarks, handedness=handedness)


@pytest.fixture
def cfg(tmp_path):
    return Config(path=tmp_path / "config.json")


def _make_tracker(monkeypatch, cfg, results):
    fake = FakeLandmarker(results)
    monkeypatch.setattr(
        tracker_mod.vision.HandLandmarker,
        "create_from_options",
        lambda options: fake,
    )
    # Avoid constructing the real mp.Image (loads native GL libs not present in
    # headless CI); the mapping under test doesn't care about its internals.
    monkeypatch.setattr(
        tracker_mod.mp, "Image", lambda image_format, data: SimpleNamespace(data=data)
    )
    return HandTracker(config=cfg), fake


def _frame(timestamp=0.0):
    return CapturedFrame.model_construct(
        image=np.zeros((10, 10, 3), dtype=np.uint8), timestamp=timestamp, frame_index=0
    )


def test_no_hands_returns_empty(monkeypatch, cfg):
    tracker, _ = _make_tracker(monkeypatch, cfg, _results([], []))
    assert tracker.detect(_frame()) == []


def test_single_hand_maps_21_points(monkeypatch, cfg):
    results = _results([_landmarks()], [_handedness("Right")])
    tracker, fake = _make_tracker(monkeypatch, cfg, results)

    hands = tracker.detect(_frame())

    assert len(hands) == 1
    assert len(hands[0].landmarks) == NUM_LANDMARKS
    assert hands[0].handedness == "Right"
    assert hands[0].confidence == pytest.approx(0.9)
    # detect() built an mp.Image and passed a timestamp through.
    assert fake.received is not None


def test_two_hands(monkeypatch, cfg):
    results = _results(
        [_landmarks(), _landmarks()],
        [_handedness("Left"), _handedness("Right")],
    )
    tracker, _ = _make_tracker(monkeypatch, cfg, results)

    hands = tracker.detect(_frame())
    assert [h.handedness for h in hands] == ["Left", "Right"]


def test_mirrored_swaps_handedness(monkeypatch, cfg):
    cfg.set("tracker.mirrored", True)
    results = _results([_landmarks()], [_handedness("Right")])
    tracker, _ = _make_tracker(monkeypatch, cfg, results)

    hands = tracker.detect(_frame())
    assert hands[0].handedness == "Left"  # relabeled for a mirrored feed


def test_timestamp_is_strictly_increasing(monkeypatch, cfg):
    tracker, fake = _make_tracker(monkeypatch, cfg, _results([], []))

    # Two frames with the same capture time must still yield increasing stamps.
    tracker.detect(_frame(timestamp=1.0))
    first = fake.received[1]
    tracker.detect(_frame(timestamp=1.0))
    second = fake.received[1]
    assert second > first


def test_close_releases_graph(monkeypatch, cfg):
    tracker, fake = _make_tracker(monkeypatch, cfg, _results([], []))
    with tracker:
        pass
    assert fake.closed
