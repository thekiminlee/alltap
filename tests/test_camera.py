"""Mock-based tests for the camera layer (headless / CI-safe).

A fake ``cv2.VideoCapture`` stands in for real hardware so we can exercise fps
fallback, index hot-swap, the not-found error, and disconnect detection without
a webcam.
"""

import cv2
import numpy as np
import pytest

from alltap import camera as camera_mod
from alltap.camera import Camera, CameraDisconnectedError, CameraError
from alltap.utils.config import Config


class FakeCapture:
    """Minimal stand-in for cv2.VideoCapture."""

    def __init__(self, index, *, opened=True, max_fps=60, good_frames=None):
        self.index = index
        self._opened = opened
        self.max_fps = max_fps
        self.good_frames = good_frames  # None => always succeed
        self.props = {}
        self.released = False
        self.read_count = 0

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_FPS:
            value = min(value, self.max_fps)  # device caps fps
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0)

    def read(self):
        self.read_count += 1
        if self.good_frames is not None and self.read_count > self.good_frames:
            return False, None
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def cfg(tmp_path):
    return Config(path=tmp_path / "config.json")


def _patch_factory(monkeypatch, factory):
    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", factory)


def test_fps_fallback(monkeypatch, cfg):
    cfg.set("camera.target_fps", 60)
    cfg.set("camera.fallback_fps", 30)
    fake = FakeCapture(0, max_fps=30)  # device can't do 60
    _patch_factory(monkeypatch, lambda idx: fake)

    cam = Camera(config=cfg)
    cam.open()

    # Negotiation fell back to 30 (the device ceiling).
    assert fake.get(cv2.CAP_PROP_FPS) == 30
    assert cam.measured_fps == 30


def test_camera_not_found_raises_and_releases(monkeypatch, cfg):
    fake = FakeCapture(0, opened=False)
    _patch_factory(monkeypatch, lambda idx: fake)

    cam = Camera(config=cfg)
    with pytest.raises(CameraError):
        cam.open()
    assert fake.released  # we don't leak the handle on failure


def test_index_hot_swap_reopens(monkeypatch, cfg):
    cfg.set("camera.index", 0)
    captures = {}

    def factory(idx):
        cap = FakeCapture(idx)
        captures[idx] = cap
        return cap

    _patch_factory(monkeypatch, factory)

    cam = Camera(config=cfg)
    gen = cam.frames()

    first = next(gen)
    assert first.image.shape == (480, 640, 3)

    cfg.set("camera.index", 1)  # user changes camera at runtime
    next(gen)

    assert captures[0].released
    assert 1 in captures
    assert cam._index == 1

    gen.close()


def test_disconnect_after_consecutive_failures(monkeypatch, cfg):
    fake = FakeCapture(0, good_frames=2)  # 2 good frames, then fails forever
    _patch_factory(monkeypatch, lambda idx: fake)

    cam = Camera(config=cfg)
    gen = cam.frames()

    assert next(gen).frame_index == 0
    assert next(gen).frame_index == 1

    # The next pull loops internally over failed reads until the threshold.
    with pytest.raises(CameraDisconnectedError):
        next(gen)


def test_context_manager_releases(monkeypatch, cfg):
    fake = FakeCapture(0)
    _patch_factory(monkeypatch, lambda idx: fake)

    with Camera(config=cfg):
        pass
    assert fake.released
