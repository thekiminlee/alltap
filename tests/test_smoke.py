"""Smoke tests: every module imports, and the scaffold's basics work.

These run headless (no camera, no display) and are safe for CI.
"""

import importlib

import pytest

# Modules that have no heavy/hardware imports yet at Section 0.
SCAFFOLD_MODULES = [
    "alltap",
    "alltap.types",
    "alltap.main",
    "alltap.utils.config",
    "alltap.utils.logger",
    "alltap.gestures.events",
    "alltap.gestures.geometry",
    "alltap.gestures.pointer",
    "alltap.gestures.scroll",
    "alltap.gestures.classifier",
]


@pytest.mark.parametrize("module_name", SCAFFOLD_MODULES)
def test_module_imports(module_name):
    assert importlib.import_module(module_name) is not None


def test_shared_types_construct():
    from alltap.types import Hand, Point, ScreenPoint, INDEX_TIP, NUM_LANDMARKS

    landmarks = [Point(x=i / NUM_LANDMARKS, y=0.5) for i in range(NUM_LANDMARKS)]
    hand = Hand(landmarks=landmarks, handedness="Right", confidence=0.9)

    assert len(hand.landmarks) == NUM_LANDMARKS
    assert hand.landmark(INDEX_TIP) == landmarks[INDEX_TIP]
    assert ScreenPoint(x=10, y=20).x == 10

    # pydantic gives us JSON round-tripping and type coercion for free.
    restored = Hand.model_validate_json(hand.model_dump_json())
    assert restored == hand
    assert ScreenPoint(x="10", y=20).x == 10  # coerced to int


def test_point_validates_types():
    from pydantic import ValidationError

    from alltap.types import Point

    with pytest.raises(ValidationError):
        Point(x="not-a-number", y=0.5)


def test_config_loads_and_round_trips(tmp_path):
    from alltap.utils.config import Config

    cfg = Config(path=tmp_path / "config.json")

    # Defaults are present and dot-access works.
    assert cfg.get("camera.target_fps") == 60
    assert cfg.get("app.debug_mode") is False
    assert cfg.get("nonexistent.key", "fallback") == "fallback"

    # First run writes the file.
    assert cfg.path.exists()

    # set + save + reload preserves the value.
    cfg.set("app.debug_mode", True)
    cfg.save()
    reloaded = Config(path=cfg.path)
    assert reloaded.get("app.debug_mode") is True


def test_main_runs_clean(tmp_path, monkeypatch):
    import alltap.utils.config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "config.json")
    config_mod.reset_config()

    from alltap.main import main

    assert main() == 0
