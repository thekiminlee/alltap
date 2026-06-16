"""Configuration loading, merging, validation, and dot-notation access.

The full schema lives in ``config/default_config.json`` (shipped with the
package). On first launch the defaults are written to ``~/.alltap/config.json``;
on every load the user's file is deep-merged over the defaults so that missing
keys are filled in and the user's values win.

Values are range-validated on load: out-of-range numbers are clamped to the
nearest valid bound and a warning is logged, so a hand-edited config can never
put the app into an invalid state.

Usage::

    from alltap.utils.config import get_config
    cfg = get_config()
    fps = cfg.get("camera.target_fps")
    cfg.set("app.debug_mode", True)
    cfg.save()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

#: Root directory for all user state (config + logs).
CONFIG_DIR = Path.home() / ".alltap"
CONFIG_PATH = CONFIG_DIR / "config.json"

#: Bundled defaults, resolved relative to the repository/package root.
_DEFAULTS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default_config.json"

#: Validation bounds for numeric keys: dot-path -> (min, max). Anything outside
#: the range is clamped on load with a warning. Keys not listed are unvalidated.
_RANGES: Dict[str, Tuple[float, float]] = {
    "camera.index": (0, 16),
    "camera.target_fps": (1, 240),
    "camera.fallback_fps": (1, 240),
    "camera.width": (160, 7680),
    "camera.height": (120, 4320),
    "tracker.max_num_hands": (1, 4),
    "tracker.min_detection_confidence": (0.0, 1.0),
    "tracker.min_presence_confidence": (0.0, 1.0),
    "tracker.min_tracking_confidence": (0.0, 1.0),
    "calibration.averaging_frames": (1, 600),
    "gestures.activation_zone_threshold": (0.0, 1.0),
    "gestures.tap_threshold_px": (1, 2000),
    "gestures.tap_time_ms": (1, 5000),
    "gestures.tap_cooldown_ms": (0, 5000),
    "gestures.double_tap_window_ms": (1, 5000),
    "gestures.double_tap_position_tolerance_px": (1, 2000),
    "gestures.swipe_threshold_px": (1, 4000),
    "gestures.swipe_window_frames": (1, 120),
    "gestures.scroll_velocity_threshold": (0.0, 1000.0),
    "gestures.scroll_multiplier": (0.1, 20.0),
    "gestures.buffer_frames": (2, 300),
    "cursor.smoothing_alpha": (0.0, 1.0),
    "app.latency_warning_ms": (1, 5000),
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base`` (override wins)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_defaults() -> Dict[str, Any]:
    with open(_DEFAULTS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


class Config:
    """In-memory configuration backed by ``~/.alltap/config.json``."""

    def __init__(self, path: Path = CONFIG_PATH) -> None:
        self.path = path
        self._defaults = _load_defaults()
        self._data: Dict[str, Any] = dict(self._defaults)
        self.load()

    # -- persistence -------------------------------------------------------
    def load(self) -> None:
        """Load and validate config, writing defaults on first run."""
        user_data: Dict[str, Any] = {}
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    user_data = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Could not read config at %s (%s); falling back to defaults.",
                    self.path,
                    exc,
                )
                user_data = {}

        self._data = _deep_merge(self._defaults, user_data)
        self._validate()

        if not self.path.exists():
            logger.info("No config found; writing defaults to %s", self.path)
            self.save()

    def save(self) -> None:
        """Persist the current config to disk (creates the directory)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)

    # -- access ------------------------------------------------------------
    def get(self, key_path: str, default: Optional[Any] = None) -> Any:
        """Return the value at a dot-separated ``key_path`` (e.g. ``"camera.index"``)."""
        node: Any = self._data
        for part in key_path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, key_path: str, value: Any) -> None:
        """Set the value at a dot-separated ``key_path``, creating intermediate dicts."""
        parts = key_path.split(".")
        node = self._data
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of the full config tree."""
        return dict(self._data)

    # -- validation --------------------------------------------------------
    def _validate(self) -> None:
        for key_path, (low, high) in _RANGES.items():
            value = self.get(key_path)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            clamped = max(low, min(high, value))
            if clamped != value:
                logger.warning(
                    "Config value %s=%s out of range [%s, %s]; clamping to %s.",
                    key_path,
                    value,
                    low,
                    high,
                    clamped,
                )
                # Preserve int-ness for integer-typed defaults.
                if isinstance(value, int):
                    clamped = int(clamped)
                self.set(key_path, clamped)


_INSTANCE: Optional[Config] = None


def get_config(path: Optional[Path] = None) -> Config:
    """Return the process-wide :class:`Config` singleton.

    ``path`` is resolved at call time (falling back to the module-level
    ``CONFIG_PATH``) so tests can monkeypatch the location before first use.
    """
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = Config(path if path is not None else CONFIG_PATH)
    return _INSTANCE


def reset_config() -> None:
    """Drop the cached singleton (primarily for tests)."""
    global _INSTANCE
    _INSTANCE = None
