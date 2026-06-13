"""Structured logging to ``~/.alltap/logs/`` with daily rotation.

A single call to :func:`setup_logging` configures the root logger with a
console handler and a :class:`~logging.handlers.TimedRotatingFileHandler` that
rolls over at midnight and keeps a week of history. The level is read from the
``app.log_level`` config key by default.
"""

from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

LOG_DIR = Path.home() / ".alltap" / "logs"
_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

_configured = False


def setup_logging(level: Optional[str] = None, log_dir: Path = LOG_DIR) -> logging.Logger:
    """Configure root logging once and return the root logger.

    Args:
        level: Logging level name (e.g. ``"DEBUG"``). When ``None``, the level
            is read from config (``app.log_level``), defaulting to ``INFO``.
        log_dir: Directory to write rotating log files into.
    """
    global _configured

    if level is None:
        try:
            # Imported lazily to avoid a circular import at module load.
            from alltap.utils.config import get_config

            level = get_config().get("app.log_level", "INFO")
        except Exception:  # pragma: no cover - defensive: config not ready
            level = "INFO"

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    if _configured:
        root.setLevel(numeric_level)
        return root

    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            log_dir / "alltap.log", when="midnight", backupCount=7, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:  # pragma: no cover - filesystem edge case
        root.warning("Could not set up file logging in %s: %s", log_dir, exc)

    _configured = True
    return root
