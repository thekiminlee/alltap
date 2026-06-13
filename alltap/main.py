"""alltap entry point.

At this stage (Section 0) the entry point only initializes configuration and
logging, reports that it started, and exits cleanly. Later sections wire in the
camera/tracking/gesture pipeline and the system-tray lifecycle.
"""

from __future__ import annotations

import logging

from alltap import __version__
from alltap.utils.config import get_config
from alltap.utils.logger import setup_logging


def main() -> int:
    """Initialize config + logging and exit cleanly. Returns a process exit code."""
    config = get_config()
    setup_logging(config.get("app.log_level", "INFO"))
    logger = logging.getLogger(__name__)

    logger.info("alltap v%s starting up", __version__)
    logger.info("Config loaded from %s", config.path)
    logger.debug("Active configuration: %s", config.as_dict())

    # Section 0 scaffold: nothing to run yet. The tray/lifecycle loop arrives in
    # Section 8.
    logger.info("Scaffold ready. (Gesture pipeline not yet wired — see Section 8.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
