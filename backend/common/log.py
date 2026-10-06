"""
TerraAlert — project-wide logging helper.

Usage::

    from backend.common.log import get_logger
    logger = get_logger(__name__)
    logger.info("Processing %s", filename)

All loggers share a consistent format with timestamp, level, and module name.
Log level is controlled by the ``TERRAALERT_LOG_LEVEL`` environment variable
(default: ``INFO``).
"""
from __future__ import annotations

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def _configure_root() -> None:
    """One-time setup of the root logger."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("TERRAALERT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if called more than once
    if not root.handlers:
        root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring the root is configured."""
    _configure_root()
    return logging.getLogger(name)
