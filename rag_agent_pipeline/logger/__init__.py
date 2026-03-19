"""Centralized logging configuration — stdout only.

Usage in any module:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("message")
"""

import logging
import sys

from config.server import LOG_LEVEL


def _configure_root() -> None:
    """Set up the root logger once (idempotent)."""
    root = logging.getLogger("rag")
    if root.handlers:
        return  # already configured

    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(root.level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


_configure_root()


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'rag' namespace."""
    return logging.getLogger(f"rag.{name}")
