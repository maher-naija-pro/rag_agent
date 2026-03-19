"""Tests for logger module — real logging, no mocks."""

import logging


class TestLogger:
    def test_get_logger_returns_logger(self):
        from logger import get_logger
        log = get_logger("test_module")
        assert isinstance(log, logging.Logger)

    def test_logger_name_prefix(self):
        from logger import get_logger
        log = get_logger("mynode")
        assert log.name == "rag.mynode"

    def test_root_logger_has_handler(self):
        root = logging.getLogger("rag")
        assert len(root.handlers) >= 1

    def test_configure_root_is_idempotent(self):
        """Calling _configure_root twice doesn't add duplicate handlers."""
        from logger import _configure_root
        root = logging.getLogger("rag")
        count_before = len(root.handlers)
        _configure_root()
        assert len(root.handlers) == count_before
