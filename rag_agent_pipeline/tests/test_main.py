"""Tests for main.py entry point — real argparse, no uvicorn launch."""

import sys


class TestMain:
    def test_main_function_exists(self):
        from main import main
        assert callable(main)

    def test_argparse_defaults(self, monkeypatch):
        """Verify argument parser uses config defaults."""
        from main import main
        import uvicorn

        captured = {}
        monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: captured.update(kwargs))
        monkeypatch.setattr(sys, "argv", ["main.py"])

        main()

        assert "host" in captured
        assert "port" in captured
        assert isinstance(captured["port"], int)
        assert captured["port"] > 0

    def test_argparse_custom_port(self, monkeypatch):
        """Verify --port flag is parsed."""
        from main import main
        import uvicorn

        captured = {}
        monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: captured.update(kwargs))
        monkeypatch.setattr(sys, "argv", ["main.py", "--port", "9999"])

        main()

        assert captured["port"] == 9999

    def test_argparse_reload_flag(self, monkeypatch):
        """Verify --reload flag is parsed."""
        from main import main
        import uvicorn

        captured = {}
        monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: captured.update(kwargs))
        monkeypatch.setattr(sys, "argv", ["main.py", "--reload"])

        main()

        assert captured["reload"] is True
