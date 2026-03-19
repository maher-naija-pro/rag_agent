"""Tests for main.py entry point — real argparse, no uvicorn launch."""

import sys


class TestMain:
    def test_main_function_exists(self):
        from main import main
        assert callable(main)

    def test_argparse_defaults(self):
        """Verify argument parser uses config defaults."""
        from main import main
        import argparse

        # Capture the parser by monkey-patching parse_args
        captured = {}
        original_run = None

        import uvicorn
        original_run = uvicorn.run

        def fake_run(app, **kwargs):
            captured.update(kwargs)

        uvicorn.run = fake_run
        try:
            # Simulate: python main.py (no args)
            old_argv = sys.argv
            sys.argv = ["main.py"]
            main()
            sys.argv = old_argv

            assert "host" in captured
            assert "port" in captured
            assert isinstance(captured["port"], int)
            assert captured["port"] > 0
        finally:
            uvicorn.run = original_run

    def test_argparse_custom_port(self):
        """Verify --port flag is parsed."""
        from main import main
        import uvicorn

        captured = {}
        original_run = uvicorn.run

        def fake_run(app, **kwargs):
            captured.update(kwargs)

        uvicorn.run = fake_run
        try:
            old_argv = sys.argv
            sys.argv = ["main.py", "--port", "9999"]
            main()
            sys.argv = old_argv

            assert captured["port"] == 9999
        finally:
            uvicorn.run = original_run

    def test_argparse_reload_flag(self):
        """Verify --reload flag is parsed."""
        from main import main
        import uvicorn

        captured = {}
        original_run = uvicorn.run

        def fake_run(app, **kwargs):
            captured.update(kwargs)

        uvicorn.run = fake_run
        try:
            old_argv = sys.argv
            sys.argv = ["main.py", "--reload"]
            main()
            sys.argv = old_argv

            assert captured["reload"] is True
        finally:
            uvicorn.run = original_run
