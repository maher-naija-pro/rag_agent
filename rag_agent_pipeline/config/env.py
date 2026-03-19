"""Load environment variables from .env — must be imported first."""

from pathlib import Path

from dotenv import load_dotenv

# Resolve .env relative to the project root (one level above config/)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)
