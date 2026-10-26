"""Shared locations for checkout assets and local runtime data."""

import os
from pathlib import Path


def _root():
    configured = os.environ.get("T1000_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "apps" / "api" / "pyproject.toml").is_file():
            return parent
    # Installed distributions use the working directory unless explicitly set.
    return Path.cwd()


ROOT = _root()
CACHE = ROOT / ".cache"
MODELS = ROOT / "data" / "models"
OUTPUT = ROOT / "data" / "output"
ASSETS = ROOT / "assets"
WEB = ROOT / "apps" / "web" / "dist"
