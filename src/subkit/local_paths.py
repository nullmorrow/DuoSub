"""Optional machine paths, kept outside version control (never credentials)."""
from __future__ import annotations

import json
import os
from pathlib import Path


def local_value(name: str, default: str | Path = "") -> str:
    """Environment wins; then local JSON; then the portable default.

    The optional file lives next to the launcher. It is read in workers too,
    without modifying environment variables or importing optional AI packages.
    Missing or malformed local settings do not prevent the editor starting.
    """
    if name in os.environ:
        return os.environ[name]
    filename = Path(__file__).resolve().parents[2] / "duosub.local.json"
    try:
        settings = json.loads(filename.read_text(encoding="utf-8"))
        value = settings.get(name) if isinstance(settings, dict) else None
        if isinstance(value, str) and "\0" not in value:
            return value
    except (OSError, ValueError):
        pass
    return str(default)
