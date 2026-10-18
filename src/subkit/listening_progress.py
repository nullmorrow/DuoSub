"""Small, best-effort progress side channel for the isolated inference child."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time


def write_progress(output: Path, text: str) -> None:
    """Replace a whole message atomically; progress must never break inference."""
    temporary = output / "progress.json.tmp"
    try:
        temporary.write_text(json.dumps({"text": str(text)[:500], "updated_ns": time.time_ns()},
                                        ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, output / "progress.json")
    except OSError:
        pass


def read_progress(output: Path, *, newer_than_ns: int) -> str | None:
    """Ignore incomplete, invalid or older messages, including completed stages."""
    try:
        with (output / "progress.json").open("rb") as stream:
            data = json.loads(stream.read(4097))
        text, updated = data.get("text"), data.get("updated_ns")
        if (isinstance(text, str) and 0 < len(text) <= 500
                and isinstance(updated, int) and not isinstance(updated, bool)
                and updated > newer_than_ns):
            return text
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None
