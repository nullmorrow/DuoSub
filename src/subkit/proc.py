"""Running external tools (ffmpeg, ffprobe) without console windows.

SubKit runs as a GUI program (pythonw).  On Windows every console program it starts would get its own console
window, flashing up on screen — the user saw that constantly while watching, because the layout measurement
starts ffmpeg whenever new lines come on screen.  Every call goes through :func:`run`, which

* passes ``CREATE_NO_WINDOW`` and a hidden ``STARTUPINFO``;
* resolves Chocolatey's shim executables (``…\\chocolatey\\bin\\ffmpeg.exe``) to the real binaries — a shim starts
  the real program as a second process, which may open a window of its own (and costs a process start).

``SUBKIT_FFMPEG`` / ``SUBKIT_FFPROBE`` override the paths.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

TOOLS = ("ffmpeg", "ffprobe")


@lru_cache(maxsize=None)
def tool(name: str) -> str:
    """Full path of a tool (the real binary behind a Chocolatey shim), or the bare name if not found."""
    env = os.environ.get(f"SUBKIT_{name.upper()}")
    if env:
        return env
    found = shutil.which(name)
    if not found:
        return name
    p = Path(found)
    if "chocolatey" in str(p).lower() and p.parent.name.lower() == "bin":
        lib = p.parent.parent / "lib"
        for cand in sorted(lib.glob(f"*/tools/**/{name}.exe")):
            if cand.is_file():
                return str(cand)
    return found


def hidden() -> dict[str, Any]:
    """Keyword arguments that keep a child process's console window hidden (Windows)."""
    if sys.platform != "win32":
        return {}
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0          # SW_HIDE
    return {"creationflags": subprocess.CREATE_NO_WINDOW, "startupinfo": si}


def run(cmd: Sequence[str], **kw: Any) -> subprocess.CompletedProcess:
    """``subprocess.run`` without a console window; ``ffmpeg`` / ``ffprobe`` resolved to their real binaries."""
    cmd = [str(c) for c in cmd]
    if cmd and cmd[0] in TOOLS:
        cmd[0] = tool(cmd[0])
    for k, v in hidden().items():
        kw.setdefault(k, v)
    return subprocess.run(cmd, **kw)


class CancelledError(Exception):
    """An external-tool operation was cancelled by its owner."""


class KillableRun:
    """One sequential tool pipeline; cancel also covers the process-start race."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._process = None

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            if self._process is not None and self._process.poll() is None:
                self._process.kill()

    def check_cancelled(self) -> None:
        with self._lock:
            if self._cancelled:
                raise CancelledError()

    def run(self, cmd: Sequence[str], *, cwd: str | None = None, timeout: float | None = None,
            text: bool = False, check: bool = False, capture_output: bool = True) -> subprocess.CompletedProcess:
        cmd = [str(c) for c in cmd]
        if cmd and cmd[0] in TOOLS:
            cmd[0] = tool(cmd[0])
        with self._lock:
            if self._cancelled:
                raise CancelledError()
            process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=text, **hidden())
            self._process = process
        try:
            with process:
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                except BaseException:
                    process.kill()
                    process.communicate()
                    raise
            with self._lock:
                if self._cancelled:
                    raise CancelledError()
            result = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
            if check:
                result.check_returncode()
            return result
        finally:
            with self._lock:
                self._process = None
