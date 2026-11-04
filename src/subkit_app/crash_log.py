"""Local crash and health records, kept on the system drive so they survive a stalled data drive.

``duosub.log`` gets one line per session start and clean end, every uncaught Python error (GUI and worker
threads), Qt critical messages and, once a minute, the process's memory, handle and GUI-object counts with the
longest GUI stall, so a slow decline before a hang is visible afterwards.  ``fault.log`` gets the stacks of
every thread on a native crash, or when the GUI thread stops responding for ``STALL_DUMP_S`` seconds.  A session
whose end line is missing was cut off (crash, hang, power loss); the next start says so in the log.
"""

from __future__ import annotations

import ctypes
import faulthandler
import os
import platform
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QObject, QTimer, QtMsgType, qInstallMessageHandler, qVersion

LOG_NAME = "duosub.log"
FAULT_NAME = "fault.log"
ROTATE_BYTES = 2 * 2**20          # at start-up a larger log moves to <name>.1 (one generation kept)
HEARTBEAT_MS = 1000
HEALTH_MS = 60_000
STALL_DUMP_S = 30                 # GUI thread blocked this long → all thread stacks go to fault.log
STALL_NOTE_S = 5.0                # stalls at least this long get their own line in duosub.log
END_MARK = "session end"


def log_dir() -> Path:
    override = os.environ.get("SUBKIT_LOG_DIR")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "DuoSub" / "logs"


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _rotate(path: Path) -> None:
    try:
        if path.stat().st_size > ROTATE_BYTES:
            os.replace(path, path.with_name(path.name + ".1"))
    except OSError:
        pass


def _last_line(path: Path) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 4096))
            lines = fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return ""
    return next((line for line in reversed(lines) if line.strip()), "")


def process_health() -> dict[str, int]:
    """Private and working-set memory (MB), kernel handles and GDI/USER objects of this process (Windows)."""
    if os.name != "nt":
        return {}
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                "PrivateUsage")]

    try:
        kernel, user, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("user32"), ctypes.WinDLL("psapi")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        kernel.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        user.GetGuiResources.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        process = kernel.GetCurrentProcess()
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        handles = wintypes.DWORD()
        if not (psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
                and kernel.GetProcessHandleCount(process, ctypes.byref(handles))):
            return {}
        return {"private_mb": counters.PrivateUsage // 2**20, "working_mb": counters.WorkingSetSize // 2**20,
                "handles": handles.value, "gdi": user.GetGuiResources(process, 0),
                "user": user.GetGuiResources(process, 1)}
    except (AttributeError, OSError):
        return {}


class CrashLog(QObject):
    def __init__(self, parent: QObject, folder: Optional[Path] = None, *, version: str = "",
                 clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(parent)
        self.folder = Path(folder) if folder is not None else log_dir()
        self.clock = clock
        self._lock = threading.Lock()
        self._log = self._fault = None
        self._previous_fault = faulthandler.is_enabled()
        self._previous_thread_hook = threading.excepthook
        self._previous_qt_handler = None
        self._last_beat: Optional[float] = None       # building the main window before the first beat is no stall
        self._max_stall = 0.0
        self._open(version)
        self.heartbeat_timer = QTimer(self)
        self.heartbeat_timer.timeout.connect(self.heartbeat)
        self.heartbeat_timer.start(HEARTBEAT_MS)
        self.health_timer = QTimer(self)
        self.health_timer.timeout.connect(self.health)
        self.health_timer.start(HEALTH_MS)

    @property
    def log_path(self) -> Path:
        return self.folder / LOG_NAME

    @property
    def fault_path(self) -> Path:
        return self.folder / FAULT_NAME

    def _open(self, version: str) -> None:
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            for path in (self.log_path, self.fault_path):
                _rotate(path)
            previous = _last_line(self.log_path)
            self._log = open(self.log_path, "a", encoding="utf-8", newline="\n")
            self._fault = open(self.fault_path, "a", encoding="utf-8", newline="\n")
        except OSError:
            self._log = self._fault = None          # logging must never stop the program from starting
            return
        if previous and not previous.endswith(END_MARK):
            self.write("previous session did not end cleanly; its last record: " + previous[:200])
        self.write(f"session start: DuoSub {version} pid {os.getpid()} Python {platform.python_version()} "
                   f"Qt {qVersion()} {platform.platform()}")
        self._fault.write(f"\n--- {_stamp()} session start pid {os.getpid()}\n")
        self._fault.flush()
        faulthandler.enable(self._fault, all_threads=True)
        faulthandler.dump_traceback_later(STALL_DUMP_S, exit=False, file=self._fault)
        threading.excepthook = self._thread_exception
        self._previous_qt_handler = qInstallMessageHandler(self._qt_message)

    def write(self, text: str) -> None:
        with self._lock:
            if self._log is None:
                return
            try:
                self._log.write(f"{_stamp()} {text.rstrip()}\n")
                self._log.flush()
            except (OSError, ValueError):
                pass

    def exception(self, exc_type, exc, tb, where: str = "GUI thread") -> None:
        self.write(f"uncaught error ({where}):\n" + "".join(traceback.format_exception(exc_type, exc, tb)))

    def _thread_exception(self, args) -> None:
        name = args.thread.name if args.thread is not None else "?"
        self.exception(args.exc_type, args.exc_value, args.exc_traceback, f"thread {name}")
        self._previous_thread_hook(args)

    def _qt_message(self, kind, context, message) -> None:
        if kind in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
            self.write(f"Qt {'fatal' if kind == QtMsgType.QtFatalMsg else 'critical'}: {message}")
        if self._previous_qt_handler is not None:
            self._previous_qt_handler(kind, context, message)

    def heartbeat(self) -> None:
        """Runs on the GUI thread: a late beat measures how long the event loop was blocked, and re-arms the
        faulthandler watchdog that dumps every stack if the next beat is ``STALL_DUMP_S`` seconds late."""
        now = self.clock()
        stall = 0.0 if self._last_beat is None else max(0.0, now - self._last_beat - HEARTBEAT_MS / 1000)
        self._last_beat = now
        self._max_stall = max(self._max_stall, stall)
        if stall >= STALL_NOTE_S:
            dumped = "; stacks in fault.log" if stall >= STALL_DUMP_S else ""
            self.write(f"GUI thread did not respond for {stall:.1f} s{dumped}")
        if self._fault is not None:
            faulthandler.dump_traceback_later(STALL_DUMP_S, exit=False, file=self._fault)

    def health(self) -> None:
        values = process_health()
        parts = [f"{name} {value}" for name, value in values.items()]
        parts.append(f"threads {threading.active_count()}")
        parts.append(f"longest stall {int(self._max_stall * 1000)} ms")
        self._max_stall = 0.0
        self.write("health: " + ", ".join(parts))

    def close(self) -> None:
        if self._log is None:
            return
        self.heartbeat_timer.stop()
        self.health_timer.stop()
        faulthandler.cancel_dump_traceback_later()
        faulthandler.disable()
        if self._previous_fault and sys.stderr is not None:
            faulthandler.enable(sys.stderr)
        threading.excepthook = self._previous_thread_hook
        qInstallMessageHandler(self._previous_qt_handler)
        self.write(END_MARK)
        with self._lock:
            for fh in (self._log, self._fault):
                try:
                    fh.close()
                except OSError:
                    pass
            self._log = self._fault = None


def install(app, version: str = "") -> CrashLog:
    """One log per application, parented to it and closed when the application quits."""
    log = getattr(app, "_crash_log", None)
    if log is None:
        log = CrashLog(app, version=version)
        app._crash_log = log
        app.aboutToQuit.connect(log.close)
    return log
