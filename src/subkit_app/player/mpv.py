"""Minimal ctypes binding for libmpv (client API 2.x) and an embeddable Qt widget.

Only what the editor needs: load a file, observe time/pause/duration/tracks,
seek exactly, step frames, loop a range, select the audio track, and load /
reload the working subtitle so libass renders exactly what will be exported.

Constants come from third_party/mpv/dev/include/mpv/client.h (API 2.5).
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (CFUNCTYPE, POINTER, Structure, c_char_p, c_double, c_int, c_int64, c_uint64, c_ulong, c_void_p)
from pathlib import Path
from typing import Any, Optional

import json

from PySide6.QtCore import QObject, QPointF, Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import QWidget

MPV_FORMAT_NONE, MPV_FORMAT_STRING, MPV_FORMAT_FLAG, MPV_FORMAT_INT64, MPV_FORMAT_DOUBLE = 0, 1, 3, 4, 5
MPV_EVENT_SHUTDOWN, MPV_EVENT_LOG_MESSAGE, MPV_EVENT_END_FILE, MPV_EVENT_FILE_LOADED = 1, 2, 7, 8
MPV_EVENT_SEEK, MPV_EVENT_PLAYBACK_RESTART, MPV_EVENT_PROPERTY_CHANGE = 20, 21, 22


class MpvEvent(Structure):
    _fields_ = [("event_id", c_int), ("error", c_int), ("reply_userdata", c_uint64), ("data", c_void_p)]


class MpvEventProperty(Structure):
    _fields_ = [("name", c_char_p), ("format", c_int), ("data", c_void_p)]


class MpvEventLogMessage(Structure):
    _fields_ = [("prefix", c_char_p), ("level", c_char_p), ("text", c_char_p), ("log_level", c_int)]


class MpvRenderParam(Structure):
    _fields_ = [("type", c_int), ("data", c_void_p)]


MPV_RENDER_API_TYPE_SW = b"sw"
MPV_RENDER_PARAM_API_TYPE = 1
MPV_RENDER_PARAM_BLOCK_FOR_TARGET_TIME = 12
MPV_RENDER_PARAM_SW_SIZE, MPV_RENDER_PARAM_SW_FORMAT = 17, 18
MPV_RENDER_PARAM_SW_STRIDE, MPV_RENDER_PARAM_SW_POINTER = 19, 20


WAKEUP = CFUNCTYPE(None, c_void_p)


def find_libmpv() -> Path:
    env = os.environ.get("SUBKIT_LIBMPV")
    cands = [Path(env)] if env else []
    root = Path(__file__).resolve().parents[3]          # …/SubKit-Opus
    cands += [root / "third_party" / "mpv" / "dev" / "libmpv-2.dll", root / "third_party" / "mpv" / "libmpv-2.dll"]
    if getattr(sys, "frozen", False):
        cands.insert(0, Path(sys.executable).parent / "libmpv-2.dll")
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError("找不到 libmpv-2.dll（应在 third_party/mpv/dev/ 下，见 third_party/mpv/SOURCE.md）")


class _Lib:
    _instance: Optional["_Lib"] = None

    def __init__(self, path: Path) -> None:
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(path.parent))
        lib = ctypes.CDLL(str(path))
        lib.mpv_client_api_version.restype = c_ulong
        ver = lib.mpv_client_api_version()
        if ver >> 16 != 2:
            raise RuntimeError(f"libmpv client API {ver >> 16}.{ver & 0xFFFF} 不受支持（需要 2.x）")
        lib.mpv_create.restype = c_void_p
        lib.mpv_initialize.argtypes = [c_void_p]
        lib.mpv_terminate_destroy.argtypes = [c_void_p]
        lib.mpv_set_option_string.argtypes = [c_void_p, c_char_p, c_char_p]
        lib.mpv_set_property_string.argtypes = [c_void_p, c_char_p, c_char_p]
        lib.mpv_get_property_string.argtypes = [c_void_p, c_char_p]
        lib.mpv_get_property_string.restype = c_void_p
        lib.mpv_free.argtypes = [c_void_p]
        lib.mpv_command.argtypes = [c_void_p, POINTER(c_char_p)]
        lib.mpv_command_async.argtypes = [c_void_p, c_uint64, POINTER(c_char_p)]
        lib.mpv_observe_property.argtypes = [c_void_p, c_uint64, c_char_p, c_int]
        lib.mpv_wait_event.argtypes = [c_void_p, c_double]
        lib.mpv_wait_event.restype = POINTER(MpvEvent)
        lib.mpv_set_wakeup_callback.argtypes = [c_void_p, WAKEUP, c_void_p]
        lib.mpv_error_string.argtypes = [c_int]
        lib.mpv_error_string.restype = c_char_p
        lib.mpv_request_log_messages.argtypes = [c_void_p, c_char_p]
        lib.mpv_render_context_create.argtypes = [POINTER(c_void_p), c_void_p, POINTER(MpvRenderParam)]
        lib.mpv_render_context_create.restype = c_int
        lib.mpv_render_context_render.argtypes = [c_void_p, POINTER(MpvRenderParam)]
        lib.mpv_render_context_render.restype = c_int
        lib.mpv_render_context_free.argtypes = [c_void_p]
        lib.mpv_render_context_free.restype = None
        self.lib = lib
        self.version = (ver >> 16, ver & 0xFFFF)
        self.path = path

    @classmethod
    def get(cls) -> "_Lib":
        if cls._instance is None:
            cls._instance = cls(find_libmpv())
        return cls._instance


class Mpv:
    """One libmpv player instance (thread-safe calls; events drained on demand)."""

    def __init__(self, options: dict[str, str]) -> None:
        self._l = _Lib.get()
        self.lib = self._l.lib
        self.h = self.lib.mpv_create()
        if not self.h:
            raise RuntimeError("mpv_create failed")
        try:
            for k, v in options.items():
                self._check(self.lib.mpv_set_option_string(self.h, k.encode(), str(v).encode()), f"option {k}")
            self._check(self.lib.mpv_initialize(self.h), "initialize")
        except BaseException:
            self.terminate()
            raise

    def _check(self, rc: int, what: str) -> None:
        if rc < 0:
            raise RuntimeError(f"mpv {what}: {self.lib.mpv_error_string(rc).decode()}")

    def command(self, *args: Any) -> None:
        arr = (c_char_p * (len(args) + 1))(*[str(a).encode("utf-8") for a in args], None)
        rc = self.lib.mpv_command(self.h, arr)
        if rc < 0:
            raise RuntimeError(f"mpv command {args[0]}: {self.lib.mpv_error_string(rc).decode()}")

    def command_async(self, *args: Any) -> None:
        arr = (c_char_p * (len(args) + 1))(*[str(a).encode("utf-8") for a in args], None)
        self.lib.mpv_command_async(self.h, 0, arr)

    def set(self, name: str, value: Any) -> None:
        if isinstance(value, bool):
            value = "yes" if value else "no"
        rc = self.lib.mpv_set_property_string(self.h, name.encode(), str(value).encode("utf-8"))
        if rc < 0:
            raise RuntimeError(f"mpv set {name}: {self.lib.mpv_error_string(rc).decode()}")

    def get(self, name: str) -> Optional[str]:
        p = self.lib.mpv_get_property_string(self.h, name.encode())
        if not p:
            return None
        try:
            return ctypes.cast(p, c_char_p).value.decode("utf-8", "replace")
        finally:
            self.lib.mpv_free(p)

    def observe(self, name: str, fmt: int, userdata: int = 0) -> None:
        self._check(self.lib.mpv_observe_property(self.h, userdata, name.encode(), fmt), f"observe {name}")

    def events(self):
        while True:
            ev = self.lib.mpv_wait_event(self.h, 0).contents
            if ev.event_id == 0:
                return
            yield ev

    def terminate(self) -> None:
        if self.h:
            self.lib.mpv_terminate_destroy(self.h)
            self.h = None


def prop_value(ev: MpvEvent) -> tuple[str, Any]:
    prop = ctypes.cast(ev.data, POINTER(MpvEventProperty)).contents
    name = prop.name.decode()
    if not prop.data:
        return name, None
    if prop.format == MPV_FORMAT_DOUBLE:
        return name, ctypes.cast(prop.data, POINTER(c_double)).contents.value
    if prop.format == MPV_FORMAT_FLAG:
        return name, bool(ctypes.cast(prop.data, POINTER(c_int)).contents.value)
    if prop.format == MPV_FORMAT_INT64:
        return name, ctypes.cast(prop.data, POINTER(c_int64)).contents.value
    if prop.format == MPV_FORMAT_STRING:
        return name, ctypes.cast(ctypes.cast(prop.data, POINTER(c_char_p)).contents, c_char_p).value.decode("utf-8", "replace")
    return name, None


MSG_OVERLAY = 60


class _Waker(QObject):
    wake = Signal()


class MpvWidget(QWidget):
    """A native child window that libmpv renders into.

    Mouse input: mpv's embedded window is transparent to mouse hit-testing, so clicks on the video arrive here as
    ordinary Qt events (and keyboard focus stays in SubKit).  They are re-emitted in **OSD pixels** — the
    coordinate space of ``osd-dimensions`` and of ``osd-overlay`` drawings."""

    timeChanged = Signal(float)
    pauseChanged = Signal(bool)
    durationChanged = Signal(float)
    fileLoaded = Signal()
    tracksChanged = Signal(str)
    seeked = Signal()
    logMessage = Signal(str)
    osdChanged = Signal(object)                       # osd-dimensions dict (w, h, ml, mt, mr, mb)
    mousePressed = Signal(QPointF, object, object)    # OSD px, Qt.MouseButton, Qt.KeyboardModifiers
    mouseMoved = Signal(QPointF, object, object)      # OSD px, buttons, modifiers
    mouseReleased = Signal(QPointF, object, object)
    mouseDoubleClicked = Signal(QPointF, object, object)
    wheelTurned = Signal(QPointF, int, object)        # OSD px, angle delta, modifiers
    mouseLeft = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setStyleSheet("background: black;")
        self.setMinimumSize(320, 180)
        self.mpv: Optional[Mpv] = None
        self.time = 0.0
        self.duration = 0.0
        self.paused = True
        self._waker = _Waker()
        self._waker.wake.connect(self._drain, Qt.ConnectionType.QueuedConnection)
        self._cb = WAKEUP(lambda _d: self._waker.wake.emit())
        self._sub_path: Optional[str] = None
        self.osd: dict = {}
        self._overlays: set[int] = set()
        self._msg_timer = QTimer(self)
        self._msg_timer.setSingleShot(True)
        self._msg_timer.timeout.connect(lambda: self.clear_overlay(MSG_OVERLAY))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._autohide = False                      # 观剧模式: hide the mouse pointer when it does not move
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(1800)
        self._hide_timer.timeout.connect(self._hide_cursor)

    def ensure(self) -> Mpv:
        if self.mpv is None:
            self._speed = 1.0
            wid = int(self.winId())
            self.mpv = Mpv({
                "wid": str(wid), "vo": "gpu", "hwdec": "auto-safe", "keep-open": "yes", "idle": "yes",
                "pause": "yes", "config": "no", "osc": "no", "input-default-bindings": "no",
                "input-vo-keyboard": "no", "input-cursor": "no", "cursor-autohide": "no", "terminal": "no",
                "sub-auto": "no", "sid": "no", "hr-seek": "yes", "hr-seek-framedrop": "no",
                "screenshot-format": "png", "keepaspect": "yes", "osd-level": "1", "osd-bar": "no", "osd-on-seek": "no",
                "sub-ass-override": "no", "embeddedfonts": "yes",
            })
            for name, fmt in (("time-pos", MPV_FORMAT_DOUBLE), ("pause", MPV_FORMAT_FLAG),
                              ("duration", MPV_FORMAT_DOUBLE), ("track-list", MPV_FORMAT_STRING),
                              ("osd-dimensions", MPV_FORMAT_STRING)):
                self.mpv.observe(name, fmt)
            self.mpv.lib.mpv_request_log_messages(self.mpv.h, b"warn")
            self.mpv.lib.mpv_set_wakeup_callback(self.mpv.h, self._cb, None)
        return self.mpv

    @Slot()
    def _drain(self) -> None:
        if self.mpv is None:
            return
        for ev in self.mpv.events():
            if ev.event_id == MPV_EVENT_PROPERTY_CHANGE:
                name, value = prop_value(ev)
                if name == "time-pos" and value is not None:
                    self.time = float(value)
                    self.timeChanged.emit(self.time)
                elif name == "pause" and value is not None:
                    self.paused = bool(value)
                    self.pauseChanged.emit(self.paused)
                elif name == "duration" and value is not None:
                    self.duration = float(value)
                    self.durationChanged.emit(self.duration)
                elif name == "track-list" and value is not None:
                    self.tracksChanged.emit(value)
                elif name == "osd-dimensions" and value:
                    try:
                        self.osd = json.loads(value)
                    except ValueError:
                        continue
                    self.osdChanged.emit(self.osd)
            elif ev.event_id == MPV_EVENT_FILE_LOADED:
                self.fileLoaded.emit()
            elif ev.event_id == MPV_EVENT_PLAYBACK_RESTART:
                self.seeked.emit()
            elif ev.event_id == MPV_EVENT_LOG_MESSAGE and ev.data:
                msg = ctypes.cast(ev.data, POINTER(MpvEventLogMessage)).contents
                self.logMessage.emit(f"[{(msg.prefix or b'').decode()}] {(msg.text or b'').decode('utf-8', 'replace').strip()}")
            elif ev.event_id == MPV_EVENT_SHUTDOWN:
                return

    # ------------------------------------------------------------ high level
    def open(self, video: str, *, audio_track: Optional[int] = None) -> None:
        m = self.ensure()
        m.set("pause", True)
        m.command("loadfile", video, "replace")
        if audio_track is not None:
            m.set("aid", str(audio_track))

    def load_subtitle(self, path: str) -> None:
        """(Re)load the working subtitle; keeps it selected so libass renders the latest version."""
        m = self.ensure()
        if self._sub_path == path:
            try:
                m.command("sub-reload")
                return
            except RuntimeError:
                pass
        m.command("sub-add", path, "select", "SubKit", "und")
        self._sub_path = path

    def play(self) -> None:
        self.ensure().set("pause", False)

    def pause(self) -> None:
        self.ensure().set("pause", True)

    def toggle(self) -> None:
        self.ensure().command("cycle", "pause")

    def seek(self, seconds: float, exact: bool = True) -> None:
        self.ensure().command("seek", f"{max(0.0, seconds):.3f}", "absolute+exact" if exact else "absolute+keyframes")

    def frame_step(self, forward: bool = True) -> None:
        self.ensure().command("frame-step" if forward else "frame-back-step")

    def loop(self, a: Optional[float], b: Optional[float]) -> None:
        m = self.ensure()
        m.set("ab-loop-a", "no" if a is None else f"{a:.3f}")
        m.set("ab-loop-b", "no" if b is None else f"{b:.3f}")

    def set_audio_track(self, aid: int) -> None:
        self.ensure().set("aid", str(aid))

    def screenshot(self, path: str, *, with_subs: bool = True) -> None:
        self.ensure().command("screenshot-to-file", path, "subtitles" if with_subs else "video")

    def set_speed(self, speed: float) -> None:
        """Playback speed (mpv keeps the pitch); only talks to mpv when the value changes."""
        if self.mpv is None or getattr(self, "_speed", 1.0) == speed:
            return
        try:
            self.mpv.set("speed", f"{speed:g}")
            self._speed = speed
        except RuntimeError:
            pass

    def volume_step(self, delta: int) -> None:
        """Change the volume (0–150 %) and show it on the video."""
        if self.mpv is None:
            return
        try:
            v = float(self.mpv.get("volume") or 100.0)
            v = max(0.0, min(150.0, v + delta))
            self.mpv.set("volume", f"{v:g}")
            self.message(f"音量 {v:.0f}%", 900)
        except (RuntimeError, ValueError):
            pass

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down) and not e.modifiers():
            self.volume_step(+5 if e.key() == Qt.Key.Key_Up else -5)
            return
        super().keyPressEvent(e)

    def time_now(self) -> float:
        """Current playback position read directly from mpv (more precise than the last time-pos event)."""
        if self.mpv is None:
            return self.time
        try:
            v = self.mpv.get("time-pos")
            return float(v) if v else self.time
        except (RuntimeError, ValueError):
            return self.time

    # ------------------------------------------------------------ OSD overlays (drawn by libass over the video)
    def overlay(self, oid: int, ass_events: str) -> None:
        """Draw ASS events in OSD pixel coordinates (``osd-overlay``); an empty text removes the overlay."""
        if self.mpv is None:
            return
        if not ass_events:
            self.clear_overlay(oid)
            return
        w, h = int(self.osd.get("w") or 0), int(self.osd.get("h") or 0)
        if w <= 0 or h <= 0:
            return
        try:
            self.mpv.command("osd-overlay", str(oid), "ass-events", ass_events, str(w), str(h), str(oid), "no", "no")
            self._overlays.add(oid)
        except RuntimeError:
            pass

    def clear_overlay(self, oid: int) -> None:
        if self.mpv is None or oid not in self._overlays:
            return
        try:
            self.mpv.command("osd-overlay", str(oid), "none", "")
        except RuntimeError:
            pass
        self._overlays.discard(oid)

    def message(self, text: str, ms: int = 1600) -> None:
        """Short on-screen note (top left), e.g. "开始 → 0:12:03.45"."""
        h = int(self.osd.get("h") or self.height() or 720)
        fs = max(18, int(h * 0.028))
        safe = text.replace("{", "(").replace("}", ")").replace("\n", "\\N")
        style = f"{{\\an7\\pos({fs},{fs})\\fs{fs}\\bord2\\shad0\\1c&HFFFFFF&\\3c&H202020&\\fnMicrosoft YaHei}}"
        self.overlay(MSG_OVERLAY, style + safe)
        self._msg_timer.start(ms)

    @property
    def autohide(self) -> bool:
        return self._autohide

    @autohide.setter
    def autohide(self, on: bool) -> None:
        self._autohide = bool(on)
        if on:
            self._hide_timer.start()
        else:
            self._hide_timer.stop()
            self.unsetCursor()

    def _hide_cursor(self) -> None:
        from PySide6.QtWidgets import QApplication

        if (self._autohide and self.underMouse() and QApplication.mouseButtons() == Qt.MouseButton.NoButton
                and QApplication.activePopupWidget() is None):
            self.setCursor(Qt.CursorShape.BlankCursor)

    def _wake_cursor(self) -> None:
        if self.cursor().shape() == Qt.CursorShape.BlankCursor:
            self.unsetCursor()
        if self._autohide:
            self._hide_timer.start()

    def osd_scale(self) -> float:
        w = float(self.osd.get("w") or 0)
        return w / self.width() if w > 0 and self.width() > 0 else self.devicePixelRatioF()

    # ------------------------------------------------------------ mouse (Qt events → OSD pixels)
    def _px(self, e) -> QPointF:
        p = e.position()
        k = self.osd_scale()
        return QPointF(p.x() * k, p.y() * k)

    def mousePressEvent(self, e) -> None:  # noqa: N802
        self._wake_cursor()
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.mousePressed.emit(self._px(e), e.button(), e.modifiers())

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        self._wake_cursor()
        self.mouseMoved.emit(self._px(e), e.buttons(), e.modifiers())

    def leaveEvent(self, e) -> None:  # noqa: N802
        self.mouseLeft.emit()
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        self.mouseReleased.emit(self._px(e), e.button(), e.modifiers())

    def mouseDoubleClickEvent(self, e) -> None:  # noqa: N802
        self.mouseDoubleClicked.emit(self._px(e), e.button(), e.modifiers())

    def wheelEvent(self, e) -> None:  # noqa: N802
        k = self.osd_scale()
        p = e.position()
        self.wheelTurned.emit(QPointF(p.x() * k, p.y() * k), e.angleDelta().y(), e.modifiers())

    def shutdown(self) -> None:
        if self.mpv is not None:
            self.mpv.terminate()
            self.mpv = None

    def closeEvent(self, e) -> None:  # noqa: N802
        self.shutdown()
        super().closeEvent(e)
