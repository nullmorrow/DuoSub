"""Queued, windowless libmpv probe rendering; one owner for all native calls."""

from __future__ import annotations

import ctypes
from concurrent.futures import Future
from pathlib import Path
import queue
import tempfile
import threading
import time

import numpy as np

from subkit.proc import CancelledError
from .mpv import (Mpv, MpvEventLogMessage, MpvRenderParam, MPV_RENDER_API_TYPE_SW,
                  MPV_RENDER_PARAM_API_TYPE, MPV_RENDER_PARAM_BLOCK_FOR_TARGET_TIME,
                  MPV_RENDER_PARAM_SW_SIZE, MPV_RENDER_PARAM_SW_FORMAT,
                  MPV_RENDER_PARAM_SW_STRIDE, MPV_RENDER_PARAM_SW_POINTER)


class _SoftwareProbe:
    def __init__(self) -> None:
        self.mpv = Mpv({"vo": "libmpv", "ao": "null", "aid": "no", "pause": "yes",
                        "idle": "yes", "config": "no", "terminal": "no", "sub-auto": "no",
                        "sub-ass-override": "no", "osd-level": "0", "keep-open": "yes"})
        self.context = ctypes.c_void_p()
        self.size = None
        self.fonts: set[str] = set()
        self.scratch = None
        try:
            api = ctypes.create_string_buffer(MPV_RENDER_API_TYPE_SW)
            params = (MpvRenderParam * 2)(MpvRenderParam(MPV_RENDER_PARAM_API_TYPE, ctypes.cast(api, ctypes.c_void_p)),
                                          MpvRenderParam(0, None))
            self.mpv._check(self.mpv.lib.mpv_render_context_create(ctypes.byref(self.context), self.mpv.h, params),
                            "create software renderer")
            self.mpv.lib.mpv_request_log_messages(self.mpv.h, b"v")
            self.scratch = tempfile.TemporaryDirectory(prefix="subkit_mpv_probe_")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.context:
            self.mpv.lib.mpv_render_context_free(self.context)
            self.context = ctypes.c_void_p()
        self.mpv.terminate()
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None

    def _draw(self) -> None:
        self.mpv._check(self.mpv.lib.mpv_render_context_render(self.context, self.params), "software render")

    def _wait(self, event_id: int, check) -> None:
        while True:
            check()
            self._draw()  # Service VO requests while waiting; never wait on mpv's render thread.
            found = False
            for event in self.mpv.events():
                found |= event.event_id == event_id
                if event.event_id == 2 and event.data:
                    msg = ctypes.cast(event.data, ctypes.POINTER(MpvEventLogMessage)).contents
                    line = (msg.text or b"").decode("utf-8", "replace").strip()
                    if "fontselect" in line:
                        self.fonts.add(line)
                if event.event_id == 7 and event.error < 0:
                    self.mpv._check(event.error, "probe source")
            if found:
                self._draw()
                return
            time.sleep(0.001)

    def render(self, content: str, width: int, height: int, count: int, check):
        check()
        if self.size != (width, height):
            stride = (width * 4 + 63) // 64 * 64
            storage = np.empty(height * stride + 63, dtype=np.uint8)
            offset = (-storage.ctypes.data) % 64
            self.storage = storage
            self.buffer = storage[offset:offset + height * stride].reshape(height, stride)
            self.pixels = self.buffer[:, :width * 4].reshape(height, width, 4)
            self.dimensions = (ctypes.c_int * 2)(width, height)
            self.stride = ctypes.c_size_t(stride)
            self.format = ctypes.create_string_buffer(b"0bgr")
            self.block = ctypes.c_int(0)
            ptr = lambda value: ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
            self.params = (MpvRenderParam * 6)(
                MpvRenderParam(MPV_RENDER_PARAM_SW_SIZE, ptr(self.dimensions)),
                MpvRenderParam(MPV_RENDER_PARAM_SW_FORMAT, ptr(self.format)),
                MpvRenderParam(MPV_RENDER_PARAM_SW_STRIDE, ptr(self.stride)),
                MpvRenderParam(MPV_RENDER_PARAM_SW_POINTER, self.buffer.ctypes.data),
                MpvRenderParam(MPV_RENDER_PARAM_BLOCK_FOR_TARGET_TIME, ptr(self.block)), MpvRenderParam(0, None))
            list(self.mpv.events())
            self.mpv.command("loadfile", f"av://lavfi:color=c=black:s={width}x{height}:r=10", "replace")
            self._wait(21, check)
            self.size = (width, height)
        sid = self.mpv.get("sid")
        if sid and sid != "no":
            self.mpv.command("sub-remove", sid)
        path = Path(self.scratch.name) / "probe.ass"
        path.write_text(content, encoding="utf-8-sig")
        self.mpv.command("sub-add", str(path), "select")
        frames = []
        for frame in range(count):
            check()
            # lavfi generators cannot rewind; align ASS time to the actual video PTS.
            self.mpv.set("sub-delay", float(self.mpv.get("time-pos")) - (frame + 0.1))
            self._draw()
            # White ink on black: RGB channels coincide; padding is unspecified.
            frames.append(self.pixels[:, :, 1].copy())
        for event in self.mpv.events():
            if event.event_id == 2 and event.data:
                msg = ctypes.cast(event.data, ctypes.POINTER(MpvEventLogMessage)).contents
                line = (msg.text or b"").decode("utf-8", "replace").strip()
                if "fontselect" in line:
                    self.fonts.add(line)
        return np.stack(frames), set(self.fonts)


class ProbeRenderer:
    """Owned by a watch session; callers can cancel while a request is queued."""

    cache_id = "mpv-sw1"

    def __init__(self, *, timeout: float = 3.0, factory=_SoftwareProbe) -> None:
        self.timeout, self.factory = timeout, factory
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.thread = None
        self.failure = None

    def render(self, content, width, height, count, runner):
        runner.check_cancelled()
        future = Future()
        with self._lock:
            if self._stop.is_set():
                raise CancelledError()
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name="subkit-mpv-measurer", daemon=True)
                self.thread.start()
            self._queue.put((future, (content, width, height, count), runner))
        while not future.done():
            runner.check_cancelled()
            if self._stop.wait(0.002):
                raise CancelledError()
        return future.result()

    def _run(self) -> None:
        native = None
        try:
            while not self._stop.is_set():
                item = self._queue.get()
                if item is None:
                    break
                future, args, runner = item
                deadline = time.monotonic() + self.timeout
                def check():
                    runner.check_cancelled()
                    if self._stop.is_set():
                        raise CancelledError()
                    if time.monotonic() >= deadline:
                        raise TimeoutError("libmpv probe timed out")
                try:
                    check()
                    if self.failure is not None:
                        raise RuntimeError(self.failure)
                    if native is None:
                        native = self.factory()
                    future.set_result(native.render(*args, check))
                except Exception as error:
                    future.set_exception(error)
                    if not isinstance(error, CancelledError):
                        self.failure = str(error)
                    if native is not None:
                        native.close()
                        native = None
        finally:
            if native is not None:
                native.close()
            while not self._queue.empty():
                item = self._queue.get_nowait()
                if item is not None:
                    item[0].set_exception(CancelledError())

    def close(self) -> None:
        with self._lock:
            self._stop.set()
            self._queue.put(None)
            thread = self.thread
        if thread is not None:
            thread.join(2.0)
            if thread.is_alive():
                raise TimeoutError("libmpv measuring thread did not stop")
