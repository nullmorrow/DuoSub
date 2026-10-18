"""Interacting with the subtitles on the video (观剧模式, PotPlayer-style).

* left click a subtitle      → select its whole Group (both languages, ruby) and pause;
* left drag a subtitle        → move the Group (Alt: only that line; Shift: one axis);
* right click a character     → its reading (ruby candidates / custom input);
* right click empty space     → deselect and continue playing;
* double click a subtitle     → edit its text in place;  double click empty space → toggle 观剧模式.

Works on the small video in edit mode too.  Geometry comes from :mod:`subkit.core.layout_index` (libass
measurements of the exported preview), computed in a worker thread whenever other lines come on screen.
Frames are drawn with mpv's ``osd-overlay`` — the same renderer as the subtitles.
"""

from __future__ import annotations

from subkit.core.editor_text import editing_text

import bisect
from copy import deepcopy
from pathlib import Path
import threading
import time
from typing import Any, Optional

from PySide6.QtCore import QObject, QPoint, QPointF, QRect, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QFont, QFontMetricsF, QKeyEvent
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QLineEdit, QVBoxLayout, QPushButton

from subkit.languages import JAPANESE_ONLY, lane_label
from subkit.core.layout_index import LayoutIndex, LineGeom, Rect
from subkit.textutil import is_kanji
from subkit.proc import CancelledError, KillableRun

from .player.mpv import MpvWidget
from .transform_tool import Drag, TransformMember, TransformTool

OV_GHOST, OV_SELECT, OV_HOVER, OV_DRAG, OV_TRANSFORM, OV_BAR = 5, 10, 11, 12, 13, 20
OV_SPEAKER = 14
SELECT_COLOR = "&HFFA63E&"      # #3ea6ff (ASS colours are BGR)
DRAG_COLOR = "&H00D7FF&"        # amber
HOVER_COLOR = "&HFFFFFF&"
FRAME_DEFAULT = "#3EA6FF"       # 视图 → 观剧模式选项 → 字幕外框颜色… (SELECT_COLOR as #RRGGBB)


def ass_rgb(hex_rgb: str) -> str:
    """#RRGGBB → the &HBBGGRR& form used by override tags."""
    h = hex_rgb.lstrip("#").upper()
    return f"&H{h[4:6]}{h[2:4]}{h[0:2]}&"


class VideoMapper:
    """OSD pixels ↔ script coordinates (subtitles are drawn inside the video rectangle)."""

    def __init__(self, osd: dict, playres: tuple[int, int]) -> None:
        self.w = float(osd.get("w") or 0)
        self.h = float(osd.get("h") or 0)
        self.ml, self.mt = float(osd.get("ml") or 0), float(osd.get("mt") or 0)
        mr, mb = float(osd.get("mr") or 0), float(osd.get("mb") or 0)
        self.vw, self.vh = self.w - self.ml - mr, self.h - self.mt - mb
        self.px, self.py = playres
        self.ok = self.vw > 1 and self.vh > 1 and self.px > 0 and self.py > 0
        self.sx = self.vw / self.px if self.ok else 1.0
        self.sy = self.vh / self.py if self.ok else 1.0

    def to_script(self, p: QPointF) -> tuple[float, float]:
        return (p.x() - self.ml) / self.sx, (p.y() - self.mt) / self.sy

    def to_osd(self, x: float, y: float) -> tuple[float, float]:
        return self.ml + x * self.sx, self.mt + y * self.sy

    def rect_to_osd(self, r: Rect) -> tuple[float, float, float, float]:
        x0, y0 = self.to_osd(r.x0, r.y0)
        x1, y1 = self.to_osd(r.x1, r.y1)
        return x0, y0, x1, y1


def box_event(x0: float, y0: float, x1: float, y1: float, *, color: str, width: float = 2.0, fill_alpha: str = "&HFF&") -> str:
    """One ASS overlay event: a rectangle outline (and an optional faint fill) in OSD pixels."""
    pad = width + 2
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    return (f"{{\\an7\\pos(0,0)\\bord{width:g}\\shad0\\blur0.6\\1c{color}\\1a{fill_alpha}\\3c{color}\\p1}}"
            f"m {x0:.1f} {y0:.1f} l {x1:.1f} {y0:.1f} l {x1:.1f} {y1:.1f} l {x0:.1f} {y1:.1f}{{\\p0}}")


def quad_event(points, *, color: str, width: float = 2.0, fill_alpha: str = "&HFF&") -> str:
    path = " l ".join(f"{x:.1f} {y:.1f}" for x, y in points)
    return (f"{{\\an7\\pos(0,0)\\bord{width:g}\\shad0\\1c{color}\\1a{fill_alpha}\\3c{color}\\p1}}"
            f"m {path}{{\\p0}}")


def _ass_colour(v: str) -> tuple[str, str]:
    """Style colour &HAABBGGRR → override colour &HBBGGRR& and alpha &HAA&."""
    h = (v or "").strip().upper().replace("&H", "").replace("&", "")
    h = h.rjust(8, "0")[-8:]
    return f"&H{h[2:]}&", f"&H{h[:2]}&"


def style_override(st, sx: float, sy: float) -> str:
    """A style written out as override tags, sizes converted to OSD pixels (the OSD overlay has no styles)."""
    c1, a1 = _ass_colour(st.get("PrimaryColour"))
    c2, a2 = _ass_colour(st.get("SecondaryColour"))
    c3, a3 = _ass_colour(st.get("OutlineColour"))
    c4, a4 = _ass_colour(st.get("BackColour"))
    return (f"\\fn{st.fontname}\\fs{st.fontsize * sy:.2f}\\b{1 if st.bold else 0}\\i{1 if st.italic else 0}"
            f"\\u{int(st.num('Underline') != 0)}\\s{int(st.num('StrikeOut') != 0)}"
            f"\\fscx{st.scale_x:g}\\fscy{st.scale_y:g}\\fsp{st.spacing * sx:.2f}\\frz{st.num('Angle'):g}"
            f"\\1c{c1}\\1a{a1}\\2c{c2}\\2a{a2}\\3c{c3}\\3a{a3}\\4c{c4}\\4a{a4}"
            f"\\bord{st.outline * sy:.2f}\\shad{st.shadow * sy:.2f}")


_STATIC_DROP = {"pos", "move", "an", "a", "org", "fad", "fade", "t", "k", "K", "kf", "ko", "clip", "iclip"}


def _ass_circle(cx: float, cy: float, r: float) -> str:
    k = 0.5523 * r
    return (f"m {cx + r:.1f} {cy:.1f} b {cx + r:.1f} {cy - k:.1f} {cx + k:.1f} {cy - r:.1f} {cx:.1f} {cy - r:.1f} "
            f"b {cx - k:.1f} {cy - r:.1f} {cx - r:.1f} {cy - k:.1f} {cx - r:.1f} {cy:.1f} "
            f"b {cx - r:.1f} {cy + k:.1f} {cx - k:.1f} {cy + r:.1f} {cx:.1f} {cy + r:.1f} "
            f"b {cx + k:.1f} {cy + r:.1f} {cx + r:.1f} {cy + k:.1f} {cx + r:.1f} {cy:.1f}")


def _clock(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}"


class SeekBar:
    """A thin progress line along the bottom edge of the video (mpv OSD).  It appears only when the mouse touches
    the very bottom edge (bilingual subtitles sit right above it, so it must not cover them) and stays while the
    mouse is near it; click or drag on it to seek (fast seeks while dragging, an exact one on release).  A press
    on a subtitle always belongs to the subtitle."""

    def __init__(self, player: MpvWidget) -> None:
        self.player = player
        self.enabled = False                # only in 全屏观看 (F11); elsewhere the strip under the video is used
        self.visible = False
        self.dragging = False
        self.hover_x: Optional[float] = None
        self._last_seek = 0.0
        self._hide = QTimer()
        self._hide.setSingleShot(True)
        self._hide.setInterval(500)
        self._hide.timeout.connect(self.hide)

    def _geom(self) -> tuple[float, float, float, float, float]:
        osd = self.player.osd
        w, h = float(osd.get("w") or 0), float(osd.get("h") or 0)
        zone = max(12.0, h * 0.012)         # the trigger: the bottom edge only
        pad = max(16.0, w * 0.015)
        return w, h, zone, pad, w - pad

    def in_zone(self, p: QPointF, *, approach: bool = False) -> bool:
        """On the bottom edge (or, with ``approach`` while shown, close enough to keep it)."""
        w, h, zone, _x0, _x1 = self._geom()
        return h > 0 and p.y() >= h - zone * (2.5 if approach else 1.0)

    def time_at(self, x: float) -> float:
        _w, _h, _zone, x0, x1 = self._geom()
        frac = min(1.0, max(0.0, (x - x0) / max(1.0, x1 - x0)))
        return frac * (self.player.duration or 0.0)

    def mouse_move(self, p: QPointF, *, on_subtitle: bool = False) -> None:
        if not self.enabled:
            return
        if self.dragging:
            self._seek(p.x(), exact=False)
            return
        if (self.in_zone(p) or (self.visible and self.in_zone(p, approach=True))) and not on_subtitle:
            self.visible = True
            self.hover_x = p.x()
            self._hide.stop()
            self.draw()
        elif self.visible:
            self.hover_x = None
            self._hide.start()

    def on_track(self, p: QPointF) -> bool:
        _w, h, zone, x0, x1 = self._geom()
        return x0 - 8 <= p.x() <= x1 + 8 and p.y() >= h - zone * 2.5

    def press(self, p: QPointF, *, on_subtitle: bool = False) -> bool:
        """True if the press belongs to the bar (the caller must not treat it as a click on the subtitles)."""
        if on_subtitle or not self.visible or not self.player.duration or not self.on_track(p):
            return False
        self.dragging = True
        self._seek(p.x(), exact=False)
        return True

    def release(self, p: QPointF) -> bool:
        if not self.dragging:
            return False
        self.dragging = False
        self._seek(p.x(), exact=True)
        return True

    def _seek(self, x: float, *, exact: bool) -> None:
        import time as _time

        now = _time.monotonic()
        if not exact and now - self._last_seek < 0.07:
            self.hover_x = x
            self.draw(at=self.time_at(x))
            return
        self._last_seek = now
        t = self.time_at(x)
        self.player.seek(t, exact=exact)
        self.hover_x = x
        self.draw(at=t)

    def tick(self) -> None:
        """Playback moved: redraw at most ~6 times a second while the bar is shown."""
        import time as _time

        if self.visible and not self.dragging and _time.monotonic() - getattr(self, '_last_draw', 0.0) > 0.16:
            self.draw()

    def hide(self) -> None:
        if self.dragging:
            return
        self.visible = False
        self.hover_x = None
        self.player.clear_overlay(OV_BAR)

    def draw(self, at: Optional[float] = None) -> None:
        if not self.visible or not self.enabled:
            return
        import time as _time

        self._last_draw = _time.monotonic()
        w, h, zone, x0, x1 = self._geom()
        dur = self.player.duration or 0.0
        if w <= 0 or dur <= 0:
            return
        t = self.player.time if at is None else at
        xp = x0 + (x1 - x0) * min(1.0, max(0.0, t / dur))
        th = max(3.0, zone * 0.3)                    # a thin line on the bottom edge, no backdrop over the text
        ym = h - th * 1.4
        fs = max(13, int(zone * 1.15))
        line = "{{\\an7\\pos(0,0)\\bord0\\shad0" + "{0}\\p1}}m {1:.1f} {2:.1f} l {3:.1f} {2:.1f} l {3:.1f} {4:.1f} l {1:.1f} {4:.1f}{{\\p0}}"
        ev = [line.format("\\1c&H000000&\\1a&H60&", x0 - 1, ym - th / 2 - 1, x1 + 1, ym + th / 2 + 1),
              line.format("\\1c&HB0B0B0&\\1a&H30&", x0, ym - th / 2, x1, ym + th / 2),
              line.format("\\1c&HFFA63E&", x0, ym - th / 2, xp, ym + th / 2),
              f"{{\\an7\\pos(0,0)\\bord1\\shad0\\1c&HFFFFFF&\\3c&HFFA63E&\\p1}}{_ass_circle(xp, ym, th * 1.3)}{{\\p0}}",
              f"{{\\an3\\pos({x1:.1f},{ym - th - 2:.1f})\\fs{fs}\\bord2\\shad0\\1c&HFFFFFF&\\3c&H202020&\\fnMicrosoft YaHei}}"
              f"{_clock(t)} / {_clock(dur)}"]
        if self.hover_x is not None and not self.dragging:
            hx = min(x1, max(x0, self.hover_x))
            ev.append(f"{{\\an2\\pos({hx:.1f},{ym - th - 2:.1f})\\fs{fs}\\bord2\\shad0\\1c&HFFFFFF&\\3c&H202020&\\fnMicrosoft YaHei}}"
                      f"{_clock(self.time_at(hx))}")
        self.player.overlay(OV_BAR, "\n".join(ev))


class _GeoSignals(QObject):
    coarse = Signal(int, object, object)      # publish before the precise probes
    done = Signal(int, object, object)        # preview serial, {index: LineGeom}, {index: anchor}


class GeometryJob(QRunnable):
    def __init__(self, serial: int, ass, owners, indices: list[int], cache_path: Optional[Path], video_height=None) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.serial, self.ass, self.owners, self.indices, self.cache_path = serial, ass, owners, indices, cache_path
        self.target_serial = serial
        self.reusable_indices = set(indices)
        self.signals = _GeoSignals()
        self.video_height = video_height
        self.runner = KillableRun()
        self.cancelled = False
        self.finished = threading.Event()

    def cancel(self) -> None:
        self.cancelled = True
        self.runner.cancel()

    def run(self) -> None:
        try:
            li = LayoutIndex(self.ass, self.owners, measure_cache=self.cache_path, video_height=self.video_height,
                             runner=self.runner)
            try:
                geo = {g.index: g for g in li.geometry(self.indices)}
                bases = [i for i in self.indices if self.owners[i].get("kind") == "base"]
                anchors = {i: li.anchor(i) for i in bases}
                if self.cancelled:
                    return
                self.signals.coarse.emit(self.serial, dict(geo), anchors)
                if self.cancelled:
                    return
                geo.update({g.index: g for g in li.precise_geometry(bases)})
            finally:
                li.close()
        except CancelledError:
            geo, anchors = {}, {}
        except Exception:  # pragma: no cover - geometry is best effort (clicks then do nothing)
            import traceback

            if not self.cancelled:
                traceback.print_exc()
            geo, anchors = {}, {}
        finally:
            try:
                if not self.cancelled:
                    self.signals.done.emit(self.serial, geo, anchors)
            finally:
                self.finished.set()


class _WarmSignals(QObject):
    progress = Signal(int, int)               # measured, total
    done = Signal()


class WarmJob(QRunnable):
    """Measure every line of the episode once, in the background (results go to the shared probe cache), so that
    playing and clicking never has to start the measurement program."""

    def __init__(self, ass, owners, cache_path: Optional[Path], chunk: int = 150, video_height=None) -> None:
        super().__init__()
        self.setAutoDelete(False)             # stop_warm may tryTake a job that just finished
        self.ass, self.owners, self.cache_path, self.chunk = ass, owners, cache_path, chunk
        self.signals = _WarmSignals()
        self.cancelled = False
        self.runner = KillableRun()
        self.finished = threading.Event()
        self.video_height = video_height

    def cancel(self) -> None:
        self.cancelled = True
        self.runner.cancel()

    def run(self) -> None:
        li = None
        try:
            li = LayoutIndex(self.ass, self.owners, measure_cache=self.cache_path, video_height=self.video_height,
                             runner=self.runner)
            idx = [i for i, o in enumerate(self.owners)
                   if o.get("kind") == "base" and i < len(self.ass.events) and self.ass.events[i].kind == "Dialogue"]
            for k in range(0, len(idx), self.chunk):
                if self.cancelled:
                    return
                li.geometry(idx[k:k + self.chunk])
                if self.cancelled:
                    return
                self.signals.progress.emit(min(len(idx), k + self.chunk), len(idx))
        except CancelledError:
            pass
        except Exception:  # pragma: no cover - warming is only an optimisation
            import traceback

            if not self.cancelled:
                traceback.print_exc()
        finally:
            try:
                if li is not None:
                    li.close()
                if not self.cancelled:
                    self.signals.done.emit()
            finally:
                self.finished.set()


class WatchController(QObject):
    groupPicked = Signal(str)                          # select this group (the player is already paused)
    deselected = Signal()
    resumeRequested = Signal()                         # right click on empty space: deselect and play
    readingRequested = Signal(str, QPoint)             # annotation id, global position
    newReadingRequested = Signal(str, int, int, QPoint)   # member id, visible [a, b) in the member text, position
    pinyinRequested = Signal(str, int, int, QPoint)
    moveRequested = Signal(object)                     # {member id: {"an":…, "pos"/"move":…}}
    transformRequested = Signal(object)                # one history operation on apply
    textEditRequested = Signal(str, QRect, int)        # member id, global rect of the line, font pixel size
    doubleClickedEmpty = Signal()
    note = Signal(str)                                 # short on-screen message
    warming = Signal(int, int)                         # lines measured, total (background warm-up)
    speakerRequested = Signal(str, QPoint)
    addLanguageRequested = Signal(list, str)

    def __init__(self, player: MpvWidget, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.player = player
        self.ass = None
        self.owners: list[dict[str, Any]] = []
        self.playres = (1920, 1080)
        self.serial = 0
        self.cache_path: Optional[Path] = None
        self.geo: dict[int, LineGeom] = {}
        self.anchors: dict[int, dict] = {}
        self._starts: list[tuple[int, int, int]] = []
        self._max_dur = 0
        self._job_running = False
        self._geometry_jobs: dict[QObject, GeometryJob] = {}
        self._measuring_stopped = False
        self._job_wanted = False
        self._pending: Optional[tuple] = None        # a click waiting for geometry
        self._warm: Optional[WarmJob] = None
        self._ghost_hold = False                    # the drag preview stays until the moved subtitles are loaded
        self._ghost_timer = QTimer(self)
        self._ghost_timer.setSingleShot(True)
        self._ghost_timer.setInterval(4000)
        self._ghost_timer.timeout.connect(self._end_ghost)
        self.selected: Optional[str] = None
        self.hover: Optional[str] = None
        self.drag: Optional[dict] = None
        self.member_lane: dict[str, str] = {}
        self.enabled = True
        self.frames_visible = True                   # 视图 → 观剧模式选项 → 暂停时显示字幕外框
        self.frame_color = SELECT_COLOR              # 视图 → 观剧模式选项 → 字幕外框颜色…
        self.transform: Optional[TransformTool] = None
        self._transform_preview: Optional[TransformTool] = None
        self._transform_only_line = False
        self._transform_drag: Optional[Drag] = None
        self._transform_ruby: dict[int, Any] = {}
        self.last_clicked_member: Optional[str] = None
        self.speaker_hits = []
        self._speaker_cache = {}
        self._speaker_geoms = {}
        self._speaker_estimates = {}
        self.missing_buttons = {}
        for lane in ("original", "translation"):
            button = QPushButton(player)
            button.help_anchor = "review"
            button.clicked.connect(lambda _checked=False, lane=lane: self.selected and self.addLanguageRequested.emit([self.selected], lane))
            button.hide()
            self.missing_buttons[lane] = button
        self._single = QTimer(self)                 # single click on empty space waits for a possible double click
        self._single.setSingleShot(True)
        self._single.timeout.connect(self._empty_single_click)
        player.mousePressed.connect(self._press)
        player.mouseMoved.connect(self._move)
        player.mouseReleased.connect(self._release)
        player.mouseDoubleClicked.connect(self._double)
        player.osdChanged.connect(lambda _d: self.redraw())
        self.bar = SeekBar(player)
        player.mouseLeft.connect(lambda: self.bar.visible and self.bar._hide.start())
        player.timeChanged.connect(lambda _t: self.bar.tick())

    # ------------------------------------------------------------ data
    def set_preview(self, ass, owners: list[dict], lanes: dict[str, str], cache_path: Optional[Path]) -> None:
        self._measuring_stopped = False
        # A queued preview may arrive just after T. Rebuild untouched sessions
        # against its precise geometry, preserving the scope chosen at entry.
        tool = self._transform_preview or self.transform
        reenter = tool is not None and not tool.changes()
        only_line = self._transform_only_line
        self.finish_transform(apply=True)
        if self._ghost_hold:                          # the moved subtitles are here: drop the drag preview
            self._end_ghost()
        old_ass, old_owners = self.ass, self.owners
        same_layout = old_ass is not None and old_ass.playres == ass.playres and old_ass.styles == ass.styles
        unchanged = {i for i in range(min(len(ass.events), len(owners)))
                     if same_layout and i < len(old_ass.events) and i < len(old_owners)
                     and ass.events[i] == old_ass.events[i] and owners[i] == old_owners[i]}
        keep = {i: geom for i, geom in self.geo.items() if i in unchanged}
        for signals, job in list(self._geometry_jobs.items()):
            if job.finished.is_set() and job.cancelled:
                self._geometry_jobs.pop(signals)
                continue
            if job.cancelled:
                continue
            job.reusable_indices.intersection_update(unchanged if job.target_serial == self.serial else ())
            if job.reusable_indices:
                job.target_serial = self.serial + 1
            else:
                job.cancel()
                if QThreadPool.globalInstance().tryTake(job):
                    job.finished.set()
                    self._geometry_jobs.pop(signals)
        self._job_running = any(not job.cancelled for job in self._geometry_jobs.values())
        self.ass, self.owners, self.cache_path = ass, owners, cache_path
        self.member_lane = lanes
        self.playres = ass.playres
        self.serial += 1
        self.invalidate_speakers()
        self.geo, self.anchors = keep, {i: a for i, a in self.anchors.items() if i in keep}
        self._speaker_geoms = {i: g for i, g in self._speaker_geoms.items() if i in keep}
        self._speaker_estimates.clear()
        rows = sorted((e.start, e.end, i) for i, e in enumerate(ass.events) if e.kind == "Dialogue" and i < len(owners))
        self._starts = rows
        self._max_dur = max((e - s for s, e, _i in rows), default=0)
        if reenter:
            self.enter_transform(only_line=only_line)
        self.request_geometry()
        self.redraw()

    def warm_up(self) -> None:
        """Measure the whole episode once in the background (first preview of a session)."""
        if self.ass is None:
            return
        self._cancel_warm()
        job = WarmJob(self.ass, self.owners, self.cache_path, video_height=self._video_height())
        job.signals.progress.connect(self._warm_progress)
        job.signals.done.connect(self._warm_done)
        self._warm = job
        QThreadPool.globalInstance().start(job, -1)          # below previews and clicks

    def _warm_progress(self, done: int, total: int) -> None:
        if self._warm is not None and self.sender() is self._warm.signals:
            self.warming.emit(done, total)

    def _warm_done(self) -> None:
        if self._warm is not None and self.sender() is self._warm.signals:
            self.warming.emit(-1, -1)

    def stop_warm(self) -> None:
        """Cancel the background measuring (closing the app / switching episodes)."""
        self.finish_transform()
        self.last_clicked_member = None
        self._measuring_stopped = True
        jobs = [self._warm, *self._geometry_jobs.values()]
        self._warm = None
        self._geometry_jobs.clear()
        self._job_running = self._job_wanted = False
        self._pending = None
        self._stop_jobs(jobs)

    def _cancel_warm(self) -> None:
        if self._warm is not None:
            job = self._warm
            self._warm = None
            self._stop_jobs([job])

    @staticmethod
    def _stop_jobs(jobs) -> None:
        jobs = [job for job in jobs if job is not None]
        deadline = time.monotonic() + 2.0
        for job in jobs:
            job.cancel()
            # A queued job owns no resources; remove it instead of waiting for a
            # busy global pool. Running jobs signal only after probe/DB cleanup.
            if QThreadPool.globalInstance().tryTake(job):
                job.finished.set()
        for job in jobs:
            job.finished.wait(max(0.0, deadline - time.monotonic()))

    def upcoming(self, ahead_ms: float = 8000.0) -> list[int]:
        """Lines that start within the next few seconds (measured before they appear)."""
        if not self._starts:
            return []
        t = self.player.time * 1000 / 10.0
        lo = bisect.bisect_left(self._starts, (t, -1, -1))
        hi = bisect.bisect_right(self._starts, (t + ahead_ms / 10.0, 1 << 60, 1 << 60))
        return [i for _s, _e, i in self._starts[lo:hi]]

    def active(self, t_ms: Optional[float] = None) -> list[int]:
        if not self._starts:
            return []
        t = (self.player.time * 1000 if t_ms is None else t_ms) / 10.0
        hi = bisect.bisect_right(self._starts, (t, 1 << 60, 1 << 60))
        out = []
        k = hi - 1
        while k >= 0 and self._starts[k][0] > t - self._max_dur - 1:
            s, e, i = self._starts[k]
            if s <= t < e:
                out.append(i)
            k -= 1
        return sorted(out)

    def on_time(self, _t: float) -> None:
        if self.ass is None:
            return
        if any(i not in self.geo for i in self.active() + self.upcoming()):
            self.request_geometry()
        if self.selected or self.hover or self._speakers_enabled():
            self.redraw()

    def request_geometry(self, extra: Optional[list[int]] = None) -> None:
        if self.ass is None or self._measuring_stopped:
            return
        for signals, job in list(self._geometry_jobs.items()):
            if job.cancelled and job.finished.is_set():
                self._geometry_jobs.pop(signals)
        running = [job for job in self._geometry_jobs.values()
                   if not job.cancelled and job.target_serial == self.serial]
        measuring = {i for job in running for i in job.reusable_indices}
        active = [i for i in self.active() if i not in self.geo and i not in measuring]
        # Visible lines can overtake a running look-ahead job; avoid duplicate work.
        if running and not active:
            self._job_wanted = True
            return
        # Publish visible names before measuring the look-ahead window.
        if self._pending is not None and self._pending[0] == "edit_member":
            mid = self._pending[1][0]
            extra = list(extra or []) + [i for i, o in enumerate(self.owners)
                                        if o.get("member") == mid and o.get("kind") == "base"]
        want = active or [i for i in self.upcoming() + list(extra or []) if i not in self.geo and i not in measuring]
        if self.selected and not active:
            want += [i for i in self._group_events(self.selected) if i not in self.geo and i not in measuring and i not in want]
        if not want:
            return
        self._job_wanted = bool(active and any(i not in self.geo for i in self.upcoming()))
        self._job_running = True
        job = GeometryJob(self.serial, self.ass, self.owners, sorted(set(want)), self.cache_path, self._video_height())
        job.signals.coarse.connect(self._geometry_coarse)
        job.signals.done.connect(self._geometry_done)
        self._geometry_jobs[job.signals] = job
        QThreadPool.globalInstance().start(job, 1 if active else 0)

    def _publish_geometry(self, serial: int, job, geo: dict, anchors: dict) -> None:
        if self._measuring_stopped or job.cancelled or serial != job.serial or job.target_serial != self.serial:
            return
        self.geo.update({i: g for i, g in geo.items() if i in job.reusable_indices})
        self.anchors.update({i: a for i, a in anchors.items() if i in job.reusable_indices})
        self.redraw()
        if self._pending is not None and self._ready():
            kind, args = self._pending
            self._pending = None
            getattr(self, kind)(*args)

    def _geometry_coarse(self, serial: int, geo: dict, anchors: dict) -> None:
        job = self._geometry_jobs.get(self.sender())
        if job is not None:
            self._publish_geometry(serial, job, geo, anchors)

    def _geometry_done(self, serial: int, geo: dict, anchors: dict) -> None:
        job = self._geometry_jobs.pop(self.sender(), None)
        if job is None:
            return
        self._job_running = any(not job.cancelled for job in self._geometry_jobs.values())
        self._publish_geometry(serial, job, geo, anchors)
        if not self._measuring_stopped and self._job_wanted:
            self._job_wanted = False
            self.request_geometry()

    def _geometry_now(self) -> None:
        """Request missing measured hit boxes without cold probes on the GUI thread."""
        self.request_geometry()

    def _group_events(self, gid: str) -> list[int]:
        return [i for i, o in enumerate(self.owners) if o.get("group") == gid]

    def _current_geoms(self) -> list[LineGeom]:
        return [self.geo[i] for i in self.active() if i in self.geo]

    def _hit(self, p: QPointF):
        m = self._mapper()
        if not m.ok:
            return None, None
        x, y = m.to_script(p)
        return LayoutIndex.hit(self._current_geoms(), x, y, pad=8.0 / max(m.sx, 1e-6)), (x, y)

    def _mapper(self) -> VideoMapper:
        return VideoMapper(self.player.osd, self.playres)

    def _ready(self) -> bool:
        return self.ass is not None and all(i in self.geo for i in self.active())

    # ------------------------------------------------------------ selection
    def _video_height(self) -> Optional[int]:
        mpv = self.player.mpv
        if mpv is not None:
            try:
                return int(mpv.get("height") or 0) or None
            except (RuntimeError, ValueError, TypeError):
                pass
        return None

    def _layout_index(self) -> LayoutIndex:
        return LayoutIndex(self.ass, self.owners, measure_cache=self.cache_path, video_height=self._video_height())

    def enter_transform(self, *, only_line: bool = False) -> bool:
        if self.transform is not None:
            return True
        if self.ass is None or not self.selected or not self._mapper().ok:
            return False
        doc = self._doc()
        if doc is None:
            return False
        indices = [i for i in self._group_events(self.selected)
                   if self.owners[i].get("kind") == "base" and self.ass.events[i].kind == "Dialogue"]
        if only_line and indices:
            fallback = next((i for i in indices if self.member_lane.get(self.owners[i]["member"]) == "original"), indices[0])
            indices = [next((i for i in indices if self.owners[i]["member"] == self.last_clicked_member), fallback)]
        if not indices:
            return False
        li = None
        try:
            self.player.pause()
            self._single.stop()
            self._end_ghost()
            self.drag = None
            self.bar.hide()
            li = self._layout_index()
            from subkit.core.transform import transform_of

            members = []
            for geom in li.precise_geometry(indices):
                owner = self.owners[geom.index]
                member = doc.members[geom.member]
                event = self.ass.events[geom.index]
                # The exporter captures this after resampling/display rules,
                # before whole-line overrides; removing tags cannot recover it.
                source = event.copy(text=owner["source_text"])
                tf = transform_of(self.ass, event)
                anchor = li.anchor(geom.index)
                self.geo[geom.index], self.anchors[geom.index] = geom, anchor
                members.append(TransformMember.from_geom(geom, source, self.ass.style(event.style), anchor,
                               member.overrides, measured_scale=(tf.fscx or 100, tf.fscy or 100),
                               original=member.lane == "original"))
            self.transform = TransformTool(self.ass, members, perspective_distance=li.perspective_distance)
            self._transform_only_line = only_line
            self._transform_sources = {m.member: m.source_event for m in members}
            self._transform_ruby = {i: self.ass.events[i].copy() for i in self.active()
                                    if self.owners[i].get("kind") == "ruby" and self.owners[i].get("member") in self._transform_sources}
            self._transform_autohide = self.player.autohide
            self._transform_cursor = self.player.cursor()
            self.player.autohide = False
            self.player.setCursor(Qt.CursorShape.SizeAllCursor)
            self.hover = None
            for oid in (OV_SELECT, OV_HOVER, OV_DRAG):
                self.player.clear_overlay(oid)
            self.player.message(TransformTool.hint)
            self._draw_transform()
            return True
        except Exception as exc:
            self._transform_error(exc)
            return False
        finally:
            if li is not None:
                li.close()

    def finish_transform(self, *, apply: bool = False) -> None:
        tool = self.transform
        if tool is None:
            return
        try:
            if apply and self._transform_drag is not None and self._transform_drag.current != self._transform_drag.start:
                tool.end_drag(self._transform_drag)
            changes = tool.changes() if apply else {}
        finally:
            self.transform = self._transform_preview = self._transform_drag = None
            self._transform_ruby = {}
            self._end_ghost()
            for oid in (OV_TRANSFORM, OV_SELECT, OV_HOVER, OV_DRAG):
                self.player.clear_overlay(oid)
            self.player.autohide = getattr(self, "_transform_autohide", self.player.autohide)
            self.player.setCursor(getattr(self, "_transform_cursor", Qt.CursorShape.ArrowCursor))
        if changes:
            self.transformRequested.emit(changes)

    def _transform_error(self, exc: Exception) -> None:
        self.finish_transform()
        self.player.message(f"自由变换失败：{exc}")

    def transform_key(self, key, mods) -> bool:
        if self.transform is None:
            return False
        try:
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.finish_transform(apply=True)
            elif key == Qt.Key.Key_Escape:
                self.finish_transform()
            elif key == Qt.Key.Key_Backspace:
                self._transform_drag = self._transform_preview = None
                self.transform.clear()
                self._transform_ruby = self._ruby_operation({"kind": "clear"})
                self._show_transform_ghost()
            elif key == Qt.Key.Key_3:
                self.transform.toggle_3d()
            elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
                if self._transform_drag is not None:
                    return True
                step = 10 if mods & Qt.KeyboardModifier.ShiftModifier else 1
                dx = step * (int(key == Qt.Key.Key_Right) - int(key == Qt.Key.Key_Left))
                dy = step * (int(key == Qt.Key.Key_Down) - int(key == Qt.Key.Key_Up))
                drag = Drag("inside", (0, 0), (dx, dy))
                ruby = self._ruby_operation(self.transform.operation(drag))
                self.transform.end_drag(drag)
                self._transform_ruby = ruby
                self._show_transform_ghost()
            self.redraw()
        except Exception as exc:
            self._transform_error(exc)
        return True

    def _transform_tolerance(self) -> float:
        return 6 / max(min(self._mapper().sx, self._mapper().sy), 1e-6)

    def _transform_press(self, p: QPointF, button, mods) -> None:
        if button != Qt.MouseButton.LeftButton:
            return
        try:
            xy = self._mapper().to_script(p)
            hit = self.transform.hit_test(xy, self._transform_tolerance())
            if hit is None:
                self.finish_transform(apply=True)
            elif not hit.enabled:
                self.player.message(hit.reason)
                self.player.setCursor(Qt.CursorShape.ForbiddenCursor)
            else:
                m = self._mapper()
                self._transform_drag = Drag(hit.name, xy, xy, screen_scale=(m.sx, m.sy))
                self._transform_cursor_at(hit)
        except Exception as exc:
            self._transform_error(exc)

    def _make_transform_drag(self, p, mods) -> Drag:
        from dataclasses import replace

        return replace(self._transform_drag, current=self._mapper().to_script(p),
                       shift=bool(mods & Qt.KeyboardModifier.ShiftModifier),
                       alt=bool(mods & Qt.KeyboardModifier.AltModifier),
                       ctrl=bool(mods & Qt.KeyboardModifier.ControlModifier))

    def _transform_cursor_at(self, hit) -> None:
        shape = Qt.CursorShape.ArrowCursor
        if hit is not None:
            if not hit.enabled:
                shape = Qt.CursorShape.ForbiddenCursor
            elif hit.name.startswith("rotate_"):
                shape = Qt.CursorShape.CrossCursor
            else:
                shape = {"nw": Qt.CursorShape.SizeFDiagCursor, "se": Qt.CursorShape.SizeFDiagCursor,
                         "ne": Qt.CursorShape.SizeBDiagCursor, "sw": Qt.CursorShape.SizeBDiagCursor,
                         "n": Qt.CursorShape.SizeVerCursor, "s": Qt.CursorShape.SizeVerCursor,
                         "e": Qt.CursorShape.SizeHorCursor, "w": Qt.CursorShape.SizeHorCursor,
                         "pivot": Qt.CursorShape.CrossCursor, "inside": Qt.CursorShape.SizeAllCursor}[hit.name]
        self.player.setCursor(shape)

    def _transform_move(self, p: QPointF, buttons, mods) -> None:
        try:
            if self._transform_drag is not None and buttons & Qt.MouseButton.LeftButton:
                drag = self._make_transform_drag(p, mods)
                self._transform_drag = drag
                self._transform_preview = deepcopy(self.transform)
                self._transform_preview.end_drag(drag)
                self._show_transform_ghost(self._ruby_operation(self.transform.operation(drag)))
                self._draw_transform()
            else:
                self._transform_cursor_at(self.transform.hit_test(self._mapper().to_script(p), self._transform_tolerance()))
        except Exception as exc:
            self._transform_error(exc)

    def _draw_transform(self) -> None:
        try:
            self._draw_speakers()
            tool = self._transform_preview or self.transform
            m = self._mapper()
            if tool is None or not m.ok:
                return
            quad = [m.to_osd(*p) for p in tool.box()]
            events = [quad_event(quad, color=DRAG_COLOR)]
            for h in tool.handles(self._transform_tolerance()):
                x, y = m.to_osd(*h.point)
                color = DRAG_COLOR if h.enabled else "&H808080&"
                if h.name == "pivot":
                    for a, b in (((x-7, y), (x+7, y)), ((x, y-7), (x, y+7))):
                        events.append(quad_event((a, b), color=color, width=1))
                elif h.name.startswith("rotate_"):
                    events.append(f"{{\\an7\\pos(0,0)\\bord1\\shad0\\1c{color}\\p1}}{_ass_circle(x,y,3)}{{\\p0}}")
                else:
                    events.append(quad_event(((x-3,y-3),(x+3,y-3),(x+3,y+3),(x-3,y+3)), color=color, width=1, fill_alpha="&H00&"))
            x, y = min(p[0] for p in quad), max(20, min(p[1] for p in quad)-8)
            events.append(f"{{\\an1\\pos({x:.1f},{y:.1f})\\fs18\\bord2\\shad0\\1c&HFFFFFF&\\3c&H202020&\\fnMicrosoft YaHei}}{tool.readout()}")
            self.player.overlay(OV_TRANSFORM, "\n".join(events))
        except Exception as exc:
            self._transform_error(exc)

    def _ruby_operation(self, op: dict) -> dict[int, Any]:
        from subkit.ass.text import line_tag
        from subkit.core.export import apply_overrides
        from subkit.core.transform import TransformState, apply_group_transform, source_transform_tags, transform_of

        out = {}
        for i, event in self._transform_ruby.items():
            pos = line_tag(event.text, "pos").float_args()
            tf = transform_of(self.ass, event)
            state = TransformState(str(i), tf, self.ass.style(event.style),
                                   {"an": 2, "x": pos[0], "y": pos[1], "move": None}, {},
                                   (pos[0], pos[1]), source_tags=source_transform_tags(event.text))
            changes = apply_group_transform([state], op)[str(i)]
            out[i] = event.copy(text=apply_overrides(event.text, changes)[0])
        return out

    def _show_transform_ghost(self, ruby=None) -> None:
        self._start_ghost()
        self.player.overlay(OV_GHOST, self._transform_ghost_text(ruby))

    def _transform_ghost_text(self, ruby=None) -> str:
        from subkit.ass.resample import scale_text
        from subkit.ass.text import line_tag, remove_tags_named
        from subkit.core.export import apply_overrides

        tool = self._transform_preview or self.transform
        overrides = tool.current_overrides
        m, out = self._mapper(), []
        ruby = self._transform_ruby if ruby is None else ruby
        li = None
        try:
            for i in self.active():
                event = ruby.get(i, self.ass.events[i])
                mid = self.owners[i].get("member")
                if self.owners[i].get("kind") == "base" and mid in overrides:
                    event = self._transform_sources[mid].copy(text=apply_overrides(self._transform_sources[mid].text, overrides[mid])[0])
                st = self.ass.style(event.style)
                if st is None:
                    continue
                pos = line_tag(event.text, "pos") or line_tag(event.text, "move")
                an_tag = line_tag(event.text, "an")
                if pos:
                    x, y = pos.float_args()[:2]
                    an = int(an_tag.num()) if an_tag else st.alignment
                else:
                    if li is None:
                        li = self._layout_index()
                    anc = self.anchors.get(i) or li.anchor(i)
                    an, x, y = anc["an"], anc["x"], anc["y"]
                px, py = m.to_osd(x,y)
                org = line_tag(event.text, "org")
                org_text = ""
                if org:
                    ox, oy = m.to_osd(*org.float_args()[:2])
                    org_text = f"\\org({ox:.2f},{oy:.2f})"
                body = scale_text(remove_tags_named(event.text, _STATIC_DROP - {"clip", "iclip"}), m.sx, m.sy, m.sy)
                out.append("{"+style_override(st,m.sx,m.sy)+f"\\an{an}\\pos({px:.2f},{py:.2f})"+org_text+"}"+body)
            return "\n".join(out)
        finally:
            if li is not None:
                li.close()

    def set_selected(self, gid: Optional[str]) -> None:
        if gid != self.selected:
            self.finish_transform(apply=True)     # picking another group applies, like a click on empty space
        self.selected = gid
        if gid and any(i not in self.geo for i in self._group_events(gid)):
            self.request_geometry()
        self.redraw()

    def redraw(self) -> None:
        self._draw_missing_buttons()
        self._draw_speakers()
        if self.transform is not None:
            self._draw_transform()
            return
        m = self._mapper()
        if not m.ok or self.ass is None:
            return
        cur = self._current_geoms()
        if self.selected and self.player.paused and self.frames_visible:      # frames only on a still picture
            fill = "&HE6&"
            self.player.overlay(OV_SELECT, "\n".join(quad_event([m.to_osd(*p) for p in q], color=self.frame_color, width=2.5, fill_alpha=fill)
                                                     for g in cur if g.group == self.selected for q in g.quads))
        else:
            self.player.clear_overlay(OV_SELECT)
        if self.hover and self.hover != self.selected and self.player.paused and self.drag is None and self.frames_visible:
            self.player.overlay(OV_HOVER, "\n".join(quad_event([m.to_osd(*p) for p in q], color=HOVER_COLOR, width=1.2)
                                                    for g in cur if g.group == self.hover for q in g.quads))
        else:
            self.player.clear_overlay(OV_HOVER)

    @staticmethod
    def _member_boxes(geoms: list[LineGeom]) -> dict[str, Rect]:
        out: dict[str, Rect] = {}
        for g in geoms:
            out[g.member] = g.rect if g.member not in out else out[g.member].union(g.rect)
        return out

    # ------------------------------------------------------------ mouse
    def _draw_missing_buttons(self):
        from subkit.core.merge import MergeSettings, song_family

        main, doc = self.parent(), self._doc()
        mapper = self._mapper()
        group = next((g for g in doc.groups if g.id == self.selected), None) if doc else None
        for lane, button in self.missing_buttons.items():
            visible = bool(getattr(main, "watch_mode", False) and self.enabled and self.transform is None
                           and group and mapper.ok and self.ass is not None
                           and not any(m.lane == lane and m.role != "comment" for m in doc.members_of(group)))
            button.setVisible(visible)
            if not visible:
                continue
            label = lane_label(lane, main.project.languages)
            button.setText("＋ 补上" + label)
            button.adjustSize()
            siblings = [m for m in doc.members_of(group) if m.role != "comment"] or list(doc.members_of(group))
            ref = siblings[-1] if siblings else None
            styles = {**MergeSettings().styles, **(main.session.episode.settings().get("merge", {}).get("styles") or {})}
            role = "song" if group.kind == "song" else (
                "info" if group.kind == "info" else "sign" if group.kind == "sign" else ref.role if ref else "dialogue")
            family = song_family(ref.style, ref.start, max((m.end for m in doc.members.values()), default=0)) if ref else "IN"
            fallback = ref.style if ref and group.kind in ("sign", "info") else styles[f"{lane}/dialogue"]
            name = styles.get(f"{lane}/{role}", fallback).format(fam=family)
            style = self.ass.style(name) or main.session.template.style(name)
            an = (style.alignment or 2) if style else 2
            margin_v = style.num("MarginV") if style else 60
            x = style.num("MarginL") if an % 3 == 1 else self.playres[0] - style.num("MarginR") if an % 3 == 0 else self.playres[0] / 2
            y = margin_v if an >= 7 else self.playres[1] / 2 if an >= 4 else self.playres[1] - margin_v
            px, py = mapper.to_osd(x, y)
            px *= self.player.width() / max(1, mapper.w)
            py *= self.player.height() / max(1, mapper.h)
            left = px if an % 3 == 1 else px - button.width() if an % 3 == 0 else px - button.width() / 2
            top = py if an >= 7 else py - button.height() / 2 if an >= 4 else py - button.height()
            button.move(max(0, min(self.player.width() - button.width(), round(left))),
                        max(0, min(self.player.height() - button.height(), round(top))))
            button.raise_()

    def _speakers_enabled(self):
        main = self.parent()
        return bool(getattr(main, "watch_mode", False) and
                    getattr(main, "act_watch_speaker", None) and main.act_watch_speaker.isChecked())

    def _speaker_text(self, gid):
        main, doc = self.parent(), self._doc()
        book = main.session.editor.character_book
        lane = main.settings.value("watch/speaker_lane", "original")
        from subkit.core.people import needs_speaker
        group = doc.group(gid)
        members = [m for m in doc.members_of(group) if needs_speaker(group, m) and not book.speaker_ignored(m)]
        if not members:
            return "", ""
        members.sort(key=lambda member: member.lane != lane)
        language = main.project.languages.get(lane, "ja")
        for member in members:
            speaker = member.speaker or {}
            cid, label = speaker.get("character"), speaker.get("label", "")
            if cid == "crowd":
                return "群众", "&H808080&"
            if any(ch.id == cid for ch in book.characters):
                return book.display_name(cid, language, label, "short") + "：", "&HFFFFFF&"
        labels = [(m.speaker or {}).get("label", "") for m in members]
        from subkit.core.cc import normalize_speaker
        ignored = {normalize_speaker(label) for label in book.not_person}
        if any(labels) and all(not label or normalize_speaker(label) in ignored for label in labels):
            return "", ""
        label = next((label for label in labels if label), "说话人")
        return "? " + label, "&H0000FF&"

    def invalidate_speakers(self):
        self._speaker_cache.clear()

    def _estimate_speaker_geometry(self, i: int) -> LineGeom:
        """Display-only first-line estimate; never enters geo or the hit index."""
        from subkit.ass.document import Style
        from subkit.ass.text import VisibleText, line_tag, tokenize
        from subkit.core.transform import transform_of

        ev, owner = self.ass.events[i], self.owners[i]
        st = self.ass.style(ev.style) or Style.new(ev.style)
        family, fs, spacing, bold, italic = st.fontname, st.fontsize, st.spacing, st.bold, st.italic
        for token in tokenize(ev.text):
            if token.kind == "comment":
                continue
            if token.kind != "tags":
                break
            for tag in token.tags:
                if tag.name == "fn":
                    family = tag.arg or st.fontname
                elif tag.name == "fs":
                    fs = tag.num() or st.fontsize
                elif tag.name == "fsp":
                    spacing = tag.num() if tag.num() is not None else st.spacing
                elif tag.name in ("b", "i"):
                    if tag.name == "b":
                        bold = bool(tag.num())
                    else:
                        italic = bool(tag.num())
        font = QFont(family)
        font.setPixelSize(1000)
        font.setBold(bold)
        font.setItalic(italic)
        fm = QFontMetricsF(font)
        # ASS size specifies ascender + descender, Qt pixel size specifies em.
        unit = max(1.0, fs) / max(1.0, fm.ascent() + fm.descent())
        tf = transform_of(self.ass, ev)
        sx, sy = tf.fscx / 100, tf.fscy / 100
        vt = VisibleText(ev.text, wrap_style=int(self.ass.info("WrapStyle", "0")))
        first = vt.lines()[0]
        width = (fm.horizontalAdvance(first) * unit + max(0, len(first) - 1) * spacing) * sx
        lh = (fm.height() * unit) * sy
        an_tag = line_tag(ev.text, "an")
        an = int(an_tag.num() or st.alignment) if an_tag else st.alignment
        an = max(1, min(9, an))
        col, row = (an - 1) % 3, (an - 1) // 3
        px, py = self.playres
        ml, mr = ev.margin_l or st.num("MarginL"), ev.margin_r or st.num("MarginR")
        x = ml if col == 0 else ml + (px - ml - mr) / 2 if col == 1 else px - mr
        mv = ev.margin_v or st.num("MarginV")
        y = py - mv if row == 0 else py / 2 if row == 1 else mv
        pos = line_tag(ev.text, "pos") or line_tag(ev.text, "move")
        args = pos.float_args() if pos else None
        if args and len(args) >= 2:
            x, y = args[:2]
        left = x - col * width / 2
        top = y - (2 - row) * vt.line_count * lh / 2
        ink = fm.tightBoundingRect(first or " ")
        rect = Rect(left + ink.left() * unit * sx,
                    top + (fm.ascent() + ink.top()) * unit * sy,
                    left + (ink.right() * unit + max(0, len(first) - 1) * spacing) * sx,
                    top + (fm.ascent() + ink.bottom()) * unit * sy)
        return LineGeom(i, owner["member"], owner["group"], "base", ev.layer, rect,
                        layout_origins=((left, top),), line_height=lh)

    def _speaker_geometry(self, geom: LineGeom) -> LineGeom:
        if geom.layout_origins:
            return geom
        from subkit.ass.text import VisibleText

        estimate = self._estimate_speaker_geometry(geom.index)
        vt = VisibleText(self.ass.events[geom.index].text)
        first = [r for k, r in enumerate(geom.chars) if r is not None and vt.line_of(k) == 0]
        rect = first[0] if first else geom.rect
        for r in first[1:]:
            rect = rect.union(r)
        bearing = estimate.rect.y0 - estimate.layout_origins[0][1]
        return LineGeom(geom.index, geom.member, geom.group, geom.kind, geom.layer, rect,
                        layout_origins=((rect.x0, rect.y0 - bearing),), line_height=estimate.line_height)

    def _draw_speakers(self, offsets=None):
        from subkit.ass.document import Style
        from subkit.core.transform import transform_of, line_quad
        from subkit.ass.text import line_tag, tokenize
        from subkit.ass.resample import scale_text
        self.speaker_hits = []
        mapper, doc = self._mapper(), self._doc()
        if not self._speakers_enabled() or not mapper.ok or self.ass is None or doc is None:
            self.player.clear_overlay(OV_SPEAKER)
            return
        groups = {}
        for i in self.active():
            owner = self.owners[i]
            if owner.get("kind") != "base" or owner.get("member") not in doc.members:
                continue
            measured = self.geo.get(i)
            # Keep the first measured placement through the precise update:
            # estimates may jump once, hit-testing still gets precise geometry.
            if measured is not None:
                geom = self._speaker_geoms.get(i)
                if geom is None:
                    geom = self._speaker_geoms[i] = self._speaker_geometry(measured)
            else:
                geom = self._speaker_estimates.get(i)
                if geom is None:
                    geom = self._speaker_estimates[i] = self._estimate_speaker_geometry(i)
            groups.setdefault(geom.group, []).append(geom)
        tool = self._transform_preview or self.transform
        states = {state.member: state for state in tool.states()} if tool else {}
        events = []
        for gid, geoms in groups.items():
            geom = min(geoms, key=lambda g: (doc.members[g.member].lane != "translation", g.index))
            cache_key = (id(geom), mapper.w, mapper.h, mapper.ml, mapper.mt, mapper.sx, mapper.sy,
                         self.parent().settings.value("watch/speaker_lane", "original"),
                         tuple((m.id, repr(m.speaker)) for m in doc.members_of(doc.group(gid))))
            cached = self._speaker_cache.get(gid) if not tool and not offsets else None
            if cached and cached[0] == cache_key:
                events.append(cached[1])
                if geom.index in self.geo:
                    self.speaker_hits.append(cached[2])
                continue
            text, _color = self._speaker_text(gid)
            if not text:
                continue
            event = self.ass.events[geom.index]
            st = self.ass.style(event.style) or Style.new(event.style)
            origin = geom.layout_origins[0] if geom.layout_origins else (geom.rect.x0, geom.rect.y0)
            anchor = self.anchors.get(geom.index, {"x": origin[0], "y": origin[1]})
            tf, lh = transform_of(self.ass, event), geom.line_height or geom.rect.y1 - geom.rect.y0
            state = states.get(geom.member)
            if state:
                lh *= state.transform.fscy / max(tf.fscy, 1e-6)
                origin, anchor, tf = state.layout_origin, state.anchor, state.transform
            dx, dy = (offsets or {}).get(geom.member, (0, 0))
            x, y = origin[0] + dx, origin[1] + dy
            fontname = line_tag(event.text, "fn")
            fontsize = line_tag(event.text, "fs")
            family = fontname.arg if fontname else st.fontname
            fs = fontsize.num() if fontsize else st.fontsize
            # The first visual line has its own ink bounds; a longer second
            # line must not pull the name away from the first glyph.
            first_ink = geom._frames[0][0] if geom._frames else geom.rect
            ink_offset = first_ink.x0 - (geom.layout_origins[0][0] if geom.layout_origins else origin[0])
            x += ink_offset - 0.3 * fs * tf.fscx / 100 - 2 * st.outline - 1
            font = QFont(family)
            font.setPixelSize(max(1, round(fs * mapper.sy)))
            font.setBold(st.bold)
            fm = QFontMetricsF(font)
            available = max(0, (x - st.outline) * mapper.sx)
            width = (fm.horizontalAdvance(text) + st.outline * mapper.sy * 2) * tf.fscx / 100
            ratio = max(0.5, min(1.0, available / max(width, 1)))
            if width * ratio > available:
                while text and fm.horizontalAdvance(text + "…") * tf.fscx / 100 * ratio > available:
                    text = text[:-1]
                text = text + "…" if text else ""
            if not text:
                continue
            width = fm.horizontalAdvance(text) * tf.fscx / 100 * ratio / mapper.sx
            height = lh * ratio
            rect = Rect(x - width, y + lh - height, x, y + lh)
            pivot = tf.org or (anchor["x"] + dx, anchor["y"] + dy)
            quad = line_quad(rect, (anchor["x"] + dx, anchor["y"] + dy), tf,
                             layout_origin=(x, y), measured_scale=(tf.fscx or 100, tf.fscy or 100))
            hit = (quad, geom.member)
            if geom.index in self.geo:
                self.speaker_hits.append(hit)
            px, py = mapper.to_osd(x, y + lh)
            ox, oy = mapper.to_osd(*pivot)
            leading = ""
            for token in tokenize(event.text):
                if token.kind != "tags":
                    break
                leading += "".join(tag.raw for tag in token.tags if tag.name in (
                    "fn", "fs", "b", "i", "fsp", "c", "1c", "3c", "4c", "alpha", "1a", "3a", "4a", "bord", "shad"))
            safe = text.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", " ").replace("\r", " ")
            if safe.startswith("? "):
                primary, _alpha = _ass_colour(st.get("PrimaryColour"))
                for token in tokenize("{" + leading + "}"):
                    for tag in token.tags:
                        if tag.name in ("c", "1c"):
                            primary = tag.arg or primary
                safe = "{\\1c&H0000FF&}?{\\1c" + primary + "}" + safe[1:]
            events.append("{" + style_override(st, mapper.sx, mapper.sy) + scale_text(leading, mapper.sx, mapper.sy, mapper.sy)
                          + f"\\an3\\pos({px:.2f},{py:.2f})\\org({ox:.2f},{oy:.2f})"
                           + f"\\fs{fs * mapper.sy * ratio:.2f}\\fscx{tf.fscx:g}\\fscy{tf.fscy:g}"
                           + f"\\fsp{st.spacing * mapper.sx * ratio:.2f}"
                          + f"\\frz{tf.frz:g}\\frx{tf.frx:g}\\fry{tf.fry:g}\\fax{tf.fax:g}\\fay{tf.fay:g}"
                          + "}" + safe)
            if not tool and not offsets:
                self._speaker_cache[gid] = (cache_key, events[-1], hit)
        self.player.overlay(OV_SPEAKER, "\n".join(events))

    def _global(self, p: QPointF) -> QPoint:
        k = self.player.osd_scale()
        return self.player.mapToGlobal(QPoint(int(p.x() / k), int(p.y() / k)))

    def _on_subtitle(self, p: QPointF) -> bool:
        if self.ass is None:
            return False
        if not self._ready():
            self._geometry_now()
        return self._hit(p)[0] is not None

    def _press(self, p: QPointF, button, mods) -> None:
        if self._speakers_enabled() and button == Qt.MouseButton.LeftButton:
            from subkit.core.layout_index import in_quad
            self._draw_speakers()
            xy = self._mapper().to_script(p)
            for quad, mid in reversed(self.speaker_hits):
                if in_quad(quad, *xy, 3):
                    self._single.stop()
                    self.player.pause()
                    self.speakerRequested.emit(mid, self._global(p))
                    return
        if self.transform is not None:
            self._transform_press(p, button, mods)
            return
        if button == Qt.MouseButton.LeftButton and self.bar.enabled and self.bar.visible and self.bar.on_track(p):
            if self.bar.press(p, on_subtitle=self._on_subtitle(p)):
                return                                # the progress bar, not the subtitles
        if not self.enabled or self.ass is None:
            return
        if button == Qt.MouseButton.LeftButton and not self.player.paused:
            self.player.pause()
        if not self._ready():
            self._pending = ("_press", (p, button, mods))
            self._geometry_now()
            return
        self._press_now(p, button, mods)

    def _deferred_click(self, p: QPointF, button, mods) -> None:
        self._press(p, button, mods)
        self._release(p, button, mods)

    def _press_now(self, p: QPointF, button, mods) -> None:
        hit, _xy = self._hit(p)
        if hit is not None:
            self.last_clicked_member = hit.line.member
        if button == Qt.MouseButton.LeftButton:
            if hit is not None:
                gid = hit.line.group
                self.groupPicked.emit(gid)
                self.drag = {"p0": p, "gid": gid, "member": hit.line.member,
                             "only_line": bool(mods & Qt.KeyboardModifier.AltModifier), "moved": False}
            else:
                self._single.start(QApplication.doubleClickInterval())
        elif button == Qt.MouseButton.RightButton:
            if hit is None:
                self.resumeRequested.emit()
                return
            gid = hit.line.group
            if gid != self.selected:
                self.groupPicked.emit(gid)
            self._reading_at(hit, p)

    def _empty_single_click(self) -> None:
        if self.selected:
            self.deselected.emit()
        elif self.player.paused:
            self.player.play()

    def _reading_at(self, hit, p: QPointF) -> None:
        g = hit.line
        gp = self._global(p)
        doc = self._doc()
        member = doc.members.get(g.member) if doc else None
        if member is not None and member.lang.lower().startswith("zh"):
            project = getattr(self.parent(), "project", None)
            mode = project.data.get("settings", {}).get("pinyin", {}).get("mode", "off") if project else "off"
            if mode == "off":
                return
            if g.kind == "pinyin" and g.ann:
                self.readingRequested.emit(g.ann, gp)
                return
            from subkit.core.readings import visible_map
            from subkit.textutil import is_han

            vt, ids = visible_map(member)
            try:
                i = ids.index(hit.identity) if hit.identity is not None else -1
            except ValueError:
                return
            if 0 <= i < len(vt.text) and is_han(vt.text[i]):
                self.pinyinRequested.emit(member.id, i, i + 1, gp)
            return
        if g.kind == "ruby":
            if g.ann:
                self.readingRequested.emit(g.ann, gp)
            return
        ident = hit.identity
        if ident is None:
            return
        if member is None or member.lang.lower().split("-")[0] != "ja":
            return
        self._reading_for_identity(g, hit.char, ident, gp)

    def _reading_for_identity(self, g: LineGeom, k: int, ident: tuple, gp: QPoint) -> None:
        """The ruby annotation that covers this character, or a new reading for the kanji block around it."""
        from subkit.core.ops import _anchor_ids

        doc = self._doc()
        if doc is None:
            return
        for a in doc.annotations.values():
            if a.member == g.member and a.kind == "ruby" and ident in _anchor_ids(a):
                self.readingRequested.emit(a.id, gp)
                return
        text = g.text
        if not (0 <= k < len(text)) or not is_kanji(text[k]):
            self.note.emit("这里没有读音（只有汉字可以加振假名）")
            return
        a, b = k, k + 1
        while a > 0 and is_kanji(text[a - 1]):
            a -= 1
        while b < len(text) and is_kanji(text[b]):
            b += 1
        # visible indices of the event text → indices in the member text (a display rule may hide characters)
        from subkit.core.readings import visible_map

        m = doc.members.get(g.member)
        if m is None:
            return
        _vt, ids = visible_map(m)
        try:
            ma = ids.index(g.ids[a])
            mb = ids.index(g.ids[b - 1]) + 1
        except (ValueError, IndexError):
            return
        owner = self.owners[g.index] if 0 <= g.index < len(self.owners) else {}
        if (any(x < b and a < z for x, z in owner.get("replacement_ranges", ())) or
                any(identity in ids[ma:mb] for identity in owner.get("replacement_deleted_ids", ()))):
            self.note.emit("这里应用了批量替换，请在原文编辑区添加或修改读音。")
            return
        self.newReadingRequested.emit(g.member, ma, mb, gp)

    def _doc(self):
        w = self.parent()
        s = getattr(w, "session", None)
        return s.doc if s is not None else None

    def _move(self, p: QPointF, buttons, mods) -> None:
        if self.transform is not None:
            self._transform_move(p, buttons, mods)
            return
        if self.bar.enabled:
            near = self.bar.in_zone(p, approach=True)
            self.bar.mouse_move(p, on_subtitle=near and self.ass is not None and self._ready()
                                and self._hit(p)[0] is not None)
            if self.bar.dragging:
                return
        if not self.enabled or self.ass is None:
            return
        d = self.drag
        if d is not None and buttons & Qt.MouseButton.LeftButton:
            if not d["moved"] and (abs(p.x() - d["p0"].x()) + abs(p.y() - d["p0"].y())) > 5:
                d["moved"] = True
            if d["moved"]:
                self._draw_drag(p, mods)
            return
        if self.player.paused and self._ready():
            hit, _xy = self._hit(p)
            gid = hit.line.group if hit else None
            if gid != self.hover:
                self.hover = gid
                self.redraw()

    def _drag_delta(self, p: QPointF, mods) -> tuple[float, float]:
        d = self.drag
        dx, dy = p.x() - d["p0"].x(), p.y() - d["p0"].y()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            if abs(dx) >= abs(dy):
                dy = 0.0
            else:
                dx = 0.0
        return dx, dy

    def _drag_members(self) -> list[str]:
        d = self.drag
        if d["only_line"]:
            return [d["member"]]
        return sorted({o["member"] for o in self.owners if o.get("group") == d["gid"] and o.get("kind") == "base"})

    # ------------------------------------------------------------ drag preview (the real text, not only frames)
    def _ghost_text(self, moved: set[str], dx_px: float, dy_px: float) -> str:
        """Every line on screen as OSD events, the dragged ones moved: while dragging the real subtitles are
        hidden and this shows exactly what the frame will look like."""
        from subkit.ass.resample import scale_text
        from subkit.ass.text import line_tag, remove_tags_named

        m = self._mapper()
        out = []
        for i in self.active():
            ev = self.ass.events[i]
            o = self.owners[i] if i < len(self.owners) else {}
            st = self.ass.style(ev.style)
            if st is None or ev.kind != "Dialogue":
                continue
            if o.get("kind") == "ruby":
                tag = line_tag(ev.text, "pos")
                a = tag.float_args() if tag else None
                if not a or len(a) < 2:
                    continue
                an, x, y = 2, a[0], a[1]
            else:
                anc = self.anchors.get(i)
                if anc is None:
                    li = self._layout_index()
                    try:
                        anc = li.anchor(i)
                    finally:
                        li.close()
                    self.anchors[i] = anc
                an, x, y = anc["an"], anc["x"], anc["y"]
            px, py = m.to_osd(x, y)
            if o.get("member") in moved:
                px, py = px + dx_px, py + dy_px
            body = scale_text(remove_tags_named(ev.text, _STATIC_DROP), m.sx, m.sy, m.sy)
            out.append("{" + f"\\an{an}\\pos({px:.1f},{py:.1f})" + style_override(st, m.sx, m.sy) + "}" + body)
        return "\n".join(out)

    def _start_ghost(self) -> None:
        if self.player.mpv is not None:
            try:
                self.player.mpv.set("sub-visibility", "no")
            except RuntimeError:
                pass
        self._ghost_timer.stop()

    def _end_ghost(self) -> None:
        self._ghost_hold = False
        self._ghost_timer.stop()
        self.player.clear_overlay(OV_GHOST)
        if self.player.mpv is not None:
            try:
                self.player.mpv.set("sub-visibility", "yes")
            except RuntimeError:
                pass

    def _draw_drag(self, p: QPointF, mods) -> None:
        m = self._mapper()
        dx, dy = self._drag_delta(p, mods)
        members = set(self._drag_members())
        if not self.drag.get("ghost"):
            self.drag["ghost"] = True
            self._start_ghost()
        self.player.overlay(OV_GHOST, self._ghost_text(members, dx, dy))
        boxes = self._member_boxes([g for g in self._current_geoms() if g.member in members])
        self.player.overlay(OV_DRAG, "\n".join(
            box_event(x0 + dx, y0 + dy, x1 + dx, y1 + dy, color=DRAG_COLOR, width=2.0, fill_alpha="&HD8&")
            for x0, y0, x1, y1 in (m.rect_to_osd(r) for r in boxes.values())))
        self._draw_speakers({mid: (dx / m.sx, dy / m.sy) for mid in members})

    def _release(self, p: QPointF, button, mods) -> None:
        if self._pending is not None and self._pending[0] == "_press":
            # A mouse released before geometry arrives is a click, not a drag.
            self._pending = ("_deferred_click", self._pending[1])
            return
        if self.transform is not None:
            if button == Qt.MouseButton.LeftButton and self._transform_drag is not None:
                try:
                    drag = self._make_transform_drag(p, mods)
                    if drag.current == drag.start:
                        self._transform_drag = self._transform_preview = None
                        return
                    ruby = self._ruby_operation(self.transform.operation(drag))
                    self.transform.end_drag(drag)
                    self._transform_ruby = ruby
                    self._transform_drag = self._transform_preview = None
                    self._show_transform_ghost()
                    self._draw_transform()
                except Exception as exc:
                    self._transform_error(exc)
            return
        if button == Qt.MouseButton.LeftButton and self.bar.release(p):
            return
        d, self.drag = self.drag, None
        if d is None or button != Qt.MouseButton.LeftButton:
            return
        self.player.clear_overlay(OV_DRAG)
        if not d["moved"]:
            return
        if d.get("ghost"):                            # keep the preview until the moved subtitles are loaded
            self._ghost_hold = True
            self._ghost_timer.start()
        self.drag = d
        dx_px, dy_px = self._drag_delta(p, mods)
        members = self._drag_members()
        self.drag = None
        m = self._mapper()
        dx, dy = dx_px / m.sx, dy_px / m.sy
        changes: dict[str, dict] = {}
        for mid in members:
            idx = next((i for i, o in enumerate(self.owners) if o.get("member") == mid and o.get("kind") == "base"), None)
            if idx is None:
                continue
            anc = self.anchors.get(idx)
            if anc is None:
                li = self._layout_index()
                try:
                    anc = li.anchor(idx)
                finally:
                    li.close()
            if anc.get("move"):
                mv = list(anc["move"])
                mv[0], mv[1], mv[2], mv[3] = mv[0] + dx, mv[1] + dy, mv[2] + dx, mv[3] + dy
                changes[mid] = {"an": anc["an"], "move": [round(v, 1) for v in mv[:4]] + [int(v) for v in mv[4:6]], "pos": None}
            else:
                changes[mid] = {"an": anc["an"], "pos": [round(anc["x"] + dx, 1), round(anc["y"] + dy, 1)], "move": None}
        if changes:
            self.moveRequested.emit(changes)
        elif self._ghost_hold:
            self._end_ghost()

    def _double(self, p: QPointF, button, mods) -> None:
        if self.transform is not None:
            if button == Qt.MouseButton.LeftButton:
                from subkit.core.layout_index import in_quad

                if in_quad(self.transform.box(), *self._mapper().to_script(p), 0):
                    self.finish_transform(apply=True)
            return
        if not self.enabled or self.ass is None or button != Qt.MouseButton.LeftButton:
            return
        self._single.stop()
        if not self._ready():
            self._pending = ("_double", (p, button, mods))
            self._geometry_now()
            return
        hit, _xy = self._hit(p)
        if hit is None and self.bar.enabled and self.bar.visible and self.bar.on_track(p):
            return                                    # a double click on the progress line (a subtitle wins)
        if hit is None:
            self.doubleClickedEmpty.emit()
            return
        if hit.line.kind != "base":
            return
        self._edit_line(hit.line)

    def edit_member(self, mid: str) -> None:
        """Keyboard editing uses the same measured base line as double click."""
        if not self.enabled or self.transform is not None:
            return
        doc = self._doc()
        member = doc.members.get(mid) if doc is not None else None
        if member is not None and member.role != "comment" and not visible_map_text(member):
            self._edit_empty_member(member)
            return
        if self.ass is None:
            return
        indices = [i for i, owner in enumerate(self.owners)
                   if owner.get("member") == mid and owner.get("kind") == "base"
                   and self.ass.events[i].kind == "Dialogue"]
        active = self.active()
        indices.sort(key=lambda i: i not in active)
        if not indices:
            return
        i = indices[0]
        if i not in self.geo:
            self._pending = ("edit_member", (mid,))
            self.request_geometry(extra=indices)
            return
        self._single.stop()
        self._edit_line(self.geo[i])

    def _edit_empty_member(self, member) -> None:
        """An empty row cannot acquire ink geometry; open its editor immediately."""
        from dataclasses import replace

        self._single.stop()
        self._pending = None
        active = set(self.active())
        candidates = [i for i, owner in enumerate(self.owners)
                      if owner.get("group") == member.group and owner.get("kind") == "base"
                      and i in self.geo]
        candidates.sort(key=lambda i: i not in active)
        if candidates and self._mapper().ok:
            self._edit_line(replace(self.geo[candidates[0]], member=member.id))
            return
        # Preview/geometry may still be loading. Use a safe logical-pixel anchor
        # in the player's viewport; InlineTextEditor clamps itself to the screen.
        self.player.pause()
        width = max(80, min(560, self.player.width()))
        size = max(16, min(40, int(self.player.height() * 0.045)))
        point = self.player.mapToGlobal(QPoint(max(0, (self.player.width() - width) // 2),
                                              max(0, int(self.player.height() * 0.7))))
        self.textEditRequested.emit(member.id, QRect(point.x(), point.y(), width, size + 8), size)

    def _edit_line(self, line: LineGeom) -> None:
        m = self._mapper()
        if not m.ok:
            return
        self.player.pause()
        x0, y0, x1, y1 = m.rect_to_osd(line.rect)
        k = self.player.osd_scale()
        tl = self.player.mapToGlobal(QPoint(int(x0 / k), int(y0 / k)))
        rect = QRect(tl.x(), tl.y(), max(80, int((x1 - x0) / k)), max(24, int((y1 - y0) / k)))
        self.textEditRequested.emit(line.member, rect, max(12, int((y1 - y0) / k * 0.8)))


def visible_map_text(m) -> str:
    from subkit.core.readings import visible_map

    return visible_map(m)[0].text


class InlineTextEditor(QFrame):
    """Edit a group's lines right where they are drawn (double click a subtitle on the video).

    The furigana of a Japanese line is shown above the text (click a reading to change it).  Hotkeys:
    Enter apply · Esc cancel · Tab / Shift+Tab next / previous line of the group (applies first) ·
    Ctrl+Enter apply and confirm the group · F2 reading for the selection ·
    Ctrl+G regenerate the furigana of the selection (no selection = whole line; confirmed readings kept) ·
    Ctrl+Shift+G regenerate including confirmed readings.  Clicking elsewhere applies (like Word)."""

    committed = Signal(str, str)                       # member id, new text
    confirmRequested = Signal(str)                     # group id
    readingRequested = Signal(str, QPoint)             # annotation id
    newReadingRequested = Signal(str, int, int, QPoint)   # member id, visible [a, b)
    regenerateRequested = Signal(str, int, int, bool)  # member id, visible [a, b), include confirmed
    formatRequested = Signal(str, object, object, dict)   # member id, visible a / b (None = whole line), change
    groupFormatRequested = Signal(str, dict)
    clearFormatRequested = Signal(str)

    def __init__(self, doc, mids: list[str], current: str, rect: QRect, pixel_size: int, rules=None, parent=None) -> None:
        from .views.floating import FLAGS, OutsideClick

        super().__init__(parent, FLAGS)            # not Qt.Popup: an input method must be able to compose here
        from .views.group_editor import RubyLine, InlineLineEdit

        self._outside = OutsideClick(self, self.close, ignore_floating=True)   # clicking elsewhere applies
        self.doc = doc
        self.mids = mids
        self.mid = current
        self._cancel = False
        self._orig = ""
        self.setStyleSheet("InlineTextEditor { background: #1B1F24; border: 2px solid #3ea6ff; border-radius: 4px; }"
                           "QLineEdit, QPlainTextEdit, QComboBox, QDoubleSpinBox { background: #0d1117; color: white; border: 1px solid #30363d; padding: 2px 4px; }"
                           "QLabel { color: #8b949e; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)
        self.head = QLabel()
        self.traditional_note = QLabel()
        self.traditional_note.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.preview_caption = QLabel("预览 · 点振假名改读音，拖选汉字加读音")
        self.preview_caption.setStyleSheet("color: #8B949E; font-size: 11px;")
        self.ruby = RubyLine(background="#1B1F24", base_colour="#C9D1D9", ruby_colour="#79C0FF")
        self.ruby.rules = rules
        self.ruby.readingClicked.connect(self.readingRequested)
        self.ruby.charClicked.connect(lambda i, gp: self.newReadingRequested.emit(self.mid, i, i + 1, gp))
        self.ruby.rangeSelected.connect(lambda a, b, gp: self.newReadingRequested.emit(self.mid, a, b, gp))
        self.edit = InlineLineEdit()
        self.edit.setStyleSheet("QPlainTextEdit { background: #FFFFFF; color: #1F2328; "
                               "border: 1px solid #8B949E; padding: 2px 4px; } "
                               "QPlainTextEdit:focus { border: 2px solid #3EA6FF; }")
        f = QFont(["Noto Sans CJK JP", "Yu Gothic UI", "Microsoft YaHei"])
        f.setPixelSize(max(14, min(48, pixel_size)))
        self.edit.setFont(f)
        self.edit.setFixedHeight(int(QFontMetricsF(f).height()) + 16)
        self.edit.installEventFilter(self)
        self.hint = hint = QLabel("Enter 生效 · Esc 取消 · Tab 下一行 · Ctrl+Enter 生效并确认 · F2 读音 · Ctrl+G 重新生成振假名（Ctrl+Shift+G 含已确认）"
                      " · Ctrl+B 粗体 · Ctrl+I 斜体 · Ctrl+Shift+> / < 字号")
        hint.setStyleSheet("color: #6e7681; font-size: 11px;")
        from .views.format_bar import FormatBar

        self.fmt = FormatBar(compact=True)
        self.fmt.setStyleSheet("QLabel { color: #c9d1d9; } QToolButton { color: #e6edf3; background: #30363d; "
                               "border: 1px solid #484f58; border-radius: 3px; padding: 1px 6px; }")
        self.fmt.formatRequested.connect(self._format)
        self.fmt.scopedFormatRequested.connect(self._format_scoped)
        self.fmt.clearRequested.connect(lambda: self.clearFormatRequested.emit(self.mid))
        self.edit.selectionChanged.connect(self._show_fmt)
        for w in (self.head, self.traditional_note, self.preview_caption, self.ruby, self.fmt, self.edit, hint):
            lay.addWidget(w)
        from .views.placement import clamp_rect

        w = max(rect.width() + 120, 560)
        self.setGeometry(clamp_rect(QRect(rect.center().x() - w // 2, rect.top() - 140, w, 180), self))
        self._load()

    # ------------------------------------------------------------ content
    def _load(self) -> None:
        m = self.doc.members[self.mid]
        self._orig = editing_text(m)
        self.edit.setText(self._orig)
        lane = lane_label(m.lane, {m.lane: m.lang})
        k = self.mids.index(self.mid) + 1 if self.mid in self.mids else 1
        from html import escape

        self.head.setText(f'<span style="color:#3EA6FF">正在编辑：</span>'
                          f'{escape(lane)} · 第 {k}/{len(self.mids)} 行 · {escape(m.style)}')
        self.traditional_note.setText(self.ruby.rules.script_label if self.ruby.rules is not None else "")
        self.traditional_note.setVisible(self.ruby.rules is not None and self.ruby.rules.converted_for(m))
        japanese = m.lang.lower().split("-")[0] == "ja"
        self.hint.setText("Enter 生效 · Esc 取消 · Tab 下一行 · Ctrl+Enter 生效并确认 · "
                          + ("F2 读音 · Ctrl+G 重新生成振假名（Ctrl+Shift+G 含已确认）" if japanese else JAPANESE_ONLY)
                          + " · Ctrl+B 粗体 · Ctrl+I 斜体 · Ctrl+Shift+> / < 字号")
        self.preview_caption.setVisible(japanese)
        self.ruby.setVisible(japanese)
        self.ruby.set_member(self.doc, m if m.lang.lower().split("-")[0] == "ja" else None)
        self._show_fmt()

    def switch_member(self, mid: str) -> None:
        if mid not in self.mids:
            return
        self._apply()                         # same commit policy as Tab
        self.mid = mid
        self._load()
        self.edit.setFocus()

    def cancel(self) -> None:
        self._cancel = True
        self.close()

    def refresh(self) -> None:
        """The document changed (a reading was set, a line applied): redraw the furigana, keep what is typed."""
        if self.mid not in self.doc.members:
            self.close()
            return
        m = self.doc.members[self.mid]
        self.traditional_note.setText(self.ruby.rules.script_label if self.ruby.rules is not None else "")
        self.traditional_note.setVisible(self.ruby.rules is not None and self.ruby.rules.converted_for(m))
        typed = self.edit.text()
        self._orig = editing_text(m)
        if typed != self._orig and not self.edit.isModified():
            self.edit.setText(self._orig)
        self.ruby.set_member(self.doc, m if m.lang.lower().split("-")[0] == "ja" else None)

    def _apply(self) -> None:
        t = self.edit.text()
        if t != self._orig:
            self.committed.emit(self.mid, t)
            self._orig = t
        self.edit.setModified(False)

    def _selection_visible(self) -> Optional[tuple[int, int]]:
        """The selected characters (or the whole line) as visible indices of the applied text."""
        from subkit.core.readings import plain_range_to_visible, visible_map

        m = self.doc.members[self.mid]
        if self.edit.hasSelectedText():
            a = self.edit.selectionStart()
            r = plain_range_to_visible(m, a, a + len(self.edit.selectedText()))
            return r
        vt, _ids = visible_map(m)
        return 0, len(vt.text)

    # ------------------------------------------------------------ keys
    def eventFilter(self, obj, e) -> bool:  # noqa: N802
        if obj is self.edit and e.type() == e.Type.KeyPress:
            return self._key(e)
        return super().eventFilter(obj, e)

    def _key(self, e: QKeyEvent) -> bool:
        if self.fmt.handle_key(e):
            return True
        k, mods = e.key(), e.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        if k == Qt.Key.Key_Escape:
            self.cancel()
            return True
        if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._apply()
            if ctrl:
                self.confirmRequested.emit(self.doc.members[self.mid].group)
            self.close()
            return True
        if k in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            self._apply()
            if len(self.mids) > 1 and self.mid in self.mids:
                step = -1 if (k == Qt.Key.Key_Backtab or shift) else 1
                self.mid = self.mids[(self.mids.index(self.mid) + step) % len(self.mids)]
                QTimer.singleShot(0, self._load)
            return True
        if k == Qt.Key.Key_F2:
            self._apply()
            QTimer.singleShot(0, self._reading_for_selection)
            return True
        if k == Qt.Key.Key_G and ctrl:
            self._apply()
            QTimer.singleShot(0, lambda: self._regenerate(shift))
            return True
        return False

    def _show_fmt(self) -> None:
        from subkit.core.readings import plain_range_to_visible
        from subkit.core.textformat import shown_props

        from .views.format_bar import describe_target, format_selection

        m = self.doc.members.get(self.mid)
        if m is None:
            return
        st = self.style_lookup(m.style) if getattr(self, "style_lookup", None) else None
        vis = None
        if self.edit.hasSelectedText():
            a = self.edit.selectionStart()
            vis = plain_range_to_visible(m, a, a + len(self.edit.selectedText()))
        vis = format_selection(visible_map_text(m), vis)
        props = shown_props(m, st, vis)
        from subkit.core.fades import query_fade_inheritance

        self.fmt.show_inherited_fade(query_fade_inheritance(self.doc, m.id).raw)
        self.fmt.show_state(target=describe_target(visible_map_text(m), vis), fn=props["fn"] or "",
                            fs=float(props["fs"] or 48.0), bold=bool(props["b"]), italic=bool(props["i"]),
                            colour=props["c1"], raw=props.get("raw") or "", span=bool(vis))

    def _format(self, change: dict) -> None:
        from subkit.core.readings import plain_range_to_visible

        self._apply()
        m = self.doc.members.get(self.mid)
        if m is None:
            return
        vis = None
        if self.edit.hasSelectedText():
            a = self.edit.selectionStart()
            vis = plain_range_to_visible(m, a, a + len(self.edit.selectedText()))
        from .views.format_bar import format_selection

        vis = format_selection(visible_map_text(m), vis)
        self.formatRequested.emit(self.mid, vis[0] if vis else None, vis[1] if vis else None, change)

    def _format_scoped(self, change: dict, scope: str) -> None:
        self._apply()
        if self.mid not in self.doc.members:
            return
        if scope == "group":
            self.groupFormatRequested.emit(self.mid, change)
        else:
            self.formatRequested.emit(self.mid, None, None, change)

    def _reading_for_selection(self) -> None:
        if self.doc.members[self.mid].lang.lower().split("-")[0] != "ja":
            return
        span = self._selection_visible()
        if span is None:
            return
        a, b = span
        self.newReadingRequested.emit(self.mid, a, b, self.edit.mapToGlobal(QPoint(0, self.edit.height())))

    def _regenerate(self, include_confirmed: bool) -> None:
        if self.doc.members[self.mid].lang.lower().split("-")[0] != "ja":
            return
        span = self._selection_visible()
        if span is None:
            return
        a, b = span
        self.regenerateRequested.emit(self.mid, a, b, include_confirmed)

    def closeEvent(self, e) -> None:  # noqa: N802
        if not self._cancel:
            self._apply()           # clicking elsewhere keeps what was typed (like Word)
        super().closeEvent(e)
        parent = self.parentWidget()
        if parent is not None and hasattr(parent, "player"):
            if getattr(parent, "_inline", None) is self:
                parent._inline = None
            if self._cancel:
                parent.activateWindow()
                parent.player.setFocus()

    def show_now(self) -> None:
        from .views.floating import show_activated

        show_activated(self, self.edit)
        self.edit.end(False)
