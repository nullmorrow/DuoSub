"""A slim whole-episode progress bar under the video — outside the picture, so it never covers a subtitle.

Shown in the edit layout and in 看片打轴 (F10); 全屏观看 (F11) uses the thin overlay line at the bottom edge instead.
Click or drag to seek (fast seeks while dragging, an exact one on release); the time under the mouse is shown.
"""

from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


def _clock(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}"


class SeekStrip(QWidget):
    seekRequested = Signal(float, bool)            # seconds, exact

    BG = QColor("#16191d")
    TRACK = QColor("#3a3f46")
    DONE = QColor("#3ea6ff")
    TEXT = QColor("#c9d1d9")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(20)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("点击或拖动跳到这个时间")
        self.duration = 0.0
        self.time = 0.0
        self.hover: Optional[float] = None
        self.dragging = False
        self._last_seek = 0.0
        self._last_px = -1
        self._font = QFont("Microsoft YaHei", 8)

    # ------------------------------------------------------------ geometry
    def _track(self) -> QRectF:
        label_w = 118
        return QRectF(8, self.height() / 2 - 2, max(10.0, self.width() - 16 - label_w), 4)

    def _time_at(self, x: float) -> float:
        r = self._track()
        frac = min(1.0, max(0.0, (x - r.left()) / max(1.0, r.width())))
        return frac * self.duration

    # ------------------------------------------------------------ state
    def set_duration(self, d: float) -> None:
        self.duration = max(0.0, float(d or 0.0))
        self.update()

    def set_time(self, t: float) -> None:
        self.time = float(t or 0.0)
        if self.dragging or not self.isVisible():
            return
        r = self._track()
        px = int(r.left() + r.width() * (self.time / self.duration)) if self.duration > 0 else 0
        if px != self._last_px or int(self.time) != int(getattr(self, "_shown_sec", -1)):
            self._last_px = px
            self._shown_sec = int(self.time)
            self.update()

    # ------------------------------------------------------------ mouse
    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() != Qt.MouseButton.LeftButton or self.duration <= 0:
            return
        self.dragging = True
        self._seek(e.position().x(), exact=False)

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        self.hover = e.position().x()
        if self.dragging:
            self._seek(self.hover, exact=False)
        self.update()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self.dragging:
            self.dragging = False
            self._seek(e.position().x(), exact=True)

    def leaveEvent(self, e) -> None:  # noqa: N802
        self.hover = None
        self.update()
        super().leaveEvent(e)

    def _seek(self, x: float, *, exact: bool) -> None:
        now = time.monotonic()
        t = self._time_at(x)
        self.time = t
        self.update()
        if not exact and now - self._last_seek < 0.07:
            return
        self._last_seek = now
        self.seekRequested.emit(t, exact)

    # ------------------------------------------------------------ paint
    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), self.BG)
        r = self._track()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.TRACK)
        p.drawRoundedRect(r, 2, 2)
        if self.duration > 0:
            xp = r.left() + r.width() * min(1.0, max(0.0, self.time / self.duration))
            p.setBrush(self.DONE)
            p.drawRoundedRect(QRectF(r.left(), r.top(), xp - r.left(), r.height()), 2, 2)
            p.setBrush(QColor("white"))
            p.setPen(QPen(self.DONE, 1.5))
            p.drawEllipse(QRectF(xp - 5, r.center().y() - 5, 10, 10))
        p.setFont(self._font)
        p.setPen(self.TEXT)
        label = f"{_clock(self.time)} / {_clock(self.duration)}"
        p.drawText(QRectF(r.right() + 6, 0, self.width() - r.right() - 8, self.height()),
                   int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight), label)
        if self.hover is not None and self.duration > 0 and r.left() <= self.hover <= r.right():
            p.setPen(QPen(QColor(255, 255, 255, 150), 1))
            p.drawLine(int(self.hover), int(r.top() - 3), int(self.hover), int(r.bottom() + 3))
            txt = _clock(self._time_at(self.hover))
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(txt) + 8
            x = min(max(r.left(), self.hover - tw / 2), r.right() - tw)
            box = QRectF(x, 1, tw, self.height() - 2)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 200))
            p.drawRoundedRect(box, 3, 3)
            p.setPen(self.TEXT)
            p.drawText(box, int(Qt.AlignmentFlag.AlignCenter), txt)
        p.end()
