"""Keep popups on screen (a right click near the bottom of a full-screen video must not open a popup below it)."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget


def screen_area(anchor: QPoint, widget: QWidget) -> QRect:
    scr = QGuiApplication.screenAt(anchor) or widget.screen() or QGuiApplication.primaryScreen()
    return scr.availableGeometry()


def place_near(widget: QWidget, anchor: QPoint, *, gap: int = 12) -> None:
    """Move a (top-level) widget next to a global point: below it if it fits, otherwise above; always inside the
    screen that contains the point."""
    widget.adjustSize()
    w, h = widget.frameGeometry().width(), widget.frameGeometry().height()
    area = screen_area(anchor, widget)
    x = anchor.x()
    y = anchor.y() + gap
    if y + h > area.bottom():
        y = anchor.y() - gap - h
    x = max(area.left(), min(x, area.right() - w))
    y = max(area.top(), min(y, area.bottom() - h))
    widget.move(x, y)


def clamp_rect(rect: QRect, widget: QWidget) -> QRect:
    """The same rectangle moved (not resized) so it lies on the screen of its centre."""
    area = screen_area(rect.center(), widget)
    r = QRect(rect)
    if r.width() > area.width():
        r.setWidth(area.width())
    if r.height() > area.height():
        r.setHeight(area.height())
    r.moveLeft(max(area.left(), min(r.left(), area.right() - r.width())))
    r.moveTop(max(area.top(), min(r.top(), area.bottom() - r.height())))
    return r
