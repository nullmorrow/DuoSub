"""Small floating editors (reading popup, in-place text editor) that Windows input methods can type into.

A ``Qt.Popup`` window never becomes the active native window on Windows, so an IME composes in the main window's
focused field instead (romaji appearing in the line editor behind the popup).  These windows are therefore tool
windows that take the focus, and the popup behaviour — a press outside closes them — is done here.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtWidgets import QApplication, QWidget

FLAGS = Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint


def show_activated(win: QWidget, focus: QWidget) -> None:
    """Show a floating window as the active window with ``focus`` focused (so the IME composes there)."""
    win.show()
    win.raise_()
    win.activateWindow()
    focus.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
    QTimer.singleShot(0, lambda: _refocus(win, focus))     # Windows activates asynchronously


def _refocus(win: QWidget, focus: QWidget) -> None:
    try:
        if win.isVisible() and not focus.hasFocus():
            win.activateWindow()
            focus.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
    except RuntimeError:          # already deleted
        pass


class OutsideClick(QObject):
    """Calls ``on_outside`` when the mouse is pressed outside ``win`` (and outside windows it opened itself, e.g.
    a combo box list or a colour dialog).  The press still reaches what was clicked."""

    def __init__(self, win: QWidget, on_outside: Callable[[], None], *, ignore_floating: bool = False) -> None:
        super().__init__(win)
        self.win = win
        self.on_outside = on_outside
        self.ignore_floating = ignore_floating     # presses in other floating windows (a reading popup) are not outside
        self._pending = False
        QApplication.instance().installEventFilter(self)

    def _fire(self) -> None:
        self._pending = False
        try:
            if self.win.isVisible():
                self.on_outside()
        except RuntimeError:
            pass

    def _ours(self, w: QWidget) -> bool:
        top = w.window()
        while top is not None:
            if top is self.win:
                return True
            parent = top.parentWidget()
            top = parent.window() if parent is not None else None
        return False

    def eventFilter(self, obj, e) -> bool:  # noqa: N802
        if e.type() != QEvent.Type.MouseButtonPress:
            return False
        try:
            if not self.win.isVisible():
                return False
        except RuntimeError:
            return False
        if isinstance(obj, QWidget) and self._ours(obj):
            return False
        if isinstance(obj, QWidget) and self.ignore_floating:
            t = obj.window().windowType()
            if t in (Qt.WindowType.Tool, Qt.WindowType.Dialog, Qt.WindowType.Popup, Qt.WindowType.ToolTip):
                return False
        if not isinstance(obj, QWidget) and self.win.frameGeometry().contains(e.globalPosition().toPoint()):
            return False
        if not self._pending:                   # one press reaches several objects (window, widget, parents)
            self._pending = True
            QTimer.singleShot(0, self._fire)
        return False
