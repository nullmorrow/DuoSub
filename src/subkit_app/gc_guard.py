"""Run Python's cyclic garbage collector on the GUI thread only.

Qt wrappers caught in reference cycles are freed by the cyclic collector.  An automatic collection is triggered
by whichever thread happens to allocate, so a background job (merge, preview, measuring) can end up deleting
Qt objects that belong to the GUI thread while the GUI thread is processing events: the two threads then wait
for each other (a rare hang seen while a background merge ran).  Automatic collection is switched off and the
GUI thread collects on a timer, generation by generation, with the interpreter's own thresholds.
"""

from __future__ import annotations

import gc

from PySide6.QtCore import QObject, QTimer


class GuiThreadCollector(QObject):
    def __init__(self, parent: QObject, interval_ms: int = 500) -> None:
        super().__init__(parent)
        self.threshold = gc.get_threshold()
        gc.disable()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check)
        self.timer.start(interval_ms)

    def check(self) -> None:
        young, middle, old = gc.get_count()
        if young > self.threshold[0]:
            gc.collect(0)
            if middle > self.threshold[1]:
                gc.collect(1)
                if old > self.threshold[2]:
                    gc.collect(2)


def install(app) -> GuiThreadCollector:
    """Keep the collector alive for the application's lifetime (parented to the application object)."""
    collector = getattr(app, "_gui_thread_collector", None)
    if collector is None:
        collector = GuiThreadCollector(app)
        app._gui_thread_collector = collector
    return collector
