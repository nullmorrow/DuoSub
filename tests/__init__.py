"""Test package setup.

Cyclic garbage collection runs on the main thread only, between tests (see subkit_app/gc_guard.py: a
collection started on a worker thread can delete GUI-thread Qt objects and deadlock against event processing).
"""

import gc
import unittest

gc.disable()
_run = unittest.TestCase.run
_count = [0]


def _run_then_collect(self, result=None):
    from unittest.mock import patch
    from PySide6.QtWidgets import QMessageBox

    with patch("subkit_app.mainwindow.MainWindow._ask_save", return_value=QMessageBox.StandardButton.Save):
        try:
            return _run(self, result)
        finally:
            _count[0] += 1
            gc.collect(2 if _count[0] % 25 == 0 else 1)


unittest.TestCase.run = _run_then_collect
