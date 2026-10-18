"""GUI-thread garbage collection guard (rare cross-thread deadlock during background merges)."""

import gc
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PySide6.QtCore import QObject  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from subkit_app import gc_guard  # noqa: E402


class GcGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.was_enabled = gc.isenabled()
        self.addCleanup(lambda: gc.enable() if self.was_enabled else gc.disable())

    def test_install_disables_automatic_collection_once(self):
        owner = QObject()
        self.addCleanup(owner.deleteLater)
        first = gc_guard.install(owner)
        self.assertFalse(gc.isenabled())
        self.assertIs(gc_guard.install(owner), first)
        self.assertTrue(first.timer.isActive())

    def test_check_collects_generations_by_threshold(self):
        owner = QObject()
        self.addCleanup(owner.deleteLater)
        collector = gc_guard.GuiThreadCollector(owner)
        collector.threshold = (10, 5, 5)
        with patch.object(gc, "collect") as collect, patch.object(gc, "get_count", return_value=(3, 9, 9)):
            collector.check()
        collect.assert_not_called()
        with patch.object(gc, "collect") as collect, patch.object(gc, "get_count", return_value=(11, 2, 9)):
            collector.check()
        self.assertEqual([c.args for c in collect.call_args_list], [(0,)])
        with patch.object(gc, "collect") as collect, patch.object(gc, "get_count", return_value=(11, 6, 6)):
            collector.check()
        self.assertEqual([c.args for c in collect.call_args_list], [(0,), (1,), (2,)])


if __name__ == "__main__":
    unittest.main()
