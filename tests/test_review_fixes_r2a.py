"""R2a: real offscreen dock events and layout restoration, player double."""

import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock

from PySide6.QtCore import QPoint, Qt, qVersion
from PySide6.QtGui import QMouseEvent, QPalette
from PySide6.QtTest import QTest

from tests import test_convert_ui as fixtures
from tests import test_review_fixes_r9 as r9


class ReviewR2a(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    setUp = r9.ReviewR9.setUp
    _dispose = fixtures.ConversionUI._dispose

    def test_title_controls_double_click_close_and_unused_events(self):
        w = self.window
        for dock in (w.dock_source, w.people):
            dock.show()
            bar = dock.titleBarWidget()
            self.assertEqual(bar.toggle.text(), "浮动")
            QTest.mouseClick(bar.toggle, Qt.MouseButton.LeftButton)
            self.app.processEvents()
            self.assertTrue(dock.isFloating())
            self.assertEqual(bar.toggle.text(), "放回右侧")
            self.assertEqual(bar.toggle.toolTip(), "放回右侧")
            QTest.mouseDClick(bar, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
            self.app.processEvents()
            self.assertFalse(dock.isFloating())
            self.assertIs(dock.parentWidget(), w.right_host)
            QTest.mouseDClick(bar, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
            self.app.processEvents()
            self.assertTrue(dock.isFloating())
            QTest.mouseClick(bar.toggle, Qt.MouseButton.LeftButton)
            QTest.mouseClick(bar.close_button, Qt.MouseButton.LeftButton)
            self.assertTrue(dock.isHidden())
            for kind, handler in ((QMouseEvent.Type.MouseButtonPress, bar.mousePressEvent),
                                  (QMouseEvent.Type.MouseMove, bar.mouseMoveEvent),
                                  (QMouseEvent.Type.MouseButtonRelease, bar.mouseReleaseEvent)):
                event = QMouseEvent(kind, QPoint(10, 10), QPoint(10, 10), Qt.MouseButton.LeftButton,
                                    Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
                handler(event)
                self.assertFalse(event.isAccepted())

    def test_opaque_floating_after_switch_and_watch(self):
        w = self.window
        for layout in ("video", "columns", "list", "video"):
            w.set_edit_layout(layout)
            for dock in (w.dock_source, w.people):
                dock.setFloating(True)
                dock.show()
            self.app.processEvents()
            for pinned in (True, False, True):
                w._watch_key(pinned)
                self.app.processEvents()
                self.assertTrue(w.watch_mode)
                self.assertTrue(w.dock_source.isHidden())
                self.assertTrue(w.people.isHidden())
            w.set_watch_mode(False)
            self.app.processEvents()
            for dock in (w.dock_source, w.people):
                self.assertTrue(dock.isFloating())
                self.assertTrue(dock.isVisible())
                for widget in (dock, dock.widget()):
                    self.assertTrue(widget.autoFillBackground())
                    self.assertEqual(widget.palette().color(QPalette.ColorRole.Window).alpha(), 255)
                    self.assertFalse(widget.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground))
            w.act_docks_right.trigger()
            self.app.processEvents()
            for dock in (w.dock_source, w.people):
                self.assertFalse(dock.isFloating())
                self.assertEqual(dock.titleBarWidget().toggle.text(), "浮动")
                self.assertIs(dock.parentWidget(), w if layout == "list" else w.right_host)

    def test_floating_layout_switch_and_restart_off_screen(self):
        w = self.window
        w.people.setFloating(True)
        w.people.show()
        self.app.processEvents()
        w.set_edit_layout("columns", restore=False)
        self.app.processEvents()
        self.assertTrue(w.people.isFloating())
        w.people.move(20000, 20000)
        w.close()
        restarted = fixtures.mw.MainWindow()
        try:
            restarted.show()
            self.app.processEvents()
            self.assertEqual(restarted.edit_layout, "columns")
            self.assertTrue(restarted.people.isFloating())
            self.assertTrue(restarted.people.isVisible())
            frame = restarted.people.frameGeometry()
            self.assertTrue(restarted.screen().availableGeometry().contains(frame.topLeft()))
            self.assertEqual(restarted.people.titleBarWidget().toggle.text(), "放回右侧")
            restarted.people.titleBarWidget().toggle.click()
            self.assertFalse(restarted.people.isFloating())
            self.assertIs(restarted.people.parentWidget(), restarted.right_host)
        finally:
            restarted.close()
            restarted.deleteLater()

    def test_video_fit_right_tabs_and_evidence(self):
        w = self.window
        evidence = Path(os.environ["TEMP"]) / "r2a-evidence"
        evidence.mkdir(exist_ok=True)
        # No saved splitters: fit using the actual player height after showing.
        self.assertLessEqual(abs(w.player.width() - round(w.player.height() * 16 / 9)), 2)
        for dock in (w.dock_source, w.people):
            self.assertIs(dock.parentWidget(), w.right_host)
            self.assertTrue(dock.isHidden())
        before = (w.table.height(), w.player.size())
        self.assertTrue(w.grab().save(str(evidence / "video-docks-closed.png")))
        for dock in (w.dock_source, w.people):
            dock.toggleViewAction().trigger()
        self.app.processEvents()
        self.assertIn(w.people, w.right_host.tabifiedDockWidgets(w.dock_source))
        self.assertEqual((w.table.height(), w.player.size()), before)
        self.assertGreaterEqual(w.editor.height(), 240)
        self.assertGreater(w.people.height(), 100)
        self.assertTrue(w.grab().save(str(evidence / "video-docks-open.png")))
        w.people.titleBarWidget().toggle.click()
        self.app.processEvents()
        self.assertTrue(w.people.grab().save(str(evidence / "people-floating.png")))
        geometry = {"environment": "Windows host / Qt " + qVersion() + " offscreen / player double",
                    "client": [w.width(), w.height()], "player": [w.player.width(), w.player.height()],
                    "list_height": w.table.height(), "editor_height": w.editor.height(),
                    "floating_flags": hex(int(w.people.windowFlags()))}
        (evidence / "geometry.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")

    def test_saved_sizes_are_not_overwritten(self):
        w = self.window
        w.top_split.setSizes([1200, 720])
        self.app.processEvents()
        sizes = w.top_split.sizes()
        w.set_edit_layout("list")
        w.set_edit_layout("video")
        self.app.processEvents()
        self.assertEqual(w.top_split.sizes(), sizes)
        self.assertFalse(w._pending_video_fit)


if __name__ == "__main__":
    unittest.main()
