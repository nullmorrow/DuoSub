"""Offscreen numeric-position wiring with real history and layout-box/player doubles."""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests import test_convert_ui as conversion_ui
from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from subkit.core.export import ExportOptions, build_ass
from subkit.core.layout_index import LayoutIndex, Rect
from subkit.core.merge import MergeSettings
from subkit.model.document import Annotation
from subkit_app.views.position_bar import GRID_TIP, PositionBar


class PositionUI(unittest.TestCase):
    setUpClass = classmethod(conversion_ui.ConversionUI.setUpClass.__func__)
    _dispose = conversion_ui.ConversionUI._dispose
    group = conversion_ui.ConversionUI.group
    select = conversion_ui.ConversionUI.select
    snapshot = conversion_ui.ConversionUI.snapshot

    def setUp(self):
        conversion_ui.ConversionUI.setUp(self)
        self.session.previewReady.connect(self.window._on_preview)
        self.watch, self.player = self.window.watch, self.window.player
        self.player.load_subtitle = Mock()
        self.player.pause = Mock()
        self.player._autohide = False
        self.player._hide_timer = QTimer(self.player)
        self.player.mpv = SimpleNamespace(get=Mock(return_value="720"))
        self.watch.request_geometry = Mock()
        self.watch.warm_up = Mock()
        self.indices = []
        self.layout_calls = []
        def layout(ass, owners, **kwargs):
            self.layout_calls.append(kwargs)
            index = LayoutIndex(ass, owners, **kwargs)
            def boxes(indices):
                result = []
                for i in indices:
                    a = index.anchor(i)
                    x0 = a["x"] - (a["an"] - 1) % 3 * 50
                    y0 = a["y"] - (1 - (a["an"] - 1) // 3 / 2) * 40
                    result.append(Rect(x0, y0, x0 + 100, y0 + 40))
                return result
            index.layout_boxes = boxes  # Only measurement is doubled; anchors parse the actual preview.
            index.close = Mock(wraps=index.close)
            self.indices.append(index)
            return index
        p = patch("subkit_app.watch.LayoutIndex", side_effect=layout)
        p.start()
        self.addCleanup(p.stop)
        self.cn = self.group("g", text="HH")
        self.jp = self.group("j", text="HH")
        self.doc.groups.pop()
        self.jp.group, self.jp.lane, self.jp.style = "g", "original", "JP"
        self.doc.groups[0].members.append(self.jp.id)
        self.cn.overrides = {"an": 7, "pos": [100, 100]}
        self.jp.overrides = {"an": 7, "pos": [100, 170]}
        self.doc.structure_changed()
        self.window.model.reset()
        self.session.engine = SimpleNamespace(analyze=Mock(return_value=[]))
        self.preview()
        self.select("g")
        self.bar = self.window.editor.position
        self.session.editor.set_overrides = Mock(wraps=self.session.editor.set_overrides)

    def preview(self):
        ass, report = build_ass(self.doc, {}, self.template, self.path, ExportOptions(ruby=False))
        self.session._applied_revision = self.doc.revision
        self.session.previewReady.emit("preview.ass", ass, report)

    def assert_one_undo(self, before, label):
        tx = self.session.editor.history.undo()
        self.assertEqual(tx.label, label)
        self.assertEqual(self.snapshot(), before)
        self.assertIsNone(self.session.editor.history.undo())
        self.session.editor.history.redo()
        self.cn = self.doc.members[self.cn.id]
        self.jp = self.doc.members[self.jp.id]
        self.session.engine.analyze.assert_not_called()

    def test_row_wording_tooltips_default_scope_and_focus_values(self):
        self.assertEqual(self.bar.scope(), "group")
        self.assertEqual((self.bar.x.value(), self.bar.y.value()), (100, 170))
        self.assertTrue(self.bar.alignments[7].isChecked())
        self.assertEqual(self.bar.alignments[1].toolTip(), GRID_TIP)
        self.assertEqual([b.text() for b in self.bar.presets.values()],
                         ["恢复原位置", "左侧", "右侧", "顶部", "底部", "底部第二人"])
        mid, edit = self.window.editor.cn_edits[0]
        edit.focused.emit()
        self.assertEqual(self.window.editor.position_member(), mid)
        self.assertEqual(self.bar.y.value(), 100)
        self.window.select_group("g")
        self.assertEqual(self.bar.y.value(), 100)
        self.assertTrue(all(li.close.call_count == 1 for li in self.indices))
        self.assertTrue(all(c["video_height"] == 720 for c in self.layout_calls))

    def test_display_anchor_does_not_measure_member_boxes(self):
        with patch.object(self.window, "_position_targets", side_effect=AssertionError("display must not measure boxes")):
            self.window._refresh_position()
            self.assertEqual((self.bar.x.value(), self.bar.y.value()), (100, 170))
            mid, edit = self.window.editor.cn_edits[0]
            edit.focused.emit()
            self.assertEqual((self.bar.x.value(), self.bar.y.value()), (100, 100))

    def test_scope_is_remembered_in_isolated_settings(self):
        self.bar.scope_box.setCurrentIndex(1)
        other = PositionBar()
        self.addCleanup(other.deleteLater)
        other.set_settings(self.settings)
        self.assertEqual(other.scope(), "line")
        self.assertEqual(other.scope_box.currentText(), "只这一行")

    def test_xy_both_scopes_expected_changes_and_single_undo(self):
        for scope in ("group", "line"):
            with self.subTest(scope=scope):
                self.preview()
                self.bar.scope_box.setCurrentIndex(0 if scope == "group" else 1)
                before = self.snapshot()
                self.session.editor.set_overrides.reset_mock()
                self.bar.x.setValue(120.3)
                expected = {self.jp.id: {"pos": [120.3, 170]}}
                if scope == "group":
                    expected[self.cn.id] = {"pos": [120.3, 100]}
                self.session.editor.set_overrides.assert_called_once_with(expected, "位置")
                self.session.request_preview.assert_called()
                self.assertIn("位置：1 组（Ctrl+Z 撤销）", self.window.statusBar().currentMessage())
                self.assert_one_undo(before, "位置")
                self.session.editor.history.undo()
                self.cn = self.doc.members[self.cn.id]
                self.jp = self.doc.members[self.jp.id]

    def test_typing_enter_focus_out_and_arrow_steps(self):
        self.window.show()
        self.window.activateWindow()
        self.bar.x.setFocus()
        self.app.processEvents()
        self.bar.x.lineEdit().selectAll()
        QTest.keyClicks(self.bar.x.lineEdit(), "123.4")
        self.session.editor.set_overrides.assert_not_called()
        QTest.keyClick(self.bar.x, Qt.Key.Key_Return)
        self.session.editor.set_overrides.assert_called_once()
        self.assertEqual(self.jp.overrides["pos"], [123.4, 170])
        self.preview()
        self.session.editor.set_overrides.reset_mock()
        self.bar.y.setFocus()
        self.bar.y.lineEdit().selectAll()
        QTest.keyClicks(self.bar.y.lineEdit(), "190.2")
        self.session.editor.set_overrides.assert_not_called()
        self.window.editor.jp_edit.setFocus()
        self.app.processEvents()
        self.session.editor.set_overrides.assert_called_once()
        self.assertEqual(self.cn.overrides["pos"], [123.4, 120.2])
        self.preview()
        self.session.editor.set_overrides.reset_mock()
        self.bar.x.setFocus()
        QTest.keyClick(self.bar.x, Qt.Key.Key_Up)
        self.session.editor.set_overrides.assert_called_once()
        self.assertEqual(self.jp.overrides["pos"], [123.5, 190.2])

    def test_alignment_keeps_single_line_layout_in_both_scopes(self):
        self.bar.scope_box.setCurrentIndex(1)
        before = self.snapshot()
        self.bar.alignments[3].click()
        self.assertEqual(self.jp.overrides, {"an": 3, "pos": [200, 210]})
        self.assertEqual(self.cn.overrides, {"an": 7, "pos": [100, 100]})
        self.assert_one_undo(before, "对齐点")
        self.preview()
        self.bar.scope_box.setCurrentIndex(0)
        self.bar.alignments[2].click()
        self.assertEqual(self.cn.overrides["pos"], [150, 140])
        self.assertEqual(self.jp.overrides["pos"], [150, 210])
        self.assertEqual(self.session.editor.set_overrides.call_count, 2)

    def test_move_start_values_both_points_and_times_kept(self):
        self.jp.overrides = {"an": 7, "move": [100, 170, 140, 220, 50, 900]}
        self.preview()
        self.assertEqual((self.bar.x.value(), self.bar.y.value()), (100, 170))
        self.assertIn("两个端点一起移动", self.bar.x.toolTip())
        self.bar.y.setValue(180.1)
        self.assertEqual(self.jp.overrides["move"], [100, 180.1, 140, 230.1, 50, 900])
        self.assertEqual(self.cn.overrides["pos"], [100, 110.1])

    def test_row_side_and_restore_obey_line_scope(self):
        self.bar.scope_box.setCurrentIndex(1)
        self.jp.overrides["raw"] = r"\bord4"
        self.preview()
        before = self.snapshot()
        self.bar.presets["right"].click()
        self.assertEqual(self.jp.overrides["an"], 9)
        self.assertEqual(self.jp.overrides["pos"][1], 170)
        self.assertEqual(self.cn.overrides["pos"], [100, 100])
        self.assert_one_undo(before, "右侧说话人")
        self.preview()
        self.bar.presets["restore"].click()
        self.assertEqual(self.jp.overrides, {"raw": r"\bord4"})
        self.assertEqual(self.cn.overrides["pos"], [100, 100])

    def test_menu_wording_no_shortcuts_multiselect_and_one_transaction(self):
        second = self.group("b")
        second.overrides = {"an": 2, "pos": [900, 950]}
        self.preview()
        self.select("g", "b")
        self.window.subtitle_menu.aboutToShow.emit()
        self.assertEqual([a.text() for a in self.window.position_menu.actions()],
                         ["左侧说话人", "右侧说话人", "移到顶部", "移到底部", "底部第二人", "", "恢复原位置"])
        self.assertTrue(all(a.shortcut().isEmpty() for a in self.window.position_actions.values()))
        before = self.snapshot()
        self.window.position_actions["left"].trigger()
        self.session.editor.set_overrides.assert_called_once()
        self.assertEqual(set(self.session.editor.set_overrides.call_args.args[0]), {self.cn.id, self.jp.id, second.id})
        self.assertEqual(self.jp.overrides["pos"][1], 170)
        self.assertEqual(second.overrides["pos"][1], 950)
        self.assertIn("左侧说话人：2 组", self.window.statusBar().currentMessage())
        self.assert_one_undo(before, "左侧说话人")

    def test_menu_restore_all_groups_and_preserves_readings_text_karaoke(self):
        second = self.group("b")
        second.overrides = {"an": 2, "pos": [900, 950], "raw": r"\blur2"}
        self.doc.annotations["a"] = Annotation("a", "ruby", self.jp.id, [], "HH", "saved", status="auto")
        self.doc.karaoke["saved"] = {"member": self.jp.id, "beats": []}
        self.preview()
        self.select("g", "b")
        before = self.snapshot()
        self.window.position_actions["restore"].trigger()
        self.session.editor.set_overrides.assert_called_once()
        self.assertEqual(self.cn.overrides, {})
        self.assertEqual(second.overrides, {"raw": r"\blur2"})
        self.assertEqual(self.doc.annotations["a"].text, "saved")
        self.assertEqual(self.doc.karaoke, before["karaoke"])
        self.assert_one_undo(before, "恢复原位置")

    def test_vertical_styles_skips_and_group_row_roundtrip(self):
        self.group("song", "song")
        source = self.group("source")
        # SOURCE positioning is evaluated by the core, not inferred from manual overrides.
        from subkit.model.source import Source, SourceEvent
        from subkit.model.document import Run
        self.episode.sources["s"] = Source("s", "zh", "translation", "ass", "s.ass", "s.ass", "sha", 0, "utf-8", "lf",
                                            events=[SourceEvent(0, "Dialogue", 1, 3, r"{\pos(2,3)}HH", "CN")])
        source.runs = [Run("HH", "text", src="s", e=0, a=11, b=13)]
        self.merge["styles"] = {**MergeSettings().styles, "translation/top": "CN.CustomTop", "original/top": "JP.CustomTop"}
        self.select("g", "song", "source")
        before = self.snapshot()
        sources = copy.deepcopy(self.episode.sources)
        self.window.position_actions["top"].trigger()
        self.assertEqual(self.episode.sources, sources)
        self.assertEqual((self.cn.style, self.jp.style), ("CN.CustomTop", "JP.CustomTop"))
        self.assertEqual(self.cn.overrides, {})
        self.assertEqual(source.role, "dialogue")
        self.assertIn("移到顶部：1 组（Ctrl+Z 撤销）（2 组是歌曲、屏幕字或原字幕自带位置，没有改）",
                      self.window.statusBar().currentMessage())
        self.assert_one_undo(before, "移到顶部")
        self.select("g")
        self.window._position_row_action("bottom")
        self.assertEqual((self.cn.style, self.jp.style), ("CN", "JP"))
        self.assertEqual(self.session.editor.history.undo().label, "移到底部")

    def test_disabled_without_selection_and_pending_preview_does_not_reuse_coordinates(self):
        self.window.current_gid = None
        self.window.editor.set_group(self.doc, None, None, None)
        self.window.table.clearSelection()
        self.window.table.setCurrentIndex(self.window.proxy.index(-1, -1))
        self.window._update_position_actions()
        self.assertFalse(self.bar.isEnabled())
        self.assertTrue(all(not a.isEnabled() for a in self.window.position_actions.values()))
        self.select("g")
        self.bar.x.setValue(125)
        self.assertFalse(self.bar.isEnabled())
        self.window._position_xy(140, None)
        self.session.editor.set_overrides.assert_called_once()
        self.session._applied_revision = self.doc.revision - 1
        self.session.previewReady.emit("stale.ass", self.watch.ass, SimpleNamespace(owners=self.watch.owners))
        self.assertFalse(self.bar.isEnabled())
        self.preview()
        self.assertTrue(self.bar.isEnabled())
        self.assertEqual(self.bar.x.value(), 125)
        self.bar.x.setValue(140)
        self.assertEqual(self.jp.overrides["pos"], [140, 170])
        self.assertEqual(self.cn.overrides["pos"], [140, 100])

    def test_layout_failure_closes_index_and_keeps_document_unchanged(self):
        before = self.snapshot()
        self.window._position_cache = None
        real = self.watch._layout_index
        def broken():
            index = real()
            index.layout_boxes = Mock(side_effect=RuntimeError("measurement failed"))
            return index
        with patch.object(self.watch, "_layout_index", side_effect=broken), patch("subkit_app.mainwindow.traceback.print_exc"):
            self.window._position_xy(120, None)
        self.assertEqual(self.snapshot(), before)
        self.indices[-1].close.assert_called_once()
        self.assertIn("measurement failed", self.window.statusBar().currentMessage())
        self.session.editor.set_overrides.assert_not_called()

    def test_f10_f11_menu_access_and_position_actions(self):
        self.window.show()
        self.window.activateWindow()
        self.app.processEvents()
        for pinned, action in ((True, "left"), (False, "right")):
            self.preview()
            self.select("g")
            self.window.set_watch_mode(True, pinned=pinned)
            self.app.processEvents()
            self.assertTrue(self.window.menuBar().isHidden())
            alt_e = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_E, Qt.KeyboardModifier.AltModifier, "e")
            self.assertIs(self.window._hidden_menu_for_key(alt_e), self.window.edit_menu)
            self.player.setFocus()
            self.app.processEvents()
            QTest.keyClick(self.player, Qt.Key.Key_E, Qt.KeyboardModifier.AltModifier)
            self.app.processEvents()
            self.assertTrue(self.window.edit_menu.isVisible())
            self.window.position_actions[action].trigger()
            self.window.edit_menu.hide()
            self.assertEqual(self.cn.overrides["an"], 7 if action == "left" else 9)
            self.player.message.assert_called()
            QTest.qWait(310)  # Drain the existing fullscreen welcome timers before disposing the window.
            self.window.set_watch_mode(False)
            self.assertIsNone(self.window._hidden_menu_for_key(alt_e))
        self.assertEqual(self.session.editor.set_overrides.call_count, 2)


if __name__ == "__main__":
    unittest.main()
