"""Q1: real Qt keys/editor/history; player and rendered geometry doubles."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from tests import test_convert_ui as fixtures
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtTest import QTest, QSignalSpy
from PySide6.QtWidgets import QDialog, QLineEdit
from subkit.core.export import ExportOptions, build_ass
from subkit.core.layout_index import LineGeom, Rect
from subkit_app.views.reading_popup import ReadingPopup


class WatchInlineQoL(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.project.settings = lambda: {}
        self.cn = self.group("g", text="翻译")
        self.jp = self.group("j", text="原文")
        self.doc.groups.pop()
        self.jp.group, self.jp.lane, self.jp.lang, self.jp.style = "g", "original", "ja", "JP"
        self.doc.groups[0].members.append(self.jp.id)
        self.doc.structure_changed()
        w = self.window
        self.watch, self.player = w.watch, w.player
        self.player.osd = {"w": 1920, "h": 1080, "ml": 0, "mt": 0}
        self.player.time = 1.5
        self.player._autohide = False
        self.player._hide_timer = QTimer(self.player)
        self.player.pause = Mock(side_effect=lambda: setattr(self.player, "paused", True))
        self.player.mpv = SimpleNamespace(set=Mock(), get=Mock(return_value="1080"))
        self.watch.request_geometry = Mock()
        ass, report = build_ass(self.doc, {}, self.template, self.path, ExportOptions(ruby=False))
        self.watch.set_preview(ass, report.owners, {}, None)
        for i, owner in enumerate(report.owners):
            y = 100 if owner["member"] == self.cn.id else 170
            self.watch.geo[i] = LineGeom(i, owner["member"], "g", "base", 0, Rect(100, y, 200, y + 50))
        w.current_gid = "g"
        self.watch.selected = "g"
        self.watch.enabled = True
        w.watch_mode = True
        w.action_hub.set_scene("watch")
        w.show()
        w.activateWindow()
        self.player.setFocus()
        self.app.processEvents()
        self.addCleanup(self.close_inline)
        self.addCleanup(setattr, w, "watch_mode", False)

    def close_inline(self):
        if self.window._inline is not None:
            self.window._inline.cancel()
        self.app.processEvents()

    def key(self, key, mods=Qt.KeyboardModifier.NoModifier, target=None):
        QTest.keyClick(target or self.player, key, mods)
        self.app.processEvents()

    def open(self, key=Qt.Key.Key_Z):
        self.key(key)
        self.assertIsNotNone(self.window._inline)
        return self.window._inline

    def test_z_x_selected_and_switch_commits_like_tab(self):
        ed = self.open()
        self.assertEqual(ed.mid, self.cn.id)
        ed.edit.setText("修改")
        self.window.activateWindow()
        self.player.setFocus()
        self.app.processEvents()
        self.key(Qt.Key.Key_X)
        self.assertIs(self.window._inline, ed)
        self.assertEqual(ed.mid, self.jp.id)
        self.assertEqual(self.cn.text, "修改")

    def test_no_selection_uses_shown_group_before_list_current(self):
        self.group("other", start=5000)
        self.window.current_gid = "other"
        self.watch.selected = None
        self.assertEqual(self.open(Qt.Key.Key_X).mid, self.jp.id)

    def test_no_shown_group_uses_list_and_seeks_group_start(self):
        self.watch.selected = None
        self.player.time = 9
        self.assertEqual(self.open().mid, self.cn.id)
        self.player.seek.assert_called_once_with(1.0)

    def test_overlapping_groups_use_list_current(self):
        self.group("other")
        self.watch.selected = None
        self.window.current_gid = "g"
        self.watch.owners.append({"group": "other", "member": "m_other", "kind": "base"})
        self.watch.active = Mock(return_value=[0, 1, 2])
        self.assertEqual(self.open().mid, self.cn.id)

    def test_missing_lane_creates_empty_dialogue_and_keeps_comment(self):
        self.project.pinyin_mode = "off"
        for key, lane in ((Qt.Key.Key_X, "original"), (Qt.Key.Key_Z, "translation")):
            with self.subTest(lane=lane):
                member = self.jp if lane == "original" else self.cn
                member.role = "comment"
                before = set(self.doc.members)
                self.key(key)
                ed = self.window._inline
                self.assertIsNotNone(ed)
                added = set(self.doc.members) - before
                self.assertEqual(added, {ed.mid})
                created = self.doc.members[ed.mid]
                self.assertEqual((created.lane, created.role, created.text), (lane, "dialogue", ""))
                self.assertEqual(self.doc.members[member.id].role, "comment")
                self.assertTrue(ed.edit.hasFocus())
                self.close_inline()
                self.session.editor.history.undo()
                self.assertEqual(set(self.doc.members), before)
                self.doc.members[member.id].role = "dialogue"

    def test_modifiers_and_ctrl_z_undo(self):
        for modifier in (Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.AltModifier,
                         Qt.KeyboardModifier.ShiftModifier):
            self.key(Qt.Key.Key_X, modifier)
            self.assertIsNone(self.window._inline)
        self.session.editor.replace_text(self.cn.id, "修改")
        self.key(Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(self.doc.members[self.cn.id].text, "翻译")
        self.assertIsNone(self.window._inline)

    def test_typing_transform_modal_and_popup_ignore_z_x(self):
        ed = self.open()
        self.key(Qt.Key.Key_X, target=ed.edit)
        self.assertEqual(ed.mid, self.cn.id)
        self.close_inline()
        self.watch.transform = object()
        self.key(Qt.Key.Key_X)
        self.assertIsNone(self.window._inline)
        self.watch.transform = None
        for flags in (Qt.WindowType.Dialog, Qt.WindowType.Popup):
            dialog = QDialog(self.window, flags)
            self.addCleanup(dialog.deleteLater)
            if flags == Qt.WindowType.Dialog:
                dialog.setModal(True)
            dialog.show()
            self.app.processEvents()
            self.key(Qt.Key.Key_X, target=dialog)
            self.assertIsNone(self.window._inline)
            dialog.close()
            self.app.processEvents()

    def test_escape_from_input_raw_and_format_button_discards_text(self):
        for control in ("edit", "raw", "button"):
            with self.subTest(control=control):
                ed = self.open()
                ed.edit.setText("不要保存")
                widget = {"edit": ed.edit, "raw": ed.fmt.raw, "button": ed.fmt.btn_b}[control]
                widget.setFocus()
                self.app.processEvents()
                self.key(Qt.Key.Key_Escape, target=widget)
                self.assertIsNone(self.window._inline)
                self.assertTrue(self.window.watch_mode)
                self.assertEqual(self.doc.members[self.cn.id].text, "翻译")
                self.assertTrue(self.player.hasFocus())

    def test_escape_outside_closes_before_mode_exit(self):
        self.open().edit.setText("不要保存")
        self.window.activateWindow()
        self.player.setFocus()
        self.app.processEvents()
        self.key(Qt.Key.Key_Escape)
        self.assertIsNone(self.window._inline)
        self.assertTrue(self.window.watch_mode)
        self.assertEqual(self.cn.text, "翻译")
        self.watch.selected = None
        self.window.set_watch_mode = Mock()
        self.key(Qt.Key.Key_Escape)
        self.window.set_watch_mode.assert_called_once_with(False)

    def test_apply_on_close_keeps_focus_of_clicked_input(self):
        ed = self.open()
        ed.edit.setText("保存")
        other = QLineEdit(self.window)
        self.addCleanup(other.deleteLater)
        other.show()
        self.window.activateWindow()
        other.setFocus()
        self.app.processEvents()
        ed.close()
        self.app.processEvents()
        self.assertTrue(other.hasFocus())
        self.assertEqual(self.doc.members[self.cn.id].text, "保存")
        self.assertIsNone(self.window._inline)

    def test_child_dialog_escape_closes_dialog_first(self):
        ed = self.open()
        dialog = QDialog(ed)
        self.addCleanup(dialog.deleteLater)
        dialog.setModal(True)
        dialog.show()
        self.app.processEvents()
        self.key(Qt.Key.Key_Escape, target=dialog)
        self.assertFalse(dialog.isVisible())
        self.assertIs(self.window._inline, ed)
        self.assertTrue(self.window.watch_mode)

    def test_reading_tool_escape_closes_reading_first(self):
        ed = self.open(Qt.Key.Key_X)
        popup = ReadingPopup("原文", "げんぶん", [], "manual", "confirmed", self.window)
        self.addCleanup(popup.deleteLater)
        self.addCleanup(popup.close)
        popup.popup_at(ed.pos())
        self.app.processEvents()
        self.key(Qt.Key.Key_Escape, target=popup.edit)
        self.assertFalse(popup.isVisible())
        self.assertIs(self.window._inline, ed)
        self.assertTrue(self.window.watch_mode)

    def test_both_watch_modes_open_and_cancel_without_changing_pin(self):
        for pinned in (False, True):
            with self.subTest(pinned=pinned):
                self.window.wave_panel.btn_pin.setChecked(pinned)
                ed = self.open(Qt.Key.Key_X)
                self.assertEqual(ed.mid, self.jp.id)
                self.key(Qt.Key.Key_Escape, target=ed.edit)
                self.assertTrue(self.window.watch_mode)
                self.assertEqual(self.window.wave_panel.pinned(), pinned)

    def test_combo_dropdown_escape_closes_dropdown_first(self):
        ed = self.open()
        # The offscreen font database can be empty; populate the real Qt dropdown.
        ed.fmt.font_box.addItem("Test font")
        ed.fmt.font_box.setFocus()
        self.app.processEvents()
        ed.fmt.font_box.showPopup()
        self.app.processEvents()
        self.assertTrue(ed.fmt.font_box.view().isVisible())
        self.key(Qt.Key.Key_Escape, target=ed.fmt.font_box.view())
        self.assertIs(self.window._inline, ed)
        self.assertTrue(self.window.watch_mode)

    def test_input_palette_and_preview_caption(self):
        ed = self.open(Qt.Key.Key_X)
        self.assertEqual(ed.edit.palette().color(QPalette.ColorRole.Base), QColor("#FFFFFF"))
        self.assertEqual(ed.edit.palette().color(QPalette.ColorRole.Text), QColor("#1F2328"))
        self.assertIn("2px solid #3EA6FF", ed.edit.styleSheet())
        self.assertIn("1px solid #8B949E", ed.edit.styleSheet())
        self.assertTrue(ed.preview_caption.isVisible())
        self.assertEqual(ed.preview_caption.text(), "预览 · 点振假名改读音，拖选汉字加读音")
        self.assertEqual(ed.ruby.grab().toImage().pixelColor(0, 0), QColor("#1B1F24"))
        self.assertEqual(ed.ruby.base_colour, "#C9D1D9")
        self.assertEqual(ed.ruby.ruby_colour, "#79C0FF")
        self.assertIn("正在编辑：", ed.head.text())
        self.assertIn("#3EA6FF", ed.head.text())
        ed.switch_member(self.cn.id)
        self.assertFalse(ed.ruby.isVisible())
        self.assertFalse(ed.preview_caption.isVisible())

    def test_keyboard_and_double_click_have_identical_rect_and_font_size(self):
        spy = QSignalSpy(self.watch.textEditRequested)
        self.watch.edit_member(self.cn.id)
        expected = spy.at(0)
        self.close_inline()
        mapper = self.watch._mapper()
        x, y = mapper.to_osd(150, 125)
        self.watch._double(QPointF(x, y), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.assertEqual(spy.at(1), expected)

    def test_missing_geometry_defers_until_measured(self):
        self.watch.geo.clear()
        self.key(Qt.Key.Key_X)
        self.assertIsNone(self.window._inline)
        self.assertEqual(self.watch._pending, ("edit_member", (self.jp.id,)))
        self.assertTrue(self.watch.request_geometry.called)


if __name__ == "__main__":
    unittest.main()


class WatchFrames(WatchInlineQoL):
    """视图 → 观剧模式选项 → 暂停时显示字幕外框 / 字幕外框颜色… (user 2026-10-14)."""

    def frames(self):
        overlays = []
        self.player.overlay = lambda key, text: overlays.append((key, text))
        self.player.clear_overlay = lambda key: overlays.append((key, None))
        self.player.paused = True
        self.watch.redraw()
        from subkit_app.watch import OV_SELECT
        return [text for key, text in overlays if key == OV_SELECT]

    def test_switch_and_colour(self):
        from subkit_app.watch import ass_rgb
        w = self.window
        self.assertEqual(ass_rgb("#3ea6ff"), "&HFFA63E&")
        self.assertTrue(w.act_watch_frames.isChecked())
        self.assertIn("&HFFA63E&", self.frames()[-1])                 # default blue frame on a paused picture
        w._apply_frame_color("#FF0000")
        self.assertIn("&H0000FF&", self.frames()[-1])
        self.assertEqual(w.settings.value("watch/frame_color"), "#FF0000")
        w.act_watch_frames.setChecked(False)
        self.assertEqual(self.frames()[-1], None)                       # switched off: nothing drawn
        self.assertFalse(w.settings.value("watch/frames", True, type=bool))
        self.assertTrue(w.act_watch_frames.statusTip() and w.act_watch_frame_color.statusTip())


class WatchArrows(WatchInlineQoL):
    """User 2026-10-14: in F10/F11 ← / → jump to the previous / next group, Ctrl+← / → step a frame; F11 stays F11."""

    def test_arrows_swap_in_watch_mode_only(self):
        w = self.window
        calls = []
        w._play_neighbour = lambda d: calls.append(("group", d))
        w._frame = lambda forward: calls.append(("frame", forward))
        for watch in (True, False):
            w.watch_mode = watch
            w._arrow(-1, ctrl=False)
            w._arrow(+1, ctrl=True)
        self.assertEqual(calls, [("group", -1), ("frame", True), ("frame", False), ("group", 1)])

    def test_group_jump_in_f11_keeps_the_waveform_hidden(self):
        w = self.window
        self.group("h", text="下一句", start=4000)
        w.player.play = Mock()
        w.player.seek = Mock()
        shown = []
        w._show_watch_wave = lambda on: shown.append(on)
        for pinned, expected in ((False, []), (True, [True])):
            shown.clear()
            w.wave_panel.pinned = lambda p=pinned: p
            w.select_group("g")
            w._play_neighbour(+1)
            self.assertEqual(w.current_gid, "h")
            self.assertEqual(shown, expected)                 # F11: no waveform pop-up (it read as F10)
