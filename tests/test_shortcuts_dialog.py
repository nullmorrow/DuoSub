"""Shortcut editing through real offscreen Qt events and isolated settings."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from tests import test_convert_ui as fixtures
from PySide6.QtCore import Qt
from PySide6.QtGui import QInputMethodEvent, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QPlainTextEdit, QTextEdit, QMessageBox
from subkit_app import mainwindow as mw
from subkit_app.views.shortcuts_dialog import ShortcutsDialog


class ShortcutDialogTests(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group
    select = fixtures.ConversionUI.select
    context_menu = fixtures.ConversionUI.context_menu

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.dialog = self.make_dialog()
        self.addCleanup(self.dialog.deleteLater)

    def make_dialog(self):
        w = self.window
        prefixes = ShortcutsDialog.menu_names(w.menuBar())
        names = {s.id: " → ".join(filter(None, (prefixes.get(s.id), s.text))) for s in w.action_hub.registry}
        return ShortcutsDialog(w.action_hub.registry, w, names=names)

    def choose(self, action_id, scene="main"):
        self.dialog.table.setCurrentItem(self.dialog.rows[action_id, scene])

    def key(self, text, target):
        stroke = QKeySequence(text)[0]
        QTest.keyClick(target, Qt.Key(stroke.key()), stroke.keyboardModifiers())
        self.app.processEvents()

    def record(self, text):
        self.dialog.start_recording()
        self.key(text, self.dialog.recorder)

    def test_listing_names_search_and_fixed(self):
        d = self.dialog
        self.assertEqual([d.table.topLevelItem(i).text(0) for i in range(4)],
                         ["编辑界面", "观剧模式", "K 轴", "弹窗内部（不能改）"])
        self.assertEqual(d.rows["file.export", "main"].text(0), "导出 → 发布用 ASS…")
        self.assertEqual(d.rows["k.auto_line", "k"].text(0), "自动 K 轴 → 这一句")
        self.assertIn(("play.toggle_plain", "main"), d.rows)
        self.assertIn(("play.group_typing", "watch"), d.rows)
        d.search.setText("Ctrl+D")
        self.assertFalse(d.rows["group.duplicate", "main"].isHidden())
        self.assertTrue(d.rows["file.export", "main"].isHidden())
        d.search.setText("自动 K 轴")
        self.assertFalse(d.rows["k.auto_line", "k"].isHidden())
        d.search.clear()
        self.choose("popup.cancel", "popup")
        self.assertFalse(d.rebind.isEnabled())
        d.start_recording()
        self.assertIsNone(d.recorder)

    def test_recording_ime_escape_clear_reserved_and_tap(self):
        d = self.dialog
        d.show()
        d.activateWindow()
        self.app.processEvents()
        self.choose("file.save")
        d.start_recording()
        QApplication.sendEvent(d.recorder, QInputMethodEvent("拼", []))
        self.key("Ctrl+Alt+F6", d.recorder)
        self.assertEqual(d.registry.effective_keys("file.save"), ("Ctrl+S",))
        QApplication.sendEvent(d.recorder, QInputMethodEvent("", []))
        self.key("Esc", d.recorder)
        self.assertIsNone(d.recorder)
        self.assertEqual(d.result(), 0)
        self.record("F1")
        self.assertIn("F1 是保留键，不能分配", d.message.text())
        self.key("Ctrl+Alt+F6", d.recorder)
        self.assertEqual(d.registry.effective_keys("file.save"), ("Ctrl+Alt+F6",))
        self.record("Backspace")
        self.assertEqual(d.registry.effective_keys("file.save"), ())
        self.choose("k.tap", "k")
        self.record("Ctrl+L")
        self.assertIn("single key without modifiers", d.message.text())
        self.key("L", d.recorder)
        self.assertEqual(d.registry.effective_keys("k.tap", "k"), ("L",))
        self.assertIn("这个键在输入框里打字时会被当成文字，不会触发", d.message.text())

    def test_conflict_cross_scene_resets_and_cancel(self):
        d = self.dialog
        self.choose("file.save")
        d.record_key("Ctrl+D")
        self.assertFalse(d.ok.isEnabled())
        self.assertIn("已用于", d.message.text())
        self.assertEqual(d.rows["file.save", "main"].foreground(0).color(),
                         d.rows["group.duplicate", "main"].foreground(0).color())
        d.restore_one()
        self.assertTrue(d.ok.isEnabled())
        self.choose("group.delete")
        self.assertIn("K 轴里优先", d.message.text())
        self.choose("file.save")
        d.record_key("Ctrl+Alt+F6")
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
            d.restore_all()
        self.assertEqual(d.registry.effective_keys("file.save"), ("Ctrl+Alt+F6",))
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes) as confirm:
            d.restore_all()
        self.assertEqual(confirm.call_args.args[2], "把所有快捷键恢复为默认？")
        self.assertEqual(d.registry.effective_keys("file.save"), ("Ctrl+S",))
        d.record_key("Ctrl+Alt+F6")
        d.reject()
        self.assertEqual(self.window.action_hub.registry.effective_keys("file.save"), ("Ctrl+S",))
        self.assertFalse(self.settings.contains("shortcuts/main/file.save"))

    def test_help_buttons_and_f1(self):
        d = self.dialog
        d.show()
        d.activateWindow()
        self.app.processEvents()
        with patch("subkit_app.help.open_manual") as help_mock:
            d.manual.click()
            self.key("F1", d)
            next(b for b in d.buttons.buttons() if b.text() == "帮助").click()
        self.assertEqual(help_mock.call_args_list, [(("keys",),)] * 3)

    def test_ok_live_keys_persistence_restart_and_mode_routing(self):
        w, d = self.window, self.dialog
        edits = (("file.save", "main", "Ctrl+Alt+F6"), ("k.add_rest", "k", "Ctrl+Alt+F7"),
                 ("watch.edit_translation", "watch", "Ctrl+Alt+F8"), ("watch.transform", "watch", "Ctrl+Alt+F5"))
        for action_id, scene, key in edits:
            d.registry.set_keys(action_id, scene, key)
        # Exercise the actual Tools menu handler and accepted dialog path.
        def accept_dialog(dialog):
            dialog.registry = d.registry.copy()
            return dialog.DialogCode.Accepted
        with patch.object(ShortcutsDialog, "exec", accept_dialog):
            w.action_hub.action("tools.shortcuts").trigger()
        self.assertEqual(w.action_hub.action("file.save").shortcut(), QKeySequence("Ctrl+Alt+F6"))
        self.assertIn("Ctrl+Alt+F6", w.action_hub.action("file.save").toolTip())
        self.assertIn("\tCtrl+Alt+F7", w.action_hub.action("k.add_rest").text())
        self.assertIn("Ctrl+Alt+F7", w.wave_panel.btn_kaddrest.toolTip())
        self.assertNotIn("Shift+K", w.wave_panel.btn_kaddrest.toolTip())
        self.assertEqual(w.act_transform.shortcut(), QKeySequence("Ctrl+Alt+F5"))
        self.assertEqual(self.settings.value("shortcuts/main/file.save"), "Ctrl+Alt+F6")
        fresh = mw.MainWindow()
        try:
            self.assertEqual(fresh.action_hub.registry.effective_keys("file.save"), ("Ctrl+Alt+F6",))
            self.assertEqual(fresh.action_hub.action("file.save").shortcut(), QKeySequence("Ctrl+Alt+F6"))
        finally:
            fresh.close()
            fresh.deleteLater()
        self.group("g", "song")
        self.select("g")
        w.watch_mode = True
        self.addCleanup(setattr, w, "watch_mode", False)
        w.watch.selected = "g"
        w.action_hub.set_scene("watch")
        w.menuBar().hide()
        w._k_action = Mock()
        w._watch_inline_lane = Mock()
        w.show()
        w.activateWindow()
        w.player.setFocus()
        self.app.processEvents()
        for key in ("Shift+K", "Z"):
            self.key(key, w.player)
        w._k_action.assert_not_called()
        w._watch_inline_lane.assert_not_called()
        self.key("Ctrl+Alt+F7", w.player)
        w._k_action.assert_called_once_with("add_rest")
        self.key("Ctrl+Alt+F8", w.player)
        w._watch_inline_lane.assert_called_once_with("translation")

    def test_live_fallback_grabs_context_proxy_and_new_k_binding(self):
        w = self.window
        self.group("g", "song")
        self.select("g")
        spy = Mock()
        w.action_hub.bind_shortcuts("file.save", spy)
        w.action_hub.action("file.save").setEnabled(False)
        edited = w.action_hub.registry.copy()
        edited.set_keys("file.save", "main", "Ctrl+Alt+F6")
        edited.set_keys("group.duplicate", "main", "Ctrl+Alt+F9")
        edited.set_keys("k.clear", "k", "Ctrl+Alt+F7")
        w._apply_shortcut_overrides(edited)
        w.show()
        w.activateWindow()
        w.player.setFocus()
        self.app.processEvents()
        self.key("Ctrl+S", w.player)
        spy.assert_not_called()
        self.key("Ctrl+Alt+F6", w.player)
        spy.assert_called_once()
        actions = self.context_menu()
        duplicate = next(a for a in actions if a.text().startswith("复制这一组"))
        self.assertIn("Ctrl+Alt+F9", duplicate.text())
        w._k_action = Mock()
        w.wave_panel.btn_kclear.setEnabled(True)
        self.key("Ctrl+Alt+F7", w.player)
        w._k_action.assert_called_once_with("clear")
        first = w.action_hub.registry.copy()
        first.set_keys("k.add_rest", "k", "Ctrl+Alt+F3")
        w._apply_shortcut_overrides(first)
        second = first.copy()
        second.set_keys("k.add_rest", "k", "Ctrl+Alt+F4")
        w._apply_shortcut_overrides(second)
        for control in (w.wave_panel.btn_kaddrest, w.action_hub.action("k.add_rest")):
            self.assertIn("Ctrl+Alt+F4", control.toolTip())
            self.assertNotIn("Ctrl+Alt+F3", control.toolTip())


class NewKeysTextFieldTests(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = fixtures.ConversionUI._dispose

    def setUp(self):
        fixtures.ConversionUI.setUp(self)

    def test_six_defaults_in_three_text_fields(self):
        w = self.window
        w.show()
        w.activateWindow()
        for widget_type in (QLineEdit, QPlainTextEdit, QTextEdit):
            field = widget_type(w)
            field.show()
            field.setFocus()
            self.app.processEvents()
            try:
                for action_id in ("group.duplicate", "time.set", "time.shift", "group.hide", "group.mark", "file.export"):
                    action = w.action_hub.action(action_id)
                    key = w.action_hub.registry.effective_keys(action_id)[0]
                    spy = Mock()
                    # Replace the native keyboard route only; leave Qt's text-widget handling intact.
                    w.action_hub.bind_shortcuts(action_id, spy)
                    stroke = QKeySequence(key)[0]
                    QTest.keyClick(field, Qt.Key(stroke.key()), stroke.keyboardModifiers())
                    self.app.processEvents()
                    outcome = "action" if spy.called else "text-widget"
                    print(f"TEXT-KEY {widget_type.__name__} {key}: {outcome}")
                    self.assertEqual(spy.call_count, 1)
            finally:
                field.close()
                field.deleteLater()
                self.app.processEvents()
