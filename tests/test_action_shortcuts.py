"""Real Qt key routing: shared actions, scene changes and legacy guards."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from tests import test_convert_ui as fixtures
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QLineEdit, QMenu, QPlainTextEdit, QTextEdit
from subkit_app import mainwindow as mw


# Independent inventory of the former _build_shortcuts, including secondary keys.
CASES = [
    ("Space", "_space", ()), ("Ctrl+Space", "toggle_play", ()),
    ("R", "_key_r", ()), ("Ctrl+R", "play_group", ()),
    ("Ctrl+Return", "confirm_next", ()), ("Ctrl+Enter", "confirm_next", ()),
    ("Ctrl+Shift+Return", "_confirm", ()), ("Ctrl+Shift+Enter", "_confirm", ()),
    ("Alt+Up", "_move", ("head_to_prev",)), ("Alt+Down", "_move", ("tail_to_next",)),
    ("Alt+Shift+Up", "_move", ("take_from_prev",)), ("Alt+Shift+Down", "_move", ("take_from_next",)),
    ("Ctrl+Shift+Up", "_move_at_cursor", (True,)), ("Ctrl+Shift+Down", "_move_at_cursor", (False,)),
    ("Ctrl+J", "_on_action", ("merge",)), ("Ctrl+K", "_on_action", ("split",)),
    ("Ctrl+U", "_on_action", ("unlink",)), ("Del", "delete_selected", ()),
    ("Ctrl+Z", "undo", ()), ("Ctrl+Y", "redo", ()), ("Ctrl+Shift+Z", "redo", ()),
    ("C", "_adopt_timing", ()), ("PgUp", "_step_group", (-1,)), ("PgDown", "_step_group", (1,)),
    ("[", "_nudge", ("start", -1)), ("]", "_nudge", ("start", 1)),
    ("Shift+[", "_nudge", ("end", -1)), ("Shift+]", "_nudge", ("end", 1)),
    ("{", "_nudge", ("end", -1)), ("}", "_nudge", ("end", 1)),
    *((k, "_play_key", (k,)) for k in "SQWDE"),
    ("F2", "_f2", ()), ("Ctrl+G", "_regen_current", (False,)),
    ("Ctrl+Shift+G", "_regen_current", (True,)),
    ("F9", "toggle_karaoke_mode", ()), ("F10", "_watch_key", (True,)),
    ("F11", "_watch_key", (False,)), ("Ctrl+Shift+Space", "_set_step_playback", ()),
]
ARROWS = [("Left", False, -1), ("Right", False, 1),
          ("Ctrl+Left", True, -1), ("Ctrl+Right", True, 1)]
WATCH = [("Tab", "set_watch_mode", (False,)), ("Esc", "_escape", ()),
         ("Up", "volume_step", (5,)), ("Down", "volume_step", (-5,)),
         ("Alt+E", "popup", None)]


class ActionShortcutTests(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = fixtures.ConversionUI._dispose

    def setUp(self):
        # These two native QAction connections capture bound methods at build time.
        for method in ("toggle_karaoke_mode", "_set_step_playback"):
            patcher = patch.object(mw.MainWindow, method)
            patcher.start()
            self.addCleanup(patcher.stop)
        fixtures.ConversionUI.setUp(self)
        self.addCleanup(setattr, self.window, "watch_mode", False)
        w = self.window
        self.spies = {}
        methods = {name for _key, name, _args in CASES} | {"_frame", "_play_neighbour", "_escape"}
        for method in methods:
            if method in ("toggle_karaoke_mode", "_set_step_playback"):
                self.spies[method] = getattr(w, method)
            else:
                self.spies[method] = Mock()
                setattr(w, method, self.spies[method])
        w.player.volume_step = self.spies["volume_step"] = Mock()
        w.edit_menu.popup = self.spies["popup"] = Mock()
        w.show()
        w.activateWindow()
        w.player.setFocus()
        self.app.processEvents()

    def scene(self, scene, hidden=True):
        w = self.window
        w.watch_mode = scene == "watch"
        w.action_hub.set_scene(scene)
        w.menuBar().setVisible(not hidden)
        w.activateWindow()
        w.player.setFocus()
        self.app.processEvents()
        self.reset()

    def reset(self):
        for spy in self.spies.values():
            spy.reset_mock()

    def key(self, key, target=None):
        stroke = QKeySequence(key)[0]
        QTest.keyClick(target or self.window.player, Qt.Key(stroke.key()), stroke.keyboardModifiers())
        self.app.processEvents()

    def assert_route(self, key, name, args):
        self.reset()
        self.key(key)
        if name == "_confirm":
            self.spies[name].assert_called_once_with(move=False)
        elif name == "_set_step_playback":
            self.spies[name].assert_called_once_with(self.window.act_step_playback.isChecked())
        elif args is None:
            self.spies[name].assert_called_once()
        else:
            self.spies[name].assert_called_once_with(*args)
        for other, spy in self.spies.items():
            if other != name:
                spy.assert_not_called()

    def test_split_play_actions_own_keys_and_menus_keep_primary_keys(self):
        w = self.window
        for scene in ("main", "watch"):
            self.scene(scene)
            for action_id, key in (("play.toggle", "Space"), ("play.toggle_plain", "Ctrl+Space"),
                                   ("play.group", "R"), ("play.group_typing", "Ctrl+R")):
                self.assertEqual(w.action_hub.action(action_id).shortcuts(), [QKeySequence(key)])
            def menu_actions(menu):
                for action in menu.actions():
                    if action.menu() is not None:
                        yield from menu_actions(action.menu())
                    else:
                        yield action
            actions = list(menu_actions(w.menuBar()))
            for action_id in ("play.toggle", "play.group"):
                self.assertIn(w.action_hub.action(action_id), actions)
            for action_id in ("play.toggle_plain", "play.group_typing"):
                self.assertNotIn(w.action_hub.action(action_id), actions)

    def test_split_handlers_follow_action_identity_after_rebinding(self):
        w = self.window
        for action_id, key in (("play.toggle", "Ctrl+Alt+B"), ("play.toggle_plain", "Ctrl+Alt+N"),
                               ("play.group", "Ctrl+Alt+P"), ("play.group_typing", "Ctrl+Alt+L")):
            w.action_hub.registry.set_keys(action_id, "main", key)
        for scene in ("main", "watch"):
            self.scene(scene)
            for key, name in (("Ctrl+Alt+B", "_space"), ("Ctrl+Alt+N", "toggle_play"),
                              ("Ctrl+Alt+P", "_key_r"), ("Ctrl+Alt+L", "play_group")):
                self.assert_route(key, name, ())
            self.reset()
            for key in ("Space", "Ctrl+Space", "R", "Ctrl+R"):
                self.key(key)
            for spy in self.spies.values():
                spy.assert_not_called()

    def test_watch_keys_inactive_in_main(self):
        self.window.set_watch_mode = self.spies["set_watch_mode"] = Mock()
        self.scene("main")
        for key, _name, _args in WATCH:
            self.key(key, self.window)
        for name in ("set_watch_mode", "_escape", "volume_step", "popup"):
            self.spies[name].assert_not_called()
        for action_id in ("watch.exit", "watch.escape", "watch.volume_up", "watch.volume_down", "watch.edit_menu"):
            action = self.window.action_hub.action(action_id)
            self.assertFalse(action.isEnabled())
            self.assertEqual(action.shortcuts(), [])

    def test_letters_and_space_stay_in_all_text_widgets(self):
        for scene in ("main", "watch"):
            self.scene(scene)
            for widget_type in (QLineEdit, QPlainTextEdit, QTextEdit):
                field = widget_type(self.window)
                field.show()
                field.setFocus()
                self.app.processEvents()
                self.reset()
                QTest.keyClicks(field, "crsqwde ")
                self.app.processEvents()
                text = field.text() if isinstance(field, QLineEdit) else field.toPlainText()
                self.assertEqual(text, "crsqwde ")
                for spy in self.spies.values():
                    spy.assert_not_called()
                self.key("Ctrl+Space", field)
                self.spies["toggle_play"].assert_called_once_with()
                self.key("Ctrl+R", field)
                self.spies["play_group"].assert_called_once_with()
                field.deleteLater()
                self.app.processEvents()

    def test_disabled_and_hidden_menu_actions_retain_keyboard_entry(self):
        w = self.window
        for scene in ("main", "watch"):
            self.scene(scene)
            for key, name, args in CASES:
                if key in ("F9", "F10", "F11", "Ctrl+Shift+Space"):
                    continue  # These were QAction shortcuts already, without the legacy fallback.
                action = next(a for a in w.actions() if QKeySequence(key) in a.shortcuts())
                for state in ("disabled", "hidden"):
                    with self.subTest(scene=scene, key=key, state=state):
                        action.setEnabled(state != "disabled")
                        action.setVisible(state != "hidden")
                        self.assert_route(key, name, args)
                action.setVisible(True)
                action.setEnabled(True)
        # Text widgets retain their own Delete/undo even with the fallback registered.
        for widget_type in (QLineEdit, QPlainTextEdit, QTextEdit):
            field = widget_type(self.window)
            field.show()
            field.setFocus()
            self.app.processEvents()
            w.action_hub.action("group.delete").setEnabled(False)
            w.action_hub.action("edit.undo").setEnabled(False)
            self.reset()
            QTest.keyClicks(field, "abc")
            self.key("Ctrl+Z", field)
            self.key("Del", field)
            self.spies["delete_selected"].assert_not_called()
            self.spies["undo"].assert_not_called()
            field.deleteLater()
            self.app.processEvents()

    def test_scene_switch_rebinds_keys_and_has_no_ambiguous_actions_or_qshortcuts(self):
        w = self.window
        for scene in ("main", "watch", "main", "watch"):
            self.scene(scene)
            bindings = {}
            for action in w.findChildren(QAction):
                for key in action.shortcuts():
                    self.assertIn(action, w.actions())
                    self.assertEqual(action.shortcutContext(), Qt.ShortcutContext.WindowShortcut)
                    if action.isEnabled() and action.isVisible():
                        portable = key.toString(QKeySequence.SequenceFormat.PortableText)
                        self.assertNotIn(portable, bindings, (scene, portable))
                        bindings[portable] = action
            self.assertEqual(w.findChildren(QShortcut), [])
            self.assertFalse(w.action_hub.registry.conflicts(scene))

    def test_modal_and_popup_do_not_dispatch_disabled_actions(self):
        self.scene("main")
        self.window.action_hub.action("group.merge").setEnabled(False)
        modal = QDialog(self.window)
        modal.setModal(True)
        modal.show()
        modal.activateWindow()
        modal.setFocus()
        self.app.processEvents()
        self.reset()
        self.key("Ctrl+J", modal)
        self.spies["_on_action"].assert_not_called()
        modal.close()
        modal.deleteLater()
        self.app.processEvents()
        self.scene("main")
        popup = QMenu(self.window)
        popup.addAction("item")
        popup.popup(self.window.mapToGlobal(self.window.rect().center()))
        self.app.processEvents()
        self.key("Ctrl+J", popup)
        self.spies["_on_action"].assert_not_called()
        popup.close()
        popup.deleteLater()

    def test_set_watch_mode_updates_hub_in_both_fullscreen_variants(self):
        w = self.window
        w.player._hide_timer = QTimer(w.player)
        w.player.pause = Mock()
        self.project.settings = Mock(return_value={})
        for pinned in (False, True):
            w.set_watch_mode(True, pinned=pinned)
            self.app.processEvents()
            self.assertEqual(w.action_hub._scene, "watch")
            self.assertEqual(w.action_hub.action("play.prev_group").shortcut(), QKeySequence("Left"))
            self.assertFalse(w.menuBar().isVisible())
            w.set_watch_mode(False)
            self.app.processEvents()
            self.assertEqual(w.action_hub._scene, "main")
            self.assertEqual(w.action_hub.action("play.prev_group").shortcut(), QKeySequence("Ctrl+Left"))


def route_case(key, name, args):
    def test(self):
        for scene, hidden in (("main", False), ("main", True), ("watch", True)):
            with self.subTest(scene=scene, hidden=hidden):
                self.scene(scene, hidden)
                self.assert_route(key, name, args)
    return test


for index, (key, name, args) in enumerate(CASES):
    setattr(ActionShortcutTests, f"test_key_{index:02d}_{name.lstrip('_')}", route_case(key, name, args))

for index, (key, ctrl, direction) in enumerate(ARROWS):
    def arrow_case(self, key=key, ctrl=ctrl, direction=direction):
        for scene in ("main", "watch"):
            self.scene(scene)
            if (scene == "watch") != ctrl:
                self.assert_route(key, "_play_neighbour", (direction,))
            else:
                self.assert_route(key, "_frame", (direction > 0,))
    setattr(ActionShortcutTests, f"test_arrow_{index}", arrow_case)

for index, (key, name, args) in enumerate(WATCH):
    def watch_case(self, key=key, name=name, args=args):
        self.scene("watch")
        self.window.set_watch_mode = self.spies["set_watch_mode"] = Mock()
        self.assert_route(key, name, args)
    setattr(ActionShortcutTests, f"test_watch_{index}_{name.lstrip('_')}", watch_case)
