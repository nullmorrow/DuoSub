"""Real Qt mode-key events, registry overrides and unchanged input guards."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from tests import test_convert_ui as fixtures
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QMenu, QPlainTextEdit, QTextEdit


# ID, scene, original key, override, destination, arguments.
MODES = (
    ("watch.edit_translation", "watch", "Z", "Ctrl+Alt+F3", "inline", ("translation",)),
    ("watch.edit_original", "watch", "X", "Ctrl+Alt+F4", "inline", ("original",)),
    ("watch.transform", "watch", "T", "Ctrl+Alt+F5", "transform", False),
    ("watch.transform_line", "watch", "Alt+T", "Ctrl+Alt+F6", "transform", True),
    ("k.add_rest", "k", "Shift+K", "Ctrl+Alt+F7", "k", ("add_rest",)),
    ("k.remove_rest", "k", "Del", "Ctrl+Alt+F8", "k", ("remove_rest",)),
    ("k.adopt", "k", "Return", "Ctrl+Alt+F9", "k", ("adopt",)),
    ("k.cancel", "k", "Esc", None, "cancel", ()),
)


class ModeKeyTests(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = fixtures.ConversionUI._dispose

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        w = self.window
        self.addCleanup(setattr, w, "watch_mode", False)
        self.addCleanup(setattr, w.watch, "transform", None)
        self.addCleanup(setattr, w, "_export_job", None)
        w.watch_mode = True
        w.watch.selected = "g"
        w.action_hub.set_scene("watch")
        w.menuBar().hide()
        self.spies = {
            "inline": Mock(), "transform": Mock(), "k": Mock(), "cancel": Mock(),
        }
        w._escape = Mock()
        w._watch_inline_lane = self.spies["inline"]
        w._start_transform = self.spies["transform"]
        w._k_action = self.spies["k"]
        w.wave.clear_suggestions = self.spies["cancel"]
        w.wave.selected_rest = Mock(return_value=("original", 0))
        w.wave.k_suggestions = ("missing", [], "syllable")
        w.show()
        w.activateWindow()
        w.player.setFocus()
        self.app.processEvents()

    def reset(self):
        for spy in self.spies.values():
            spy.reset_mock()

    def key(self, key, target=None, repeat=False, release=False):
        stroke = QKeySequence(key)[0]
        target = target or self.window.player
        if repeat or release:
            event = QKeyEvent(QEvent.Type.KeyRelease if release else QEvent.Type.KeyPress,
                              stroke.key(), stroke.keyboardModifiers(), "", repeat)
            QApplication.sendEvent(target, event)
        else:
            QTest.keyClick(target, Qt.Key(stroke.key()), stroke.keyboardModifiers())
        self.app.processEvents()

    def assert_route(self, case, key):
        self.reset()
        self.key(key)
        _id, _scene, _default, _override, destination, args = case
        if destination == "transform":
            self.spies[destination].assert_called_once_with(only_line=args)
        else:
            self.spies[destination].assert_called_once_with(*args)
        for name, spy in self.spies.items():
            if name != destination:
                spy.assert_not_called()

    def override(self, case):
        action_id, scene, _default, override, *_ = case
        if override is not None:
            self.window.action_hub.registry.set_keys(action_id, scene, override)

    def test_default_modifier_tolerance_and_overrides_are_exact(self):
        for case, keys in ((MODES[2], ("T", "Shift+T")),
                           (MODES[3], ("Alt+T", "Alt+Shift+T")),
                           (MODES[6], ("Return", "Enter", "Shift+Return", "Shift+Enter")),
                           (MODES[7], ("Esc", "Shift+Esc"))):
            for key in keys:
                self.assert_route(case, key)
        for case in MODES[:2] + MODES[4:6]:
            self.reset()
            self.key(("Ctrl+" if case[0] == "k.add_rest" else "Shift+") + case[2])
            for spy in self.spies.values():
                spy.assert_not_called()
        for case in MODES:
            if case[3] is None:
                continue
            self.override(case)
            self.reset()
            self.key("Shift+" + case[3])
            for spy in self.spies.values():
                spy.assert_not_called()
        # Explicitly storing the default is still an override, with exact modifiers.
        self.window.action_hub.registry.set_keys("watch.transform", "watch", "T")
        self.reset()
        self.key("Shift+T")
        self.spies["transform"].assert_not_called()

    def test_all_mode_keys_respect_typing_export_modal_and_popup(self):
        w = self.window
        for overridden in (False, True):
            w.action_hub.registry.reset_all()
            if overridden:
                for case in MODES:
                    self.override(case)
            keys = [case[3] if overridden and case[3] else case[2] for case in MODES]
            for widget_type in (QLineEdit, QPlainTextEdit, QTextEdit):
                field = widget_type(w)
                field.show()
                field.setFocus()
                self.app.processEvents()
                self.reset()
                for key in keys:
                    self.key(key, field)
                for spy in self.spies.values():
                    spy.assert_not_called()
                field.close()
                field.deleteLater()
                self.app.processEvents()
            w.activateWindow()
            w.player.setFocus()
            self.app.processEvents()
            w._export_job = object()
            self.reset()
            for key in keys:
                self.key(key)
            for spy in self.spies.values():
                spy.assert_not_called()
            w._export_job = None
            for kind in ("modal", "popup"):
                for key in keys:
                    dialog = QDialog(w) if kind == "modal" else QMenu(w)
                    if kind == "modal":
                        dialog.setModal(True)
                        dialog.show()
                        dialog.activateWindow()
                    else:
                        dialog.addAction("item")
                        dialog.popup(w.mapToGlobal(w.rect().center()))
                    dialog.setFocus()
                    self.app.processEvents()
                    self.assertIsNotNone(QApplication.activeModalWidget() if kind == "modal"
                                         else QApplication.activePopupWidget())
                    self.reset()
                    self.key(key, dialog)
                    for spy in self.spies.values():
                        spy.assert_not_called()
                    dialog.close()
                    dialog.deleteLater()
                    self.app.processEvents()
            w.activateWindow()
            w.player.setFocus()
            self.app.processEvents()

    def test_rest_and_suggestion_guards_and_shortcut_override(self):
        w = self.window
        for overridden in (False, True):
            w.action_hub.registry.reset_all()
            for case in MODES:
                if overridden:
                    self.override(case)
                key = case[3] if overridden and case[3] else case[2]
                stroke = QKeySequence(key)[0]
                event = QKeyEvent(QEvent.Type.ShortcutOverride, stroke.key(), stroke.keyboardModifiers())
                event.ignore()
                QApplication.sendEvent(w.player, event)
                self.assertTrue(event.isAccepted(), case[0])
                self.reset()
                self.key(key, repeat=True)
                if case[4] == "transform":
                    self.spies["transform"].assert_called_once_with(only_line=case[5])
                else:
                    for spy in self.spies.values():
                        spy.assert_not_called()
                self.reset()
            for state in ("transform", "tapper", "no_suggestions", "no_rest"):
                if state == "transform":
                    w.watch.transform = object()
                    w.watch.transform_key = Mock(return_value=True)
                elif state == "tapper":
                    w.tapper.run = {"lane": "original"}
                    w.tapper.enabled = False
                elif state == "no_suggestions":
                    w.wave.k_suggestions = None
                else:
                    w.wave.selected_rest.return_value = None
                self.reset()
                cases = MODES[4:] if state in ("transform", "tapper") else (MODES[6:8] if state == "no_suggestions" else MODES[5:6])
                for case in cases:
                    self.key(case[3] if overridden and case[3] else case[2])
                for spy in self.spies.values():
                    spy.assert_not_called()
                w.watch.transform = None
                w.tapper.run = None
                w.tapper.enabled = True
                w.wave.k_suggestions = ("missing", [], "syllable")
                w.wave.selected_rest.return_value = ("original", 0)

    def test_watch_keys_require_watch_and_transform_requires_selection(self):
        w = self.window
        for overridden in (False, True):
            if overridden:
                for case in MODES:
                    self.override(case)
            w.watch_mode = False
            self.reset()
            for case in MODES[:4]:
                self.key(case[3] if overridden else case[2])
            for spy in self.spies.values():
                spy.assert_not_called()
            w.watch_mode = True
            w.watch.selected = None
            for case in MODES[2:4]:
                self.key(case[3] if overridden else case[2])
            self.spies["transform"].assert_not_called()
            w.watch.selected = "g"

    def test_transform_menu_displays_effective_key_without_qaction_shortcut(self):
        w = self.window
        for key in ("T", "Ctrl+Alt+B", ""):
            w.action_hub.registry.set_keys("watch.transform", "watch", key)
            w._update_transform_action()
            self.assertEqual(w.act_transform.text(), "自由变换")
            self.assertEqual(w.act_transform.shortcuts(), [QKeySequence(key)] if key else [])

    def test_tap_press_release_autorepeat_and_guards_with_rebound_key(self):
        w = self.window
        tapper = w.tapper
        tapper._down, tapper._up = Mock(), Mock()
        for key in ("K", "L"):
            if key == "L":
                w.action_hub.registry.set_keys("k.tap", "k", key)
                self.key("K")
                tapper._down.assert_not_called()
                tapper._up.assert_not_called()
            self.key(key)
            tapper._down.assert_called_once_with()
            tapper._up.assert_called_once_with()
            tapper._down.reset_mock()
            tapper._up.reset_mock()
            self.key(key, repeat=True)
            self.key(key, repeat=True, release=True)
            tapper._down.assert_not_called()
            tapper._up.assert_not_called()
            for prefix in ("Shift+", "Ctrl+", "Alt+"):
                self.key(prefix + key)
                tapper._down.assert_not_called()
                tapper._up.assert_not_called()
            for field_type in (QLineEdit, QPlainTextEdit, QTextEdit):
                field = field_type(w)
                field.show()
                field.setFocus()
                self.app.processEvents()
                self.key(key, field)
                tapper._down.assert_not_called()
                tapper._up.assert_not_called()
                field.deleteLater()
                self.app.processEvents()
            w.activateWindow()
            w.player.setFocus()
            self.app.processEvents()
            w._export_job = object()
            self.key(key)
            tapper._down.assert_not_called()
            tapper._up.assert_not_called()
            w._export_job = None
            for kind in ("modal", "popup"):
                dialog = QDialog(w) if kind == "modal" else QMenu(w)
                if kind == "modal":
                    dialog.setModal(True)
                    dialog.show()
                    dialog.activateWindow()
                else:
                    dialog.addAction("item")
                    dialog.popup(w.mapToGlobal(w.rect().center()))
                dialog.setFocus()
                self.app.processEvents()
                self.key(key, dialog)
                tapper._down.assert_not_called()
                tapper._up.assert_not_called()
                dialog.close()
                dialog.deleteLater()
                self.app.processEvents()
            w.activateWindow()
            w.player.setFocus()
            self.app.processEvents()
        w.action_hub.registry.clear_keys("k.tap", "k")
        self.key("L")
        tapper._down.assert_not_called()
        tapper._up.assert_not_called()


def mode_case(case):
    def test(self):
        self.assert_route(case, case[2])
        if case[3] is not None:
            self.override(case)
            self.reset()
            self.key(case[2])
            for spy in self.spies.values():
                spy.assert_not_called()
            self.assert_route(case, case[3])
    return test


for case in MODES:
    setattr(ModeKeyTests, "test_route_" + case[0].replace(".", "_"), mode_case(case))
