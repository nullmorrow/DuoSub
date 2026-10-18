"""Shared menu/toolbar actions, legacy wiring and read-only shortcut loading."""

from __future__ import annotations

from contextlib import ExitStack
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from tests import test_convert_ui as fixtures
from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QWidget

from subkit_app.action_hub import ActionHub
from subkit_app.actions import ActionRegistry
from subkit_app import mainwindow as mw
from tests.test_menu_redesign import MENU_IDS, NEW_MENU_IDS


class HubTests(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]

    def test_lazy_identity_all_ids_without_registry_shortcuts(self):
        parent = QWidget()
        try:
            registry = ActionRegistry()
            hub = ActionHub(registry, parent)
            self.assertEqual(parent.findChildren(QAction), [])
            actions = []
            for spec in registry:
                action = hub.action(spec.id)
                self.assertIs(hub.action(spec.id), action)
                self.assertEqual(action.text(), spec.text)
                self.assertEqual(action.shortcuts(), [])
                self.assertIsNone(action.property("help_anchor"))
                self.assertEqual(action.property("action_id"), spec.id)
                actions.append(action)
            self.assertEqual(len(set(actions)), len(list(registry)))
            with self.assertRaises(KeyError):
                hub.action("unknown.command")
            self.assertEqual(len(parent.findChildren(QAction)), len(actions))
        finally:
            parent.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_native_factory_runs_once(self):
        parent = QWidget()
        try:
            native = QAction("native", parent)
            factory = Mock(return_value=native)
            hub = ActionHub(ActionRegistry(), parent)
            self.assertIs(hub.action("view.source_panel", factory=factory), native)
            self.assertIs(hub.action("view.source_panel", factory=factory), native)
            factory.assert_called_once_with()
        finally:
            parent.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_load_and_hub_never_write_settings(self):
        with tempfile.TemporaryDirectory(prefix="action-hub-") as directory:
            path = Path(directory) / "settings.ini"
            settings = QSettings(str(path), QSettings.Format.IniFormat)
            settings.setValue("shortcuts/version", 1)
            settings.setValue("shortcuts/main/file.save", "Ctrl+Alt+S")
            settings.setValue("shortcuts/main/edit.undo", "")
            settings.sync()
            before = path.read_bytes()
            parent = QWidget()
            try:
                with patch.object(settings, "setValue", side_effect=AssertionError("write")), \
                        patch.object(settings, "remove", side_effect=AssertionError("remove")):
                    registry = ActionRegistry()
                    registry.load(settings)
                    hub = ActionHub(registry, parent)
                    for spec in registry:
                        hub.action(spec.id)
                self.assertEqual(registry.effective_keys("file.save"), ("Ctrl+Alt+S",))
                self.assertEqual(registry.effective_keys("edit.undo"), ())
                settings.sync()
                self.assertEqual(path.read_bytes(), before)
            finally:
                parent.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


class WindowHubTests(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    setUp = fixtures.ConversionUI.setUp
    _dispose = fixtures.ConversionUI._dispose

    def menu_actions(self):
        def walk(menu):
            for action in menu.actions():
                if action.menu() is not None:
                    yield from walk(action.menu())
                elif not action.isSeparator():
                    yield action
        return list(walk(self.window.menuBar()))

    def test_all_registered_entries_and_legacy_aliases_share_actions(self):
        w = self.window
        actions = self.menu_actions()
        registered = {a.property("action_id"): a for a in actions if a.property("action_id")}
        self.assertEqual(set(registered), MENU_IDS)
        for action in actions:
            action_id = action.property("action_id")
            if action_id:
                self.assertIs(action, w.action_hub.action(action_id))
        aliases = {
            "act_undo": "edit.undo", "act_redo": "edit.redo", "act_watch": "view.watch_full",
            "act_watch2": "view.watch_timing", "act_karaoke": "view.karaoke",
            "act_step_playback": "play.step_mode", "act_private": "tools.private",
            "act_ask_save": "file.ask_save", "act_project_readings": "project.words",
            "act_replacements": "display.replacements", "act_punctuation": "display.punctuation",
            "act_dialogue_align": "time.dialogue_align", "act_find_passage": "tools.find_passage",
            "act_brackets": "tools.brackets_to_text", "act_shift_groups": "time.shift",
            "act_transform": "watch.transform", "act_convert_song": "group.to_song",
            "act_convert_dialogue": "group.to_dialogue", "act_convert_sign": "group.to_sign",
            "act_remove_episode": "episode.remove", "act_docks_right": "view.dock_reset",
            "act_watch_speaker": "watch.speakers", "act_watch_frames": "watch.frames",
            "act_watch_frame_color": "watch.frame_color", "act_hide_period": "display.hide_period",
        }
        for name, action_id in aliases.items():
            self.assertIs(getattr(w, name), w.action_hub.action(action_id), name)
        for key in ("left", "right", "top", "bottom", "alt", "restore"):
            self.assertIs(w.position_actions[key], w.action_hub.action("position." + key))
        self.assertIs(w.batch_actions["5.1"], w.action_hub.action("episode.batch"))
        self.assertIs(w.batch_actions["5.2"], w.action_hub.action("episode.batch_bilingual"))
        for value in vars(w).values():
            values = value.values() if isinstance(value, dict) else (value,)
            for action in values:
                if isinstance(action, QAction) and action.property("action_id"):
                    self.assertIs(action, w.action_hub.action(action.property("action_id")))
        for action_id, dock in (("view.source_panel", w.dock_source), ("view.people_panel", w.people)):
            shared = w.action_hub.action(action_id)
            self.assertIs(shared, dock.toggleViewAction())
            self.assertIn(shared, actions)
            self.assertIn(shared, w.toolbar.actions())
        punctuation = [a for a in actions if a.property("action_id") == "display.punctuation"]
        self.assertEqual(len(punctuation), 1)
        self.assertIs(punctuation[0], w.act_punctuation)

    def test_punctuation_refresh_keeps_identity_single_connection_and_legacy_states(self):
        w = self.window
        w.project.settings = Mock(return_value={})
        action = w.act_punctuation
        with patch.object(mw, "PunctuationDialog") as dialog:
            for _ in range(3):
                w._refresh_punctuation_profiles()
                self.assertIs(w.punctuation_profiles_menu.actions()[-1], action)
                dialog.reset_mock()
                action.trigger()
                dialog.assert_called_once()
                dialog.return_value.exec.assert_called_once_with()
                dialog.return_value.deleteLater.assert_called_once_with()
        w.project = None
        w.display_menu.aboutToShow.emit()
        self.assertTrue(action.isEnabled())
        w.punctuation_profiles_menu.aboutToShow.emit()
        self.assertFalse(action.isEnabled())
        w.punctuation_profiles_menu.aboutToHide.emit()
        self.assertFalse(action.isEnabled())

    def test_native_dock_check_and_enable_states_still_follow_layout(self):
        w = self.window
        w.show()
        self.app.processEvents()
        for layout in ("list", "columns", "karaoke", "video"):
            w.set_edit_layout(layout)
            self.app.processEvents()
            for action_id, dock in (("view.source_panel", w.dock_source), ("view.people_panel", w.people)):
                action = w.action_hub.action(action_id)
                self.assertEqual(action.isEnabled(), layout != "karaoke")
                self.assertEqual(action.isChecked(), dock.isVisible())
                if action.isEnabled():
                    visible = dock.isVisible()
                    action.trigger()
                    self.app.processEvents()
                    self.assertEqual(dock.isVisible(), not visible)
                    self.assertEqual(action.isChecked(), dock.isVisible())

    def test_migrated_actions_use_registry_keys_without_future_bindings(self):
        hub = self.window.action_hub
        for action_id in hub._shortcuts:
            self.assertEqual([key.toString() for key in hub.action(action_id).shortcuts()],
                             list(hub.registry.effective_keys(action_id, "main")), action_id)
        for action_id in ("time.shift", "watch.transform", "group.duplicate", "group.hide", "group.mark",
                          "time.set", "file.export"):
            self.assertEqual([k.toString() for k in hub.action(action_id).shortcuts()],
                             list(hub.registry.effective_keys(action_id, "watch" if action_id == "watch.transform" else "main")), action_id)

    def test_window_loads_overrides_read_only_and_applies_migrated_keys(self):
        self.settings.setValue("shortcuts/version", 1)
        self.settings.setValue("shortcuts/main/file.save", "Ctrl+Alt+S")
        self.settings.setValue("shortcuts/main/edit.undo", "")
        before = {k: self.settings.value(k) for k in self.settings.allKeys() if k.startswith("shortcuts/")}
        original_set = self.settings.setValue

        def set_existing_preference(key, value):
            self.assertFalse(key.startswith("shortcuts/"), key)
            original_set(key, value)

        with patch.object(self.settings, "setValue", side_effect=set_existing_preference), \
                patch.object(self.settings, "remove", side_effect=AssertionError("remove")), \
                patch.object(ActionRegistry, "save", side_effect=AssertionError("save")):
            w = mw.MainWindow()
            try:
                self.assertEqual(w.action_hub.registry.effective_keys("file.save"), ("Ctrl+Alt+S",))
                self.assertEqual(w.action_hub.registry.effective_keys("edit.undo"), ())
                self.assertEqual(w.action_hub.action("file.save").shortcut().toString(), "Ctrl+Alt+S")
            finally:
                w.close()
                w.deleteLater()
                self.app.processEvents()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertEqual({k: self.settings.value(k) for k in before}, before)

    def test_each_registered_menu_command_calls_legacy_handler(self):
        # Expected destinations are independent of the construction helper.
        direct = {
            "file.new": "new_project", "file.open": "open_project_dialog", "file.save": "save",
            "file.export": "export", "file.export_season": "export_season",
            "file.export_review_list": "_export_review_list", "file.quit": "close",
            "project.template": "change_template", "project.pinyin_settings": "_pinyin_settings",
            "project.ruby_settings": "_ruby_settings", "project.words": "_project_readings",
            "tools.passage_library": "_passage_library", "tools.clear_cache": "_clear_layout_cache",
            "edit.undo": "undo", "edit.redo": "redo", "watch.transform": "_start_transform",
            "edit.delete_member": "delete_member", "group.delete": "delete_selected",
            "episode.add": "add_episode", "episode.remove": "remove_episode", "episode.merge": "merge",
            "episode.reanalyse": "reanalyze", "episode.source_copies": "_source_copies",
            "episode.video_links": "_video_links", "tools.brackets_to_text": "brackets_to_text",
            "tools.check": "check", "tools.check_fonts": "check_episode_fonts",
            "tools.font_library": "_choose_font_library", "tools.check_selfref": "check_self_references",
            "tools.export_review": "export_review", "tools.import_review": "import_review_answers",
            "tools.find_passage": "_find_passages", "time.autotime": "_autotime",
            "time.dialogue_align": "_start_dialogue_align", "play.toggle": "toggle_play",
            "play.group": "play_group", "group.confirm_next": "confirm_next",
            "watch.frame_color": "_choose_frame_color", "view.dock_reset": "_dock_panels_right",
            "display.punctuation": "_open_punctuation_menu", "display.replacements": "_open_replacements",
            "help.manual": "show_help", "help.about": "show_about", "view.karaoke": "toggle_karaoke_mode",
        }
        routed = {
            "time.shift": ("_shift_groups_dialog", (["selected"],)),
            "tools.record_passage": ("_record_passage", (["selected"],)),
            "episode.batch": ("_batch", ("5.1",)), "episode.batch_bilingual": ("_batch", ("5.2",)),
            "view.watch_full": ("_watch_key", (False,)), "view.watch_timing": ("_watch_key", (True,)),
            "time.unify_original": ("_unify_lane_times", ("original",)),
            "time.unify_translation": ("_unify_lane_times", ("translation",)),
        }
        for key in ("head_to_prev", "tail_to_next", "take_from_prev", "take_from_next"):
            routed["group." + key] = ("_move", (key,))
        for key in ("merge", "split", "unlink"):
            routed["group." + key] = ("_on_action", (key,))
        for key in ("song", "dialogue", "sign"):
            routed["group.to_" + key] = ("_convert_groups", (["selected"], key))
        for key in ("left", "right", "top", "bottom", "alt", "restore"):
            routed["position." + key] = ("_position_action", (key, ["selected"]))
        for key in "sqwde":
            routed["play." + key] = ("_play_key", (key.upper(),))
        direct["k.parameters"] = "_k_parameters"
        for key, signal in (("tap", "start"), ("import", "import"), ("adopt", "adopt"),
                            ("add_rest", "add_rest"), ("remove_rest", "remove_rest"),
                            ("clear", "clear"), ("confirm", "confirm"), ("mapping", "mapping"),
                            ("english", "english"), ("suggest_onset", "suggest")):
            routed["k." + key] = ("_k_action", (signal,))
        for scope in ("line", "selected", "song"):
            routed["k.auto_" + scope] = ("_k_action", ("auto:" + scope,))
            for kind in ("redistribute", "align_times", "fade"):
                routed[f"k.{kind}_{scope}"] = ("_k_action", (f"{kind}:{scope}",))
        toggled = {"tools.private": "_set_private_enabled", "play.step_mode": "_set_step_playback",
                   "watch.speakers": "_speaker_display_changed", "watch.frames": "_frame_display_changed",
                   "display.hide_period": "_toggle_hide_period"}
        methods = set(direct.values()) | {v[0] for v in routed.values()} | set(toggled.values())
        with ExitStack() as stack:
            spies = {name: stack.enter_context(patch.object(mw.MainWindow, name)) for name in methods}
            w = mw.MainWindow()
            try:
                stack.enter_context(patch.object(w.table, "selected_gids", return_value=["selected"]))
                for name in ("k", "kimport", "kadopt", "kaddrest", "kdelrest", "kclear", "kconfirm", "kmap", "kenglish", "kparams"):
                    getattr(w.wave_panel, "btn_" + name).setEnabled(True)
                for action_id in (*direct, *routed, *toggled):
                    with self.subTest(action_id=action_id):
                        for spy in spies.values():
                            spy.reset_mock()
                        action = w.action_hub.action(action_id)
                        action.setEnabled(True)
                        action.setVisible(True)
                        action.trigger()
                        if action_id in direct:
                            spies[direct[action_id]].assert_called_once_with()
                        elif action_id in routed:
                            name, args = routed[action_id]
                            spies[name].assert_called_once_with(*args)
                        else:
                            spies[toggled[action_id]].assert_called_once_with(action.isChecked())
                with patch.object(mw.app_help, "open_manual") as opened:
                    w.action_hub.action("help.shortcuts").trigger()
                    opened.assert_called_once_with("keys")
                    opened.reset_mock()
                    w.action_hub.action("help.changes").trigger()
                    opened.assert_called_once_with("changes")
                follow = w.action_hub.action("play.follow_list")
                follow.trigger()
                self.assertEqual(w._follow, follow.isChecked())
                ask = w.action_hub.action("file.ask_save")
                ask.trigger()
                self.assertEqual(w.settings.value("save/ask_on_close", type=bool), ask.isChecked())
                covered = set(direct) | set(routed) | set(toggled) | {
                    "help.shortcuts", "help.changes", "play.follow_list", "file.ask_save",
                    "view.source_panel", "view.people_panel"}
                self.assertEqual(covered, MENU_IDS - NEW_MENU_IDS)
            finally:
                w.deleteLater()
                self.app.processEvents()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
