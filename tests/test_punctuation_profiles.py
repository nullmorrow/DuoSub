"""Profile files and the quick-switch/new-project wiring, on offscreen Qt."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from subkit.core.display import PunctuationRules
from subkit.core.punctuation_profiles import (FORMAT, PunctuationProfileStore, ProfileError,
                                              ProfileConflictError, apply, create_profile, matches)
from subkit.core.workspace import Project
from tests import test_punctuation_menu as menu_tests


def standard(name="标准"):
    return create_profile(name, PunctuationRules(comma="spaces", tilde="half", dash="em",
        fonts=(("comma", "Arial"), ("long_vowel", "Arial")), scope_chars=("dialogue", "song"),
        scope_fonts=("song", "other")), PunctuationRules(quotes="nested", question="full"))


class ProfileFiles(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp(prefix="punctuation-profiles-", dir=os.environ["TEMP"]))
        self.store = PunctuationProfileStore(self.path / "a")

    def test_roundtrip_scopes_fonts_sorted_and_detached(self):
        first = standard()
        self.assertEqual(self.store.list(), [])
        self.store.save(first)
        self.store.save(standard("A"))
        self.assertEqual([p["name"] for p in self.store.list()], ["A", "标准"])
        self.assertEqual(self.store.load("标准"), first)
        loaded = self.store.load("标准")
        loaded["translation"]["scope_fonts"].clear()
        self.assertEqual(self.store.load("标准"), first)
        text = (self.store.directory / "标准.json").read_text(encoding="utf-8")
        self.assertIn('"标准"', text)
        self.assertNotIn('"font":', text)
        self.assertEqual(json.loads(text), first)

    def test_conflict_overwrite_and_case_insensitive_names(self):
        self.store.save(standard("Standard"))
        changed = standard("standard")
        changed["original"]["question"] = "half"
        with self.assertRaises(ProfileConflictError):
            self.store.save(changed)
        self.store.save(changed, overwrite=True)
        self.assertEqual(self.store.load("STANDARD")["original"]["question"], "half")
        self.assertEqual(len(self.store.list()), 1)

    def test_rename_delete_and_conflicts(self):
        self.store.save(standard("A"))
        self.store.save(standard("B"))
        with self.assertRaises(ProfileConflictError):
            self.store.rename("A", "B")
        self.assertEqual(self.store.load("A")["name"], "A")
        renamed = self.store.rename("A", "改名")
        self.assertEqual(renamed["name"], "改名")
        self.assertFalse((self.store.directory / "A.json").exists())
        self.assertEqual(self.store.rename("B", "b")["name"], "b")
        self.assertEqual(self.store.load("b")["name"], "b")
        self.store.delete("改名")
        with self.assertRaises(FileNotFoundError):
            self.store.load("改名")

    def test_export_import_across_directories_conflict_override_and_rename(self):
        original = self.store.save(standard())
        exported = self.path / "export.json"
        self.store.export_to("标准", exported)
        other = PunctuationProfileStore(self.path / "b")
        self.assertEqual(other.import_from(exported), original)
        with self.assertRaises(ProfileConflictError):
            other.import_from(exported)
        self.assertEqual(other.import_from(exported, overwrite=True), original)
        self.assertEqual(other.import_from(exported, name="另一套")["name"], "另一套")
        self.assertEqual(json.loads(exported.read_text(encoding="utf-8")), original)

    def test_invalid_format_version_structure_rules_and_encoding(self):
        invalid = [[], None, {}, {"format": "wrong", "version": 1},
                   dict(standard(), version=2), dict(standard(), version=True),
                   dict(standard(), original=[]), dict(standard(), translation=None)]
        missing = standard()
        del missing["original"]
        invalid.append(missing)
        for key, value in (("dash", "bad"), ("comma", 1), ("period", "yes"),
                           ("fonts", []), ("fonts", {"unknown": "Arial"}),
                           ("scope_fonts", ["everything"]), ("scope_chars", "dialogue"),
                           ("garbage", True)):
            profile = standard()
            profile["translation"][key] = value
            invalid.append(profile)
        path = self.path / "bad.json"
        for value in invalid:
            with self.subTest(value=value):
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ProfileError):
                    self.store.import_from(path)
        for content in (b"garbage", b"\xff\xfe", b'{"format":'):
            path.write_bytes(content)
            with self.assertRaisesRegex(ProfileError, "JSON"):
                self.store.import_from(path)
        self.assertEqual(self.store.list(), [])

    def test_invalid_names_never_write_outside_store(self):
        for name in ("", " ", "../outside", "a/b", "a\\b", "a:b", "name.", " leading", "CON", "nul.json", "a\n"):
            with self.subTest(name=name), self.assertRaises(ProfileError):
                self.store.save(dict(standard(), name=name))
        self.assertFalse(self.store.directory.exists())

    def test_legacy_comparison_and_migration_preserves_dialogue_font_scope(self):
        old = {"comma": True, "quotes": True, "font": "Arial"}
        profile = {"format": FORMAT, "version": 1, "name": "旧方案", "translation": old, "original": {}}
        saved = self.store.save(profile)
        settings = {"display": {"punctuation": {"translation": old, "original": {}}}}
        self.assertTrue(matches(settings, saved))
        self.assertEqual(saved["translation"]["scope_fonts"], ["dialogue"])
        apply(settings, saved)
        self.assertTrue(matches(settings, profile))
        settings["display"]["punctuation"]["translation"]["scope_fonts"] = ["dialogue", "song", "other"]
        self.assertFalse(matches(settings, profile))

    def test_apply_replaces_both_lanes_and_preserves_unrelated_settings(self):
        settings = {"ruby": {"gap": 2}, "display": {"hide_final_period": True,
            "punctuation": {"translation": {"font": "Old"}, "original": {"period": True}}}}
        profile = standard()
        apply(settings, profile)
        self.assertEqual(settings["display"]["punctuation"], {lane: profile[lane] for lane in ("translation", "original")})
        self.assertEqual(settings["display"]["punctuation_profile"], "标准")
        self.assertTrue(settings["display"]["hide_final_period"])
        self.assertEqual(settings["ruby"], {"gap": 2})
        settings["display"]["punctuation"]["original"]["question"] = "half"
        self.assertEqual(profile["original"]["question"], "full")

    def test_missing_default_leaves_settings_untouched(self):
        settings = {"display": {"hide_final_period": True}}
        before = copy.deepcopy(settings)
        for name in (None, "", "missing"):
            self.assertFalse(self.store.apply_default(settings, name))
            self.assertEqual(settings, before)

    def test_misnamed_local_file_has_clear_error(self):
        self.store.save(standard())
        (self.store.directory / "标准.json").rename(self.store.directory / "不同名称.json")
        with self.assertRaisesRegex(ProfileError, "名称与文件名"):
            self.store.list()


class ProfileMenu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = menu_tests.QApplication.instance() or menu_tests.QApplication([])
        if cls.app.platformName() != "offscreen":
            raise RuntimeError("Profile menu tests require offscreen Qt")

    def setUp(self):
        menu_tests.PunctuationMenu.setUp(self)

    def tearDown(self):
        menu_tests.PunctuationMenu.tearDown(self)

    def test_settings_file_directory_empty_menu_and_existing_menu_entry(self):
        w = self.window
        self.assertEqual(w.punctuation_profiles.directory, self.path / "punctuation-profiles")
        actions = w.punctuation_profiles_menu.actions()
        self.assertEqual([a.text() for a in actions], ["（还没有方案）", "", "标点和空格…"])
        self.assertFalse(actions[0].isEnabled())
        self.assertTrue(actions[1].isSeparator())
        with patch.object(menu_tests.mw.PunctuationDialog, "exec", return_value=0) as opened:
            actions[-1].trigger()
            opened.assert_called_once()
        self.assertEqual(actions[-1].property("help_anchor"), "display-rules")

    def test_one_click_switch_saves_refreshes_preview_and_leaves_document(self):
        w = self.window
        w.punctuation_profiles.save(standard())
        w.punctuation_profiles.save(create_profile("番剧", PunctuationRules(), PunctuationRules(dash="half")))
        before = copy.deepcopy(self.ep.document)
        w._refresh_punctuation_profiles()
        with patch.object(self.session, "request_preview") as preview:
            w.punctuation_profile_actions["番剧"].trigger()
            preview.assert_called_once_with()
        w._set_punctuation("original", "dash", "em")
        self.assertEqual(w.punctuation_profile_actions["番剧"].text(), "番剧（已修改）")
        self.assertFalse(w.punctuation_profile_actions["番剧"].isChecked())
        with patch.object(self.session, "request_preview") as preview:
            w.punctuation_profile_actions["标准"].trigger()
            preview.assert_called_once_with()
        self.assertTrue(w.punctuation_profile_actions["标准"].isChecked())
        self.assertEqual(PunctuationRules.from_settings(self.project.settings()["display"]["punctuation"]["translation"]).comma, "spaces")
        self.assertEqual(PunctuationRules.from_settings(self.project.settings()["display"]["punctuation"]["original"]).quotes, "nested")
        self.assertTrue(matches(Project.open(self.project.root).settings(), standard()))
        self.assertEqual(self.ep.document, before)
        self.assertFalse(self.session.dirty)

    def test_default_modified_and_equal_radio_states(self):
        w = self.window
        w.punctuation_profiles.save(standard())
        w.punctuation_profiles.save(standard("相同"))
        self.settings.setValue("display/default_punctuation_profile", "标准")
        apply(self.project.data.setdefault("settings", {}), standard("相同"))
        w._apply_rules()
        self.assertEqual(w.punctuation_profile_actions["标准"].text(), "标准（默认）")
        self.assertTrue(w.punctuation_profile_actions["相同"].isChecked())
        self.assertEqual(sum(a.isChecked() for a in w.punctuation_profile_actions.values()), 1)
        self.assertTrue(w.punctuation_profile_actions["标准"].actionGroup().isExclusive())
        self.project.data["settings"]["display"]["punctuation_profile"] = "标准"
        w._set_punctuation("original", "question", "half")
        self.assertEqual(w.punctuation_profile_actions["标准"].text(), "标准（已修改）（默认）")
        self.assertFalse(any(a.isChecked() for a in w.punctuation_profile_actions.values()))

    def test_profiles_deleted_or_changed_refresh_when_menu_opens(self):
        w = self.window
        w.punctuation_profiles.save(standard())
        w._select_punctuation_profile("标准")
        before = copy.deepcopy(self.project.data["settings"])
        w.punctuation_profiles.delete("标准")
        w.punctuation_profiles_menu.aboutToShow.emit()
        self.assertEqual(w.punctuation_profile_actions, {})
        self.assertEqual(self.project.data["settings"], before)
        w.punctuation_profiles.save(standard("新方案"))
        w.punctuation_profiles_menu.aboutToShow.emit()
        self.assertTrue(w.punctuation_profile_actions["新方案"].isChecked())

    def test_no_project_disables_switching(self):
        w = self.window
        w.punctuation_profiles.save(standard())
        w.project = None
        w._refresh_punctuation_profiles()
        self.assertFalse(w.punctuation_profile_actions["标准"].isEnabled())
        self.assertFalse(w.punctuation_profiles_menu.actions()[-1].isEnabled())

    def test_save_failure_restores_settings_and_does_not_request_preview(self):
        w = self.window
        w.punctuation_profiles.save(standard())
        before = copy.deepcopy(self.project.data["settings"])
        with patch.object(self.project, "save", side_effect=OSError("cannot save")), \
                patch.object(menu_tests.mw.QMessageBox, "warning") as warning, \
                patch.object(self.session, "request_preview") as preview:
            w._select_punctuation_profile("标准")
            warning.assert_called_once()
            preview.assert_not_called()
        self.assertEqual(self.project.data["settings"], before)

    def test_profile_popup_f1_routes_to_display_rules(self):
        w = self.window
        w.punctuation_profiles.save(standard())
        w._refresh_punctuation_profiles()
        popup = w.punctuation_profiles_menu
        popup.setActiveAction(w.punctuation_profile_actions["标准"])
        with patch.object(menu_tests.QApplication, "activePopupWidget", return_value=popup):
            self.assertEqual(menu_tests.help_ui.resolve_anchor(w), "display-rules")

    def test_registry_settings_use_app_data_directory(self):
        with patch.object(self.settings, "fileName", return_value="HKEY_CURRENT_USER\\Software\\SubKit\\DuoSub"), \
                patch.object(menu_tests.mw.QStandardPaths, "writableLocation", return_value=str(self.path)) as location, \
                patch.object(menu_tests.mw, "MpvWidget", menu_tests.PlayerDouble), \
                patch.object(menu_tests.mw, "QSettings", return_value=self.settings), \
                patch.object(menu_tests.mw.MainWindow, "_open_last", Mock()):
            window = menu_tests.mw.MainWindow()
        try:
            self.assertEqual(window.punctuation_profiles.directory, self.path / "punctuation-profiles")
            location.assert_called_once_with(menu_tests.mw.QStandardPaths.StandardLocation.AppDataLocation)
        finally:
            window.close()
            window.deleteLater()
            self.app.processEvents()
            menu_tests.QCoreApplication.sendPostedEvents(None, menu_tests.QEvent.Type.DeferredDelete)

    def test_read_errors_and_missing_selected_profile_report_without_mutation(self):
        w = self.window
        before = copy.deepcopy(self.project.data["settings"])
        with patch.object(menu_tests.mw.QMessageBox, "warning") as warning:
            w._select_punctuation_profile("missing")
            warning.assert_called_once()
        self.assertEqual(self.project.data["settings"], before)
        w.punctuation_profiles.directory.mkdir()
        (w.punctuation_profiles.directory / "bad.json").write_text("garbage", encoding="utf-8")
        w._refresh_punctuation_profiles()
        self.assertIn("读取标点方案失败", w.statusBar().currentMessage())

    def test_new_project_default_and_missing_default(self):
        w = self.window
        w.punctuation_profiles.save(standard())
        for default in ("标准", "missing", ""):
            with self.subTest(default=default):
                folder = self.path / ("new-" + (default or "empty"))
                dialog = Mock()
                dialog.exec.return_value = menu_tests.PunctuationDialog.DialogCode.Accepted
                dialog.folder.text.return_value = str(folder)
                dialog.name.text.return_value = "new"
                dialog.original.currentData.return_value = "ja"
                dialog.translation.currentData.return_value = "zh-Hant"
                dialog.editor.styles_text.return_value = menu_tests.default_styles_ass()
                dialog.editor.scheme.return_value = {}
                dialog.dict_from.text.return_value = ""
                self.settings.setValue("display/default_punctuation_profile", default)
                with patch.object(menu_tests.mw, "NewProjectDialog", return_value=dialog) as factory, \
                        patch.object(w, "open_project") as opened:
                    factory.DialogCode = menu_tests.PunctuationDialog.DialogCode
                    w.new_project()
                    opened.assert_called_once_with(folder)
                settings = Project.open(folder).settings()
                if default == "标准":
                    self.assertTrue(matches(settings, standard()))
                    self.assertEqual(settings["display"]["punctuation_profile"], "标准")
                else:
                    self.assertNotIn("punctuation_profile", settings["display"])
