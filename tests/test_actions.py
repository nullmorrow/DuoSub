"""Pure registry and isolated IniFormat shortcut persistence tests (U1)."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import QSettings
from PySide6.QtGui import QKeySequence
from subkit_app.actions import ActionRegistry, ActionSpec, SCENES, SCENE_PRIORITY, is_reserved
from subkit_app.help import HELP_ANCHORS


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = ActionRegistry()

    def test_register_lookup_and_reject_duplicate_or_invalid_id(self):
        registry = ActionRegistry(())
        spec = ActionSpec("test.command", "测试", {"main": ("ctrl+b",)}, "keys")
        registry.register(spec)
        self.assertIs(registry.lookup("test.command"), spec)
        self.assertEqual(spec.keys["main"], ("Ctrl+B",))
        self.assertEqual(tuple(registry), (spec,))
        with self.assertRaises(ValueError):
            registry.register(spec)
        with self.assertRaises(KeyError):
            registry.lookup("missing.command")
        for action_id in ("test", "测试.command", "Test.command", "test/command"):
            with self.subTest(action_id=action_id), self.assertRaises(ValueError):
                ActionSpec(action_id, "测试", {"main": ()}, "keys")

    def test_specs_copy_and_freeze_input_defaults(self):
        keys = {"main": ["Ctrl+B"]}
        spec = ActionSpec("test.command", "测试", keys, "keys")
        keys["main"].append("Ctrl+L")
        self.assertEqual(spec.keys["main"], ("Ctrl+B",))
        with self.assertRaises(TypeError):
            spec.keys["main"] = ()

    def test_scene_names_and_priority(self):
        self.assertEqual(dict(SCENES), {"main": "编辑界面", "watch": "观剧模式", "k": "K 轴", "popup": "弹窗内部（固定）"})
        self.assertEqual(SCENE_PRIORITY, ("popup", "k", "watch", "main"))

    def test_all_explicit_section_three_ids_registered(self):
        design = (ROOT / "docs/ui-redesign/DESIGN.md").read_text(encoding="utf-8")
        section = design.split("## 3. 菜单栏", 1)[1].split("## 4.", 1)[0]
        expected = set()
        for group in re.findall(r"〔([^〕]+)〕", section):
            if not re.match(r"[a-z]+\.", group):
                continue  # Skip the format legend, which is not an action ID.
            parts = group.split("／")
            area = parts[0].rsplit(".", 1)[0]
            for part in parts:
                if "*" not in part:
                    expected.add(part if "." in part else area + "." + part)
        self.assertGreater(len(expected), 75)
        self.assertFalse(expected - {spec.id for spec in self.registry})

    def test_unlabelled_commands_are_registered_but_dynamic_choices_are_not(self):
        expected = {
            "position.left": "左侧说话人", "position.right": "右侧说话人", "position.top": "移到顶部",
            "position.bottom": "移到底部", "position.alt": "底部第二人", "position.restore": "恢复原位置",
            "tools.export_review": "导出审议包（发布前给大模型复查）…", "tools.import_review": "导回审议意见…",
            "tools.passage_library": "重复段落库…", "tools.record_passage": "记为重复段落…",
            "tools.find_passage": "查找并套用…", "tools.private": "私有功能",
            "tools.font_library": "字体库位置…", "tools.clear_cache": "清除排版缓存",
            "display.punctuation": "标点和空格…", "display.replacements": "批量替换…",
            "k.mapping": "对应…", "k.english": "英文按音节拆开", "k.parameters": "参数…",
            "help.manual": "使用说明", "help.shortcuts": "快捷键一览", "help.changes": "版本记录",
            "help.about": "关于双语字幕工坊", "watch.speakers": "显示说话人",
            "watch.frames": "暂停时显示字幕外框", "watch.frame_color": "字幕外框颜色…",
        }
        for action_id, text in expected.items():
            with self.subTest(action_id=action_id):
                self.assertEqual(self.registry.lookup(action_id).text, text)
        for kind in ("redistribute", "align_times", "fade"):
            for scope, text in (("line", "这一句"), ("selected", "选中的句"), ("song", "整首歌")):
                self.assertEqual(self.registry.lookup(f"k.{kind}_{scope}").text, text + ("…" if kind == "fade" else ""))
        texts = {spec.text for spec in self.registry}
        for choice in ("视频优先（推荐）", "列表优先", "左右分栏", "台湾（默认）", "原样（不转换）", "kf"):
            self.assertNotIn(choice, texts)

    def test_help_anchors_all_known_and_existing_in_manual(self):
        manual = (ROOT / "docs/manual/index.html").read_text(encoding="utf-8")
        ids = set(re.findall(r'\bid="([^"]+)"', manual))
        for spec in self.registry:
            with self.subTest(action_id=spec.id):
                self.assertIn(spec.help_anchor, HELP_ANCHORS)
                self.assertIn(spec.help_anchor, ids)
        for action_id, anchor in (("file.save", "save"), ("file.export_season", "export-season"),
                                  ("episode.video_links", "video-links"), ("episode.source_copies", "source-copies"),
                                  ("time.unify_original", "unify-times"), ("play.toggle", "playback"),
                                  ("display.replacements", "display-replacements"), ("tools.export_review", "review-packet")):
            self.assertEqual(self.registry.lookup(action_id).help_anchor, anchor)

    def test_default_keys_including_secondary_and_new_keys(self):
        expected = {
            "file.new": ("Ctrl+Shift+N",), "file.open": ("Ctrl+O",), "file.save": ("Ctrl+S",),
            "file.export": ("Ctrl+E",), "group.duplicate": ("Ctrl+D",), "time.set": ("Ctrl+T",),
            "time.shift": ("Ctrl+Shift+T",), "group.hide": ("Ctrl+H",), "group.mark": ("Ctrl+M",),
            "edit.redo": ("Ctrl+Y", "Ctrl+Shift+Z"), "group.confirm_next": ("Ctrl+Enter", "Ctrl+Return"),
            "group.confirm": ("Ctrl+Shift+Enter", "Ctrl+Shift+Return"),
            "time.end_minus": ("Shift+[", "{"), "time.end_plus": ("Shift+]", "}"),
            "play.group": ("R",), "play.toggle": ("Space",),
            "play.group_typing": ("Ctrl+R",), "play.toggle_plain": ("Ctrl+Space",),
            "play.step_mode": ("Ctrl+Shift+Space",), "time.adopt": ("C",),
            "nav.prev": ("PgUp",), "nav.next": ("PgDown",), "ruby.edit": ("F2",),
            "ruby.regen": ("Ctrl+G",), "ruby.regen_all": ("Ctrl+Shift+G",),
        }
        for action_id, keys in expected.items():
            with self.subTest(action_id=action_id):
                self.assertEqual(self.registry.effective_keys(action_id), keys)
        for key in "SQWDE":
            self.assertEqual(self.registry.effective_keys("play." + key.lower()), (key,))

    def test_watch_swaps_replace_main_keys(self):
        for action_id, main, watch in (("play.prev_group", "Ctrl+Left", "Left"),
                                       ("play.next_group", "Ctrl+Right", "Right"),
                                       ("play.frame_back", "Left", "Ctrl+Left"),
                                       ("play.frame_forward", "Right", "Ctrl+Right")):
            self.assertEqual(self.registry.effective_keys(action_id, "main"), (main,))
            self.assertEqual(self.registry.effective_keys(action_id, "watch"), (watch,))
        self.assertEqual(self.registry.conflicts(), ())

    def test_watch_only_keys_and_fallback(self):
        for action_id, key in (("watch.edit_original", "X"), ("watch.edit_translation", "Z"),
                               ("watch.transform", "T"), ("watch.transform_line", "Alt+T")):
            self.assertEqual(self.registry.effective_keys(action_id, "main"), ())
            self.assertEqual(self.registry.effective_keys(action_id, "watch"), (key,))
        self.registry.set_keys("file.save", "main", "Ctrl+B")
        self.assertEqual(self.registry.effective_keys("file.save", "watch"), ("Ctrl+B",))
        self.assertEqual(self.registry.effective_keys("file.save", "k"), ())

    def test_per_scene_override_clear_and_reset(self):
        self.registry.set_keys("file.save", "main", ("Ctrl+B", QKeySequence("Ctrl+L")))
        self.registry.set_keys("file.save", "watch", "Alt+B")
        self.assertEqual(self.registry.effective_keys("file.save"), ("Ctrl+B", "Ctrl+L"))
        self.assertEqual(self.registry.effective_keys("file.save", "watch"), ("Alt+B",))
        self.registry.clear_keys("file.save", "watch")
        self.assertEqual(self.registry.effective_keys("file.save", "watch"), ())
        self.registry.reset("file.save", "watch")
        self.assertEqual(self.registry.effective_keys("file.save", "watch"), ("Ctrl+B", "Ctrl+L"))
        self.registry.reset("file.save", "main")
        self.assertEqual(self.registry.effective_keys("file.save"), ("Ctrl+S",))

    def test_reset_all_keeps_other_registry_defaults_independent(self):
        self.registry.clear_keys("file.save", "main")
        self.registry.set_keys("k.tap", "k", "L")
        self.registry.reset_all()
        self.assertEqual(self.registry.effective_keys("file.save"), ("Ctrl+S",))
        self.assertEqual(self.registry.effective_keys("k.tap", "k"), ("K",))
        self.assertEqual(ActionRegistry().effective_keys("file.save"), ("Ctrl+S",))

    def test_same_scene_conflict_uses_normalized_secondary_keys(self):
        self.registry.set_keys("file.open", "main", ("Ctrl+B", "ctrl+s"))
        conflict, = self.registry.conflicts("main")
        self.assertEqual((conflict.scene, conflict.key, conflict.action_ids),
                         ("main", "Ctrl+S", ("file.open", "file.save")))
        self.registry.clear_keys("file.open", "main")
        self.assertEqual(self.registry.conflicts("main"), ())

    def test_watch_conflict_with_inherited_main_binding(self):
        self.registry.set_keys("file.open", "watch", "Ctrl+S")
        self.assertEqual(self.registry.conflicts("main"), ())
        conflict, = self.registry.conflicts("watch")
        self.assertEqual(conflict.action_ids, ("file.open", "file.save"))
        self.registry.clear_keys("file.save", "watch")
        self.assertEqual(self.registry.conflicts("watch"), ())
        self.registry.reset("file.save", "watch")
        self.assertEqual(len(self.registry.conflicts("watch")), 1)

    def test_cross_scene_equal_keys_allowed(self):
        self.registry.set_keys("file.save", "main", "Ctrl+B")
        self.registry.set_keys("file.open", "watch", "Ctrl+B")
        self.registry.set_keys("file.save", "watch", "Ctrl+L")
        self.registry.set_keys("k.import", "k", "Ctrl+B")
        self.assertEqual(self.registry.conflicts(), ())
        self.assertEqual(self.registry.effective_keys("group.delete"), ("Del",))
        self.assertEqual(self.registry.effective_keys("k.remove_rest", "k"), ("Del",))

    def test_reserved_keys_rejected_without_mutating_binding(self):
        for key in ("F1", "Escape", "Tab", "Shift+Tab", "Enter", "Return", "Alt+F4", "Alt",
                    "Meta+B", "Ctrl+Meta+B", "Ctrl+A", "Ctrl+C", "Ctrl+V", "Ctrl+X",
                    "Ctrl+Shift", "Alt+Shift", "Shift+Space", "Kanji", "Hangul", "Ctrl+B, F1"):
            with self.subTest(key=key):
                self.assertTrue(is_reserved(key))
                with self.assertRaises(ValueError):
                    self.registry.set_keys("file.save", "main", key)
                self.assertEqual(self.registry.effective_keys("file.save"), ("Ctrl+S",))
        for key in ("Ctrl+Space", "Ctrl+Shift+Space", "Ctrl+Enter", "Ctrl+Return", "Alt+B", "R"):
            self.assertFalse(is_reserved(key), key)

    def test_popup_keys_fixed_and_complete(self):
        popup = [spec for spec in self.registry if "popup" in spec.keys]
        self.assertTrue(popup)
        keys = {key for spec in popup for key in spec.keys["popup"]}
        self.assertTrue({"Enter", "Esc", "Tab", "Left", "Right", "Up", "Down", "Alt+Left", "Alt+Right",
                         "Backspace", *tuple("123456789")} <= keys)
        for spec in popup:
            self.assertFalse(spec.customizable)
            with self.assertRaises(ValueError):
                self.registry.set_keys(spec.id, "popup", "Ctrl+B")

    def test_typing_metadata_for_mixed_and_text_keys(self):
        for action_id in ("play.toggle_plain", "play.group_typing", "file.save", "group.confirm_next"):
            self.assertTrue(self.registry.lookup(action_id).typing)
        for action_id in ("play.toggle", "play.group", "play.s", "time.adopt", "time.start_minus", "watch.transform", "k.tap"):
            self.assertFalse(self.registry.lookup(action_id).typing)

    def test_tap_requires_one_unmodified_non_modifier_key(self):
        for keys in ("Ctrl+B", "Shift+K", "Alt+L", "Meta+L", "Ctrl", "Shift", "AltGr",
                     "L, B", ("L", "B")):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                self.registry.set_keys("k.tap", "k", keys)
            self.assertEqual(self.registry.effective_keys("k.tap", "k"), ("K",))
        self.registry.set_keys("k.tap", "k", "L")
        self.assertEqual(self.registry.effective_keys("k.tap", "k"), ("L",))
        self.registry.clear_keys("k.tap", "k")
        self.assertEqual(self.registry.effective_keys("k.tap", "k"), ())

    def test_invalid_scenes_and_keys_do_not_change_registry(self):
        for scene, key in (("unknown", "Ctrl+B"), ("k", "Ctrl+B"), ("main", "not a key")):
            with self.assertRaises(ValueError):
                self.registry.set_keys("file.save", scene, key)
        with self.assertRaises(ValueError):
            self.registry.effective_keys("file.save", "unknown")
        with self.assertRaises(ValueError):
            self.registry.conflicts("unknown")
        self.assertEqual(self.registry.effective_keys("file.save"), ("Ctrl+S",))

    def test_module_import_needs_no_application_or_widgets(self):
        code = ("import sys; sys.path.insert(0, 'src'); from subkit_app.actions import ActionRegistry; "
                "from PySide6.QtCore import QCoreApplication; "
                "assert QCoreApplication.instance() is None; "
                "assert 'PySide6.QtWidgets' not in sys.modules; assert ActionRegistry().lookup('file.save')")
        process = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", code], cwd=ROOT,
                                 capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)
        tree = ast.parse((ROOT / "src/subkit_app/actions.py").read_text(encoding="utf-8"))
        qt_imports = {node.module for node in ast.walk(tree)
                      if isinstance(node, ast.ImportFrom) and node.module.startswith("PySide6")}
        self.assertEqual(qt_imports, {"PySide6.QtCore", "PySide6.QtGui"})


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="duosub_actions_")
        self.path = str(Path(self.tmp.name) / "shortcuts.ini")
        self.settings = QSettings(self.path, QSettings.Format.IniFormat)
        self.registry = ActionRegistry()

    def tearDown(self):
        self.settings.sync()
        del self.settings
        self.tmp.cleanup()

    def test_portable_text_round_trip_missing_empty_and_overrides(self):
        self.settings.setValue("layouts/current", "karaoke")
        self.registry.set_keys("file.save", "main", ("Ctrl+B", QKeySequence("Alt+L")))
        self.registry.clear_keys("file.open", "main")
        self.registry.clear_keys("file.save", "watch")
        self.registry.set_keys("k.tap", "k", "L")
        self.registry.save(self.settings)
        self.settings.sync()
        self.assertEqual(self.settings.status(), QSettings.Status.NoError)
        self.assertEqual(self.settings.value("shortcuts/version", type=int), 1)
        self.assertEqual(self.settings.value("shortcuts/main/file.save"), "Ctrl+B;Alt+L")
        self.assertEqual(self.settings.value("shortcuts/main/file.open"), "")
        self.assertFalse(self.settings.contains("shortcuts/main/file.new"))
        del self.settings
        self.settings = QSettings(self.path, QSettings.Format.IniFormat)
        restored = ActionRegistry()
        restored.load(self.settings)
        self.assertEqual(restored.effective_keys("file.save"), ("Ctrl+B", "Alt+L"))
        self.assertEqual(restored.effective_keys("file.save", "watch"), ())
        self.assertEqual(restored.effective_keys("file.open"), ())
        self.assertEqual(restored.effective_keys("file.new"), ("Ctrl+Shift+N",))
        self.assertEqual(restored.effective_keys("k.tap", "k"), ("L",))
        self.assertEqual(self.settings.value("layouts/current"), "karaoke")

    def test_unknown_fixed_reserved_and_malformed_entries_ignored(self):
        for path, value in (("main/unknown.command", "Ctrl+B"), ("invalid/file.open", "Ctrl+B"),
                            ("main/file.save", "F1"), ("main/file.new", "invalid key"),
                            ("popup/popup.cancel", "Ctrl+B"), ("main/file.open/extra", "Ctrl+B")):
            self.settings.setValue("shortcuts/" + path, value)
        self.settings.setValue("shortcuts/main/time.set", "Ctrl+L")
        self.registry.load(self.settings)
        self.assertEqual(self.registry.effective_keys("time.set"), ("Ctrl+L",))
        self.assertEqual(self.registry.effective_keys("file.save"), ("Ctrl+S",))
        self.assertEqual(self.registry.effective_keys("file.open"), ("Ctrl+O",))
        self.assertEqual(self.registry.effective_keys("file.new"), ("Ctrl+Shift+N",))
        self.assertEqual(self.registry.effective_keys("popup.cancel", "popup"), ("Esc",))

    def test_reset_persistence_removes_overrides_and_restores_fallback(self):
        self.registry.clear_keys("file.save", "watch")
        self.registry.set_keys("file.save", "main", "Ctrl+B")
        self.registry.save(self.settings)
        self.registry.reset("file.save", "watch")
        self.registry.save(self.settings)
        self.assertFalse(self.settings.contains("shortcuts/watch/file.save"))
        self.registry.load(self.settings)
        self.assertEqual(self.registry.effective_keys("file.save", "watch"), ("Ctrl+B",))
        self.registry.reset_all()
        self.registry.save(self.settings)
        self.assertEqual(self.settings.allKeys(), ["shortcuts/version"])

    def test_load_replaces_previous_overrides_and_ignores_future_version(self):
        self.registry.clear_keys("file.save", "main")
        self.registry.load(self.settings)
        self.assertEqual(self.registry.effective_keys("file.save"), ("Ctrl+S",))
        self.settings.setValue("shortcuts/version", 99)
        self.settings.setValue("shortcuts/main/file.save", "Ctrl+B")
        self.registry.load(self.settings)
        self.assertEqual(self.registry.effective_keys("file.save"), ("Ctrl+S",))

    def test_semicolon_keys_and_multistroke_portable_round_trip(self):
        keys = ("Ctrl+;", ";", "Ctrl+L, Ctrl+B")
        self.registry.set_keys("file.save", "main", keys)
        self.registry.save(self.settings)
        self.settings.sync()
        restored = ActionRegistry()
        restored.load(self.settings)
        self.assertEqual(restored.effective_keys("file.save"), keys)


if __name__ == "__main__":
    unittest.main()
