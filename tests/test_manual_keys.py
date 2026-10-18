"""Compare the visible manual shortcut tables with registry defaults and menu names."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import unittest

from tests import test_convert_ui as fixtures
from PySide6.QtGui import QKeySequence

from subkit_app.actions import ActionRegistry
from subkit_app.views.shortcuts_dialog import ShortcutsDialog


ROOT = Path(__file__).resolve().parents[1]


class ShortcutTables(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables = {}
        self.scene = None
        self.row = None
        self.cell = None
        self.in_keys = False

    def handle_starttag(self, tag, attrs) -> None:
        attrs = dict(attrs)
        if tag == "section":
            self.in_keys = attrs.get("id") == "keys"
        elif tag == "table" and self.in_keys:
            self.scene = attrs.get("data-scene") or attrs.get("data-defaults")
            if self.scene:
                if self.scene in self.tables:
                    raise ValueError(f"Duplicate shortcut table: {self.scene}")
                self.tables[self.scene] = []
        elif tag == "tr" and self.scene:
            self.row = (attrs.get("data-action"), [])
        elif tag == "td" and self.row is not None:
            self.cell = []

    def handle_data(self, data) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag) -> None:
        if tag == "td" and self.cell is not None:
            self.row[1].append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row[1]:
                self.tables[self.scene].append(self.row)
            self.row = None
        elif tag == "table":
            self.scene = None
        elif tag == "section":
            self.in_keys = False


def portable_keys(text: str) -> tuple[str, ...]:
    if text == "—":
        return ()
    for source, target in (("PgDn", "PgDown"), ("←", "Left"), ("→", "Right"),
                           ("↑", "Up"), ("↓", "Down")):
        text = text.replace(source, target)
    # Read the visible cell, including text outside kbd tags, so a stray
    # documented key cannot hide behind the row's metadata.
    return tuple(QKeySequence(key.strip()).toString(QKeySequence.SequenceFormat.PortableText)
                 for key in text.split(" / "))


class ManualShortcutTests(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    setUp = fixtures.ConversionUI.setUp
    _dispose = fixtures.ConversionUI._dispose

    def tables(self):
        parser = ShortcutTables()
        parser.feed((ROOT / "docs/manual/index.html").read_text(encoding="utf-8"))
        return parser.tables

    def assert_scene(self, scene):
        registry = ActionRegistry()  # Ignore user/test overrides: the manual lists defaults.
        rows = self.tables()[scene]
        prefixes = ShortcutsDialog.menu_names(self.window.menuBar())
        # Every command with a default key in this scene is listed, in menu order; a
        # keyless command is listed only when its row explains something.
        listed = [action_id for action_id, _ in rows]
        order = [spec.id for spec in registry if scene in spec.keys]
        self.assertLessEqual({spec.id for spec in registry if spec.keys.get(scene)}, set(listed))
        self.assertEqual(listed, [action_id for action_id in order if action_id in set(listed)])
        for action_id, cells in rows:
            spec = registry.lookup(action_id)
            with self.subTest(scene=scene, action=action_id):
                self.assertEqual(len(cells), 3)
                name = " → ".join(filter(None, (prefixes.get(spec.id), spec.text)))
                self.assertEqual(cells[0], name)
                self.assertEqual(portable_keys(cells[1]), spec.keys[scene])
                if not spec.keys[scene]:
                    self.assertTrue(cells[2], "A keyless row must explain something")

    def test_main_defaults_names_secondary_keys_and_menu_order(self):
        self.assert_scene("main")

    def test_watch_and_k_defaults_names_and_scene_order(self):
        self.assert_scene("watch")
        self.assert_scene("k")

    def test_popup_fixed_keys_and_command_names(self):
        self.assert_scene("popup")

    def test_watch_inheritance_and_direction_swaps(self):
        tables = self.tables()
        self.assertEqual(list(tables), ["main", "watch", "k", "popup", "new"])
        main = {action_id: portable_keys(cells[1]) for action_id, cells in tables["main"]}
        watch = {action_id: portable_keys(cells[1]) for action_id, cells in tables["watch"]}
        registry = ActionRegistry()
        for spec in registry:
            if spec.customizable and ("main" in spec.keys or "watch" in spec.keys):
                with self.subTest(action=spec.id):
                    documented = watch.get(spec.id, main.get(spec.id, ()))
                    self.assertEqual(documented, registry.effective_keys(spec.id, "watch"))
        text = (ROOT / "docs/manual/index.html").read_text(encoding="utf-8")
        section = text.split('<section id="keys">', 1)[1].split('</section>', 1)[0]
        self.assertIn("未单列的命令沿用编辑界面默认键", section)
        self.assertIn("Alt+T 仍是自由变换这一行，不弹时间轴菜单", section)

    def test_new_default_summary_matches_registry(self):
        rows = self.tables()["new"]
        self.assertEqual(rows[-1][1][0:2], ["菜单栏隐藏时弹出菜单", "Alt+各菜单助记键"])
        rows = rows[:-1]
        self.assertEqual([action_id for action_id, _ in rows], [
            "group.duplicate", "time.set", "time.shift", "group.hide", "group.mark", "file.export"])
        prefixes = ShortcutsDialog.menu_names(self.window.menuBar())
        registry = ActionRegistry()
        for action_id, cells in rows:
            with self.subTest(action=action_id):
                spec = registry.lookup(action_id)
                self.assertEqual(cells[0], " → ".join(filter(None, (prefixes.get(action_id), spec.text))))
                self.assertEqual(portable_keys(cells[1]), registry.effective_keys(action_id))
