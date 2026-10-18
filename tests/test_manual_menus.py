"""Resolve documented menu paths against an isolated offscreen MainWindow."""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
import unittest

from tests import test_convert_ui as fixtures


ROOT = Path(__file__).resolve().parents[1]
OLD_LOCATIONS = (
    r"集\s*→\s*(?:检查|查找并套用重复段落|自动调轴|统一两种语言的时间|精确对话轴|记为重复段落|导出审议包|导回审议意见|字体库位置|原字幕括号)",
    r"视图\s*→\s*(?:显示规则|打轴方式|观剧模式显示说话人|暂停时显示字幕外框|字幕外框颜色)",
    r"视图\s*→\s*布局\s*→\s*卡拉 OK",
    r"文件\s*→\s*(?:振假名设置|拼音设置|工程词语与分词规则|重复段落库|私有功能|清除排版缓存|导出发布用|导出整季发布用|导出标记和隐藏)",
    r"编辑\s*→\s*(?:位置|与下一组合并|转为|整体平移|自由变换)",
    r"(?:「集」|集)菜单.{0,12}(?:检查|重复段落|自动调轴|统一两种语言的时间)",
    r"(?:「文件」|文件)菜单.{0,12}(?:重复段落|振假名设置|拼音设置|工程词语)",
    r"编辑菜单.{0,12}(?:整体平移|转为)",
)


class MenuPaths(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []
        self.parts: list[str] | None = None

    def handle_starttag(self, tag, attrs) -> None:
        if tag == "span" and "menu" in dict(attrs).get("class", "").split():
            self.parts = []

    def handle_data(self, data) -> None:
        if self.parts is not None:
            self.parts.append(data)

    def handle_endtag(self, tag) -> None:
        if tag == "span" and self.parts is not None:
            self.paths.append("".join(self.parts))
            self.parts = None


def action_name(action) -> str:
    # Keep the actual label (including meaningful parentheses and ellipsis).
    # Only omit Qt's mnemonic, displayed key and temporary disabled reason.
    text = action.property("enabled_text") or action.text()
    return re.sub(r"\(&.\)", "", text.split("\t")[0]).strip()


def resolve_path(menu, path: str) -> None:
    parts = [part.strip().removesuffix("▸").strip() for part in path.split("→")]
    for index, part in enumerate(parts):
        matches = [a for a in menu.actions() if not a.isSeparator() and action_name(a) == part]
        if len(matches) != 1:
            raise AssertionError(f"{path}: {part!r} resolves to {len(matches)} items")
        if index < len(parts) - 1:
            menu = matches[0].menu()
            if menu is None:
                raise AssertionError(f"{path}: {part!r} is not a submenu")


class ManualMenuTests(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    setUp = fixtures.ConversionUI.setUp
    _dispose = fixtures.ConversionUI._dispose

    def test_every_documented_path_resolves_item_by_item(self):
        self.project.languages = {"original": "ja", "translation": "zh-Hant"}
        self.window._update_group_menu_actions()
        parser = MenuPaths()
        parser.feed((ROOT / "docs/manual/index.html").read_text(encoding="utf-8"))
        self.assertGreater(len(parser.paths), 50)
        for path in parser.paths:
            with self.subTest(path=path):
                resolve_path(self.window.menuBar(), path)

    def test_resolver_rejects_missing_parent_wrong_branch_and_abbreviation(self):
        for path in ("发布用 ASS…", "编辑 → 位置", "文件 → 导出 → 发布用 ASS",
                     "播放 → 播放本组 → 不存在"):
            with self.subTest(path=path), self.assertRaises(AssertionError):
                resolve_path(self.window.menuBar(), path)


class ManualLocationTests(unittest.TestCase):
    def test_no_obsolete_locations_in_manual_and_passages(self):
        found = []
        for name in ("docs/manual/index.html", "docs/PASSAGES.md"):
            for number, line in enumerate((ROOT / name).read_text(encoding="utf-8").splitlines(), 1):
                if any(re.search(pattern, line) for pattern in OLD_LOCATIONS):
                    found.append(f"{name}:{number}: {line.strip()}")
        self.assertEqual(found, [])
