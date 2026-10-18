"""List context menu matrix and proxy dispatch, using offscreen Qt."""
from __future__ import annotations

import ast
import itertools
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QWidget
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subkit.model.document import Document, Group, Member
from subkit_app.action_hub import ActionHub
from subkit_app.actions import ActionRegistry
from subkit_app.views.group_table import GroupTable, GroupTableModel, GroupFilter


class ContextMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.owner = QWidget()
        self.registry = ActionRegistry()
        self.hub = ActionHub(self.registry, self.owner)
        self.table = GroupTable(self.owner)
        self.doc = Document("context-matrix")
        self.model = GroupTableModel(self.doc)
        self.proxy = GroupFilter()
        self.proxy.setSourceModel(self.model)
        self.table.setModel(self.proxy)
        self.table.context_action_lookup = self.hub.action
        self.table.context_action_spec = self.registry.lookup
        self.table.context_action_keys = self.registry.effective_keys
        self.table.karaoke_effect_choices = lambda: [("none", "不用特效"), ("kf", "逐字扫色")]
        self.addCleanup(self.owner.close)
        self.addCleanup(self.owner.deleteLater)

    def populate(self, kinds, lanes, confirmed=False, hidden=False, marked=False, mixed=False):
        self.doc.groups.clear()
        self.doc.members.clear()
        for i, kind in enumerate(kinds):
            gid = str(i)
            g = Group(gid, kind, [])
            g.state = "confirmed" if confirmed ^ (mixed and i % 2 == 1) else "draft"
            g.flags = [flag for flag, value in (("hidden", hidden), ("marked", marked))
                       if value ^ (mixed and i % 2 == 1)]
            for lane in lanes[i % len(lanes)]:
                mid = gid + lane
                self.doc.members[mid] = Member(mid, gid, "ja" if lane == "original" else "zh-Hant",
                                               lane, "dialogue", 1000, 2000)
                g.members.append(mid)
            self.doc.groups.append(g)
        return [g.id for g in self.doc.groups]

    @staticmethod
    def labels(menu):
        return [None if a.isSeparator() else a.text().split("\t")[0] for a in menu.actions()]

    def test_matrix(self):
        count = 0
        for kind, scope, lane_mask, confirmed, hidden, marked in itertools.product(
                ("dialogue", "song", "sign"), ("single", "multi", "mixed"), range(4), (False, True), (False, True), (False, True)):
            with self.subTest(kind=kind, scope=scope, lanes=lane_mask, confirmed=confirmed, hidden=hidden, marked=marked):
                lanes = [lane for bit, lane in enumerate(("original", "translation")) if lane_mask & (1 << bit)]
                kinds = [kind] if scope == "single" else [kind, kind] if scope == "multi" else [kind, "dialogue", "song", "sign"]
                lane_sets = [lanes] if scope != "mixed" else [lanes, [], ["original"], ["translation"]]
                gids = self.populate(kinds, lane_sets, confirmed, hidden, marked, scope == "mixed")
                shared = {spec.id: self.hub.action(spec.id).text() for spec in self.registry}
                menu = self.table._context_menu(gids)
                groups = self.doc.groups
                expected = ["播放这一组", None]
                if any(g.state != "confirmed" for g in groups): expected += ["确认这一组"]
                if any(g.state == "confirmed" for g in groups): expected += ["取消确认"]
                expected += [None, "复制这一组" if len(gids) == 1 else f"复制这 {len(gids)} 组", "设置开始和结束时间…" + ("（只能对一组设置）" if len(gids) > 1 else ""), "整体平移…"]
                missing = [lane for lane in ("original", "translation") if any(not any(m.lane == lane for m in self.doc.members_of(g)) for g in groups)]
                if missing or set(kinds) & {"dialogue", "sign"}:
                    expected += [None]
                    expected += ["补上原声（日语）…" if lane == "original" else "补上翻译（繁中）…" for lane in missing]
                    if set(kinds) & {"dialogue", "sign"}:
                        from subkit.core.listening import unavailable_reason
                        reason = unavailable_reason(self.doc, gids, self.model.languages)
                        expected += ["AI听译…" + (f"（{reason}）" if reason else "")]
                expected += [None, "转为"]
                targets = (["歌曲（K 轴）…"] if "dialogue" in kinds else []) + (["普通台词…"] if set(kinds) & {"song", "sign"} else []) + (["屏幕字…"] if "dialogue" in kinds else [])
                if "song" in kinds: expected += ["K 轴特效", f"确认 K 轴（{kinds.count('song')} 句歌词）"]
                expected += [None]
                for flag, on, off in (("hidden", "隐藏（不显示、不导出）", "取消隐藏"), ("marked", "标记待复查", "取消标记")):
                    if any(flag not in g.flags for g in groups): expected += [on]
                    if any(flag in g.flags for g in groups): expected += [off]
                if any("marked" in g.flags for g in groups): expected += ["编辑标记备注…"]
                expected += [None, "记为重复段落…", None]
                both = {m.lane for g in groups for m in self.doc.members_of(g)} >= {"original", "translation"}
                if both: expected += ["删除"]
                expected += ["删除这一组（原声和翻译都删）"]
                self.assertEqual(self.labels(menu), expected)
                convert = next(a.menu() for a in menu.actions() if a.text() == "转为")
                self.assertEqual(self.labels(convert), targets)
                self.assertTrue(all(a.isEnabled() for a in convert.actions()))
                if len(targets) < 3: self.assertTrue(convert.menuAction().toolTip())
                times = next(a for a in menu.actions() if a.property("action_id") == "time.set")
                self.assertEqual(times.isEnabled(), len(gids) == 1)
                self.assertTrue(times.text().endswith("\tCtrl+T"))
                listen = next((a for a in menu.actions() if a.property("action_id") == "group.ai_listen"), None)
                if listen is not None: self.assertEqual(listen.isEnabled(), not reason)
                if both:
                    delete = next(a.menu() for a in menu.actions() if a.text() == "删除")
                    self.assertEqual(self.labels(delete), ["只删原声", "只删翻译"])
                self.assertEqual(shared, {spec.id: self.hub.action(spec.id).text() for spec in self.registry})
                menu.deleteLater()
                count += 1
        self.assertEqual(count, 288)

    def test_registry_keys_and_shared_action_state_stay_independent(self):
        gids = self.populate(["dialogue"], [["original", "translation"]])
        self.registry.set_keys("group.duplicate", "main", "Ctrl+Alt+D")
        shared = self.hub.action("group.duplicate")
        shared.setText("复制这一组（菜单状态）")
        shared.setEnabled(False)
        shared.setCheckable(True)
        shared.setChecked(True)
        before = (shared.text(), shared.isEnabled(), shared.isChecked(), shared.shortcuts())
        menu = self.table._context_menu(gids)
        proxy = next(a for a in menu.actions() if a.property("action_id") == "group.duplicate")
        self.assertEqual(proxy.text(), "复制这一组\tCtrl+Alt+D")
        self.assertTrue(proxy.isEnabled())
        self.assertEqual(before, (shared.text(), shared.isEnabled(), shared.isChecked(), shared.shortcuts()))

    def test_native_menu_activation_dispatches_once(self):
        from PySide6.QtCore import QTimer, Qt
        from PySide6.QtGui import QContextMenuEvent
        from PySide6.QtTest import QTest
        gids = self.populate(["dialogue"], [["translation"]])
        self.model.reset()
        self.table.select_gid(gids[0])
        menu = self.table._context_menu(gids)
        action = next(a for a in menu.actions() if a.property("action_id") == "group.duplicate")
        calls = []
        self.table.duplicateGroupsRequested.connect(calls.append)
        pos = self.table.visualRect(self.table.currentIndex()).center()
        event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, pos, self.table.mapToGlobal(pos))
        QTimer.singleShot(0, lambda: QTest.mouseClick(menu, Qt.MouseButton.LeftButton,
                                                   pos=menu.actionGeometry(action).center()))
        with patch.object(self.table, "_context_menu", return_value=menu):
            self.table.contextMenuEvent(event)
        self.assertEqual(calls, [gids])

    def test_no_conversion_and_comment_does_not_fill_lane(self):
        gids = self.populate(["info"], [["original"]])
        self.doc.members["0original"].role = "comment"
        menu = self.table._context_menu(gids)
        self.assertNotIn("转为", self.labels(menu))
        self.assertIn("补上原声（日语）…", self.labels(menu))
        self.assertNotIn("AI听译…", self.labels(menu))

    def test_every_proxy_dispatches_original_signal_with_selection(self):
        gids = self.populate(["dialogue", "song"], [["original", "translation"], []], mixed=True)
        calls = []
        names = ["playRequested", "confirmGroupsRequested", "duplicateGroupsRequested", "groupTimesRequested", "shiftGroupsRequested", "addLanguageRequested", "listeningRequested", "convertRequested", "karaokeEffectRequested", "karaokeConfirmRequested", "hiddenGroupsRequested", "markedGroupsRequested", "markNoteRequested", "recordPassageRequested", "deleteLaneRequested", "deleteGroupsRequested"]
        for name in names:
            getattr(self.table, name).connect(lambda *args, name=name: calls.append((name, args)))
        expected = {
            "播放这一组": ("playRequested", ("0",)), "确认这一组": ("confirmGroupsRequested", (["0"], True)), "取消确认": ("confirmGroupsRequested", (["1"], False)),
            "复制这 2 组": ("duplicateGroupsRequested", (gids,)), "整体平移…": ("shiftGroupsRequested", (gids,)),
            "补上原声（日语）…": ("addLanguageRequested", (gids, "original")), "补上翻译（繁中）…": ("addLanguageRequested", (gids, "translation")),
            "歌曲（K 轴）…": ("convertRequested", (gids, "song")), "普通台词…": ("convertRequested", (gids, "dialogue")), "屏幕字…": ("convertRequested", (gids, "sign")),
            "不用特效": ("karaokeEffectRequested", (["1"], "none")), "逐字扫色": ("karaokeEffectRequested", (["1"], "kf")), "确认 K 轴（1 句歌词）": ("karaokeConfirmRequested", (["1"],)),
            "隐藏（不显示、不导出）": ("hiddenGroupsRequested", (["0"], True)), "取消隐藏": ("hiddenGroupsRequested", (["1"], False)), "标记待复查": ("markedGroupsRequested", (["0"], True)), "取消标记": ("markedGroupsRequested", (["1"], False)),
            "编辑标记备注…": ("markNoteRequested", (gids,)), "记为重复段落…": ("recordPassageRequested", (gids,)),
            "只删原声": ("deleteLaneRequested", (gids, "original")), "只删翻译": ("deleteLaneRequested", (gids, "translation")), "删除这一组（原声和翻译都删）": ("deleteGroupsRequested", (gids,))}
        menu = self.table._context_menu(gids)
        def check(menu):
            for action in menu.actions():
                if action.menu(): check(action.menu())
                elif action.isEnabled() and not action.isSeparator():
                    label = action.text().split("\t")[0]
                    action.trigger()
                    self.assertEqual(calls[-1], expected.pop(label))
        check(menu)
        self.assertFalse(expected)
        gids = self.populate(["dialogue"], [[]])
        with patch("subkit.core.listening.unavailable_reason", return_value=""):
            menu = self.table._context_menu(gids)
        for action_id, signal in (("time.set", "groupTimesRequested"), ("group.ai_listen", "listeningRequested")):
            action = next(a for a in menu.actions() if a.property("action_id") == action_id)
            action.trigger()
            self.assertEqual(calls[-1], (signal, (["0"],) if action_id == "time.set" else ("0",)))


class ButtonTexts(unittest.TestCase):
    def test_no_bracketed_keys_in_button_text(self):
        root = Path(__file__).resolve().parents[1] / "src" / "subkit_app"
        bad = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and (isinstance(node.func, ast.Name) and node.func.id in {"QPushButton", "QToolButton", "QCheckBox", "QRadioButton"} or isinstance(node.func, ast.Attribute) and node.func.attr == "setText"):
                    for arg in node.args[:1]:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and re.search(r"[（(](?:Esc|Space|[A-Z])[）)]", arg.value):
                            bad.append((str(path.relative_to(root)), node.lineno, arg.value))
        self.assertEqual(bad, [])


class ReadingCancel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_reading_cancel_tooltip(self):
        from subkit_app.views.reading_popup import ReadingPopup
        popup = ReadingPopup("学校", "がっこう", ["がっこう"], "manual", "confirmed")
        self.addCleanup(popup.close)
        self.addCleanup(popup.deleteLater)
        self.assertEqual(popup.cancel_button.text(), "取消")
        self.assertEqual(popup.cancel_button.toolTip(), "Esc")
        popup._open_kanji()
        popup._cancel()
        self.assertEqual(popup.cancel_button.text(), "取消")
        self.assertEqual(popup.cancel_button.toolTip(), "Esc")
