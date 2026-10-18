"""S8 editor layout, remembered sections and real Qt input (isolated settings)."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import unittest

from tests import test_convert_ui as fixtures
from PySide6.QtCore import QEvent, Qt, qVersion
from PySide6.QtGui import QInputMethodEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractButton, QApplication, QScrollBar
from subkit.model.document import Run
from subkit_app.views.group_editor import GroupEditor


class EditorPanel(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.group("previous", text="上下文" * 100, start=0)
        translation = self.group("current", text="翻译正文")
        original = copy.deepcopy(translation)
        original.id, original.lane, original.lang = "original", "original", "ja"
        original.runs = [Run("原文", "text", ins="original", a=0, b=2)]
        self.doc.members[original.id] = original
        self.doc.group("current").members.append(original.id)
        self.group("next", text="后文" * 100, start=4000)
        self.doc.structure_changed()
        self.window.select_group("current")
        self.editor = self.window.editor

    def show(self):
        self.window.show()
        self.window.activateWindow()
        self.app.processEvents()

    def test_no_truncation_eight_window_layout_combinations(self):
        evidence = Path(os.environ["TEMP"]) / "s8-layouts"
        evidence.mkdir(exist_ok=True)
        results = []
        for width, height in ((1600, 1000), (1280, 800)):
            for layout in ("video", "list", "columns", "karaoke"):
                with self.subTest(size=(width, height), layout=layout):
                    self.window.set_edit_layout(layout)
                    self.window.select_group("current", restore=True)
                    for key, section in self.editor.sections.items():
                        section.expanded = True
                        section._refresh()
                    self.editor._update_content_height()
                    self.window.resize(width, height)
                    self.show()
                    self.assertEqual(self.editor.group.id, "current")
                    self.app.processEvents()
                    self.assertEqual((self.window.width(), self.window.height()), (width, height))
                    buttons = []
                    for button in self.editor.findChildren(QAbstractButton):
                        if not button.isVisible() or not button.text():
                            continue
                        text_width = button.fontMetrics().horizontalAdvance(button.text().replace("&", ""))
                        available = button.contentsRect().width()
                        self.assertLessEqual(text_width, available, button.text())
                        self.assertNotIn("…", button.text())
                        # Also reject full-size buttons clipped by a narrower ancestor.
                        parent = button.parentWidget()
                        while parent is not self.editor:
                            self.assertLessEqual(button.mapTo(parent, button.rect().topRight()).x(), parent.width(), button.text())
                            parent = parent.parentWidget()
                        buttons.append([button.text(), text_width, available])
                    for scrollbar in self.editor.findChildren(QScrollBar):
                        if scrollbar.orientation() == Qt.Orientation.Horizontal:
                            self.assertFalse(scrollbar.isVisible())
                    results.append({"size": [self.window.width(), self.window.height()], "layout": layout,
                                    "editor_width": self.editor.width(), "buttons": buttons})
                    self.assertTrue(self.window.grab().save(str(evidence / f"{width}-{layout}.png")))
        (evidence / "geometry.json").write_text(json.dumps({"qt": qVersion(), "platform": self.app.platformName(),
            "player": "test double", "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_section_defaults_persist_and_restore_after_unavailable(self):
        self.show()
        sections = self.editor.sections
        self.assertEqual([s.expanded for s in sections.values()], [True, False, False])
        self.editor.position.show_anchor({"x": 100, "y": 200, "an": 2})
        for section in sections.values():
            QTest.mouseClick(section.header, Qt.MouseButton.LeftButton)
        self.assertEqual([s.expanded for s in sections.values()], [False, True, True])
        self.editor.set_group(self.doc, None, None, None)
        for section in sections.values():
            self.assertFalse(section.header.isChecked())
            self.assertTrue(section.content.isHidden())
            self.assertIn("（先选中字幕组）", section.header.text())
        self.editor.set_group(self.doc, self.doc.group("current"), None, None)
        self.editor.position.show_anchor({"x": 100, "y": 200, "an": 2})
        self.assertEqual([s.header.isChecked() for s in sections.values()], [False, True, True])
        self.editor.position.show_anchor(None)
        self.assertFalse(sections["position"].header.isChecked())
        self.assertIn("（先等待字幕预览更新）", sections["position"].header.text())
        self.editor.position.show_anchor({"x": 100, "y": 200, "an": 2})
        self.assertTrue(sections["position"].header.isChecked())
        self.settings.sync()
        fresh = GroupEditor()
        self.addCleanup(fresh.deleteLater)
        fresh.set_settings(self.settings)
        self.assertEqual([s.expanded for s in fresh.sections.values()], [False, True, True])
        self.assertEqual([self.settings.value("editor/section/" + key, type=bool)
                          for key in sections], [False, True, True])

    def test_real_tab_order_and_ime_commit_escape(self):
        self.show()
        original, translation = self.editor.jp_edit, self.editor.cn_edits[0][1]
        actions, commits = [], []
        self.editor.action.connect(actions.append)
        self.editor.textCommitted.connect(lambda *args: commits.append(args))
        for edit in (translation, original):
            edit.setFocus()
            self.app.processEvents()
            before = edit.toPlainText()
            count = len(commits)
            QApplication.sendEvent(edit, QInputMethodEvent("输入法組字", []))
            self.assertEqual(actions, [])
            self.assertEqual(len(commits), count)
            self.assertEqual(edit.toPlainText(), before)
            event = QInputMethodEvent()
            event.setCommitString("字")
            QApplication.sendEvent(edit, event)
            QTest.keyClick(edit, Qt.Key.Key_Return)
            self.assertEqual(commits[-1][1], edit.toPlainText())
            self.assertEqual(actions, [])
            committed = edit.toPlainText()
            QTest.keyClicks(edit, "temporary")
            QTest.keyClick(edit, Qt.Key.Key_Escape)
            self.assertEqual(edit.toPlainText(), committed)
            self.assertEqual(len(commits), 1 if edit is translation else 2)
        original.setFocus()
        QTest.keyClick(original, Qt.Key.Key_Tab)
        self.assertIs(self.app.focusWidget(), self.editor.action_buttons[0])
        expected = [*self.editor.action_buttons[1:-1], *self.editor.review_buttons.values(),
                    self.editor.action_buttons[-1], self.editor.sections["format"].header]
        for widget in expected:
            QTest.keyClick(self.app.focusWidget(), Qt.Key.Key_Tab)
            self.assertIs(self.app.focusWidget(), widget)
        QTest.keyClick(self.app.focusWidget(), Qt.Key.Key_Tab)
        self.assertTrue(self.editor.fmt.isAncestorOf(self.app.focusWidget()))

    def test_tooltip_rebind_and_full_context(self):
        self.assertGreater(len(self.editor.prev_lbl.toolTip()), 100)
        self.assertEqual(self.editor.prev_lbl.text(), self.editor.prev_lbl.toolTip())
        for name, index, key in (("merge", 4, "Ctrl+Alt+J"), ("split", 5, "Ctrl+Alt+K")):
            registry = self.window.action_hub.registry
            registry.set_keys("group." + name, "main", key)
            self.window._refresh_shortcuts()
            tooltip = self.editor.action_buttons[index].toolTip()
            self.assertIn(registry.lookup("group." + name).text, tooltip)
            self.assertIn(key, tooltip)
