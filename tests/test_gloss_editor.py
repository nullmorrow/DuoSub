"""G2: real Qt text input, hidden gloss runs, identity/reading preservation and export."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QRect, Qt
from PySide6.QtGui import QKeyEvent, QTextCursor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from subkit.core.editor_text import editing_text, ordinary_offset
from subkit.core.export import build_ass, ExportOptions
from subkit.core.ops import Editor
from subkit.core.readings import anchor_of, plain_range_to_visible, reading_spans, visible_map
from subkit.model.document import Annotation, Run
from subkit.model.validate import validate
from subkit_app.views.check_result import issue_lines
from subkit_app.views.group_editor import GroupEditor, InlineLineEdit, LineEdit
from subkit_app import help as help_ui
from subkit_app.watch import InlineTextEditor
from tests.test_gloss import fixture
from tests import test_convert_ui as ui


def type_text(edit, text):
    for char in text:
        QApplication.sendEvent(edit, QKeyEvent(QEvent.Type.KeyPress, 0, Qt.KeyboardModifier.NoModifier, char))


class GlossEditor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        assert cls.app.platformName() == "offscreen"

    def setUp(self):
        self.doc, self.template = fixture("校庭へ", lane="original")
        self.ed = Editor(self.doc, {})
        self.widget = GroupEditor()
        self.widget.textCommitted.connect(self.ed.replace_editor_text)
        self.widget.set_group(self.doc, self.doc.groups[0], None, None)
        self.addCleanup(self.dispose)

    def dispose(self):
        self.widget.close()
        self.widget.deleteLater()
        self.app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def type_mark(self, mark, pos=2):
        edit = self.widget.jp_edit
        cursor = edit.textCursor()
        cursor.setPosition(pos)
        edit.setTextCursor(cursor)
        type_text(edit, mark)
        QTest.keyClick(edit, Qt.Key.Key_Return)
        return edit

    def test_aliases_fullwidth_typing_position_grey_and_one_undo(self):
        for alias in ("注", "g", "gloss", "GLOSS"):
            for colon in (":", "："):
                with self.subTest(alias=alias, colon=colon):
                    before = copy.deepcopy(self.doc.members)
                    mark = "{@" + alias + colon + "校庭=Garden}"
                    edit = self.type_mark(mark)
                    self.assertEqual(edit.toPlainText(), "校庭" + mark + "へ")
                    self.assertEqual(self.ed.visible_text("m"), "校庭へ")
                    self.assertEqual([(r.kind, r.text) for r in self.doc.members["m"].runs],
                                     [("text", "校庭"), ("tag", mark), ("text", "へ")])
                    self.app.processEvents()
                    formats = edit.document().firstBlock().layout().formats()
                    self.assertTrue(any(f.start == 2 and f.length == len(mark) and
                                        f.format.foreground().color().name() == "#8c959f" for f in formats))
                    self.assertEqual(len(self.ed.history.done), 1)
                    self.ed.history.undo()
                    self.assertEqual(self.doc.members, before)
                    self.widget.set_group(self.doc, self.doc.groups[0], None, None)

    def test_raw_markers_grey_in_both_lanes_and_inline_utf16(self):
        text = "𠀀前{@Raw=漢字}後{@注:後=word}"
        for edit in (self.widget.jp_edit, LineEdit(self.widget), InlineLineEdit()):
            try:
                edit.set_text(text)
                self.app.processEvents()
                formats = edit.document().firstBlock().layout().formats()
                grey = {i for f in formats if f.format.foreground().color().name() == "#8c959f"
                        for i in range(f.start, f.start + f.length)}
                self.assertEqual(grey, set(range(3, 9)) | {11} | set(range(13, 24)))
            finally:
                if edit is not self.widget.jp_edit:
                    edit.close()
                    edit.deleteLater()

    def test_edit_delete_mark_and_redo(self):
        self.type_mark("{@注:校庭=Garden}")
        self.widget.set_group(self.doc, self.doc.groups[0], None, None)
        edit = self.widget.jp_edit
        edit.setPlainText("校庭{@注:校庭=Schoolyard}へ")
        edit.commit()
        self.assertEqual(editing_text(self.doc.members["m"]), "校庭{@注:校庭=Schoolyard}へ")
        edit.setPlainText("校庭へ")
        edit.commit()
        self.assertFalse(any(r.kind == "tag" for r in self.doc.members["m"].runs))
        self.ed.history.undo()
        self.assertEqual(editing_text(self.doc.members["m"]), "校庭{@注:校庭=Schoolyard}へ")
        self.ed.history.redo()
        self.assertEqual(editing_text(self.doc.members["m"]), "校庭へ")

    def test_marks_preserve_readings_pinyin_identities_pairing_and_beats(self):
        m = self.doc.members["m"]
        ids = visible_map(m)[1]
        for kind in ("ruby", "pinyin"):
            self.doc.annotations[kind] = Annotation(kind, kind, "m", anchor_of(ids[:2]), "校庭", "こうてい",
                                                   status="confirmed", origin="manual")
        self.doc.karaoke["track"] = {"member": "m", "beats": [{"anchor": anchor_of(ids), "start": 0, "end": 1}]}
        before = copy.deepcopy((self.doc.annotations, self.doc.karaoke, self.doc.groups))
        self.type_mark("{@注:校庭=ガーデン}")
        m = self.doc.members["m"]
        self.assertEqual(visible_map(m)[1], ids)
        self.assertEqual((self.doc.annotations, self.doc.karaoke, self.doc.groups), before)
        self.assertTrue(self.ed.history.done[-1].keeps_text)
        self.assertIsNone(plain_range_to_visible(m, 2, 2 + len("{@注:校庭=ガーデン}")))
        self.assertEqual(plain_range_to_visible(m, len(editing_text(m))-1, len(editing_text(m))), (2, 3))
        engine = Mock()
        engine.analyze.return_value = []
        reading_spans(self.doc, m, engine)
        self.assertNotIn("ガーデン", str(engine.mock_calls))
        cursor = self.widget.jp_edit.textCursor()
        cursor.setPosition(len(editing_text(m)))
        self.widget.jp_edit.setTextCursor(cursor)
        self.assertEqual(self.widget.jp_cursor(), 3)
        self.assertEqual(ordinary_offset(editing_text(m), 5), 2)

    def test_other_braces_and_unclosed_marks_stay_text(self):
        text = "校庭{ordinary}{\\b1}{@private:x}{@注:unfinished"
        self.ed.replace_editor_text("m", text)
        self.assertEqual(self.ed.visible_text("m"), text)
        self.assertTrue(all(r.kind == "text" for r in self.doc.members["m"].runs))
        self.assertEqual(editing_text(self.doc.members["m"]), text)

    def test_adjacent_marks_at_start_middle_end_preserve_order(self):
        for pos in (0, 2, 3):
            raw = "校庭へ"[:pos] + "{@g:=One}{@注:=Two}" + "校庭へ"[pos:]
            self.ed.replace_editor_text("m", raw)
            self.assertEqual(self.doc.members["m"].text, raw)
            self.assertEqual(self.ed.visible_text("m"), "校庭へ")

    def test_existing_ass_tags_survive_mark_edit(self):
        m = self.doc.members["m"]
        m.runs.insert(0, Run(r"{\fnArial}", "tag", ins="tag", b=10))
        tag = copy.deepcopy(m.runs[0])
        self.ed.replace_editor_text("m", "校庭{@注:校庭=Garden}へ")
        self.assertIn(tag, self.doc.members["m"].runs)
        self.assertEqual(editing_text(self.doc.members["m"]), "校庭{@注:校庭=Garden}へ")

    def test_invalid_marks_kept_and_check_export_reported(self):
        mark = "{@注:校庭=Garden;0}"
        self.type_mark(mark)
        expected = "旁注写法有误：" + mark + "（位置要写 1–9）"
        self.assertEqual(issue_lines(validate(self.doc, {}), self.doc, {}), [expected])
        self.assertEqual(editing_text(self.doc.members["m"]), "校庭"+mark+"へ")
        with tempfile.TemporaryDirectory() as folder:
            ass, report = build_ass(self.doc, {}, self.template, Path(folder), ExportOptions(ruby=False))
        self.assertEqual(report.warnings, [expected])
        self.assertEqual(len(ass.events), 1)
        self.assertEqual(ass.events[0].text, "校庭へ")

    def test_editor_typed_gloss_exports_real_independent_event(self):
        self.type_mark("{@注:校庭=Garden}")
        with tempfile.TemporaryDirectory() as folder:
            ass, report = build_ass(self.doc, {}, self.template, Path(folder), ExportOptions(ruby=False))
        self.assertEqual(ass.events[0].text, "校庭へ")
        self.assertEqual(len(ass.events), 2)
        self.assertTrue(ass.events[1].text.endswith("Garden"))
        self.assertEqual(report.owners[1]["kind"], "gloss")
        self.assertEqual((ass.events[1].start, ass.events[1].end), (100, 300))

    def test_inline_grey_commit_reading_selection_and_cancel(self):
        inline = InlineTextEditor(self.doc, ["m"], "m", QRect(0, 0, 400, 100), 24)
        inline.committed.connect(self.ed.replace_editor_text)
        try:
            inline.edit.end(False)
            type_text(inline.edit, "{@g:校庭=Garden}")
            inline._apply()
            self.assertEqual(self.ed.visible_text("m"), "校庭へ")
            inline.refresh()
            self.assertEqual(inline.edit.text(), "校庭へ{@g:校庭=Garden}")
            self.app.processEvents()
            self.assertTrue(any(f.format.foreground().color().name() == "#8c959f"
                                for f in inline.edit.document().firstBlock().layout().formats()))
            cursor = inline.edit.textCursor()
            cursor.setPosition(3)
            cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
            inline.edit.setTextCursor(cursor)
            popup = Mock()
            inline.newReadingRequested.connect(popup)
            inline._reading_for_selection()
            popup.assert_not_called()
            before = copy.deepcopy(self.doc.members)
            inline.edit.setPlainText("cancelled")
            QTest.keyClick(inline.edit, Qt.Key.Key_Escape)
            self.assertEqual(self.doc.members, before)
        finally:
            inline._cancel = True
            inline.close()
            inline.deleteLater()

    def test_missing_original_stores_tag_run(self):
        m = self.doc.members["m"]
        m.lane, m.lang = "translation", "zh-Hans"
        mids = self.ed.add_missing_language({"g": "校庭{@注:校庭=Garden}"}, "original")
        added = self.doc.members[mids[0]]
        self.assertEqual([r.text for r in added.runs if r.kind == "tag"], ["{@注:校庭=Garden}"])
        self.assertEqual(visible_map(added)[0].text, "校庭")


class GlossMainWindow(unittest.TestCase):
    setUpClass = ui.ConversionUI.__dict__["setUpClass"]
    setUp = ui.ConversionUI.setUp
    _dispose = ui.ConversionUI._dispose
    group = ui.ConversionUI.group
    select = ui.ConversionUI.select

    def test_copy_buttons_clipboard_status_focus_revision_and_f1(self):
        member = self.group("g", text="学校")
        self.select("g")
        self.window.show()
        self.app.processEvents()
        edit = self.window.editor.cn_edits[0][1]
        edit.setFocus()
        type_text(edit, "未提交")
        self.app.processEvents()
        before, revision = copy.deepcopy(self.doc), self.doc.revision
        pending = edit.toPlainText()
        expected = (
            ("旁注", "{@注:被注的字=注释文字}", "gloss"),
            ("旁注（全部参数）", "{@注:被注的字=注释文字;位置;距离;font=字体;size=字号;color=#RRGGBB}", "gloss"),
            ("Raw", "{@Raw=文字}", "display-rules"))
        for button, (label, text, anchor) in zip(self.window.editor.copy_buttons, expected):
            self.assertEqual(button.text(), label)
            self.assertEqual(button.focusPolicy(), Qt.FocusPolicy.NoFocus)
            self.assertTrue(button.toolTip())
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            self.assertEqual(QApplication.clipboard().text(), text)
            self.assertEqual(self.window.statusBar().currentMessage(), "已复制：" + text)
            self.assertIs(QApplication.focusWidget(), edit)
            self.assertEqual(edit.toPlainText(), pending)
            self.assertEqual(self.doc.revision, revision)
            self.assertEqual(self.doc, before)
            self.assertIn('id="' + anchor + '"', help_ui.manual_path().read_text(encoding="utf-8"))
            with patch.object(help_ui, "open_manual") as opened:
                QTest.keyClick(button, Qt.Key.Key_F1)
                opened.assert_called_once_with(anchor)
            with patch.object(QApplication, "widgetAt", return_value=button), patch.object(help_ui, "open_manual") as opened:
                QTest.keyClick(edit, Qt.Key.Key_F1)
                opened.assert_called_once_with(anchor)
        edit.set_text("学校")

    def test_buttons_wrap_the_selected_text_in_place(self):
        member = self.group("g", text="糧食庫就是怪物儲藏食物的地方吧")
        self.select("g")
        self.window.show()
        self.app.processEvents()
        edit = self.window.editor.cn_edits[0][1]
        gloss, full, raw = self.window.editor.copy_buttons
        QApplication.clipboard().setText("")

        def select(a, b):
            edit.setFocus()
            edit.set_text("糧食庫就是怪物儲藏食物的地方吧")
            cursor = edit.textCursor()
            cursor.setPosition(a)
            cursor.setPosition(b, QTextCursor.MoveMode.KeepAnchor)
            edit.setTextCursor(cursor)
            self.app.processEvents()

        select(0, 3)
        QTest.mouseClick(gloss, Qt.MouseButton.LeftButton)
        self.assertEqual(edit.toPlainText(), "糧食庫{@注:糧食庫=}就是怪物儲藏食物的地方吧")
        self.assertEqual(edit.textCursor().position(), len("糧食庫{@注:糧食庫="))      # type the gloss here
        self.assertEqual(QApplication.clipboard().text(), "")
        self.assertIn("已写入：糧食庫{@注:糧食庫=}", self.window.statusBar().currentMessage())
        self.assertEqual(self.session.editor.visible_text(member.id), "糧食庫就是怪物儲藏食物的地方吧")   # not committed
        type_text(edit, "儲藏室")
        QTest.keyClick(edit, Qt.Key.Key_Return)
        self.assertEqual(edit.toPlainText(), "糧食庫{@注:糧食庫=儲藏室}就是怪物儲藏食物的地方吧")
        self.assertEqual(self.session.editor.visible_text(member.id), "糧食庫就是怪物儲藏食物的地方吧")

        select(7, 9)
        QTest.mouseClick(raw, Qt.MouseButton.LeftButton)
        self.assertEqual(edit.toPlainText(), "糧食庫就是怪物{@Raw=儲藏}食物的地方吧")
        self.assertEqual(edit.textCursor().position(), len("糧食庫就是怪物{@Raw=儲藏}"))

        select(0, 3)
        QTest.mouseClick(full, Qt.MouseButton.LeftButton)
        style = self.template.style(member.style)
        self.assertTrue(edit.toPlainText().startswith(
            f"糧食庫{{@注:糧食庫=;8;0;font={style.fontname};size={style.fontsize / 2:g};color=#"), edit.toPlainText())

        select(3, 3)                                     # nothing selected: copy as before
        QTest.mouseClick(raw, Qt.MouseButton.LeftButton)
        self.assertEqual(QApplication.clipboard().text(), "{@Raw=文字}")
        self.assertEqual(edit.toPlainText(), "糧食庫就是怪物儲藏食物的地方吧")
        edit.set_text("糧食庫{@注:糧食庫=x}就是")            # a selection holding a mark: copy instead
        cursor = edit.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(8, QTextCursor.MoveMode.KeepAnchor)
        edit.setTextCursor(cursor)
        QTest.mouseClick(gloss, Qt.MouseButton.LeftButton)
        self.assertEqual(edit.toPlainText(), "糧食庫{@注:糧食庫=x}就是")
        self.assertEqual(QApplication.clipboard().text(), "{@注:被注的字=注释文字}")
        edit.set_text(self.session.editor.visible_text(member.id))

    def test_translation_real_commit_skips_pinyin_refresh(self):
        member = self.group("a", text="学校")
        self.select("a")
        self.project.data["settings"] = {"pinyin": {"mode": "PY"}}
        self.session._reannotate = Mock()
        ids = visible_map(member)[1]
        self.doc.annotations["p"] = Annotation("p", "pinyin", member.id, anchor_of(ids), "学校", "xue2 xiao4",
                                               status="confirmed", origin="manual")
        annotations = copy.deepcopy(self.doc.annotations)
        before = copy.deepcopy(self.doc.groups)
        edit = self.window.editor.cn_edits[0][1]
        cursor = edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        edit.setTextCursor(cursor)
        type_text(edit, "{@注:学校=学术}")
        QTest.keyClick(edit, Qt.Key.Key_Return)
        self.assertEqual(self.session.editor.visible_text(member.id), "学校")
        self.assertEqual(self.doc.groups, before)
        self.assertEqual(self.doc.annotations, annotations)
        self.session._reannotate.assert_not_called()
        self.assertEqual(len(self.session.editor.history.done), 1)
        self.session.editor.history.undo()
        self.assertEqual(editing_text(self.doc.members[member.id]), "学校")


if __name__ == "__main__":
    unittest.main()
