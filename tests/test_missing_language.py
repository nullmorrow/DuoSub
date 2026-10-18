"""L1 identity, readings, atomic history and offscreen entry points."""
import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QInputMethodEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from subkit.core.ops import Editor
from subkit.model.document import Document, Group, Member, Run
from subkit.model.serialize import document_to_json, save_document, load_document
from subkit.model.validate import validate
from subkit.model.source import Source, SourceEvent
from subkit_app.views.missing_language import MissingLanguageDialog, distribute_lines
from tests import test_convert_ui as fixtures


def snapshot(doc):
    data = copy.deepcopy(document_to_json(doc))
    data.pop("revision", None)
    return data


class Core(unittest.TestCase):
    def make(self, kind="dialogue", lane="translation", style=None):
        doc = Document("E1", groups=[Group("g_a", kind)])
        text = "歌词" if lane == "translation" else "学校"
        m = Member("m_a", "g_a", "zh-Hans" if lane == "translation" else "ja", lane,
                   "song" if kind == "song" else "dialogue", 1234, 5678, layer=4,
                   style=style or ("ED.CN" if kind == "song" else "CN"),
                   runs=[Run(text, ins="i_existing", b=len(text))])
        doc.members[m.id] = m
        doc.groups[0].members.append(m.id)
        return doc

    def test_readings_validation_roundtrip_exact_history(self):
        doc = self.make("song")
        editor = Editor(doc, {})
        before = snapshot(doc)
        mids = editor.add_missing_language({"g_a": "学校へ"}, "original")
        m = doc.members[mids[0]]
        self.assertEqual((m.start, m.end, m.style, m.layer, m.role), (1234, 5678, "ED.JP", 4, "song"))
        self.assertEqual(doc.groups[0].members, [m.id, "m_a"])
        self.assertIsNone(m.runs[0].src)
        self.assertTrue(m.runs[0].ins.startswith("i_"))
        self.assertTrue(any(a.kind == "ruby" and a.member == m.id and a.text for a in doc.annotations.values()))
        self.assertTrue(validate(doc, {}).ok)
        after = snapshot(doc)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "doc.json"
            save_document(doc, path)
            self.assertEqual(snapshot(load_document(path)), after)
        self.assertEqual(len(editor.history.done), 1)
        self.assertEqual(editor.history.done[0].label, "补上原声（日语）")
        editor.history.undo()
        self.assertEqual(snapshot(doc), before)
        editor.history.redo()
        self.assertEqual(snapshot(doc), after)

    def test_pinyin_setting_on_off_and_dialogue_styles(self):
        for enabled in (True, False):
            doc = self.make(lane="original")
            editor = Editor(doc, {}, pinyin_words_factory=lambda lang: {} if enabled else None)
            mid = editor.add_missing_language({"g_a": "学校"}, "translation",
                    styles={"translation/dialogue": "Custom.CN"})[0]
            m = doc.members[mid]
            self.assertEqual((m.layer, m.style, m.role), (1, "Custom.CN", "dialogue"))
            self.assertEqual(bool(doc.annotations), enabled)
            after = snapshot(doc)
            editor.history.undo()
            editor.history.redo()
            self.assertEqual(snapshot(doc), after)

    def test_multi_skip_empty_existing_atomic_invalid_and_no_source_mutation(self):
        doc = self.make()
        second = self.make("song")
        g = second.groups[0]
        g.id, g.members = "g_b", ["m_b"]
        m = second.members["m_a"]
        m.id, m.group = "m_b", g.id
        doc.groups.append(g)
        doc.members[m.id] = m
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source.srt"
            source.write_bytes("1\r\n00:00:01,234 --> 00:00:05,678\r\n歌词\r\n".encode("utf-8"))
            sha = hashlib.sha256(source.read_bytes()).hexdigest()
            src = Source("cn1", "zh-Hans", "translation", "srt", str(source), "source.srt", sha,
                         source.stat().st_size, "utf-8", "crlf", events=[SourceEvent(0, "Dialogue", 1234, 5678, "歌词")])
            doc.members["m_a"].runs = [Run("歌词", src="cn1", e=0, b=2)]
            source_before = copy.deepcopy(src.events_json())
            editor = Editor(doc, {src.id: src})
            before = snapshot(doc)
            with self.assertRaises(KeyError):
                editor.add_missing_language({"g_a": "学校", "bad": "学校"}, "original")
            self.assertEqual(snapshot(doc), before)
            ids = editor.add_missing_language({"g_a": "学校", "g_b": "学校"}, "original")
            self.assertEqual(len(ids), 2)
            self.assertEqual(len(editor.history.done), 1)
            self.assertNotEqual(doc.members[ids[0]].runs[0].ins, doc.members[ids[1]].runs[0].ins)
            self.assertEqual(editor.add_missing_language({"g_a": "别改", "g_b": " "}, "original"), [])
            self.assertEqual(len(editor.history.done), 1)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), sha)
            self.assertEqual(src.events_json(), source_before)
            self.assertTrue(validate(doc, {src.id: src}).ok)
            editor.history.undo()
            self.assertEqual(snapshot(doc), before)

    def test_clipboard_more_fewer_empty_and_crlf(self):
        self.assertEqual(distribute_lines("甲\r\n\r\n乙\r\n丙", 2), (["甲", "乙"], "粘贴了 3 行，选中了 2 组：多的没有用"))
        self.assertEqual(distribute_lines("甲\n", 3), (["甲", "", ""], "粘贴了 1 行，选中了 3 组：少的空着"))
        self.assertEqual(distribute_lines("甲\n乙", 2), (["甲", "乙"], ""))

    def test_top_sign_info_english_and_delete_inserted_member(self):
        for kind, role in (("dialogue", "top"), ("sign", "sign"), ("info", "comment")):
            doc = self.make(kind)
            ref = doc.members["m_a"]
            ref.role, ref.style = role, "Custom.Existing"
            editor = Editor(doc, {})
            before = snapshot(doc)
            mid = editor.add_missing_language({"g_a": "school"}, "original", languages={"original": "en"},
                    styles={"original/top": "Custom.Top"})[0]
            m = doc.members[mid]
            self.assertEqual(m.role, "info" if kind == "info" else role)
            self.assertEqual(m.style, "Custom.Top" if kind == "dialogue" else "Custom.Existing")
            self.assertEqual(m.lang, "en")
            self.assertFalse(doc.annotations)
            self.assertTrue(validate(doc, {}).ok)
            after = snapshot(doc)
            editor.delete_member(mid)
            self.assertNotIn(mid, doc.members)
            self.assertFalse(doc.tombstones)
            editor.history.undo()
            self.assertEqual(snapshot(doc), after)
            editor.history.undo()
            self.assertEqual(snapshot(doc), before)


class UI(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    setUp = fixtures.ConversionUI.setUp
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group
    select = fixtures.ConversionUI.select
    context_menu = fixtures.ConversionUI.context_menu

    def test_menu_and_empty_editor_enter_one_step(self):
        self.group("a")
        self.select("a")
        self.window._flush_group_selection()
        actions = self.context_menu()
        add = [a for a in actions if a.text().startswith("补上")]
        self.assertEqual(len(add), 1)
        self.assertTrue(add[0].isEnabled())
        self.assertIn("补上原声", add[0].text())
        self.assertFalse(any("补上翻译" in a.text() for a in actions))
        edit = self.window.editor.missing_edits["original"]
        self.assertFalse(edit.isHidden())
        self.assertIn("这一组没有原声", edit.placeholderText())
        edit.setText("学校")
        QTest.keyClick(edit, Qt.Key.Key_Return)
        self.assertEqual(len(self.session.editor.history.done), 1)
        self.assertEqual(len(self.doc.groups[0].members), 2)
        after = snapshot(self.doc)
        self.session.editor.history.undo()
        self.session.editor.history.redo()
        self.assertEqual(snapshot(self.doc), after)

    def test_single_tool_ime_event_enter_and_cancel(self):
        self.group("a")
        self.window._add_language_dialog(["a"], "original")
        d = self.window.findChild(MissingLanguageDialog)
        self.assertEqual(d.windowType(), Qt.WindowType.Tool)
        self.assertEqual(d.help_anchor, "review")
        event = QInputMethodEvent()
        event.setCommitString("学校")
        QApplication.sendEvent(d.inputs[0], event)
        self.assertEqual(d.inputs[0].text(), "学校")
        QTest.keyClick(d.inputs[0], Qt.Key.Key_Return)
        self.assertEqual(len(self.session.editor.history.done), 1)
        self.assertTrue(any(a.kind == "ruby" for a in self.doc.annotations.values()))
        d2 = MissingLanguageDialog(self.doc, ["a"], "original", self.project.languages, self.window)
        self.addCleanup(d2.deleteLater)
        d2.reject()
        self.assertEqual(len(self.session.editor.history.done), 1)

    def test_translation_empty_row_and_navigation_reset(self):
        self.project.pinyin_mode = "off"
        m = self.group("a")
        m.lang, m.lane, m.style = "ja", "original", "JP"
        self.select("a")
        self.window._flush_group_selection()
        self.window.editor.set_languages({"original": "ja", "translation": "zh-Hant"})
        self.window.editor.set_group(self.doc, self.doc.groups[0], None, None)
        edit = self.window.editor.missing_edits["translation"]
        self.assertFalse(edit.isHidden())
        self.assertIn("翻译（繁中）", edit.placeholderText())
        edit.setText("学校")
        QTest.keyClick(edit, Qt.Key.Key_Return)
        self.assertEqual(len(self.doc.groups[0].members), 2)
        self.window.editor.set_group(self.doc, None, None, None)
        self.assertTrue(all(e.isHidden() for e in self.window.editor.missing_edits.values()))

    def test_multi_table_filter_paste_ctrl_v_readonly_and_blank(self):
        self.group("a")
        self.group("b")
        self.group("c")
        self.session.editor.add_missing_language({"c": "学校"}, "original")
        d = MissingLanguageDialog(self.doc, ["c", "b", "a"], "original", self.project.languages, self.window)
        self.addCleanup(d.deleteLater)
        self.assertEqual([g.id for g in d.groups], ["a", "b"])
        self.assertFalse(d.table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable)
        QApplication.clipboard().setText("甲\n\n乙\n丙")
        QTest.keyClick(d.inputs[0], Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(d.texts(), {"a": "甲", "b": "乙"})
        self.assertIn("多的没有用", d.note.text())
        QApplication.clipboard().setText("甲")
        d.paste()
        self.assertEqual(d.texts(), {"a": "甲"})
        self.assertIn("少的空着", d.note.text())

    def test_watch_chip_click_hide_and_empty_double_click(self):
        self.group("a", "song")
        self.window.watch_mode = True
        self.addCleanup(setattr, self.window, "watch_mode", False)
        watch = self.window.watch
        watch.ass = self.template
        watch.selected = "a"
        self.window.player.resize(1000, 700)
        self.window.player.osd = {"w": 1000, "h": 700}
        watch._draw_missing_buttons()
        button = watch.missing_buttons["original"]
        self.assertFalse(button.isHidden())
        self.assertTrue(button.text().startswith("＋ 补上原声"))
        self.assertTrue(self.window.player.rect().contains(button.geometry()))
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        self.assertIsNotNone(self.window.findChild(MissingLanguageDialog))
        watch.selected = None
        watch._draw_missing_buttons()
        self.assertTrue(button.isHidden())
        with patch.object(watch, "_hit", return_value=(None, None)), patch.object(watch, "_ready", return_value=True), \
                patch.object(self.window, "set_watch_mode"):
            # Existing double-click signal remains wired to leave watch mode.
            events = []
            watch.doubleClickedEmpty.connect(lambda: events.append(True))
            watch._double(QPointF(10, 10), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
            self.assertEqual(events, [True])
