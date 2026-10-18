"""R4: defaults, scoped undo, recorded Utena rows and speaker eligibility."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox
from PySide6.QtCore import Qt
from subkit.core.people import CharacterBook, NameForm, file_name_labels, needs_speaker
from subkit.core.bangumi import apply_import, build_candidates
from subkit.core.ops import Editor
from subkit.core.self_check import check
from subkit.model.document import Document, Group, Member, Run
from subkit.model.source import Source, SourceEvent
from subkit_app.views.people import ScopeEditor, BangumiImportDialog
from subkit_app.views.group_table import GroupTableModel, C_SPEAKER
from tests.test_self_check import SelfCheckTests


class LineDecision(SelfCheckTests):
    def test_line_exception_hints_and_undo(self):
        self.member("m", "私とあなたの長い台詞の続きです")
        self.annotate("m", 0, 1, "あたし")
        row = self.check()["mismatches"][0]
        raw = self.book.path.read_bytes()
        before = copy.deepcopy(self.doc.annotations)
        self.editor.keep_self_reference(row, line_exception=True)
        ref = self.book.character(self.ch.id).self_refs[-1]
        self.assertEqual(ref["scope"], {"line": "m"})
        self.assertEqual(ref["display_hint"], {"episode": "E1", "number": 1, "text": "私とあなたの長い台詞の続"})
        self.assertEqual(len(self.editor.history.done), 1)
        self.editor.history.undo()
        self.assertEqual(self.book.path.read_bytes(), raw)
        self.assertEqual(self.doc.annotations, before)
        self.editor.history.redo()
        self.assertEqual(self.book.character(self.ch.id).self_refs[-1], ref)


class R4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "characters.json"

    def test_default_original_language_once_and_removed_undo(self):
        book = CharacterBook.load(self.path, original="ja")
        ch = book.add_character({"ja": "太郎"})
        self.assertEqual(ch.self_refs, [{"surface": "私", "reading": "わたし", "scope": {}}])
        ed = Editor(Document("E1"), {}, character_book=book)
        raw = book.path.read_bytes()
        ed.edit_characters(lambda b: b.update_character(ch.id, self_refs=[]))
        self.assertTrue(book.character(ch.id).extra["default_self_ref_removed"])
        self.assertEqual(CharacterBook.load(self.path, original="ja").character(ch.id).self_refs, [])
        ed.history.undo()
        self.assertEqual(book.path.read_bytes(), raw)
        ed.history.redo()
        self.assertEqual(book.character(ch.id).self_refs, [])
        en = CharacterBook(Path(self.temp.name) / "en.json", original="en")
        self.assertEqual(en.add_character({"ja": "太郎"}).self_refs, [])

    def test_existing_custom_reading_and_one_time_fill(self):
        book = CharacterBook(self.path)
        ch = book.add_character({"ja": "太郎"})
        other = book.add_character({"ja": "次郎"}, self_refs=[{"surface": "私", "reading": "わたくし", "scope": {}}])
        book = CharacterBook.load(self.path, original="ja")
        self.assertEqual(book.character(ch.id).self_refs[0]["reading"], "わたし")
        self.assertEqual(book.character(other.id).self_refs[0]["reading"], "わたくし")
        raw = self.path.read_bytes()
        CharacterBook.load(self.path, original="ja")
        self.assertEqual(self.path.read_bytes(), raw)

    def test_source_name_threshold_and_comment_denominator(self):
        src = Source("cn", "zh-Hant", "translation", "ass", "", "", "", 0, "utf-8", "crlf")
        src.events = [SourceEvent(i, "Dialogue", 0, 1, "台词", name="NTP" if i < 20 else "") for i in range(40)]
        self.assertEqual(file_name_labels(src), {"NTP"})
        src.events.append(SourceEvent(40, "Comment", 0, 1, "备注"))
        self.assertEqual(file_name_labels(src), {"NTP"})
        src.events.append(SourceEvent(41, "Dialogue", 0, 1, "台词"))
        self.assertEqual(file_name_labels(src), set())
        src.events = src.events[:19]
        self.assertEqual(file_name_labels(src), set())

    def test_file_labels_and_non_dialogue_no_pending_or_unknown(self):
        book = CharacterBook(self.path)
        src = Source("ja", "ja", "original", "ass", "", "", "", 0, "utf-8", "crlf",
                     events=[SourceEvent(i, "Dialogue", 0, 1, "私", name="NTP") for i in range(20)])
        g = Group("g", members=["m"])
        m = Member("m", "g", "ja", "original", "dialogue", 0, 1,
                   speaker={"label": "NTP"}, runs=[Run("私", src="ja", e=0, b=1)])
        doc = Document("E", groups=[g], members={"m": m})
        book.link_document(doc, sources={"ja": src})
        self.assertEqual(book.pending, [])
        self.assertEqual(check(doc, book)["unknown_speakers"], [])
        model = GroupTableModel(doc, character_book=book)
        self.assertEqual(model.index(0, C_SPEAKER).data(), "")
        self.assertEqual(m.speaker["label"], "NTP")
        m.speaker["by"] = "user"
        self.assertFalse(book.speaker_ignored(m))
        for kind in ("song", "sign", "info"):
            g.kind = kind
            self.assertFalse(needs_speaker(g, m))
            self.assertEqual(check(doc, book)["unknown_speakers"], [])
            self.assertEqual(model.index(0, C_SPEAKER).data(), "")
        g.kind = "dialogue"
        for role in ("dialogue", "top", "alt"):
            m.role = role
            self.assertTrue(needs_speaker(g, m))

    def test_scope_choices_and_read_only_hint(self):
        scope = ScopeEditor({})
        self.assertEqual([scope.kind.itemText(i) for i in range(scope.kind.count())], ["全部", "季", "集"])
        line = ScopeEditor({"line": "m_internal"}, {"episode": "E1", "number": 2, "text": "私です"})
        self.assertEqual(line.findChild(QLabel).text(), "第 E1 第 2 句：私です")
        self.assertEqual(line.scope(), {"line": "m_internal"})
        old = ScopeEditor({"line": "missing"})
        self.assertEqual(old.findChild(QLabel).text(), "某一句（找不到了）")
        for widget in (scope, line, old):
            widget.close()
            widget.deleteLater()

    def test_utena_recorded_shape_and_empty_guard(self):
        fixture = json.loads(Path(__file__).with_name("bangumi_utena_fixture.json").read_text(encoding="utf-8"))
        book = CharacterBook(self.path, original="ja")
        languages = {"original": "ja", "translation": "zh-Hant"}
        data = (fixture["subject"], fixture["characters"], fixture["details"])
        rows = build_candidates(*data, book, languages)
        self.assertEqual(len(rows), 33)
        self.assertFalse(any(r["missing_name"] for r in rows))
        cid = apply_import(book, rows, [0])[0]
        self.assertEqual(book.character(cid).self_refs[0]["reading"], "わたし")
        # Preserve the recorded list/detail structure; blank names are synthetic.
        chars = [{**fixture["characters"][0], "name": "  "}]
        details = {chars[0]["id"]: {"name": " ", "infobox": []}}
        data = (fixture["subject"], chars, details)
        empty = build_candidates(*data, book, languages)
        self.assertFalse(empty[0]["checked"])
        raw = book.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "这一行没有名字，请先填名字"):
            apply_import(book, empty, [0])
        self.assertEqual(book.path.read_bytes(), raw)
        dialog = BangumiImportDialog(None, data, book, languages, lambda ch: ch.full_name("ja"))
        self.assertEqual(dialog.table.item(0, 1).text(), "缺名字")
        with patch.object(QMessageBox, "warning") as warning:
            dialog.table.item(0, 0).setCheckState(Qt.CheckState.Checked)
            warning.assert_called_once()
        self.assertEqual(dialog.chosen(), [])
        empty[0]["names"] = {"ja": NameForm(given="補った名前").with_defaults("ja")}
        self.assertTrue(apply_import(book, empty, [0]))
        dialog.close()
        dialog.deleteLater()
