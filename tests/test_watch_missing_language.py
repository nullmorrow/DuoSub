"""F10/F11 missing dialogue lanes: real keyboard focus, empty history and provenance."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from tests import test_convert_ui as conversion
from tests import test_watch_inline_qol as watch_ui
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit
from subkit.core.export import ExportOptions, build_ass
from subkit.core.layout_index import LineGeom, Rect
from subkit.core.ops import Editor
from subkit.core.workspace import Project
from subkit.languages import DEFAULT_LANGUAGES
from subkit.model.document import Document, Group, Member, Run
from subkit.model.serialize import document_to_json, load_document, save_document
from subkit.model.source import Source, SourceEvent
from subkit.model.validate import validate


def snapshot(doc):
    data = copy.deepcopy(document_to_json(doc))
    data.pop("revision", None)
    return data


def single_dialogue(lane="translation", *, role="dialogue"):
    lang, style, text = ("ja", "JP", "原文") if lane == "original" else ("zh-Hant", "CN", "译文")
    member = Member("m_ref", "g", lang, lane, role, 1234, 5678, layer=4, style=style,
                    runs=[Run(text, ins="i_ref", b=len(text))])
    doc = Document("E1", groups=[Group("g", "dialogue", [member.id])], members={member.id: member})
    return doc, member


class EmptyMissingLanguageCore(unittest.TestCase):
    def test_default_blank_inputs_keep_previous_noop_contract(self):
        doc, _ = single_dialogue()
        factory = Mock(side_effect=AssertionError("empty input must not initialize readings"))
        editor = Editor(doc, {}, reading_engine_factory=factory)
        before = snapshot(doc)
        for text in ("", " ", "\t"):
            self.assertEqual(editor.add_missing_language({"g": text}, "original"), [])
        self.assertEqual(snapshot(doc), before)
        self.assertEqual(editor.history.done, [])
        factory.assert_not_called()

    def test_explicit_empty_both_lanes_default_languages_styles_times_and_no_annotations(self):
        for lane, opposite in (("original", "translation"), ("translation", "original")):
            with self.subTest(lane=lane):
                doc, ref = single_dialogue(opposite)
                factory = Mock(side_effect=AssertionError("empty line needs no reading engine"))
                words = Mock(side_effect=AssertionError("empty line needs no pinyin lookup"))
                editor = Editor(doc, {}, reading_engine_factory=factory, pinyin_words_factory=words)
                saved_ref = copy.deepcopy(ref)
                mid = editor.add_missing_language({"g": ""}, lane, allow_empty=True)[0]
                member = doc.members[mid]
                self.assertEqual((member.lane, member.lang, member.role, member.text),
                                 (lane, DEFAULT_LANGUAGES[lane], "dialogue", ""))
                self.assertEqual((member.start, member.end), (ref.start, ref.end))
                self.assertEqual(member.style, "JP" if lane == "original" else "CN")
                self.assertEqual(member.layer, int(lane == "translation"))
                self.assertEqual(doc.members[ref.id], saved_ref)
                self.assertFalse(doc.annotations)
                self.assertFalse(doc.tombstones)
                self.assertTrue(all(run.src is None for run in member.runs))
                self.assertTrue(validate(doc, {}).ok)
                factory.assert_not_called()
                words.assert_not_called()

    def test_custom_language_and_top_alt_style_follow_existing_missing_lane_rules(self):
        for lane, opposite, lang, role in (("original", "translation", "en", "top"),
                                            ("translation", "original", "ko", "alt")):
            with self.subTest(lane=lane):
                doc, ref = single_dialogue(opposite, role=role)
                editor = Editor(doc, {})
                mid = editor.add_missing_language({"g": ""}, lane, allow_empty=True,
                            languages={lane: lang}, styles={f"{lane}/{role}": "Custom.Missing"})[0]
                member = doc.members[mid]
                self.assertEqual((member.lang, member.role, member.style), (lang, role, "Custom.Missing"))
                self.assertEqual((member.start, member.end), (ref.start, ref.end))
                self.assertFalse(doc.annotations)

    def test_empty_creation_one_undo_step_exact_redo_and_serialization(self):
        doc, _ = single_dialogue()
        editor = Editor(doc, {})
        before = snapshot(doc)
        mid = editor.add_missing_language({"g": ""}, "original", allow_empty=True)[0]
        after = snapshot(doc)
        self.assertEqual(len(editor.history.done), 1)
        editor.history.undo()
        self.assertEqual(snapshot(doc), before)
        editor.history.redo()
        self.assertEqual(snapshot(doc), after)
        with tempfile.TemporaryDirectory(prefix="duosub_empty_lane_") as folder:
            path = Path(folder) / "document.json"
            save_document(doc, path)
            reopened = load_document(path)
            self.assertEqual(snapshot(reopened), after)
            self.assertEqual(reopened.members[mid].text, "")

    def test_repeated_empty_add_does_not_duplicate_existing_blank_or_nonblank_member(self):
        doc, _ = single_dialogue()
        editor = Editor(doc, {})
        mid = editor.add_missing_language({"g": ""}, "original", allow_empty=True)[0]
        after = snapshot(doc)
        for text in ("", "replace this"):
            self.assertEqual(editor.add_missing_language({"g": text}, "original", allow_empty=True), [])
        self.assertEqual(snapshot(doc), after)
        self.assertEqual(doc.members[mid].text, "")
        self.assertEqual(len(editor.history.done), 1)

    def test_comment_is_retained_and_new_blank_goes_before_translation(self):
        doc, ref = single_dialogue()
        comment = Member("m_comment", "g", "ja", "original", "comment", ref.start, ref.end,
                         style="JP", runs=[Run("comment", ins="i_comment", b=7)])
        doc.members[comment.id] = comment
        doc.groups[0].members.insert(0, comment.id)
        editor = Editor(doc, {})
        original = copy.deepcopy(comment)
        mid = editor.add_missing_language({"g": ""}, "original", allow_empty=True)[0]
        self.assertEqual(doc.members[comment.id], original)
        self.assertLess(doc.groups[0].members.index(mid), doc.groups[0].members.index(ref.id))
        self.assertEqual(doc.members[mid].role, "dialogue")

    def test_invalid_group_is_atomic_even_when_allowing_empty(self):
        doc, _ = single_dialogue()
        editor = Editor(doc, {})
        before = snapshot(doc)
        with self.assertRaises(KeyError):
            editor.add_missing_language({"g": "", "missing": ""}, "original", allow_empty=True)
        self.assertEqual(snapshot(doc), before)
        self.assertFalse(editor.history.done)

    def test_empty_add_and_edit_never_change_source_bytes_or_source_events(self):
        doc, ref = single_dialogue()
        with tempfile.TemporaryDirectory(prefix="duosub_empty_source_") as folder:
            path = Path(folder) / "original.srt"
            data = "1\r\n00:00:01,234 --> 00:00:05,678\r\n译文\r\n".encode("utf-8")
            path.write_bytes(data)
            sha = hashlib.sha256(data).hexdigest()
            source = Source("s_ref", ref.lang, ref.lane, "srt", str(path), "original.srt", sha,
                            len(data), "utf-8", "crlf",
                            events=[SourceEvent(0, "Dialogue", ref.start, ref.end, ref.text)])
            ref.runs = [Run(ref.text, src=source.id, e=0, b=len(ref.text))]
            saved_ref, saved_events = copy.deepcopy(ref), copy.deepcopy(source.events_json())
            editor = Editor(doc, {source.id: source})
            mid = editor.add_missing_language({"g": ""}, "original", allow_empty=True)[0]
            editor.replace_editor_text(mid, "new original")
            self.assertEqual(doc.members[ref.id], saved_ref)
            self.assertEqual(source.events_json(), saved_events)
            self.assertEqual(path.read_bytes(), data)
            self.assertTrue(all(run.src is None for run in doc.members[mid].runs))
            self.assertFalse(doc.tombstones)
            self.assertTrue(validate(doc, {source.id: source}).ok)


class WatchMissingLanguageUI(unittest.TestCase):
    setUpClass = conversion.ConversionUI.__dict__["setUpClass"]
    _dispose = conversion.ConversionUI._dispose
    group = conversion.ConversionUI.group
    close_inline = watch_ui.WatchInlineQoL.close_inline
    key = watch_ui.WatchInlineQoL.key

    def setUp(self):
        watch_ui.WatchInlineQoL.setUp(self)
        self.project.languages = dict(DEFAULT_LANGUAGES)
        self.project.pinyin_mode = "off"

    def missing(self, lane="original", *, pinned=False, geometry=True):
        member = self.jp if lane == "original" else self.cn
        self.doc.groups[0].members.remove(member.id)
        self.doc.members.pop(member.id)
        self.doc.structure_changed()
        self.window.wave_panel.btn_pin.setChecked(pinned)
        self.refresh_preview(geometry=geometry)
        return self.cn if lane == "original" else self.jp

    def refresh_preview(self, *, geometry=True):
        ass, report = build_ass(self.doc, {}, self.template, self.path, ExportOptions(ruby=False, mode="preview"))
        self.watch.set_preview(ass, report.owners, {}, None)
        self.watch.geo.clear()
        if geometry:
            for i, owner in enumerate(report.owners):
                self.watch.geo[i] = LineGeom(i, owner["member"], owner["group"], "base", 0,
                                             Rect(100, 170, 300, 220))
        self.watch._pending = None
        self.watch.request_geometry.reset_mock()
        self.watch.selected = "g"
        self.window.current_gid = "g"
        self.window.activateWindow()
        self.player.setFocus()
        self.app.processEvents()

    def create(self, lane="original"):
        before = set(self.doc.members)
        self.key(Qt.Key.Key_X if lane == "original" else Qt.Key.Key_Z)
        ed = self.window._inline
        self.assertIsNotNone(ed, self.window.statusBar().currentMessage())
        self.assertEqual(set(self.doc.members) - before, {ed.mid})
        member = self.doc.members[ed.mid]
        self.assertEqual((member.lane, member.text), (lane, ""))
        self.assertIn(member.role, ("dialogue", "top", "alt"))
        self.assertEqual(ed.edit.text(), "")
        self.assertTrue(ed.edit.isVisible())
        self.assertTrue(ed.edit.hasFocus())
        self.assertIs(QApplication.focusWidget(), ed.edit)
        return ed, member

    def assert_key_creates_and_commits(self, lane, pinned):
        ref = self.missing(lane, pinned=pinned)
        saved_ref = copy.deepcopy(ref)
        ed, member = self.create(lane)
        self.assertEqual(member.role, "dialogue")
        self.assertEqual(member.lang, DEFAULT_LANGUAGES[lane])
        self.assertEqual(member.style, "JP" if lane == "original" else "CN")
        self.assertEqual((member.start, member.end), (ref.start, ref.end))
        self.assertTrue(self.window.watch_mode)
        self.assertEqual(self.window.wave_panel.pinned(), pinned)
        # Send typing to the focus target that the real X/Z path selected.
        QTest.keyClicks(QApplication.focusWidget(), "new line")
        self.assertEqual(ed.edit.text(), "new line")
        self.key(Qt.Key.Key_Return, target=QApplication.focusWidget())
        self.assertIsNone(self.window._inline)
        self.assertEqual(self.doc.members[member.id].text, "new line")
        self.assertEqual(self.doc.members[ref.id], saved_ref)
        self.assertTrue(self.window.watch_mode)
        self.assertEqual(self.window.wave_panel.pinned(), pinned)

    def test_x_creates_original_and_focuses_input_in_f11(self):
        self.assert_key_creates_and_commits("original", False)

    def test_x_creates_original_and_focuses_input_in_f10(self):
        self.assert_key_creates_and_commits("original", True)

    def test_z_creates_translation_and_focuses_input_in_f11(self):
        self.assert_key_creates_and_commits("translation", False)

    def test_z_creates_translation_and_focuses_input_in_f10(self):
        self.assert_key_creates_and_commits("translation", True)

    def test_custom_original_language_and_style_from_project_settings(self):
        self.project.languages["original"] = "en"
        self.merge["styles"] = {"original/dialogue": "Custom.English"}
        self.missing()
        ed, member = self.create()
        self.assertEqual((member.lang, member.style), ("en", "Custom.English"))
        self.assertIn("英语", ed.head.text())

    def test_custom_translation_language_and_alt_style_from_project_settings(self):
        self.project.languages["translation"] = "ko"
        self.jp.role = "alt"
        self.merge["styles"] = {"translation/alt": "Custom.Korean.Alt"}
        self.missing("translation")
        _ed, member = self.create("translation")
        self.assertEqual((member.lang, member.role, member.style), ("ko", "alt", "Custom.Korean.Alt"))

    def test_no_measured_geometry_still_opens_now_with_safe_geometry(self):
        self.missing(geometry=False)
        ed, _member = self.create()
        self.assertTrue(ed.isVisible())
        self.assertGreater(ed.geometry().width(), 0)
        self.assertGreater(ed.geometry().height(), 0)
        self.assertTrue(ed.screen().availableGeometry().intersects(ed.frameGeometry()))
        self.assertFalse(self.watch._pending and self.watch._pending[0] == "edit_member")

    def test_preview_not_loaded_still_opens_empty_editor_immediately(self):
        self.missing(geometry=False)
        self.watch.ass = None
        self.watch.owners = []
        self.watch.geo.clear()
        # Match a controller before its first preview; stale event indices from
        # the fixture's earlier preview would be an inconsistent test state.
        self.watch._starts = []
        self.watch._max_dur = 0
        self.assertEqual(self.watch.active(), [])
        ed, _member = self.create()
        self.assertTrue(ed.isVisible())
        self.assertTrue(ed.screen().availableGeometry().intersects(ed.frameGeometry()))
        self.assertIsNone(self.watch._pending)

    def test_active_group_empty_creation_keeps_current_playhead(self):
        self.missing()
        self.assertTrue(self.watch.active())
        self.player.seek.reset_mock()
        self.create()
        self.player.seek.assert_not_called()

    def test_inactive_group_empty_creation_seeks_its_start_then_opens(self):
        self.missing()
        self.player.time = 9
        self.assertFalse(self.watch.active())
        self.player.seek.reset_mock()
        self.create()
        self.player.seek.assert_called_once_with(1.0)

    def test_empty_partner_creation_first_commits_existing_inline_draft(self):
        ref = self.missing()
        self.key(Qt.Key.Key_Z)
        current = self.window._inline
        self.assertIsNotNone(current)
        self.assertEqual(current.mid, ref.id)
        current.edit.setText("save existing draft")
        self.window.activateWindow()
        self.player.setFocus()
        self.app.processEvents()
        ed, _member = self.create()
        self.assertEqual(self.doc.members[ref.id].text, "save existing draft")
        self.assertNotEqual(ed.mid, ref.id)

    def test_escape_keeps_created_empty_line_and_discards_uncommitted_typing(self):
        self.missing()
        ed, member = self.create()
        QTest.keyClicks(QApplication.focusWidget(), "discard this")
        self.key(Qt.Key.Key_Escape, target=ed.edit)
        self.assertIsNone(self.window._inline)
        self.assertEqual(self.doc.members[member.id].text, "")
        self.assertEqual(len(self.session.editor.history.done), 1)
        self.assertTrue(self.session.dirty)
        self.assertTrue(self.window.watch_mode)
        self.assertTrue(self.player.hasFocus())

    def test_reopen_existing_empty_line_does_not_create_another(self):
        self.missing()
        ed, member = self.create()
        self.key(Qt.Key.Key_Escape, target=ed.edit)
        before = set(self.doc.members)
        self.key(Qt.Key.Key_X)
        self.assertIsNotNone(self.window._inline)
        self.assertEqual(self.window._inline.mid, member.id)
        self.assertEqual(set(self.doc.members), before)
        self.assertEqual(len(self.session.editor.history.done), 1)
        self.assertTrue(self.window._inline.edit.hasFocus())

    def test_existing_visible_line_is_edited_without_duplicate_or_history_change(self):
        before = snapshot(self.doc)
        self.key(Qt.Key.Key_X)
        self.assertIsNotNone(self.window._inline)
        self.assertEqual(self.window._inline.mid, self.jp.id)
        self.assertEqual(snapshot(self.doc), before)
        self.assertFalse(self.session.editor.history.done)

    def test_cancel_then_keyboard_undo_and_redo_restore_exact_creation(self):
        self.missing()
        before = snapshot(self.doc)
        ed, member = self.create()
        self.key(Qt.Key.Key_Escape, target=ed.edit)
        after = snapshot(self.doc)
        self.key(Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(snapshot(self.doc), before)
        self.assertNotIn(member.id, self.doc.members)
        self.window.redo()
        self.assertEqual(snapshot(self.doc), after)
        self.assertEqual(self.doc.members[member.id].text, "")

    def test_empty_member_survives_project_save_reopen_and_is_cleaned_only_on_release(self):
        self.missing()
        ed, member = self.create()
        self.key(Qt.Key.Key_Escape, target=ed.edit)
        project = Project.create(self.path / "saved-project", "空白补行")
        episode = project.add_episode("E1")
        episode.document = copy.deepcopy(self.doc)
        episode.save()
        reopened = Project.open(project.root).episode("E1")
        self.assertEqual(reopened.document.members[member.id].text, "")
        saved = snapshot(reopened.document)
        _ass, report = reopened.build_export(ruby=False)
        self.assertGreaterEqual(report.empty_events_removed, 1)
        self.assertFalse(any(owner.get("member") == member.id for owner in report.owners))
        self.assertEqual(snapshot(reopened.document), saved)

    def test_modified_shortcuts_and_non_watch_mode_do_not_create(self):
        self.missing()
        before = snapshot(self.doc)
        for modifier in (Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.AltModifier,
                         Qt.KeyboardModifier.ShiftModifier):
            self.key(Qt.Key.Key_X, modifier)
            self.assertIsNone(self.window._inline)
            self.assertEqual(snapshot(self.doc), before)
        self.window.watch_mode = False
        self.key(Qt.Key.Key_X)
        self.assertIsNone(self.window._inline)
        self.assertEqual(snapshot(self.doc), before)

    def test_typing_transform_modal_and_export_guards_do_not_create(self):
        self.missing()
        before = snapshot(self.doc)
        edit = QLineEdit(self.window)
        self.addCleanup(edit.deleteLater)
        edit.show()
        edit.setFocus()
        self.app.processEvents()
        self.key(Qt.Key.Key_X, target=edit)
        self.assertEqual(edit.text(), "x")
        self.assertEqual(snapshot(self.doc), before)
        edit.hide()
        self.player.setFocus()
        self.app.processEvents()
        self.watch.transform = object()
        self.key(Qt.Key.Key_X)
        self.watch.transform = None
        self.assertEqual(snapshot(self.doc), before)
        dialog = QDialog(self.window)
        self.addCleanup(dialog.deleteLater)
        dialog.setModal(True)
        dialog.show()
        self.app.processEvents()
        self.key(Qt.Key.Key_X, target=dialog)
        self.assertEqual(snapshot(self.doc), before)
        dialog.close()
        self.player.setFocus()
        self.app.processEvents()
        self.window._export_job = Mock()
        try:
            self.key(Qt.Key.Key_X)
        finally:
            self.window._export_job = None
        self.assertIsNone(self.window._inline)
        self.assertEqual(snapshot(self.doc), before)

    def test_song_sign_and_info_never_auto_create_missing_lane(self):
        ref = self.missing()
        group = self.doc.groups[0]
        for kind, role in (("song", "song"), ("sign", "sign"), ("info", "info")):
            with self.subTest(kind=kind):
                group.kind, ref.role = kind, role
                before = snapshot(self.doc)
                self.key(Qt.Key.Key_X)
                self.assertIsNone(self.window._inline)
                self.assertEqual(snapshot(self.doc), before)
                self.assertFalse(self.session.editor.history.done)


if __name__ == "__main__":
    unittest.main()
