"""X1a: displayed labels, fill-down boundaries and atomic no-speaker picks."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from subkit.ass import AssDocument
from subkit.core.display import DisplayRules, display_name
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.ops import Editor
from subkit.core.people import CharacterBook, NameForm, shown_speaker_label, preserve_user_speakers
from subkit.core.review import build_packet
from subkit.core.self_check import check
from subkit.model.document import Document, Group, Member, Run
from subkit.model.serialize import document_from_json, document_to_json
from subkit.styles_default import default_styles_ass
from tests import test_convert_ui as fixtures
from PySide6.QtCore import QPoint, QSettings, QCoreApplication, QEvent
from PySide6.QtWidgets import QPushButton, QDialog
from subkit_app.views.people import SpeakerPicker


class SpeakerFillDown(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="speaker_fill_")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.book = CharacterBook.load(self.root / "characters.json")
        self.old = self.book.add_character({"en": NameForm(given="ABC")})
        self.new = self.book.add_character({"en": NameForm(given="XYZ")})
        self.doc = Document("X1a")
        self.editor = Editor(self.doc, {}, character_book=self.book)

    def add(self, gid, label="ABC", cid=None, by=None, kind="dialogue", role="dialogue",
            start=None, lane="original", mid=None):
        group = next((g for g in self.doc.groups if g.id == gid), None)
        if group is None:
            group = Group(gid, kind=kind)
            self.doc.groups.append(group)
        mid = mid or gid
        start = start if start is not None else len(self.doc.groups) * 2000
        speaker = {"label": label, "character": cid}
        if by:
            speaker["by"] = by
        m = Member(mid, gid, "ja", lane, role, start, start + 1000,
                   style="JP", speaker=speaker, runs=[Run("私", ins=mid, b=1)])
        self.doc.members[mid] = m
        group.members.append(mid)
        self.doc.structure_changed()
        return m

    def state(self):
        return (copy.deepcopy(self.doc.members), copy.deepcopy(self.doc.annotations), self.book.snapshot())

    def round_trip(self, before):
        after = self.state()
        self.assertEqual(len(self.editor.history.done), 1)
        self.editor.history.undo()
        self.assertEqual(self.state(), before)
        self.editor.history.redo()
        self.assertEqual(self.state(), after)

    def test_displayed_label_preferred_lane_fallback_and_cross_lane_candidates(self):
        self.add("a", label="", mid="clicked", lane="translation")
        self.add("a", label="ABC", mid="source")
        self.add("a", label="DEF", mid="translation", lane="translation")
        self.add("b", label="ABC", lane="translation")
        self.add("c", label="ABC", kind="song")
        self.add("d", label="ABC", role="sign")
        self.assertEqual(shown_speaker_label(self.doc, self.doc.group("a")), "ABC")
        self.assertEqual(shown_speaker_label(self.doc, self.doc.group("a"), "translation"), "DEF")
        self.assertEqual(shown_speaker_label(self.doc, self.doc.group("d")), "")
        self.assertEqual(self.editor.same_label_candidates("clicked"), ["b"])
        self.assertEqual(self.editor.same_label_candidates("clicked", shown_label="ABC"), ["b"])
        self.editor.set_speaker(["clicked"], self.new.id, same_label=True, shown_label="ABC")
        self.assertEqual(self.doc.members["b"].speaker["character"], self.new.id)
        self.assertEqual(self.book.character(self.new.id).aliases, ["ABC"])

    def test_user_example_and_whole_group_even_when_single_clicked_member(self):
        for gid, label in zip("abcdef", ["ABC", "ABC", "ABC", "ABC", "", "DEF"]):
            self.add(gid, label)
        self.add("c", label="", lane="translation", mid="partner")
        before = self.state()
        self.assertEqual(self.editor.fill_down_candidates("b"), ["c", "d", "e"])
        self.assertEqual(self.editor.set_speaker(["b"], self.new.id, whole_group=False, fill_down=True), 5)
        self.assertEqual([self.doc.members[x].speaker["character"] for x in "abcdef"],
                         [None, self.new.id, self.new.id, self.new.id, self.new.id, None])
        self.assertEqual(self.doc.members["partner"].speaker["character"], self.new.id)
        self.round_trip(before)

    def test_each_stop_rule_does_not_skip_boundary(self):
        boundaries = [dict(kind="song"), dict(kind="sign"), dict(role="sign"), dict(start=13001),
                      dict(by="user"), dict(by="user", cid=self.old.id), dict(by="user", cid=self.new.id),
                      dict(label="DEF"), dict(cid=self.new.id), dict(cid="crowd"), dict(cid="nobody"),
                      dict(by="user", cid="nobody")]
        for boundary in boundaries:
            with self.subTest(boundary=boundary):
                self.doc = Document("boundary")
                self.editor = Editor(self.doc, {}, character_book=self.book)
                self.add("a", cid=self.old.id)
                self.add("b", **boundary)
                self.add("c", label="")
                self.assertEqual(self.editor.fill_down_candidates("a"), [])
                before = copy.deepcopy(self.doc.members)
                self.editor.set_speaker(["a"], self.new.id, fill_down=True)
                self.assertEqual(self.doc.members["b"], before["b"])
                self.assertEqual(self.doc.members["c"], before["c"])

    def test_gap_exactly_ten_seconds_and_gap_uses_last_accepted_group(self):
        self.add("a", start=0)
        self.add("b", start=11000)
        self.add("c", start=22000)
        self.add("d", start=33001)
        self.assertEqual(self.editor.fill_down_candidates("a"), ["b", "c"])

    def test_same_character_with_different_labels_and_empty_groups_continue(self):
        self.add("a", cid=self.old.id)
        self.add("b", label="alias", cid=self.old.id)
        self.add("c", label="")
        self.add("d", label="DEF")
        self.assertEqual(self.editor.fill_down_candidates("a"), ["b", "c"])

    def test_any_user_pick_in_partner_stops_whole_group(self):
        self.add("a")
        self.add("b")
        self.add("b", label="", by="user", lane="translation", mid="partner")
        self.assertEqual(self.editor.fill_down_candidates("a"), [])

    def test_selection_same_label_fill_down_and_book_are_one_exact_undo(self):
        self.add("a")
        self.add("b", label="")
        self.add("c", label="DEF")
        self.add("d")
        self.book.crowd = ["ABC"]
        self.book.save()
        before = self.state()
        self.assertEqual(self.editor.set_speaker(["a"], self.new.id, same_label=True, fill_down=True), 3)
        self.assertEqual(self.doc.members["c"], before[0]["c"])
        self.assertEqual(self.doc.members["d"].speaker["character"], self.new.id)
        self.round_trip(before)

    def test_nobody_without_same_label_retains_book_and_round_trips_saved_document(self):
        m = self.add("a")
        before = self.state()
        self.editor.set_speaker(["a"], "nobody")
        self.assertEqual(self.book.snapshot(), before[2])
        self.assertEqual(m.speaker, {"label": "ABC", "character": "nobody", "by": "user"})
        loaded = document_from_json(document_to_json(self.doc))
        self.assertEqual(loaded.members["a"].speaker, m.speaker)
        self.book.link_document(loaded)
        self.assertEqual(loaded.members["a"].speaker, m.speaker)
        self.assertEqual(self.book.pending, [])
        self.editor.history.undo()
        self.assertEqual(self.state(), before)

    def test_nobody_single_pick_refreshes_existing_pending_counts_and_undo(self):
        self.add("a", label="Unknown")
        self.add("b", label="Unknown")
        self.book.link_document(self.doc)
        self.assertEqual(self.book.pending[0]["count"], 2)
        before = self.state()
        self.editor.set_speaker(["a"], "nobody")
        self.assertEqual(self.book.pending[0]["count"], 1)
        self.assertEqual(self.book.not_person, [])
        self.round_trip(before)

    def test_nobody_same_label_not_person_and_exact_book_undo(self):
        self.add("a")
        self.add("b")
        self.book.crowd = ["ABC", "keep"]
        self.book.not_person = ["keep"]
        self.book.save()
        before = self.state()
        self.editor.set_speaker(["a"], "nobody", same_label=True)
        self.assertEqual(self.book.not_person, ["keep", "ABC"])
        self.assertEqual(self.book.crowd, ["keep"])
        self.assertEqual(self.book.pending, [])
        self.assertTrue(all(m.speaker["character"] == "nobody" for m in self.doc.members.values()))
        self.round_trip(before)
        self.add("future")
        self.book.link_document(self.doc)
        self.assertEqual(self.book.pending, [])
        self.assertTrue(self.book.speaker_ignored(self.doc.members["future"]))
        self.assertEqual(display_name(self.doc.members["future"], DisplayRules(), self.book), "")
        _, jl = build_packet(self.doc, episode="X1a", chars=self.book)
        self.assertNotIn("speaker", json.loads(jl.splitlines()[-1])["original"][0])

    def test_nobody_display_self_check_review_and_export_with_and_without_book(self):
        m = self.add("a", cid="nobody", by="user")
        self.assertEqual(display_name(m, DisplayRules()), "")
        self.assertEqual(self.book.display_name("nobody", "ja", "ABC", "label"), "")
        self.assertEqual(check(self.doc, self.book), {"mismatches": [], "unknown_speakers": []})
        template = AssDocument.parse(default_styles_ass())
        for book in (None, self.book):
            with self.subTest(book=book is not None):
                md, jl = build_packet(self.doc, episode="X1a", chars=book)
                self.assertNotIn("speaker", json.loads(jl.splitlines()[0])["original"][0])
                self.assertNotIn("［ABC］", md)
                for fmt in ("label", "short", "full"):
                    out, _ = build_ass(self.doc, {}, template, self.root,
                                       ExportOptions(ruby=False, name_format=fmt), character_book=book)
                    self.assertEqual([e.name for e in out.events], [""])

    def test_export_cache_invalidates_nobody_without_book(self):
        m = self.add("a")
        template = AssDocument.parse(default_styles_ass())
        cache = ExportCache()
        out, _ = build_ass(self.doc, {}, template, self.root, ExportOptions(ruby=False), cache=cache)
        self.assertEqual(out.events[0].name, "ABC")
        self.editor.set_speaker(["a"], "nobody")
        out, report = build_ass(self.doc, {}, template, self.root, ExportOptions(ruby=False), cache=cache)
        self.assertEqual(out.events[0].name, "")
        self.assertEqual(report.laid_out, 1)

    def test_preserve_nobody_through_remerge_by_source_range(self):
        old = self.add("a", cid="nobody", by="user")
        old.runs = [Run("私", src="source", e=0, a=0, b=1)]
        new = copy.deepcopy(self.doc)
        new.members["a"].speaker = {"label": "ABC"}
        preserve_user_speakers(self.doc, new)
        self.assertEqual(new.members["a"].speaker, old.speaker)

    def test_clearing_nobody_returns_to_original_label_and_removes_not_person(self):
        self.add("a")
        self.editor.set_speaker(["a"], "nobody", same_label=True)
        self.editor.set_speaker(["a"], None, same_label=True, shown_label="ABC")
        self.assertEqual(self.book.not_person, [])
        self.assertEqual(self.doc.members["a"].speaker, {"label": "ABC", "character": None, "by": "user"})


class SpeakerUI(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.project.pinyin_mode = "off"
        self.window.player.pause = Mock()
        self.book = self.session.editor.character_book
        self.clicked = self.group("a", text="hello", start=0)
        self.source = Member("source", "a", "en", "original", "dialogue", 0, 2000,
                             speaker={"label": "ABC"}, runs=[Run("hello", ins="source", b=5)])
        self.doc.members["source"] = self.source
        self.doc.groups[0].members.append("source")
        self.same = self.group("b", text="hello", start=2000)
        self.same.speaker = {"label": "ABC"}
        self.empty = self.group("c", text="hello", start=4000)
        self.boundary = self.group("d", text="hello", start=6000)
        self.boundary.speaker = {"label": "DEF"}
        self.doc.structure_changed()

    def open_picker(self):
        self.window._open_speaker_picker(self.clicked.id, QPoint(100, 100))
        return self.window._speaker_picker

    def test_displayed_label_live_union_counts_and_assignment(self):
        picker = self.open_picker()
        self.assertEqual(picker.same_label.text(), "标着「ABC」的 1 行一起改")
        self.assertEqual(picker.count_line.text(), "这次会改 2 组")
        self.assertFalse(picker.fill_down.isChecked())
        picker.fill_down.setChecked(True)
        self.assertEqual(picker.count_line.text(), "这次会改 3 组")
        picker.same_label.setChecked(False)
        self.assertEqual(picker.count_line.text(), "这次会改 3 组")
        picker.fill_down.setChecked(False)
        self.assertEqual(picker.count_line.text(), "这次会改 1 组")
        picker.same_label.setChecked(True)
        picker.fill_down.setChecked(True)
        ch = self.book.add_character({"en": NameForm(given="XYZ")})
        before = copy.deepcopy(self.doc.members), self.book.snapshot()
        picker.choose(ch.id)
        self.assertTrue(all(self.doc.members[mid].speaker["character"] == ch.id
                            for mid in (self.clicked.id, "source", self.same.id, self.empty.id)))
        self.assertNotIn("character", self.boundary.speaker)
        self.assertEqual(len(self.session.editor.history.done), 1)
        self.session.editor.history.undo()
        self.assertEqual((self.doc.members, self.book.snapshot()), before)

    def test_options_remembered_after_cancel_and_settings_reload(self):
        picker = self.open_picker()
        picker.whole.setChecked(False)
        picker.same_label.setChecked(False)
        picker.fill_down.setChecked(True)
        picker.reject()
        self.settings.sync()
        reloaded = QSettings(self.settings.fileName(), QSettings.Format.IniFormat)
        picker = SpeakerPicker(self.window, self.book, self.clicked, [], label="ABC",
                               single_count=1, whole_count=1, settings=reloaded)
        self.addCleanup(picker.close)
        self.assertFalse(picker.whole.isChecked())
        self.assertFalse(picker.same_label.isChecked())
        self.assertTrue(picker.fill_down.isChecked())
        self.assertEqual(len(self.session.editor.history.done), 0)

    def test_nobody_button_below_crowd_editor_and_watch_hide_name(self):
        picker = self.open_picker()
        buttons = picker.findChildren(QPushButton)
        labels = [button.text() for button in buttons]
        self.assertEqual(labels[labels.index("群众") + 1], "无说话人（不是对白）")
        next(button for button in buttons if button.text() == "无说话人（不是对白）").click()
        self.assertEqual(self.book.not_person, ["ABC"])
        self.assertEqual(self.window.watch._speaker_text("a"), ("", ""))
        from subkit_app.watch import OV_SPEAKER
        self.window.watch_mode = True
        self.window.act_watch_speaker.setChecked(True)
        self.window.player.osd = dict(w=1920, h=1080)
        self.window.watch.ass = self.template
        geom = SimpleNamespace(kind="base", group="a", member=self.clicked.id, index=0)
        with patch.object(self.window.watch, "_current_geoms", return_value=[geom]):
            self.window.watch._draw_speakers()
        self.window.player.overlay.assert_called_with(OV_SPEAKER, "")
        self.assertEqual(self.window.watch.speaker_hits, [])
        self.window.watch_mode = False
        self.window.editor.set_group(self.doc, self.doc.group("a"), None, None)
        for combo in self.window.editor.speaker_combos.values():
            self.assertEqual(combo.currentText(), "")
            self.assertEqual(combo.lineEdit().placeholderText(), "")
        self.assertNotIn("nobody", self.window._recent_speakers)

    def test_sign_and_song_have_no_speaker_controls_or_name_label(self):
        self.source.lang = "ja"
        for kind, role in (("dialogue", "sign"), ("sign", "dialogue"), ("song", "song")):
            group = self.doc.group("a")
            group.kind = kind
            for member in self.doc.members_of(group):
                member.role = role
                member.speaker = {"label": "FX"}
            self.window.editor.set_group(self.doc, group, None, None)
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self.assertEqual(self.window.editor.speaker_combos, {})
            self.assertTrue(self.window.editor.whole_speaker.isHidden())
            picker = self.open_picker()
            self.assertTrue(picker.same_label.isHidden())
            self.assertEqual(picker._label, "")
            picker.close()
            self.assertTrue(all(m.speaker["label"] == "FX" for m in self.doc.members_of(group)))

    def test_new_character_captured_translation_label_fill_down_one_undo(self):
        self.clicked.speaker = {"label": "DEF"}
        self.same.speaker = {"label": "DEF"}
        self.settings.setValue("watch/speaker_lane", "translation")
        picker = self.open_picker()
        self.assertEqual(picker._label, "DEF")
        picker.fill_down.setChecked(True)
        self.assertEqual(picker.count_line.text(), "这次会改 4 组")
        self.settings.setValue("watch/speaker_lane", "original")
        before = copy.deepcopy(self.doc.members), self.book.snapshot()
        class DetailsDouble:
            DialogCode = QDialog.DialogCode
            def __init__(self, parent, character, languages, label):
                self.character = character
                self.label = label
            def exec(self):
                self.character.names["zh-Hans"].given = "XYZ"
                return QDialog.DialogCode.Accepted
            def changes(self):
                return {"names": self.character.names, "aliases": self.character.aliases}
            def deleteLater(self):
                pass
        with patch("subkit_app.mainwindow.CharacterDetailsDialog", DetailsDouble):
            picker.choose("new")
        cid = self.doc.members[self.clicked.id].speaker["character"]
        self.assertEqual(self.book.character(cid).aliases, ["DEF"])
        self.assertTrue(all(m.speaker["character"] == cid for m in self.doc.members.values()))
        self.assertEqual(len(self.session.editor.history.done), 1)
        self.session.editor.history.undo()
        self.assertEqual((self.doc.members, self.book.snapshot()), before)
