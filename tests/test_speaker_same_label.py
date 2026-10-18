"""KF: same-label speaker picks, atomic book history and offscreen picker wiring."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests import test_convert_ui as fixtures
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QDialog
from subkit.core.ops import Editor
from subkit.core.people import CharacterBook, NameForm
from subkit.model.document import Document, Group, Member, Run
from subkit_app.views.people import SpeakerPicker


class SameLabelCore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="subkit_same_label_")
        self.addCleanup(self.tmp.cleanup)
        self.book = CharacterBook.load(Path(self.tmp.name) / "characters.json")
        self.ch = self.book.add_character({"en": NameForm(given="Juri")})
        self.other = self.book.add_character({"en": NameForm(given="Other")})
        self.doc = Document("KF")
        self.editor = Editor(self.doc, {}, character_book=self.book)
        self.add("clicked", "a")
        self.add("local", "a", label="有栖川・回想")
        self.add("auto", "b", character=self.other.id)
        self.add("partner", "b", label="different")
        self.add("protected_partner", "b", label="different", by="user", character=None)
        self.add("unresolved", "c")
        self.add("picked", "d", by="user", character=self.other.id)
        self.add("cleared", "d", by="user")
        self.add("different", "e", label="別人")
        self.add("song", "song", kind="song")
        self.add("sign", "f", role="sign")
        self.add("comment", "f", source_kind="Comment")
        self.add("ignored", "g", src="ignored-src")
        self.book.source_labels = {"ignored-src": {"有栖川"}}

    def add(self, mid, gid, *, label="有栖川", by=None, character=None, kind="dialogue",
            role="dialogue", source_kind="Dialogue", src=None):
        group = next((g for g in self.doc.groups if g.id == gid), None)
        if group is None:
            group = Group(gid, kind=kind)
            self.doc.groups.append(group)
        speaker = {"label": label, "character": character}
        if by:
            speaker["by"] = by
        self.doc.members[mid] = Member(mid, gid, "en", "original", role, 1000, 2000,
            speaker=speaker, source_kind=source_kind, runs=[Run("hello", ins=mid, src=src, a=0, b=5)])
        group.members.append(mid)
        self.doc.structure_changed()

    def state(self):
        return copy.deepcopy(self.doc.members), copy.deepcopy(self.doc.annotations), self.book.snapshot()

    def assert_round_trip(self, before):
        after = self.state()
        self.assertEqual(len(self.editor.history.done), 1)
        self.editor.history.undo()
        self.assertEqual(self.state(), before)
        self.editor.history.redo()
        self.assertEqual(self.state(), after)

    def test_candidates_normalized_order_auto_unresolved_and_exclusions(self):
        # Member dictionary insertion order must not define document order.
        self.doc.members = dict(reversed(list(self.doc.members.items())))
        self.assertEqual(self.editor.same_label_candidates("clicked"), ["auto", "unresolved"])
        self.assertEqual(self.editor.same_label_candidates("clicked", whole_group=False),
                         ["local", "auto", "unresolved"])
        self.doc.members["clicked"].speaker["label"] = ""
        self.assertEqual(self.editor.same_label_candidates("clicked"), ["auto", "unresolved"])

    def test_character_whole_group_protects_other_user_picks_and_atomic_undo(self):
        self.book.crowd = ["有栖川・回想", "another"]
        self.book.pending = [{"label": "有栖川", "by_episode": {"KF": {"count": 3, "examples": ["hello"]}}}]
        self.book.save()
        before = self.state()
        count = self.editor.set_speaker(["clicked"], self.ch.id, same_label=True)
        self.assertEqual(count, 5)
        for mid in ("clicked", "local", "auto", "partner", "unresolved"):
            self.assertEqual(self.doc.members[mid].speaker["character"], self.ch.id)
            self.assertEqual(self.doc.members[mid].speaker["by"], "user")
        for mid in ("picked", "cleared", "protected_partner", "song", "sign", "comment", "ignored"):
            self.assertEqual(self.doc.members[mid], before[0][mid])
        self.assertEqual(self.book.character(self.ch.id).aliases, ["有栖川"])
        self.assertEqual(self.book.crowd, ["another"])
        self.assert_round_trip(before)
        self.add("future", "future", label="有栖川・心の声")
        self.book.link_document(self.doc)
        self.assertEqual(self.doc.members["future"].speaker["character"], self.ch.id)

    def test_single_selection_does_not_change_partners(self):
        before = self.state()
        self.assertEqual(self.editor.set_speaker(["clicked"], self.ch.id, whole_group=False, same_label=True), 4)
        self.assertEqual(self.doc.members["partner"], before[0]["partner"])
        self.assert_round_trip(before)

    def test_alias_not_duplicated_when_normalized_alias_or_name_matches(self):
        for name, aliases in (("Juri", ["有栖川・回想"]), ("有栖川", [])):
            with self.subTest(name=name):
                ch = self.book.character(self.ch.id)
                ch.names = {"en": NameForm(given=name).with_defaults("en")}
                ch.aliases = aliases.copy()
                self.editor.set_speaker(["clicked"], ch.id, same_label=True)
                self.assertEqual(ch.aliases, aliases)
                self.editor.history.undo()

    def test_crowd_mark_adds_once_and_round_trips(self):
        before = self.state()
        self.editor.set_speaker(["clicked"], "crowd", same_label=True)
        self.assertEqual(self.book.crowd, ["有栖川"])
        self.assertEqual(self.doc.members["auto"].speaker["character"], "crowd")
        self.assert_round_trip(before)
        self.editor.set_speaker(["clicked"], "crowd", same_label=True)
        self.assertEqual(self.book.crowd, ["有栖川"])

    def test_clear_removes_all_normalized_aliases_crowd_keeps_names(self):
        self.book.character(self.ch.id).names = {"en": NameForm(given="有栖川").with_defaults("en")}
        self.book.character(self.ch.id).aliases = ["有栖川", "keep"]
        self.book.character(self.other.id).aliases = ["有栖川・回想"]
        self.book.crowd = ["有栖川・心の声", "keep"]
        self.book.save()
        before = self.state()
        self.editor.set_speaker(["clicked"], None, same_label=True)
        for mid in ("clicked", "local", "auto", "partner", "unresolved"):
            self.assertEqual(self.doc.members[mid].speaker,
                             {**before[0][mid].speaker, "character": None, "by": "user"})
        self.assertEqual(self.book.character(self.ch.id).aliases, ["keep"])
        self.assertEqual(self.book.character(self.other.id).aliases, [])
        self.assertEqual(self.book.crowd, ["keep"])
        self.assertEqual(self.book.character(self.ch.id).full_name("en"), "有栖川")
        self.assert_round_trip(before)

    def test_flag_off_preserves_book_and_other_groups(self):
        before = self.state()
        self.editor.set_speaker(["clicked"], self.ch.id)
        self.assertEqual(self.book.snapshot(), before[2])
        self.assertEqual(self.doc.members["auto"], before[0]["auto"])
        self.assert_round_trip(before)


class SameLabelUI(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.window.player.pause = Mock()
        self.book = self.session.editor.character_book
        self.clicked = self.group("a", text="hello")
        self.clicked.lang = "en"
        self.clicked.speaker = {"label": "有栖川"}
        self.local = Member("local", "a", "en", "original", "dialogue", 1000, 3000,
            speaker={"label": "有栖川"}, runs=[Run("hi", ins="local", a=0, b=2)])
        self.doc.members[self.local.id] = self.local
        self.doc.groups[0].members.append(self.local.id)
        self.remote = self.group("b", text="hello", start=4000)
        self.remote.lang = "en"
        self.remote.speaker = {"label": "有栖川"}
        self.doc.structure_changed()

    def picker(self, label="有栖川", single=2, whole=1):
        for key, value in (("speaker_whole_group", True), ("speaker_same_label", True), ("speaker_fill_down", False)):
            self.settings.setValue(key, value)
        picker = SpeakerPicker(self.window, self.book, self.clicked, [], label=label,
                               single_count=single, whole_count=whole)
        self.addCleanup(picker.close)
        return picker

    def test_picker_default_text_counts_and_signal(self):
        picker = self.picker()
        self.assertTrue(picker.same_label.isChecked())
        self.assertFalse(picker.same_label.isHidden())
        self.assertEqual(picker.same_label.text(), "标着「有栖川」的 1 行一起改")
        picker.whole.setChecked(False)
        self.assertEqual(picker.same_label.text(), "标着「有栖川」的 2 行一起改")
        chosen = Mock()
        picker.chosen.connect(chosen)
        picker.choose("crowd")
        chosen.assert_called_once_with("crowd", False, True, False)

    def test_picker_hidden_empty_or_zero_and_toggle_restores_visibility(self):
        for label, single, whole in (("", 2, 1), ("有栖川", 0, 0), ("有栖川", 1, 0)):
            picker = self.picker(label, single, whole)
            self.assertTrue(picker.same_label.isHidden())
            picker.whole.setChecked(False)
            self.assertEqual(picker.same_label.isHidden(), not bool(label and single))
            chosen = Mock()
            picker.chosen.connect(chosen)
            picker.choose(None)
            chosen.assert_called_once_with(None, False, bool(label and single), False)

    def test_unchecked_emits_false(self):
        picker = self.picker()
        picker.same_label.setChecked(False)
        chosen = Mock()
        picker.chosen.connect(chosen)
        picker.choose(None)
        chosen.assert_called_once_with(None, True, False, False)

    def test_mainwindow_real_picker_counts_and_wiring(self):
        ch = self.book.add_character({"en": NameForm(given="Juri")})
        self.window._open_speaker_picker(self.clicked.id, QPoint(100, 100))
        picker = self.window._speaker_picker
        self.assertEqual(picker.same_label.text(), "标着「有栖川」的 1 行一起改")
        picker.whole.setChecked(False)
        self.assertEqual(picker.same_label.text(), "标着「有栖川」的 2 行一起改")
        picker.choose(ch.id)
        self.assertEqual(self.doc.members[self.remote.id].speaker["character"], ch.id)
        self.assertEqual(self.book.character(ch.id).aliases, ["有栖川"])
        self.assertEqual(len(self.session.editor.history.done), 1)

    def test_new_character_uses_displayed_label_and_one_transaction(self):
        # Same-label assignment now uses the label selected for the group display.
        self.local.speaker = {"label": "Juri"}
        self.remote.speaker = {"label": "Juri"}
        self.settings.setValue("watch/speaker_lane", "original")
        before_members, before_book = copy.deepcopy(self.doc.members), self.book.snapshot()
        class DetailsDouble:
            DialogCode = QDialog.DialogCode
            def __init__(self, parent, character, languages, label):
                self.character = character
            def exec(self):
                return QDialog.DialogCode.Accepted
            def changes(self):
                return {"names": self.character.names, "aliases": self.character.aliases}
            def deleteLater(self):
                pass
        with patch("subkit_app.mainwindow.CharacterDetailsDialog", DetailsDouble):
            self.window._set_speaker(self.clicked.id, "new", True, True)
        cid = self.doc.members[self.clicked.id].speaker["character"]
        self.assertEqual(self.doc.members[self.remote.id].speaker["character"], cid)
        self.assertIn("Juri", self.book.character(cid).aliases)
        self.assertEqual(len(self.session.editor.history.done), 1)
        after_members, after_book = copy.deepcopy(self.doc.members), self.book.snapshot()
        self.session.editor.history.undo()
        self.assertEqual(self.doc.members, before_members)
        self.assertEqual(self.book.snapshot(), before_book)
        self.session.editor.history.redo()
        self.assertEqual(self.doc.members, after_members)
        self.assertEqual(self.book.snapshot(), after_book)
