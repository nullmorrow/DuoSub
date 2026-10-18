"""Offscreen People dock with writable soma copies and a fake player."""

import os
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt, QTimer
from PySide6.QtGui import QInputMethodEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QDialogButtonBox, QLineEdit, QMessageBox, QPushButton

from subkit.core.people import NameForm, Character
from subkit.core import bangumi
from subkit.core.workspace import Project
from subkit.model.document import Document, Group, Member, Run
from subkit_app import help as app_help, mainwindow as mw
from subkit_app.session import EpisodeSession
from subkit_app.views import people as ui
from subkit_app.views.group_table import C_SPEAKER, GroupFilter, GroupTableModel
from subkit.core.readings import anchor_of, visible_map
from subkit.model.document import Annotation
from tests.test_convert_ui import PlayerDouble


class PeopleUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.platformName() != "offscreen":
            raise RuntimeError("People tests require offscreen Qt")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="subkit_people_ui_")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        shutil.copytree(ROOT / "test_projects" / "soma", root / "project", copy_function=shutil.copyfile,
                        ignore=shutil.ignore_patterns("cache"))
        self.project = Project.open(root / "project")
        self.settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
        for key, value in (("MpvWidget", PlayerDouble), ("QSettings", Mock(return_value=self.settings)),
                           ("MainWindow._open_last", Mock())):
            p = patch("subkit_app.mainwindow." + key, value)
            p.start()
            self.addCleanup(p.stop)
        self.window = mw.MainWindow()
        self.window.player._hide_timer = QTimer(self.window.player)
        self.window.player._autohide = False
        self.window.player.pause = Mock()
        self.doc = Document("people-ui", groups=[Group("g_1")])
        for i, (text, label) in enumerate((("私です", "不明"), ("生徒です", "生徒A"), ("効果音", "音"))):
            mid = f"m_{i}"
            self.doc.members[mid] = Member(mid, "g_1", "ja", "original", "dialogue", 1000, 3000,
                speaker={"label": label, "character": None}, runs=[Run(text, ins=mid, a=0, b=len(text))])
            self.doc.groups[0].members.append(mid)
        self.episode = SimpleNamespace(id="people-ui", dir=root, document=self.doc, sources={}, save=Mock(),
            reading_engine=lambda: self.project.reading_engine(episode="people-ui"))
        self.session = EpisodeSession(self.project, self.episode)
        self.session.request_preview = Mock()
        self.window.project, self.window.session = self.project, self.session
        self.panel = self.window.people
        self.book = self.session.editor.character_book
        self.book.link_document(self.doc)
        self.panel.set_context(self.project, self.session)
        self.addCleanup(self.dispose)

    def test_r2b_candidate_tables_layout_tooltips_order_and_selection(self):
        from PySide6.QtWidgets import QTableWidget
        long_example = "長い例文" * 50 + "\n二行目"
        ch = self.book.add_character({"ja": NameForm(given="アルベール・子爵")})
        self.book.pending = [{"label": "アルベール", "count": 3, "examples": [long_example]},
                             {"label": "未確認", "count": 1, "examples": ["例文"]}]
        candidates = [{"text": text, "lang": "ja", "reading": text, "count": count,
                       "examples": [{"text": long_example}]} for text, count in
                      [("アキラ", 1), ("田中", 1), ("ミキ", 5), ("花子", 2)]]
        self.panel._name_cache = None
        self.panel._name_cache = None
        from subkit.core import name_finder
        # 0.11.6 caches find_names and filters known names in the panel; the fixture book knows アキラ etc.
        with patch.object(ui, "find_names", return_value=candidates), \
                patch.object(name_finder, "_excluded", return_value=set()):
            self.panel.refresh()
        for table, headers in ((self.panel.pending, ["标签", "次数", "例句", "建议"]),
                               (self.panel.found_names, ["名字", "读音", "次数", "例句"])):
            self.assertIsInstance(table, QTableWidget)
            self.assertEqual([table.horizontalHeaderItem(i).text() for i in range(4)], headers)
            self.assertFalse(table.wordWrap())
            self.assertEqual(table.textElideMode(), Qt.TextElideMode.ElideRight)
            self.assertEqual(table.editTriggers(), table.EditTrigger.NoEditTriggers)
        pending = self.panel.pending
        self.assertEqual(pending.item(0, 0).text(), "アルベール")
        self.assertEqual(pending.item(0, 3).text(), "")
        self.assertEqual(pending.item(0, 2).toolTip(), long_example)
        self.assertNotIn("\n", pending.item(0, 2).text())
        button = pending.cellWidget(0, 3)
        self.assertEqual(button.text(), "归到 アルベール・子爵")
        self.assertGreaterEqual(pending.rowHeight(0), button.sizeHint().height() + 8)
        self.assertIsNone(pending.cellWidget(0, 0))
        found = self.panel.found_names
        self.assertEqual([found.item(i, 0).text() for i in range(4)], ["ミキ", "花子", "田中", "アキラ"])
        for col in range(4):
            self.assertEqual(found.item(3, col).foreground().color().name(), "#8b949e")
            self.assertIn("只出现一次的片假名，可能不是人名", found.item(3, col).toolTip())
        self.assertIn(long_example, found.item(3, 3).toolTip())
        self.assertEqual(found.item(0, 3).toolTip(), long_example)
        pending.setCurrentCell(1, 2)
        with patch.object(self.panel, "change") as change:
            self.panel.mark_pending(True)
            book = Mock()
            change.call_args.args[0](book)
            book.mark_label.assert_called_once_with("未確認", crowd=True)
        found.setCurrentCell(2, 3)
        with patch.object(self.panel, "change") as change:
            self.panel.reject_found()
            book = Mock()
            change.call_args.args[0](book)
            book.mark_not_name.assert_called_once_with("田中")

    def test_r2b_japanese_only_character_display_everywhere(self):
        from subkit.core.export import build_ass, ExportOptions
        from subkit.core.review import build_packet
        name = "高槻枝織"
        ch = self.book.add_character({"ja": name})
        for member in self.doc.members.values():
            member.lang = "zh-Hans"
            member.speaker = {"character": ch.id, "label": "旧标签"}
        self.panel.refresh()
        self.assertEqual(self.panel.display(ch), name)
        for col in range(len(self.panel.model.languages)):
            self.assertEqual(self.panel.model.index(len(self.book.characters) - 1, col).data(), name)
        picker = ui.SpeakerPicker(self.window, self.book, self.doc.members["m_0"], [])
        details = ui.CharacterDetailsDialog(self.panel, ch, self.project.languages)
        choice = ui.CharacterChoiceDialog(self.panel, "归到人物…", self.book.characters, self.panel.display)
        check = ui.SelfCheckDialog(self.window, self.session)
        try:
            self.assertEqual(picker.list.item(picker.list.count() - 1).text(), name)
            self.assertEqual(details.windowTitle(), "人物：" + name)
            self.assertEqual(choice.choice.itemText(choice.choice.count() - 1), name)
            self.assertIn(name, check.describe({"member": "m_0", "time": 1000}))
            self.setup_speakers()
            self.assertEqual(self.window.editor.speaker_combos["m_0"].currentText(), name)
            self.assertIn(name, self.window.model.index(0, C_SPEAKER).data())
            ass, _ = build_ass(self.doc, {}, self.project.template(), Path(self.tmp.name),
                               ExportOptions(ruby=False), character_book=self.book)
            self.assertTrue(all(event.name == name for event in ass.events))
            md, jl = build_packet(self.doc, episode="test", chars=self.book)
            self.assertIn(name, md)
            self.assertNotIn(ch.id, md)
            rec = json.loads(jl.splitlines()[0])
            self.assertEqual(rec["original"][0]["speaker"]["name"], name)
            # The packet retains identity keys for answer import; display names never use them.
            self.assertEqual(rec["original"][0]["speaker"]["character"], ch.id)
            self.watch_fixture()
            self.window.speaker_lane_actions["translation"].trigger()
            self.assertIn(name, self.speaker_overlay())
        finally:
            check.done(0)
            for dialog in (picker, details, choice, check):
                dialog.close()
                dialog.deleteLater()

    def dispose(self):
        self.session.close()
        self.window.close()
        self.window.deleteLater()
        self.session.deleteLater()
        self.app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.settings.sync()

    def select(self, cid):
        self.panel.search.clear()
        self.panel.refresh()
        for row in range(self.panel.proxy.rowCount()):
            if self.panel.proxy.index(row, 0).data(Qt.ItemDataRole.UserRole) == cid:
                self.panel.table.selectRow(row)
                return
        self.fail("Character is not shown")

    def pending(self, label):
        self.panel.refresh()
        for row in range(self.panel.pending.rowCount()):
            if self.panel.pending.item(row, 0).data(Qt.ItemDataRole.UserRole)["label"] == label:
                self.panel.pending.selectRow(row)
                return
        self.fail("Pending label is not shown")

    def test_pending_suggestion_row_explicit_link_alias_and_one_step_undo(self):
        label = "アルベール"
        self.doc.members["m_0"].speaker["label"] = label
        ch = self.book.add_character({"ja": NameForm(given="アルベール・ド・モルセール子爵")})
        self.book.link_document(self.doc)
        before_file = self.book.path.read_bytes()
        before_doc = copy.deepcopy(self.doc)
        count = len(self.session.editor.history.done)
        self.pending(label)
        item = self.panel.pending.item(self.panel.pending.currentRow(), 0)
        self.assertEqual(item.text(), label)
        button = self.panel.pending.cellWidget(self.panel.pending.currentRow(), 3)
        self.assertEqual(button.text(), "归到 アルベール・ド・モルセール子爵")
        self.assertIsNone(self.doc.members["m_0"].speaker["character"])
        self.assertNotIn(label, self.book.character(ch.id).aliases)
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        self.assertEqual(self.doc.members["m_0"].speaker["character"], ch.id)
        self.assertEqual(self.doc.members["m_0"].speaker["label"], label)
        self.assertIn(label, self.book.character(ch.id).aliases)
        self.assertEqual(len(self.session.editor.history.done), count + 1)
        self.panel.refresh()
        self.assertFalse(any(self.panel.pending.item(i, 0).data(Qt.ItemDataRole.UserRole)["label"] == label
                             for i in range(self.panel.pending.rowCount())))
        self.session.editor.history.undo()
        self.assertEqual(self.book.path.read_bytes(), before_file)
        self.assertEqual(self.doc.members, before_doc.members)
        self.assertEqual(self.doc.annotations, before_doc.annotations)
        self.assertEqual(self.doc.groups, before_doc.groups)
        self.assertEqual(self.doc.karaoke, before_doc.karaoke)
        self.session.editor.history.redo()
        self.assertEqual(self.doc.members["m_0"].speaker["character"], ch.id)

    def test_pending_ambiguous_suggestion_hidden(self):
        for family in ("甲", "乙"):
            self.book.add_character({"ja": NameForm(family=family, given="不明")})
        self.pending("不明")
        item = self.panel.pending.item(self.panel.pending.currentRow(), 0)
        self.assertNotIn("可能是：", item.text())
        self.assertIsNone(self.panel.pending.cellWidget(self.panel.pending.currentRow(), 3))

    def setup_speakers(self):
        self.window.model = GroupTableModel(self.doc, character_book=self.book, languages=self.project.languages)
        self.window.proxy = GroupFilter()
        self.window.proxy.setSourceModel(self.window.model)
        self.window.table.setModel(self.window.proxy)
        self.window.wave.set_document(self.doc)
        self.window.source.set_data(self.doc, {})
        self.session.changed.connect(self.window._on_changed)
        self.window.select_group("g_1", seek=False)

    def choose_speaker(self, mid, cid):
        combo = self.window.editor.speaker_combos[mid]
        combo.setCurrentIndex(combo.findData(cid))
        combo.activated.emit(combo.currentIndex())

    def watch_fixture(self):
        from subkit.core.export import build_ass, ExportOptions
        from subkit.core.layout_index import LayoutIndex
        w = self.window
        self.doc.members["m_1"].lang, self.doc.members["m_1"].lane = "zh-Hans", "translation"
        for member in self.doc.members.values():
            member.style = "CN" if member.lane == "translation" else "JP"
        self.setup_speakers()
        ass, report = build_ass(self.doc, {}, self.project.template(), Path(self.tmp.name),
                                ExportOptions(ruby=False), character_book=self.book)
        w.player.time = 1.5
        w.player.osd = dict(w=1920, h=1080)
        with patch.object(w.watch, "request_geometry"):
            w.watch.set_preview(ass, report.owners, {m.id: m.lane for m in self.doc.members.values()}, None)
        with_index = LayoutIndex(ass, report.owners)
        try:
            bases = [i for i, owner in enumerate(report.owners) if owner["kind"] == "base"]
            w.watch.geo = {g.index: g for g in with_index.precise_geometry(bases)}
            w.watch.anchors = {i: with_index.anchor(i) for i in bases}
        finally:
            with_index.close()
        w.watch_mode = True
        w.act_watch_speaker.setChecked(True)
        return ass

    def speaker_overlay(self):
        from subkit_app.watch import OV_SPEAKER
        self.window.watch.redraw()
        return next(call.args[1] for call in reversed(self.window.player.overlay.call_args_list)
                    if call.args[0] == OV_SPEAKER)

    def test_watch_layout_translation_first_line_and_export_unchanged(self):
        from subkit.ass.text import line_tag
        ch = self.book.add_character({"ja": NameForm(family="幸平", given="創真", short="given"),
                                      "zh-Hans": NameForm(given="创真")})
        for m in self.doc.members.values():
            m.speaker["character"] = ch.id
        self.doc.members["m_1"].runs[0].text = "第一行\\N第二行"
        ass = self.watch_fixture()
        before = ass.serialize().encode("utf-8")
        overlay = self.speaker_overlay()
        self.assertIn("創真：", overlay)
        self.assertEqual(overlay.count("創真："), 1)
        geom = next(g for g in self.window.watch.geo.values() if g.member == "m_1")
        st = ass.style(ass.events[geom.index].style)
        expected = (geom._frames[0][0].x0 - .3 * st.fontsize - 2 * st.outline - 1,
                    geom.layout_origins[0][1] + geom.line_height)
        actual = line_tag(overlay, "pos").float_args()
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, delta=0.01)
        self.assertAlmostEqual(line_tag(overlay, "fs").num(), ass.style(ass.events[geom.index].style).fontsize)
        self.window.speaker_lane_actions["translation"].trigger()
        self.assertIn("创真：", self.speaker_overlay())
        self.assertEqual(ass.serialize().encode("utf-8"), before)

    def test_watch_warning_crowd_not_person_fallback_and_active_only(self):
        self.watch_fixture()
        # 0.11.5 (R17): only the "?" is red; the name uses the attached line's colours.
        self.assertIn("{\\1c&H0000FF&}?", self.speaker_overlay())
        self.assertIn(" 不明", self.speaker_overlay())
        self.doc.members["m_0"].speaker["character"] = "crowd"
        self.assertNotIn("\\1c&H808080&", self.speaker_overlay())
        self.assertIn("群众", self.speaker_overlay())
        for m in self.doc.members.values():
            m.speaker = {"character": None, "label": "音・回想"}
        self.book.mark_label("音", crowd=False)
        self.assertEqual(self.speaker_overlay(), "")
        self.book.not_person.clear()
        for m in self.doc.members.values():
            m.speaker["label"] = ""
        self.assertIn("?{\\1c&H5F1FA8&} 说话人", self.speaker_overlay())
        self.window.player.time = 10
        self.assertEqual(self.speaker_overlay(), "")

    def test_watch_click_picker_search_ime_apply_undo_escape(self):
        from PySide6.QtCore import QPointF
        ch = self.book.add_character({"ja": "太郎"}, aliases=["Taro"])
        self.watch_fixture()
        self.speaker_overlay()
        quad, mid = self.window.watch.speaker_hits[0]
        point = QPointF(sum(p[0] for p in quad) / 4, sum(p[1] for p in quad) / 4)
        self.window.watch._press(point, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        picker = self.window._speaker_picker
        self.assertTrue(picker.isVisible())
        self.assertEqual(picker.windowType(), Qt.WindowType.Tool)
        self.assertTrue(picker.whole.isChecked())
        self.window.player.pause.assert_called()
        picker.search.setText("tArO")
        self.assertFalse(picker.list.item(picker.list.count() - 1).isHidden())
        event = QInputMethodEvent()
        event.setCommitString("太郎")
        picker.search.clear()
        self.app.sendEvent(picker.search, event)
        self.assertEqual(picker.search.text(), "太郎")
        old = copy.deepcopy(self.doc.members)
        count = len(self.session.editor.history.done)
        picker.choose(ch.id)
        self.assertTrue(all(m.speaker["character"] == ch.id for m in self.doc.members.values()))
        self.assertEqual(len(self.session.editor.history.done), count + 1)
        self.session.editor.history.undo()
        self.assertEqual(self.doc.members, old)
        self.app.processEvents()
        self.window._open_speaker_picker(mid, self.window.pos())
        picker = self.window._speaker_picker
        QTest.keyClick(picker.search, Qt.Key.Key_Escape)
        self.assertFalse(picker.isVisible())

    def test_watch_toggle_lane_remembered_and_drag_follows(self):
        from subkit.ass.text import line_tag
        self.watch_fixture()
        self.window.speaker_lane_actions["translation"].trigger()
        self.assertEqual(self.settings.value("watch/speaker_lane"), "translation")
        self.assertTrue(self.settings.value("watch/speakers", type=bool))
        before = line_tag(self.speaker_overlay(), "pos").float_args()
        self.window.watch._draw_speakers({"m_1": (100, -20)})
        from subkit_app.watch import OV_SPEAKER
        after = line_tag(self.window.player.overlay.call_args.args[1], "pos").float_args()
        self.assertEqual([round(b - a, 2) for a, b in zip(before, after)], [100, -20])
        self.window.act_watch_speaker.setChecked(False)
        self.window.player.clear_overlay.assert_any_call(OV_SPEAKER)
        other = mw.MainWindow()
        try:
            self.assertFalse(other.act_watch_speaker.isChecked())
            self.assertTrue(other.speaker_lane_actions["translation"].isChecked())
        finally:
            other.close()
            other.deleteLater()

    def test_export_name_format_options_wired_and_remembered(self):
        from subkit_app.views.dialogs import ExportDialog
        self.episode.video = None
        self.episode.settings = lambda: {"release": {}}
        self.episode.validate = Mock(return_value=SimpleNamespace(ok=True))
        build_options = []
        def build_export(**options):
            build_options.append(options)
            raise RuntimeError("stop before writing export")
        self.episode.build_export = build_export
        selected = []
        def accept(dialog):
            selected.append(dialog.name_format.currentData())
            self.assertEqual([dialog.name_format.itemText(i) for i in range(5)],
                             ["常用称呼", "全名", "只写姓", "只写名", "原标签"])
            dialog.name_format.setCurrentIndex(dialog.name_format.findData("family"))
            return dialog.DialogCode.Accepted
        with patch.object(ExportDialog, "exec", accept), patch.object(QMessageBox, "warning"):
            self.window.export()
            self.window.export()
        self.assertEqual(selected, ["short", "family"])
        self.assertEqual(build_options[-1]["name_format"], "family")
        self.assertEqual(self.settings.value("export/name_format"), "family")

    def test_picker_all_lines_alias_search_recent_whole_group_and_undo(self):
        first = self.book.add_character({"ja": NameForm(family="幸平", given="創真", short="given"),
                                         "zh-Hans": NameForm(given="创真")}, aliases=["ソーマ", "Soma"])
        second = self.book.add_character({"ja": "恵"})
        self.doc.members["m_1"].lang, self.doc.members["m_1"].lane = "zh-Hans", "translation"
        self.setup_speakers()
        self.assertEqual(set(self.window.editor.speaker_combos), set(self.doc.members))
        combo = self.window.editor.speaker_combos["m_0"]
        combo.completer().setCompletionPrefix("sOmA")
        self.assertEqual(combo.completer().completionCount(), 1)
        index = combo.completer().completionModel().index(0, 0)
        self.assertEqual(combo.completer().pathFromIndex(index), "創真")
        old = copy.deepcopy(self.doc.members)
        count = len(self.session.editor.history.done)
        combo.completer().activated[type(index)].emit(index)
        self.assertTrue(all(m.speaker["character"] == first.id for m in self.doc.members.values()))
        self.assertEqual(len(self.session.editor.history.done), count + 1)
        self.assertEqual(self.window.editor.speaker_combos["m_1"].currentText(), "创真")
        self.session.editor.history.undo()
        self.assertEqual(self.doc.members, old)
        self.window.editor.whole_speaker.setChecked(False)
        self.choose_speaker("m_0", second.id)
        self.assertIsNone(self.doc.members["m_1"].speaker["character"])
        self.assertEqual(self.window.editor.speaker_combos["m_0"].itemData(0), second.id)
        self.assertFalse(self.settings.value("speaker_whole_group", True, type=bool))
        self.session.editor.history.undo()
        self.assertEqual(self.doc.members, old)

    def test_picker_crowd_clear_and_new_details_prefill_cancel(self):
        self.setup_speakers()
        self.choose_speaker("m_0", "crowd")
        self.assertTrue(all(m.speaker["character"] == "crowd" for m in self.doc.members.values()))
        self.assertEqual(self.window.editor.speaker_combos["m_0"].currentText(), "群众")
        self.choose_speaker("m_0", None)
        self.assertTrue(all(m.speaker["character"] is None and m.speaker["by"] == "user" for m in self.doc.members.values()))
        count = len(self.session.editor.history.done)
        def cancel():
            dialog = self.window.findChild(ui.CharacterDetailsDialog)
            self.assertEqual(dialog.names["ja"].value().given, "不明")
            self.assertEqual(dialog.aliases.toPlainText(), "不明")
            dialog.reject()
        QTimer.singleShot(0, cancel)
        self.choose_speaker("m_0", "new")
        self.assertEqual(len(self.session.editor.history.done), count)
        self.assertEqual(self.window.editor.speaker_combos["m_0"].currentText(), "（原标签：不明）")

    def test_picker_keyboard_alias_completion_is_one_selection(self):
        ch = self.book.add_character({"ja": NameForm(family="幸平", given="創真", short="given")}, aliases=["Soma"])
        self.setup_speakers()
        self.window.show()
        combo = self.window.editor.speaker_combos["m_0"]
        combo.setFocus()
        combo.lineEdit().selectAll()
        QTest.keyClicks(combo.lineEdit(), "Soma")
        self.app.processEvents()
        popup = combo.completer().popup()
        self.assertTrue(popup.isVisible())
        popup.setCurrentIndex(combo.completer().completionModel().index(0, 0))
        QTest.keyClick(popup, Qt.Key.Key_Return)
        self.app.processEvents()
        self.assertTrue(all(m.speaker["character"] == ch.id for m in self.doc.members.values()))
        self.assertEqual(len(self.session.editor.history.done), 1)

    def test_picker_new_character_details_applies_and_undo(self):
        self.setup_speakers()
        def accept():
            dialog = self.window.findChild(ui.CharacterDetailsDialog)
            dialog.accept()
        QTimer.singleShot(0, accept)
        self.choose_speaker("m_0", "new")
        cid = self.doc.members["m_0"].speaker["character"]
        self.assertEqual(self.book.character(cid).full_name("ja"), "不明")
        self.assertTrue(all(m.speaker["character"] == cid for m in self.doc.members.values()))
        self.session.editor.history.undo()
        # Creation remains an independent character-table decision; the pick is one undo step.
        self.assertIsNone(self.doc.members["m_1"].speaker["character"])

    def test_list_speaker_column_language_short_names_unknown_and_crowd(self):
        ch = self.book.add_character({"ja": NameForm(family="幸平", given="創真", short="given"),
                                      "zh-Hans": NameForm(given="创真")})
        self.doc.members["m_0"].speaker["character"] = ch.id
        self.doc.members["m_1"].speaker["character"] = "crowd"
        model = GroupTableModel(self.doc, character_book=self.book)
        self.assertEqual(model.headerData(C_SPEAKER, Qt.Orientation.Horizontal), "说话人")
        self.assertEqual(model.index(0, C_SPEAKER).data(), "創真 ／ 群众 ／ 音?")
        self.doc.members["m_0"].lang = "zh-Hans"
        self.assertEqual(model.index(0, C_SPEAKER).data(), "创真 ／ 群众 ／ 音?")

    def self_dialog(self):
        ch = self.book.add_character({"ja": "太郎"}, self_refs=[
            {"surface": "私", "reading": "わたくし", "scope": {}},
            {"surface": "私", "reading": "わたし", "scope": {"line": "m_1"}}])
        for mid, reading in (("m_0", "わたし"), ("m_1", "わたくし"), ("m_2", "わたし")):
            member = self.doc.members[mid]
            member.runs = [Run("私です", ins=mid, a=0, b=3)]
            member.speaker["character"] = ch.id if mid != "m_2" else None
            ids = visible_map(member)[1]
            self.doc.annotations[mid] = Annotation(mid, "ruby", mid, anchor_of(ids[:1]), "私", reading)
        self.setup_speakers()
        dialog = ui.SelfCheckDialog(self.window, self.session)
        self.addCleanup(dialog.deleteLater)
        self.addCleanup(dialog.reject)
        return dialog, ch

    def test_self_check_scene_rows_unknown_fix_keep_undo_and_help(self):
        dialog, ch = self.self_dialog()
        self.assertEqual([r["expected"] for r in dialog.rows], ["わたくし", "わたし"])
        self.assertEqual(dialog.unknown.count(), 1)
        actions = dialog.table.cellWidget(0, 1)
        fix = next(b for b in actions.findChildren(QPushButton) if b.text() == "改成 わたくし")
        fix.click()
        self.assertEqual(len(self.session.editor.history.done), 1)
        self.assertEqual(len(dialog.rows), 1)
        self.assertTrue(any(a.text == "わたくし" and a.status == "confirmed" for a in self.doc.annotations_of("m_0")))
        self.session.editor.history.undo()
        self.assertEqual(len(dialog.rows), 2)
        actions = dialog.table.cellWidget(0, 1)
        next(b for b in actions.findChildren(QPushButton) if b.text() == "只保留这一句").click()
        self.assertEqual(len(dialog.rows), 1)
        self.assertTrue(any(a.text == "わたし" and a.status == "confirmed" for a in self.doc.annotations_of("m_0")))
        with patch.object(app_help, "open_manual") as opener:
            next(b for b in dialog.findChildren(QPushButton) if b.text() == "帮助").click()
            opener.assert_called_once_with("people")

    def test_self_check_audition_jump_and_consistent_message(self):
        dialog, ch = self.self_dialog()
        self.window.player.play = Mock()
        actions = dialog.table.cellWidget(0, 1)
        next(b for b in actions.findChildren(QPushButton) if b.text() == "试听").click()
        self.window.player.seek.assert_called_with(0.5)
        self.window.player.play.assert_called_once_with()
        self.assertAlmostEqual(self.window._stop_at, 3.0 - 1001 / 24000)
        dialog.table.cellDoubleClicked.emit(0, 0)
        self.window.player.seek.assert_called_with(1.0)
        self.window.player.pause.assert_called_once_with()
        while dialog.rows:
            dialog.decide(dialog.rows[0])
        self.doc.members["m_2"].speaker["character"] = "crowd"
        dialog.refresh()
        self.assertEqual(dialog.message.text(), "自称都和人物表一致。")

    def test_self_check_episode_exception_file_undo(self):
        dialog, ch = self.self_dialog()
        button = next(b for b in dialog.table.cellWidget(0, 1).findChildren(QPushButton)
                      if b.text() == "这一集都这样读")
        self.assertTrue(button.isEnabled())
        raw = self.book.path.read_bytes()
        dialog.decide(dialog.rows[0], keep=True, exception=True)
        self.assertEqual(self.book.character(ch.id).self_refs[-1],
                         {"surface": "私", "reading": "わたし", "scope": {"episodes": ["people-ui"]}})
        self.assertEqual(len(self.session.editor.history.done), 1)
        self.session.editor.history.undo()
        self.assertEqual(self.book.path.read_bytes(), raw)
        self.assertEqual(len(dialog.rows), 2)

    def test_conversion_display_search_and_read_only_names(self):
        self.assertGreater(self.panel.model.rowCount(), 0)
        self.assertTrue(self.book.path.exists())
        ch = self.book.characters[0]
        text = self.panel.model.index(0, 0).data()
        self.assertIn(ch.full_name("ja"), text)
        self.assertIn("（", text)
        self.assertFalse(self.panel.model.flags(self.panel.model.index(0, 0)) & Qt.ItemFlag.ItemIsEditable)
        self.panel.search.setText(ch.full_name("ja"))
        self.assertGreater(self.panel.proxy.rowCount(), 0)
        self.panel.search.setText("绝不存在的姓名")
        self.assertEqual(self.panel.proxy.rowCount(), 0)
        self.assertFalse(self.panel.buttons["删除人物"].isEnabled())

    def test_inline_alias_and_note_persist_and_undo(self):
        ch = self.book.characters[0]
        col = len(self.panel.model.languages)
        before = self.book.path.read_bytes()
        self.assertTrue(self.panel.model.setData(self.panel.model.index(0, col), "不明、测试别名"))
        self.assertEqual(self.doc.members["m_0"].speaker["character"], ch.id)
        loaded = self.project.character_book()
        self.assertIn("测试别名", loaded.character(ch.id).aliases)
        self.session.editor.history.undo()
        self.assertEqual(self.book.path.read_bytes(), before)
        self.assertIsNone(self.doc.members["m_0"].speaker["character"])
        self.panel.refresh()
        self.assertTrue(self.panel.model.setData(self.panel.model.index(0, col + 2), "新备注"))
        self.assertEqual(self.project.character_book().character(ch.id).persona["note_ja"], "新备注")

    def test_pending_assign_crowd_not_person_and_undo(self):
        cid = self.book.characters[0].id
        self.pending("不明")
        with patch.object(self.panel, "choose", return_value=cid):
            self.panel.assign_pending()
        self.assertEqual(self.doc.members["m_0"].speaker["character"], cid)
        self.pending("生徒A")
        self.panel.mark_pending(True)
        self.assertEqual(self.doc.members["m_1"].speaker["character"], "crowd")
        self.pending("音")
        self.panel.mark_pending(False)
        self.assertIn("音", self.project.character_book().not_person)
        self.panel.refresh()
        self.assertEqual(self.panel.pending.rowCount(), 0)
        self.session.editor.history.undo()
        self.panel.refresh()
        self.assertEqual(self.panel.pending.rowCount(), 1)

    def test_pending_new_and_cancel(self):
        self.pending("不明")
        count = len(self.book.characters)
        with patch.object(ui.NewCharacterDialog, "exec", return_value=QDialog.DialogCode.Rejected):
            self.panel.new_pending()
        self.assertEqual(len(self.book.characters), count)
        with patch.object(ui.NewCharacterDialog, "exec", return_value=QDialog.DialogCode.Accepted):
            self.panel.new_pending()
        self.assertEqual(len(self.book.characters), count + 1)
        self.assertEqual(self.doc.members["m_0"].speaker["character"], self.book.characters[-1].id)

    def test_merge_confirmation_one_undo_restores_file_and_user_speaker(self):
        source, target = self.book.characters[:2]
        self.session.editor.set_speaker(["m_0"], source.id, whole_group=False)
        before = self.book.path.read_bytes()
        self.select(source.id)
        with patch.object(self.panel, "choose", return_value=target.id), \
                patch.object(ui.QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
            self.panel.merge_character()
        self.assertEqual(self.book.path.read_bytes(), before)
        with patch.object(self.panel, "choose", return_value=target.id), \
                patch.object(ui.QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes), \
                patch.object(self.session.editor, "merge_characters", wraps=self.session.editor.merge_characters) as merge:
            self.panel.merge_character()
        merge.assert_called_once_with(source.id, target.id)
        self.assertEqual(self.doc.members["m_0"].speaker["character"], target.id)
        self.session.editor.history.undo()
        self.assertEqual(self.book.path.read_bytes(), before)
        self.assertEqual(self.doc.members["m_0"].speaker["character"], source.id)
        self.panel.refresh()
        self.assertEqual(self.panel.selected_id(), source.id)

    def test_delete_exact_confirmation_cancel_and_undo(self):
        ch = self.book.characters[0]
        self.session.editor.set_speaker(["m_0"], ch.id, whole_group=False)
        self.select(ch.id)
        before = self.book.path.read_bytes()
        with patch.object(ui.QMessageBox, "question", return_value=QMessageBox.StandardButton.No) as question:
            self.panel.delete_character()
        self.assertEqual(question.call_args.args[2], f"删除人物「{self.panel.display(ch)}」？1 行的说话人会变回原来的标签。")
        self.assertEqual(self.book.path.read_bytes(), before)
        with patch.object(ui.QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.panel.delete_character()
        self.assertIsNone(self.doc.members["m_0"].speaker["character"])
        self.assertEqual(self.doc.members["m_0"].speaker["label"], "不明")
        self.session.editor.history.undo()
        self.assertEqual(self.book.path.read_bytes(), before)
        self.assertEqual(self.doc.members["m_0"].speaker["character"], ch.id)

    def add_names(self):
        for i in range(2):
            gid = f"g_n{i}"
            group = Group(gid)
            self.doc.groups.append(group)
            for lang, lane, text in (("ja", "original", "田中さんです"), ("zh-Hant", "translation", "田中")):
                mid = f"m_{lane}{i}"
                self.doc.members[mid] = Member(mid, gid, lang, lane, "dialogue", 4000, 5000,
                    runs=[Run(text, ins=mid, a=0, b=len(text))])
                group.members.append(mid)
        self.doc.structure_changed()
        self.panel.refresh()
        for row in range(self.panel.found_names.rowCount()):
            if self.panel.found_names.item(row, 0).data(Qt.ItemDataRole.UserRole)["text"] == "田中":
                self.panel.found_names.selectRow(row)
                return
        self.fail("Sudachi candidate not found")

    def test_found_new_prefills_translation_reading_and_preserves_text(self):
        self.add_names()
        before = {mid: m.text for mid, m in self.doc.members.items()}
        def accept(dialog):
            self.assertEqual(dialog.fields["ja"][0].text(), "田中")
            self.assertEqual(dialog.fields["ja"][1].text(), "たなか")
            self.assertEqual(dialog.fields[self.project.languages["translation"]][0].text(), "田中")
            return QDialog.DialogCode.Accepted
        with patch.object(ui.NewCharacterDialog, "exec", accept):
            self.panel.new_found()
        self.assertEqual(self.book.characters[-1].names["ja"].family, "田中")
        self.assertEqual(self.book.characters[-1].names["ja"].family_reading, "たなか")
        self.assertEqual({mid: m.text for mid, m in self.doc.members.items()}, before)

    def test_found_add_alias_or_missing_language_and_reject_persist(self):
        self.add_names()
        cid = self.book.characters[0].id
        with patch.object(self.panel, "choose", return_value=cid):
            self.panel.assign_found()
        self.assertIn("田中", self.project.character_book().character(cid).aliases)
        self.session.editor.history.undo()
        other = self.book.add_character({"en": NameForm(given="Tanaka")})
        self.add_names_selection()
        with patch.object(self.panel, "choose", return_value=other.id):
            self.panel.assign_found()
        self.assertEqual(self.project.character_book().character(other.id).full_name("ja"), "田中")
        self.session.editor.history.undo()
        self.add_names_selection()
        self.panel.reject_found()
        self.assertIn("田中", self.project.character_book().not_name)
        self.panel.refresh()
        self.assertFalse(any(self.panel.found_names.item(i, 0).data(Qt.ItemDataRole.UserRole)["text"] == "田中"
                             for i in range(self.panel.found_names.rowCount())))

    def add_names_selection(self):
        self.panel.refresh()
        for row in range(self.panel.found_names.rowCount()):
            if self.panel.found_names.item(row, 0).data(Qt.ItemDataRole.UserRole)["text"] == "田中":
                self.panel.found_names.selectRow(row)
                return
        self.fail("Candidate missing after undo")

    def test_creation_ime_tool_window_help_and_placeholder(self):
        dialog = ui.NewCharacterDialog(self.panel, self.project.languages)
        self.assertEqual(dialog.windowType(), Qt.WindowType.Tool)
        self.assertFalse(dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled())
        event = QInputMethodEvent()
        event.setCommitString("新人物")
        QCoreApplication.sendEvent(dialog.fields["ja"][0], event)
        self.assertTrue(dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled())
        self.assertEqual(dialog.names()["ja"].compose(), "新人物")
        self.assertEqual(app_help.resolve_anchor(self.window, active=self.window, focus=self.panel.search), "people")
        self.select(self.book.characters[0].id)
        with patch.object(ui.QDialog, "exec", return_value=QDialog.DialogCode.Rejected) as execute:
            self.panel.table.doubleClicked.emit(self.panel.proxy.index(0, 0))
        execute.assert_called_once()

    def test_dock_toggle_remembered_watch_restores_and_context_cleared(self):
        self.window.show()
        self.app.processEvents()
        self.panel.show()
        self.assertTrue(self.panel.toggleViewAction().isChecked())
        # 0.10.14: in 视频优先 the docks live in the right column's own main window.
        host = self.window.right_host if self.window.right_host.isAncestorOf(self.panel) else self.window
        state = host.saveState()
        self.panel.hide()
        host.restoreState(state)
        self.assertFalse(self.panel.isHidden())
        self.window.set_watch_mode(True)
        self.assertTrue(self.panel.isHidden())
        QTest.qWait(320)  # let the existing fullscreen introduction finish before destroying its widgets
        self.window.set_watch_mode(False)
        self.assertFalse(self.panel.isHidden())
        self.panel.set_context(None)
        self.assertEqual(self.panel.model.rowCount(), 0)
        self.assertEqual(self.panel.found_names.rowCount(), 0)
        self.assertFalse(self.panel.buttons["新建人物"].isEnabled())

    def test_inline_editor_accepts_input_method_commit(self):
        self.window.show()
        self.panel.show()
        self.select(self.book.characters[0].id)
        col = len(self.panel.model.languages)
        index = self.panel.proxy.index(0, col)
        self.panel.table.edit(index)
        self.app.processEvents()
        editor = self.panel.table.findChild(QLineEdit)
        self.assertIsNotNone(editor)
        editor.selectAll()
        event = QInputMethodEvent()
        event.setCommitString("输入法别名")
        QCoreApplication.sendEvent(editor, event)
        QTest.keyClick(editor, Qt.Key.Key_Return)
        self.app.processEvents()
        self.assertIn("输入法别名", self.project.character_book().characters[0].aliases)

    def test_real_soma_session_connected_and_closed(self):
        self.window.close_session()
        episode = self.project.episode(self.project.episode_ids()[0])
        with patch.object(self.window, "_load_media"), patch.object(EpisodeSession, "request_preview"):
            self.window._start_session(episode)
        self.session = self.window.session
        self.doc = self.session.doc
        self.assertIs(self.panel.book, self.session.editor.character_book)
        self.assertIs(self.panel.session, self.session)
        self.assertGreater(len(self.doc.members), 800)
        self.assertGreater(self.panel.model.rowCount(), 0)
        self.assertGreater(self.panel.found_names.rowCount(), 0)
        self.window.close_session()
        self.assertIsNone(self.panel.session)
        self.assertFalse(self.panel._refresh_timer.isActive())

    def details_dialog(self):
        ch = self.book.characters[0]
        return ui.CharacterDetailsDialog(self.panel, ch, self.project.languages, self.panel.display(ch))

    def test_details_round_trip_parts_readings_name_choice_aliases_persona_and_undo(self):
        ch = self.book.characters[0]
        self.select(ch.id)
        before = self.book.path.read_bytes()
        def edit(dialog):
            self.assertEqual(dialog.real_group.title(), "真实姓名（用于读音）")
            self.assertEqual(dialog.name_group.title(), "字幕名字栏")
            ja = dialog.names["ja"]
            for key, value in (("family", "幸平"), ("given", "創真"),
                               ("family_reading", "ユキヒラ"), ("given_reading", "そうま")):
                ja.fields[key].setText(value)
            ja.separator.setCurrentIndex(ja.separator.findData("・"))
            ja.order.setCurrentIndex(ja.order.findData("given_first"))
            ja.short.setCurrentIndex(ja.short.findData("custom"))
            ja.custom.setText("Soma @ #")
            self.assertEqual(ja.preview.text(), "創真・幸平")
            self.assertEqual(ja.short_preview.text(), "Soma @ #")
            zh = dialog.names[self.project.languages["translation"]]
            for key, value in (("family", "幸平"), ("given", "創真"),
                               ("family_reading", "xing4 ping2"), ("given_reading", "chuang4 zhen1")):
                zh.fields[key].setText(value)
            dialog.add_language("en")
            en = dialog.names["en"]
            en.fields["given"].setText("Sheldon")
            en.fields["family"].setText("Cooper")
            self.assertEqual(en.preview.text(), "Sheldon Cooper")
            self.assertTrue(en.fields["given_reading"].isHidden())
            dialog.aliases.setPlainText("不明\nソーマ\nソーマ")
            dialog.persona["note_tr"].setPlainText("测试\n第二行")
            return QDialog.DialogCode.Accepted
        with patch.object(ui.CharacterDetailsDialog, "exec", edit):
            self.panel.details()
        saved = self.project.character_book().character(ch.id)
        self.assertEqual(saved.names["ja"].compose(), "創真・幸平")
        self.assertEqual(saved.names["ja"].family_reading, "ゆきひら")
        self.assertEqual(saved.names["ja"].compose("short"), "Soma @ #")
        self.assertEqual(saved.names[self.project.languages["translation"]].family_reading, "xìng píng")
        self.assertEqual(saved.aliases, ["不明", "ソーマ"])
        self.assertEqual(saved.persona["note_tr"], "测试\n第二行")
        self.assertEqual(self.doc.members["m_0"].speaker["character"], ch.id)
        self.assertEqual(len(self.session.editor.history.done), 1)
        self.session.editor.history.undo()
        self.assertEqual(self.book.path.read_bytes(), before)
        self.assertIsNone(self.doc.members["m_0"].speaker["character"])

    def test_details_invalid_kana_pinyin_messages_and_cancel_no_writes(self):
        before = self.book.path.read_bytes()
        dialog = self.details_dialog()
        dialog.names["ja"].fields["given_reading"].setText("watashi")
        with patch.object(ui.QMessageBox, "warning") as warning:
            dialog.accept()
        self.assertIn("假名无效", warning.call_args.args[2])
        self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)
        dialog.names["ja"].fields["given_reading"].clear()
        zh = dialog.names[self.project.languages["translation"]]
        zh.fields["family_reading"].clear()
        zh.fields["given"].setText("单")
        zh.fields["given_reading"].setText("invalid9")
        with patch.object(ui.QMessageBox, "warning") as warning:
            dialog.accept()
        self.assertIn("拼音无效", warning.call_args.args[2])
        dialog.reject()
        self.assertEqual(self.book.path.read_bytes(), before)

    def test_self_refs_all_season_episodes_line_edit_save_and_confirmed_untouched(self):
        ch = self.book.characters[0]
        dialog = self.details_dialog()
        dialog.refs.setRowCount(0)
        for scope in ({}, {"season": 2}, {"episodes": ["e1", "e2"]}, {"line": "m_0"}):
            dialog.add_ref({"surface": "私", "reading": "わたくし", "scope": scope})
        self.assertEqual([r["scope"] for r in dialog.changes()["self_refs"]],
                         [{}, {"season": 2}, {"episodes": ["e1", "e2"]}, {"line": "m_0"}])
        scope = dialog.refs.cellWidget(2, 2)
        scope.value.setText("e3、e4")
        self.assertEqual(dialog.changes()["self_refs"][2]["scope"], {"episodes": ["e3", "e4"]})
        self.panel.change(lambda b: b.update_character(ch.id, **dialog.changes(),
                         ))
        self.session.editor.set_speaker(["m_0"], ch.id, whole_group=False)
        ann = next(a for a in self.doc.annotations.values() if a.member == "m_0" and a.base == "私")
        ann.status = "confirmed"
        kept = copy.deepcopy(ann)
        dialog.refs.item(3, 1).setText("わたし")
        self.panel.change(lambda b: b.update_character(ch.id, **dialog.changes()))
        self.assertEqual(self.doc.annotations[ann.id], kept)
        self.assertEqual(self.book.character(ch.id).self_refs[3]["reading"], "わたし")

    def import_dialog(self, guest=False):
        fixture = json.loads((ROOT / "tests/bangumi_fixture.json").read_text(encoding="utf-8"))
        if guest:
            fixture["characters"][0]["relation"] = "客串"
            # Recorded Chinese names contain no convertible characters; supplement this boundary.
            cid = str(fixture["characters"][0]["id"])
            fixture["details"][cid]["infobox"].append({"key": "简体中文名", "value": "测试"})
        data = fixture["subject"], fixture["characters"], fixture["details"]
        return ui.BangumiImportDialog(self.panel, data, self.book,
                                     {"original": "ja", "translation": "zh-Hant"}, self.panel.display)

    def test_bangumi_preview_defaults_matching_traditional_skip_and_cancel(self):
        before = self.book.path.read_bytes()
        dialog = self.import_dialog(guest=True)
        self.assertTrue(dialog.traditional.isChecked())
        self.assertEqual(len(dialog.rows), 14)
        self.assertNotIn(0, dialog.chosen())
        self.assertEqual(dialog.table.horizontalHeaderItem(6).text(), "对应")
        from opencc import OpenCC
        index = next(i for i, r in enumerate(dialog.rows) if OpenCC("t2s").convert(r["zh_name"]) != r["zh_name"])
        traditional = dialog.rows[index]["zh_name"]
        dialog.table.cellWidget(index, 6).setCurrentIndex(dialog.table.cellWidget(index, 6).findData("skip"))
        dialog.traditional.setChecked(False)
        self.assertNotEqual(dialog.rows[index]["zh_name"], traditional)
        self.assertNotIn(index, dialog.chosen())
        dialog.reject()
        self.assertEqual(self.book.path.read_bytes(), before)
        ch = self.book.add_character({"ja": NameForm(given=dialog.rows[1]["ja_name"])})
        matched = self.import_dialog()
        self.assertEqual(matched.table.cellWidget(1, 6).currentData(), ch.id)
        self.assertTrue(matched.table.cellWidget(1, 6).currentText().startswith("更新："))
        self.assertEqual(matched.table.cellWidget(1, 6).currentText(), "更新：" + ch.full_name("ja"))

    def test_bangumi_panel_apply_and_one_step_undo(self):
        before = self.book.path.read_bytes()
        count = len(self.book.characters)
        fixture = json.loads((ROOT / "tests/bangumi_fixture.json").read_text(encoding="utf-8"))
        data = fixture["subject"], fixture["characters"], fixture["details"]
        def choose(dialog):
            for i in range(len(dialog.rows)):
                dialog.table.item(i, 0).setCheckState(Qt.CheckState.Checked if i == 0 else Qt.CheckState.Unchecked)
            return QDialog.DialogCode.Accepted
        with patch.object(ui.QInputDialog, "getText", return_value=("2903", True)), \
                patch.object(ui.BangumiFetchDialog, "fetch", return_value=data), \
                patch.object(ui.BangumiImportDialog, "exec", choose):
            self.panel.import_bangumi()
        self.assertEqual(len(self.book.characters), count + 1)
        self.assertEqual(self.book.characters[-1].extra["source"]["bangumi"], data[1][0]["id"])
        self.session.editor.history.undo()
        self.assertEqual(self.book.path.read_bytes(), before)

    def test_bangumi_preview_name_split_edit_retained_on_toggle(self):
        dialog = self.import_dialog()
        dialog.table.selectRow(0)
        def edit(details):
            ja = details.names["ja"]
            ja.fields["family"].setText("変更")
            ja.fields["family_reading"].setText("へんこう")
            ja.fields["given"].setText("名前")
            ja.fields["given_reading"].setText("なまえ")
            return QDialog.DialogCode.Accepted
        with patch.object(ui.CharacterDetailsDialog, "exec", edit):
            dialog.edit_names()
        self.assertEqual(dialog.rows[0]["names"]["ja"].family, "変更")
        dialog.traditional.setChecked(False)
        self.assertEqual(dialog.rows[0]["names"]["ja"].family, "変更")
        self.assertEqual(dialog.table.item(0, 1).text(), "名前・変更")

    def test_details_every_separator_and_name_choice_preview(self):
        dialog = self.details_dialog()
        ja = dialog.names["ja"]
        ja.fields["family"].setText("幸平")
        ja.fields["given"].setText("創真")
        ja.order.setCurrentIndex(ja.order.findData("family_first"))
        for separator in ("", " ", "・", "·"):
            ja.separator.setCurrentIndex(ja.separator.findData(separator))
            self.assertEqual(ja.preview.text(), "幸平" + separator + "創真")
        for kind, expected in (("family", "幸平"), ("given", "創真"), ("full", "幸平·創真")):
            ja.short.setCurrentIndex(ja.short.findData(kind))
            self.assertEqual(ja.short_preview.text(), expected)

    def test_bangumi_fetch_thread_success_failure_cancel_cursor_and_status(self):
        self.window.statusBar().showMessage("原状态")
        fixture = json.loads((ROOT / "tests/bangumi_fixture.json").read_text(encoding="utf-8"))
        data = fixture["subject"], fixture["characters"], fixture["details"]
        with patch.object(bangumi, "fetch_subject", return_value=data):
            dialog = ui.BangumiFetchDialog(self.panel, "2903")
            self.assertEqual(dialog.fetch(), data)
            self.assertFalse(dialog.worker.isRunning())
        with patch.object(bangumi, "fetch_subject", side_effect=bangumi.BangumiError("HTTP 404")):
            dialog = ui.BangumiFetchDialog(self.panel, "0")
            with self.assertRaisesRegex(ValueError, "无法从 Bangumi 读取：HTTP 404"):
                dialog.fetch()
        def cancellable(text, *, cancelled, progress):
            progress(0, 14)
            for _ in range(100):
                if cancelled():
                    raise bangumi.BangumiCancelled()
                import time
                time.sleep(0.005)
            raise AssertionError("Cancel was not delivered")
        with patch.object(bangumi, "fetch_subject", side_effect=cancellable):
            dialog = ui.BangumiFetchDialog(self.panel, "2903")
            QTimer.singleShot(10, dialog.reject)
            self.assertIsNone(dialog.fetch())
            self.assertFalse(dialog.worker.isRunning())
        self.assertIsNone(QApplication.overrideCursor())
        self.assertEqual(self.window.statusBar().currentMessage(), "原状态")


if __name__ == "__main__":
    unittest.main()
