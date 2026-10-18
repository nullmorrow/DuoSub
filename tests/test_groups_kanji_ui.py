"""S1/S2 Qt wiring, selection, cancellation, input composition and context help.

Uses isolated settings and temporary synthetic documents; no private projects,
native player, preview workers or models are opened.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QEvent, QPoint, Qt, QTimer
from PySide6.QtGui import QInputMethodEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from subkit.core.readings import anchor_of, visible_map
from subkit.model.document import Annotation
from subkit.reading.engine import Analyzer
from subkit.reading.kanji_split import split_kanji_reading
from subkit_app import help as app_help
from subkit_app.views.group_times import GroupTimesDialog
from subkit_app.views.reading_popup import ReadingPopup
from tests import test_convert_ui as fixture_ui


class GroupsKanjiUI(unittest.TestCase):
    setUp = fixture_ui.ConversionUI.setUp
    _dispose = fixture_ui.ConversionUI._dispose
    group = fixture_ui.ConversionUI.group
    select = fixture_ui.ConversionUI.select
    snapshot = fixture_ui.ConversionUI.snapshot
    context_menu = fixture_ui.ConversionUI.context_menu

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def japanese(self, text="学校", reading="がっこう", *, kind="dialogue"):
        m = self.group("g", kind, text=text)
        m.lang, m.lane, m.style = "ja", "original", "JP"
        ids = visible_map(m)[1]
        a = Annotation("a", "ruby", m.id, anchor_of(ids), text, reading,
                       status="auto", origin="analyzer", parts=[[0,len(text),reading]])
        self.doc.annotations[a.id] = a
        self.project.dict_path = lambda key: self.path / (key + ".json")
        self.select("g")
        return m

    def current_popup(self):
        return next(w for w in self.window.findChildren(ReadingPopup) if w.isVisible())

    def show_reading(self):
        self.window._on_reading("a", QPoint(100,100))
        self.app.processEvents()
        return self.current_popup()

    def test_single_and_multiple_context_copy_selection_and_one_undo(self):
        self.group("g1")
        self.group("g2", "sign")
        self.select("g1","g2")
        before = self.snapshot()
        menu = self.context_menu()
        times = next(a for a in menu if a.text().startswith("设置开始和结束时间"))
        self.assertFalse(times.isEnabled())
        self.assertIn("（只能对一组设置）", times.text())
        self.assertIn("复制这 2 组\tCtrl+D", [a.text() for a in menu])
        self.context_menu("复制这 2 组")
        new = self.window.table.selected_gids()
        self.assertEqual(len(new),2)
        self.assertEqual([g.id for g in self.doc.groups],["g1",new[0],"g2",new[1]])
        self.assertEqual(self.window.current_gid,new[0])
        self.assertEqual(self.window.statusBar().currentMessage(),"已复制 2 组（Ctrl+Z 撤销）")
        self.assertEqual(len(self.session.editor.history.done),1)
        self.session.editor.history.undo()
        self.assertEqual(self.snapshot(),before)

    def test_copy_keeps_auto_readings_in_session_and_filters_reveal_copy(self):
        m=self.japanese()
        self.session.editor.mark_many(["g"])
        self.window.filter_box.setCurrentText("只看已标记")
        self.select("g")
        old=copy.deepcopy(self.doc.annotations["a"])
        with patch.object(self.session,"_reannotate") as refresh:
            self.context_menu("复制这一组")
        refresh.assert_not_called()
        new=self.doc.members[self.doc.group(self.window.current_gid).members[0]]
        self.assertEqual(self.window.filter_box.currentText(),"全部")
        self.assertNotEqual(new.id,m.id)
        a=self.doc.annotations_of(new.id)[0]
        self.assertEqual((a.text,a.status,a.origin),(old.text,old.status,old.origin))
        self.assertTrue(self.session.dirty)

    def test_time_dialog_validation_formats_targets_and_beats_visibility(self):
        m=self.japanese()
        m.start,m.end=83456,85000
        dialog=GroupTimesDialog(self.doc,"g",self.window)
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog.start.text(),"0:01:23.456")
        self.assertFalse(dialog.shift_beats.isVisible())
        self.assertFalse(dialog.target.model().item(1).isEnabled())
        dialog.start.setText("1:23.45")
        dialog.end.setText("83450")
        self.assertFalse(dialog.ok.isEnabled())
        self.assertEqual(dialog.hint.text(),"结束要晚于开始")
        dialog.end.setText("bad")
        self.assertEqual(dialog.hint.text(),"时间格式：0:01:23.45")
        dialog.end.setText("85001")
        self.assertTrue(dialog.ok.isEnabled())
        self.assertEqual(dialog.values()[:2],(83450,85001))

    def test_time_menu_modal_apply_cancel_and_help(self):
        self.japanese()
        before=self.snapshot()
        def cancel():
            dialog=QApplication.activeModalWidget()
            self.assertIsInstance(dialog,GroupTimesDialog)
            dialog.start.setText("2000")
            dialog.reject()
        QTimer.singleShot(0,cancel)
        self.context_menu("设置开始和结束时间")
        self.assertEqual(self.snapshot(),before)
        def apply():
            dialog=QApplication.activeModalWidget()
            with patch.object(app_help,"open_manual") as opener:
                QTest.keyClick(dialog.start,Qt.Key.Key_F1)
                opener.assert_called_once_with("timing")
            dialog.start.setText("1:23.456")
            dialog.end.setText("85001")
            dialog.target.setCurrentIndex(2)
            dialog.ok.click()
        QTimer.singleShot(0,apply)
        self.context_menu("设置开始和结束时间")
        m=self.doc.members["m_g"]
        self.assertEqual((m.start,m.end),(83456,85001))
        self.assertEqual(self.session.editor.history.done[-1].label,"设置时间")
        self.session.editor.history.undo()
        self.assertEqual(self.snapshot(),before)

    def test_popup_prefill_romaji_cancel_returns_and_reading_help(self):
        self.japanese()
        before=self.snapshot()
        pop=self.show_reading()
        self.assertTrue(pop.btn_kanji.isVisible())
        pop.btn_kanji.click()
        self.assertTrue(pop._kanji_mode)
        self.assertEqual([edit.text() for _,edit,_ in pop._kanji_rows],["がっ","こう"])
        first=pop._kanji_rows[0][1]
        first.setText("gatsu")
        self.assertEqual(pop._kanji_rows[0][2].text(),"がつ")
        with patch.object(app_help,"open_manual") as opener:
            QTest.keyClick(first,Qt.Key.Key_F1)
            opener.assert_called_once_with("readings")
        QTest.keyClick(first,Qt.Key.Key_Escape)
        self.assertFalse(pop._kanji_mode)
        self.assertTrue(pop.isVisible())
        self.assertTrue(pop.edit.isVisible())
        self.assertEqual(self.snapshot(),before)
        pop.close()

    def test_range_changed_hides_button_and_restore_brings_it_back(self):
        scope={"text":"学校生活","a":0,"b":2,"spans":[
            {"a":0,"b":2,"reading":"がっこう"},{"a":2,"b":4,"reading":"せいかつ"}]}
        pop=ReadingPopup("学校","がっこう",[],"manual","confirmed",self.window,scope=scope)
        self.addCleanup(pop.deleteLater)
        self.addCleanup(pop.close)
        pop.show()
        self.assertTrue(pop.btn_kanji.isVisible())
        pop._extend(1)
        self.assertFalse(pop.btn_kanji.isVisible())
        pop._reset_range()
        self.assertTrue(pop.btn_kanji.isVisible())

    def test_irregular_prefill_empty_manual_entry_and_scope_only_here(self):
        for scope in (1,2):
            with self.subTest(scope=scope):
                self.japanese("今日","きょう") if not self.doc.groups else None
                before=self.snapshot()
                pop=self.show_reading()
                pop.scope_group.button(scope).setChecked(True)
                pop.btn_kanji.click()
                self.assertIn("自动分不开（可能是熟字訓）",pop.kanji_hint.text())
                self.assertEqual([edit.text() for _,edit,_ in pop._kanji_rows],["",""])
                pop._kanji_rows[0][1].setText("kyou")
                pop.kanji_ok.click()
                anns=self.doc.annotations_of("m_g")
                self.assertEqual([(a.base,a.text,a.status) for a in anns],[("今","きょう","confirmed"),("日","","hidden")])
                self.assertIn("只改了这一处",self.window.statusBar().currentMessage())
                self.assertEqual(self.doc.rules,[])
                self.assertFalse(self.project.dict_path("readings").exists())
                self.session.editor.history.undo()
                self.assertEqual(self.snapshot(),before)

    def test_ime_preedit_does_not_apply_and_empty_second_character_is_preserved(self):
        self.japanese()
        pop=self.show_reading()
        pop.btn_kanji.click()
        got=[]
        pop.kanjiChosen.connect(lambda readings,scope:got.append((readings,scope)))
        first=pop._kanji_rows[0][1]
        event=QInputMethodEvent("g",[])
        QApplication.sendEvent(first,event)
        pop._accept_kanji()
        self.assertEqual(got,[])
        commit=QInputMethodEvent()
        commit.setCommitString("がっ")
        QApplication.sendEvent(first,commit)
        first.setText("gatt")
        pop._kanji_rows[1][1].clear()
        pop.kanji_ok.click()
        self.assertEqual(len(got),1)
        self.assertEqual(got[0][0][1],"")

    def test_single_kanji_and_pinyin_never_show_per_kanji_button(self):
        for base,kind in (("学","ruby"),("学校","pinyin")):
            pop=ReadingPopup(base,"",[],"manual","",self.window,kind=kind,
                             scope={"text":base,"a":0,"b":len(base),"spans":[]})
            self.addCleanup(pop.deleteLater)
            pop.show()
            self.assertFalse(pop.btn_kanji.isVisible())
            pop.close()

    def test_delayed_popup_does_not_write_another_session(self):
        self.japanese()
        before=self.snapshot()
        pop=self.show_reading()
        pop.btn_kanji.click()
        self.window.session=None
        pop.kanji_ok.click()
        self.assertEqual(self.snapshot(),before)

    def test_new_project_import_merge_edit_save_reopen_and_release(self):
        """Start from real empty project APIs; attach it through MainWindow.open_project."""
        from hashlib import sha256
        from subkit.core.workspace import Project
        from subkit.model.serialize import document_to_json
        from subkit_app.session import EpisodeSession
        cn=self.path/"translation.srt"
        ja=self.path/"original.srt"
        cn.write_text("1\n00:00:01,000 --> 00:00:04,000\n去学校\n",encoding="utf-8")
        ja.write_text("1\n00:00:01,000 --> 00:00:04,000\n学校へ行く\n",encoding="utf-8")
        originals={p:sha256(p.read_bytes()).hexdigest() for p in (cn,ja)}
        project=Project.create(self.path/"real-project","S1 S2 acceptance",translation="zh-Hans")
        episode=project.add_episode("S01E001")
        episode.add_source(cn,lang="zh-Hans",lane="translation")
        episode.add_source(ja,lang="ja",lane="original")
        episode.merge()
        self.assertTrue(episode.validate().ok)
        initial=(episode.dir/"document.json").read_bytes()
        with patch.object(EpisodeSession,"request_preview"):
            self.window.open_project(project.root)
            self.session=self.window.session
            self.doc=self.session.doc
            self.project=self.window.project
            self.assertEqual(len(self.doc.groups),1)
            gid=self.doc.groups[0].id
            self.window._duplicate_groups([gid])
            copied=self.window.current_gid
            self.assertNotEqual(copied,gid)
            self.session.editor.set_exact_group_time(copied,83456,85555)
            m=next(m for m in self.doc.members_of(self.doc.group(copied)) if m.lane=="original")
            ann=next(a for a in self.doc.annotations_of(m.id) if a.base=="学校")
            self.window._on_reading(ann.id,QPoint(100,100))
            pop=self.current_popup()
            pop.btn_kanji.click()
            self.assertEqual([e.text() for _,e,_ in pop._kanji_rows],["がっ","こう"])
            pop.kanji_ok.click()
            self.assertEqual([(a.base,a.text) for a in self.doc.annotations_of(m.id) if a.base in ("学","校")],
                             [("学","がっ"),("校","こう")])
            self.session.save()
            directory=self.session.episode.dir
            self.assertEqual((directory/"document.v1.bak.json").read_bytes(),initial)
            saved=self.snapshot()
            self.window.close_session()
            self.window.open_project(project.root)
            self.session=self.window.session
            self.doc=self.session.doc
            self.assertEqual(self.snapshot(),saved)
            self.assertEqual(document_to_json(self.doc)["format_version"],2)
            self.assertTrue(self.session.episode.validate().ok)
            output,report=self.session.episode.export(self.path/"release.ass")
            self.assertTrue(output.is_file())
            self.assertGreater(report.ruby_lines,0)
            self.assertEqual(self.doc.group_time(self.doc.group(copied)),(83456,85555))
            for source in self.session.episode.sources.values():
                copy_path=self.session.episode.dir/source.copy
                self.assertEqual(sha256(copy_path.read_bytes()).hexdigest(),source.sha256)
        for p,digest in originals.items(): self.assertEqual(sha256(p.read_bytes()).hexdigest(),digest)


if __name__ == "__main__":
    unittest.main()
