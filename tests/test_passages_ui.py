"""Offscreen passage dialogs with copied projects and a fake media player."""

from __future__ import annotations

import copy
import os
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

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QItemSelectionModel, QSettings, Qt, QTimer
from PySide6.QtGui import QContextMenuEvent, QInputMethodEvent
from PySide6.QtTest import QSignalSpy
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLabel, QMenu, QMessageBox

from subkit.audio import AudioFeatures
from subkit.audio_match import Match, SearchResult, template_from
from subkit.core import karaoke as kara
from subkit.core.passages import ApplySummary, PassageStore, snapshot
from subkit.core.workspace import Project
from subkit.model.document import Document, Group
from subkit.model.serialize import document_to_json
from subkit_app import mainwindow as mw
from subkit_app.session import EpisodeSession
from subkit_app.views import passages as ui
from subkit_app.views.group_table import GroupFilter, GroupTableModel
from subkit_app.views.waveform import WaveformView
from tests.test_convert_ui import PlayerDouble
from tests.test_passages import annotation, member


class PassageUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.platformName() != "offscreen":
            raise RuntimeError("Passage UI tests require offscreen Qt")

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="subkit_passages_ui_")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        # Copy bytes into ordinary writable scratch files, without source read-only attributes.
        shutil.copytree(ROOT / "test_projects" / "soma", self.path / "project",
                        copy_function=shutil.copyfile, ignore=shutil.ignore_patterns("cache"))
        self.project = Project.open(self.path / "project")
        self.settings = QSettings(str(self.path / "settings.ini"), QSettings.Format.IniFormat)
        self.doc = Document("passages-ui", groups=[Group("g_1", "song"), Group("g_2", "dialogue")])
        self.original = member(self.doc, "g_1", "m_jp", "学" * 24, start=1013, end=6000)
        member(self.doc, "g_1", "m_cn", "译文", start=2000, end=6500, lane="translation")
        member(self.doc, "g_2", "m_other", "未选", start=8000, end=10000)
        annotation(self.doc, self.original, 0, 1, "がく")
        annotation(self.doc, self.original, 1, 2, "がく", status="auto")
        self.features = AudioFeatures(np.zeros(1200, np.float32), np.ones(1200, np.float32),
                                      np.arange(1200 * 128, dtype=np.uint8).reshape(1200, 128),
                                      np.arange(1200, dtype=np.float32))
        self.episode = SimpleNamespace(id="E01", dir=self.path, document=self.doc, sources={}, video=None,
                                       save=Mock(), settings=lambda: {"merge": {}, "release": {}},
                                       reading_engine=lambda: self.project.reading_engine(episode="E01"))
        for target, value in (("MpvWidget", PlayerDouble), ("QSettings", Mock(return_value=self.settings)),
                              ("MainWindow._open_last", Mock())):
            p = patch("subkit_app.mainwindow." + target, value)
            p.start()
            self.addCleanup(p.stop)
        self.window = mw.MainWindow()
        self.session = EpisodeSession(self.project, self.episode)
        self.session.request_preview = Mock()
        self.session.engine = SimpleNamespace(analyze=Mock(return_value=[]))
        self.window.project, self.window.session = self.project, self.session
        self.window.model = GroupTableModel(self.doc)
        self.window.proxy = GroupFilter()
        self.window.proxy.setSourceModel(self.window.model)
        self.window.table.setModel(self.window.proxy)
        self.window.wave.set_document(self.doc)
        self.window.wave._loaded(self.features, self.window.wave._load_generation)
        self.store = PassageStore(self.project)
        self.addCleanup(self._dispose)
        self.select("g_1")

    def _dispose(self) -> None:
        self.session._autosave.stop()
        self.session._debounce.stop()
        self.window.close()
        self.window.deleteLater()
        self.session.deleteLater()
        self.app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.settings.sync()

    def select(self, *gids: str) -> None:
        self.window.select_group(gids[0])
        sm = self.window.table.selectionModel()
        sm.clearSelection()
        for gid in gids:
            index = self.window.proxy.mapFromSource(self.window.model.index(self.doc.group_index(gid), 0))
            sm.select(index, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)

    def record(self, name="学" * 20, version="TV", choice=None) -> None:
        def accept(dialog):
            dialog.name.setText(name)
            dialog.version.setText(version)
            dialog.category.setCurrentIndex(dialog.category.findData("ED"))
            return QDialog.DialogCode.Accepted

        with patch.object(ui.RecordPassageDialog, "exec", accept), patch.object(mw, "duplicate_choice", return_value=choice):
            self.window._record_passage(self.window.table.selected_gids())

    def test_record_template_origin_and_no_document_or_project_metadata_edits(self) -> None:
        before = copy.deepcopy(document_to_json(self.doc))
        metadata = (self.project.root / "subkit-project.json").read_bytes()
        with patch.object(mw, "template_from", wraps=mw.template_from) as template:
            self.record()
        template.assert_called_once_with(self.features, 1013, 6500)
        p = self.store.list()[0]
        self.assertEqual((p["name"], p["category"], p["version"], p["start"]), ("学" * 20, "ED", "TV", 1013))
        self.assertEqual((p["summary"]["groups"], p["summary"]["lines"]), (1, 2))
        self.assertEqual(self.window.statusBar().currentMessage(), "已记为重复段落：片尾曲《" + "学" * 20 + "》")
        np.testing.assert_array_equal(self.store.template(p["id"]).onset[:3], self.features.onset[102:105])
        self.assertEqual(document_to_json(self.doc), before)
        self.assertEqual((self.project.root / "subkit-project.json").read_bytes(), metadata)

    def test_dialog_default_name_summary_modal_and_input_method_event(self) -> None:
        def inspect(dialog):
            self.assertEqual(dialog.name.text(), "学" * 20)
            self.assertTrue(dialog.isModal())
            self.assertEqual(dialog.windowType(), Qt.WindowType.Tool)
            self.assertEqual(dialog.version.placeholderText(), "例如 TV 版、完整版")
            self.assertIn("共 1 组 2 行，5.487 秒", dialog.range_label.text())
            self.assertIn("读音 1 处 · K 轴", dialog.lines.item(0).text())
            labels = [label.text() for label in dialog.findChildren(QLabel)]
            self.assertIn("有 1 行在没确认的组里，它们的自动读音不记下（别的集会按同样的文字自动注出来）；"
                          "手动改过、选过的读音照样记下。把这些组标为已确认后，组里的读音也会记下。", labels)
            dialog.name.clear()
            self.assertFalse(dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled())
            event = QInputMethodEvent()
            event.setCommitString("中文名称")
            QCoreApplication.sendEvent(dialog.name, event)
            self.assertEqual(dialog.name.text(), "中文名称")
            self.assertTrue(dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled())
            return QDialog.DialogCode.Rejected

        with patch.object(ui.RecordPassageDialog, "exec", inspect):
            self.window._record_passage(["g_1"])
        self.assertEqual(self.store.list(), [])

    def test_duplicate_replace_save_new_cancel_and_distinct_version(self) -> None:
        self.record(name="段落")
        first = self.store.list()[0]["id"]
        self.original.end = 6200
        self.record(name="段落", choice="replace")
        self.assertEqual([p["id"] for p in self.store.list()], [first])
        self.assertEqual(self.store.list()[0]["groups"][0]["members"][0]["end"], 6200 - 1013)
        before = self.store.path.read_bytes()
        self.record(name="段落", choice="cancel")
        self.assertEqual(self.store.path.read_bytes(), before)
        self.record(name="段落", choice="new")
        self.assertEqual(len(self.store.list()), 2)
        self.assertEqual(len({p["id"] for p in self.store.list()}), 2)
        self.record(name="段落", version="完整版")
        self.assertEqual(len(self.store.list()), 3)

    def test_duplicate_prompt_verbatim_and_each_button(self) -> None:
        p = {"name": "段落", "version": "TV"}
        for label, expected in (("替换", "replace"), ("另存为新的", "new"), ("取消", "cancel")):
            def choose(box):
                self.assertEqual(box.text(), "库里已经有《段落》（TV）。")
                self.assertEqual({b.text() for b in box.buttons()}, {"替换", "另存为新的", "取消"})
                next(b for b in box.buttons() if b.text() == label).click()
                return 0
            with patch.object(QMessageBox, "exec", choose):
                self.assertEqual(ui.duplicate_choice(self.window, p), expected)

    def test_refusal_missing_selection_missing_features_and_too_short(self) -> None:
        with patch.object(QMessageBox, "information") as info:
            self.window._record_passage([])
            self.assertEqual(info.call_args.args[2], "先在列表里选中这一段的几组。")
            self.window.wave.load(None, None, self.path)
            self.window._record_passage(["g_1"])
            self.assertEqual(info.call_args.args[2], "音频还在读取，读完后再试。")
        self.window.wave._loaded(self.features)
        for m in self.doc.members_of(self.doc.groups[0]):
            m.end = m.start + 100
        with patch.object(QMessageBox, "warning") as warning:
            self.window._record_passage(["g_1"])
            self.assertEqual(warning.call_args.args[2], "段落太短（不到 3 秒），没法按音频找。")
        self.assertEqual(self.store.list(), [])

    def test_library_rename_version_and_delete_confirmation(self) -> None:
        self.record(name="段落")
        before = copy.deepcopy(document_to_json(self.doc))
        dialog = ui.PassageLibraryDialog(self.store, self.window)
        self.addCleanup(dialog.deleteLater)
        self.assertTrue(dialog.isModal())
        self.assertEqual(dialog.windowType(), Qt.WindowType.Tool)
        self.assertEqual([dialog.table.horizontalHeaderItem(i).text() for i in range(7)],
                         ["名称", "类别", "版本", "来自", "行数", "时长", "记录时间"])
        self.assertFalse(dialog.rename_button.isEnabled())
        dialog.table.selectRow(0)
        self.assertTrue(dialog.rename_button.isEnabled())
        def rename(d):
            self.assertEqual(d.windowType(), Qt.WindowType.Tool)
            d.name.setText("新名称")
            d.version.setText("完整版")
            return QDialog.DialogCode.Accepted
        with patch.object(ui._RenameDialog, "exec", rename):
            dialog.rename_button.click()
        p = self.store.list()[0]
        self.assertEqual((p["name"], p["version"]), ("新名称", "完整版"))
        path = self.store.templates / (p["id"] + ".npz")
        for label in ("取消", "删除"):
            def choose(box):
                self.assertEqual(box.text(), "从库里删除《新名称》？已经套用到各集的内容不受影响。")
                next(b for b in box.buttons() if b.text() == label).click()
                return 0
            with patch.object(QMessageBox, "exec", choose):
                dialog.delete_button.click()
            self.assertEqual(path.exists(), label == "取消")
        self.assertEqual(self.store.list(), [])
        self.assertFalse(dialog.delete_button.isEnabled())
        self.assertEqual(document_to_json(self.doc), before)

    def test_menu_entry_points_and_context_selection(self) -> None:
        self.assertTrue(self.window.act_find_passage.isEnabled())
        with patch.object(self.window, "_record_passage") as record:
            next(a for a in self.window.passages_menu.actions() if a.text() == "记为重复段落…").trigger()
            record.assert_called_once_with(["g_1"])
        def library_exec(d):
            self.assertEqual(d.store.path, self.store.path)
            return QDialog.DialogCode.Rejected
        menu_actions = self.window.menuBar().actions()
        tools = next(a.menu() for a in menu_actions if a.text() == "工具(&O)")
        menu = next(a.menu() for a in tools.actions() if a.text() == "重复段落")
        with patch.object(ui.PassageLibraryDialog, "exec", library_exec):
            next(a for a in menu.actions() if a.text() == "重复段落库…").trigger()
        got = []
        self.window.table.recordPassageRequested.disconnect(self.window._record_passage)
        self.window.table.recordPassageRequested.connect(got.append)
        self.select("g_1", "g_2")
        class MenuDouble(QMenu):
            def exec(self, pos):
                return next(a for a in self.actions() if a.text() == "记为重复段落…")
        table = self.window.table
        table.resize(1000, 400)
        pos = table.visualRect(table.currentIndex()).center()
        event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, pos, table.mapToGlobal(pos))
        with patch("subkit_app.views.group_table.QMenu", MenuDouble):
            table.contextMenuEvent(event)
        self.assertEqual(got, [["g_1", "g_2"]])

    def test_features_signal_generation_accessor_and_reset(self) -> None:
        wave = WaveformView()
        self.addCleanup(wave.deleteLater)
        spy = QSignalSpy(wave.featuresLoaded)
        observed = []
        wave.featuresLoaded.connect(lambda: observed.append(wave.audio_features()))
        wave.load(None, None, self.path)
        generation = wave._load_generation
        wave._loaded(self.features, generation - 1)
        self.assertEqual(spy.count(), 0)
        self.assertIsNone(wave.audio_features())
        wave._loaded(self.features, generation)
        self.assertEqual(spy.count(), 1)
        self.assertIs(observed[0], self.features)
        wave.load(None, None, self.path)
        self.assertIsNone(wave.audio_features())
        wave._loaded(self.features, generation)
        self.assertEqual(spy.count(), 1)
        self.assertIsNone(wave.spectrogram)

    def test_record_real_modal_event_loop_cancel(self) -> None:
        passage = snapshot(self.doc, ["g_1"], name="名称", category="OP", version="", episode_id="E01")
        d = ui.RecordPassageDialog(passage, self.window, karaoke_states=["还有占位拍", "无"])
        self.addCleanup(d.deleteLater)
        QTimer.singleShot(0, d.reject)
        self.assertEqual(d.exec(), QDialog.DialogCode.Rejected)

    def test_record_karaoke_summary_complete_partial_and_none(self) -> None:
        beats = kara.derive(self.doc, self.original)
        for i, beat in enumerate(beats):
            beat.t0, beat.t1, beat.by = 1013 + i * 30, 1043 + i * 30, "manual"
        track = kara.beats_to_track(self.original, beats)
        self.doc.karaoke[track["id"]] = track
        for partial in (False, True):
            if partial:
                track["beats"] = track["beats"][:1]
            def inspect(d):
                self.assertTrue(d.lines.item(0).text().endswith("K 轴 " + ("还有占位拍" if partial else "有")))
                self.assertTrue(d.lines.item(1).text().endswith("K 轴 无"))
                labels = [label.text() for label in d.findChildren(QLabel)]
                self.assertEqual("有 1 行 K 轴还没打完，占位拍不会记下。" in labels, partial)
                return QDialog.DialogCode.Rejected
            with patch.object(ui.RecordPassageDialog, "exec", inspect):
                self.window._record_passage(["g_1"])
        self.assertEqual(self.store.list(), [])

    def find_dialog(self, *, missing=False, karaoke=False):
        if karaoke:
            beats = kara.derive(self.doc, self.original)
            for i, beat in enumerate(beats):
                beat.t0, beat.t1, beat.by = 1013 + i * 30, 1043 + i * 30, "manual"
            track = kara.beats_to_track(self.original, beats)
            self.doc.karaoke[track['id']] = track
        self.original.overrides = {"pos": [100, 200]}
        p = snapshot(self.doc, ["g_1"], name="段落", category="OP", version="TV", episode_id="E01")
        self.store.add(p, template_from(self.features, p['start'], p['start'] + p['duration']))
        if missing:
            self.doc.groups[0].members.clear()
            self.doc.members = {'m_other': self.doc.members['m_other']}
            self.doc.annotations.clear()
            self.doc.karaoke.clear()
            self.doc.groups = [self.doc.groups[1]]
            self.doc._gindex_valid = False
        else:
            self.doc.groups[0].kind = "dialogue"
            for m in self.doc.members_of(self.doc.groups[0]):
                m.role, m.style = "dialogue", "Default"
            self.original.overrides = {"pos": [300, 400]}
            self.doc.annotations['a_0'].text = "まなぶ"
        match = Match(1513, .95, (0, p['duration']), [], 55, 1.01)
        result = SearchResult([match], .95)
        d = ui.PassagesDialog(self.session, [(p, result)], self.window.player, self.window)
        self.addCleanup(d.deleteLater)
        return d, p, result

    def test_find_defaults_conflicts_notes_scale_and_group_conversion_once(self):
        d, p, result = self.find_dialog(karaoke=True)
        before = copy.deepcopy(document_to_json(self.doc))
        self.assertEqual(d.windowTitle(), "重复段落 — E01")
        self.assertTrue(d.isModal())
        self.assertEqual(d.windowType(), Qt.WindowType.Tool)
        self.assertFalse(d.scale.isChecked())
        self.assertIn("结尾比库里晚 55 毫秒（TV可能不同）", d.drift.text())
        actions = {a.id: a for l in d.current_plan.lines for a in l.actions}
        for aid, item in d.action_items.items():
            self.assertEqual(item.checkState(0) == Qt.CheckState.Checked, actions[aid].checked)
            if actions[aid].conflict:
                self.assertEqual(item.child(0).foreground(0).color(), ui.QColor('red'))
                self.assertFalse(item.child(0).flags() & Qt.ItemFlag.ItemIsUserCheckable)
        shown = [actions[aid].kind for aid in d.action_items]
        self.assertEqual(shown.count('convert'), 1)
        self.assertIn('karaoke', shown)
        kitem = next(item for aid, item in d.action_items.items() if actions[aid].kind == 'karaoke')
        self.assertIn('读音和库里不同，要同时补读音，K 轴才对得上', [kitem.child(i).text(0) for i in range(kitem.childCount())])
        d.check_all(False)
        aid = next(iter(d.action_items))
        d.action_items[aid].setCheckState(0, Qt.CheckState.Checked)
        states = {a: i.checkState(0) for a, i in d.action_items.items()}
        old_end = d.current_plan.lines[0].end
        d.scale.setChecked(True)
        self.assertEqual({a: i.checkState(0) for a, i in d.action_items.items()}, states)
        self.assertGreater(d.current_plan.lines[0].end, old_end)
        self.assertEqual(document_to_json(self.doc), before)
        with patch.object(self.window.player, 'play') as play:
            d.audition_button.click()
            self.window.player.seek.assert_called_once_with(1.513)
            play.assert_called_once()

    def test_find_insert_group_once_and_apply_ctrl_z_restores_document(self):
        d, p, result = self.find_dialog(missing=True)
        before = copy.deepcopy(document_to_json(self.doc))
        self.assertEqual(len(d.action_items), 1)
        self.assertEqual(next(iter(d.action_items.values())).text(0), '补上这一组')
        d.show()
        self.app.processEvents()
        d.apply_button.click()
        self.assertIn('补上 2 行（Ctrl+Z 撤销）', self.window.statusBar().currentMessage())
        self.assertIn('已套用', d.list.item(0).text())
        QTest.keyClick(d, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        after = document_to_json(self.doc)
        after['revision'] = before['revision']
        self.assertEqual(after, before)
        self.assertNotIn('已套用', d.list.item(0).text())
        d.close()

    def test_coverage_labels_defaults_toggle_scale_apply_and_undo(self):
        from tests.test_passages import CoverageTests
        for gap, state, checked, label in ((20000, "unknown", False, "未知"),
                                            (5000, "inferred", True, "按连续性推断 · 证据弱")):
            p = CoverageTests().passage([(0, 10000), (10000, 10000 + gap),
                                        (10000 + gap, 20000 + gap)])
            m = Match(50000, .95, (0, 20000 + gap), [], 50, 1.01,
                      ((0, 10000), (10000 + gap, 20000 + gap)))
            d = ui.PassagesDialog(self.session, [(p, SearchResult([m], .95))],
                                  self.window.player, self.window)
            self.addCleanup(d.deleteLater)
            line = d.current_plan.lines[1]
            self.assertEqual(line.coverage, state)
            item = d.action_items[line.actions[0].id]
            self.assertEqual(item.checkState(0) == Qt.CheckState.Checked, checked)
            self.assertIn(label, d.tree.topLevelItem(1).text(0))
            self.assertIn("直接匹配到", d.tree.topLevelItem(0).text(0))
            item.setCheckState(0, Qt.CheckState.Checked)
            d.scale.setChecked(True)
            self.assertEqual(d.action_items[line.actions[0].id].checkState(0), Qt.CheckState.Checked)
            before = document_to_json(self.doc)
            d.apply()
            self.assertIn("补上 3 行", self.window.statusBar().currentMessage())
            d.undo()
            after = document_to_json(self.doc); after["revision"] = before["revision"]
            self.assertEqual(after, before)

    def test_find_apply_preserves_conflicts_and_passes_episode_styles(self):
        d, p, result = self.find_dialog(karaoke=True)
        before = copy.deepcopy(document_to_json(self.doc))
        readings = copy.deepcopy(self.doc.annotations)
        overrides = copy.deepcopy(self.original.overrides)
        self.episode.settings = lambda: {'merge': {'styles': {'original/song': '{fam}.JP', 'translation/song': '{fam}.CN'}}}
        d.styles = self.episode.settings()['merge']['styles']
        with patch.object(self.session.editor, 'apply_passage', wraps=self.session.editor.apply_passage) as apply:
            d.apply()
            self.assertEqual(apply.call_args.kwargs['styles'], d.styles)
        self.assertEqual(self.original.start, 1513)
        self.assertEqual(self.doc.groups[0].kind, 'song')
        self.assertEqual(self.doc.annotations, readings)
        self.assertEqual(self.original.overrides, overrides)
        self.assertEqual(len(self.session.editor.history.done), 1)
        self.session.engine.analyze.assert_not_called()
        d.undo()
        after = document_to_json(self.doc)
        after['revision'] = before['revision']
        self.assertEqual(after, before)

    def test_find_skipped_messages_and_value_error_rollback(self):
        d, p, result = self.find_dialog(karaoke=True)
        aid = next(iter(d.action_items))
        reasons = list(ui.SKIP_REASONS)
        summary = ApplySummary(skipped=[{'id': aid, 'reason': reason} for reason in reasons])
        with patch.object(self.session.editor, 'apply_passage', return_value=summary), patch.object(QMessageBox, 'information') as info:
            d.apply()
            self.assertEqual(info.call_args.args[1], '重复段落')
            self.assertTrue(info.call_args.args[2].startswith(f'有 {len(reasons)} 项没有套用：\n'))
            for text in ui.SKIP_REASONS.values():
                self.assertIn(text, info.call_args.args[2])
        before = copy.deepcopy(document_to_json(self.doc))
        with patch('subkit.model.validate.validate', side_effect=ValueError('校验失败')), patch.object(QMessageBox, 'warning') as warning:
            d.apply()
            self.assertEqual(warning.call_args.args[2], '没有套用：校验失败')
        self.assertEqual(document_to_json(self.doc), before)

    def test_find_not_found_multiple_matches_and_partial_information(self):
        d, p, result = self.find_dialog(missing=True)
        result.matches.append(Match(9000, .94, (1000, 3000), [], -20, .99))
        d.results.append(({**p, 'name': '未找到'}, SearchResult([], .4)))
        d.refresh()
        self.assertEqual(d.list.count(), 3)
        self.assertIn('放了 1–3 秒', d.list.item(1).text())
        self.assertIn('没找到（最高相关 0.40）', d.list.item(2).text())
        self.assertEqual(d.list.item(2).foreground().color(), ui.QColor('gray'))
        d.list.setCurrentRow(1)
        self.assertEqual(len(d.action_items), 2)
        self.assertTrue(all(i.checkState(0) == Qt.CheckState.Unchecked for i in d.action_items.values()))
        info = d.tree.topLevelItem(d.tree.topLevelItemCount() - 1)
        self.assertEqual(info.text(0), '覆盖未知：2 行默认不勾选，可手动勾选。')
        self.assertFalse(info.flags() & Qt.ItemFlag.ItemIsUserCheckable)
        d.list.setCurrentRow(2)
        self.assertFalse(d.apply_button.isEnabled())

    def test_search_cursor_exception_cancel_success_and_index_reuse(self):
        d, p, result = self.find_dialog()
        seen = []
        def search(template):
            self.assertEqual(QApplication.overrideCursor().shape(), Qt.CursorShape.WaitCursor)
            self.assertEqual(self.window.statusBar().currentMessage(), '正在查找重复段落…')
            seen.append(template)
            return result
        index_class = mw.EpisodeIndex
        with patch.object(mw, 'EpisodeIndex', wraps=index_class) as index, patch.object(index_class, 'find', side_effect=search):
            self.window._search_passages(self.store, [p])
            with patch.object(ui.PassagesDialog, 'exec', return_value=QDialog.DialogCode.Rejected):
                self.window.act_find_passage.trigger()
            self.assertEqual(index.call_count, 1)
        self.assertEqual(len(seen), 2)
        self.assertIsNone(QApplication.overrideCursor())
        self.window.statusBar().showMessage('原状态')
        with patch.object(self.window._passage_index, 'find', side_effect=ValueError('搜索失败')), patch.object(QMessageBox, 'warning') as warning:
            self.window._find_passages()
            self.assertEqual(warning.call_args.args[2], '搜索失败')
        self.assertIsNone(QApplication.overrideCursor())
        self.assertEqual(self.window.statusBar().currentMessage(), '原状态')

    def test_automatic_prompt_only_new_matches_and_quick_episode_switch(self):
        d, p, result = self.find_dialog()
        self.window._load_media(self.episode)
        with patch.object(mw.EpisodeIndex, 'find', return_value=result), patch.object(QMessageBox, 'information') as info:
            self.window.wave._loaded(self.features, self.window.wave._load_generation)
            self.assertEqual(self.window._auto_process_pending[4], ['片头曲《段落》（TV）'])
            info.assert_not_called()
            self.window._drop_auto_process()
            self.doc.groups[0].extra['passage'] = {'id': p['id'], 'at': result.matches[0].start_ms}
            self.window.statusBar().clearMessage()
            self.window.wave.featuresLoaded.emit()
            self.assertEqual(self.window.statusBar().currentMessage(), '')
            result.matches.append(Match(10000, .95, result.matches[0].covered, [], 0, 1))
            self.window.wave.featuresLoaded.emit()
            self.assertEqual(self.window._auto_process_pending[4], ['片头曲《段落》（TV）'])
            self.window._drop_auto_process()
            self.window.statusBar().clearMessage()
            result.matches.clear()
            self.window.wave.featuresLoaded.emit()
            self.assertEqual(self.window.statusBar().currentMessage(), '')
        old_generation = self.window.wave._load_generation
        self.window.session = None
        other = SimpleNamespace(video=None)
        self.window._load_media(other)
        self.window.statusBar().clearMessage()
        with patch.object(self.window, '_search_passages') as search:
            self.window.wave._loaded(self.features, old_generation)
            self.window.wave.featuresLoaded.emit()
            search.assert_not_called()
        self.assertEqual(self.window.statusBar().currentMessage(), '')
        self.window.session = self.session

    def test_record_default_category_uses_song_style_family_or_narration(self):
        for family in ('OP', 'ED', 'IN', 'Default'):
            self.original.style = family + '.JP'
            self.doc.members['m_cn'].style = family + '.CN'
            def inspect(d):
                self.assertEqual(d.category.currentData(), family if family != 'Default' else 'NA')
                return QDialog.DialogCode.Rejected
            with patch.object(ui.RecordPassageDialog, 'exec', inspect):
                self.window._record_passage(['g_1'])
        self.doc.groups[0].kind = 'dialogue'
        self.original.style = 'OP.JP'
        with patch.object(ui.RecordPassageDialog, 'exec', inspect):
            self.window._record_passage(['g_1'])

    def test_library_rename_cancel_and_empty_usable_selection(self) -> None:
        self.record(name="段落")
        d = ui.PassageLibraryDialog(self.store, self.window)
        self.addCleanup(d.deleteLater)
        d.table.selectRow(0)
        before = self.store.path.read_bytes()
        with patch.object(ui._RenameDialog, "exec", return_value=QDialog.DialogCode.Rejected):
            d.rename()
        self.assertEqual(self.store.path.read_bytes(), before)
        for m in self.doc.members_of(self.doc.groups[0]):
            m.role = "comment"
        with patch.object(QMessageBox, "warning") as warning:
            self.window._record_passage(["g_1"])
            self.assertEqual(warning.call_args.args[2], "先在列表里选中这一段的几组。")


if __name__ == "__main__":
    unittest.main()
