"""Offscreen conversion entry points; isolated settings and no libmpv or preview workers."""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import QCoreApplication, QEvent, QItemSelectionModel, QSettings, QTimer, qVersion  # noqa: E402
from PySide6.QtGui import QContextMenuEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMenu, QMessageBox, QWidget  # noqa: E402

from subkit.ass import AssDocument  # noqa: E402
from subkit.ass.text import tokenize  # noqa: E402
from subkit.core import karaoke as kara  # noqa: E402
from subkit.core.convert import K_HINT, export_stripped_text  # noqa: E402
from subkit.core.export import ExportOptions, build_ass  # noqa: E402
from subkit.core.people import CharacterBook  # noqa: E402
from subkit.reading.engine import ReadingEngine  # noqa: E402
from subkit.core.fontcheck import FontCheck  # noqa: E402
from subkit.core.merge import MergeSettings  # noqa: E402
from subkit.model.document import Annotation, Document, Group, Member, Run  # noqa: E402
from subkit.model.serialize import document_to_json  # noqa: E402
from subkit.styles_default import default_styles_ass  # noqa: E402
from subkit_app import mainwindow as mw  # noqa: E402
from subkit_app.player.mpv import MpvWidget  # noqa: E402
from subkit_app.session import EpisodeSession  # noqa: E402
from subkit_app.views.dialogs import ConvertToSongDialog, ExportDialog  # noqa: E402
from subkit_app.views.group_table import GroupFilter, GroupTableModel  # noqa: E402


class PlayerDouble(MpvWidget):
    """Keep the real Qt signal interface without creating a native player."""

    def __init__(self) -> None:
        QWidget.__init__(self)
        self.time, self.duration, self.paused = 0.0, 0.0, True
        self.mpv, self.osd = None, {}
        self.message = Mock()
        self.seek = Mock()
        self.overlay = Mock()
        self.clear_overlay = Mock()
        self.shutdown = Mock()

    def showEvent(self, event) -> None:  # noqa: N802
        QWidget.showEvent(self, event)


class ConversionUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.platformName() != "offscreen":
            raise RuntimeError("Conversion UI tests require QT_QPA_PLATFORM=offscreen")

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="subkit_convert_ui_")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.settings = QSettings(str(self.path / "settings.ini"), QSettings.Format.IniFormat)
        self.doc = Document("convert-ui")
        self.template = AssDocument.parse(default_styles_ass())
        self.merge = {}
        self.project = SimpleNamespace(data={"id": "convert-ui"}, languages={"translation": "zh-Hans"},
                                       template=lambda: self.template, cache_dir=self.path,
                                       character_book=lambda: CharacterBook.load(self.path / "characters.json"))
        self.episode = SimpleNamespace(id="E01", dir=self.path, document=self.doc, sources={}, video=None,
                                       save=Mock(), settings=lambda: {"merge": self.merge, "release": {}},
                                       reading_engine=ReadingEngine)
        self.addCleanup(self._dispose)
        for target, value in (("MpvWidget", PlayerDouble), ("QSettings", Mock(return_value=self.settings)),
                              ("MainWindow._open_last", Mock())):
            p = patch("subkit_app.mainwindow." + target, value)
            p.start()
            self.addCleanup(p.stop)
        self.window = mw.MainWindow()
        self.session = EpisodeSession(self.project, self.episode)
        self.session.request_preview = Mock()
        self.window.project, self.window.session = self.project, self.session
        self.window.model = GroupTableModel(self.doc)
        self.window.proxy = GroupFilter()
        self.window.proxy.setSourceModel(self.window.model)
        self.window.table.setModel(self.window.proxy)
        self.window.wave.set_document(self.doc)
        self.window.source.set_data(self.doc, {})
        self.session.changed.connect(self.window._on_changed)

    def _dispose(self) -> None:
        if hasattr(self, "session"):
            self.session._autosave.stop()
            self.session._debounce.stop()
        if hasattr(self, "window"):
            self.window.close()
            self.window.deleteLater()
        if hasattr(self, "session"):
            self.session.deleteLater()
        self.app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.settings.sync()

    def group(self, gid: str, kind: str = "dialogue", text: str = "歌词", start: int = 1000) -> Member:
        mid = "m_" + gid
        runs = [Run(t.raw, "tag" if t.kind in ("tags", "comment") else "text", ins=mid, a=t.start, b=t.end)
                for t in tokenize(text)]
        member = Member(mid, gid, "zh-Hans", "translation", "song" if kind == "song" else "dialogue",
                        start, start + 2000, style="OP.CN" if kind == "song" else "CN", runs=runs)
        self.doc.groups.append(Group(gid, kind, [mid]))
        self.doc.members[mid] = member
        self.doc.structure_changed()
        self.window.model.reset()
        return member

    def select(self, *gids: str) -> None:
        self.window.select_group(gids[0])
        sm = self.window.table.selectionModel()
        sm.clearSelection()
        for gid in gids:
            index = self.window.proxy.mapFromSource(self.window.model.index(self.doc.group_index(gid), 0))
            sm.select(index, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)

    def dialog(self, gids: list[str], **kwargs) -> ConvertToSongDialog:
        d = ConvertToSongDialog(self.doc, gids, self.window, styles=kwargs.get("styles", MergeSettings().styles),
                                template_styles=[s.name for s in self.template.styles], settings=self.settings)
        self.addCleanup(d.deleteLater)
        return d

    def snapshot(self) -> dict:
        result = copy.deepcopy(document_to_json(self.doc))
        result.pop("revision")
        return result

    def context_menu(self, choose: str | None = None) -> list:
        actions = []

        class MenuDouble(QMenu):
            def exec(self, _pos):
                actions.extend(self.actions())
                def find(items):
                    for action in items:
                        if choose and choose in action.text():
                            return action
                        if action.menu():
                            found = find(action.menu().actions())
                            if found is not None:
                                return found
                    return None
                return find(actions)

        table = self.window.table
        table.resize(1000, 400)
        pos = table.visualRect(table.currentIndex()).center()
        event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, pos, table.mapToGlobal(pos))
        with patch("subkit_app.views.group_table.QMenu", MenuDouble):
            table.contextMenuEvent(event)
        return actions

    def test_sign_menu_reasons_conversion_speaker_and_one_undo(self) -> None:
        m = self.group("sign", "sign", text="我")
        m.role, m.style = "sign", "Sign"
        self.select("sign")
        self.window._update_convert_actions()
        self.assertTrue(self.window.act_convert_dialogue.isEnabled())
        self.assertFalse(self.window.act_convert_sign.isEnabled())
        self.assertIn("已经是屏幕字", self.window.act_convert_sign.text())
        self.assertEqual(self.window.model.data(self.window.model.index(0, 7)), "")
        before = self.snapshot()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.window._convert_groups(["sign"], "dialogue")
        self.assertEqual(self.window.model.data(self.window.model.index(0, 7)), "?")
        self.window._update_convert_actions()
        self.assertTrue(self.window.act_convert_sign.isEnabled())
        self.assertIn("已经是对白", self.window.act_convert_dialogue.text())
        self.assertIn("已转为普通台词：1 组", self.window.statusBar().currentMessage())
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.context_menu("屏幕字…")
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.window.model.data(self.window.model.index(0, 7)), "")
        self.window.undo()
        self.assertEqual(self.doc.group("sign").kind, "dialogue")
        self.window.redo()
        self.assertEqual(self.snapshot(), before)
        self.group("song", "song")
        self.select("song")
        self.window._update_convert_actions()
        self.assertIn("歌曲先转为对白", self.window.act_convert_sign.text())

    def test_context_menu_wording_order_and_enabled_states(self) -> None:
        self.group("d", text="普通台词")
        self.group("s", "song")
        self.group("i", "info")
        for gids, enabled in ((["d"], (True, False)), (["s"], (False, True)),
                              (["i"], (False, False)), (["d", "s", "i"], (True, True))):
            with self.subTest(gids=gids):
                self.select(*gids)
                actions = self.context_menu()
                names = [a.text() for a in actions]
                convert = next((a for a in actions if a.text() == "转为"), None)
                if any(enabled):
                    self.assertIsNotNone(convert)
                    choices = convert.menu().actions()
                    labels = [a.text() for a in choices]
                    self.assertEqual("歌曲（K 轴）…" in labels, enabled[0])
                    self.assertEqual("普通台词…" in labels, enabled[1])
                    self.assertTrue(all(a.isEnabled() for a in choices))
                    index = names.index("转为")
                    self.assertTrue(actions[index - 1].isSeparator())
                else:
                    self.assertIsNone(convert)
                self.assertIn("确认这一组\tCtrl+Shift+Enter", names)
                self.assertNotIn("取消确认", names)
                self.assertEqual(names[-1], "删除这一组（原声和翻译都删）\tDel")

    def test_context_menu_emits_selected_gids_and_target(self) -> None:
        self.group("a", text="台词一")
        self.group("b", text="台词二")
        self.select("a", "b")
        got = []
        self.window.table.convertRequested.disconnect(self.window._convert_groups)
        self.window.table.convertRequested.connect(lambda gids, to: got.append((gids, to)))
        self.context_menu("歌曲（K 轴）…")
        self.assertEqual(got, [(["a", "b"], "song")])
        self.session.editor.convert_to_song(["a", "b"], "IN")
        self.select("a", "b")
        self.context_menu("普通台词…")
        self.assertEqual(got[-1], (["a", "b"], "dialogue"))

    def test_edit_menu_uses_selection_and_disables_ineligible_actions(self) -> None:
        self.window._update_convert_actions()
        self.assertFalse(self.window.act_convert_song.isEnabled())
        self.assertFalse(self.window.act_convert_dialogue.isEnabled())
        self.group("a", text="一")
        self.group("b", text="二")
        self.select("a", "b")
        menu_actions = self.window.menuBar().actions()
        menu = next(a.menu() for a in menu_actions if a.text() == "字幕(&S)")
        menu.aboutToShow.emit()
        self.assertTrue(self.window.act_convert_song.isEnabled())
        self.assertFalse(self.window.act_convert_dialogue.isEnabled())
        names = [a.property("enabled_text") or a.text() for a in menu.actions()]
        self.assertEqual(names.index("转为"), names.index("AI听译…") + 2)
        convert = next(a.menu() for a in menu.actions() if a.text() == "转为")
        self.assertEqual(convert.actions()[0].text(), "歌曲（K 轴）…")
        with patch.object(self.window, "_convert_groups") as convert:
            self.window.act_convert_song.trigger()
        convert.assert_called_once_with(["a", "b"], "song")

    def test_dialog_family_default_last_choice_and_saved_song_priority(self) -> None:
        self.group("d")
        self.assertEqual(self.dialog(["d"]).family(), "IN")
        self.settings.setValue("convert/song_family", "ED")
        self.assertEqual(self.dialog(["d"]).family(), "ED")
        self.session.editor.convert_to_song(["d"], "OP")
        self.session.editor.convert_to_dialogue(["d"])
        self.assertEqual(self.dialog(["d"]).family(), "OP")
        self.group("plain", text="另一句")
        self.settings.setValue("convert/song_family", "invalid")
        self.assertEqual(self.dialog(["plain"]).family(), "IN")

    def test_dialog_replans_custom_styles_and_missing_template_warning(self) -> None:
        self.group("d")
        styles = {**MergeSettings().styles, "translation/song": "{fam}.自定义", "original/song": "{fam}.原声"}
        d = self.dialog(["d"], styles=styles)
        self.assertEqual(d.style_label.text(), "样式：IN.自定义 / IN.原声")
        self.assertIn("模板里没有 IN.自定义 / IN.原声", d.explanation.text())
        d.families["ED"].click()
        self.assertEqual(d.plan.family, "ED")
        self.assertEqual(d.style_label.text(), "样式：ED.自定义 / ED.原声")
        self.assertIn("模板里没有 ED.自定义 / ED.原声", d.explanation.text())
        self.assertNotIn("IN.自定义", d.explanation.text())
        self.assertEqual(d.windowTitle(), "转为歌曲（K 轴）")
        bb = d.findChild(QDialogButtonBox)
        self.assertEqual(bb.button(QDialogButtonBox.StandardButton.Ok).text(), "转换")
        self.assertEqual(bb.button(QDialogButtonBox.StandardButton.Cancel).text(), "取消")

    def test_dialog_lists_twins_skips_and_dormant_timing(self) -> None:
        m = self.group("a", "song")
        self.group("twin", "song", start=1010)
        beats = kara.derive(self.doc, m)
        for b in beats:
            b.t0, b.t1, b.by = m.start, m.end, "tap"
        self.session.editor.set_karaoke(m.id, beats, twins=True)
        self.session.editor.convert_to_dialogue(["a"])
        self.group("skip", "sign")
        d = self.dialog(["a", "skip"])
        self.assertIn("另外 1 组是同一句的另一图层", d.explanation.text())
        self.assertIn("1 组不是普通台词", d.explanation.text())
        self.assertIn("会恢复 2 行之前打过的 K 轴", d.explanation.text())

    def test_cancel_song_dialog_changes_neither_document_nor_last_family(self) -> None:
        self.group("a")
        before = self.snapshot()

        original = ConvertToSongDialog.exec
        with patch.object(ConvertToSongDialog, "exec", lambda d: (d.families["ED"].click(),
                          QTimer.singleShot(0, d.reject), original(d))[-1]):
            self.window._convert_groups(["a"], "song")
        self.assertEqual(self.snapshot(), before)
        self.assertIsNone(self.settings.value("convert/song_family"))
        self.assertIsNone(self.session.editor.history.undo())
        self.session.request_preview.assert_not_called()

    def accept_song(self, gids: list[str], family: str = "ED") -> None:
        original = ConvertToSongDialog.exec

        def accept(d):
            d.families[family].click()
            bb = d.findChild(QDialogButtonBox)
            QTimer.singleShot(0, bb.button(QDialogButtonBox.StandardButton.Ok).click)
            return original(d)

        with patch.object(ConvertToSongDialog, "exec", accept):
            self.window._convert_groups(gids, "song")

    def test_accept_song_refreshes_list_k_preview_and_one_undo_restores(self) -> None:
        self.group("a")
        self.select("a")
        before = self.snapshot()
        self.accept_song(["a"])
        self.assertEqual(self.doc.group("a").kind, "song")
        self.assertEqual(self.doc.members["m_a"].style, "ED.CN")
        self.assertIn("歌曲", self.window.model.data(self.window.model.index(0, 6)))
        self.assertIn("translation", self.window.wave.kviews)
        self.assertIn("已转为歌曲：1 组", self.window.statusBar().currentMessage())
        self.assertEqual(self.settings.value("convert/song_family"), "ED")
        self.session.request_preview.assert_called_once()
        self.window.undo()
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(self.window.wave.kviews)
        self.assertEqual(self.session.request_preview.call_count, 2)
        self.assertIsNone(self.session.editor.history.undo())
        self.window.redo()
        self.assertEqual(self.doc.group("a").kind, "song")
        self.assertIn("translation", self.window.wave.kviews)

    def test_conversion_uses_episode_style_mapping_for_both_directions(self) -> None:
        self.group("a")
        self.group("imported", "song", text="另一首")
        self.merge["styles"] = {**MergeSettings().styles, "translation/song": "{fam}.定制",
                                "translation/dialogue": "中文对白", "original/dialogue": "原声对白"}
        self.accept_song(["a"], "OP")
        self.assertEqual(self.doc.members["m_a"].style, "OP.定制")
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes) as question:
            self.window._convert_groups(["imported"], "dialogue")
        self.assertIn("没有记录的用 中文对白 / 原声对白", question.call_args.args[2])
        self.assertEqual(self.doc.members["m_imported"].style, "中文对白")

    def test_dialogue_cancel_and_accept_keep_timing_and_refresh_k(self) -> None:
        m = self.group("a", "song", text=r"{\k20}歌{\kf30}词")
        self.select("a")
        self.session.editor.import_karaoke(m.id)
        before = self.snapshot()
        self.session.request_preview.reset_mock()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No) as question:
            self.window._convert_groups(["a"], "dialogue")
        self.assertIn("已打的 K 轴（1 行）留在工程里但不导出", question.call_args.args[2])
        self.assertIn("原字幕里自带的 \\k（1 行）不写进输出", question.call_args.args[2])
        self.assertEqual(self.snapshot(), before)
        self.session.request_preview.assert_not_called()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.window._convert_groups(["a"], "dialogue")
        self.assertEqual(self.doc.group("a").kind, "dialogue")
        self.assertFalse(self.window.wave.kviews)
        self.assertTrue(all(t["dormant"] for t in self.doc.karaoke.values()))
        self.assertIn("已转为普通台词：1 组", self.window.statusBar().currentMessage())
        self.window.undo()
        self.assertEqual(self.snapshot(), before)
        self.assertIn("translation", self.window.wave.kviews)

    def test_k_on_dialogue_shows_conversion_hint(self) -> None:
        self.group("a")
        self.select("a")
        self.window._k_action("start")
        self.window.player.message.assert_called_once_with(K_HINT)

    def test_session_conversion_and_undo_preserve_original_readings(self) -> None:
        m = self.group("a", text="歌詞")
        m.lane, m.lang, m.style = "original", "ja", "JP"
        self.doc.annotations["saved"] = Annotation("saved", "ruby", m.id, [{"ins": m.id, "a": 0, "b": 2}],
                                                   "歌詞", "かし", status="auto")
        self.session.engine = SimpleNamespace(analyze=Mock(return_value=[]))
        self.select("a")
        readings = copy.deepcopy(self.doc.annotations)
        self.accept_song(["a"])
        self.assertEqual(self.doc.annotations, readings)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.window._convert_groups(["a"], "dialogue")
        self.assertEqual(self.doc.annotations, readings)
        self.window.undo()
        self.window.undo()
        self.window.redo()
        self.assertEqual(self.doc.annotations, readings)
        self.session.engine.analyze.assert_not_called()

    def test_session_refreshes_readings_only_when_the_text_changes(self) -> None:
        m = self.group("a", text="歌詞")
        m.lane, m.lang, m.style = "original", "ja", "JP"
        self.doc.annotations["saved"] = Annotation("saved", "ruby", m.id, [{"ins": m.id, "a": 0, "b": 2}],
                                                   "歌詞", "かし", status="auto")
        self.session.engine = SimpleNamespace(analyze=Mock(return_value=[]))
        readings = copy.deepcopy(self.doc.annotations)
        ed = self.session.editor
        ed.set_overrides({m.id: {"an": 7, "pos": [100.0, 200.0]}}, "位置")       # position, format, time: text kept
        ed.set_format(m.id, {"fs": 60})
        ed.set_member_time(m.id, start=m.start + 100)
        self.window.undo()
        self.window.redo()
        self.assertEqual(self.doc.annotations, readings)
        self.session.engine.analyze.assert_not_called()
        ed.replace_text(m.id, "歌詞だ")                                          # a text edit still refreshes
        self.session.engine.analyze.assert_called()

    def test_export_info_and_utf8_report_include_stripped_line_and_six_refs(self) -> None:
        for i in range(8):
            self.group(str(i), "song", text=rf"{{\k20}}第{i}句", start=1000 + i * 4000)
        self.session.editor.convert_to_dialogue([g.id for g in self.doc.groups])
        ass, report = build_ass(self.doc, {}, self.template, self.path, ExportOptions(ruby=False))
        self.assertEqual(len(report.karaoke_stripped), 8)
        out = self.path / "output.ass"
        self.episode.validate = Mock(return_value=SimpleNamespace(ok=True, summary=lambda: "通过"))
        self.episode.build_export = Mock(return_value=(ass, report))

        def write_export(result, path):
            result.save(path)
            return path

        self.episode.write_export = Mock(side_effect=write_export)
        original = ExportDialog.exec

        def accept_export(d):
            d.path.setText(str(out))
            QTimer.singleShot(0, d.accept)
            return original(d)

        with patch.object(ExportDialog, "exec", accept_export), \
                patch.object(mw, "check_fonts", return_value=FontCheck()), \
                patch.object(QMessageBox, "information") as info:
            self.window.export()
        expected = export_stripped_text(report.karaoke_stripped)
        self.assertIn(expected, info.call_args.args[2])
        written = out.with_suffix(".export-report.txt").read_text(encoding="utf-8")
        self.assertIn(expected, written)
        self.assertNotIn(report.karaoke_stripped[6], written)
        self.assertIsNone(QApplication.overrideCursor())
        self.assertNotIn(r"\k20", out.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    print(f"Qt {qVersion()}, platform=offscreen, player/preview/font checks=test doubles")
    unittest.main()
