"""KR2b K bar input and re-tapping, Qt offscreen with no native player."""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import time
import unittest

import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, QSettings, Qt, qVersion
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from subkit.ass import AssDocument
from subkit.core import karaoke as kara
from subkit.core.people import CharacterBook
from subkit.reading.engine import ReadingEngine
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.model.serialize import document_to_json
from subkit.styles_default import default_styles_ass
from subkit_app import mainwindow as mw
from subkit_app.player.mpv import MpvWidget
from subkit_app.session import EpisodeSession
from subkit_app.views.group_table import GroupFilter, GroupTableModel


class PlayerDouble(MpvWidget):
    def __init__(self) -> None:
        QWidget.__init__(self)
        self.time, self.duration, self.paused = 0.0, 10.0, True
        self.mpv, self.osd = None, {}
        self.message, self.overlay, self.clear_overlay, self.shutdown = Mock(), Mock(), Mock(), Mock()
        self.seek = Mock(side_effect=lambda seconds: setattr(self, "time", seconds))
        self.play = Mock(side_effect=lambda: setattr(self, "paused", False))
        self.pause = Mock(side_effect=lambda: setattr(self, "paused", True))

    def showEvent(self, event) -> None:  # noqa: N802
        QWidget.showEvent(self, event)


class KaraokeUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.platformName() != "offscreen":
            raise RuntimeError("Karaoke UI tests require QT_QPA_PLATFORM=offscreen")
        print(f"Karaoke UI: Qt {qVersion()} offscreen; fake player, no libmpv")

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="subkit_k_ui_")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.settings = QSettings(str(self.path / "settings.ini"), QSettings.Format.IniFormat)
        self.doc = Document("k-ui")
        template = AssDocument.parse(default_styles_ass())
        self.project = SimpleNamespace(data={"id": "k-ui"}, languages={"translation": "zh-Hans"},
                                       template=lambda: template, cache_dir=self.path,
                                       character_book=lambda: CharacterBook.load(self.path / "characters.json"))
        self.episode = SimpleNamespace(id="E01", dir=self.path, document=self.doc, sources={}, video=None,
                                       save=Mock(), settings=lambda: {"merge": {}, "release": {}},
                                       reading_engine=ReadingEngine)
        for target, value in (("MpvWidget", PlayerDouble), ("QSettings", Mock(return_value=self.settings)),
                              ("MainWindow._open_last", Mock())):
            p = patch("subkit_app.mainwindow." + target, value)
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.dispose)
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
        self.wave, self.player = self.window.wave, self.window.player
        self.wave.resize(1000, 200)
        self.app.processEvents()

    def dispose(self) -> None:
        if hasattr(self, "session"):
            self.session._autosave.stop()
            self.session._debounce.stop()
        if hasattr(self, "window"):
            self.window.tapper.cancel()
            self.app.removeEventFilter(self.window.tapper)
            self.window.close()
            self.window.deleteLater()
        if hasattr(self, "session"):
            self.session.deleteLater()
        self.app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.settings.sync()

    def line(self, text: str = "ガーな", *, lane: str = "original", reading: str = "",
             unit: str = "syllable", timed: bool = True, gid: str = "g_song") -> Member:
        mid = "m_" + lane + gid
        m = Member(mid, gid, "ja" if lane == "original" else "zh-Hans", lane, "song", 1000, 3000,
                   style="JP" if lane == "original" else "CN", runs=[Run(text, ins=mid, a=0, b=len(text))])
        self.doc.members[mid] = m
        if any(g.id == gid for g in self.doc.groups):
            self.doc.group(gid).members.append(mid)
        else:
            self.doc.groups.append(Group(gid, "song", [mid]))
        if reading:
            self.doc.annotations["a_" + mid] = Annotation("a_" + mid, "ruby", mid,
                [{"ins": mid, "a": 0, "b": len(text)}], text, reading, status="confirmed",
                parts=[[0, len(text), reading]])
        self.doc.structure_changed()
        if timed:
            beats = kara.derive(self.doc, m, unit)
            for i, b in enumerate(beats):
                b.t0, b.t1, b.by = 1100 + i * 400, 1500 + i * 400, "tap"
            self.session.editor.set_karaoke(mid, beats, unit=unit)
        self.window.model.reset()
        self.window.select_group(gid)
        self.wave.t0, self.wave.span = 0.0, 4000.0
        return m

    def beats(self, lane: str = "original") -> list:
        return self.wave.kviews[lane][1].beats

    def point(self, ms: float, lane: str = "original") -> QPoint:
        y0, y1 = self.wave._k_rows()[lane]
        return QPoint(round(self.wave.x_of(ms)), round((y0 + y1) / 2))

    def click(self, ms: float, *, lane: str = "original", ctrl: bool = False) -> None:
        QTest.mouseClick(self.wave, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.ControlModifier if ctrl else Qt.KeyboardModifier.NoModifier,
                         self.point(ms, lane))

    def snapshot(self) -> dict:
        out = copy.deepcopy(document_to_json(self.doc))
        out.pop("revision")
        return out

    def key(self, key: Qt.Key, *, release: bool = False) -> None:
        if release:
            QTest.keyRelease(self.wave, key)
        else:
            QTest.keyPress(self.wave, key)

    def test_rest_toolbar_shortcut_selection_delete_and_drag_undo(self):
        self.line("かなよ")
        self.click(1200)
        panel = self.window.wave_panel
        self.assertTrue(panel.btn_kaddrest.isEnabled())
        self.player.play.reset_mock()
        before, count = self.snapshot(), len(self.session.editor.history.done)
        QTest.keyClick(self.wave, Qt.Key.Key_K, Qt.KeyboardModifier.ShiftModifier)
        self.assertFalse(self.window.tapper.active)
        self.assertEqual((self.beats()[0].t1, self.beats()[1].t0), (1300, 1500))
        self.assertEqual(len(self.session.editor.history.done), count + 1)
        self.click(1400)
        self.assertEqual(self.wave.selected_rest(), ("original", 0))
        self.assertIsNone(self.wave.selected_beat())
        self.assertTrue(panel.btn_kdelrest.isEnabled())
        self.player.play.assert_not_called()
        self.wave.grab()  # Exercise the hatch painting offscreen.
        self.drag_original(1300, 1350, alt=True)
        self.assertAlmostEqual(self.beats()[0].t1, 1350, delta=3)
        self.assertEqual(self.beats()[1].t0, 1500)
        self.assertEqual(len(self.session.editor.history.done), count + 2)
        self.window.undo()
        self.drag_original(1500, 1550, alt=True)
        self.assertEqual(self.beats()[0].t1, 1300)
        self.assertAlmostEqual(self.beats()[1].t0, 1550, delta=3)
        self.window.undo()
        self.click(1400)
        QTest.keyClick(self.wave, Qt.Key.Key_Delete)
        self.assertEqual((self.beats()[0].t1, self.beats()[1].t0), (1500, 1500))
        self.assertEqual(len(self.session.editor.history.done), count + 2)
        self.window.undo()
        self.window.undo()
        self.assertEqual(self.snapshot(), before)

    def test_tapper_long_release_stored_and_threshold_is_legato(self):
        from subkit_app.karaoke_ui import GAP_MS

        for gap in (GAP_MS - 1, GAP_MS, GAP_MS + 1, 300):
            with self.subTest(gap=gap):
                self.line("かな", gid="gap" + str(gap))
                self.window._k_action("start")
                self.player.time = 1.1
                self.window.tapper._down()
                self.player.time = 1.5
                self.window.tapper._up()
                self.player.time = (1500 + gap) / 1000
                self.window.tapper._down()
                self.player.time = 2.1
                self.window.tapper._up()
                rests = kara.rests_of(self.beats())
                self.assertEqual(len(rests), int(gap > GAP_MS))
                if rests:
                    self.assertEqual((rests[0].t0, rests[0].t1), (1500, 1500 + gap))
                self.assertTrue(self.wave.kviews["original"][1].complete)
                self.wave.grab()

    def test_narrow_rest_click_selects_without_playing_and_edges_still_drag(self):
        self.line("かな")
        beats = self.beats()
        beats[0].t1 = 1492
        self.session.editor.set_karaoke(self.wave.kviews["original"][0], beats)
        self.window._refresh_k()
        self.player.play.reset_mock()
        self.click(1496)
        self.assertEqual(self.wave.selected_rest(), ("original", 0))
        self.player.play.assert_not_called()
        QTest.mouseDClick(self.wave, Qt.MouseButton.LeftButton, pos=self.point(1496))
        QTest.mouseRelease(self.wave, Qt.MouseButton.LeftButton, pos=self.point(1496))
        self.assertEqual(self.wave.selected_rest(), ("original", 0))
        self.player.play.assert_not_called()
        self.drag_original(1492, 1452, alt=True)
        self.assertEqual((self.beats()[0].t1, self.beats()[1].t0), (1452, 1500))

    def test_onset_tone_silence_onset_proposes_rest_and_overlay_end(self):
        from subkit import audio

        self.line("かな", timed=False)
        samples = np.zeros(4 * audio.RATE, np.float32)
        for a, z in ((1100, 1450), (2000, 2400)):
            n = (z - a) * audio.RATE // 1000
            samples[a * audio.RATE // 1000:z * audio.RATE // 1000] = (
                0.5 * np.sin(2 * np.pi * 440 * np.arange(n) / audio.RATE))
        self.wave._loaded(audio.compute_features(samples))
        before = self.snapshot()
        self.window._k_action("suggest")
        beats = self.wave.k_suggestions[1]
        self.assertLessEqual(abs(beats[0].t0 - 1100), 30)
        self.assertLessEqual(abs(beats[1].t0 - 2000), 30)
        self.assertLessEqual(abs(beats[0].t1 - 1450), 30)
        self.assertGreaterEqual(beats[1].t0 - beats[0].t1, 120)
        self.assertEqual(self.snapshot(), before)
        self.wave.grab()  # Includes both rest edges in the suggestion overlay.

    def test_ctrl_split_kana_at_click_time_and_single_undo(self) -> None:
        m = self.line()
        before, count = self.snapshot(), len(self.session.editor.history.done)
        self.click(1312, ctrl=True)  # three pixels past the displayed character boundary
        self.assertEqual([b.sung for b in self.beats()], ["が", "ー", "な"])
        self.assertEqual([(b.t0, b.t1) for b in self.beats()], [(1100, 1312), (1312, 1500), (1500, 1900)])
        self.assertEqual(len(self.session.editor.history.done), count + 1)
        self.assertTrue(self.session.editor.history.done[-1].keeps_text)
        self.player.play.assert_not_called()
        self.window.undo()
        self.assertEqual(self.snapshot(), before)
        self.window.redo()
        self.assertEqual(self.doc.members[m.id].text, "ガーな")
        self.assertEqual(len(self.beats()), 3)

    def test_ctrl_split_reading_kana_preserves_reading(self) -> None:
        self.line("学", reading="がっ")
        original = copy.deepcopy(self.doc.annotations)
        self.click(1300, ctrl=True)
        self.assertEqual([b.sung for b in self.beats()], ["が", "っ"])
        self.assertEqual(self.doc.annotations, original)
        self.click(1300, ctrl=True)
        self.assertEqual([b.sung for b in self.beats()], ["がっ"])

    def test_word_split_uses_displayed_character_key(self) -> None:
        self.line("Just!")
        self.click(1200, ctrl=True)
        self.assertEqual([b.sung for b in self.beats()], ["j", "ust"])
        self.assertEqual([(b.t0, b.t1) for b in self.beats()], [(1100, 1200), (1200, 1500)])

    def test_ctrl_hit_tolerance_is_inclusive_five_pixels(self) -> None:
        self.line()
        self.click(1320, ctrl=True)
        self.assertEqual(len(self.beats()), 3)
        self.window.undo()
        self.click(1324, ctrl=True)
        self.assertEqual(len(self.beats()), 2)

    def test_ctrl_merge_cross_slots_is_single_undo(self) -> None:
        self.line("メッキ", unit="mora")
        before, count = self.snapshot(), len(self.session.editor.history.done)
        self.click(1500, ctrl=True)
        self.assertEqual([b.sung for b in self.beats()], ["めっ", "き"])
        self.assertEqual((self.beats()[0].t0, self.beats()[0].t1), (1100, 1900))
        self.assertEqual(len(self.session.editor.history.done), count + 1)
        self.window.undo()
        self.assertEqual(self.snapshot(), before)

    def test_ctrl_miss_and_outer_edges_do_not_change_line(self) -> None:
        self.line()
        before = self.snapshot()
        for ms in (1100, 1220, 1900, 2800):
            self.click(ms, ctrl=True)
        self.assertEqual(self.snapshot(), before)
        self.player.play.assert_not_called()

    def test_ctrl_hover_shows_split_and_merge_positions(self) -> None:
        self.line()
        for ms, word in ((1300, "拆开"), (1500, "合并")):
            event = QMouseEvent(QEvent.Type.MouseMove, QPointF(self.point(ms)), QPointF(self.point(ms)),
                                Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.ControlModifier)
            QApplication.sendEvent(self.wave, event)
            self.assertEqual(self.wave.cursor().shape(), Qt.CursorShape.PointingHandCursor)
            self.assertIn(word, self.wave.toolTip())

    def test_placeholder_split_remains_unfinished(self) -> None:
        self.line("ガー", timed=False)
        self.click(2000, ctrl=True)
        self.assertEqual([b.sung for b in self.beats()], ["が", "ー"])
        self.assertFalse(self.wave.kviews["original"][1].complete)

    def test_click_selects_highlights_and_plays_only_beat(self) -> None:
        self.line()
        before = self.snapshot()
        unselected = self.wave.grab().toImage()
        self.click(1700)
        self.assertEqual(self.wave.selected_beat(), ("original", 1))
        self.assertNotEqual(unselected, self.wave.grab().toImage())
        self.player.seek.assert_called_with(1.5)
        self.player.play.assert_called_once()
        self.assertEqual(self.window._stop_at, 1.9)
        self.assertEqual(self.snapshot(), before)

    def test_plain_boundary_click_selects_without_editing(self) -> None:
        self.line()
        before = self.snapshot()
        self.click(1500)
        self.assertEqual(self.wave.selected_beat(), ("original", 1))
        self.assertEqual(self.snapshot(), before)

    def test_double_click_plays_to_member_end_and_release_does_not_shorten(self) -> None:
        self.line()
        self.click(1700)
        QTest.mouseDClick(self.wave, Qt.MouseButton.LeftButton, pos=self.point(1700))
        QTest.mouseRelease(self.wave, Qt.MouseButton.LeftButton, pos=self.point(1700))
        self.assertEqual(self.window._stop_at, 3.0)
        self.player.seek.assert_called_with(1.5)
        self.assertEqual(self.wave.selected_beat(), ("original", 1))

    def test_boundary_drag_keeps_existing_edit_path(self) -> None:
        self.line()
        count = len(self.session.editor.history.done)
        QTest.mousePress(self.wave, Qt.MouseButton.LeftButton, pos=self.point(1500))
        QTest.mouseMove(self.wave, self.point(1600))
        QTest.mouseRelease(self.wave, Qt.MouseButton.LeftButton, pos=self.point(1600))
        self.assertEqual([(b.t0, b.t1) for b in self.beats()], [(1100, 1600), (1600, 1900)])
        self.assertEqual(len(self.session.editor.history.done), count + 1)

    def test_k_after_playing_click_retaps_selected_and_keeps_earlier_beats(self) -> None:
        self.line("かなよ")
        before, count = self.snapshot(), len(self.session.editor.history.done)
        earlier = copy.deepcopy(self.beats()[0])
        self.click(1700)
        self.assertFalse(self.player.paused)
        self.key(Qt.Key.Key_K)
        self.key(Qt.Key.Key_K, release=True)
        self.assertEqual(self.window.tapper.run["i"], 1)
        self.assertFalse(self.window.tapper.run["auto"])
        self.assertIsNone(self.window._stop_at)
        self.player.seek.assert_called_with(0.9)
        self.assertEqual(self.window.statusBar().currentMessage(), "从第 2 拍开始重打（前面的拍不变），Esc 取消")
        for start, end in ((1.6, 1.8), (1.85, 2.1)):
            self.player.time = start
            self.key(Qt.Key.Key_K)
            self.player.time = end
            self.key(Qt.Key.Key_K, release=True)
        self.assertFalse(self.window.tapper.active)
        self.assertEqual(self.beats()[0], earlier)
        self.assertEqual([(b.t0, b.t1) for b in self.beats()[1:]], [(1600, 1850), (1850, 2100)])
        self.assertEqual(len(self.session.editor.history.done), count + 1)
        self.window.undo()
        self.assertEqual(self.snapshot(), before)

    def test_esc_during_selected_retap_changes_nothing(self) -> None:
        self.line("かなよ")
        self.click(1700)
        before, count = self.snapshot(), len(self.session.editor.history.done)
        self.key(Qt.Key.Key_K)
        self.key(Qt.Key.Key_K, release=True)
        self.player.time = 1.6
        self.key(Qt.Key.Key_K)
        self.key(Qt.Key.Key_Escape)
        self.key(Qt.Key.Key_K, release=True)
        self.assertFalse(self.window.tapper.active)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(len(self.session.editor.history.done), count)
        self.assertIsNone(self.wave.k_active)

    def test_translation_selected_retap_uses_translation_not_original(self) -> None:
        self.line("かな")
        self.line("你好", lane="translation")
        original = copy.deepcopy(self.beats())
        self.click(1700, lane="translation")
        self.key(Qt.Key.Key_K)
        self.key(Qt.Key.Key_K, release=True)
        self.assertEqual(self.window.tapper.run["lane"], "translation")
        self.assertEqual(self.window.tapper.run["i"], 1)
        self.assertEqual(self.beats(), original)

    def test_selected_first_beat_has_retap_message_and_refresh_preserves_selection(self) -> None:
        self.line()
        self.click(1200)
        self.window._refresh_k()
        self.assertEqual(self.wave.selected_beat(), ("original", 0))
        self.window._k_action("start")
        self.assertIn("从第 1 拍开始重打", self.window.statusBar().currentMessage())

    def test_group_switch_and_removed_beat_clear_selection(self) -> None:
        self.line()
        self.click(1200)
        self.click(1300, ctrl=True)
        self.assertIsNone(self.wave.selected_beat())
        self.click(1600)
        self.line("よ", gid="g_other")
        self.assertIsNone(self.wave.selected_beat())

    def test_invalid_edit_issue_shows_in_toolbar_for_either_lane(self) -> None:
        self.line("かな")
        m = self.line("ガー", lane="translation")
        self.click(1300, lane="translation", ctrl=True)
        self.session.editor.replace_text(m.id, "ガ")
        self.assertIn(kara.EDIT_MISSING, self.window.wave_panel.k_state.text())

    def test_unselected_paused_and_playing_k_keep_existing_modes(self) -> None:
        self.line("かな")
        self.key(Qt.Key.Key_K)
        self.key(Qt.Key.Key_K, release=True)
        self.assertEqual(self.window.tapper.run["i"], 0)
        self.assertFalse(self.window.tapper.run["auto"])
        self.window.tapper.cancel()
        self.player.time, self.player.paused = 1.2, False
        self.key(Qt.Key.Key_K)
        self.player.time = 1.4
        self.key(Qt.Key.Key_K, release=True)
        self.assertTrue(self.window.tapper.run["auto"])
        self.assertEqual(self.window.tapper.run["i"], 1)

    def test_arm_empty_and_invalid_indices_do_not_start_player(self) -> None:
        m = self.line()
        for beats, index in (([], 0), (self.beats(), -1), (self.beats(), 2)):
            self.window.tapper.arm(m.id, m.lane, beats, (m.start, m.end), from_index=index)
        self.assertFalse(self.window.tapper.active)
        self.player.play.assert_not_called()

    def onset_line(self, *, timed=False):
        m = self.line("かなよ", timed=timed)
        self.wave.onset = np.zeros(400, np.float32)
        self.wave.onset[[110, 170, 240]] = 1
        return m

    def test_suggestions_preview_hide_without_document_or_history_changes(self):
        self.onset_line()
        before, count = self.snapshot(), len(self.session.editor.history.done)
        image = self.wave.grab().toImage()
        self.window.wave_panel.kauto_menu.actions()[-1].trigger()
        self.assertIsNotNone(self.wave.k_suggestions)
        self.assertTrue(self.window.wave_panel.btn_kadopt.isEnabled())
        self.assertNotEqual(image, self.wave.grab().toImage())
        self.assertEqual(self.snapshot(), before)
        self.key(Qt.Key.Key_Escape)
        self.assertIsNone(self.wave.k_suggestions)
        self.assertFalse(self.window.wave_panel.btn_kadopt.isEnabled())
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(len(self.session.editor.history.done), count)

    def test_adopt_suggestions_is_one_undo_and_release_still_lists_line(self):
        m = self.onset_line()
        before, count = self.snapshot(), len(self.session.editor.history.done)
        self.window._k_action("suggest")
        self.window.wave_panel.btn_kadopt.click()
        view = self.wave.kviews["original"][1]
        self.assertEqual(view.state, "suggested")
        self.assertFalse(view.complete)
        self.assertEqual([b.by for b in view.beats], ["onset"] * 3)
        self.assertEqual([(b.t0, b.t1) for b in view.beats], [(1100, 1700), (1700, 2400), (2400, 3000)])
        self.assertIn("建议", self.window.wave_panel.k_state.text())
        self.assertIn((m.id, ["自动建议还没确认"]), kara.unfinished(self.doc))
        from subkit.core.export import build_ass
        _ass, report = build_ass(self.doc, {}, self.project.template(), self.path)
        self.assertEqual(len(report.karaoke_unfinished), 1)
        self.assertIn("自动建议还没确认", report.karaoke_unfinished[0])
        self.assertEqual(len(self.session.editor.history.done), count + 1)
        self.assertTrue(self.session.editor.history.done[-1].keeps_text)
        self.assertIsNone(self.wave.k_suggestions)
        self.window.undo()
        self.assertEqual(self.snapshot(), before)
        self.window.redo()
        self.window._k_action("confirm")
        self.assertTrue(self.wave.kviews["original"][1].complete)
        self.assertEqual(kara.unfinished(self.doc), [])

    def test_return_and_keypad_enter_adopt_without_confirming_group(self):
        self.onset_line()
        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.window._k_action("suggest")
            self.key(key)
            self.assertEqual(self.wave.kviews["original"][1].state, "suggested")
            self.assertNotEqual(self.doc.group("g_song").state, "confirmed")
            self.window.undo()

    def test_suggestion_keys_leave_text_editors_and_ctrl_enter_alone(self):
        self.onset_line()
        self.window._k_action("suggest")
        proposal = self.wave.k_suggestions
        with patch.object(self.window, "_typing", return_value=True):
            self.key(Qt.Key.Key_Return)
        self.assertIs(self.wave.k_suggestions, proposal)
        from PySide6.QtGui import QKeyEvent
        event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
        self.assertFalse(self.window.eventFilter(self.wave, event))
        self.assertIs(self.wave.k_suggestions, proposal)

    def test_selected_translation_suggestions_do_not_touch_original_or_links(self):
        self.onset_line(timed=True)
        cn = self.line("你好", lane="translation")
        original = copy.deepcopy(self.beats())
        self.click(1700, lane="translation")
        self.window._k_action("suggest")
        self.assertEqual(self.wave.k_suggestions[0], cn.id)
        self.window._k_action("adopt")
        self.assertEqual(self.beats(), original)
        self.assertEqual(self.wave.kviews["translation"][1].state, "suggested")
        self.assertFalse(any(b.link for b in self.beats("translation")))

    def test_suggestions_survive_edit_clear_on_group_switch_and_audio_reload(self):
        self.onset_line()
        self.window._k_action("suggest")
        self.window._k_split(self.wave.kviews["original"][0], self.beats()[1].parts[0].at, 1700)
        self.assertIsNotNone(self.wave.k_suggestions)
        self.window._k_action("suggest")
        self.line("?", gid="other")
        self.assertIsNone(self.wave.k_suggestions)
        self.window._k_action("suggest")
        self.wave.load(None, None, self.path)
        self.assertIsNone(self.wave.k_suggestions)
        self.window._k_action("suggest")
        self.assertIsNone(self.wave.k_suggestions)
        self.assertIn("音频尚未加载", self.window.statusBar().currentMessage())

    def test_suggestion_during_tapping_does_not_interrupt_or_write(self):
        self.onset_line()
        self.window._k_action("start")
        before = self.snapshot()
        self.window._k_action("suggest")
        self.assertTrue(self.window.tapper.active)
        self.assertIsNone(self.wave.k_suggestions)
        self.assertEqual(self.snapshot(), before)

    def test_group_selection_and_follow_keep_zoom_in_edit_and_f10(self):
        self.player.seek = Mock(side_effect=lambda seconds, **kwargs: setattr(self.player, "time", seconds))
        self.player._hide_timer = Mock()
        self.player._autohide = False
        self.line("かな", gid="short")
        second = self.line("よる", gid="long")
        second.start, second.end = 20000, 50000
        for watch in (False, True):
            if watch:
                self.window._watch_key(True)
            for gid in ("short", "long"):
                self.wave.span = 2000.0
                self.window.select_group(gid)
                self.assertEqual(self.wave.span, 2000)
            self.window._on_table_group("short")
            self.window._flush_group_selection()
            self.assertEqual(self.wave.span, 2000)
            self.window._on_wave_group("long")
            self.assertEqual(self.wave.span, 2000)
            self.wave.set_playhead(30000)
            self.assertEqual(self.wave.span, 2000)
        self.window.wave_panel.btn_fit.click()
        self.assertEqual(self.wave.span, 42000)
        self.assertEqual(self.wave.t0, 14000)

    def test_spectrum_view_and_onset_snap_preferences_are_remembered(self):
        panel = self.window.wave_panel
        self.assertEqual(self.wave.display_mode, "waveform")
        self.assertFalse(self.wave.snap_onset)
        panel.btn_spectrum.click()
        panel.btn_onset.setChecked(True)
        from subkit_app.views.waveform import WaveformPanel, WaveformView
        other = WaveformView()
        restored = WaveformPanel(other)
        try:
            restored.set_settings(self.settings)
            self.assertEqual(other.display_mode, "spectrogram")
            self.assertTrue(other.snap_onset)
        finally:
            restored.close()
            restored.deleteLater()

    def test_spectrum_visible_range_frequency_direction_and_k_interactions(self):
        self.line()
        self.wave.setFixedSize(1000, 200)  # Pixel assertions use this explicit viewport, independent of edit layout.
        self.wave.mins = self.wave.maxs = np.zeros(400, np.float32)
        self.wave.spectrogram = np.zeros((400, 128), np.uint8)
        self.wave.spectrogram[200:300, :32] = 255
        self.window.wave_panel.btn_spectrum.click()
        image = self.wave.grab().toImage()
        x = round(self.wave.x_of(2500))
        self.assertGreater(image.pixelColor(x, 160).red(), image.pixelColor(x, 75).red())
        self.wave.t0, self.wave.span = 4000, 1000
        beyond = self.wave.grab().toImage()
        self.assertEqual(beyond.pixelColor(500, 100).name(), "#101418")
        self.wave.t0, self.wave.span = 0, 4000
        self.click(1300, ctrl=True)
        self.assertEqual(len(self.beats()), 3)
        self.click(1700)
        self.assertEqual(self.wave.selected_beat(), ("original", 2))
        self.player.play.assert_called_once()

    def drag_original(self, start, end, *, alt=False):
        mods = Qt.KeyboardModifier.AltModifier if alt else Qt.KeyboardModifier.NoModifier
        QTest.mousePress(self.wave, Qt.MouseButton.LeftButton, pos=self.point(start))
        point = self.point(end)
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(point), QPointF(self.wave.mapToGlobal(point)),
                           Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, mods)
        QApplication.sendEvent(self.wave, event)
        QTest.mouseRelease(self.wave, Qt.MouseButton.LeftButton, mods, point)

    def test_onset_snap_off_on_six_pixel_boundary_alt_and_undo(self):
        self.onset_line(timed=True)
        self.wave.onset = np.zeros(400, np.float32)
        self.wave.onset[170] = 1
        self.drag_original(1500, 1676)
        self.assertEqual(self.beats()[1].t0, 1676)
        self.window.undo()
        self.window.wave_panel.btn_onset.setChecked(True)
        self.load_hint_features()
        count = len(self.session.editor.history.done)
        self.drag_original(1500, 1676)
        self.assertEqual(self.beats()[1].t0, 1700)
        self.assertEqual(len(self.session.editor.history.done), count + 1)
        self.window.undo()
        self.drag_original(1500, 1672)
        self.assertEqual(self.beats()[1].t0, 1672)
        self.window.undo()
        self.drag_original(1500, 1676, alt=True)
        self.assertEqual(self.beats()[1].t0, 1676)

    def test_adjusting_adopted_boundary_keeps_suggestion_unfinished(self):
        self.onset_line()
        self.window._k_action("suggest")
        self.window._k_action("adopt")
        self.drag_original(1700, 1800)
        view = self.wave.kviews["original"][1]
        self.assertEqual(view.state, "suggested")
        self.assertFalse(view.complete)
        self.assertTrue(kara.unfinished(self.doc))

    def load_hint_features(self):
        from subkit.audio import AudioFeatures
        n = len(self.wave.onset)
        self.wave._loaded(AudioFeatures(np.zeros(n), np.zeros(n), np.zeros((n, 128), np.uint8), self.wave.onset))
        key = self.wave._group_hint_key(self.doc.group(self.wave.current_gid))
        self.assertIn(key, self.wave._hint_cache)

    def test_translation_link_and_onset_snap_choose_nearest(self):
        self.snap_lines()
        self.wave.onset = np.zeros(400, np.float32)
        self.wave.onset[151] = 1
        self.window.wave_panel.btn_onset.setChecked(True)
        self.load_hint_features()
        self.drag_translation(1400, 1524)
        self.assertEqual(self.beats("translation")[1].t0, 1510)
        self.assertFalse(self.beats("translation")[1].link)

    def snap_lines(self):
        self.line("かな")
        cn = self.line("你好", lane="translation")
        beats = self.beats("translation")
        beats[0].t1 = beats[1].t0 = 1400
        self.session.editor.set_karaoke(cn.id, beats)
        self.window._refresh_k()
        return cn

    def drag_translation(self, start, end, *, alt=False):
        mods = Qt.KeyboardModifier.AltModifier if alt else Qt.KeyboardModifier.NoModifier
        QTest.mousePress(self.wave, Qt.MouseButton.LeftButton, pos=self.point(start, "translation"))
        point = self.point(end, "translation")
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(point), QPointF(self.wave.mapToGlobal(point)),
                           Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, mods)
        QApplication.sendEvent(self.wave, event)
        tooltip = self.wave.toolTip()
        QTest.mouseRelease(self.wave, Qt.MouseButton.LeftButton, mods, self.point(end, "translation"))
        return tooltip

    def test_translation_snap_six_pixel_threshold_creates_link_and_undo(self):
        cn = self.snap_lines()
        before, count = self.snapshot(), len(self.session.editor.history.done)
        message = self.drag_translation(1400, 1524)  # exactly six pixels from the original edge
        self.assertEqual(message, "已吸附到原声拍（以后跟着原声拍走）")
        beats = self.beats("translation")
        self.assertEqual([(b.t0, b.t1) for b in beats], [(1100, 1500), (1500, 1900)])
        self.assertTrue(beats[0].link["end"])
        self.assertEqual(beats[0].link["end"], beats[1].link["start"])
        self.assertTrue(kara.tracks_of(self.doc, cn.id)[0]["beats"][1]["link"])
        self.assertEqual(len(self.session.editor.history.done), count + 1)
        self.window.undo()
        self.assertEqual(self.snapshot(), before)

    def test_translation_alt_during_drag_disables_snap(self):
        self.snap_lines()
        message = self.drag_translation(1400, 1480, alt=True)
        self.assertEqual(message, "")
        self.assertEqual([(b.t0, b.t1) for b in self.beats("translation")], [(1100, 1480), (1480, 1900)])
        self.assertFalse(any(b.link for b in self.beats("translation")))

    def test_translation_outside_six_pixels_does_not_snap(self):
        self.snap_lines()
        self.drag_translation(1400, 1528)
        self.assertEqual(self.beats("translation")[1].t0, 1528)
        self.assertFalse(any(b.link for b in self.beats("translation")))

    def test_drag_away_unlinks_shared_boundary_and_original_no_longer_moves_it(self):
        self.snap_lines()
        self.drag_translation(1400, 1480)
        self.drag_translation(1500, 1600)
        self.assertFalse(any(b.link for b in self.beats("translation")))
        original = self.beats()
        original[0].t1 = original[1].t0 = 1550
        self.session.editor.set_karaoke(self.wave.kviews["original"][0], original)
        self.assertEqual(self.beats("translation")[1].t0, 1600)

    def test_snapped_end_and_start_follow_original_drag(self):
        self.snap_lines()
        self.drag_translation(1400, 1480)
        QTest.mousePress(self.wave, Qt.MouseButton.LeftButton, pos=self.point(1500))
        QTest.mouseMove(self.wave, self.point(1560))
        QTest.mouseRelease(self.wave, Qt.MouseButton.LeftButton, pos=self.point(1560))
        self.assertEqual([(b.t0, b.t1) for b in self.beats("translation")], [(1100, 1560), (1560, 1900)])

    def test_translation_last_end_snaps_to_original_end(self):
        self.snap_lines()
        self.drag_translation(1900, 1916)
        self.assertEqual(self.beats("translation")[-1].t1, 1900)
        self.assertEqual(self.beats("translation")[-1].link["end"]["edge"], "end")

    def test_unreachable_snap_after_minimum_duration_clamp_has_no_link(self):
        self.snap_lines()
        beats = self.beats("translation")
        beats[1].t1 = 1496
        self.session.editor.set_karaoke(self.wave.kviews["translation"][0], beats)
        self.drag_translation(1400, 1504)
        self.assertEqual(self.beats("translation")[1].t0, 1486)
        self.assertFalse(any(b.link for b in self.beats("translation")))

    def test_alt_release_removes_preview_snap(self):
        self.snap_lines()
        QTest.mousePress(self.wave, Qt.MouseButton.LeftButton, pos=self.point(1400, "translation"))
        QTest.mouseMove(self.wave, self.point(1480, "translation"))
        self.assertEqual(self.beats("translation")[1].t0, 1500)
        QTest.mouseRelease(self.wave, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier,
                           self.point(1480, "translation"))
        self.assertEqual(self.beats("translation")[1].t0, 1480)
        self.assertFalse(any(b.link for b in self.beats("translation")))


if __name__ == "__main__":
    unittest.main()
