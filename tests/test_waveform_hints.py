"""KC: real synthetic PCM, per-group synchronous caching and timing undo; offscreen."""
from __future__ import annotations

import os
import sys
import gc
import weakref
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcutEvent
from PySide6.QtTest import QTest
from subkit import audio, autotime
from subkit.model.document import Group, Member, Run
from subkit_app.views import waveform
from tests import test_karaoke_ui as ku


def speech_pcm():
    samples = np.zeros(audio.RATE * 5, np.float32)
    rng = np.random.default_rng(12)
    for start, end in ((1200, 1900), (3200, 4100)):
        a, b = start * 16, end * 16
        samples[a:b] = rng.normal(0, 0.15, b - a)
    return audio.compute_features(samples)


class HintDSP(unittest.TestCase):
    def test_known_speech_segments_within_30ms_and_abstention(self):
        features = speech_pcm()
        runs = autotime.speech_segments(autotime._snap_activity(features), 500, 4800)
        self.assertEqual(len(runs), 2)
        for actual, expected in zip(runs, ((1200, 1900), (3200, 4100))):
            for a, b in zip(actual, expected):
                self.assertLessEqual(abs(a - b), 30)
        self.assertEqual(autotime.speech_segments(np.ones(300, np.float32), 0, 3000), [])
        self.assertEqual(autotime.speech_segments(np.zeros(300), 0, 3000), [])
        cut = autotime.speech_segments(autotime._snap_activity(features), 1400, 2400)
        self.assertIsNone(cut[0][0])

    def test_offset_sustained_energy_drop_and_short_quiet_rejected(self):
        envelope = np.zeros(200, np.float32)
        envelope[20:65], envelope[100:150] = 0.5, 0.4
        f = audio.AudioFeatures(-envelope, envelope, np.zeros((200, 128), np.uint8), np.zeros(200))
        self.assertEqual(audio.offset_edges(f, np.array([200, 1000]), 0, 2000).tolist(), [650, 1500])
        envelope[68:100] = 0.5
        self.assertEqual(audio.offset_edges(f, np.array([200, 1000]), 0, 2000).tolist(), [1500])


class HintUI(ku.KaraokeUI):
    def test_one_sided_white_label_shortcut_and_exact_undo(self):
        from PySide6.QtGui import QPainter
        labels = []
        class RecordingPainter(QPainter):
            def drawText(self, *args):
                labels.extend(a for a in args if isinstance(a, str))
                return super().drawText(*args)
        self.dialogue()
        shortcut = self.window.action_hub.action("time.adopt")
        for runs, expected, label, absent in (([(1200, None)], (930, None), "建议开始", "建议结束"),
                                              ([(None, 4100)], (None, 4100), "建议结束", "建议开始")):
            with self.subTest(side=label), patch.object(waveform, "speech_segments", return_value=runs), \
                    patch.object(autotime, "speech_segments", return_value=runs):
                self.wave._hint_cache.clear()
                self.wave._refresh_hints()
                self.assertEqual(self.wave.suggested_times(), expected)
                self.assertTrue(self.window.wave_panel.btn_adopt_time.isEnabled())
                labels.clear()
                with patch.object(waveform, "QPainter", RecordingPainter):
                    self.wave.grab()
                self.assertIn(label, labels)
                self.assertNotIn(absent, labels)
                count = len(self.session.editor.history.done)
                QCoreApplication.sendEvent(shortcut, QShortcutEvent(QKeySequence("C"), 0))
                m = self.doc.members["m_dialogue"]
                self.assertEqual((m.start, m.end), tuple(c if p is None else p
                                                       for c, p in zip((1000, 4300), expected)))
                self.assertEqual(len(self.session.editor.history.done), count + 1)
                self.assertTrue(m.extra["time_manual"])
                self.window.undo()
                m = self.doc.members["m_dialogue"]
                self.assertEqual((m.start, m.end), (1000, 4300))
                self.assertNotIn("time_manual", m.extra)

    def test_c_shortcut_ignored_while_typing(self):
        self.dialogue()
        shortcut = self.window.action_hub.action("time.adopt")
        with patch.object(self.window, "_adopt_timing") as adopt:
            with patch.object(self.window, "_typing", return_value=True):
                QCoreApplication.sendEvent(shortcut, QShortcutEvent(QKeySequence("C"), 0))          # a "c" typed into a text field
            adopt.assert_not_called()
            with patch.object(self.window, "_typing", return_value=False):
                QCoreApplication.sendEvent(shortcut, QShortcutEvent(QKeySequence("C"), 0))
            adopt.assert_called_once()

    def test_run_on_cache_reuses_detection_and_tracks_neighbour_time(self):
        self.dialogue()
        m = self.doc.members["m_dialogue"]
        m.start, m.end = 1000, 2000
        following = self.neighbour("following", 2200, 3600, "dialogue")
        curve = np.zeros(500, np.float32)
        curve[90:370], curve[205], curve[250] = .5, .2, .25
        self.wave._features.snap_activity = curve
        self.wave._hint_cache.clear()
        with patch.object(waveform, "speech_segments", wraps=waveform.speech_segments) as detector:
            hints = self.wave._group_hints(self.doc.group(m.group))
            self.assertEqual(hints["end"], [2050])
            self.assertEqual(self.wave.suggested_times(), (630, 2050))
            following.start = 1900  # Now overlapping: no fabricated split.
            self.assertIs(self.wave._group_hints(self.doc.group(m.group)), hints)
            self.assertEqual(hints["end"], [])
            self.assertEqual(self.wave.suggested_times(), (630, None))
            self.assertEqual(detector.call_count, 1)

    def current_hints(self):
        self.wave._nearby_hints()
        key = self.wave._group_hint_key(self.doc.group(self.wave.current_gid))
        self.assertIn(key, self.wave._hint_cache)
        return self.wave._hint_cache[key]

    def dialogue(self):
        m = Member("m_dialogue", "g_dialogue", "ja", "original", "dialogue", 1000, 4300,
                   runs=[Run("台詞", ins="i_dialogue", a=0, b=2)])
        self.doc.members[m.id] = m
        self.doc.groups.append(Group(m.group, "dialogue", [m.id]))
        self.doc.structure_changed()
        self.window.select_group(m.group)
        # Existing hint tests exercise explicit autotime padding (270/0).
        panel = self.window.wave_panel
        panel.padding_mode.setCurrentIndex(panel.padding_mode.findData("autotime"))
        panel.sync_hint_padding()
        self.wave._loaded(speech_pcm())
        self.wave.t0, self.wave.span = 0, 5000
        self.window.wave_panel.btn_hint_lines.setChecked(True)
        self.wave.viewChanged.emit()
        self.current_hints()
        return m

    def test_dialogue_adopt_button_shortcut_one_undo_manual_and_other_group_untouched(self):
        song = self.line(gid="other")
        m = self.dialogue()
        times = self.wave.suggested_times()
        self.assertIsNotNone(times)
        self.assertLessEqual(abs(times[0] - 930), 30)
        self.assertLessEqual(abs(times[1] - 4100), 30)
        count = len(self.session.editor.history.done)
        self.window.wave_panel.btn_adopt_time.click()
        self.assertEqual((self.doc.members[m.id].start, self.doc.members[m.id].end), times)
        self.assertTrue(self.doc.members[m.id].extra["time_manual"])
        self.assertEqual(self.doc.members[m.id].extra["time_src"], {"start": "adopted", "end": "adopted"})
        self.assertEqual((self.doc.members[song.id].start, self.doc.members[song.id].end), (1000, 3000))
        self.assertEqual(len(self.session.editor.history.done), count + 1)
        self.assertEqual(self.session.editor.history.done[-1].label, "采用建议时间")
        self.window.undo()
        self.assertEqual((self.doc.members[m.id].start, self.doc.members[m.id].end), (1000, 4300))
        self.assertNotIn("time_manual", self.doc.members[m.id].extra)
        shortcuts = [s for s in self.window.findChildren(QAction) if QKeySequence("C") in s.shortcuts()]
        self.assertEqual(len(shortcuts), 1)
        QCoreApplication.sendEvent(shortcuts[0], QShortcutEvent(QKeySequence("C"), 0))
        self.assertEqual((self.doc.members[m.id].start, self.doc.members[m.id].end), times)
        self.assertTrue(self.wave.hint_lines)

    def neighbour(self, gid, start, end, kind="song"):
        m = Member("m_" + gid, gid, "ja", "original", "dialogue", start, end,
                   runs=[Run("隣", ins=gid, a=0, b=1)])
        self.doc.members[m.id] = m
        self.doc.groups.append(Group(gid, kind, [m.id]))
        self.doc.structure_changed()
        return m

    def test_suggestion_uses_current_neighbours_even_with_cached_hints(self):
        m = self.dialogue()
        m.end = 4000
        previous = self.neighbour("previous", 0, 980)
        following = self.neighbour("following", 4050, 4700)
        self.neighbour("sign", 900, 1100, "sign")
        hints = self.current_hints()
        self.assertEqual(self.wave.suggested_times(), (980, 4050))
        previous.end, following.start = 995, 4020
        self.assertIs(self.current_hints(), hints)
        self.assertEqual(self.wave.suggested_times(), (995, 4020))
        previous.end, following.start = 1100, 3900
        self.assertEqual(self.wave.suggested_times(), hints["suggested"])

    def test_adopt_shortcut_clamps_both_sides_and_undo_preserves_neighbours(self):
        m = self.dialogue()
        m.end = 4000
        previous = self.neighbour("previous", 0, 980)
        following = self.neighbour("following", 4050, 4700)
        shortcuts = [s for s in self.window.findChildren(QAction) if QKeySequence("C") in s.shortcuts()]
        QCoreApplication.sendEvent(shortcuts[0], QShortcutEvent(QKeySequence("C"), 0))
        self.assertEqual((self.doc.members[m.id].start, self.doc.members[m.id].end), (980, 4050))
        self.assertEqual((previous.start, previous.end, following.start, following.end), (0, 980, 4050, 4700))
        self.window.undo()
        self.assertEqual((self.doc.members[m.id].start, self.doc.members[m.id].end), (1000, 4000))

    def test_empty_clamped_suggestion_disables_adoption_and_shortcut_is_noop(self):
        self.dialogue()
        self.neighbour("previous", 0, 980)
        # The shared core rejects an empty neighbour-clamped interval.
        with patch.object(waveform, "suggest_group_times", return_value=None):
            self.wave._hint_cache.clear()
            self.wave._refresh_hints()
        self.assertIsNone(self.wave.suggested_times())
        self.assertFalse(self.window.wave_panel.btn_adopt_time.isEnabled())
        count = len(self.session.editor.history.done)
        self.window._adopt_timing()
        self.assertEqual(len(self.session.editor.history.done), count)

    @patch.object(waveform, "speech_segments", return_value=[(1200, 1900)])
    def test_nearest_targets_edge_types_six_pixels_and_click_modifiers(self, segments):
        self.dialogue()
        self.window.wave_panel.btn_onset.setChecked(True)
        self.current_hints()
        hints = self.current_hints()
        hints["start"], hints["end"] = np.array([1200]), np.array([1900])
        self.wave.keyframes = np.array([1210])
        self.assertEqual(self.wave.snap_ms(1208, edge="start"), 1210)
        self.assertEqual(self.wave.snap_ms(1202, edge="start"), 1200)
        self.wave.snap = False
        tolerance = 6 * self.wave.span / self.wave.width()
        self.assertEqual(self.wave.snap_ms(1200 + tolerance, edge="start"), 1200)
        self.assertEqual(self.wave.snap_ms(1200 + tolerance + 1, edge="start"), 1200 + tolerance + 1)
        self.assertEqual(self.wave.snap_ms(1202, edge="end"), 1202)
        recorded = []
        self.wave.timeClicked.connect(lambda ms, edge: recorded.append((ms, edge)))
        for mode, free in (("aegisub", Qt.KeyboardModifier.ShiftModifier), ("select", Qt.KeyboardModifier.AltModifier)):
            self.wave.timing_mode = mode
            modifiers = Qt.KeyboardModifier.ShiftModifier if mode == "select" else Qt.KeyboardModifier.NoModifier
            for bypass in (False, True):
                self.doc.members["m_dialogue"].start = 1000
                point = ku.QPoint(round(self.wave.x_of(1215)), 100)
                QTest.mouseClick(self.wave, Qt.MouseButton.LeftButton, modifiers | (free if bypass else Qt.KeyboardModifier.NoModifier), point)
                self.assertEqual(recorded[-1][0], round(self.wave.ms_of(point.x())) if bypass else 1200)

    def test_per_group_cache_scroll_edits_and_padding_preferences(self):
        other = self.line(gid="other")
        m = self.dialogue()
        key = self.wave._hint_group_keys[m.group]
        old = self.current_hints()
        song_key = self.wave._hint_group_keys[other.group]
        song_hints = self.wave._hint_cache[song_key]
        with patch.object(waveform, "speech_segments", wraps=waveform.speech_segments) as edges:
            for offset in (100, 200, 300):
                self.wave.t0 = offset
                self.wave.repaint()
                self.wave.viewChanged.emit()
                self.assertIs(self.current_hints(), old)
            self.assertEqual(edges.call_count, 0)
            # No event processing or background delivery between edit and snap/paint.
            m.start += 20
            self.wave.snap_onset = True
            self.wave.snap = False
            self.assertEqual(self.wave.snap_ms(old["start"][0] + 2, edge="start"), old["start"][0])
            self.assertEqual(edges.call_count, 1)
            self.assertNotEqual(key, self.wave._hint_group_keys[m.group])
            self.assertNotIn(key, self.wave._hint_cache)
            self.assertIs(self.wave._hint_cache[song_key], song_hints)
            self.assertTrue(self.current_hints()["start"])
            self.assertEqual(edges.call_count, 1)
        self.settings.setValue("autotime/lead_in", 0.1)
        self.settings.setValue("autotime/lead_out", 0.05)
        self.window.wave_panel.sync_hint_padding()
        self.assertLessEqual(abs(self.wave.suggested_times()[0] - 1100), 30)
        self.assertLessEqual(abs(self.wave.suggested_times()[1] - 4150), 30)
        self.assertLessEqual(len(self.wave._hint_cache), len(self.doc.groups))

    def test_activity_once_per_features_and_released_on_switch_close(self):
        self.dialogue()
        features = speech_pcm()
        with patch.object(waveform, "_snap_activity", wraps=waveform._snap_activity) as activity:
            self.wave._loaded(features)
            curve_ref = weakref.ref(features.snap_activity)
            self.wave._loaded(features)
            self.wave.viewChanged.emit()
            self.wave.repaint()
            self.wave.lead_in_ms = 100
            self.wave._refresh_hints()
            self.assertEqual(activity.call_count, 1)
            self.assertTrue(self.current_hints()["start"])
            self.wave.load(None, None, self.path)
            gc.collect()
            self.assertIsNone(curve_ref())
            self.assertIsNone(features.snap_activity)
            self.assertEqual(self.wave._hint_cache, {})
            self.assertEqual(self.wave._hint_group_keys, {})
            self.assertIsNone(self.wave.suggested_times())
            self.wave._loaded(features)
            curve_ref = weakref.ref(features.snap_activity)
            self.assertEqual(activity.call_count, 2)
            self.wave.close()
            gc.collect()
            self.assertIsNone(curve_ref())

    def test_song_hints_use_whole_group_and_survive_scroll_and_beat_edits(self):
        self.line()
        envelope = np.zeros(500, np.float32)
        envelope[110:160], envelope[200:250] = 0.5, 0.4
        onset = np.zeros(500, np.float32)
        onset[110], onset[200] = 1, 0.8
        self.wave._loaded(audio.AudioFeatures(-envelope, envelope,
                          np.zeros((500, 128), np.uint8), onset))
        self.window.wave_panel.btn_hint_lines.setChecked(True)
        hints = self.current_hints()
        self.assertEqual(hints["onset"].tolist(), [1100, 2000])
        self.assertEqual(hints["offset"].tolist(), [1600, 2500])
        self.wave.snap_onset, self.wave.snap = True, False
        self.assertEqual(self.wave.snap_ms(1602, beat=True), 1600)
        self.wave.t0, self.wave.span = 1500, 1000
        with patch.object(waveform, "onset_peaks", wraps=waveform.onset_peaks) as peaks:
            self.wave.viewChanged.emit()
            self.drag_original(1500, 1550)
            self.assertIs(self.current_hints(), hints)
            self.assertEqual(peaks.call_count, 0)

    def test_suggestions_survive_unrelated_beat_drag_then_clear_on_group_change(self):
        self.onset_line(timed=True)
        self.window._k_action("suggest")
        proposal = self.wave.k_suggestions
        self.drag_original(1500, 1550)
        self.assertIs(self.wave.k_suggestions, proposal)
        self.assertTrue(self.window.wave_panel.btn_kadopt.isEnabled())
        self.line(gid="next")
        self.assertIsNone(self.wave.k_suggestions)

    def test_hint_preferences_keep_old_snap_setting_and_song_controls(self):
        panel = self.window.wave_panel
        self.settings.setValue("karaoke/snap_onset", True)
        panel.set_settings(self.settings)
        self.assertTrue(self.wave.snap_onset)
        panel.btn_hint_lines.setChecked(True)
        self.assertTrue(self.settings.value("waveform/hint_lines", False, type=bool))
        self.line()
        self.assertTrue(panel.btn_khintlines.isChecked())
        panel.btn_khintsnap.click()
        self.assertFalse(self.settings.value("karaoke/snap_onset", True, type=bool))
        panel.set_settings(self.settings)
        self.assertTrue(self.wave.hint_lines)
        self.assertFalse(self.wave.snap_onset)


    def test_one_nearest_pair_and_omitted_edges_do_not_snap(self):
        self.dialogue()
        self.wave._hint_cache.clear()
        with patch.object(waveform, "speech_segments", return_value=[(800, 1100), (1200, 1900), (3200, 4100), (4400, 4700)]):
            hints = self.current_hints()
        self.assertEqual(hints["start"], [1200])
        self.assertEqual(hints["end"], [4100])
        self.wave.snap_onset, self.wave.snap = True, False
        self.assertEqual(self.wave.snap_ms(1202, edge="start"), 1200)
        self.assertEqual(self.wave.snap_ms(802, edge="start"), 802)
        self.assertEqual(self.wave.snap_ms(4102, edge="end"), 4100)

    def test_legend_modes_fixed_cancel_project_isolation_and_shortcut(self):
        m = self.dialogue()
        panel = self.window.wave_panel
        self.project.save = unittest.mock.Mock()
        panel.set_hint_context(self.project)
        self.assertEqual(panel.padding_mode.currentText(), "按我调过的行（推荐）")
        self.assertFalse(panel.legend.isHidden())
        for label in ("▷ 人声开始", "◁ 人声结束", "│ 起音", "│ 收音", "粉线 画面切换"):
            self.assertIn(label, panel.legend.text())
        panel.btn_hint_lines.setChecked(False)
        self.assertTrue(panel.legend.isHidden())
        panel.btn_hint_lines.setChecked(True)
        panel.padding_mode.setCurrentIndex(3)
        with patch.object(panel, "_fixed_padding_dialog", return_value=(100, -50)):
            panel.padding_mode.activated.emit(3)
        self.assertEqual(self.project.data["settings"]["hint_padding"]["fixed"], [100, -50])
        self.assertIn("提前 100 毫秒、结束前 50 毫秒（手动设定）", panel.legend.text())
        hints = self.current_hints()
        expected = (hints["start"][0] - 100, hints["end"][0] - 50)
        self.assertEqual(self.wave.suggested_times(), expected)
        self.window._adopt_timing()
        self.assertEqual((m.start, m.end), expected)
        self.window.undo()
        panel.set_hint_context(self.project)
        self.assertEqual(panel.padding_mode.currentData(), "fixed")
        panel.padding_mode.setCurrentIndex(2)
        panel.padding_mode.activated.emit(2)
        self.assertIn("同自动调轴设置", panel.legend.text())
        panel.padding_mode.setCurrentIndex(3)
        with patch.object(panel, "_fixed_padding_dialog", return_value=None):
            panel.padding_mode.activated.emit(3)
        self.assertEqual(panel.padding_mode.currentData(), "autotime")
        from types import SimpleNamespace
        panel.set_hint_context(SimpleNamespace(data={}, save=unittest.mock.Mock()))
        self.assertEqual(panel.padding_mode.currentData(), "manual")
        self.assertEqual((self.wave.lead_in_ms, self.wave.lead_out_ms), waveform.HABIT_INITIAL_MS)

    def test_episode_calibration_once_recompute_and_project_fallback(self):
        self.project.data["settings"] = {"hint_padding": {"mode": "episode", "mode_explicit": True}}
        self.dialogue()
        for i in range(19):
            self.neighbour("cal" + str(i), 1000, 4300, "dialogue")
        panel = self.window.wave_panel
        self.project.save = unittest.mock.Mock()
        panel.set_hint_context(self.project)
        with patch.object(waveform, "calibrate_hint_padding", wraps=autotime.calibrate_hint_padding) as calibrate:
            self.wave._loaded(speech_pcm())
            self.assertEqual(calibrate.call_count, 1)
            values = (self.wave.lead_in_ms, self.wave.lead_out_ms)
            self.assertLessEqual(abs(values[0] - 200), 30)
            self.assertLessEqual(abs(values[1] - 200), 30)
            self.assertEqual(self.project.data["settings"]["hint_padding"]["last_episode"], list(values))
            self.doc.members["m_dialogue"].start += 10
            self.wave.viewChanged.emit()
            self.wave._loaded(speech_pcm())
            self.assertEqual(calibrate.call_count, 1)
            panel.btn_recalibrate.click()
            self.assertEqual(calibrate.call_count, 2)
            self.doc.groups = self.doc.groups[:1]
            self.doc.structure_changed()
            panel.set_hint_context(self.project)
            self.wave._loaded(speech_pcm())
            self.assertEqual(panel._episode_padding.source, "project")
            self.assertIn("按本集字幕算出", panel.legend.text())

    def test_manual_mode_uses_habit_then_learns_twenty_eligible_groups(self):
        self.dialogue()
        panel = self.window.wave_panel
        panel.set_hint_context(self.project)
        self.assertEqual([panel.padding_mode.itemText(i) for i in range(4)],
                         ["按我调过的行（推荐）", "按本集字幕", "同自动调轴设置", "固定值…"])
        self.settings.setValue("waveform/habit/lead_in_ms", 150)
        self.settings.setValue("waveform/habit/lead_out_ms", 200)
        # Imported groups and manually timed signs/songs must not train habits.
        for i in range(25):
            m = self.neighbour("import" + str(i), 1000, 4300, "dialogue")
            m.extra["text_manual"] = True
        for i in range(19):
            m = self.neighbour("mine" + str(i), 1000, 4300, "dialogue")
            if i % 2:
                self.doc.group(m.group).state = "confirmed"
            else:
                m.extra["time_manual"] = True
        song = self.neighbour("manual_song", 1000, 4300, "song")
        song.extra["time_manual"] = True
        self.doc.group(song.group).state = "confirmed"
        panel.recalibrate_hint_padding()
        self.assertEqual((self.wave.lead_in_ms, self.wave.lead_out_ms), (150, 200))
        self.assertIn("按我的习惯", panel.legend.text())
        self.assertEqual(panel._manual_padding.onset_samples, 19)
        self.doc.group("g_dialogue").state = "confirmed"
        panel.btn_recalibrate.click()
        self.assertEqual(panel._manual_padding.source, "manual")
        self.assertIn("按我调过的行算出", panel.legend.text())
        learned = self.wave.lead_in_ms, self.wave.lead_out_ms
        self.assertLessEqual(abs(learned[0] - 200), 30)
        self.assertLessEqual(abs(learned[1] - 200), 30)
        self.assertEqual((self.settings.value("waveform/habit/lead_in_ms", type=int),
                          self.settings.value("waveform/habit/lead_out_ms", type=int)), learned)
        self.assertNotIn("habit", self.project.data.get("settings", {}))
        from types import SimpleNamespace
        panel.set_hint_context(SimpleNamespace(data={}, save=unittest.mock.Mock()))
        self.assertEqual((self.wave.lead_in_ms, self.wave.lead_out_ms), learned)
        self.assertIn("按我的习惯", panel.legend.text())

    def test_manual_habit_pools_cached_episode_without_writing_it(self):
        import hashlib
        from subkit.model.document import Document
        from subkit.model.serialize import save_document, write_json_atomic
        self.dialogue()
        self.doc.group("g_dialogue").state = "confirmed"
        for i in range(9):
            m = self.neighbour("mine" + str(i), 1000, 4300, "dialogue")
            m.extra["time_manual"] = True
        other = Document("other")
        for i in range(10):
            mid, gid = "m" + str(i), "g" + str(i)
            other.groups.append(Group(gid, "dialogue", [mid], state="confirmed"))
            other.members[mid] = Member(mid, gid, "ja", "original", "dialogue", 1000, 4300)
        folder = self.path / "other"
        folder.mkdir()
        video = self.path / "media.mkv"
        video.write_bytes(b"metadata-only video")
        metadata = folder / "episode.json"
        document = folder / "document.json"
        write_json_atomic(metadata, {"media": {"video": {"path": str(video)}, "audio_track": 1}})
        save_document(other, document)
        features = speech_pcm()
        cache = waveform._waveform_cache_file(video, 1, self.path / "waveform")
        cache.parent.mkdir()
        with cache.open("wb") as stream:
            np.savez_compressed(stream, **{k: getattr(features, k) for k in ("mins", "maxs", "spectrogram", "onset")})
        tracked = (metadata, document, video, cache)
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked]
        self.project.root = self.path
        self.project.data["episodes"] = [{"id": "k-ui", "path": "current"}, {"id": "other", "path": "other"}]
        panel = self.window.wave_panel
        panel.set_hint_context(self.project)
        with patch.object(waveform, "_decode_samples", side_effect=AssertionError("must not decode")):
            panel.recalibrate_hint_padding()
        self.assertEqual(panel._manual_padding.source, "manual")
        self.assertEqual(panel._manual_padding.onset_samples, 20)
        self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked])
        cache.unlink()
        panel.recalibrate_hint_padding()
        self.assertEqual(panel._manual_padding.source, "habit")
        self.assertEqual(panel._manual_padding.onset_samples, 10)

    def test_episode_migration_preserves_explicit_old_choices(self):
        self.project.save = unittest.mock.Mock()
        panel = self.window.wave_panel
        for saved, expected in (({}, "manual"), ({"mode": "episode", "mode_explicit": False}, "manual"),
                                ({"mode": "episode"}, "episode"),
                                ({"mode": "episode", "mode_explicit": True}, "episode"),
                                ({"mode": "fixed", "fixed": [30, -40]}, "fixed")):
            self.project.data["settings"] = {"hint_padding": saved}
            panel.set_hint_context(self.project)
            self.assertEqual(panel.padding_mode.currentData(), expected)
        panel.padding_mode.setCurrentIndex(panel.padding_mode.findData("episode"))
        panel.padding_mode.activated.emit(panel.padding_mode.currentIndex())
        self.assertTrue(self.project.data["settings"]["hint_padding"]["mode_explicit"])

    def test_fixed_dialog_ranges_accept_cancel_and_help(self):
        from PySide6.QtWidgets import QDialog, QSpinBox
        panel = self.window.wave_panel
        def accept(dialog):
            self.assertEqual(dialog.help_anchor, "timing")
            boxes = dialog.findChildren(QSpinBox)
            self.assertEqual([(b.minimum(), b.maximum()) for b in boxes], [(0, 600), (-200, 600)])
            boxes[0].setValue(140)
            boxes[1].setValue(-20)
            return QDialog.DialogCode.Accepted
        with patch.object(QDialog, "exec", accept):
            self.assertEqual(panel._fixed_padding_dialog((270, 0)), (140, -20))
        with patch.object(QDialog, "exec", return_value=QDialog.DialogCode.Rejected):
            self.assertIsNone(panel._fixed_padding_dialog((270, 0)))

    def test_yellow_markers_current_opaque_others_dim_in_both_views(self):
        self.dialogue()
        self.neighbour("dim", 3000, 4300, "dialogue")
        self.wave._loaded(speech_pcm())
        self.wave.t0, self.wave.span = 0, 5000
        from PySide6.QtGui import QColor
        for mode in ("waveform", "spectrogram"):
            self.wave.display_mode = mode
            image = self.wave.grab().toImage()
            current = image.pixelColor(round(self.wave.x_of(1200)) + 2, 32)
            other = image.pixelColor(round(self.wave.x_of(3200)) + 2, 32)
            self.assertEqual(current.name(), QColor("#ffd33d").name())
            self.assertLess(other.red(), current.red())
            self.assertGreater(other.red(), 60)
            evidence = Path(os.environ["TEMP"]) / ("w3a-hints-" + mode + ".png")
            self.assertTrue(image.save(str(evidence)))
