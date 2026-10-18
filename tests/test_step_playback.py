"""Step timing and offscreen shortcuts with isolated settings/player doubles."""

import unittest
from unittest.mock import Mock, patch

from tests import test_convert_ui as fixtures
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit
from subkit.ass import AssDocument, Event
from subkit.core.stepping import fully_visible_time
from subkit.model.document import Member, Run


class StepTiming(unittest.TestCase):
    def visible(self, *events, start=1000, end=5000, frame=40, owners=None, members=()):
        return fully_visible_time('g', start, end, frame, events=events,
                                  owners=owners if owners is not None else [{'group': 'g'}] * len(events),
                                  members=members)

    def event(self, text, start=100, end=500):
        return Event(start=start, end=end, text=text)

    def test_fad_and_fade_directions(self):
        for text, expected in ((r'{\fad(450,900)}x', 1450),
                               (r'{\fade(255,0,255,0,700,900,1200)}x', 1700),
                               (r'{\fade(0,255,0,0,700,900,1200)}x', 1000),
                               (r'{\fade(0,0,255,0,700,900,1200)}x', 1000)):
            with self.subTest(text=text):
                self.assertEqual(self.visible(self.event(text)), expected)

    def test_alpha_transforms_threshold_acceleration_and_nested_tags(self):
        for alpha in ('alpha', '1a', '2a', '3a', '4a'):
            for a, expected in ((0, 1800), (300, 1800), (301, 1000)):
                with self.subTest(alpha=alpha, a=a):
                    text = rf'{{\t({a},800,2,\clip(1,2,3,4)\{alpha}&H00&)}}x'
                    self.assertEqual(self.visible(self.event(text)), expected)
        self.assertEqual(self.visible(self.event(r'{\t(0,1200,\fs50)}x')), 1000)
        self.assertEqual(self.visible(self.event(r'{\fad(200,0)\t(100,600,\alpha&H00&)}x')), 1600)

    def test_caps_and_clamps(self):
        self.assertEqual(self.visible(self.event(r'{\fad(9000,0)}x')), 3000)
        self.assertEqual(self.visible(self.event(r'{\fad(9000,0)}x', end=120)), 1200)
        self.assertEqual(self.visible(self.event(r'{\fad(9000,0)}x'), end=1300), 1260)
        self.assertEqual(self.visible(self.event(r'{\fad(9000,0)}x'), end=1010), 1000)
        self.assertEqual(self.visible(self.event(r'{\fad(50,0)}x', start=50)), 1000)

    def test_lead_in_initial_frame_and_later_events(self):
        events = (self.event(r'{\fad(800,0)}x', start=70),
                  self.event(r'{\fad(1000,0)}x', start=99),
                  self.event(r'{\fad(2000,0)}later highlight', start=100),
                  self.event(r'{\fad(2000,0)}later line', start=130))
        self.assertEqual(self.visible(*events), 1990)
        events = (self.event(r'{\fad(100,0)}x'),
                  self.event(r'{\fad(500,0)}ruby', start=104),
                  self.event(r'{\fad(2000,0)}late', start=105))
        self.assertEqual(self.visible(*events), 1540)

    def test_owner_comment_invalid_duration_and_member_fallback(self):
        member = Member('m', 'g', 'ja', 'original', 'dialogue', 1000, 5000,
                        runs=[Run(r'{\fad(350,0)}', kind='tag'), Run('x')])
        self.assertEqual(self.visible(members=[member]), 1350)
        event = self.event(r'{\fad(900,0)}x')
        self.assertEqual(self.visible(event, owners=[{'group': 'other'}], members=[member]), 1350)
        self.assertEqual(self.visible(event.copy(kind='Comment'), members=[member]), 1350)
        self.assertEqual(self.visible(event.copy(end=100), members=[member]), 1350)
        self.assertEqual(self.visible(), 1000)
        self.assertEqual(self.visible(self.event('x'), members=[member]), 1000)

    def test_invalid_tags_and_fractional_frame(self):
        self.assertEqual(self.visible(self.event(r'{\fad(bad,0)\t(no,800,\alpha&H00&)}x')), 1000)
        self.assertAlmostEqual(self.visible(self.event(r'{\fad(9000,0)}x'), end=2000,
                                            frame=1001 / 24), 2000 - 1001 / 24)


class StepPlaybackUI(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__['setUpClass']
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        w = self.window
        w.player._hide_timer = Mock()
        w.player._autohide = False
        w.player.play = Mock(side_effect=lambda: setattr(w.player, 'paused', False))
        w.player.pause = Mock(side_effect=lambda: setattr(w.player, 'paused', True))
        w.player.toggle = Mock(side_effect=lambda: setattr(w.player, 'paused', not w.player.paused))
        w.player.mpv = Mock()
        w.player.mpv.get.return_value = 25
        w.watch.request_geometry = Mock()
        w.act_step_playback.setChecked(True)

    def show(self):
        w = self.window
        w.show()
        w.activateWindow()
        w.table.setFocus()
        self.app.processEvents()

    def test_in_group_holds_selection_no_lead_in_and_pause(self):
        self.group('a', start=1000)
        self.group('b', start=2500)
        w = self.window
        w.select_group('a')
        w.player.time = 1.7
        w.player.seek.reset_mock()
        w._space()
        self.assertEqual(w._stop_at, 2.96)
        w.player.seek.assert_not_called()
        w._on_time(2.7)
        self.assertEqual(w.current_gid, 'a')
        w._on_time(3.1)
        self.assertTrue(w.player.paused)
        self.assertEqual(w.player.seek.call_args.args, (2.96,))
        self.assertEqual((w.current_gid, w.wave.current_gid, w.watch.selected), ('a', 'a', 'a'))
        w.player.paused = False
        w._space()
        self.assertTrue(w.player.paused)
        self.assertIsNone(w._stop_at)

    def test_playhead_reported_slightly_before_group_start_counts_as_inside(self):
        # mpv reports frame-aligned times a few ms before the stop point; Space must not replay the same group
        self.group('a', start=1000)
        self.group('b', start=4000)
        w = self.window
        w.select_group('b')
        for early in (0.001, 0.02, 0.06):
            with self.subTest(early=early):
                w.player.paused = True
                w.player.time = 4.0 - early
                w._space()
                self.assertEqual(w._stop_at, 5.96)            # to the end of b, not b's start again
                self.assertIsNone(w._step_select_gid)
                self.assertEqual(w._group_playback_gid, 'b')
        w.player.paused = True
        w.player.time = 3.5                                   # a real gap: still steps to b
        w._space()
        self.assertEqual(w._step_select_gid, 'b')
        self.assertEqual(w._stop_at, 4.0)

    def test_end_next_group_preview_stop_seek_back_selection_focus_zoom(self):
        self.group('a', start=1000)
        self.group('b', start=4000)
        w = self.window
        w.select_group('a')
        w.watch.ass = AssDocument()
        w.watch.ass.events = [Event(start=400, end=600, text=r'{\fad(600,0)}b')]
        w.watch.owners = [{'group': 'b'}]
        self.show()
        w.wave.span = 1234
        focus = QApplication.focusWidget()
        w.player.time = 2.96
        w._space()
        self.assertEqual(w._stop_at, 4.6)
        self.assertEqual(w.current_gid, 'a')
        w._on_time(4.4)
        self.assertEqual(w.current_gid, 'a')
        w._on_time(4.85)
        self.assertEqual(w.player.seek.call_args.args, (4.6,))
        self.assertEqual((w.current_gid, w.wave.current_gid, w.watch.selected), ('b', 'b', 'b'))
        self.assertEqual(w.table.selected_gids(), ['b'])
        self.assertEqual(w.wave.span, 1234)
        self.assertIs(QApplication.focusWidget(), focus)
        w.player.time = 4.6
        w._space()
        self.assertEqual(w._stop_at, 5.96)

    def test_blank_sorted_next_last_and_other_kinds(self):
        self.group('late', start=8000)
        self.group('sign', kind='sign', start=2000)
        self.group('early', text=r'{\fad(350,0)}x', start=4000)
        w = self.window
        w.player.time = .5
        w._space()
        self.assertEqual(w._stop_at, 4.35)
        w._on_time(4.4)
        self.assertEqual(w.current_gid, 'early')
        w.player.time = 10
        calls = w.player.play.call_count
        w._space()
        self.assertEqual(w.player.play.call_count, calls)
        self.assertEqual(w.statusBar().currentMessage(), '已经是最后一组')

    def test_containment_threshold_selected_preferred_other_group(self):
        self.group('a', start=1000)
        self.group('b', start=2000)
        self.group('c', start=6000)
        w = self.window
        w.select_group('a')
        w.player.time = 2.949
        w._space()
        self.assertEqual(w._group_playback_gid, 'a')
        w.player.paused = True
        w.player.time = 2.950
        w._space()
        self.assertEqual(w._group_playback_gid, 'b')
        self.assertEqual(w.current_gid, 'b')
        w.player.paused = True
        w.player.time = 3.96
        w._space()
        self.assertEqual(w._step_select_gid, 'c')

    def test_karaoke_only_songs(self):
        self.group('dialogue', start=1000)
        self.group('song', kind='song', start=4000)
        w = self.window
        w.set_edit_layout('karaoke')
        w.player.time = 1.5
        w._space()
        self.assertEqual(w._step_select_gid, 'song')
        w._on_time(4.1)
        self.assertEqual(w.current_gid, 'song')

    def test_f10_step_and_button_context_help(self):
        self.group('a', start=1000)
        w = self.window
        self.show()
        w.wave_panel.btn_step.setFocus()
        self.app.processEvents()
        with patch('subkit_app.mainwindow.app_help.open_manual') as opener:
            QTest.keyClick(w.wave_panel.btn_step, Qt.Key.Key_F1)
            opener.assert_called_once_with('playback')
        w.set_watch_mode(True, pinned=True)
        w.table.setFocus()
        self.app.processEvents()
        w.player.time = 1.5
        QTest.keyClick(w, Qt.Key.Key_Space)
        self.assertEqual(w._stop_at, 2.96)
        w._on_time(3.1)
        self.assertEqual(w.current_gid, 'a')
        self.assertFalse(w.wave_panel.isHidden())
        QTest.keyClick(w, Qt.Key.Key_Space,
                       Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        self.assertFalse(w.act_step_playback.isChecked())
        self.assertFalse(w.wave_panel.btn_step.isChecked())

    def test_f11_step_keeps_waveform_hidden_and_off_resume(self):
        self.group('a', start=1000)
        self.group('b', start=4000)
        w = self.window
        self.show()
        w.set_watch_mode(True)
        w.select_group('a')
        w.player.time = 1.5
        w._show_watch_wave = Mock()
        QTest.keyClick(w, Qt.Key.Key_Space)
        self.assertEqual(w._stop_at, 2.96)
        w._on_time(3.1)
        w.player.time = 2.96
        QTest.keyClick(w, Qt.Key.Key_Space)
        w._on_time(4.1)
        self.assertEqual(w.current_gid, 'b')
        w._show_watch_wave.assert_not_called()
        self.assertTrue(w.wave_panel.isHidden())
        w.act_step_playback.setChecked(False)
        w._space()
        self.assertIsNone(w.watch.selected)
        self.assertFalse(w.player.paused)
        w._show_watch_wave.assert_called_with(False)

    def test_shortcuts_sync_persistence_off_behaviour_and_typing(self):
        self.group('a')
        w = self.window
        self.show()
        modifier = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        QTest.keyClick(w, Qt.Key.Key_Space, modifier)
        self.assertFalse(w.act_step_playback.isChecked())
        self.assertFalse(w.wave_panel.btn_step.isChecked())
        self.assertFalse(self.settings.value('playback/step', type=bool))
        QTest.keyClick(w, Qt.Key.Key_Space)
        w.player.toggle.assert_called_once()
        w.player.paused = True
        w.wave_panel.btn_step.click()
        self.assertTrue(w.act_step_playback.isChecked())
        self.settings.sync()
        other = fixtures.mw.MainWindow()
        try:
            self.assertTrue(other.act_step_playback.isChecked())
            self.assertTrue(other.wave_panel.btn_step.isChecked())
        finally:
            other.close()
            other.deleteLater()
        field = QLineEdit(w)
        field.show()
        field.setFocus()
        self.app.processEvents()
        before = w.player.play.call_count
        QTest.keyClick(field, Qt.Key.Key_Space)
        self.assertEqual(w.player.play.call_count, before)
        self.assertIsNone(w._stop_at)
        self.assertEqual(field.text(), ' ')

    def test_other_playback_actions_and_selection_cancel_pending_step(self):
        self.group('a')
        self.group('b', start=4000)
        w = self.window
        w.select_group('a')
        actions = [w._key_r, w.toggle_play, lambda: w._play_neighbour(1),
                   lambda: w.select_group('b'), lambda: w._play_range(4500, 4800)]
        actions += [lambda key=key: w._play_key(key) for key in 'SQWDE']
        for action in actions:
            with self.subTest(action=action):
                w.select_group('a')
                w.player.paused = True
                w.player.time = .5
                w._space()
                self.assertEqual(w._step_select_gid, 'a')
                action()
                self.assertIsNone(w._step_select_gid)
        w.player.paused = True
        w.player.time = .5
        w._space()
        self.show()
        QTest.keyClick(w, Qt.Key.Key_Space, Qt.KeyboardModifier.ControlModifier)
        self.assertIsNone(w._step_select_gid)


if __name__ == '__main__':
    unittest.main()
