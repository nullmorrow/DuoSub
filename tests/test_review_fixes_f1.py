"""F1 regressions: offscreen Qt events and isolated player/session doubles."""
import copy
import time
import unittest
from unittest.mock import Mock, patch
from tests import test_convert_ui as fixtures
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog
from subkit.core.export import ExportOptions, build_ass
from subkit.model.document import Member, Run


class ReviewF1(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__['setUpClass']
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group
    select = fixtures.ConversionUI.select

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.project.pinyin_mode = 'off'
        w = self.window
        w.player._hide_timer = Mock()
        w.player._autohide = False
        w.player.play = Mock(side_effect=lambda: setattr(w.player, 'paused', False))
        w.player.pause = Mock(side_effect=lambda: setattr(w.player, 'paused', True))
        w.watch.request_geometry = Mock()

    def show(self):
        self.window.show()
        self.window.activateWindow()
        self.window.table.setFocus()
        self.app.processEvents()

    def test_missing_lane_uses_template_then_defaults_and_repeated_ctrl_left(self):
        for lane in ('original', 'translation'):
            with self.subTest(missing=lane):
                self.doc.groups.clear()
                self.doc.members.clear()
                for i in range(4):
                    m = self.group(str(i), start=i * 3000)
                    if lane == 'translation':
                        m.lane, m.lang, m.style = 'original', 'ja', 'JP'
                ass, owners = build_ass(self.doc, {}, self.template, self.path, ExportOptions(ruby=False))
                w = self.window
                w.watch.ass, w.watch.owners = ass, owners.owners
                w.player.resize(1000, 700)
                w.player.osd = {'w': 1000, 'h': 700}
                w.watch_mode = True
                w.action_hub.set_scene("watch")
                self.addCleanup(setattr, w, 'watch_mode', False)
                self.assertIsNone(ass.style('JP' if lane == 'original' else 'CN'))
                w.select_group('3')
                self.show()
                for _ in range(3):
                    QTest.keyClick(w, Qt.Key.Key_Left)       # F10/F11: ← jumps to the previous group (Ctrl+← in edit mode)
                self.assertEqual(w.current_gid, '0')
                button = w.watch.missing_buttons[lane]
                self.assertFalse(button.isHidden())
                self.assertTrue(w.player.rect().contains(button.geometry()))
                style = self.session.template.style('JP' if lane == 'original' else 'CN')
                style.values['Alignment'] = '8'
                style.values['MarginV'] = '120'
                w.watch._draw_missing_buttons()
                self.assertLess(button.y(), 150)
                with patch.object(self.session.template, 'style', return_value=None):
                    w.watch._draw_missing_buttons()
                    expected = round((1080 - 60) * w.player.height() / 1080 - button.height())
                    self.assertEqual(button.y(), expected)
                w.watch_mode = False
                w.action_hub.set_scene("main")

    def test_group_playback_holds_current_through_lead_in_overlap_and_stop(self):
        self.group('a', start=1000)
        self.group('b', start=2500)
        self.group('c', start=4000)
        w = self.window
        for watch_mode in (False, True):
            with self.subTest(watch=watch_mode):
                w.watch_mode = watch_mode
                w.select_group('b')
                w.player.mpv = Mock()
                w.player.mpv.get.return_value = 25
                w.play_group()
                self.assertEqual(w._stop_at, 4.46)
                self.assertEqual(w.player.seek.call_args.args, (2.0,))
                for t in (2.0, 4.2, 4.7, 4.8):
                    w._on_time(t)
                    self.assertEqual(w.current_gid, 'b')
                    self.assertEqual(w.table.selected_gids(), ['b'])
                    self.assertEqual(w.watch.selected, 'b')
                    self.assertEqual(w.wave.current_gid, 'b')
                self.assertTrue(w.player.paused)
                self.assertEqual(w.player.seek.call_args.args, (4.46,))
                w._key_r()
                self.assertFalse(w.player.paused)
                self.assertEqual(w._stop_at, 4.46)
                if watch_mode:
                    self.assertFalse(w.wave_panel.isHidden())
        w.watch_mode = False

    def test_follow_keeps_group_while_translation_text_is_uncommitted(self):
        self.group('a', start=1000)
        self.group('b', start=2500)
        w = self.window
        w.select_group('a')
        mid, edit = w.editor.cn_edits[0]
        edit.setPlainText(edit.toPlainText() + " 加")          # typed, not yet committed (Enter / focus out)
        w.player.paused = False
        w._on_time(2.6)                                         # playback crosses into b
        self.assertEqual(w.current_gid, 'a')
        self.assertTrue(edit.toPlainText().endswith(" 加"))
        edit.commit()
        self.assertTrue(self.doc.members[mid].text.endswith(" 加"))
        w._on_time(2.7)                                         # committed: following resumes
        self.assertEqual(w.current_gid, 'b')

    def test_fps_fallback_lead_in_clamp_and_short_group(self):
        m = self.group('a', start=100)
        w = self.window
        w.select_group('a')
        for value in (None, 'bad', 0, -1, float('nan'), float('inf')):
            w.player.mpv = Mock()
            w.player.mpv.get.return_value = value
            w.play_group()
            self.assertAlmostEqual(w._stop_at, 2.1 - 1001 / 24000)
            self.assertEqual(w.player.seek.call_args.args, (0.0,))
        m.end = 110
        w.play_group()
        self.assertEqual(w._stop_at, .1)

    def test_s_d_e_keys_stop_before_group_end_and_keep_it(self):
        self.group('a', start=1000)
        self.group('b', start=2500)
        w = self.window
        w.player.mpv = Mock()
        w.player.mpv.get.return_value = 25
        for key, stop in (('S', 2.96), ('D', 1.5), ('E', 2.96)):
            with self.subTest(key=key):
                w.select_group('a')
                w._play_key(key)
                self.assertEqual(w._stop_at, stop)
                self.assertEqual(w._group_playback_gid, 'a')
                for t in (stop - 0.2, stop + 0.04):            # b already started; a stays current
                    w._on_time(t)
                    self.assertEqual(w.current_gid, 'a')
                self.assertTrue(w.player.paused)
                self.assertEqual(w.player.seek.call_args.args, (stop,))
                self.assertEqual((w.table.selected_gids(), w.watch.selected), (['a'], 'a'))
        for key in 'QW':                                       # around the group: no hold
            with self.subTest(key=key):
                w.select_group('a')
                w._play_key(key)
                self.assertIsNone(w._group_playback_gid)

    def test_explicit_resume_range_and_new_selection_release_group_hold(self):
        self.group('a', start=1000)
        self.group('b', start=4000)
        w = self.window
        w.select_group('a')
        w.play_group()
        w.select_group('b')
        self.assertIsNone(w._stop_at)
        self.assertIsNone(w._group_playback_gid)
        w.play_group()
        w._play_range(1000, 2000)
        self.assertIsNone(w._group_playback_gid)
        w.play_group()
        w.player.toggle = Mock()
        w.toggle_play()
        self.assertIsNone(w._group_playback_gid)
        w.play_group()
        w._video_resume()
        self.assertIsNone(w._group_playback_gid)

    def test_ctrl_shift_clicks_keep_scroll_and_native_selection_paused_and_playing(self):
        for i in range(80):
            self.group(str(i), start=i * 3000)
        self.show()
        w, table = self.window, self.window.table
        for playing in (False, True):
            for modifier in (Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.ShiftModifier):
                with self.subTest(playing=playing, modifier=modifier):
                    self.select('30')
                    w.player.paused = not playing
                    before = (table.horizontalScrollBar().value(), table.verticalScrollBar().value())
                    pos = table.visualRect(w.proxy.index(32, 4)).center()
                    self.assertTrue(table.viewport().rect().contains(pos))
                    QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, modifier, pos)
                    expected = ['30', '32'] if modifier == Qt.KeyboardModifier.ControlModifier else ['30', '31', '32']
                    self.assertEqual(table.selected_gids(), expected)
                    self.assertEqual(w.current_gid, '32')
                    w._on_time(0.5)
                    deadline = time.monotonic() + 2   # the deferred refresh (~150 ms) may run late on a busy machine
                    while w._pending_group is not None and time.monotonic() < deadline:
                        QTest.qWait(10)
                    QTest.qWait(20)
                    self.assertEqual(table.selected_gids(), expected)
                    self.assertEqual((table.horizontalScrollBar().value(), table.verticalScrollBar().value()), before)
                    w.player.paused = False
                    w._on_time(1.0)
                    self.assertEqual(w.current_gid, '0')
                    self.assertEqual(table.selected_gids(), expected)
                    self.assertEqual((table.horizontalScrollBar().value(), table.verticalScrollBar().value()), before)

    def new_character(self, lane, original_label, translation_label, whole=True, accepted=True):
        tr = self.group('a')
        tr.speaker = {'label': translation_label}
        jp = Member('jp', 'a', 'ja', 'original', 'dialogue', 1000, 3000,
                    speaker={'label': original_label}, runs=[Run('私', ins='jp', a=0, b=1)])
        self.doc.members[jp.id] = jp
        self.doc.groups[0].members.append(jp.id)
        self.doc.structure_changed()
        self.window.select_group('a')
        self.settings.setValue('watch/speaker_lane', lane)
        captured = []
        class DetailsDouble:
            DialogCode = QDialog.DialogCode
            def __init__(self, parent, character, languages, label):
                captured.append((copy.deepcopy(character), label))
            def exec(self):
                return QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected
            def changes(self):
                ch = captured[0][0]
                return {'names': ch.names, 'aliases': ch.aliases}
            def deleteLater(self):
                pass
        with patch('subkit_app.mainwindow.CharacterDetailsDialog', DetailsDouble):
            self.window._set_speaker(tr.id, 'new', whole)
        return captured[0], tr, jp

    def test_new_character_prefers_display_lane_language_alias_and_one_undo(self):
        (ch, label), tr, jp = self.new_character('original', 'ウテナ', '欧蒂娜')
        self.assertEqual(label, 'ウテナ')
        self.assertEqual(ch.names['ja'].given, label)
        self.assertEqual(ch.aliases, [label])
        self.assertEqual(tr.speaker['character'], jp.speaker['character'])
        self.assertEqual(len(self.session.editor.history.done), 1)
        cid = tr.speaker['character']
        self.assertTrue(any(c.id == cid for c in self.session.editor.character_book.characters))
        self.session.editor.history.undo()
        self.assertIsNone(self.doc.members[tr.id].speaker.get('character'))
        self.assertIsNone(self.doc.members[jp.id].speaker.get('character'))
        self.assertFalse(any(c.id == cid for c in self.session.editor.character_book.characters))
        self.session.editor.history.redo()
        self.assertEqual(self.doc.members[tr.id].speaker['character'], cid)
        self.assertEqual(self.doc.members[jp.id].speaker['character'], cid)

    def test_new_character_translation_lane_and_single_member_option(self):
        (ch, label), tr, jp = self.new_character('translation', 'ウテナ', '欧蒂娜', whole=False)
        self.assertEqual(ch.names['zh-Hans'].given, '欧蒂娜')
        self.assertEqual(label, '欧蒂娜')
        self.assertTrue(tr.speaker.get('character'))
        self.assertIsNone(jp.speaker.get('character'))

    def test_new_character_falls_back_to_any_label(self):
        (ch, label), _, _ = self.new_character('original', '', '欧蒂娜')
        self.assertEqual(ch.names['zh-Hans'].given, '欧蒂娜')
        self.assertEqual(label, '欧蒂娜')

    def test_new_character_cancel_keeps_document_and_history(self):
        self.new_character('original', 'ウテナ', '欧蒂娜', accepted=False)
        self.assertEqual(self.session.editor.history.done, [])
        self.assertTrue(all(not m.speaker.get('character') for m in self.doc.members.values()))
