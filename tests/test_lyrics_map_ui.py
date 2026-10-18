"""Qt offscreen mapping dialog; player double and no native browser/player."""
import copy
import os
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import Qt, qVersion
from PySide6.QtWidgets import QApplication, QMessageBox
from subkit.core import lyrics_map as lm
from subkit_app import help as app_help
from subkit_app.views.lyrics_map import LyricsMapDialog
from subkit_app.views.waveform import WaveformPanel, WaveformView
from tests.test_lyrics_map import document


class LyricsMapUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        print("Lyrics mapping UI: Qt", qVersion(), "offscreen; no native mpv/browser")

    def test_button_disabled_reasons_and_signal(self):
        doc, _ = document()
        panel = WaveformPanel(WaveformView())
        self.addCleanup(panel.deleteLater)
        reasons = []
        panel.kAction.connect(reasons.append)
        panel.set_mapping_reason(lm.disabled_reason(doc, doc.groups[0], {"original": "ja", "translation": "zh-Hant"}))
        self.assertTrue(panel.btn_kmap.isEnabled())
        panel.btn_kmap.click()
        self.assertEqual(reasons, ["mapping"])
        for lane in ("original", "translation"):
            target = next(m for m in doc.members.values() if m.lane == lane)
            target.runs = []
            reason = lm.disabled_reason(doc, doc.groups[0], {"original": "ja", "translation": "zh-Hant"})
            self.assertIn("这一组没有", reason)
            panel.set_mapping_reason(reason)
            self.assertFalse(panel.btn_kmap.isEnabled())
            self.assertIn(reason, panel.btn_kmap.text())
        doc.groups[0].kind = "dialogue"
        self.assertEqual(lm.disabled_reason(doc, doc.groups[0], {}), "（不是歌曲）")

    def test_song_scope_suggestions_edit_auto_play_help_and_partial_apply(self):
        doc, ed = document()
        group = copy.deepcopy(doc.groups[0])
        group.id, group.members = "g2", ["jp2", "cn2"]
        doc.groups.append(group)
        for m in list(doc.members.values()):
            new = copy.deepcopy(m)
            new.id, new.group, new.start, new.end = m.id + "2", "g2", 20000, 23000
            new.runs[0].ins += "2"
            doc.members[new.id] = new
        doc.structure_changed()
        dialog = LyricsMapDialog(doc, "g2", {"original": "ja", "translation": "zh-Hant"})
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog.windowFlags() & Qt.WindowType.WindowType_Mask, Qt.WindowType.Tool)
        self.assertEqual(dialog.table.rowCount(), 2)
        self.assertEqual(dialog.table.currentRow(), 1)
        self.assertEqual(dialog.suggestions, {0, 1})
        first = dialog.table.item(0, 2).text()
        dialog.auto()
        self.assertNotIn(1, dialog.suggestions)
        self.assertEqual(dialog.table.item(0, 2).text(), first)
        played = []
        dialog.playRequested.connect(lambda a, b: played.append((a, b)))
        dialog.play()
        self.assertEqual(played, [(20000, 23000)])
        dialog.table.item(1, 2).setText("坏（")
        self.assertEqual(dialog.table.item(1, 3).text(), "括号不成对")
        # A grey (unadopted) suggestion is never applied: it may replace finer existing timing.
        self.assertEqual(dialog.table.item(0, 3).text(), "✓ 建议，没采用（应用时不改）")
        self.assertEqual(dialog.mappings(), {})
        self.assertFalse(dialog.apply_button.isEnabled())
        dialog.table.setCurrentCell(0, 2)
        dialog.adopt()
        self.assertEqual(dialog.table.item(0, 3).text(), "✓")
        self.assertTrue(dialog.apply_button.isEnabled())
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No) as question:
            dialog.accept()
            self.assertIn("1 行还没对上，这些行不改。继续吗？", question.call_args.args)
            self.assertEqual(dialog.result(), 0)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            dialog.accept()
        self.assertEqual(set(dialog.mappings()), {"g"})
        ed.apply_lyrics_mapping(dialog.mappings())
        ed.history.undo()
        self.assertEqual(doc.karaoke, {})
        with patch.object(app_help, "open_manual", return_value=True) as opener:
            from PySide6.QtWidgets import QPushButton
            next(b for b in dialog.findChildren(QPushButton) if b.text() == "帮助").click()
            opener.assert_called_once_with("karaoke")

    def test_whole_note_setting_controls_initial_and_repeated_proposals(self):
        for follow in (True, False):
            with self.subTest(follow_notes=follow):
                doc, _ = document("てこう", "一二三", sung=[("て", 300), ("こ", 300), ("う", 300)])
                dialog = LyricsMapDialog(doc, "g", {"original": "ja", "translation": "zh-Hant"},
                                         follow_notes=follow)
                self.addCleanup(dialog.deleteLater)
                expected = "一（て）二三（こう）" if follow else "一（て）二（こ）三（う）"
                self.assertEqual(dialog.table.item(0, 2).text(), expected)
                dialog.table.item(0, 2).setText("一二三（てこう）")
                dialog.auto_button.click()
                self.assertEqual(dialog.table.item(0, 2).text(), expected)

    def test_existing_mapping_shown_black_and_original_tapping_target(self):
        doc, ed = document()
        ed.apply_lyrics_mapping({"g": "你好（かな）世界（よ）"})
        dialog = LyricsMapDialog(doc, "g", {"original": "ja", "translation": "zh-Hant"})
        self.addCleanup(dialog.deleteLater)
        self.assertFalse(dialog.suggestions)
        self.assertEqual(dialog.table.item(0, 2).text(), "你好（かな）世界（よ）")
        self.assertEqual(dialog.mappings(), {})                 # unchanged rows keep their timing
        dialog.table.item(0, 2).setText("你（かな）好世界（よ）")
        self.assertEqual(dialog.mappings(), {"g": "你（かな）好世界（よ）"})
        from types import SimpleNamespace
        from subkit_app.mainwindow import MainWindow
        fake = SimpleNamespace(session=SimpleNamespace(doc=doc), _k_unit_pref="syllable")
        fake._k_member = lambda g, lane: lm.member(doc, g, lane)
        self.assertEqual(MainWindow._k_line_at(fake, 2000)[:2], ("jp", "original"))
        doc.members["jp"].runs = []
        self.assertEqual(MainWindow._k_line_at(fake, 2000)[:2], ("cn", "translation"))
