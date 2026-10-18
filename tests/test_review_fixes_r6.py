"""R6 wording and form checks with Qt offscreen and deterministic timing maps."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel
from subkit.autotime import Segment, TimingMap
from subkit.model.validate import Report
from subkit_app.views.autotime import segment_text, segment_shifts, AutoTimeDetailsDialog
from subkit_app.views.dialogs import AddEpisodeDialog, MergeDialog
from subkit_app.views.check_result import CheckResultDialog, issue_lines, reanalyze_text
from tests.test_autotime_ui import AutoTimeUI


class Wording(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_shift_drift_boundary_and_rounding(self):
        first = Segment(0, 10000, -210, 1.01, 'medium')
        mapping = TimingMap([first, Segment(10000, 20000, 500, confidence='high')])
        self.assertEqual(segment_shifts(first), (-210, -110))
        self.assertEqual(mapping.map(10000) - 10000, 500)
        self.assertIn('开头提前 0.21 秒，结尾提前 0.11 秒（把握一般）', segment_text(first))
        self.assertIn('整段推后 0.50 秒（把握大）', segment_text(mapping.segments[1]))
        self.assertIn('把握小，没改', segment_text(Segment(0, 1000, -200)))
        self.assertIn('整段', segment_text(Segment(0, 10000, 1.5, 1.0049, 'high')))
        self.assertIn('开头', segment_text(Segment(0, 10000, 0, 1.005, 'high')))

    def test_autofill_missing_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TEMP']) as folder:
            root = Path(folder)
            for name in ['episode.mkv', 'episode.zh-Hant.ass', 'episode.jp.srt', 'episode.en.srt']:
                (root / name).write_text('', encoding='utf-8')
            dialog = AddEpisodeDialog(suggested_id='')
            try:
                self.assertEqual(dialog.missing_hint.text(), '还缺：集编号、翻译字幕、原声字幕')
                self.assertFalse(dialog.ok_button.isEnabled())
                self.assertFalse(dialog.audio.isEnabled())
                with patch('subkit_app.views.dialogs.probe_audio_tracks', return_value=[]):
                    dialog.video.setText(str(root / 'episode.mkv'))
                self.assertEqual(Path(dialog.cn.text()).name, 'episode.zh-Hant.ass')
                self.assertEqual(Path(dialog.jp.text()).name, 'episode.jp.srt')
                self.assertFalse(dialog.found_notes['original'].isHidden())
                self.assertEqual(dialog.missing_hint.text(), '还缺：集编号')
                dialog.eid.setText('S01E017')
                self.assertTrue(dialog.ok_button.isEnabled())
                dialog.jp.setText(str(root / 'episode.en.srt'))
                dialog._fill_subtitles(root / 'episode.mkv')
                self.assertEqual(Path(dialog.jp.text()).name, 'episode.en.srt')
                dialog.cn.setText(str(root))
                self.assertFalse(dialog.ok_button.isEnabled())
                self.assertEqual(dialog.windowTitle(), '添加一集')
            finally:
                dialog.deleteLater()

    def test_language_aliases_and_ambiguous_candidates(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TEMP']) as folder:
            root = Path(folder)
            for suffix, code in [('chs','zh-Hans'), ('cht','zh-Hant'), ('ja','ja'), ('jp','ja'), ('en','en'), ('zh-Hans','zh-Hans')]:
                path = root / ('video.' + suffix + '.srt')
                path.write_text('', encoding='utf-8')
                dialog = AddEpisodeDialog(languages={'original':code, 'translation':'ko'})
                try:
                    dialog._fill_subtitles(root / 'video.mkv')
                    self.assertEqual(Path(dialog.jp.text()).name, path.name)
                    self.assertEqual(dialog.cn.text(), '')
                finally:
                    dialog.deleteLater()
                    path.unlink()

    def test_plain_language_name_preferred_over_working_copies(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TEMP']) as folder:
            root = Path(folder)
            for name in ['ep.zh-Hant.ass', 'ep.zh-Hant.v6_4.ass', 'ep.jp.srt']:
                (root / name).write_text('', encoding='utf-8')
            dialog = AddEpisodeDialog(languages={'original': 'ja', 'translation': 'zh-Hant'})
            try:
                dialog._fill_subtitles(root / 'ep.mkv')
                self.assertEqual(Path(dialog.cn.text()).name, 'ep.zh-Hant.ass')
                self.assertEqual(Path(dialog.jp.text()).name, 'ep.jp.srt')
            finally:
                dialog.deleteLater()

    def test_unmarked_song_part_styles_are_translation_lyrics(self):
        from subkit.ass.document import AssDocument
        from subkit.sources.cn import classify
        lines = ['[Script Info]', 'PlayResX: 640', 'PlayResY: 480', '', '[V4+ Styles]',
                 'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding']
        styles = ['Default', 'IN', 'ED', 'OP2', 'OPCH', 'Info', 'Edit']
        lines += [f'Style: {n},Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,1,2,0,0,30,1' for n in styles]
        lines += ['', '[Events]', 'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text']
        for i, n in enumerate(styles * 3):
            lines.append(f'Dialogue: 0,0:00:{i:02d}.00,0:00:{i:02d}.50,{n},,0,0,0,,文字{i}')
        roles = classify(AssDocument.parse(chr(10).join(lines) + chr(10)), {}).style_roles
        for name in ('IN', 'ED', 'OP2', 'OPCH'):
            self.assertEqual(roles[name], 'song_cn', name)
        self.assertEqual(roles['Default'], 'dialogue')
        self.assertEqual(roles['Info'], 'keep')
        self.assertEqual(roles['Edit'], 'keep')

    def test_merge_result_and_categories_hide_internal_ids(self):
        source = SimpleNamespace(lane='original', events=[SimpleNamespace(text='你好', start=1200)])
        sources = {'src_secret':source}
        doc = SimpleNamespace(groups=[], members={}, comments={}, annotations={})
        rep = Report(ledger={'src_secret':{'paired':1,'missing':1,'duplicate':0}})
        for code in ['order', 'ledger.missing', 'ledger.duplicate', 'ref.member']:
            rep.add('error', code, 'raw src_secret g_secret', source='src_secret', event=0)
        episode = SimpleNamespace(document=doc, sources=sources, project=SimpleNamespace(languages={'original':'en','translation':'ja'}),
                                  data={'merge':{'summary':{'offset_ms':-200,'original_units':1,'translation_rows':2,'cuts':[(1200,500)]}}})
        dialog = CheckResultDialog(episode, rep, merged=True)
        try:
            text = dialog.text.toPlainText()
            self.assertIn('这一集共 0 组：', text)
            self.assertIn('只有原声（英语） 0 组', text)
            self.assertIn('原声（英语）字幕 2 字', dialog.details_text.toPlainText())
            self.assertIn('整体差 -0.20 秒', text)
            self.assertIn('两个版本剪辑不同', text)
            self.assertNotIn('src_secret', text)
            self.assertNotIn('已删除 0', text)
            self.assertTrue(any('有 1 个字没有着落' in l.text() for l in dialog.findChildren(QLabel)))
            lines = '\n'.join(issue_lines(rep,doc,sources))
            for category in ['顺序','没有着落','重复','内部数据（请把检查结果发给开发者）']:
                self.assertIn(category,lines)
            self.assertNotIn('src_secret',lines)
            self.assertIn('已重新分析原声（英语）字幕', reanalyze_text({'src_secret':(1,2)}, sources,episode.project.languages))
        finally:
            dialog.deleteLater()

    def test_merge_settings_unavailable_reason(self):
        dialog = MergeDialog(semantic_available=False)
        try:
            self.assertIn('本机没有安装，用不了', dialog.semantic.text())
            self.assertFalse(dialog.semantic.isEnabled())
            self.assertIn('全部显示（推荐）', dialog.keep.text())
        finally:
            dialog.deleteLater()


class GroupNumbers(AutoTimeUI):
    def test_details_group_numbers_preserve_audition_identity(self):
        with patch('subkit.autotime.estimate', return_value=self.map):
            dialog = AutoTimeDetailsDialog(self.session, self.window.wave.audio_features(), self.window.player, self.window.settings, self.window)
            self.wait_plan(dialog)
        try:
            self.assertEqual(dialog.groups.item(0,0).text(), '1')
            self.assertEqual(dialog.groups.item(0,0).data(Qt.ItemDataRole.UserRole), self.doc.groups[0].id)
            self.assertEqual(dialog.segments.horizontalHeaderItem(3).text(), '结尾多挪（秒）')
            dialog.groups.setCurrentCell(0,0)
            with patch.object(dialog.player, 'play') as play, patch.object(dialog.player, 'seek') as seek:
                dialog.audition()
                play.assert_called_once()
                seek.assert_called_once()
        finally:
            dialog.deleteLater()
