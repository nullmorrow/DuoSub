"""Host Qt offscreen import flow; copied files and cancellable worker pipelines."""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hashlib
import html
import re
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QMainWindow
from subkit.core import oped_import
from subkit.core.passages import PassageStore
from subkit.core.workspace import Project
from subkit_app.views import passages as ui
from tests import test_passages_ui as fixture

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = "# 歌名：试歌\n# 类别：片头曲\n时间：0:00:01.00 - 0:00:05.00\n原文：光\n读音：光（ひかり）\n假名拍：{\\kf100}ひ{\\kf100}か{\\kf100}り\n翻译：光\n对应：光（ひかり）\n备注：检查特殊读法\n"

class ImportUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="l3c-ui-")
        self.addCleanup(self.tmp.cleanup)
        self.project = Project.create(Path(self.tmp.name) / "project", "导入测试")
        self.store = PassageStore(self.project)
        self.window = QMainWindow()
        self.dialog = ui.OPEDImportDialog(self.store, self.window)
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.dialog.deleteLater)

    def wait(self):
        end = time.monotonic() + 15
        while self.dialog._job is not None and time.monotonic() < end:
            QTest.qWait(10)
        self.assertIsNone(self.dialog._job)
        self.assertNotIn("未完成", self.dialog.progress.text())

    def test_paste_errors_notes_prompt_and_tool_windows(self):
        paste = ui.PasteLyricsDialog(self.dialog)
        self.addCleanup(paste.deleteLater)
        self.assertEqual(paste.windowType(), Qt.WindowType.Tool)
        self.assertEqual(self.dialog.windowType(), Qt.WindowType.Tool)
        paste.copy_prompt()
        prompt = QApplication.clipboard().text()
        manual = (ROOT / "docs/manual/index.html").read_text(encoding="utf-8")
        manual_prompt = html.unescape(re.search(r'<pre id="lyrics-prompt">(.*?)</pre>', manual, re.S).group(1))
        design = (ROOT / "docs/LYRICS.md").read_text(encoding="utf-8").split('### 4.5')[1].split('```')[1].strip('\n')
        self.assertEqual(prompt, manual_prompt)
        self.assertEqual(prompt, design)
        self.dialog.load_text(SAMPLE + "\n原文：坏块\n")
        self.wait()
        self.assertEqual(len(self.dialog.rows), 1)
        self.assertIn("检查特殊读法", self.dialog.table.item(0, 1).toolTip())
        self.assertIn("歌词块 第 2 块", self.dialog.error_list.item(0).text())
        self.assertEqual(self.dialog.table.item(0, 9).text(), "没有")
        self.dialog.start_import()
        self.wait()
        self.assertEqual(len(self.store.list()), 1)
        self.assertIn("已导入 1 首：试歌", self.window.statusBar().currentMessage())

    def test_real_file_copy_and_source_hash(self):
        source = Path('./sample-media/OPED.txt')
        if not source.exists():
            self.skipTest("真实 OPED.txt 不存在")
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        copied = Path(self.tmp.name) / "OPED.txt"
        shutil.copyfile(source, copied)
        self.dialog.load_files([copied])
        self.wait()
        self.assertEqual(len(self.dialog.rows), 3)
        # One style family for three songs: default names come from each song's first line.
        names = [self.dialog.table.item(row, 1).text() for row in range(3)]
        self.assertEqual(len(set(names)), 3)
        self.assertTrue(names[0].startswith("潔く"))
        self.assertEqual(names[1], "絶対運命黙示録")
        self.assertTrue(names[2].startswith("Missing truth"))
        categories = [self.dialog.table.cellWidget(row, 2).currentData() for row in range(3)]
        self.assertEqual(categories, ["OP", "IN", "ED"])
        self.dialog.start_import()
        self.wait()
        self.assertEqual(len(self.store.list()), 3)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
        self.assertEqual(source.read_bytes(), copied.read_bytes())

    def test_conflict_default_skip_replace_new_and_edit(self):
        song = oped_import.parse_text(SAMPLE).songs[0]
        passage = oped_import.build_passage(song, "试歌", "OP", "")
        self.store.add(passage, None)
        self.dialog.load_text(SAMPLE)
        self.wait()
        choice = self.dialog.table.cellWidget(0, 0)
        self.assertIsInstance(choice, QComboBox)
        self.assertEqual(choice.currentData(), "skip")
        self.dialog.start_import()
        self.wait()
        self.assertEqual(len(self.store.list()), 1)
        for option, count in (("replace", 1), ("new", 2)):
            choice.setCurrentIndex(choice.findData(option))
            self.dialog.start_import()
            self.wait()
            self.assertEqual(len(self.store.list()), count)
        self.dialog.table.item(0, 1).setText("另一首")
        self.assertNotIsInstance(self.dialog.table.cellWidget(0, 0), QComboBox)

    def test_background_cancel_publishes_no_rows_or_store_writes(self):
        def slow(source):
            time.sleep(.12)
            return oped_import.ImportResult()
        with patch.object(oped_import, "parse_text", side_effect=slow):
            self.dialog.load_text(SAMPLE)
            self.assertIsNotNone(self.dialog._job)
            job = self.dialog._job
            self.dialog.reject()
            end = time.monotonic() + 3
            while self.dialog._job is not None and time.monotonic() < end:
                QTest.qWait(10)
            self.assertTrue(job.finished.is_set())
        self.assertEqual(self.store.list(), [])
        self.assertEqual(self.dialog.rows, [])
        self.assertEqual(self.dialog.result(), QDialog.DialogCode.Rejected)

    def test_audio_options_and_help_anchors(self):
        self.assertEqual(self.dialog.help_anchor, "oped-import")
        self.assertEqual(ui.PasteLyricsDialog.help_anchor, "lyrics-block")
        self.dialog.audio.setCurrentIndex(1)
        self.dialog.load_text(SAMPLE)
        self.assertIn("先打开", self.dialog.progress.text())
        self.assertIsNone(self.dialog._job)
        self.assertEqual(self.dialog.table.columnCount(), 10)

class ManualApplyUI(fixture.PassageUI):
    # Reuse the established copied-project setup without rerunning its test methods.
    def test_no_audio_anchor_unscaled_and_undo(self):
        song = oped_import.parse_text(SAMPLE).songs[0]
        p = oped_import.build_passage(song, "试歌", "OP", "")
        self.store.add(p, None)
        dialog = ui.PassagesDialog(self.session, [(p, None)], self.window.player, self.window)
        self.addCleanup(dialog.deleteLater)
        self.window.current_gid = None
        dialog._replan()
        self.assertIn("没有音频，只能手动套用", dialog.list.item(0).text())
        self.assertFalse(dialog.manual_button.isEnabled())
        self.assertIn("先在列表里选中", dialog.manual_button.toolTip())
        self.window.current_gid = "g_1"
        dialog._replan()
        self.assertTrue(dialog.manual_button.isEnabled())
        dialog.manual_apply()
        self.assertEqual(dialog.current_plan.at_ms, 1013)
        self.assertEqual(dialog.current_plan.scale, 1)
        self.assertFalse(dialog.scale.isChecked())
        self.assertTrue(dialog.current_plan.lines)

# Keep inherited helper setup, but discover only this batch's manual-apply test.
for _name in dir(fixture.PassageUI):
    if _name.startswith("test_"):
        setattr(ManualApplyUI, _name, None)
