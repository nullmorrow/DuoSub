"""文件 → 导出整季发布用 ASS: names, conflicts, the dialog and the real write (offscreen Qt, no ffmpeg)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import test_convert_ui as fixtures
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDialog, QMessageBox
from subkit.core.fontcheck import FontCheck
from subkit.core.season_export import SeasonEntry, check_suffix, plan, season_entries, target_name
from subkit.core.workspace import Project
from subkit.model.document import Document
from subkit_app.views.season_export import SeasonExportDialog, SeasonReportDialog

VIDEO = "S01E001 - [Suzu-Kaze&LoliHouse] Sword Oratoria - 01 [BDRip 1080p].mkv"


class Names(unittest.TestCase):
    def test_two_copy_names_and_conflicts(self):
        from subkit.core.season_export import script_targets
        for source, stem in (("第05集.zh-Hant.ass", "第05集"), ("第05集.ass", "第05集"),
                             ("show.cht.ass", "show"), ("show.1080p.ass", "show.1080p")):
            self.assertEqual([p.name for _, p in script_targets(Path(source))],
                             [stem + ".zh-Hans.ass", stem + ".zh-Hant.ass"])
        self.assertEqual(script_targets(Path("show.zh-Hant.v1.ass"))[0][1].name, "show.zh-Hans.v1.ass")
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / "a.zh-Hant.ass"
            source.write_text("source", encoding="utf-8")
            entries = [SeasonEntry("E1", Path("a.mkv"), True, (source,)),
                       SeasonEntry("E2", Path("A.mkv"), True)]
            items = plan(entries, folder, "zh-Hans", traditional_suffix="zh-Hant")
            self.assertEqual([i.direction for i in items], ["simplified", "traditional"] * 2)
            self.assertIsNone(items[1].target)
            self.assertIn("原字幕", items[1].note)
            self.assertEqual(items[2].duplicate_of, "E1")
            self.assertTrue(items[3].exists)
            for suffix in ("zh-Hans", "ZH-HANS", " .zh-Hans. "):
                with self.assertRaisesRegex(ValueError, "不能相同"):
                    plan(entries, folder, "zh-Hans", traditional_suffix=suffix)

    def test_suffix_and_names(self):
        self.assertEqual(check_suffix(" .zh-Hant. "), ("zh-Hant", ""))
        self.assertIn("/", check_suffix("zh/Hant")[1])
        self.assertTrue(check_suffix("zh Hant")[1])
        self.assertTrue(check_suffix("")[1])
        self.assertEqual(target_name("S01E001", Path("E:/v") / VIDEO, "zh-Hant"), VIDEO[:-4] + ".zh-Hant.ass")
        self.assertEqual(target_name("S01E002", None, "zh-Hant"), "S01E002.zh-Hant.ass")

    def test_plan_marks_existing_duplicates_unmerged_and_source_files(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "a.zh-Hant.ass").write_text("old", encoding="utf-8")
            (folder / "c.cht.ass").write_text("fansub", encoding="utf-8")
            entries = [SeasonEntry("E1", Path("v/a.mkv"), True), SeasonEntry("E2", Path("w/A.mkv"), True),
                       SeasonEntry("E3", None, False), SeasonEntry("E4", None, True)]
            items = plan(entries, folder, "zh-Hant")
            self.assertEqual([(i.eid, i.target.name if i.target else None, i.exists, i.duplicate_of, i.note) for i in items],
                             [("E1", "a.zh-Hant.ass", True, "", ""), ("E2", "A.zh-Hant.ass", True, "E1", ""),
                              ("E3", None, False, "", "还没有合并，跳过"), ("E4", "E4.zh-Hant.ass", False, "", "没有视频，用集名")])
            source = SeasonEntry("E5", Path("v/c.mkv"), True, (folder / "c.cht.ass",))
            item = plan([source], folder, "cht")[0]
            self.assertIsNone(item.target)
            self.assertIn("原字幕，不能覆盖", item.note)


class SeasonExportUI(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = fixtures.ConversionUI._dispose

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.root = Path(self.tmp.name)
        self.videos = self.root / "videos"
        self.videos.mkdir()
        (self.videos / VIDEO).write_bytes(b"")
        self.project = Project.create(self.root / "project", "剑姬", translation="zh-Hant")
        for eid, video, merged in (("S01E001", self.videos / VIDEO, True), ("S01E002", None, True),
                                   ("S01E003", None, False)):
            ep = self.project.add_episode(eid)
            if video:
                ep.set_video(video)
                ep.save_meta()
            if merged:
                ep.document = Document(episode=eid)
                ep.save()
        self.out = self.root / "out"
        self.out.mkdir()
        self.settings = QSettings(str(self.root / "season.ini"), QSettings.Format.IniFormat)

    def dialog(self):
        d = SeasonExportDialog(self.project, self.settings, None)
        self.addCleanup(d.deleteLater)
        return d

    def test_two_copy_dialog_suffixes_skip_and_remember(self):
        d = self.dialog()
        d.folder.setText(str(self.out))
        d.zh_both.setChecked(True)
        self.assertTrue(d.suffix.isHidden())
        self.assertFalse(d.simplified_suffix.isHidden())
        self.assertEqual(len(d.chosen()), 4)
        self.assertEqual([x[2] for x in d.chosen()], ["simplified", "traditional"] * 2)
        d.traditional_suffix.setText("ZH-HANS")
        self.assertFalse(d.export_button.isEnabled())
        self.assertIn("不能相同", d.reason.text())
        d.traditional_suffix.setText("cht")
        existing = self.out / "S01E002.cht.ass"
        existing.write_text("keep", encoding="utf-8")
        with patch.object(QMessageBox, "exec", lambda box: next(
                b for b in box.buttons() if b.text() == "跳过这些文件").click()):
            d.accept()
        self.assertEqual(len(d.chosen()), 3)
        self.assertEqual(existing.read_text(encoding="utf-8"), "keep")
        reopened = self.dialog()
        self.assertTrue(reopened.zh_both.isChecked())
        self.assertEqual(reopened.traditional_suffix.text(), "cht")
        self.assertTrue(any("S01E002.cht.ass" in reason for _, reason in d.skipped()))

    def test_two_copy_season_cancel_preserves_existing_and_writes_nothing(self):
        self.window.project, self.window.session = self.project, None
        existing = self.out / "S01E002.zh-Hant.ass"
        existing.write_bytes(b"original")
        def choose(dialog):
            dialog.folder.setText(str(self.out))
            dialog.zh_both.setChecked(True)
            with patch.object(QMessageBox, "exec", lambda box: next(
                    b for b in box.buttons() if b.text() == "覆盖").click()):
                dialog.accept()
            return dialog.result()
        def cancel(ass, runner=None):
            self.window._export_job.cancel()
            return FontCheck()
        with patch.object(SeasonExportDialog, "exec", choose), \
                patch("subkit_app.mainwindow.check_fonts", cancel), patch.object(SeasonReportDialog, "exec") as shown:
            self.window.export_season()
        shown.assert_not_called()
        self.assertEqual(list(self.out.iterdir()), [existing])
        self.assertEqual(existing.read_bytes(), b"original")
        self.assertIsNone(self.window._export_job)

    def test_two_copy_season_writes_scripts_and_report_sections(self):
        from tests.test_pinyin_export import document
        for eid in ("S01E001", "S01E002"):
            ep = self.project.episode(eid)
            ep.document = document("这里是軟體")
            ep.save()
        self.window.project, self.window.session = self.project, None
        def choose(dialog):
            dialog.folder.setText(str(self.out))
            dialog.zh_both.setChecked(True)
            dialog.accept()
            return dialog.result()
        with patch.object(SeasonExportDialog, "exec", choose), patch.object(SeasonReportDialog, "exec", return_value=1), \
                patch("subkit_app.mainwindow.check_fonts", lambda ass, runner=None: FontCheck()):
            self.window.export_season()
        for name in (VIDEO[:-4], "S01E002"):
            self.assertIn("这里是软体", (self.out / (name + ".zh-Hans.ass")).read_text(encoding="utf-8-sig"))
            self.assertIn("這裡是軟體", (self.out / (name + ".zh-Hant.ass")).read_text(encoding="utf-8-sig"))
        report = (self.out / "project.zh-Hans+zh-Hant.export-report.txt").read_text(encoding="utf-8")
        self.assertIn("已导出 2 集，共 4 个文件", report)
        self.assertEqual(report.count("【S01E"), 4)

    def test_dialog_defaults_names_and_overwrite_choice(self):
        d = self.dialog()
        self.assertEqual(d.folder.text(), str(self.videos))                        # next to the videos
        self.assertEqual(d.suffix.currentText(), "zh-Hant")
        self.assertEqual([d.table.item(r, 1).text() for r in range(3)],
                         [VIDEO[:-4] + ".zh-Hant.ass", "S01E002.zh-Hant.ass", ""])
        self.assertEqual([d.table.item(r, 2).text() for r in range(3)],
                         ["将导出", "没有视频，用集名；将导出", "还没有合并，跳过"])
        d.suffix.setCurrentText("zh/Hant")
        self.assertFalse(d.export_button.isEnabled())
        self.assertIn("不能有", d.reason.text())
        d.suffix.setCurrentText("cht")
        d.folder.setText(str(self.out))
        (self.out / "S01E002.cht.ass").write_text("old", encoding="utf-8")
        d.refresh()
        self.assertEqual(d.table.item(1, 2).text(), "没有视频，用集名；已存在（导出时询问）")

        def answer(text):
            def run(box):
                next(b for b in box.buttons() if b.text() == text).click()
            return run

        with patch.object(QMessageBox, "exec", answer("跳过这些集")):
            d.accept()
        self.assertEqual(d.result(), QDialog.DialogCode.Accepted)
        self.assertEqual([eid for eid, _ in d.chosen()], ["S01E001"])
        self.assertEqual(d.skipped(), [("S01E002", "已存在，按你的选择跳过"), ("S01E003", "还没有合并，跳过")])
        self.assertEqual(self.settings.value("export/season_suffix"), "cht")
        d2 = self.dialog()
        self.assertEqual((d2.folder.text(), d2.suffix.currentText()), (str(self.out), "cht"))   # remembered
        with patch.object(QMessageBox, "exec", answer("覆盖")):
            d2.accept()
        self.assertEqual([eid for eid, _ in d2.chosen()], ["S01E001", "S01E002"])

    def test_export_season_writes_every_episode_and_one_report(self):
        w = self.window
        w.project, w.session = self.project, None
        (self.out / "S01E002.zh-Hant.ass").write_text("old", encoding="utf-8")
        shown = {}

        def choose(dialog):
            dialog.folder.setText(str(self.out))
            with patch.object(QMessageBox, "exec", lambda box: next(
                    b for b in box.buttons() if b.text() == "覆盖").click()):
                dialog.accept()
            return dialog.result()

        def report(dialog):
            shown["text"] = dialog.text.toPlainText()
            return QDialog.DialogCode.Accepted

        with patch.object(SeasonExportDialog, "exec", choose), patch.object(SeasonReportDialog, "exec", report), \
                patch("subkit_app.mainwindow.check_fonts", lambda ass, runner=None: FontCheck()):
            w.export_season()
        first = self.out / (VIDEO[:-4] + ".zh-Hant.ass")
        self.assertIn("[Events]", first.read_text(encoding="utf-8-sig"))
        self.assertIn("[Events]", (self.out / "S01E002.zh-Hant.ass").read_text(encoding="utf-8-sig"))
        self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                         sorted([first.name, "S01E002.zh-Hant.ass", "project.zh-Hant.export-report.txt"]))
        self.assertIn("已导出 2 集", shown["text"])
        self.assertIn("跳过：S01E003（还没有合并，跳过）", shown["text"])
        self.assertIn("【S01E001】" + first.name, shown["text"])
        self.assertEqual(season_entries(self.project)[0].video, self.videos / VIDEO)


if __name__ == "__main__":
    unittest.main()
