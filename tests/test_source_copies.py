"""The project's copies of imported subtitles: missing / changed copies are found and only identical bytes go back."""
from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import test_convert_ui as fixtures
from PySide6.QtWidgets import QFileDialog, QMessageBox
from subkit.core.importer import SourceChanged, copy_state, restore_copy, sha256_bytes
from subkit.core.source_copies import find_in_folder, fix_from_original, scan_project
from subkit.core.workspace import Project
from subkit_app.views.source_copies import SourceCopiesDialog

SRT = "1\n00:00:01,000 --> 00:00:02,000\n{}\n"


def writable(path: Path) -> None:
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)


class Copies(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="subkit_copies_")
        self.addCleanup(self.cleanup)
        self.root = Path(self.tmp.name)
        self.originals = self.root / "originals"
        self.originals.mkdir()
        self.project = Project.create(self.root / "project", "副本")
        self.ep = self.project.add_episode("S01E001")
        for lane, text in (("translation", "你好"), ("original", "こんにちは")):
            path = self.originals / f"ep01.{lane}.srt"
            path.write_text(SRT.format(text), encoding="utf-8")
            self.ep.add_source(path, lang="zh-Hant" if lane == "translation" else "ja", lane=lane)
        self.copies = sorted((self.ep.dir / "sources").glob("*.srt"))

    def cleanup(self):
        for p in Path(self.tmp.name).rglob("*"):
            if p.is_file():
                writable(p)
        self.tmp.cleanup()

    def test_scan_and_restore_only_identical_bytes(self):
        self.assertEqual(scan_project(self.project), [])
        missing, changed = self.copies
        writable(missing)
        missing.unlink()
        writable(changed)
        changed.write_text("被改过", encoding="utf-8")
        problems = sorted(scan_project(self.project), key=lambda p: p.copy.name)
        self.assertEqual([(p.eid, p.copy.name, p.state) for p in problems],
                         [("S01E001", missing.name, "missing"), ("S01E001", changed.name, "changed")])
        wrong = self.root / "wrong.srt"
        wrong.write_text("不一样", encoding="utf-8")
        with self.assertRaises(SourceChanged):
            restore_copy(problems[0].copy, problems[0].sha, wrong)
        self.assertFalse(missing.exists())
        self.assertTrue(all(fix_from_original(p) for p in problems))      # originals still where imported
        self.assertEqual([copy_state(p.copy, p.sha) for p in problems], ["ok", "ok"])
        kept = list((self.ep.dir / "sources").glob(changed.name + ".changed-*"))
        self.assertEqual([k.read_text(encoding="utf-8") for k in kept], ["被改过"])   # never deleted
        self.assertFalse(os.access(missing, os.W_OK))                                # read-only again
        self.assertEqual(scan_project(self.project), [])

    def test_find_renamed_file_in_a_folder_by_content(self):
        moved = self.root / "archive" / "sub"
        moved.mkdir(parents=True)
        for path in self.originals.iterdir():
            path.rename(moved / ("改名-" + path.name))
        (moved / "其他.srt").write_text(SRT.format("别的"), encoding="utf-8")
        for copy in self.copies:
            writable(copy)
            copy.unlink()
        problems = scan_project(self.project)
        self.assertFalse(any(fix_from_original(p) for p in problems))
        found = find_in_folder(self.root / "archive", problems)
        self.assertEqual(sorted(path.name for path in found.values()),
                         ["改名-ep01.original.srt", "改名-ep01.translation.srt"])
        for i, path in found.items():
            self.assertEqual(sha256_bytes(path.read_bytes()), problems[i].sha)


class CopiesDialog(Copies):
    @classmethod
    def setUpClass(cls):
        fixtures.ConversionUI.setUpClass()

    def test_dialog_restores_from_originals_then_folder_and_reports(self):
        for copy in self.copies:
            writable(copy)
            copy.unlink()
        moved = self.originals / "ep01.original.srt"
        archive = self.root / "archive"
        archive.mkdir()
        moved.rename(archive / "renamed.srt")
        dialog = SourceCopiesDialog(self.project)
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog.table.rowCount(), 2)
        self.assertIn("还有 2 份", dialog.message.text())
        fixed = []
        dialog.fixed.connect(fixed.append)
        dialog.fix_from_originals()
        self.assertIn("放回 1 份", dialog.message.text())
        with patch.object(QFileDialog, "getExistingDirectory", return_value=str(archive)):
            dialog.search_folder()
        self.assertEqual(fixed, [["S01E001"], ["S01E001"]])
        self.assertEqual(dialog.message.text().splitlines()[0], "全部副本都已放回。")
        self.assertFalse(dialog.folder_button.isEnabled())
        self.assertEqual(scan_project(self.project), [])

    def test_choose_file_refuses_different_bytes(self):
        writable(self.copies[0])
        self.copies[0].unlink()
        dialog = SourceCopiesDialog(self.project)
        self.addCleanup(dialog.deleteLater)
        dialog.table.setCurrentCell(0, 0)
        other = self.root / "other.srt"
        other.write_text("不同", encoding="utf-8")
        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(other), "")), \
                patch.object(QMessageBox, "warning") as warned:
            dialog.choose_file()
        self.assertIn("SHA-256 不一致", warned.call_args.args[2])
        self.assertEqual(len(dialog.open_problems()), 1)


if __name__ == "__main__":
    unittest.main()
