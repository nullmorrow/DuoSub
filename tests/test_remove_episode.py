"""Episode archival on temporary projects; offscreen dialogs and player doubles."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests import test_help as help_tests
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox, QPushButton
from subkit.core.workspace import Project
from subkit.model.document import Document
from subkit_app import help as help_ui
from subkit_app.session import EpisodeSession


class RemoveEpisodeCore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="duosub_remove_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = Project.create(self.root / "project", "临时工程")
        self.original = self.root / "original.ass"
        self.original.write_bytes(b"original source bytes")
        for eid in ("A", "B", "C"):
            ep = self.project.add_episode(eid)
            (ep.dir / "work.bin").write_bytes(eid.encode())
            ep.data["media"]["video"] = {"path": str(self.original)}
            ep.save_meta()

    def test_moves_only_target_and_persists_list(self):
        others = {eid: (self.project.episode(eid).dir / "episode.json").read_bytes() for eid in ("A", "C")}
        dst = self.project.remove_episode("B")
        self.assertEqual(dst.parent, self.project.root / "已移除")
        self.assertTrue(dst.name.startswith("B-"))
        self.assertEqual((dst / "work.bin").read_bytes(), b"B")
        self.assertFalse((self.project.root / "episodes/B").exists())
        self.assertEqual(Project.open(self.project.root).episode_ids(), ["A", "C"])
        self.assertEqual(self.original.read_bytes(), b"original source bytes")
        for eid, data in others.items():
            self.assertEqual((self.project.episode(eid).dir / "episode.json").read_bytes(), data)

    def test_same_second_repeated_removal_has_unique_destination(self):
        with patch("subkit.core.workspace.time.strftime", return_value="20261006-000000"):
            first = self.project.remove_episode("B")
            self.project.add_episode("B")
            second = self.project.remove_episode("B")
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_dir() and second.is_dir())

    def test_save_failure_restores_folder_metadata_and_list(self):
        previous = copy.deepcopy(self.project.data)
        disk = (self.project.root / "subkit-project.json").read_bytes()
        with patch.object(self.project, "save", side_effect=OSError("test save failure")):
            with self.assertRaisesRegex(OSError, "test save failure"):
                self.project.remove_episode("B")
        self.assertEqual(self.project.data, previous)
        self.assertEqual((self.project.root / "subkit-project.json").read_bytes(), disk)
        self.assertEqual((self.project.episode("B").dir / "work.bin").read_bytes(), b"B")

    def test_rejects_external_and_shared_episode_paths(self):
        entry = self.project.data["episodes"][1]
        for unsafe in (str(self.root), "../outside", ".", "episodes/A"):
            with self.subTest(path=unsafe):
                entry["path"] = unsafe
                with self.assertRaises(ValueError):
                    self.project.remove_episode("B")
        self.assertEqual(self.project.episode_ids(), ["A", "B", "C"])
        self.assertFalse((self.project.root / "已移除").exists())

    def test_rename_failure_and_missing_id_preserve_list(self):
        with patch.object(Path, "rename", side_effect=OSError("test rename failure")):
            with self.assertRaises(OSError):
                self.project.remove_episode("B")
        with self.assertRaises(KeyError):
            self.project.remove_episode("missing")
        self.assertEqual(self.project.episode_ids(), ["A", "B", "C"])
        self.assertTrue(self.project.episode("B").dir.is_dir())


    def test_original_media_inside_episode_is_not_moved(self):
        ep = self.project.episode("B")
        original = ep.dir / "original.mkv"
        original.write_bytes(b"source media")
        ep.data["media"]["video"]["path"] = str(original)
        ep.save_meta()
        with self.assertRaisesRegex(ValueError, "原视频或原字幕"):
            self.project.remove_episode("B")
        self.assertEqual(original.read_bytes(), b"source media")
        self.assertEqual(self.project.episode_ids(), ["A", "B", "C"])


class RemoveEpisodeUI(unittest.TestCase):
    setUpClass = classmethod(help_tests.HelpTests.setUpClass.__func__)

    def setUp(self):
        help_tests.HelpTests.setUp(self)
        p = patch.object(EpisodeSession, "request_preview", Mock())
        p.start()
        self.addCleanup(p.stop)
        w = self.window
        w._load_media = Mock()
        project = Project.create(Path(self.tmp.name) / "project", "测试")
        for eid in ("A", "B", "C"):
            ep = project.add_episode(eid)
            ep.document = Document(eid)
            ep.save()
        w.open_project(project.root)
        w._switch_episode("B")

    def dispose(self):
        self.window.close_session()
        help_tests.HelpTests.dispose(self)

    def choose(self, accept=True, help_check=False):
        seen = []
        def interact(dialog):
            expected = "把 B 从工程里移除？这一集的工作文件会移到工程文件夹里的「已移除」文件夹，需要时可以手动找回；原视频和原字幕不受影响。"
            self.assertEqual(dialog.text(), expected)
            self.assertEqual(dialog.informativeText(), "不能用 Ctrl+Z 撤销。")
            buttons = {b.text(): b for b in dialog.findChildren(QPushButton)}
            self.assertEqual(set(buttons), {"移除", "取消", "帮助"})
            self.assertIs(dialog.defaultButton(), buttons["取消"])
            if help_check:
                with patch.object(help_ui, "open_manual") as opener:
                    QTest.keyClick(dialog, Qt.Key.Key_F1)
                    opener.assert_called_once_with("project")
            seen.append(True)
            buttons["移除" if accept else "取消"].click()
        original = QMessageBox.exec
        def execute(dialog):
            QTimer.singleShot(0, lambda: interact(dialog))
            return original(dialog)
        with patch.object(QMessageBox, "exec", execute):
            self.window.remove_episode()
        self.assertEqual(seen, [True])

    def test_confirm_moves_and_switches_to_next(self):
        self.choose(help_check=True)
        w = self.window
        self.assertEqual(w.project.episode_ids(), ["A", "C"])
        self.assertEqual([w.ep_box.itemText(i) for i in range(w.ep_box.count())], ["A", "C"])
        self.assertEqual(w.session.episode.id, "C")
        self.assertEqual(len(list((w.project.root / "已移除").iterdir())), 1)

    def test_cancel_preserves_session_and_files(self):
        old = self.window.session
        self.choose(False)
        self.assertIs(self.window.session, old)
        self.assertEqual(self.window.project.episode_ids(), ["A", "B", "C"])
        self.assertFalse((self.window.project.root / "已移除").exists())

    def test_background_job_blocks_action_and_direct_call(self):
        w = self.window
        w._episode_job = object()
        try:
            w._update_remove_episode_action()
            self.assertFalse(w.act_remove_episode.isEnabled())
            self.assertIn(w.act_remove_episode, w._episode_actions)
            with patch.object(QMessageBox, "exec", side_effect=AssertionError("must not open")):
                w.remove_episode()
            self.assertEqual(w.project.episode_ids(), ["A", "B", "C"])
        finally:
            w._episode_job = None
        w._update_remove_episode_action()
        self.assertTrue(w.act_remove_episode.isEnabled())

    def test_last_episode_switches_previous_then_empty(self):
        w = self.window
        self.choose()
        # The text assertion above specifically checks B; subsequent confirms use the real dialog too.
        def accept(dialog):
            button = next(b for b in dialog.buttons() if b.text() == "移除")
            QTimer.singleShot(0, button.click)
            return original(dialog)
        original = QMessageBox.exec
        with patch.object(QMessageBox, "exec", accept):
            w.remove_episode()
            self.assertEqual(w.session.episode.id, "A")
            w.remove_episode()
        self.assertIsNone(w.session)
        self.assertEqual(w.ep_box.count(), 0)
        self.assertIsNone(w.wave.doc)
        self.assertIsNone(w.current_gid)
        self.assertIsNone(w.table.model())
        self.assertIsNone(w.editor.group)
        self.assertEqual(w.source.toPlainText(), "")
        self.assertFalse(w.act_remove_episode.isEnabled())


if __name__ == "__main__":
    unittest.main()
