"""视频文件位置: moved / renamed videos are found again; caches survive the move (2026-10-15)."""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import test_convert_ui as fixtures
from PySide6.QtWidgets import QFileDialog, QMessageBox
from subkit.audio import legacy_waveform_cache_file, waveform_cache_file
from subkit.core.video_link import match_folder, relink, video_entries
from subkit.core.workspace import Project
from subkit.kalign import cache as kcache
from subkit.model.serialize import read_json
from subkit_app.views.video_links import VideoLinksDialog

NAMES = ("[Group] Show - 01 [1080p].mkv", "[Group] Show - 02 [1080p].mkv", "[Group] Show - 10 [HEVC-10bit].mkv")


class Links(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="subkit_videos_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old = self.root / "old"
        self.old.mkdir()
        self.project = Project.create(self.root / "project", "视频")
        for i, name in enumerate(NAMES):
            video = self.old / name
            video.write_bytes(b"x" * (100 + i))
            ep = self.project.add_episode(f"S01E{int(name.split(' - ')[1][:2]):03d}",
                                          number=int(name.split(" - ")[1][:2]))
            ep.set_video(video, audio_track=2 if i == 0 else None)

    def move(self, renames=None):
        new = self.root / "archive" / "season"
        new.mkdir(parents=True)
        for name in NAMES:
            stamp = (self.old / name).stat().st_mtime_ns
            target = new / (renames or {}).get(name, name)
            (self.old / name).rename(target)
            os.utime(target, ns=(stamp, stamp))
        return new

    def test_match_by_name_size_and_number(self):
        new = self.move({NAMES[1]: "第02話.mkv"})
        (new / "Show - 10 extra.mkv").write_bytes(b"yy")                    # another "10": number stays ambiguous
        entries = video_entries(self.project)
        self.assertEqual([(e.eid, e.missing, e.track) for e in entries],
                         [("S01E001", True, 2), ("S01E002", True, None), ("S01E010", True, None)])
        found = match_folder(entries, self.root / "archive")
        self.assertEqual({eid: (path.name, rule) for eid, (path, rule) in found.items()},
                         {"S01E001": (NAMES[0], "name"), "S01E002": ("第02話.mkv", "size"),
                          "S01E010": (NAMES[2], "name")})
        entries[1].size = None                                                  # no size recorded: by number
        self.assertEqual(match_folder(entries[1:2], self.root / "archive")["S01E002"][1], "number")

    def test_relink_writes_episode_json_and_carries_caches(self):
        cache = self.project.cache_dir
        entry = video_entries(self.project)[0]
        old_video = entry.recorded
        old_wave = legacy_waveform_cache_file(old_video, 2, cache / "waveform")
        old_wave.parent.mkdir(parents=True)
        old_wave.write_bytes(b"wave")
        old_kalign = cache / "kalign" / kcache.legacy_cache_key(old_video, 1, None)
        old_kalign.mkdir(parents=True)
        (old_kalign / "rms.npy").write_bytes(b"rms")
        new = self.move()
        relink(entry, new / NAMES[0], cache)
        media = read_json(entry.folder / "episode.json")["media"]
        self.assertEqual((media["video"]["path"], media["audio_track"]), (str(new / NAMES[0]).replace("\\", "/"), 2))
        self.assertFalse(video_entries(self.project)[0].missing)
        self.assertEqual(waveform_cache_file(new / NAMES[0], 2, cache / "waveform").read_bytes(), b"wave")
        self.assertEqual((kcache.cache_directory(cache / "kalign", new / NAMES[0], 1, None) / "rms.npy").read_bytes(), b"rms")
        self.assertFalse(old_wave.exists())

    def test_caches_no_longer_depend_on_the_path(self):
        video = self.old / NAMES[0]
        cache = self.root / "cache"
        legacy = legacy_waveform_cache_file(video, None, cache)
        cache.mkdir()
        legacy.write_bytes(b"old name")
        path = waveform_cache_file(video, None, cache)                           # renamed on first use
        self.assertEqual(path.read_bytes(), b"old name")
        self.assertFalse(legacy.exists())
        key = kcache.cache_key(video, 0, (1000, 2000))
        new = self.move()
        self.assertEqual(waveform_cache_file(new / NAMES[0], None, cache), path)
        self.assertEqual(kcache.cache_key(new / NAMES[0], 0, (1000, 2000)), key)


class LinksDialog(Links):
    @classmethod
    def setUpClass(cls):
        fixtures.ConversionUI.setUpClass()

    def test_choose_one_file_pairs_the_folder_then_apply(self):
        new = self.move({NAMES[1]: "renamed.mkv"})
        dialog = VideoLinksDialog(self.project, focus_eid="S01E002")
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog.table.currentRow(), 1)
        self.assertIn("还有 3 集找不到视频", dialog.message.text())
        self.assertFalse(dialog.apply_button.isEnabled())
        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(new / "renamed.mkv"), "")):
            dialog.choose_file()
        self.assertEqual([dialog.table.item(r, 4).text() for r in range(3)], ["同名", "你选的", "同名"])
        self.assertIn("另外配上 2 集", dialog.message.text())
        dialog.apply()
        self.assertEqual(sorted(eid for eid, _ in dialog.applied), ["S01E001", "S01E002", "S01E010"])
        self.assertFalse(any(e.missing for e in video_entries(self.project)))

    def test_open_episode_warns_and_relinks_the_open_one(self):
        new = self.move()
        ep = self.project.episode("S01E001")
        from subkit_app.mainwindow import MainWindow
        with patch("subkit_app.mainwindow.MpvWidget", fixtures.PlayerDouble), \
                patch("subkit_app.mainwindow.MainWindow._open_last"):
            window = MainWindow()
        self.addCleanup(window.deleteLater)
        window.project = self.project
        window.session = type("S", (), {"episode": ep})()
        shown = []

        def answer(box):
            shown.append(box.text())
            next(b for b in box.buttons() if b.text() == "重新指定…").click()

        def relink_all(dialog):
            with patch.object(QFileDialog, "getExistingDirectory", return_value=str(new)):
                dialog.search_folder()
            dialog.apply()
            return dialog.result()

        with patch.object(QMessageBox, "exec", answer), patch.object(VideoLinksDialog, "exec", relink_all), \
                patch.object(window, "_load_media") as reload:
            window._warn_missing_video(ep)
        self.assertIn("找不到这一集的视频", shown[0])
        self.assertEqual(ep.video, new / NAMES[0])                              # the open episode in memory too
        self.assertEqual((ep.data["media"]["audio_track"]), 2)
        reload.assert_called_once_with(ep)
        window.session = None


if __name__ == "__main__":
    unittest.main()
