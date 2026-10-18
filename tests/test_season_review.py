"""One season export writes ASS and the existing, importable review packets."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import test_season_export as fixtures
from tests.test_pinyin_export import document
from PySide6.QtWidgets import QApplication, QMessageBox
from subkit.core.display import DisplayRules
from subkit.core.fontcheck import FontCheck
from subkit.core.ops import Editor
from subkit.core.review import (available_packet_paths, build_packet, packet_paths,
                                write_packet, write_packet_files)
from subkit.core.review_import import Answer, apply, match
from subkit.core.workspace import Episode, Project
from subkit.model.document import Document, Group
from subkit.proc import CancelledError
from subkit_app.views.season_export import SeasonExportDialog, SeasonReportDialog


class PacketFiles(unittest.TestCase):
    def test_fresh_names_and_exclusive_pair_preserve_old_files(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            md, jl = packet_paths(folder, episode="E1", stamp="fixed")
            md.write_bytes(b"old md")
            _, jl1 = packet_paths(folder, episode="E1", stamp="fixed-1")
            jl1.write_bytes(b"old jsonl")
            paths = available_packet_paths(folder, episode="E1", stamp="fixed")
            self.assertEqual(paths[0].name, "E1.review-fixed-2.md")
            write_packet_files(("汉字(かんじ)\n", '{"group":"g"}\n'), paths)
            self.assertEqual(paths[0].read_text(encoding="utf-8"), "汉字(かんじ)\n")
            self.assertEqual(md.read_bytes(), b"old md")
            self.assertEqual(jl1.read_bytes(), b"old jsonl")
            self.assertFalse(jl.exists())
            # Another file appears after planning: never truncate it, and remove
            # only the empty MD file this attempt created before opening JSONL.
            raced = packet_paths(folder, episode="E2", stamp="fixed")
            raced[1].write_bytes(b"new external file")
            with self.assertRaises(FileExistsError):
                write_packet_files(("md", "jl"), raced)
            self.assertFalse(raced[0].exists())
            self.assertEqual(raced[1].read_bytes(), b"new external file")

    def test_packet_output_and_single_episode_writer_stay_the_same(self):
        doc = document("这里是軟體")
        before = copy.deepcopy(doc)
        expected = build_packet(doc, episode="E1")
        self.assertEqual(build_packet(doc, episode="E1", check_cancelled=lambda: None), expected)
        with tempfile.TemporaryDirectory() as folder:
            paths = write_packet(doc, Path(folder), episode="E1", stamp="fixed")
            self.assertEqual(tuple(p.read_text(encoding="utf-8") for p in paths), expected)
        self.assertEqual(doc, before)

    def test_cancellation_is_checked_while_building_large_packets(self):
        doc = Document("E1", groups=[Group(str(i), kind="sign") for i in range(100)])
        checks = 0
        def cancel():
            nonlocal checks
            checks += 1
            if checks == 4:
                raise CancelledError()
        with self.assertRaises(CancelledError):
            build_packet(doc, episode="E1", check_cancelled=cancel)
        self.assertEqual(checks, 4)


class SeasonReviewUI(unittest.TestCase):
    setUpClass = fixtures.SeasonExportUI.__dict__["setUpClass"]
    setUp = fixtures.SeasonExportUI.setUp
    _dispose = fixtures.SeasonExportUI._dispose
    dialog = fixtures.SeasonExportUI.dialog

    def populate(self, text="这里是軟體"):
        for eid in ("S01E001", "S01E002"):
            ep = self.project.episode(eid)
            ep.document = document(text)
            ep.document.episode = eid
            ep.save()
        self.window.project, self.window.session = self.project, None

    def choose(self, dialog, *, both=False, review=True):
        dialog.folder.setText(str(self.out))
        dialog.zh_both.setChecked(both)
        dialog.review_packet.setChecked(review)
        dialog._review_stamp = "fixed"
        with patch.object(QMessageBox, "exec", lambda box: next(
                b for b in box.buttons() if b.text() == "覆盖").click()):
            dialog.accept()
        return dialog.result()

    def export(self, *, both=False, review=True):
        with patch.object(SeasonExportDialog, "exec", lambda d: self.choose(d, both=both, review=review)), \
                patch.object(SeasonReportDialog, "exec", return_value=1), \
                patch("subkit_app.mainwindow.check_fonts", return_value=FontCheck()):
            self.window.export_season()

    def report(self, both=False):
        suffix = "zh-Hans+zh-Hant" if both else "zh-Hant"
        return (self.out / ("project." + suffix + ".export-report.txt")).read_text(encoding="utf-8")

    def test_option_default_off_remembered_and_only_chosen_episodes(self):
        d = self.dialog()
        self.assertFalse(d.review_packet.isChecked())
        self.assertTrue(d.review_note.isHidden())
        self.assertEqual(d.review_targets(), {})
        d.folder.setText(str(self.out))
        d.review_packet.setChecked(True)
        d._review_stamp = "fixed"
        (self.out / "S01E002.zh-Hant.ass").write_bytes(b"keep")
        with patch.object(QMessageBox, "exec", lambda box: next(
                b for b in box.buttons() if b.text() == "跳过这些集").click()):
            d.accept()
        self.assertEqual(list(d.review_targets()), ["S01E001"])
        self.assertTrue(self.dialog().review_packet.isChecked())
        self.assertEqual(len(d.review_targets()["S01E001"]), 2)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["S01E002.zh-Hant.ass"])

    def test_whole_season_writes_packets_once_for_dual_ass_in_background(self):
        self.populate()
        gui_thread = threading.get_ident()
        threads, opened = [], []
        original_open, original_packet = self.project.episode, build_packet
        def open_episode(eid):
            opened.append(eid)
            return original_open(eid)
        def packet(*args, **kw):
            threads.append(threading.get_ident())
            return original_packet(*args, **kw)
        with patch.object(self.project, "episode", side_effect=open_episode), \
                patch("subkit.core.review.build_packet", side_effect=packet):
            self.export(both=True)
        self.assertEqual(opened, ["S01E001", "S01E002"])
        self.assertEqual(len(threads), 2)
        self.assertTrue(all(t != gui_thread for t in threads))
        self.assertEqual(len(list(self.out.glob("*.ass"))), 4)
        self.assertEqual(len(list(self.out.glob("*.review-fixed.md"))), 2)
        self.assertEqual(len(list(self.out.glob("*.review-fixed.jsonl"))), 2)
        for eid in ("S01E001", "S01E002"):
            row = json.loads((self.out / (eid + ".review-fixed.jsonl")).read_text(encoding="utf-8"))
            self.assertEqual(row["group"], "g_1")
            self.assertEqual(row["translation"][0]["text"], "这里是軟體")
            self.assertIn("请只用下面的格式回答", (self.out / (eid + ".review-fixed.md")).read_text(encoding="utf-8"))
        self.assertIn("审议包已导出 2 份", self.report(both=True))
        self.assertIsNone(self.window._export_job)
        self.assertIsNone(QApplication.overrideCursor())

    def test_unsaved_current_episode_and_saved_others_share_packet_and_ass_snapshot(self):
        self.populate("已经保存")
        current = self.project.episode("S01E001")
        current.document = document("还没保存")
        current.document.episode = current.id
        self.session.episode = current
        self.window.session = self.session
        saved = (current.dir / "document.json").read_bytes()
        self.export()
        self.assertIn("还没保存", (self.out / (fixtures.VIDEO[:-4] + ".zh-Hant.ass")).read_text(encoding="utf-8-sig"))
        self.assertIn("还没保存", (self.out / "S01E001.review-fixed.md").read_text(encoding="utf-8"))
        self.assertIn("已经保存", (self.out / "S01E002.review-fixed.md").read_text(encoding="utf-8"))
        self.assertEqual((current.dir / "document.json").read_bytes(), saved)

    def test_packet_failure_does_not_discard_successful_ass(self):
        self.populate()
        original_packet = build_packet
        def packet(*args, **kw):
            if kw["episode"] == "S01E001":
                raise OSError("packet failed")
            return original_packet(*args, **kw)
        with patch("subkit.core.review.build_packet", side_effect=packet):
            self.export()
        self.assertEqual(len(list(self.out.glob("*.ass"))), 2)
        self.assertEqual(len(list(self.out.glob("*.md"))), 1)
        report = self.report()
        self.assertIn("审议包已导出 1 份", report)
        self.assertIn("packet failed（ASS 已导出）", report)
        self.assertIn("审议包没有导出：1 集（S01E001）", report)

    def test_cancel_during_or_after_last_packet_leaves_all_old_bytes(self):
        self.populate()
        for during in (True, False):
            with self.subTest(during=during):
                old = self.out / "S01E002.zh-Hant.ass"
                old.write_bytes(b"old ass")
                for path, data in zip(packet_paths(self.out, episode="S01E001", stamp="fixed"), (b"old md", b"old jl")):
                    path.write_bytes(data)
                (self.out / "project.zh-Hans+zh-Hant.export-report.txt").write_bytes(b"old report")
                snapshot = {p.name: p.read_bytes() for p in self.out.iterdir()}
                original_packet = build_packet
                def packet(*args, **kw):
                    if kw["episode"] != "S01E002":
                        return original_packet(*args, **kw)
                    if during:
                        original_check = kw["check_cancelled"]
                        checks = 0
                        def cancel():
                            nonlocal checks
                            checks += 1
                            if checks == 3:
                                self.window._export_job.cancel()
                            original_check()
                        kw["check_cancelled"] = cancel
                        return original_packet(*args, **kw)
                    value = original_packet(*args, **kw)
                    self.window._export_job.cancel()
                    return value
                with patch("subkit.core.review.build_packet", side_effect=packet), \
                        patch.object(SeasonExportDialog, "exec", lambda d: self.choose(d, both=True)), \
                        patch("subkit_app.mainwindow.check_fonts", return_value=FontCheck()), \
                        patch.object(SeasonReportDialog, "exec") as shown:
                    self.window.export_season()
                shown.assert_not_called()
                self.assertEqual({p.name: p.read_bytes() for p in self.out.iterdir()}, snapshot)
                self.assertIsNone(self.window._export_job)
                self.assertEqual(self.window.statusBar().currentMessage(), "已取消整季导出，没有写入文件")

    def test_packet_write_or_close_failure_cleans_only_new_pair_and_keeps_ass(self):
        self.populate()
        old_pair = packet_paths(self.out, episode="S01E001", stamp="fixed")
        for path, data in zip(old_pair, (b"old markdown", b"old jsonl")):
            path.write_bytes(data)
        new_pair = packet_paths(self.out, episode="S01E001", stamp="fixed-1")
        original_open = Path.open
        for fail_at in ("write", "close"):
            with self.subTest(fail_at=fail_at):
                class BrokenWriter:
                    def __init__(self, stream):
                        self.stream = stream
                    def __enter__(self):
                        return self
                    def write(self, text):
                        if fail_at == "write":
                            raise OSError("simulated packet disk failure")
                        return self.stream.write(text)
                    def __exit__(self, *_exc):
                        self.stream.close()
                        if fail_at == "close":
                            raise OSError("simulated packet disk failure")
                def open_path(path, mode="r", *args, **kw):
                    stream = original_open(path, mode, *args, **kw)
                    return BrokenWriter(stream) if path == new_pair[1] and mode == "x" else stream
                with patch.object(Path, "open", open_path):
                    self.export()
                self.assertEqual(tuple(p.read_bytes() for p in old_pair), (b"old markdown", b"old jsonl"))
                self.assertFalse(any(p.exists() for p in new_pair))
                self.assertEqual(len(list(self.out.glob("*.ass"))), 2)
                self.assertIn("simulated packet disk failure（ASS 已导出）", self.report())

    def test_new_project_real_import_export_packet_answer_can_be_applied(self):
        project = Project.create(self.root / "fresh", "审议集成", translation="zh-Hans")
        project.data.setdefault("settings", {})["display"] = {
            "zh_script": "traditional", "zh_traditional": {"variant": "tw", "phrases": True}}
        project.save()
        cn, ja = self.root / "cn.srt", self.root / "ja.srt"
        cn.write_text("1\n00:00:01,000 --> 00:00:03,000\n这里是软件\n", encoding="utf-8")
        ja.write_text("1\n00:00:01,000 --> 00:00:03,000\n学校です\n", encoding="utf-8")
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (cn, ja)}
        ep = project.add_episode("E1")
        ep.add_source(cn, lang="zh-Hans", lane="translation")
        ep.add_source(ja, lang="ja", lane="original")
        ep.merge()
        self.project = project
        self.window.project, self.window.session = project, None
        self.export(both=True)
        payload = json.loads((self.out / "E1.review-fixed.jsonl").read_text(encoding="utf-8"))
        text = payload["translation"][0]["text"]
        self.assertEqual(text, "這裡是軟體")
        self.assertIn("这里是软件", (self.out / "E1.zh-Hans.ass").read_text(encoding="utf-8-sig"))
        self.assertIn("這裡是軟體", (self.out / "E1.zh-Hant.ass").read_text(encoding="utf-8-sig"))
        item = Answer(payload["group"], "zh", "text", "軟體", "軟體", "工具", "审议测试")
        reopened = project.episode("E1")
        ed = Editor(reopened.document, reopened.sources)
        suggestion = match(ed.doc, ed.sources, DisplayRules.from_settings(project.settings()), item)
        self.assertTrue(apply(ed, suggestion))
        self.assertTrue(any("工具" in m.text for m in ed.doc.members.values()))
        for p, digest in hashes.items():
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
