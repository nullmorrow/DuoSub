"""Optional reference reports: real Qt export paths, detached jobs and protected outputs."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from tests import test_convert_ui as ui
from tests import test_season_export as season_ui
from PySide6.QtWidgets import QDialog, QMessageBox
from subkit.core.export import ExportReport
from subkit.core.fontcheck import FontCheck
from subkit.core.reference_report import preflight_targets, render_reference_report, write_reference_report
from subkit.core.workspace import Episode, Project
from subkit.model.document import Document, Group, Member, Run
from subkit.proc import CancelledError
from subkit_app import mainwindow as mw
from subkit_app.views.dialogs import ExportDialog
from subkit_app.views.season_export import SeasonExportDialog, SeasonReportDialog


def packet(episode="E1", current="当前中文", *, protected=()):
    return dict(schema=1, episode=episode, direction="traditional", issues=["没有唯一参考时自动回退。"],
        protected_input_paths=list(protected),
        sources=[dict(source_id="t1", source_path="原始.ass", source_sha256="a" * 64,
                      target_path="原配.ass", target_sha256="b" * 64, baseline_path="底稿.ass",
                      baseline_sha256="c" * 64, issues=[])],
        rows=[dict(group="g1", member="m1", start=1000, end=3000, lang="zh-Hans", source_path="原始.ass",
                   source_event=0, imported="导入中文", baseline="原版中文", current=current, reference="原配中文",
                   automatic="自动中文", generated="采用中文", final="最终中文", strategy="opencc",
                   reason="自动回退原因", changed=True, preimport_changed=True)])


def document(eid="E1", text="这里是軟體"):
    m = Member("m1", "g1", "zh-Hans", "translation", "dialogue", 1000, 3000, style="CN",
               runs=[Run(text, ins="i1", b=len(text))])
    return Document(eid, groups=[Group("g1", members=[m.id])], members={m.id: m})


class ReportFiles(unittest.TestCase):
    def test_html_is_escaped_self_contained_and_has_all_comparison_fields(self):
        data = packet(current="</pre><script>alert('x')</script>\n第二行")
        data["sources"][0]["target_path"] = '<img src=x onerror="bad">.ass'
        value = render_reference_report(data)
        self.assertNotIn("<script", value)
        self.assertNotIn("<img", value)
        self.assertIn("&lt;script&gt;", value)
        self.assertIn("&lt;img", value)
        self.assertIn("a" * 64, value)
        self.assertIn("<style>", value)
        for field in ("导入中文", "原版中文", "原配中文", "自动中文", "采用中文", "最终中文", "自动回退原因"):
            self.assertIn(field, value)
        self.assertNotIn("<script src", value)
        self.assertNotIn("<link", value)

    def test_render_checks_cancellation(self):
        class Runner:
            calls = 0
            def check_cancelled(self):
                self.calls += 1
                if self.calls == 3:
                    raise CancelledError()
        runner = Runner()
        with self.assertRaises(CancelledError):
            render_reference_report(packet(), runner=runner)
        self.assertEqual(runner.calls, 3)

    def test_absolute_samefile_protection_and_atomic_report_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.ass"
            source.write_bytes(b"source original")
            alias = root / "alias.ass"
            os.link(source, alias)
            for output in (source, alias, source.parent / "sub" / ".." / source.name):
                with self.subTest(output=output):
                    with self.assertRaises(PermissionError):
                        preflight_targets([root / "safe.ass", output], [source])
            data = packet(protected=[str(source)])
            data["html"] = render_reference_report(data)
            target = root / "output.ass"
            report = target.with_suffix(".reference.html")
            report.write_bytes(b"old report")
            with patch("subkit.core.reference_report.os.replace", side_effect=OSError("disk failed")):
                with self.assertRaises(OSError):
                    write_reference_report(data, target)
            self.assertEqual(report.read_bytes(), b"old report")
            self.assertEqual(source.read_bytes(), b"source original")
            self.assertFalse(list(root.glob("*.tmp")))
            self.assertEqual(write_reference_report(data, target), report)


class SingleReferenceUI(unittest.TestCase):
    setUpClass = classmethod(ui.ConversionUI.setUpClass.__func__)
    _dispose = ui.ConversionUI._dispose

    def setUp(self):
        ui.ConversionUI.setUp(self)
        self.project = Project.create(self.path / "reference-project", "原配测试")
        self.episode = self.project.add_episode("E1")
        self.episode.document = document()
        self.session.episode = self.episode
        self.window.project = self.project
        self.protected, self.calls = [], []

    def choose(self, dialog, *, enabled=True, both=False, hint=""):
        dialog.path.setText(str(self.path / "out.ass"))
        dialog.zh_both.setChecked(both)
        dialog.reference.setChecked(enabled)
        dialog.reference_path.setText(hint)
        return QDialog.DialogCode.Accepted

    def build(self, original):
        def run(ep, **options):
            enabled = options.pop("reference_enabled", False)
            hint = options.pop("reference_path", None)
            self.calls.append((enabled, hint, options.get("zh_direction")))
            ass, report = original(ep, **options)
            if enabled:
                report.reference = packet(ep.id, ep.document.members["m1"].text, protected=self.protected)
                report.reference["direction"] = options.get("zh_direction") or "traditional"
            return ass, report
        return run

    def export(self, *, enabled=True, both=False, hint="", fake=True):
        original = Episode.build_export
        with patch.object(ExportDialog, "exec", lambda d: self.choose(d, enabled=enabled, both=both, hint=hint)), \
                patch.object(mw, "check_fonts", return_value=FontCheck()), \
                patch.object(QMessageBox, "information"), \
                patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes), \
                patch.object(Episode, "build_export", self.build(original) if fake else original):
            self.window.export()

    def test_checkbox_default_off_optional_picker_and_off_preserves_outputs(self):
        d = ExportDialog(has_chinese=True)
        self.addCleanup(d.deleteLater)
        self.assertFalse(d.reference.isChecked())
        self.assertFalse(d.reference_row.isEnabled())
        d.reference.setChecked(True)
        self.assertTrue(d.reference_row.isEnabled())
        self.assertEqual(d.reference_path.text(), "")
        self.export(enabled=False)
        self.assertTrue((self.path / "out.ass").exists())
        self.assertFalse(list(self.path.glob("*.reference.html")))
        self.assertFalse(self.calls[0][0])

    def test_dual_reports_are_built_in_worker_and_unsaved_snapshot_is_unchanged(self):
        before = copy.deepcopy(self.episode.document)
        threads, original = [], render_reference_report
        def render(data, **kw):
            threads.append(threading.get_ident())
            return original(data, **kw)
        hint = str(self.path / "selected-reference.ass")
        with patch("subkit.core.reference_report.render_reference_report", side_effect=render):
            self.export(both=True, hint=hint)
        self.assertEqual(len(threads), 2)
        self.assertTrue(all(t != threading.get_ident() for t in threads))
        self.assertEqual([c[2] for c in self.calls], ["simplified", "traditional"])
        self.assertTrue(all(c[1] == Path(hint) for c in self.calls))
        for suffix in ("zh-Hans", "zh-Hant"):
            report = self.path / ("out." + suffix + ".reference.html")
            self.assertIn("这里是軟體", report.read_text(encoding="utf-8"))
        self.assertEqual(self.episode.document, before)

    def test_real_core_missing_reference_exports_without_extra_confirmation(self):
        hint = self.path / "missing-reference.ass"
        self.export(both=True, hint=str(hint), fake=False)
        for script in ("zh-Hans", "zh-Hant"):
            target = self.path / f"out.{script}.ass"
            self.assertTrue(target.is_file())
            html = target.with_suffix(".reference.html").read_text(encoding="utf-8")
            self.assertIn("没有可唯一核对的原始事件", html)
            self.assertIn("最终 ASS 可见文字", html)

    def test_cancel_after_html_generation_writes_nothing(self):
        old = self.path / "out.ass"
        old.write_bytes(b"old ass")
        def cancel(data, **kw):
            self.window._export_job.cancel()
            return "html"
        with patch("subkit.core.reference_report.render_reference_report", side_effect=cancel):
            self.export()
        self.assertEqual(old.read_bytes(), b"old ass")
        self.assertFalse(list(self.path.glob("*.html")))
        self.assertFalse(list(self.path.glob("*.export-report.txt")))

    def test_html_generation_and_write_errors_keep_ass_and_are_reported(self):
        with patch("subkit.core.reference_report.render_reference_report", side_effect=OSError("render failed")):
            self.export()
        self.assertTrue((self.path / "out.ass").exists())
        self.assertIn("render failed（ASS 已导出）", (self.path / "out.export-report.txt").read_text(encoding="utf-8"))
        with patch("subkit.core.reference_report.write_reference_report", side_effect=OSError("write failed")):
            self.export()
        self.assertIn("write failed（ASS 已导出）", (self.path / "out.export-report.txt").read_text(encoding="utf-8"))

    def test_second_output_reference_collision_prevents_first_write(self):
        protected = self.path / "out.zh-Hant.ass"
        protected.write_bytes(b"original reference")
        self.protected = [str(protected)]
        with patch.object(QMessageBox, "warning") as warning:
            self.export(both=True)
        warning.assert_called_once()
        self.assertEqual(protected.read_bytes(), b"original reference")
        self.assertFalse((self.path / "out.zh-Hans.ass").exists())
        self.assertFalse(list(self.path.glob("*.html")))
        self.assertFalse(list(self.path.glob("*.export-report.txt")))


class SeasonReferenceUI(unittest.TestCase):
    setUpClass = season_ui.SeasonExportUI.__dict__["setUpClass"]
    _dispose = season_ui.SeasonExportUI._dispose

    def setUp(self):
        season_ui.SeasonExportUI.setUp(self)
        self.window.project, self.window.session = self.project, None
        for eid in ("S01E001", "S01E002"):
            ep = self.project.episode(eid)
            ep.document = document(eid, "已经保存")
            ep.save()
        self.protected = []

    def choose(self, dialog, *, both=True, review=True, skip=False):
        dialog.folder.setText(str(self.out))
        dialog.zh_both.setChecked(both)
        dialog.review_packet.setChecked(review)
        dialog.reference.setChecked(True)
        dialog.reference_path.setText("")
        with patch.object(QMessageBox, "exec", lambda box: next(b for b in box.buttons()
                if b.text() == ("跳过这些文件" if skip and both else "跳过这些集" if skip else "覆盖")).click()):
            dialog.accept()
        return dialog.result()

    def export(self, *, both=True, review=True, skip=False):
        original = Episode.build_export
        def build(ep, **options):
            enabled = options.pop("reference_enabled", False)
            options.pop("reference_path", None)
            ass, report = original(ep, **options)
            if enabled:
                report.reference = packet(ep.id, ep.document.members["m1"].text, protected=self.protected)
            return ass, report
        with patch.object(SeasonExportDialog, "exec", lambda d: self.choose(d, both=both, review=review, skip=skip)), \
                patch.object(SeasonReportDialog, "exec", return_value=1), \
                patch.object(mw, "check_fonts", return_value=FontCheck()), patch.object(Episode, "build_export", build):
            self.window.export_season()

    def test_default_off_and_dual_reports_coexist_with_review_packets_and_unsaved_current(self):
        d = SeasonExportDialog(self.project, self.settings)
        self.addCleanup(d.deleteLater)
        self.assertFalse(d.reference.isChecked())
        self.assertFalse(d.reference_path.isEnabled())
        d.reference.setChecked(True)
        self.assertTrue(d.reference_path.isEnabled())
        current = self.project.episode("S01E001")
        current.document = document(current.id, "还没保存")
        self.session.episode = current
        self.window.session = self.session
        saved = (current.dir / "document.json").read_bytes()
        self.export()
        reports = list(self.out.glob("*.reference.html"))
        self.assertEqual(len(reports), 4)
        self.assertEqual(len(list(self.out.glob("*.review-*.md"))), 2)
        self.assertEqual(len(list(self.out.glob("*.review-*.jsonl"))), 2)
        self.assertEqual(sum("还没保存" in p.read_text(encoding="utf-8") for p in reports), 2)
        self.assertEqual((current.dir / "document.json").read_bytes(), saved)

    def test_skip_and_cross_episode_input_protection_before_all_writes(self):
        old = self.out / "S01E002.zh-Hant.ass"
        old.write_bytes(b"input reference")
        self.protected = [str(old)]
        with patch.object(QMessageBox, "warning") as warning:
            self.export()
        warning.assert_called_once()
        self.assertEqual(list(self.out.iterdir()), [old])
        self.assertEqual(old.read_bytes(), b"input reference")
        self.protected = []
        self.export(both=False, review=False, skip=True)
        self.assertEqual(len(list(self.out.glob("*.reference.html"))), 1)
        self.assertFalse((self.out / "S01E002.zh-Hant.reference.html").exists())
        self.assertEqual(old.read_bytes(), b"input reference")

    def test_cancel_last_report_preserves_every_old_byte_and_writes_no_review(self):
        old = self.out / "old.html"
        old.write_bytes(b"keep old report")
        def render(data, **kw):
            if data["episode"] == "S01E002":
                self.window._export_job.cancel()
            return "html"
        with patch("subkit.core.reference_report.render_reference_report", side_effect=render):
            self.export()
        self.assertEqual(list(self.out.iterdir()), [old])
        self.assertEqual(old.read_bytes(), b"keep old report")


if __name__ == "__main__":
    unittest.main()
