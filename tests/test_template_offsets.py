"""TP1: host libass placement and offscreen Qt standard-button regression checks."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from subkit import templates as T
from subkit.ass.document import AssDocument
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.layout_index import LayoutIndex
from subkit.core.workspace import Project
from subkit.model.document import Annotation
from subkit.proc import run
from .test_pinyin_export import document


class OffsetControls(unittest.TestCase):
    def test_scheme_defaults_round_trip_and_range(self):
        fonts, pal = T.scheme_from_dict({"fonts": "nana_black", "palette": "classic"})
        self.assertEqual((fonts.ruby_offset, fonts.pinyin_offset), (0, 0))
        changed = replace(fonts, ruby_offset=-40, pinyin_offset=40)
        self.assertEqual(T.scheme_from_dict(T.scheme_dict(changed, pal)), (changed, pal))
        for value in (-40.5, 40.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                replace(fonts, ruby_offset=value)

    def test_editor_live_values_preserved_by_presets_and_render(self):
        from PySide6.QtWidgets import QApplication
        from subkit_app.views.template_editor import TemplateEditor

        QApplication.instance() or QApplication([])
        editor = TemplateEditor(scheme=T.scheme_dict(replace(T.FONT_SETS[0], ruby_offset=3.5, pinyin_offset=-4),
                                                     T.PALETTES[0]))
        try:
            editor._timer.stop()
            for key in ("ruby_offset", "pinyin_offset"):
                spin = editor.w[key]
                self.assertEqual((spin.minimum(), spin.maximum(), spin.singleStep()), (-40, 40, .5))
                self.assertEqual(spin.toolTip(), "正数离正文更远，负数更近")
            editor.w["ruby_offset"].setValue(8.5)
            self.assertEqual(editor.fonts.ruby_offset, 8.5)
            self.assertTrue(editor._timer.isActive())
            editor.font_box_set.setCurrentIndex(editor.font_box_set.findData("utena_serif"))
            editor._font_preset(0)
            self.assertEqual((editor.fonts.ruby_offset, editor.fonts.pinyin_offset), (8.5, -4))
            with patch.object(editor, "_run_job") as job, patch.object(T, "render_scene") as render:
                editor._render()
                job.call_args.args[0](None)
                self.assertEqual(render.call_args.kwargs["ruby_offset"], 8.5)
                self.assertEqual(render.call_args.kwargs["pinyin_offset"], -4)
        finally:
            editor.close()

    def test_qt_chinese_standard_buttons_and_missing_translation(self):
        from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox
        from subkit_app.app import install_chinese_translator

        app = QApplication.instance() or QApplication([])
        previous = getattr(app, "_chinese_translator", None)
        self.assertTrue(install_chinese_translator(app))
        translator = app._chinese_translator
        try:
            box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
            message = QMessageBox()
            message.setStandardButtons(QMessageBox.StandardButton.Cancel)
            self.assertEqual(box.button(QDialogButtonBox.StandardButton.Cancel).text(), "取消")
            self.assertEqual(message.button(QMessageBox.StandardButton.Cancel).text(), "取消")
            box.close()
            message.close()
            with patch("PySide6.QtCore.QLibraryInfo.path", return_value="missing-qt-translations"):
                self.assertFalse(install_chinese_translator(app))
        finally:
            app.removeTranslator(translator)
            app._chinese_translator = previous


@unittest.skipUnless(shutil.which("ffmpeg"), "requires host ffmpeg/libass")
class OffsetLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="tp1-offset-")
        cls.path = Path(cls.tmp.name)
        cls.fonts = T.FontSet("test", "test", "Microsoft YaHei", "Microsoft YaHei", "Microsoft YaHei",
                              pinyin_font="Arial")
        cls.pal = T.palette("classic")
        cls.styles = T.build_styles(cls.fonts, cls.pal)
        cls.template = AssDocument.parse(cls.styles)
        cls.measure_cache = cls.path / "measure.sqlite3"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def export(self, doc, **kw):
        return build_ass(doc, {}, self.template, self.path, ExportOptions(pinyin_mode="PY", **kw),
                         measure_cache=self.measure_cache)

    def render(self, ass):
        ass.save(self.path / "offset.ass")
        result = run(["ffmpeg", "-hide_banner", "-v", "info", "-f", "lavfi", "-i",
                      "color=black:s=1920x1080:r=1:d=2", "-vf", "ass=offset.ass", "-frames:v", "1",
                      "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"], cwd=self.path, capture_output=True, check=True)
        frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape(1080, 1920)
        y, x = np.where(frame > 24)
        self.assertGreater(len(x), 0)
        return np.array([x.min(), y.min(), x.max(), y.max()]), result.stderr.decode("utf-8", "replace")

    def test_export_positive_negative_top_rotated_hit_and_cache(self):
        errors = []
        for kind in ("ruby", "pinyin"):
            for style, overrides in (("JP" if kind == "ruby" else "CN", {}),
                                     ("JP.Top" if kind == "ruby" else "CN.Top", {"pos": [960, 200]}),
                                     ("JP" if kind == "ruby" else "CN", {"raw": r"\frz90", "pos": [960, 540]})):
                doc = document("行", style=style)
                if kind == "ruby":
                    doc.members["m_1"].lang = "ja"
                    doc.annotations = {"a": Annotation("a", "ruby", "m_1", [{"ins": "i_1", "a": 0, "b": 1}],
                                                        "行", "こう")}
                doc.members["m_1"].overrides = overrides
                cache = ExportCache()
                for offset in (0, 12, -12):
                    ass, rep = build_ass(doc, {}, self.template, self.path,
                                        ExportOptions(pinyin_mode="PY", **{kind + "_offset": offset}),
                                        measure_cache=self.measure_cache, cache=cache)
                    row = next(i for i, o in enumerate(rep.owners) if o["kind"] == kind)
                    index = LayoutIndex(ass, rep.owners, measure_cache=self.measure_cache)
                    try:
                        geom = index.geometry([row])[0]
                        center = np.asarray(geom.quads[0]).mean(axis=0)
                        self.assertIsNotNone(index.hit([geom], *center, pad=0))
                    finally:
                        index.close()
                    ass.events = [ass.events[row]]
                    ink, log = self.render(ass)
                    if offset == 0:
                        baseline_ink, baseline_center = ink, center
                    else:
                        self.assertEqual(rep.laid_out, 1)  # changing offset invalidates the live cache
                        delta = (np.array([-offset, 0, -offset, 0]) if "raw" in overrides
                                 else np.array([0, -offset, 0, -offset]))
                        error = float(np.abs(ink - baseline_ink - delta).max())
                        errors.append(error)
                        self.assertLessEqual(error, 1, (kind, style, overrides, offset, ink, baseline_ink))
                        np.testing.assert_allclose(center - baseline_center, delta[:2], atol=.15)
        versions = [line for line in log.splitlines() if "libass" in line]
        print(f"TP1 host libass export: {len(errors)} shifts, max ink error={max(errors)} px; {versions}")

    def test_preview_all_scenes_and_offsets_render(self):
        for scene in T.SCENES:
            original = T.scene_events(self.styles, scene, pinyin=True, measure_cache=self.measure_cache)
            shifted = T.scene_events(self.styles, scene, pinyin=True, ruby_offset=7.5, pinyin_offset=-5.5,
                                     measure_cache=self.measure_cache)
            for old, new in zip(original, shifted):
                style = old.split(",", 4)[3]
                if style.endswith((".Ruby", ".Pinyin")):
                    before = float(re.search(r"\\pos\([^,]+,([^)]+)\)", old)[1])
                    after = float(re.search(r"\\pos\([^,]+,([^)]+)\)", new)[1])
                    self.assertAlmostEqual(after - before, -7.5 if style.endswith(".Ruby") else 5.5)
                else:
                    self.assertEqual(old, new)
        for kind in ("Ruby", "Pinyin"):
            boxes = []
            for offset in (0, 12, -12):
                events = T.scene_events(self.styles, "dialogue", pinyin=True, ruby_offset=offset,
                                        pinyin_offset=offset, measure_cache=self.measure_cache)
                ass = AssDocument.parse(T._HEAD + self.styles + T._EVH + "\n".join(events))
                ass.events = [e for e in ass.events if e.style.endswith("." + kind)]
                boxes.append(self.render(ass)[0])
            for offset, box in zip((12, -12), boxes[1:]):
                self.assertLessEqual(np.abs(box - boxes[0] - [0, -offset, 0, -offset]).max(), 1)
        print("TP1 host libass preview: ruby/pinyin ±12 px ink error <=1 px; all four scenes ±fractional anchors exact")

    def test_fit_measures_offsets_and_retains_values(self):
        fonts = replace(self.fonts, ruby_offset=8, pinyin_offset=10)
        old = T.clearances(self.fonts, self.pal, pinyin=True, measure_cache=self.measure_cache)
        moved = T.clearances(fonts, self.pal, pinyin=True, measure_cache=self.measure_cache)
        self.assertLessEqual(abs(moved["cn_ruby"] - old["cn_ruby"] + 8), 1)
        self.assertLessEqual(abs(moved["pinyin"] - old["pinyin"] - 10), 1)
        fitted, gaps = T.fit_layout(fonts, self.pal, pinyin=True, measure_cache=self.measure_cache)
        self.assertEqual((fitted.ruby_offset, fitted.pinyin_offset), (8, 10))
        for key in T.TARGET:
            self.assertLessEqual(abs(gaps[key] - T.TARGET[key]), 1)

    def test_project_round_trip_preview_and_template_version(self):
        from PySide6.QtWidgets import QApplication
        from subkit_app.session import EpisodeSession, PreviewJob

        QApplication.instance() or QApplication([])
        project = Project.create(self.path / "project", "offset", template_text=self.styles,
                                 template_scheme=T.scheme_dict(replace(self.fonts, pinyin_offset=9.5), self.pal))
        project.set_pinyin_mode("PY")
        episode = project.add_episode("E1")
        episode.document = document("行")
        baseline, rep0 = self.export(episode.document)
        ass, rep = episode.build_export(include_actor=False)
        old_y = next(o["y"] for o in rep0.owners if o["kind"] == "pinyin")
        self.assertAlmostEqual(next(o["y"] for o in rep.owners if o["kind"] == "pinyin"), old_y - 9.5)
        session = EpisodeSession(project, episode)
        try:
            job = PreviewJob(session, 0, self.path / "preview.ass")
            done, failed = [], []
            job.signals.done.connect(lambda *args: done.append(args))
            job.signals.failed.connect(lambda *args: failed.append(args))
            job.run()
            self.assertFalse(failed, failed)
            self.assertEqual([e.text for e in done[0][2].events], [e.text for e in ass.events])
        finally:
            session.close()
        project.replace_template(self.styles, T.scheme_dict(replace(self.fonts, pinyin_offset=9.5), self.pal))
        reopened = Project.open(project.root)
        self.assertEqual(reopened.data["template"]["version"], 2)
        self.assertEqual(T.scheme_from_dict(reopened.data["template"]["scheme"])[0].pinyin_offset, 9.5)
        reopened.replace_template(self.styles)
        self.assertEqual(T.scheme_from_dict(Project.open(project.root).data["template"]["scheme"])[0].pinyin_offset, 9.5)

    def test_generator_offset_scales_with_resolution(self):
        from subkit.ass.document import Event
        from subkit.ruby.generate import RubyGenerator, RubyJob, RubyPiece
        from subkit.ruby.measure import Measurer

        ass = AssDocument.parse(T._HEAD + self.styles + T._EVH)
        ass.set_info("PlayResY", 540)
        base = Event(style="JP", text=r"{\pos(960,350)\fs40}行{\fs70}行")
        jobs = [RubyJob(base, "JP.Ruby", [RubyPiece(0, 1, "こう"), RubyPiece(1, 2, "こう")], offset=offset)
                for offset in (0, 12, -12)]
        with Measurer(ass, cache_path=self.measure_cache) as measurer:
            RubyGenerator(ass, measurer).run(jobs)
        for job in jobs[1:]:
            for original, shifted in zip(jobs[0].geoms, job.geoms):
                self.assertAlmostEqual(shifted["y"] - original["y"], -job.offset / 2)

    def test_zero_soma_copy_bytes_and_geometry(self):
        path = self.path / "soma"
        shutil.copytree(ROOT / "test_projects/soma", path, copy_function=shutil.copyfile)
        project = Project.open(path)
        episode = project.episode(project.episode_ids()[0])
        before, before_rep = episode.build_export()
        fonts, pal = T.scheme_from_dict(project.data["template"].get("scheme", {}))
        project.data["template"]["scheme"] = T.scheme_dict(replace(fonts, ruby_offset=0, pinyin_offset=0), pal)
        after, after_rep = episode.build_export()
        self.assertEqual(before.serialize().encode("utf-8"), after.serialize().encode("utf-8"))
        self.assertEqual(before_rep.owners, after_rep.owners)
        print("TP1 soma copy zero offset UTF8 SHA256=" + hashlib.sha256(after.serialize().encode("utf-8")).hexdigest())


if __name__ == "__main__":
    unittest.main()
