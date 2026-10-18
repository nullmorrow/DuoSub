"""Horizontal furigana centring, retained collision limits and real export entry points."""
from __future__ import annotations

import copy
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subkit import templates as T
from subkit import workfile
from subkit.ass.document import AssDocument, Event, Style
from subkit.ass.text import VisibleText, line_tag, remove_tags_named
from subkit.config import Config, DEFAULTS
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.pinyin_notes import annotate_pinyin
from subkit.core.readings import anchor_of, visible_map
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.model.serialize import document_to_json
from subkit.proc import run
from subkit.project import Episode as WorkEpisode
from subkit.reading.engine import Entry, Part
from subkit.ruby import generate as generator
from subkit.ruby.generate import RubyGenerator, RubyJob, RubyPiece
from subkit.ruby.measure import Box, Measurer
from tests.ruby_policy import legacy_ruby_layout


def fake_job(text="状況に", span=(0, 2), *, width=140, scale=None, percent=150,
             prefer_center=True, vertical=False, offset=0, pieces=None, widths=None):
    doc = AssDocument()
    doc.script_info = [("PlayResX", "1920"), ("PlayResY", "1080")]
    doc.styles = [Style.new("Default", Fontsize=48, Fontname="@MS Gothic" if vertical else "MS Gothic",
                            Angle=270 if vertical else 0, Outline=1, Shadow=0),
                  Style.new("Default.Ruby", Fontsize=24, Fontname="MS Gothic", Outline=1, Shadow=0)]
    measurer = SimpleNamespace(playres_x=1920, playres_y=1080, anchor_x=lambda *args: 960)
    gen = RubyGenerator(doc, measurer)
    gen._metrics = {"Default": (60, 5, 50), "Default.Ruby": (30, 2, 25)}
    parts = pieces or [RubyPiece(*span, "じょうきょう", scale=scale)]
    job = RubyJob(Event(style="Default", text=text), "Default.Ruby", parts,
                  max_width_percent=percent, prefer_center=prefer_center, offset=offset)
    boxes = [Box(900 + i * 40, 940 + i * 40, 5, 50) for i in range(len(text))]
    ruby_boxes = [Box(960 - w / 2, 960 + w / 2, 2, 26) for w in (widths or [width] * len(parts))]
    limits = []
    real_layout = generator.layout_line

    def layout(items, **kwargs):
        limits.extend((item.left_limit, item.right_limit) for item in items)
        return real_layout(items, **kwargs)

    with patch.object(generator, "layout_line", side_effect=layout), patch.object(gen, "_limit", wraps=gen._limit) as limit:
        gen._layout_job(job, boxes, ruby_boxes)
    return job, boxes, ruby_boxes, limits, limit.call_count


def actual_template():
    doc = AssDocument()
    doc.script_info = [("ScriptType", "v4.00+"), ("PlayResX", "1920"), ("PlayResY", "1080"), ("WrapStyle", "2")]
    doc.styles = [Style.new("JP", Fontname="MS Gothic", Fontsize=72, Outline=0, Shadow=0,
                            Alignment=2, MarginV=100, PrimaryColour="&H0000FF00"),
                  Style.new("JP.Ruby", Fontname="MS Gothic", Fontsize=54, Outline=0, Shadow=0,
                            Alignment=2, PrimaryColour="&H000000FF"),
                  Style.new("CN", Fontname="MS Gothic", Fontsize=72, Outline=0, Shadow=0, Alignment=2, MarginV=220),
                  Style.new("CN.Pinyin", Fontname="Arial", Fontsize=32, Outline=0, Shadow=0, Alignment=2)]
    return doc


def document(cases=(("状況に", 0, 2, "じょうきょう"),)):
    doc = Document("ruby-centering")
    for i, (text, start, end, reading) in enumerate(cases):
        mid, gid = f"m{i}", f"g{i}"
        member = Member(mid, gid, "ja", "original", "dialogue", 1000 + i * 5000, 5000 + i * 5000,
                        style="JP", runs=[Run(text, ins=f"i{i}", b=len(text))])
        doc.groups.append(Group(gid, "dialogue", [mid]))
        doc.members[mid] = member
        ann = Annotation(f"a{i}", "ruby", mid, anchor_of(visible_map(member)[1][start:end]),
                         text[start:end], reading, status="confirmed", origin="manual",
                         parts=[[0, end - start, reading]])
        doc.annotations[ann.id] = ann
    return doc


class HorizontalRubyPlacement(unittest.TestCase):
    def test_kana_punctuation_unannotated_kanji_and_edges_keep_the_same_center(self):
        for text, span in (("状況に", (0, 2)), ("い状況", (1, 3)), ("状況！", (0, 2)),
                           ("状況語", (0, 2)), ("語状況", (1, 3)), ("状況", (0, 2)),
                           ("前のいろいろ状況に続く", (6, 8))):
            with self.subTest(text=text):
                job, boxes, _ruby, bounds, calls = fake_job(text, span)
                expected = (boxes[span[0]].x0 + boxes[span[1] - 1].x1) / 2
                self.assertAlmostEqual(job.placements[0][1], expected)
                self.assertEqual(bounds, [(float("-inf"), float("inf"))])
                self.assertEqual(calls, 0)

    def test_short_long_and_floor_limited_ruby_stay_centered(self):
        for width in (40, 100, 140, 240, 400):
            with self.subTest(width=width):
                job, _boxes, _ruby, _bounds, _calls = fake_job(width=width)
                self.assertAlmostEqual(job.placements[0][1], 940)
                self.assertGreaterEqual(job.placements[0][2], 60)

    def test_width_ceiling_and_compression_floor_are_still_enforced(self):
        for percent, expected in ((100, 60), (120, 80 * 120 / 140), (150, 80 * 150 / 140), (200, 100)):
            with self.subTest(percent=percent):
                job, _boxes, _ruby, _bounds, _calls = fake_job(percent=percent)
                self.assertAlmostEqual(job.placements[0][2], expected)
                self.assertAlmostEqual(job.placements[0][1], 940)
        uncapped, *_rest = fake_job(percent=None)
        self.assertEqual(uncapped.placements[0][2], 100)

    def test_manual_scale_bypasses_automatic_cap_and_keeps_center(self):
        for scale in (42, 125):
            with self.subTest(scale=scale):
                job, _boxes, _ruby, _bounds, _calls = fake_job(percent=100, scale=scale)
                self.assertEqual(job.placements[0][2], scale)
                self.assertAlmostEqual(job.placements[0][1], 940)
                self.assertFalse(any("振假名太长" in warning for warning in job.warnings))

    def test_actual_adjacent_ruby_collision_is_still_separated(self):
        parts = [RubyPiece(0, 2, "じょうきょう"), RubyPiece(2, 4, "きょうふ")]
        job, _boxes, rb, _bounds, _calls = fake_job("状況恐怖", pieces=parts, widths=[150, 150])
        first, second = job.placements
        widths = [(box.width + 2) * placed[2] / 100 for box, placed in zip(rb, (first, second))]
        self.assertLessEqual(first[1] + widths[0] / 2 + 2, second[1] - widths[1] / 2 + 0.5)
        self.assertTrue(all(60 <= placed[2] <= 80 for placed in (first, second)))
        self.assertTrue(first[1] != 940 or second[1] != 1020)

    def test_vertical_delta_changes_only_y_for_horizontal_ruby(self):
        jobs = [fake_job(offset=delta)[0] for delta in (-12, 0, 12)]
        for job, delta in zip(jobs, (-12, 0, 12)):
            self.assertAlmostEqual(job.geoms[0]["x"], jobs[1].geoms[0]["x"])
            self.assertAlmostEqual(job.geoms[0]["y"] - jobs[1].geoms[0]["y"], -delta)
            self.assertEqual(job.placements[0][2], jobs[1].placements[0][2])

    def test_default_false_keeps_old_neighbor_limits(self):
        self.assertFalse(RubyJob(Event(), "Ruby", []).prefer_center)
        old, _boxes, _ruby, bounds, calls = fake_job("状況語", prefer_center=False)
        centered, *_rest = fake_job("状況語")
        self.assertEqual(calls, 2)
        self.assertTrue(all(np.isfinite(value) for value in bounds[0]))
        self.assertLess(old.placements[0][1], centered.placements[0][1])

    def test_vertical_prefer_center_keeps_the_same_legacy_limits_and_events(self):
        old, _boxes, _ruby, old_bounds, old_calls = fake_job("状況語", vertical=True, prefer_center=False)
        new, _boxes, _ruby, new_bounds, new_calls = fake_job("状況語", vertical=True, prefer_center=True)
        self.assertEqual((old_calls, new_calls), (2, 2))
        self.assertEqual(new_bounds, old_bounds)
        self.assertEqual(new.placements, old.placements)
        self.assertEqual(new.geoms, old.geoms)
        self.assertEqual([event.text for event in new.out], [event.text for event in old.out])


class RubyCenteringEntryPoints(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="duosub_ruby_centering_")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def recorder(self, captured):
        original = RubyGenerator._layout_job

        def layout(gen, job, boxes, ruby_boxes):
            original(gen, job, boxes, ruby_boxes)
            captured.append((job, copy.deepcopy(boxes), copy.deepcopy(ruby_boxes)))

        return layout

    def assert_measured_centers(self, captured):
        count = 0
        for job, boxes, _ruby_boxes in captured:
            if not job.prefer_center:
                continue
            for piece, x, _scale, _overlap in job.placements:
                used = [box for box in boxes[piece.start:piece.end] if box is not None]
                expected = (min(box.x0 for box in used) + max(box.x1 for box in used)) / 2
                self.assertAlmostEqual(x, expected, delta=0.01)
                count += 1
        self.assertGreater(count, 0)

    def build(self, doc, *, mode="preview", cache=None, offset=0, pinyin="off"):
        return build_ass(doc, {}, actual_template(), self.path,
                         ExportOptions(mode=mode, ruby_offset=offset, pinyin_mode=pinyin), cache=cache,
                         measure_cache=self.path / "render.sqlite3")

    def test_real_preview_release_cache_and_varied_neighbors_center_over_measured_base(self):
        cases = (("状況に", 0, 2, "じょうきょう"), ("い状況", 1, 3, "じょうきょう"),
                 ("状況！", 0, 2, "じょうきょう"), ("状況語", 0, 2, "じょうきょう"),
                 ("語状況", 1, 3, "じょうきょう"), ("状況", 0, 2, "じょうきょう"))
        doc = document(cases)
        before = copy.deepcopy(document_to_json(doc))
        captured, cache = [], ExportCache()
        with patch.object(RubyGenerator, "_layout_job", self.recorder(captured)):
            preview, preview_report = self.build(doc, cache=cache)
        self.assertTrue(all(job.prefer_center for job, _boxes, _ruby in captured))
        self.assertEqual(len(captured), len(cases))
        self.assert_measured_centers(captured)
        cached, cached_report = self.build(doc, cache=cache)
        self.assertEqual(cached_report.laid_out, 0)
        self.assertEqual(cached.serialize(), preview.serialize())
        release, release_report = self.build(doc, mode="release")
        self.assertEqual([event.text for event in release.events], [event.text for event in preview.events])
        self.assertEqual(release_report.ruby_lines, preview_report.ruby_lines)
        self.assertEqual(document_to_json(doc), before)

    def test_export_pinyin_and_gloss_retain_default_policy(self):
        doc = document()
        member = doc.members["m0"]
        member.runs[0].text += "{@注:に=注釈}"
        member.runs[0].b = len(member.runs[0].text)
        cn = Member("cn", "g0", "zh-Hans", "translation", "dialogue", 1000, 5000,
                    style="CN", runs=[Run("情况", ins="i_cn", b=2)])
        doc.members[cn.id] = cn
        doc.groups[0].members.append(cn.id)
        annotate_pinyin(doc, cn)
        captured = []
        with patch.object(RubyGenerator, "_layout_job", self.recorder(captured)):
            output, report = self.build(doc, pinyin="PY")
        kinds = {job.ruby_style.rsplit(".", 1)[-1]: job.prefer_center for job, _boxes, _ruby in captured}
        self.assertIs(kinds["Ruby"], True)
        self.assertIs(kinds["Pinyin"], False)
        # Core glosses use their independent layout path, not RubyJob. Compare
        # actual gloss output with old-policy ruby instead of expecting a job.
        gloss_lines = lambda ass, rep: [event.to_line(ass.event_format)
            for event, owner in zip(ass.events, rep.owners) if owner["kind"] == "gloss"]
        self.assertTrue(gloss_lines(output, report))
        with legacy_ruby_layout():
            baseline, baseline_report = self.build(doc, pinyin="PY")
        self.assertEqual(gloss_lines(output, report), gloss_lines(baseline, baseline_report))

    def test_real_export_horizontal_x_is_independent_of_delta(self):
        doc = document()
        collected = []
        for offset in (-9, 0, 9):
            captured = []
            with patch.object(RubyGenerator, "_layout_job", self.recorder(captured)):
                self.build(doc, offset=offset)
            job = next(job for job, _boxes, _ruby in captured if job.prefer_center)
            collected.append(job.geoms[0])
        for geom, offset in zip(collected, (-9, 0, 9)):
            self.assertEqual(geom["x"], collected[1]["x"])
            self.assertAlmostEqual(geom["y"] - collected[1]["y"], -offset)

    def test_template_preview_ruby_uses_center_policy_and_pinyin_does_not(self):
        fonts = T.FontSet("center-test", "test", "MS Gothic", "MS Gothic", "MS Gothic", pinyin_font="Arial")
        styles = T.build_styles(fonts, T.palette("classic"))
        captured = []
        samples = {"centering-test": [("JP", "test"), ("CN", "test")]}

        def sentence(lang, _sample):
            return ("状況に", [(0, 2, "じょうきょう")]) if lang == "ja" else ("情况", [])

        with patch.dict(T._SCENE_LINES, samples), patch.object(T, "preview_sentence", side_effect=sentence), \
                patch.object(RubyGenerator, "_layout_job", self.recorder(captured)):
            lines = T.scene_events(styles, "centering-test", pinyin=True, measure_cache=self.path / "template.sqlite3")
        self.assertTrue(lines)
        self.assertTrue(any(job.ruby_style.endswith(".Ruby") and job.prefer_center for job, _boxes, _ruby in captured))
        self.assertTrue(any(job.ruby_style.endswith(".Pinyin") and not job.prefer_center for job, _boxes, _ruby in captured))
        self.assert_measured_centers(captured)

    def test_legacy_workfile_refresh_ruby_centers_but_pinyin_and_gloss_keep_policy(self):
        cfg = Config(copy.deepcopy(DEFAULTS), self.path)
        cfg.data["tools"]["global_dict"] = str(self.path / "global.json")
        cfg.data["pinyin"]["enabled"] = True
        project = SimpleNamespace(config=cfg, cache_dir=self.path / "cache")
        episode = WorkEpisode(1, 1, project)
        episode.work_dir.mkdir(parents=True)
        work = actual_template()
        work.events = [Event(start=100, end=500, style="JP", effect="#1", text="状況に{@注:に=注釈}"),
                       Event(start=100, end=500, style="CN", effect="#1", text="情况")]
        work.save(episode.work_ass)
        fake_engine = SimpleNamespace(analyze=lambda *_args, **_kwargs: [
            Entry(0, 2, "状況", "じょうきょう", [Part(0, 2, "じょうきょう")], "analyzer")])
        captured, measurers = [], []

        def measurer(*args, **kwargs):
            instance = Measurer(*args, **kwargs)
            measurers.append(instance)
            return instance

        try:
            with patch.object(workfile, "ReadingEngine", return_value=fake_engine), \
                    patch.object(workfile, "Measurer", side_effect=measurer), \
                    patch.object(RubyGenerator, "_layout_job", self.recorder(captured)):
                workfile.refresh(episode, log=lambda _message: None)
        finally:
            for instance in measurers:
                instance.close()
        policies = {job.effect.rsplit(":", 1)[-1]: job.prefer_center for job, _boxes, _ruby in captured}
        self.assertEqual(policies, {"ruby": True, "gloss": False, "py": False})
        self.assert_measured_centers(captured)
        refreshed = AssDocument.load(episode.work_ass)
        self.assertTrue(any(event.effect == "#1:ruby" for event in refreshed.events))

    def test_soma_default_policy_changes_only_ruby_horizontal_geometry(self):
        from subkit.core.workspace import Project
        from tests.ruby_policy import legacy_ruby_layout

        source = Path(__file__).resolve().parents[1] / "test_projects/soma"
        target = self.path / "soma"
        shutil.copytree(source, target, copy_function=shutil.copyfile)
        project = Project.open(target)
        episode = project.episode(project.episode_ids()[0])
        saved = copy.deepcopy(document_to_json(episode.document))
        with legacy_ruby_layout():
            old, old_report = episode.build_export(include_actor=False)
        new, new_report = episode.build_export(include_actor=False)
        old_head, new_head = old.copy(), new.copy()
        old_head.events, new_head.events = [], []
        self.assertEqual(new_head.serialize(), old_head.serialize())
        # Ruby hit owners carry the same horizontal x/width that is supposed
        # to change. Keep identity, range, y and height strict.
        stable_owner = lambda owner: {key: value for key, value in owner.items()
            if owner["kind"] != "ruby" or key not in ("x", "w")}
        self.assertEqual([stable_owner(owner) for owner in new_report.owners],
                         [stable_owner(owner) for owner in old_report.owners])
        self.assertEqual(len(new.events), len(old.events))
        changed = 0
        for before, after, owner in zip(old.events, new.events, new_report.owners):
            if owner["kind"] != "ruby":
                self.assertEqual(after.to_line(new.event_format), before.to_line(old.event_format))
                continue
            self.assertEqual(VisibleText(after.text).text, VisibleText(before.text).text)
            for name in ("pos", "move"):
                a, b = line_tag(before.text, name), line_tag(after.text, name)
                self.assertEqual(a is None, b is None)
                if a is not None:
                    aa, bb = a.float_args(), b.float_args()
                    self.assertEqual(aa[1::2] if name == "pos" else [aa[1], aa[3], *aa[4:]],
                                     bb[1::2] if name == "pos" else [bb[1], bb[3], *bb[4:]])
                    if name == "move":
                        self.assertAlmostEqual(aa[2] - aa[0], bb[2] - bb[0])
            strip = lambda event: event.copy(text=remove_tags_named(event.text, {"pos", "move", "fscx"}))
            self.assertEqual(strip(after).to_line(new.event_format), strip(before).to_line(old.event_format))
            changed += before.text != after.text
        self.assertGreater(changed, 0)
        self.assertEqual(document_to_json(episode.document), saved)


    def test_one_real_1080p_frame_centers_ruby_ink_over_its_kanji(self):
        ass, report = self.build(document())
        base = next(event.copy() for event, owner in zip(ass.events, report.owners) if owner["kind"] == "base")
        ruby = next(event for event, owner in zip(ass.events, report.owners) if owner["kind"] == "ruby")
        # Paint transparency keeps に's advance, so the original layout remains
        # intact while the green ink mask isolates only the annotated kanji.
        base.text = r"{\alpha&H00&}状況{\alpha&HFF&}に"
        ass.events = [base, ruby]
        ass.save(self.path / "center.ass")
        rendered = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "color=c=black:s=1920x1080:r=1:d=1", "-vf",
                        "setpts=PTS+1.5/TB,ass=center.ass,format=rgb24", "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       cwd=self.path, capture_output=True, timeout=30)
        self.assertEqual(rendered.returncode, 0, rendered.stderr.decode("utf-8", "replace"))
        frame = np.frombuffer(rendered.stdout, np.uint8).reshape(1080, 1920, 3)

        def center(channel):
            mask = (frame[:, :, channel] >= 128) & (frame[:, :, 1 - channel] < 64)
            _ys, xs = np.where(mask)
            self.assertGreater(xs.size, 0)
            return (int(xs.min()) + int(xs.max()) + 1) / 2

        self.assertAlmostEqual(center(0), center(1), delta=2)


if __name__ == "__main__":
    unittest.main()
