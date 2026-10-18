"""Separate word ruby stays centred before collision fallback, through real exports."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.ass.document import AssDocument, Event, Style
from subkit.ass.text import VisibleText, line_tag
from subkit.core import karaoke
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.ops import Editor
from subkit.core.readings import anchor_of, visible_map
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.model.serialize import document_to_json
from subkit.reading.engine import Entry, Part
from subkit.ruby import generate as generator
from subkit.ruby.generate import RubyGenerator, RubyJob, RubyPiece
from subkit.ruby.measure import Box


WORDS = ((0, 2, "代表", "だいひょう"),
         (2, 5, "取締役", "とりしまりやく"),
         (5, 7, "社長", "しゃちょう"))


def fake_job(*, text="代表取締役社長", pieces=None, widths=(94, 158, 100), percent=150,
             prefer_center=True, vertical=False):
    """Known 40 px base cells and independently measured ruby ink, without ffmpeg."""
    doc = AssDocument()
    doc.script_info = [("PlayResX", "1920"), ("PlayResY", "1080")]
    doc.styles = [Style.new("JP", Fontsize=48, Fontname="@MS Gothic" if vertical else "MS Gothic",
                            Angle=270 if vertical else 0, Outline=1, Shadow=0),
                  Style.new("JP.Ruby", Fontsize=24, Fontname="MS Gothic", Outline=1, Shadow=0)]
    gen = RubyGenerator(doc, SimpleNamespace(playres_x=1920, playres_y=1080,
                                            anchor_x=lambda *args: 960))
    gen._metrics = {"JP": (60, 5, 50), "JP.Ruby": (30, 2, 25)}
    if pieces is None:
        pieces = [RubyPiece(a, b, reading, tag=f"word-{index}")
                  for index, (a, b, _surface, reading) in enumerate(WORDS)]
    job = RubyJob(Event(style="JP", text=text), "JP.Ruby", pieces,
                  max_width_percent=percent, prefer_center=prefer_center)
    boxes, column, row = [], 0, 0
    for char in VisibleText(text).text:
        if char == "\n":
            boxes.append(None)
            column, row = 0, row + 1
        else:
            boxes.append(Box(900 + column * 40, 940 + column * 40, 5 + row * 60, 50 + row * 60))
            column += 1
    job.base_advances = {(piece.start, piece.end): 40 * (piece.end - piece.start) for piece in pieces}
    rubies = [Box(960 - width / 2, 960 + width / 2, 2, 26) for width in widths]
    captured = []
    original = generator.layout_line

    def record(items, **kwargs):
        captured.append(items)
        return original(items, **kwargs)

    with patch.object(generator, "layout_line", side_effect=record):
        gen._layout_job(job, boxes, rubies)
    return job, boxes, rubies, captured


def template():
    doc = AssDocument()
    doc.script_info = [("ScriptType", "v4.00+"), ("PlayResX", "1920"),
                       ("PlayResY", "1080"), ("WrapStyle", "2")]
    doc.styles = [Style.new("JP", Fontname="MS Gothic", Fontsize=60, Outline=1, Shadow=0,
                            Alignment=2, MarginV=100),
                  Style.new("JP.Ruby", Fontname="MS Gothic", Fontsize=30, Outline=1, Shadow=0,
                            Alignment=2),
                  Style.new("CN", Fontname="MS Gothic", Fontsize=60, Outline=1, Shadow=0,
                            Alignment=2, MarginV=180)]
    return doc


def document(*, status="confirmed", combined=False, kind="dialogue"):
    doc = Document("ruby-word-boundaries")
    text = "代表取締役社長の"
    member = Member("m", "g", "ja", "original", kind, 1000, 5000,
                    style="JP", runs=[Run(text, ins="i", b=len(text))])
    doc.groups = [Group("g", kind, ["m"])]
    doc.members["m"] = member
    ids = visible_map(member)[1]
    entries = [(0, 7, text[:7], "".join(word[3] for word in WORDS))] if combined else WORDS
    for index, (a, b, surface, reading) in enumerate(entries):
        parts = [[x, y, value] for x, y, _word, value in WORDS] if combined else [[0, b - a, reading]]
        ann = Annotation(f"a{index}", "ruby", "m", anchor_of(ids[a:b]), surface, reading,
                         status=status, origin="analyzer" if status == "auto" else "manual", parts=parts)
        doc.annotations[ann.id] = ann
    return doc


class WordBoundaryGeometry(unittest.TestCase):
    def assert_centered(self, job, boxes):
        for piece, x, _scale, _overlap in job.placements:
            span = [box for box in boxes[piece.start:piece.end] if box is not None]
            self.assertAlmostEqual(x, (min(box.x0 for box in span) + max(box.x1 for box in span)) / 2,
                                   places=5)

    def assert_gap(self, items, gap=7.2):
        for left, right in zip(items, items[1:]):
            self.assertGreaterEqual(right.x - right.w() / 2 - left.x - left.w() / 2, gap - 0.001)

    def test_three_adjacent_words_keep_own_centres_and_distinct_compression(self):
        job, boxes, _ruby, lines = fake_job()
        self.assert_centered(job, boxes)
        self.assert_gap(lines[0])
        scales = [placement[2] for placement in job.placements]
        self.assertTrue(all(60 < scale < 100 for scale in scales), scales)
        self.assertGreater(len({round(scale, 4) for scale in scales}), 1, scales)
        self.assertFalse(any(placement[3] for placement in job.placements))

    def test_explicit_word_identity_overrides_owner_tag(self):
        pieces = [RubyPiece(a, b, reading, tag="one-owner", word=index)
                  for index, (a, b, _surface, reading) in enumerate(WORDS)]
        job, boxes, _ruby, lines = fake_job(pieces=pieces)
        self.assert_centered(job, boxes)
        self.assert_gap(lines[0])
        self.assertTrue(all(placement[2] < 100 for placement in job.placements))

    def test_unknown_words_keep_legacy_shift_and_common_compression(self):
        pieces = [RubyPiece(a, b, reading) for a, b, _surface, reading in WORDS]
        job, _boxes, _ruby, lines = fake_job(pieces=pieces)
        self.assertTrue(any(abs(item.x - item.target) > 1 for item in lines[0]))
        self.assertLessEqual(max(placement[2] for placement in job.placements)
                             - min(placement[2] for placement in job.placements), 0.01)

    def test_noncolliding_parts_of_same_word_form_one_centred_item(self):
        same_word = [RubyPiece(a, b, reading, tag="one-annotation")
                     for a, b, _surface, reading in WORDS]
        grouped, _boxes, _ruby, lines = fake_job(pieces=same_word, widths=(40, 80, 40))
        self.assertEqual(len(lines[0]), 1)
        self.assertAlmostEqual(lines[0][0].x, 1040, places=6)
        self.assertEqual([p[2] for p in grouped.placements], [100, 100, 100])
        self.assertEqual(len(grouped.out), 3)

    def test_legacy_policy_ignores_known_words(self):
        known, *_rest = fake_job(prefer_center=False)
        pieces = [RubyPiece(a, b, reading) for a, b, _surface, reading in WORDS]
        unknown, *_rest = fake_job(pieces=pieces, prefer_center=False)
        self.assertEqual([p[1:] for p in known.placements], [p[1:] for p in unknown.placements])
        self.assertEqual([event.text for event in known.out], [event.text for event in unknown.out])

    def test_vertical_ruby_ignores_word_boundaries(self):
        known, *_rest = fake_job(vertical=True)
        pieces = [RubyPiece(a, b, reading) for a, b, _surface, reading in WORDS]
        unknown, *_rest = fake_job(pieces=pieces, vertical=True)
        self.assertEqual([p[1:] for p in known.placements], [p[1:] for p in unknown.placements])
        self.assertEqual([event.text for event in known.out], [event.text for event in unknown.out])

    def test_short_reading_lends_spare_width_without_unnecessary_compression(self):
        pieces = [RubyPiece(0, 2, "だいひょう", tag="a"), RubyPiece(2, 5, "やく", tag="b")]
        job, boxes, _ruby, lines = fake_job(text="代表取締役", pieces=pieces, widths=(120, 40))
        self.assert_centered(job, boxes)
        self.assertEqual([placement[2] for placement in job.placements], [100, 100])
        self.assert_gap(lines[0])

    def test_one_long_reading_uses_its_own_scale_and_short_neighbour_stays_full(self):
        pieces = [RubyPiece(0, 2, "だいひょう", tag="a"), RubyPiece(2, 5, "やく", tag="b")]
        job, boxes, _ruby, lines = fake_job(text="代表取締役", pieces=pieces, widths=(180, 40), percent=None)
        self.assert_centered(job, boxes)
        self.assertGreater(job.placements[0][2], 60)
        self.assertLess(job.placements[0][2], 100)
        self.assertEqual(job.placements[1][2], 100)
        self.assert_gap(lines[0])

    def test_disabling_global_width_cap_retains_word_collision_rules(self):
        job, boxes, _ruby, lines = fake_job(percent=None)
        self.assert_centered(job, boxes)
        self.assert_gap(lines[0])
        self.assertTrue(all(60 <= placement[2] < 100 for placement in job.placements))

    def test_literal_long_vowel_separates_bases_without_being_replaced_or_annotated(self):
        pieces = [RubyPiece(0, 2, "だいひょう", tag="a"), RubyPiece(3, 5, "しゃちょう", tag="b")]
        job, boxes, _ruby, lines = fake_job(text="代表ー社長", pieces=pieces, widths=(90, 90))
        self.assert_centered(job, boxes)
        self.assertEqual([placement[2] for placement in job.placements], [100, 100])
        self.assertEqual(job.event.text, "代表ー社長")
        self.assert_gap(lines[0])

    def test_newline_does_not_make_separate_visual_lines_collide(self):
        pieces = [RubyPiece(0, 2, "だいひょう", tag="a"), RubyPiece(3, 5, "しゃちょう", tag="b")]
        job, boxes, _ruby, lines = fake_job(text=r"代表\N社長", pieces=pieces, widths=(110, 110))
        self.assert_centered(job, boxes)
        self.assertEqual(len(lines), 2)
        self.assertEqual([placement[2] for placement in job.placements], [100, 100])
        self.assertNotEqual(job.geoms[0]["y"], job.geoms[1]["y"])

    def test_manual_scale_survives_impossible_centred_pair_and_fallback_separates(self):
        pieces = [RubyPiece(0, 2, "だいひょう", scale=125, tag="a"),
                  RubyPiece(2, 5, "とりしまりやく", tag="b")]
        job, _boxes, _ruby, lines = fake_job(text="代表取締役", pieces=pieces, widths=(180, 180))
        self.assertEqual(job.placements[0][2], 125)
        self.assertGreaterEqual(job.placements[1][2], 60)
        self.assert_gap(lines[0])
        self.assertTrue(any(abs(item.x - item.target) > 0.1 for item in lines[0]))

    def test_automatic_floor_is_never_broken_to_preserve_centres(self):
        job, _boxes, _ruby, lines = fake_job(widths=(350, 350, 350), percent=None)
        self.assertTrue(all(placement[2] >= 60 for placement in job.placements))
        self.assert_gap(lines[0])
        self.assertTrue(any(abs(item.x - item.target) > 0.1 for item in lines[0]))

    def test_emitted_scale_never_rounds_up_into_the_reserved_word_gap(self):
        job, _boxes, rubies, _lines = fake_job(percent=None)
        emitted = []
        for placement, ruby, event in zip(job.placements, rubies, job.out):
            scale_tag = line_tag(event.text, "fscx")
            scale = scale_tag.num() if scale_tag is not None else 100
            self.assertLessEqual(scale, placement[2] + 1e-7)
            position = line_tag(event.text, "pos").float_args()[0]
            emitted.append((position, (ruby.width + 2) * scale / 100))
        for (left_x, left_width), (right_x, right_width) in zip(emitted, emitted[1:]):
            self.assertGreaterEqual(right_x - right_width / 2 - left_x - left_width / 2, 7.2 - 0.001)


class WordBoundaryExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ruby_word_boundaries_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def build(self, doc, *, mode="release", percent=150, cache=None, order="end"):
        settings = {kind: {"enabled": percent is not None, "percent": percent or 150}
                    for kind in ("dialogue", "song")}
        captured = []
        original = RubyGenerator._layout_job

        def record(gen, job, boxes, ruby_boxes):
            original(gen, job, boxes, ruby_boxes)
            captured.append((job, copy.deepcopy(boxes), copy.deepcopy(ruby_boxes)))

        with patch.object(RubyGenerator, "_layout_job", record):
            out, report = build_ass(doc, {}, template(), self.root,
                                    ExportOptions(mode=mode, ruby_max_width=settings, annotation_order=order),
                                    cache=cache, measure_cache=self.root / "measure.sqlite3")
        return out, report, captured

    def assert_real_word_centres(self, captured):
        job, boxes, _ruby = captured[0]
        self.assertEqual(len(job.placements), 3)
        for piece, x, scale, _overlap in job.placements:
            span = [box for box in boxes[piece.start:piece.end] if box is not None]
            self.assertAlmostEqual(x, (min(box.x0 for box in span) + max(box.x1 for box in span)) / 2,
                                   places=4)
            self.assertGreaterEqual(scale, 60)
        return job

    def test_confirmed_annotations_export_as_independently_centred_words(self):
        doc = document()
        before = copy.deepcopy(document_to_json(doc))
        _out, report, captured = self.build(doc)
        job = self.assert_real_word_centres(captured)
        self.assertEqual(report.ruby_lines, 3)
        self.assertEqual([piece.tag for piece in job.pieces], list(doc.annotations))
        self.assertGreater(len({round(placement[2], 3) for placement in job.placements}), 1)
        self.assertEqual(document_to_json(doc), before)

    def test_automatic_annotations_have_same_geometry_as_confirmed_annotations(self):
        auto = document(status="auto")
        before = copy.deepcopy(document_to_json(auto))
        _out, _report, auto_jobs = self.build(auto)
        _out, _report, manual_jobs = self.build(document())
        self.assert_real_word_centres(auto_jobs)
        self.assertEqual([p[1:] for p in auto_jobs[0][0].placements],
                         [p[1:] for p in manual_jobs[0][0].placements])
        self.assertEqual(document_to_json(auto), before)

    def test_one_annotation_parts_keep_one_word_identity(self):
        doc = document(combined=True)
        _out, _report, captured = self.build(doc)
        job = captured[0][0]
        self.assertEqual(len(job.pieces), 3)
        self.assertEqual({piece.tag for piece in job.pieces}, {"a0"})
        self.assertTrue(all(piece.word is None or piece.word == job.pieces[0].word for piece in job.pieces))
        self.assertEqual(len({p[2] for p in job.placements}), 1)
        boxes, rubies = captured[0][1:]
        left = job.placements[0][1] - (rubies[0].width + 2) * job.placements[0][2] / 200
        right = job.placements[-1][1] + (rubies[-1].width + 2) * job.placements[-1][2] / 200
        self.assertAlmostEqual((left + right) / 2, (boxes[0].x0 + boxes[6].x1) / 2, places=5)

    def test_song_internal_parts_keep_beats_while_sharing_one_word_layout(self):
        doc = document(combined=True, kind="song")
        beats = karaoke.derive(doc, doc.members["m"])
        for index, beat in enumerate(beats):
            beat.t0, beat.t1, beat.by = 1100 + index * 180, 1280 + index * 180, "tap"
        Editor(doc, {}).set_karaoke("m", beats)
        before = copy.deepcopy(document_to_json(doc))
        _out, _report, captured = self.build(doc)
        job = captured[0][0]
        self.assertEqual(len(job.pieces), 3)
        self.assertEqual(len({p[2] for p in job.placements}), 1)
        self.assertTrue(all(piece.k_text for piece in job.pieces))
        self.assertTrue(all(r"\k" in event.text for event in job.out))
        self.assertEqual(document_to_json(doc), before)

    def test_preview_release_and_cached_preview_preserve_positions_and_input(self):
        doc = document()
        before = copy.deepcopy(document_to_json(doc))
        cache = ExportCache()
        preview, preview_report, captured = self.build(doc, mode="preview", cache=cache)
        cached, cached_report, _capture = self.build(doc, mode="preview", cache=cache)
        release, release_report, release_jobs = self.build(doc)
        self.assert_real_word_centres(captured)
        self.assertEqual(preview.serialize(), cached.serialize())
        self.assertEqual(cached_report.laid_out, 0)
        self.assertEqual(preview.styles, release.styles)
        self.assertEqual(preview.events, release.events)
        self.assertEqual(preview_report.owners, release_report.owners)
        self.assertEqual(captured[0][0].geoms, release_jobs[0][0].geoms)
        self.assertEqual(document_to_json(doc), before)

    def test_annotation_output_order_does_not_change_ruby_geometry(self):
        doc = document()
        end, _report, jobs_end = self.build(doc)
        after, _report, jobs_after = self.build(doc, order="after_parent")
        self.assertEqual(jobs_end[0][0].geoms, jobs_after[0][0].geoms)
        self.assertCountEqual(end.events, after.events)

    def test_real_export_word_spacing_remains_when_global_cap_is_disabled(self):
        _out, _report, captured = self.build(document(), percent=None)
        job = self.assert_real_word_centres(captured)
        self.assertTrue(any(placement[2] < 100 for placement in job.placements))
        self.assertEqual(job.base_advances, {})

    def test_song_karaoke_preserves_beats_source_and_per_word_centres(self):
        doc = document(kind="song")
        beats = karaoke.derive(doc, doc.members["m"])
        for index, beat in enumerate(beats):
            beat.t0, beat.t1, beat.by = 1100 + index * 180, 1280 + index * 180, "tap"
        Editor(doc, {}).set_karaoke("m", beats)
        before = copy.deepcopy(document_to_json(doc))
        out, report, captured = self.build(doc)
        job = self.assert_real_word_centres(captured)
        self.assertTrue(out.events)
        self.assertGreater(report.karaoke_lines, 0)
        self.assertTrue(all(piece.k_text for piece in job.pieces))
        self.assertTrue(all(r"\k" in event.text for event in job.out))
        self.assertEqual(document_to_json(doc), before)

    def test_workfile_entry_parts_keep_word_identity(self):
        from subkit.workfile import _pieces

        entries = [Entry(0, 4, "取り扱い", "とりあつかい",
                         [Part(0, 1, "と"), Part(2, 3, "あつか")], "record"),
                   Entry(5, 7, "代表", "だいひょう", [Part(5, 7, "だいひょう")], "record")]
        pieces = _pieces(entries)
        self.assertEqual(len(pieces), 3)
        self.assertIsNotNone(pieces[0].word)
        self.assertEqual(pieces[0].word, pieces[1].word)
        self.assertNotEqual(pieces[1].word, pieces[2].word)


if __name__ == "__main__":
    unittest.main()
