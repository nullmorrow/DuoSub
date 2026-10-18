"""Real libass advances, width caps, preserved centres and export contracts."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subkit.ass.document import AssDocument, Style
from subkit.ass.text import line_tag
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.readings import anchor_of, visible_map
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.model.serialize import document_to_json
from subkit.proc import CancelledError
from subkit.ruby.generate import RubyGenerator
from subkit.ruby.measure import Measurer


def styles():
    doc = AssDocument()
    doc.script_info = [("ScriptType", "v4.00+"), ("PlayResX", "1920"), ("PlayResY", "1080"), ("WrapStyle", "2")]
    doc.styles = [Style.new("JP", Fontname="MS Gothic", Fontsize=60, Outline=0, Shadow=0,
                            Alignment=2, MarginV=100),
                  Style.new("JP.Ruby", Fontname="MS Gothic", Fontsize=30, Outline=0, Shadow=0,
                            Alignment=2),
                  Style.new("Reset", Fontname="Arial", Fontsize=90, ScaleX=125, Spacing=3,
                            Outline=12, Shadow=9, PrimaryColour="&HFF000000", Bold=-1)]
    return doc


def document(cases=(("自", "みずか"), ("私", "わたし")), *, scale=None, kind="dialogue"):
    doc = Document("advance-contract")
    for index, (text, reading) in enumerate(cases):
        mid, gid = f"m{index}", f"g{index}"
        member = Member(mid, gid, "ja", "original", kind, 1000 + index * 5000, 5000 + index * 5000,
                        style="JP", runs=[Run(text, ins=f"i{index}", b=len(text))])
        doc.groups.append(Group(gid, kind, [mid]))
        doc.members[mid] = member
        ann = Annotation(f"a{index}", "ruby", mid, anchor_of(visible_map(member)[1]), text, reading,
                         status="confirmed", origin="manual", parts=[[0, len(text), reading]],
                         layout={} if scale is None else {"scale": scale})
        doc.annotations[ann.id] = ann
    return doc


class RealSpanAdvances(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ruby_advance_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.doc = styles()
        self.m = Measurer(self.doc, cache_path=self.root / "advance.sqlite3")
        self.addCleanup(self.m.close)

    def widths(self, *texts, style="JP"):
        return self.m.span_advances([(style, text, 0, len(text)) for text in texts])

    def test_narrow_kanji_and_wide_kanji_have_equal_advance_despite_different_ink(self):
        advances = self.widths("自", "私")
        ink = self.m.text_boxes([("JP", "自"), ("JP", "私")])
        self.assertTrue(all(value is not None and value > 0 for value in advances))
        self.assertAlmostEqual(advances[0], advances[1], delta=0.25)
        self.assertGreater(abs(ink[0].width - ink[1].width), 2)
        self.assertGreater(advances[0], ink[0].width)

    def test_multiple_kanji_include_each_character_cell(self):
        single, pair = self.widths("自", "自私")
        self.assertAlmostEqual(pair, 2 * single, delta=0.5)

    def test_style_rotation_does_not_change_unrotated_advance(self):
        angled = self.doc.style("JP").copy("Angled")
        angled.set("Angle", 45)
        self.doc.styles.append(angled)
        plain, rotated = self.m.span_advances([("JP", "自私", 0, 2), ("Angled", "自私", 0, 2)])
        self.assertIsNotNone(rotated)
        self.assertAlmostEqual(plain, rotated, delta=0.25)

    def test_preceding_and_following_text_do_not_change_the_selected_span(self):
        spans = [("JP", "自", 0, 1), ("JP", "前自後", 1, 2),
                 ("JP", r"前{\fs80}自{\fs15}後", 1, 2), ("JP", r"{\fs80}自", 0, 1)]
        bare, context, inherited, direct = self.m.span_advances(spans)
        self.assertEqual(bare, context)
        self.assertEqual(inherited, direct)
        self.assertGreater(inherited, bare)

    def test_inline_font_size_uses_libass_size(self):
        base, bigger = self.m.span_advances([("JP", "自", 0, 1), ("JP", r"{\fs90}自", 0, 1)])
        self.assertAlmostEqual(bigger, base * 1.5, delta=0.5)

    def test_inline_horizontal_scale_is_applied_once(self):
        base, stretched = self.m.span_advances([("JP", "自", 0, 1), ("JP", r"{\fscx150}自", 0, 1)])
        self.assertAlmostEqual(stretched, base * 1.5, delta=0.5)

    def test_inline_spacing_expands_a_multicharacter_span(self):
        base, spaced = self.m.span_advances([("JP", "自私", 0, 2), ("JP", r"{\fsp7}自私", 0, 2)])
        self.assertGreaterEqual(spaced - base, 6.5)

    def test_inline_font_selection_matches_an_equivalent_style(self):
        alternate = self.doc.style("JP").copy("ArialStyle")
        alternate.set("Fontname", "Arial")
        self.doc.styles.append(alternate)
        inline, explicit, original = self.m.span_advances([
            ("JP", r"{\fnArial}Wi", 0, 2), ("ArialStyle", "Wi", 0, 2), ("JP", "Wi", 0, 2)])
        self.assertEqual(inline, explicit)
        self.assertGreater(abs(original - inline), 1)

    def test_bare_reset_restores_all_inherited_font_metrics(self):
        base, reset = self.m.span_advances([
            ("JP", "自私", 0, 2),
            ("JP", r"{\fnArial\fs20\fscx40\fsp8\b1\i1}前{\r}自私", 1, 3)])
        self.assertEqual(base, reset)

    def test_named_reset_measures_invisible_black_style_without_its_paint(self):
        reset, explicit = self.m.span_advances([
            ("JP", r"前{\rReset}Wi", 1, 3),
            ("JP", r"{\fnArial\fs90\fscx125\fscy100\fsp3\b-1\i0\fe1}Wi", 0, 2)])
        self.assertIsNotNone(reset)
        self.assertGreater(reset, 0)
        self.assertAlmostEqual(reset, explicit, delta=0.5)

    def test_named_reset_style_edit_invalidates_persistent_probe(self):
        request = [("JP", r"{\rReset}Wi", 0, 2)]
        first = self.m.span_advances(request)[0]
        self.m.close()
        self.doc.style("Reset").set("Fontsize", 120)
        with Measurer(self.doc, cache_path=self.root / "advance.sqlite3") as second:
            changed = second.span_advances(request)[0]
            self.assertGreater(second.renders, 0)
        self.assertGreater(changed, first * 1.2)

    def test_font_changes_inside_span_are_retained(self):
        plain, mixed, full_large = self.m.span_advances([
            ("JP", "自私", 0, 2), ("JP", r"自{\fs90}私", 0, 2),
            ("JP", r"{\fs90}自私", 0, 2)])
        self.assertGreater(mixed, plain)
        self.assertLess(mixed, full_large)

    def test_hard_newline_preserves_indices_and_cross_line_span_is_rejected(self):
        plain, second, crossed = self.m.span_advances([
            ("JP", "私", 0, 1), ("JP", r"自\N私", 1, 2), ("JP", r"自\N私", 0, 2)])
        self.assertEqual(plain, second)
        self.assertIsNone(crossed)

    def test_soft_newline_obeys_wrap_style(self):
        with Measurer(self.doc, wrap_style=2) as wrapped:
            self.assertIsNone(wrapped.span_advances([("JP", r"自\n私", 0, 2)])[0])
            self.assertEqual(wrapped.span_advances([("JP", r"自\n私", 1, 2)]), self.widths("私"))
        actual, expected = self.m.span_advances([
            ("JP", r"自\n私", 0, 3), ("JP", r"自\h私", 0, 3)])
        self.assertEqual(actual, expected)

    def test_hardspace_has_advance_and_space_only_span_has_no_ink(self):
        pair, spaced, empty = self.m.span_advances([
            ("JP", "自私", 0, 2), ("JP", r"自\h私", 0, 3), ("JP", r"\h", 0, 1)])
        self.assertGreater(spaced, pair)
        self.assertIsNone(empty)

    def test_missing_requested_font_is_measured_through_real_libass_fallback(self):
        width = self.m.span_advances([("JP", r"{\fnDuoSub_Missing_Advance_Test_Font}自", 0, 1)])[0]
        self.assertIsNotNone(width)
        self.assertGreater(width, 0)
        self.assertTrue(any("DuoSub_Missing_Advance_Test_Font" in item for item in self.m.font_log))

    def test_memory_and_persistent_warm_cache_reproduce_cold_measurement(self):
        requests = [("JP", "自", 0, 1), ("JP", r"{\fs75}私", 0, 1)]
        cold = self.m.span_advances(requests)
        self.assertTrue(all(value is not None for value in cold))
        with patch.object(self.m, "_render_ffmpeg", side_effect=AssertionError("memory cache missed")):
            self.assertEqual(self.m.span_advances(requests), cold)
        self.m.close()
        with Measurer(self.doc, cache_path=self.root / "advance.sqlite3") as warm, \
                patch.object(warm, "_render_ffmpeg", side_effect=AssertionError("persistent cache missed")):
            self.assertEqual(warm.span_advances(requests), cold)
            self.assertEqual(warm.renders, 0)

    def test_probe_precision_is_restored_after_success_and_error(self):
        self.assertEqual(self.m.raster_scale, 1)
        self.widths("自")
        self.assertEqual(self.m.raster_scale, 1)
        with patch.object(self.m, "_render", side_effect=RuntimeError("probe failed")):
            with self.assertRaisesRegex(RuntimeError, "probe failed"):
                self.widths("私")
        self.assertEqual(self.m.raster_scale, 1)
        with Measurer(self.doc, raster_scale=8) as precise:
            self.assertIsNotNone(precise.span_advances([("JP", "自", 0, 1)])[0])
            self.assertEqual(precise.raster_scale, 8)

    def test_cancelled_measurement_restores_precision_and_starts_no_child(self):
        self.m.runner.cancel()
        with patch.object(self.m, "_render_ffmpeg", side_effect=AssertionError("cancelled child started")):
            with self.assertRaises(CancelledError):
                self.widths("自")
        self.assertEqual(self.m.raster_scale, 1)

    def test_invalid_or_out_of_range_spans_fail_closed(self):
        self.assertEqual(self.m.span_advances([
            ("JP", "自", -1, 1), ("JP", "自", 0, 0), ("JP", "自", 0, 2), ("JP", "自", 1, 2)]),
            [None, None, None, None])

    def test_large_request_keeps_high_resolution_render_batches_bounded(self):
        requests = [("JP", char, 0, 1) for char in "自私国字世界"]
        with patch.object(self.m, "_render", wraps=self.m._render) as render:
            widths = self.m.span_advances(requests)
        self.assertTrue(all(value is not None and value > 0 for value in widths))
        self.assertGreater(len(render.call_args_list), 1)
        self.assertTrue(all(len(call.args[0]) <= 8 for call in render.call_args_list))


class AdvanceExportContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ruby_advance_export_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def build(self, doc, *, mode="release", cache=None, percent=150, captured=None, template=None):
        ceiling = {kind: {"enabled": percent is not None, "percent": percent or 150}
                   for kind in ("dialogue", "song")}
        options = ExportOptions(mode=mode, ruby_max_width=ceiling)
        original = RubyGenerator._layout_job

        def record(generator, job, boxes, ruby_boxes):
            original(generator, job, boxes, ruby_boxes)
            if captured is not None:
                captured.append((job, copy.deepcopy(boxes), copy.deepcopy(ruby_boxes)))

        with patch.object(RubyGenerator, "_layout_job", record):
            return build_ass(doc, {}, template or styles(), self.root, options,
                             cache=cache, measure_cache=self.root / "render.sqlite3")

    def test_three_kana_over_narrow_and_wide_kanji_keep_natural_width(self):
        captured = []
        self.build(document(), captured=captured)
        self.assertEqual(len(captured), 2)
        for job, boxes, _ruby_boxes in captured:
            self.assertEqual(job.placements[0][2], 100)
            self.assertAlmostEqual(job.placements[0][1], boxes[0].cx)
            self.assertGreater(job.base_advances[0, 1], boxes[0].width)
            self.assertFalse(any("占位宽度" in warning for warning in job.warnings))

    def test_real_width_ceiling_uses_advance_and_preserves_ink_center(self):
        captured = []
        self.build(document((("自", "みずから"),)), captured=captured)
        job, boxes, ruby_boxes = captured[0]
        expected = min(100, max(60, job.base_advances[0, 1] * 150 / ruby_boxes[0].width))
        self.assertGreater(expected, 60)
        self.assertLess(expected, 100)
        self.assertAlmostEqual(job.placements[0][2], expected)
        self.assertAlmostEqual(job.placements[0][1], boxes[0].cx)
        legacy_cap = boxes[0].width * 150 / ruby_boxes[0].width
        self.assertGreater(job.placements[0][2], legacy_cap + 5)
        emitted = line_tag(job.out[0].text, "fscx").num()
        self.assertLessEqual(emitted, expected + 1e-8)
        self.assertLess(expected - emitted, 0.011)

    def test_manual_scale_and_disabled_ceiling_do_not_request_automatic_advance(self):
        for scale, percent, expected in ((42, 100, 42), (125, 100, 125), (None, None, 100)):
            with self.subTest(scale=scale, percent=percent):
                captured = []
                with patch.object(Measurer, "span_advances", side_effect=AssertionError("unnecessary cap probe")):
                    self.build(document((("自", "みずから"),), scale=scale), percent=percent, captured=captured)
                job, boxes, _ruby_boxes = captured[0]
                self.assertEqual(job.placements[0][2], expected)
                self.assertAlmostEqual(job.placements[0][1], boxes[0].cx)
                self.assertEqual(job.base_advances, {})

    def test_preview_release_and_export_cache_are_identical_and_preserve_document(self):
        doc = document((("自", "みずから"), ("私", "わたし")))
        before = copy.deepcopy(document_to_json(doc))
        cache = ExportCache()
        preview, preview_report = self.build(doc, mode="preview", cache=cache)
        cached, cached_report = self.build(doc, mode="preview", cache=cache)
        release, release_report = self.build(doc, mode="release")
        self.assertEqual(preview.serialize(), cached.serialize())
        self.assertEqual(cached_report.laid_out, 0)
        self.assertEqual([event.text for event in preview.events], [event.text for event in release.events])
        self.assertEqual(preview_report.ruby_lines, release_report.ruby_lines)
        self.assertEqual(preview_report.owners, release_report.owners)
        self.assertEqual(document_to_json(doc), before)

    def test_song_karaoke_retains_width_limit_and_original_beat_records(self):
        from tests.test_karaoke_fx import fixture
        doc, template = fixture("kf", ruby=True)
        member, ann = doc.members["m_fx"], doc.annotations["a_fx"]
        member.runs[0].text = "自か"
        ann.base, ann.text, ann.parts = "自", "みずから", [[0, 1, "みずから"]]
        before = copy.deepcopy(document_to_json(doc))
        captured = []
        output, report = self.build(doc, captured=captured, template=template)
        self.assertTrue(output.events)
        self.assertGreater(report.ruby_lines, 0)
        job, _boxes, ruby_boxes = next(row for row in captured if row[0].pieces)
        self.assertIn((0, 1), job.base_advances)
        expected = min(100, max(60, job.base_advances[0, 1] * 150 / ruby_boxes[0].width))
        self.assertAlmostEqual(job.placements[0][2], expected)
        self.assertIn(r"\k", job.out[0].text)
        self.assertEqual(document_to_json(doc), before)


if __name__ == "__main__":
    unittest.main()
