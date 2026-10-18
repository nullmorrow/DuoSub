"""Quad outlines and readings, measured against host ffmpeg/libass renders."""

from __future__ import annotations

import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .test_transform import LibassHarness, document, ink_box
from subkit.ass.document import Event, Style
from subkit.core.layout_index import LayoutIndex, LineGeom, Rect, in_quad, unproject
from subkit.core.transform import LineTransform, line_quad, measurement_copy
from subkit.ruby.generate import RubyGenerator, RubyJob, RubyPiece
from subkit.ruby.measure import LineRequest, Measurer


def owners(kind="base"):
    return [{"member": "m", "group": "g", "kind": kind, "ids": [("s", "e", k) for k in range(10)]}]


class QuadHit(unittest.TestCase):
    def test_padding_order_and_degenerate_quad(self):
        g = LineGeom(0, "m", "g", "base", 0, Rect(10, 20, 30, 40), chars=[Rect(10, 20, 30, 40)])
        self.assertEqual(g.quad, g.rect.corners)
        for x, y in ((4, 14), (36, 46), (10, 20), (30, 40), (3.99, 20), (20, 46.01)):
            self.assertEqual(LayoutIndex.hit([g], x, y) is not None, g.rect.contains(x, y, 6))
        self.assertIsNone(LayoutIndex.hit([g], math.nan, 20))
        q = ((10, 20),) * 4
        self.assertTrue(in_quad(q, 10, 20, 0))
        self.assertFalse(in_quad(q, 100, 200, 6))
        top = LineGeom(1, "top", "g", "ruby", 2, g.rect)
        self.assertIs(LayoutIndex.hit([top, g], 15, 25).line, top)

    def test_inverse_perspective_and_reflected_winding(self):
        rect = Rect(100, 150, 240, 230)
        tf = LineTransform(frz=30, frx=35, fry=-25, fax=.3, fay=-.2)
        q = line_quad(rect, (100, 150), tf, layout_origin=(100, 150))
        for x, y in ((100, 150), (240, 230), (160, 190)):
            p = line_quad((x, y, x, y), (100, 150), tf, layout_origin=(100, 150))[0]
            self.assertTrue(in_quad(q, *p, 0))
            np.testing.assert_allclose(unproject(q, rect, *p), (x, y), atol=1e-8)
        self.assertTrue(in_quad(tuple(reversed(q)), *q[0], 0))

    def test_perspective_context_precedence(self):
        doc = document()
        li = LayoutIndex(doc, [], video_height=1080)
        self.assertEqual(li.perspective_distance, 156.25)
        li.close()
        doc.set_info("LayoutResY", 720)
        li = LayoutIndex(doc, [], video_height=1080)
        self.assertEqual(li.perspective_distance, 234.375)
        li.close()
        doc.remove_info("LayoutResY")
        li = LayoutIndex(doc, [])
        self.assertEqual(li.perspective_distance, 312.5)
        li.close()


class LayoutLibass(LibassHarness):
    @classmethod
    def tearDownClass(cls):
        try:
            errors = [r["error_px"] for r in cls.results if "error_px" in r]
            print(f"T3a outline: {len(errors)} renders, max error={max(errors, default=0):.6f} px")
            for r in cls.results:
                if "cold_ms_per_line" in r:
                    print("T3a cost: " + str(r))
            if cls.logs:
                print("\n".join(s for s in cls.logs[0].splitlines() if "libass" in s or "fontselect" in s))
        finally:
            super().tearDownClass()

    def geometry(self, doc, text, *, owner=None, **kw):
        copy = doc.copy()
        copy.events = [Event(text=text)]
        li = LayoutIndex(copy, owner or owners(), **kw)
        try:
            return li.geometry([0])[0]
        finally:
            li.close()

    def assert_outline(self, g, frame, label, scale=1):
        q = np.asarray([p for quad in g.quads for p in quad])
        predicted = (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max())
        error = float(np.abs(np.asarray(predicted) - ink_box(frame, scale)).max())
        self.results.append({"case": label, "error_px": error})
        self.assertLessEqual(error, 2, (label, error, predicted, ink_box(frame, scale)))

    def test_typical_transform_outline_all_alignments_and_hits(self):
        doc = document()
        tags = [r"\frz45", r"\frz-45", r"\fscx50\fscy50", r"\fscx200\fscy200",
                r"\fscx200\fscy50", r"\fax.5", r"\fax-.5", r"\fay.5", r"\fay-.5",
                r"\frx45", r"\fry-45", r"\frz35\frx30\fry-30\fscx150\fscy120\fax.3\fay.2"]
        texts = [f"{{\\an{an}\\pos(480,270){tag}}}██" for an in range(1, 10) for tag in tags]
        frames = self.render(doc, texts, "layout-typical")
        copy = doc.copy()
        copy.events = [Event(text=t) for t in texts]
        li = LayoutIndex(copy, owners() * len(texts))
        before = copy.serialize()
        try:
            geoms = li.geometry(range(len(texts)))
            self.assertEqual(copy.serialize(), before)
            for g, frame, text in zip(geoms, frames, texts):
                with self.subTest(text=text):
                    self.assert_outline(g, frame, text)
                    y, x = np.where(frame > 24)
                    # Check points across actual ink, not merely quad centres.
                    for k in np.linspace(0, len(x) - 1, 9).astype(int):
                        self.assertIsNotNone(LayoutIndex.hit([g], x[k] + .5, y[k] + .5, pad=2))
                    self.assertIsNone(LayoutIndex.hit([g], 30, 30, pad=0))
        finally:
            li.close()

    def test_changed_position_is_hit_and_old_ink_is_not(self):
        doc = document()
        plain = r"{\an7\pos(480,270)}██"
        texts = [r"{\an7\pos(480,270)\org(300,200)\frz45}██",
                 r"{\an7\pos(480,270)\fscx200}██", r"{\an7\pos(480,270)\fax.5}██",
                 r"{\an7\pos(480,270)\org(300,200)\frx45\fry-30}██"]
        frames = self.render(doc, [plain, *texts], "layout-old-position")
        old_y, old_x = np.where(frames[0] > 24)
        for text, frame in zip(texts, frames[1:]):
            g = self.geometry(doc, text)
            y, x = np.where(frame > 24)
            changed = [k for k in range(0, len(x), 20) if frames[0][y[k], x[k]] <= 24]
            self.assertTrue(changed, text)
            k = changed[len(changed) // 2]
            self.assertIsNotNone(LayoutIndex.hit([g], x[k] + .5, y[k] + .5, pad=2))
            outside = [k for k in range(0, len(old_x), 20)
                       if not in_quad(g.quad, old_x[k] + .5, old_y[k] + .5, 0)]
            # Enlargement contains the old ink; its new extension still hits.
            if r"\fscx200" not in text:
                self.assertTrue(outside, text)
                k = outside[len(outside) // 2]
                self.assertIsNone(LayoutIndex.hit([g], old_x[k] + .5, old_y[k] + .5, pad=0))

    def test_readings_inverse_map_real_glyphs(self):
        doc = document()
        tags = [r"\frz45", r"\fscx150\fscy80\fax.5\fay.2\frz30", r"\frx35\fry-30\frz25"]
        for tag in tags:
            head = r"{\an7\pos(400,220)" + tag + "}"
            g = self.geometry(doc, head + "HHH")
            texts = [head + r"{\alpha&HFF&}" + "H" * k + r"{\alpha&H00&}H{\alpha&HFF&}" + "H" * (2-k)
                     for k in range(3)]
            frames = self.render(doc, texts, "layout-reading")
            for k, frame in enumerate(frames):
                y, x = np.where(frame > 24)
                hit = LayoutIndex.hit([g], float(x.mean()) + .5, float(y.mean()) + .5, pad=0)
                self.assertIsNotNone(hit)
                self.assertEqual((hit.char, hit.identity), (k, ("s", "e", k)))

    def test_multiline_shear_uses_separate_layout_origins(self):
        doc = document()
        text = r"{\an2\pos(480,350)\fax.5\fay.3\frz30}██\N█"
        frame = self.render(doc, [text], "layout-multiline")[0]
        g = self.geometry(doc, text)
        self.assertEqual(len(g.quads), 2)
        self.assert_outline(g, frame, text)
        for rect, quad, ks in g._frames:
            # Quad's projective centre mapped back finds this visual line.
            p = tuple(np.asarray(quad).mean(axis=0))
            hit = LayoutIndex.hit([g], *p, pad=0)
            self.assertIn(hit.char, ks)

    def test_ruby_uses_own_tags_and_generated_position(self):
        doc = document()
        doc.styles.append(Style.new("Ruby", Fontname="Arial", Fontsize="40", Outline="0", Shadow="0", Alignment="2"))
        base = Event(text=r"{\an2\pos(480,350)\frz30\fax.3}HHHH")
        with Measurer(doc) as m:
            job = RubyJob(base, "Ruby", [RubyPiece(1, 3, "██")])
            RubyGenerator(doc, m).run([job])
        rd = doc.copy()
        rd.styles = [doc.style("Ruby").copy("Default")]
        ev = job.out[0]
        owner = [{**owners("ruby")[0], **job.geoms[0], "ann": "reading"}]
        g = self.geometry(rd, ev.text, owner=owner)
        frame = self.render(rd, [ev.text], "layout-ruby")[0]
        self.assert_outline(g, frame, ev.text)
        y, x = np.where(frame > 24)
        self.assertEqual(LayoutIndex.hit([g], float(x.mean()), float(y.mean())).line.ann, "reading")

    def test_storage_and_layoutres_camera_distance(self):
        doc = document()
        text = r"{\an7\pos(400,220)\frx35\fry-30\frz15}██"
        frame = self.render(doc, [text], "layout-storage", scale=2)[0]
        g = self.geometry(doc, text, video_height=1080)
        self.assert_outline(g, frame, "storage-1080", scale=2)
        doc.set_info("LayoutResX", 960)
        doc.set_info("LayoutResY", 540)
        frame = self.render(doc, [text], "layout-res", scale=2)[0]
        g = self.geometry(doc, text, video_height=1080)
        self.assert_outline(g, frame, "layoutres-over-storage", scale=2)

    def test_style_transforms_margin_anchor_and_explicit_origin(self):
        doc = document()
        doc.styles[0].set("Angle", 35)
        doc.styles[0].set("ScaleX", 150)
        doc.styles[0].set("ScaleY", 120)
        texts = ["██", r"{\an2\org(480,270)\fax.3}██", r"{\an5\frz0}██"]
        frames = self.render(doc, texts, "layout-style")
        for text, frame in zip(texts, frames):
            g = self.geometry(doc, text)
            self.assert_outline(g, frame, text)

    def test_plain_geometry_unchanged_and_precise_cache_cost(self):
        doc = document()
        doc.events = [Event(text=r"{\an2\pos(480,350)}HH\NHH")]
        li = LayoutIndex(doc, owners())
        try:
            with patch("subkit.core.layout_index.Measurer", side_effect=AssertionError("ordinary line supersampled")):
                g = li.geometry([0])[0]
            boxes = li.measurer.line_char_boxes([LineRequest("Default", doc.events[0].text, 2, 480)])[0]
            expected = li._base_geom(0, boxes)
            self.assertEqual((g.rect, g.chars, g.quad), (expected.rect, expected.chars, expected.rect.corners))
        finally:
            li.close()
        doc.events = [Event(text=f"{{\\an7\\pos(400,220)\\frz30}}HH{i}") for i in range(24)]
        with tempfile.TemporaryDirectory(prefix="subkit_layout_cache_") as tmp:
            cache = Path(tmp) / "probe.sqlite3"
            elapsed = []
            for repeat in range(2):
                li = LayoutIndex(doc, owners() * 24, measure_cache=cache)
                start = time.perf_counter()
                try:
                    g = li.geometry(range(24))
                    self.assertEqual(len(g), 24)
                finally:
                    li.close()
                elapsed.append((time.perf_counter() - start) * 1000 / 24)
            self.results.append({"lines": 24, "cold_ms_per_line": round(elapsed[0], 3),
                                 "disk_cache_ms_per_line": round(elapsed[1], 3), "raster_scale": 4})
            neutral, _ = measurement_copy(doc, doc.events[0])
            with Measurer(neutral, cache_path=cache, raster_scale=4) as m:
                m.line_char_boxes([LineRequest("Default", doc.events[0].text, 7, 400)])
                self.assertEqual(m.renders, 0)


if __name__ == "__main__":
    unittest.main()
