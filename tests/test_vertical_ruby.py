"""Vertical ruby placement checked against real libass ink and hit geometry."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subkit.ass.document import AssDocument, Style
from subkit.ass.text import VisibleText, line_tag
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.layout_index import LayoutIndex
from subkit.core.readings import anchor_of, visible_map
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.proc import run
from subkit.ruby.generate import RubyGenerator


def fixture(angle=270, *, inline=False, edge=False, readings=("あね", "いもうと"), text="姉妹"):
    raw = (r"{\frz270}" if inline else "") + text
    m = Member("m", "g", "ja", "original", "dialogue", 1000, 3000,
               style="JP", runs=[Run(raw, ins="i", b=len(raw))])
    doc = Document("vertical", groups=[Group("g", "dialogue", [m.id])], members={m.id: m})
    ids = visible_map(m)[1]
    for i, reading in enumerate(readings):
        doc.annotations[str(i)] = Annotation(str(i), "ruby", "m", anchor_of(ids[i:i+1]), text[i], reading,
                                             status="confirmed")
    template = AssDocument()
    template.script_info = [("PlayResX", "960"), ("PlayResY", "540")]
    template.styles = [Style.new("JP", Fontname="@MS Gothic", Fontsize=57, Outline=0, Shadow=0,
                                Alignment=3 if edge else 5, MarginR=72, MarginV=20,
                                Angle=0 if inline else angle, PrimaryColour="&H0000FF00"),
                       Style.new("JP.Ruby", Fontname="MS Gothic", Fontsize=28.5, Outline=0, Shadow=0,
                                 PrimaryColour="&H000000FF")]
    return doc, template


class VerticalRuby(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def export(self, doc, template, cache=None, **kwargs):
        return build_ass(doc, {}, template, self.root, ExportOptions(playres=(960, 540), ruby_gap=4, **kwargs),
                         cache=cache, measure_cache=self.root / "measure.sqlite3")

    def render(self, ass, *, label=None):
        ass.save(self.root / "vertical.ass")
        proc = run(["ffmpeg", "-hide_banner", "-loglevel", "verbose", "-f", "lavfi", "-i",
                    "color=c=black:s=1920x1080:r=1:d=1", "-vf",
                    "setpts=PTS+1.4/TB,ass=vertical.ass,format=rgb24", "-frames:v", "1",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], cwd=self.root, capture_output=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
        frame = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(1080, 1920, 3)
        if label:
            evidence = Path(os.environ["TEMP"])
            from PySide6.QtGui import QImage
            image = QImage(frame.data, 1920, 1080, frame.strides[0], QImage.Format.Format_RGB888)
            self.assertTrue(image.save(str(evidence / (label + ".png"))))
            (evidence / (label + "-libass.log")).write_bytes(proc.stderr)
            ass.save(evidence / (label + ".ass"))
        return frame

    def bounds(self, frame, channel):
        mask = (frame[:, :, channel] >= 128) & (frame[:, :, 1-channel] < 64)
        yy, xx = np.where(mask)
        self.assertGreater(len(xx), 0)
        return (xx.min()/2, yy.min()/2, (xx.max()+1)/2, (yy.max()+1)/2)

    def assert_right(self, base, ruby):
        self.assertAlmostEqual(ruby[0] - base[2], 4, delta=1.5)
        self.assertGreater(min(base[3], ruby[3]) - max(base[1], ruby[1]), 0)

    def test_real_ink_style_270_style_90_inline_270_and_hit_testing(self):
        for angle, inline in ((270, False), (90, False), (270, True)):
            with self.subTest(angle=angle, inline=inline):
                doc, template = fixture(angle, inline=inline)
                before = copy.deepcopy(doc)
                ass, report = self.export(doc, template)
                self.assertEqual(doc, before)
                self.assertEqual(report.ruby_lines, 2)
                self.assertEqual(line_tag(ass.events[1].text, "frz").num(), angle)
                # Isolate corresponding kanji without changing advances (alpha preserves layout).
                for i in range(2):
                    probe = copy.deepcopy(ass)
                    base = probe.events[0]
                    chars = VisibleText(base.text).text
                    prefix = base.text[:base.text.index(chars)] if chars in base.text else ""
                    base.text = prefix + "".join((r"{\alpha&H00&}" if k==i else r"{\alpha&HFF&}")+ch
                                                 for k, ch in enumerate(chars))
                    probe.events = [base, probe.events[i+1]]
                    frame = self.render(probe)
                    self.assert_right(self.bounds(frame, 1), self.bounds(frame, 0))
                index = LayoutIndex(ass, report.owners)
                try:
                    geoms = index.geometry(range(3))
                    for geom in geoms:
                        x = sum(p[0] for p in geom.quad)/4
                        y = sum(p[1] for p in geom.quad)/4
                        hit = index.hit(geoms, x, y, pad=0)
                        self.assertIsNotNone(hit)
                        self.assertEqual(hit.line.index, geom.index)
                finally:
                    index.close()
                if angle == 90:
                    self.render(ass, label="vertical-90")

    def test_nonbiri_edge_shift_cache_and_preview_equal(self):
        doc, template = fixture(edge=True)
        cache = ExportCache()
        ass, report = self.export(doc, template, cache)
        self.assertEqual(len([w for w in report.warnings if "整行内移" in w]), 1)
        frame = self.render(ass, label="vertical-nonbiri-270-edge-shift")
        ruby = self.bounds(frame, 0)
        self.assertGreaterEqual(ruby[0], 0)
        self.assertLessEqual(ruby[2], 958)
        self.assertGreaterEqual(ruby[1], 0)
        self.assertLessEqual(ruby[3], 540)
        cached, again = self.export(doc, template, cache)
        self.assertEqual(cached.serialize(), ass.serialize())
        self.assertEqual(again.laid_out, 0)
        preview, _ = self.export(doc, template, mode="preview")
        self.assertEqual([e.text for e in preview.events], [e.text for e in ass.events])
        self.assertEqual(line_tag(ass.events[0].text, "org").float_args(),
                         line_tag(ass.events[1].text, "org").float_args())

    def test_long_reading_neighbours_and_vertical_choonpu(self):
        for angle in (90, 270):
            doc, template = fixture(angle, readings=("ああああー", "いもうと"))
            ass, _ = self.export(doc, template)
            base_frame = self.render(ass)
            base = self.bounds(base_frame, 1)
            rubies = []
            for ev in ass.events[1:]:
                probe = copy.deepcopy(ass)
                probe.events = [ev]
                rubies.append(self.bounds(self.render(probe), 0))
            a, b = sorted(rubies, key=lambda r:r[1])
            self.assertLessEqual(a[3]+1, b[1])
            self.assertTrue(all(r[0] > base[2] for r in rubies))
            # Vertical font substitution must also reach the ruby, including punctuation.
            probe = copy.deepcopy(ass)
            probe.events = [ass.events[1].copy(text=ass.events[1].text.split("}",1)[0]+"}ー")]
            r = self.bounds(self.render(probe), 0)
            self.assertGreater(r[3]-r[1], r[2]-r[0])

    def test_shared_probes_do_not_move_other_lines_and_explicit_origins(self):
        doc, template = fixture()
        m = doc.members["m"]
        m.overrides = {"an": 5, "pos": [400, 270], "raw": r"\org(390,260)"}
        other = copy.deepcopy(m)
        other.id, other.group = "other", "other_g"
        other.overrides = {"an": 5, "pos": [650, 270], "raw": r"\org(640,260)"}
        doc.members[other.id] = other
        doc.groups.append(Group(other.group, "dialogue", [other.id]))
        for ann in list(doc.annotations.values()):
            copied = copy.deepcopy(ann)
            copied.id += "b"
            copied.member = other.id
            doc.annotations[copied.id] = copied
        ass, report = self.export(doc, template)
        self.assertEqual(report.ruby_lines, 4)
        bases = [e for e, o in zip(ass.events, report.owners) if o["kind"] == "base"]
        ruby = [e for e, o in zip(ass.events, report.owners) if o["kind"] == "ruby"]
        self.assertAlmostEqual(line_tag(ruby[2].text, "pos").float_args()[0] -
                               line_tag(ruby[0].text, "pos").float_args()[0], 250, delta=.1)
        self.assertEqual(line_tag(ruby[0].text, "org").float_args(), [390, 260])
        self.assertEqual(line_tag(ruby[2].text, "org").float_args(), [640, 260])

    def test_each_frame_edge_and_borders(self):
        for x, y in ((10, 270), (950, 270), (480, 10), (480, 530)):
            for angle in (90, 270):
                with self.subTest(x=x, y=y, angle=angle):
                    doc, template = fixture(angle)
                    doc.members["m"].overrides = {"an": 5, "pos": [x, y]}
                    template.styles[0].set("Outline", 2)
                    template.styles[1].set("Outline", 1)
                    ass, report = self.export(doc, template)
                    frame = self.render(ass)
                    for channel in (0, 1):
                        bounds = self.bounds(frame, channel)
                        self.assertGreaterEqual(min(bounds[:2]), 0)
                        self.assertLessEqual(bounds[2], 960)
                        self.assertLessEqual(bounds[3], 540)
                    doc.members["m"].overrides["pos"] = [480, 270]
                    centred, _ = self.export(doc, template)
                    reference = self.render(centred)
                    for channel in (0, 1):
                        def ink_count(image):
                            return np.count_nonzero((image[:,:,channel] >= 128) & (image[:,:,1-channel] < 64))
                        self.assertGreaterEqual(ink_count(frame), ink_count(reference)*.95)
                    self.assertTrue(any("整行内移" in w for w in report.warnings))

    def test_detection_and_all_edge_translations(self):
        doc, template = fixture()
        gen = RubyGenerator(template, None)
        from subkit.ass.document import Event
        for font, angle, expected in (("MS Gothic", 270, False), ("@MS Gothic", 0, False),
                                      ("@MS Gothic", 90, True), ("@MS Gothic", -90, True)):
            template.styles[0].set("Fontname", font)
            template.styles[0].set("Angle", angle)
            self.assertEqual(gen.vertical(Event(style="JP", text="姉")) is not None, expected)


if __name__ == "__main__":
    unittest.main()
