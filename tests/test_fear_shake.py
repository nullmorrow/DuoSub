"""Fear trembling: actual ruby anchors, bounded events, ASS continuity and motion."""
from __future__ import annotations

import copy
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.ass.document import Event, Style
from subkit.core import karaoke as kara
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.readings import visible_map
from subkit.karaoke_fx import Effect, EventLimit, FxContext, FxGlyph, REGISTRY, _register, get_effect
from subkit.model.document import Annotation, Run
from tests.test_karaoke_fx import fixture


def pos(event):
    match = re.search(r"\\(?:pos|move)\(([-\d.]+),([-\d.]+)", event.text)
    return tuple(float(v) for v in match.groups())


class FearContract(unittest.TestCase):
    def context(self, *, count=3, duration=1000, cap=2000, params=None, movement=None):
        fx = get_effect("tremble")
        p = fx.resolve({"lead_in_ms": 0, "fade_in_ms": 0, "fade_out_ms": 0, **(params or {})})
        glyphs = [FxGlyph(Event(style="Default"), (100+i*50, 180, 140+i*50, 230), "", "H",
                         100+i*50, 180, [(i, 1000, 1000+duration)],
                         {"kind": "base", "ids": [("ins", "i", i)]}, movement=movement,
                         anchor_ids=(("ins", "i", i),)) for i in range(count)]
        return FxContext("fear", 1000, 1000+duration, Style.new("Default"), glyphs, p,
                         rng=FxContext.seeded("fear", p), event_cap=cap)

    def test_deterministic_bounded_discrete_independent_positions(self):
        fx = get_effect("tremble")
        ctx = self.context()
        events = fx.generate(ctx)
        self.assertEqual(events, fx.generate(self.context()))
        for i in range(3):
            cells = events[i::3]
            self.assertEqual(cells[0].start, 100)
            self.assertEqual(cells[-1].end, 200)
            self.assertTrue(all(a.end == z.start for a, z in zip(cells, cells[1:])))
            offsets = [(pos(e)[0]-(100+i*50), pos(e)[1]-180) for e in cells]
            self.assertGreater(len(set(offsets)), 10)
            self.assertTrue(all(abs(x) <= 3.01 and abs(y) <= 3.01 for x, y in offsets))
        self.assertNotEqual(pos(events[0])[0]-100, pos(events[1])[0]-150)
        self.assertTrue(all("\\frz" not in e.text for e in events))
        self.assertEqual(len(events), len(ctx.owners))

    def test_ruby_piece_shares_offsets_with_all_its_base_characters(self):
        ctx = self.context()
        ctx.glyphs.append(FxGlyph(Event(style="Default.Ruby"), (110, 145, 135, 170), "", "い",
                                 110, 145, [(1, 1500, 1800)], {"kind": "ruby"},
                                 anchor_ids=ctx.glyphs[0].anchor_ids+ctx.glyphs[1].anchor_ids))
        events = get_effect("tremble").generate(ctx)
        for a, b, _independent, ruby in zip(events[::4], events[1::4], events[2::4], events[3::4]):
            self.assertEqual((a.start, a.end), (ruby.start, ruby.end))
            self.assertAlmostEqual(pos(a)[0]-100, pos(ruby)[0]-110)
            self.assertAlmostEqual(pos(b)[0]-150, pos(ruby)[0]-110)
            self.assertAlmostEqual(pos(a)[1]-180, pos(ruby)[1]-145)

    def test_whole_line_and_inherited_move(self):
        ctx = self.context(params={"whole_line": True}, movement=(100, -20, 0, 1000))
        events = get_effect("tremble").generate(ctx)
        for a, b, c in zip(events[::3], events[1::3], events[2::3]):
            self.assertAlmostEqual(pos(a)[0]-100, pos(b)[0]-150)
            self.assertAlmostEqual(pos(a)[0]-100, pos(c)[0]-200)
            values = [float(v) for v in re.search(r"\\move\(([^)]+)\)", a.text).group(1).split(",")]
            self.assertAlmostEqual(values[2]-values[0], (a.end-a.start), places=2)
            self.assertAlmostEqual(values[3]-values[1], -(a.end-a.start)*.2, places=2)

    def test_typical_ruby_line_adapts_and_extreme_line_falls_back(self):
        fx = get_effect("tremble")
        for count, duration in ((24, 4000), (40, 10000)):
            with self.subTest(count=count, duration=duration):
                ctx = self.context(count=count, duration=duration)
                events = fx.generate(ctx)
                self.assertLessEqual(len(events), 2000)
                self.assertTrue(ctx.warnings)
                self.assertEqual(len(events), len(ctx.owners))
                self.assertEqual(events[0].start, 100)
                self.assertEqual(events[-1].end, (1000+duration)//10)
        with self.assertRaises(EventLimit):
            fx.generate(self.context(count=40, duration=20000))
        with self.assertRaises(EventLimit):
            fx.generate(self.context(count=40, duration=10000, params={"lead_in_ms": 300}))
        with self.assertRaises(EventLimit):
            fx.generate(self.context(count=3, cap=2))

    def test_centisecond_boundaries_keep_colour_history_and_line_fade_clock(self):
        ctx = self.context(count=1, params={"fade_in_ms": 150, "fade_out_ms": 150, "interval_ms": 33})
        ctx.glyphs[0].beats = [(0, 1137, 1303), (1, 1527, 1821)]
        events = get_effect("tremble").generate(ctx)
        self.assertTrue(all(a.end == z.start for a, z in zip(events, events[1:])))
        for e in events:
            if e.start*10 >= 1140:
                self.assertIn("\\1c&HFFFFFF&", e.text)
            self.assertIn(f"\\fade(255,0,255,{1000-e.start*10},{1150-e.start*10},"
                          f"{1850-e.start*10},{2000-e.start*10})", e.text)
        self.assertTrue(any("\\t(19,20,\\alpha&H00&)" in e.text for e in events))

    def test_move_endpoints_between_grid_and_centiseconds_preserve_original_clock(self):
        ctx = self.context(count=1, params={"jitter_x": 0, "jitter_y": 0, "interval_ms": 200},
                           movement=(100, -20, 125, 875))
        events = get_effect("tremble").generate(ctx)
        self.assertTrue(any(e.end == 113 for e in events))
        for t in range(1000, 2000, 5):
            e = next(e for e in events if e.start*10 <= t < e.end*10)
            match = re.search(r"\\move\(([^)]+)\)", e.text)
            if match:
                x0, y0, x1, y1, a, z = map(float, match.group(1).split(","))
                f = min(1, max(0, (t-e.start*10-a)/(z-a)))
                actual = (x0+(x1-x0)*f, y0+(y1-y0)*f)
            else:
                actual = pos(e)
            expected_f = min(1, max(0, (t-1125)/750))
            self.assertAlmostEqual(actual[0], 100+100*expected_f, delta=.011)
            self.assertAlmostEqual(actual[1], 180-20*expected_f, delta=.011)
        # Extra endpoint slices are included in the event budget too.
        small = self.context(count=24, duration=4000, movement=(100, -20, 125, 875))
        self.assertLessEqual(len(get_effect("tremble").generate(small)), small.event_cap)

    def test_register_rejects_duplicate_or_reserved_ids_without_replacement(self):
        old = REGISTRY["kf"]
        registry = {"kf": old}
        with self.assertRaisesRegex(ValueError, "代号重复"):
            _register(registry, [Effect("kf", "覆盖")])
        self.assertIs(registry["kf"], old)
        with self.assertRaises(ValueError):
            _register(registry, [Effect("preset:fake", "预设")])
        extra = Effect("future_effect", "未来效果")
        _register(registry, [extra])
        self.assertIs(registry["future_effect"], extra)


class FearExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def export(self, doc, tpl, **options):
        return build_ass(doc, {}, tpl, self.path, ExportOptions(**options), measure_cache=self.path/"measure.sqlite3")

    def test_actual_multikanji_ruby_anchors_determinism_and_cached_preview(self):
        doc, tpl = fixture("tremble", ruby=True)
        m = doc.members["m_fx"]
        m.runs = [Run("恐怖だ", ins="i_fx", b=3)]
        ann = doc.annotations["a_fx"]
        ann.base, ann.text, ann.anchor, ann.parts = "恐怖", "きょうふ", [{"ins": "i_fx", "a": 0, "b": 2}], [[0, 2, "きょうふ"]]
        beats = kara.derive(doc, m)
        for i, b in enumerate(beats):
            b.t0, b.t1, b.by = 1000+i*300, 1300+i*300, "manual"
        doc.karaoke["k_fx"] = kara.beats_to_track(m, beats, effect={"preset": "tremble"})
        original = copy.deepcopy(doc)
        seen = []
        effect = get_effect("tremble")
        generate = effect.generate
        from unittest.mock import patch
        with patch.object(effect, "generate", side_effect=lambda ctx: (seen.append(ctx), generate(ctx))[1]):
            first, report = self.export(doc, tpl)
        ctx = seen[0]
        identities = visible_map(m)[1][:2]
        bases = [g for g in ctx.glyphs if g.owner["kind"] == "base" and g.owner["ids"][0] in identities]
        ruby = [g for g in ctx.glyphs if g.owner["kind"] == "ruby"]
        self.assertEqual(len(bases), 2)
        self.assertGreaterEqual(len(ruby), 4)
        self.assertTrue(all(set(g.anchor_ids) == set(identities) for g in ruby))
        at = min(e.start for e in first.events)
        actual = [(e, o) for e, o in zip(first.events, report.owners) if e.start == at]
        offsets = []
        for g in bases+ ruby:
            e = next(e for e, o in actual if o == g.owner and e.text.endswith(g.text))
            offsets.append((round(pos(e)[0]-g.x, 2), round(pos(e)[1]-g.y, 2)))
        self.assertLessEqual(max(x for x, y in offsets)-min(x for x, y in offsets), .02)
        self.assertLessEqual(max(y for x, y in offsets)-min(y for x, y in offsets), .02)
        second, _ = self.export(doc, tpl)
        self.assertEqual(first.serialize(), second.serialize())
        cache = ExportCache()
        cached, _ = build_ass(doc, {}, tpl, self.path, ExportOptions(), cache=cache, measure_cache=self.path/"measure.sqlite3")
        again, _ = build_ass(doc, {}, tpl, self.path, ExportOptions(), cache=cache, measure_cache=self.path/"measure.sqlite3")
        self.assertEqual(cached.serialize(), again.serialize())
        self.assertEqual(first.serialize(), cached.serialize())
        self.assertEqual(doc, original)

    def test_single_kanji_multiple_readings_and_follow_off(self):
        doc, tpl = fixture("tremble", ruby=True)
        enabled, report = self.export(doc, tpl)
        self.assertTrue(any(o["kind"] == "ruby" for o in report.owners))
        self.assertTrue(all("\\kf" not in e.text for e in enabled.events))
        doc.karaoke["k_fx"]["effect"]["params"]["follow_furigana"] = False
        disabled, _ = self.export(doc, tpl)
        self.assertTrue(any("\\kf" in e.text for e in disabled.events if e.style.endswith(".Ruby")))
        self.assertGreater(len(enabled.events), len(disabled.events))

    def test_libass_base_and_ruby_move_together_across_frames(self):
        params = {"fade_in_ms": 0, "fade_out_ms": 0, "unsung_alpha": 0}
        doc, tpl = fixture("tremble", params, ruby=True)
        moving, report = self.export(doc, tpl)
        doc.karaoke["k_fx"]["effect"]["params"].update(jitter_x=0, jitter_y=0)
        still, still_report = self.export(doc, tpl)
        first_id = visible_map(doc.members["m_fx"])[1][0]
        selections = (("base", lambda o: o["kind"] == "base" and o["ids"] == [first_id]),
                      ("ruby", lambda o: o["kind"] == "ruby"))
        for label, select in selections:
            for name, ass, rep in (("moving", moving, report), ("still", still, still_report)):
                isolated = copy.deepcopy(ass)
                isolated.events = [e for e, o in zip(ass.events, rep.owners) if select(o)]
                (self.path/f"{name}-{label}.ass").write_text(isolated.serialize(), encoding="utf-8-sig")
        displacements = {label: [] for label, _ in selections}
        for moment in (1.2, 1.24, 1.28, 1.32):
            for label, _ in selections:
                boxes = []
                for name in ("moving", "still"):
                    proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "color=c=black:s=1920x1080:r=25:d=0.04", "-vf",
                        f"setpts=PTS+{moment}/TB,ass={name}-{label}.ass,format=rgb24", "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], cwd=self.path, capture_output=True, timeout=30)
                    self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
                    frame = np.frombuffer(proc.stdout, np.uint8).reshape(1080, 1920, 3)
                    yy, xx = np.where(frame.max(axis=2) >= 80)
                    self.assertGreater(len(xx), 10)
                    boxes.append(np.array([xx.min(), yy.min(), xx.max(), yy.max()]))
                displacements[label].append(boxes[0]-boxes[1])
        for base, ruby in zip(displacements["base"], displacements["ruby"]):
            self.assertTrue(np.all(np.abs(base-ruby) <= 1), (base, ruby))
        self.assertGreater(len({tuple(v) for v in displacements["base"]}), 1)

    def test_unsung_transparency_hides_outline_too_and_restores_when_sung(self):
        params = {"lead_in_ms": 0, "fade_in_ms": 0, "fade_out_ms": 0, "unsung_alpha": 255}
        doc, tpl = fixture("tremble", params)
        m = doc.members["m_fx"]
        beats = kara.derive(doc, m)
        for b in beats:
            b.t0, b.t1, b.by = 1600, 2100, "manual"
        doc.karaoke["k_fx"] = kara.beats_to_track(m, beats, effect={"preset": "tremble", "params": params})
        tpl.styles[0].set("OutlineColour", "&H00FFFFFF")
        ass, _ = self.export(doc, tpl)
        (self.path/"opacity.ass").write_text(ass.serialize(), encoding="utf-8-sig")
        for moment, visible in ((1.2, False), (1.7, True)):
            proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                "color=c=black:s=1920x1080:r=25:d=0.04", "-vf",
                f"setpts=PTS+{moment}/TB,ass=opacity.ass,format=rgb24", "-frames:v", "1",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], cwd=self.path, capture_output=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
            ink = np.frombuffer(proc.stdout, np.uint8).max()
            self.assertEqual(bool(ink), visible, (moment, ink))
