"""Effect engine/export contract and actual host ffmpeg/libass rendering."""
from __future__ import annotations

import copy
import hashlib
import os
import re
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subkit.ass.document import AssDocument, Event, Style
from subkit.core import karaoke as kara
from subkit.core.export import ExportCache, ExportError, ExportOptions, build_ass
from subkit.core.layout_index import LayoutIndex
from subkit.core.readings import visible_map
from subkit.core.songs import song_span
from subkit.ass.text import insert_at_visible
from subkit.karaoke_fx import EventLimit, FxContext, FxGlyph, INLINE, REGISTRY, get_effect
from subkit.karaoke_fx import ktv
from subkit.karaoke_fx.base import paint, rounded_ms
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.proc import run

MOTION = ("bounce", "shake", "particles", "fly", "morph")


def fixture(name="pop", params=None, ruby=False):
    text = "生か" if ruby else "HH"
    m = Member("m_fx", "g_fx", "ja", "original", "song", 1000, 3000, style="Default",
               runs=[Run(text, ins="i_fx", b=2)])
    doc = Document("FX", groups=[Group("g_fx", "song", [m.id])], members={m.id: m})
    if ruby:
        doc.annotations["a_fx"] = Annotation("a_fx", "ruby", m.id, [{"ins": "i_fx", "a": 0, "b": 1}],
                                               "生", "いき", status="confirmed", parts=[[0, 1, "いき"]])
    beats = kara.derive(doc, m)
    for i, b in enumerate(beats):
        b.t0, b.t1, b.by = 1000+i*500, 1500+i*500, "manual"
    doc.karaoke["k_fx"] = kara.beats_to_track(m, beats, effect={"preset": name, "params": params or {}})
    template = AssDocument()
    template.styles = [Style.new("Default", Fontname="Arial", Fontsize="60", Outline="1", Shadow="0",
                                 PrimaryColour="&H00FFFFFF", SecondaryColour="&H000000FF",
                                 Alignment="7", MarginL="200", MarginV="180")]
    return doc, template


def ktv_fixture(name="kf", mode="corners", starts=(1000, 2000, 3000), ruby=False):
    doc = Document("KTV")
    _, tpl = fixture()
    tpl.styles = [tpl.styles[0].copy("OP.JP"), tpl.styles[0].copy("OP.CN")]
    for st in tpl.styles:
        st.set("Alignment", 2)
        st.set("MarginL", 80)
        st.set("MarginR", 100)
        st.set("Fontsize", 40)
        st.set("MarginV", 50 if st.name.endswith("JP") else 110)
    for i, start in enumerate(starts):
        g = Group(f"g_{i}", "song")
        doc.groups.append(g)
        for lane, suffix, text in (("original", "JP", "生か" if ruby else "HH"),
                                   ("translation", "CN", "Hi")):
            mid = f"{suffix}_{i}"
            m = Member(mid, g.id, "ja" if suffix == "JP" else "zh-Hans", lane, "song", start, start+1200,
                       style=f"OP.{suffix}", runs=[Run(text, ins=mid, b=len(text))])
            doc.members[mid] = m
            g.members.append(mid)
            if suffix == "JP":
                if ruby:
                    aid = f"a_{i}"
                    doc.annotations[aid] = Annotation(aid, "ruby", mid, [{"ins": mid, "a": 0, "b": 1}],
                        "生", "いき", status="confirmed", parts=[[0, 1, "いき"]])
                beats = kara.derive(doc, m)
                for j, b in enumerate(beats):
                    b.t0, b.t1, b.by = start+j*200, start+(j+1)*200, "manual"
                doc.karaoke[mid] = kara.beats_to_track(m, beats, effect={"preset": name,
                    "params": {"ktv": mode} if i == 0 else {}})
    return doc, tpl


class FxContract(unittest.TestCase):
    def test_segment_state_and_centisecond_tiling(self):
        g = FxGlyph(Event(), (200, 180, 260, 250), "", "潔", 200, 180,
                    [(0, 1200, 1527), (1, 1527, 1855), (2, 1855, 2182), (3, 2182, 2509)])
        for name in ("pop", "morph", "bounce", "fly", "glow", "wipe", "shake"):
            fx = get_effect(name)
            ctx = FxContext("m", 1000, 4800, Style.new("Default"), [g], fx.resolve())
            events = fx.generate(ctx)
            # Different layers intentionally coexist; segments in each layer
            # have exclusive [start,end) intervals, including rounded boundaries.
            for layer in {e.layer for e in events}:
                segments = sorted((e for e in events if e.layer == layer), key=lambda e: e.start)
                for previous, current in zip(segments, segments[1:]):
                    self.assertEqual(previous.end, current.start, (name, layer))
            if name in ("pop", "morph"):
                for e in events:
                    if e.start*10 >= 1200:
                        self.assertIn("\\1c&HFFFFFF&", e.text)
                        self.assertNotIn("\\1c&H0000FF&", e.text)
        self.assertEqual(rounded_ms(1527), 1530)
        self.assertEqual(paint(ctx, g, 2480), "\\1c&HFFFFFF&")
        # Completed original movement must not restart in a later segment.
        g.movement = (100, 20, 0, 200)
        events = []
        ctx.emit(events, g, 1500, 1600)
        self.assertIn("\\pos(300.00,200.00)", events[0].text)
        events = []
        ctx.emit(events, g, 1100, 1200)
        self.assertIn("\\move(250.00,190.00,300.00,200.00,0,100)", events[0].text)

    def test_ktv_slots_waits_and_song_boundaries(self):
        doc, _ = ktv_fixture(starts=(1000, 2000, 2500, 2100))
        views = {m.id: kara.effective(doc, m) for m in doc.members.values() if m.lane == "original"}
        slots = ktv.song_slots(doc, views)
        self.assertEqual([slots[f"JP_{i}"].index for i in range(4)], [0, 1, 0, 1])
        self.assertEqual([slots[f"JP_{i}"].start for i in range(4)], [400, 1400, 2200, 3200])
        self.assertEqual(slots["CN_2"], slots["JP_2"])
        doc.groups.insert(2, Group("gap", "dialogue"))
        self.assertEqual(ktv.song_slots(doc, views), slots)
        doc.groups.pop(2)
        for m in list(doc.members.values())[4:]:
            m.style = m.style.replace("OP", "ED")
        self.assertNotIn("JP_2", ktv.song_slots(doc, views))
        views["JP_2"].params["ktv"] = "stacked"
        self.assertEqual(ktv.song_slots(doc, views)["JP_2"].index, 0)
        for value in (-1, 10001, True):
            with self.assertRaises(ExportError):
                ktv.PARAMETERS["line_advance_ms"].validate(value)

    def test_song_span_credits_gap_boundary_and_families(self):
        doc, _ = ktv_fixture(starts=(1000, 17200, 33401, 34401))
        doc.groups.insert(1, Group("credits", "sign", ["credit"]))
        doc.members["credit"] = Member("credit", "credits", "ja", "original", "sign", 0, 999999)
        # Credit timing never bridges a lyric gap. Exactly 15 seconds is included.
        self.assertEqual(song_span(doc, "g_0"), ["g_0", "g_1"])
        self.assertEqual(song_span(doc, "g_1"), ["g_0", "g_1"])
        self.assertEqual(song_span(doc, "g_2"), ["g_2", "g_3"])
        self.assertEqual(song_span(doc, "credits"), [])
        self.assertEqual(song_span(doc, "missing"), [])
        views = {m.id: kara.effective(doc, m) for m in doc.members.values() if m.role == "song"}
        self.assertNotIn("JP_2", ktv.song_slots(doc, views))
        views["JP_2"].params["ktv"] = "corners"
        slots = ktv.song_slots(doc, views)
        self.assertEqual([slots[f"JP_{i}"].index for i in range(4)], [0, 1, 0, 1])
        # A different song family is a boundary even if OP appears again later.
        for mid in doc.groups[-2].members:
            doc.members[mid].style = "ED.JP"
        self.assertEqual(song_span(doc, "g_3"), ["g_3"])
        for mid in doc.groups[-2].members:
            doc.members[mid].style = "IN.JP"
        self.assertEqual(song_span(doc, "g_2"), ["g_2"])

    def test_ktv_alternation_continues_across_staff_credits(self):
        doc, _ = ktv_fixture(starts=(1000, 3000, 5000))
        doc.groups.insert(1, Group("staff1", "sign"))
        doc.groups.insert(3, Group("staff2", "info"))
        before = copy.deepcopy(doc)
        views = {m.id: kara.effective(doc, m) for m in doc.members.values()}
        slots = ktv.song_slots(doc, views)
        self.assertEqual(song_span(doc, "g_1"), ["g_0", "g_1", "g_2"])
        self.assertEqual([slots[f"JP_{i}"].index for i in range(3)], [0, 1, 0])
        self.assertEqual(slots["CN_1"], slots["JP_1"])
        self.assertEqual(doc, before)

    def test_schema_defaults_ranges_labels_and_seed(self):
        for name in REGISTRY.keys()-INLINE.keys():
            fx = get_effect(name)
            self.assertTrue(fx.display_name)
            defaults = fx.resolve()
            self.assertEqual(defaults["lead_in_ms"], 300)
            self.assertTrue(defaults["follow_furigana"])
            for key, p in fx.schema.items():
                self.assertTrue(p.label)
                if p.minimum is not None:
                    fx.resolve({key: p.minimum})
                    fx.resolve({key: p.maximum})
                    with self.assertRaises(ExportError):
                        fx.resolve({key: p.maximum+1})
            with self.assertRaises(ExportError):
                fx.resolve({"follow_furigana": "yes"})
            self.assertEqual(FxContext.seeded("m", defaults).random(), FxContext.seeded("m", defaults).random())

    def test_timing_determinism_and_cap_boundaries(self):
        g = FxGlyph(Event(style="Default"), (200, 180, 240, 230), "", "H", 200, 180,
                    [(0, 1000, 1500)], {"member": "m", "group": "g", "kind": "base"})
        for name in REGISTRY.keys()-INLINE.keys():
            fx = get_effect(name)
            p = fx.resolve()
            def context(cap=2000):
                return FxContext("m", 1000, 3000, Style.new("Default"), [g], p,
                                 rng=FxContext.seeded("m", p), event_cap=cap)
            events = fx.generate(context())
            self.assertEqual(events, fx.generate(context()))
            self.assertEqual(min(e.start for e in events), 70)
            self.assertEqual(max(e.end for e in events), 165 if name == "fade" else 300)
            self.assertTrue(all(e.end > e.start for e in events))
            self.assertTrue(all(e.layer >= 0 for e in events))
            fx.generate(context(len(events)))
            if name == "tremble":
                # This effect deliberately slows down to fit the budget.
                self.assertLessEqual(len(fx.generate(context(len(events)-1))), len(events)-1)
                with self.assertRaises(EventLimit):
                    fx.generate(context(1))
            else:
                with self.assertRaises(EventLimit):
                    fx.generate(context(len(events)-1))
        ctx = context()
        events = []
        ctx.emit(events, g, -8, 4)
        self.assertEqual(events, [])
        ctx.emit(events, g, 995, 1005)
        self.assertEqual((events[0].start, events[0].end), (100, 101))

    def test_lower_layers_raise_text_and_preserve_order(self):
        for source_layer in (-1, 0, 3):
            g = FxGlyph(Event(layer=source_layer), (200, 180, 240, 230), "", "H", 200, 180,
                        [(0, 1000, 1500)])
            original = copy.deepcopy(g)
            for name, params in (("glow", {}), ("wipe", {"soft_edge": True})):
                fx = get_effect(name)
                ctx = FxContext("m", 1000, 3000, Style.new("Default"), [g], fx.resolve(params))
                events = fx.generate(ctx)
                self.assertTrue(all(e.layer >= 0 for e in events))
                base = next(e for e in events if e.start == 70)
                self.assertEqual(base.layer, max(1, source_layer))
                below = next(e for e in events if "\\blur" in e.text)
                self.assertEqual(below.layer, base.layer-1)
                if name == "wipe":
                    overlay = next(e for e in events if "\\clip" in e.text and "\\blur" not in e.text)
                    self.assertEqual(overlay.layer, base.layer+1)
            self.assertEqual(g, original)

    def test_motion_parameters_and_geometry(self):
        g = FxGlyph(Event(), (200, 180, 240, 230), "", "H", 200, 180,
                    [(0, 1000, 1500)], scale_x=120, scale_y=80)
        def generate(name, values):
            fx = get_effect(name)
            p = fx.resolve(values)
            return fx.generate(FxContext("m", 1000, 3000, Style.new("Default"), [g], p,
                                         rng=FxContext.seeded("m", p)))
        bounce = generate("bounce", {"count": 2, "height": 12})
        moving = [e for e in bounce if "\\move" in e.text]
        self.assertEqual(len(moving), 4)
        self.assertIn("200.00,180.00,200.00,168.00", moving[0].text)
        self.assertIn("200.00,168.00,200.00,180.00", moving[-1].text)
        for e in moving:
            self.assertEqual(e.text.count("\\pos"), 0)
        shake = generate("shake", {"count": 3, "angle": 8})[0]
        self.assertIn("\\org(220.00,205.00)", shake.text)
        self.assertEqual(shake.text.count("\\frz8"), 3)
        self.assertEqual(shake.text.count("\\frz-8"), 3)
        self.assertIn("\\frz0", shake.text)
        for mode in ("拉伸", "挤压", "波浪"):
            text = "".join(e.text for e in generate("morph", {"mode": mode}))
            self.assertIn("\\fscx120\\fscy80", text)
            self.assertNotIn("\\org", text)
            self.assertIn("\\move", text)
        for direction, coord in (("左", "128.00,180.00"), ("右", "272.00,180.00"),
                                 ("上", "200.00,108.00"), ("下", "200.00,252.00")):
            self.assertIn(coord, generate("fly", {"direction": direction})[0].text)
        for after in ("散开", "落下", "淡出"):
            events = generate("fly", {"after": after})
            self.assertIn("\\fad(0,150)", events[-1].text)
            if after == "落下":
                self.assertIn("200.00,180.00,200.00,252.00", events[-1].text)
            if after == "淡出":
                self.assertIn("200.00,180.00,200.00,180.00", events[-1].text)
        paths = set()
        for shape in ("星", "心", "圆", "花瓣"):
            particles = [e for e in generate("particles", {"shape": shape}) if "\\p1" in e.text]
            self.assertEqual(len(particles), 6)
            paths.add(particles[0].text.split("}", 1)[1])
            for e in particles:
                coords = re.search(r"\\move\(([^)]*)\)", e.text).group(1).split(",")
                x0, y0, x1, y1 = map(float, coords[:4])
                self.assertTrue(200 <= x0 <= 240 and 180 <= y0 <= 230)
                self.assertLessEqual(((x1-x0)**2+(y1-y0)**2)**.5, 40.02)
                self.assertEqual((e.start, e.end), (100, 160))
        self.assertEqual(len(paths), 4)

    def test_short_beats_rounding_and_fly_stagger(self):
        glyphs = [FxGlyph(Event(), (200+i*40, 180, 240+i*40, 230), "", "H", 200+i*40, 180,
                          [(0, 1000, 1008)], {"kind": "base"}) for i in range(2)]
        for name in MOTION:
            fx = get_effect(name)
            p = fx.resolve()
            ctx = FxContext("m", 1000, 1100, Style.new("Default"), glyphs, p,
                            rng=FxContext.seeded("m", p))
            events = fx.generate(ctx)
            self.assertTrue(events)
            self.assertTrue(all(e.end > e.start and e.layer >= 0 for e in events))
            self.assertEqual(len(events), len(ctx.owners))
            if name == "fly":
                entries = [e for e in events if e.end == 100]
                self.assertEqual([e.start for e in entries], [70, 74])
        # The glyph-centred origin must replace an existing line origin (ASS uses the first).
        glyphs[0].tags = "\\org(1,2)"
        fx = get_effect("shake")
        event = fx.generate(FxContext("m", 1000, 1100, Style.new("Default"), glyphs[:1], fx.resolve()))[0]
        self.assertEqual(event.text.count("\\org"), 1)
        self.assertIn("\\org(220.00,205.00)", event.text)

    def test_unknown_and_placeholders(self):
        doc, tpl = fixture("unknown")
        with self.assertRaisesRegex(ExportError, "未知的卡拉 OK 特效：unknown"):
            build_ass(doc, {}, tpl, ROOT, ExportOptions(ruby=False))
        doc.karaoke["k_fx"]["effect"]["preset"] = "pop"
        doc.karaoke["k_fx"]["beats"] = []
        ass, rep = build_ass(doc, {}, tpl, ROOT, ExportOptions(ruby=False))
        self.assertGreater(len(ass.events), 1)
        self.assertEqual(rep.karaoke_fx_lines, 1)
        self.assertTrue(any("\\fscx150" in e.text for e in ass.events))
        self.assertTrue(rep.karaoke_unfinished)


@unittest.skipUnless(shutil.which("ffmpeg"), "requires host ffmpeg/libass")
class FxExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fx1-", dir=os.environ.get("TEMP"))
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def export(self, doc, tpl, **kw):
        return build_ass(doc, {}, tpl, self.path, ExportOptions(playres=(960, 540), **kw),
                         measure_cache=self.path / "measure.sqlite3")

    def render_mask(self, ass, moment):
        (self.path / "frame.ass").write_text(ass.serialize(), encoding="utf-8-sig")
        proc = run(["ffmpeg", "-hide_banner", "-loglevel", "verbose", "-f", "lavfi", "-i",
                    "color=c=black:s=960x540:r=100:d=0.01", "-vf",
                    f"setpts=PTS+{moment}/TB,ass=frame.ass,format=rgb24", "-frames:v", "1",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], cwd=self.path,
                   capture_output=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
        frame = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(540, 960, 3)
        # Compare the solid text mask, excluding low-opacity antialias fringes.
        return frame.max(axis=2) >= 128

    def japanese_fixture(self, name="kf", params=None):
        doc, tpl = fixture(name, params)
        m = doc.members["m_fx"]
        m.end = 6000
        text = "潔く生きて行こう"
        m.runs = [Run(text, ins="i_fx", b=len(text))]
        for i, reading in ((0, "いさぎよ"), (2, "い"), (5, "ゆ")):
            aid = f"a_{i}"
            doc.annotations[aid] = Annotation(aid, "ruby", m.id, [{"ins": "i_fx", "a": i, "b": i+1}],
                text[i], reading, status="confirmed", parts=[[0, 1, reading]])
        beats = kara.derive(doc, m)
        for i, b in enumerate(beats):
            b.t0, b.t1, b.by = 2500+i*200, 2700+i*200, "manual"
        doc.karaoke["k_fx"] = kara.beats_to_track(m, beats, effect={"preset": name, "params": params or {}})
        tpl.styles[0].set("Fontname", "Yu Gothic UI")
        tpl.styles[0].set("Fontsize", 72)
        tpl.styles[0].set("Bold", -1)
        return doc, tpl

    def test_libass_every_10ms_colour_and_presence_all_registered(self):
        """FX6: full-line 100 fps, isolated 潔/生, real 1080p libass.

        Sung area is normalised by current fill area: pop/morph shrink and
        shake changes raster area by design. kf/wipe progressively reveal,
        none never highlights; fade/fly and common fades are checked only
        until their documented exits start. Ruby is exported but isolated
        away so it cannot mask a missing base glyph.
        """
        for name in sorted(REGISTRY):
            with self.subTest(effect=name):
                doc, tpl = self.japanese_fixture(name)
                m = doc.members["m_fx"]
                m.start, m.end = 1200, 4800
                doc.annotations["a_5"].text = "い"
                doc.annotations["a_5"].parts = [[0, 1, "い"]]
                beats = kara.derive(doc, m)
                for i, b in enumerate(beats):
                    b.t0 = round(1200+3600*i/len(beats))
                    b.t1 = round(1200+3600*(i+1)/len(beats))
                    b.by = "manual"
                # Keep unsung ink visible here to isolate colour/presence from
                # tremble's intentional ghosting (covered in test_fear_shake).
                params = {"unsung_alpha": 0} if name == "tremble" else {}
                doc.karaoke["k_fx"] = kara.beats_to_track(m, beats, effect={"preset": name, "params": params})
                ass, report = build_ass(doc, {}, tpl, self.path, ExportOptions(playres=(1920,1080)),
                                        measure_cache=self.path/"measure.sqlite3")
                selected = []
                first_beats = []
                exits = []
                for offset in (0, 2):
                    identity = visible_map(m)[1][offset]
                    glyph_events = [e for e, o in zip(ass.events, report.owners)
                                    if o["kind"] == "base" and o["ids"] == [identity]]
                    if name in INLINE:
                        # Inline events are whole lines; isolate with transparent
                        # characters while retaining original k tags and spacing.
                        whole = next(e for e, o in zip(ass.events, report.owners) if o["kind"] == "base")
                        inserts = {i: "{\\1a&HFF&\\3a&HFF&}" for i in range(len("潔く生きて行こう"))}
                        inserts[offset] = "{\\1a&H00&\\3a&H00&}"
                        glyph_events = [whole.copy(text=insert_at_visible(whole.text, inserts))]
                    # Put the two independent glyph timelines in separate renders
                    # packed side by side by ffmpeg; neighbours cannot hide a flash.
                    isolated = copy.deepcopy(ass)
                    isolated.events = glyph_events
                    path = self.path/f"fx6-{offset}.ass"
                    path.write_text(isolated.serialize(), encoding="utf-8-sig")
                    selected.append(path.name)
                    own = [b for b in beats if b.a <= offset < b.b]
                    first_beats.append(min(b.t0 for b in own))
                    exits.append(max(b.t1 for b in own) if name == "fade" else 4650)
                proc = run(["ffmpeg", "-hide_banner", "-loglevel", "verbose", "-f", "lavfi", "-i",
                            "color=c=black:s=1920x1080:r=100:d=4.81", "-filter_complex",
                            f"[0:v]split[a][b];[a]ass={selected[0]},crop=480:260:100:80,format=rgb24[a1];"
                            f"[b]ass={selected[1]},crop=480:260:100:80,format=rgb24[b1];[a1][b1]hstack",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], cwd=self.path,
                           capture_output=True, timeout=40)
                self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
                frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1,260,960,3)
                for i in range(2):
                    rgb = frames[:,:,i*480:(i+1)*480,:]
                    fill = rgb[:,:,:,0] > 80
                    sung = fill & (rgb[:,:,:,1] > 80) & (np.abs(rgb[:,:,:,0].astype(int)-rgb[:,:,:,1]) < 15)
                    area = fill.sum(axis=(1,2))
                    ratios = sung.sum(axis=(1,2))/np.maximum(1, area)
                    # Lead-in is finished by 1200; fade-in never masks a gap.
                    lo, hi = max(120, (first_beats[i]+9)//10), exits[i]//10
                    self.assertTrue(np.all(area[120:hi] > 20), (name, i, "missing glyph"))
                    if name != "none":
                        drops = np.diff(ratios[lo:hi])
                        self.assertTrue(np.all(drops >= -.015),
                                        (name, i, "sung coverage regressed", float(drops.min())))
                    if name == "pop" and i == 0:
                        self.assertTrue(np.all(ratios[247:250] > .98), "2.48 s hand-over flash")
                        print(f"FX6 pop sung coverage at 2.47/2.48/2.49 s: {ratios[247:250].tolist()}")
                print(f"FX6 libass {name}: {len(frames)} frames at 100 fps, 潔/生 present; sung coverage retained (exit exemptions above)")
                print(next((s for s in proc.stderr.decode('utf-8','replace').splitlines() if 'libass API' in s), ''))

    def test_libass_idle_positions_match_kf_every_registered_effect(self):
        doc, tpl = self.japanese_fixture()
        plain, _ = self.export(doc, tpl)
        expected = self.render_mask(plain, 1.8)
        self.assertTrue(expected.any())

        def expanded(mask):
            pad = np.pad(mask, 1)
            return np.logical_or.reduce([pad[y:y+540, x:x+960] for y in range(3) for x in range(3)])

        for name in sorted(REGISTRY):
            with self.subTest(effect=name):
                params = {"jitter_x": 0, "jitter_y": 0, "unsung_alpha": 0} if name == "tremble" else {}
                doc.karaoke["k_fx"]["effect"] = {"preset": name, "params": params}
                ass, _ = self.export(doc, tpl)
                actual = self.render_mask(ass, 1.8)
                self.assertFalse(np.any(actual & ~expanded(expected)), name + " extra/shifted ink")
                self.assertFalse(np.any(expected & ~expanded(actual)), name + " missing/shifted ink")
                print(f"FX5 libass idle {name}: full text/ruby binary masks match kf within 1 px")

    def test_libass_pop_peak_symmetric_and_morph_bottom_fixed(self):
        doc, tpl = self.japanese_fixture("pop")
        ass, rep = self.export(doc, tpl)
        # Isolate the real first glyph, so adjacent text/ruby cannot hide drift.
        ass.events = [e for e, o in zip(ass.events, rep.owners)
                      if o["kind"] == "base" and o["ids"] == [rep.owners[0]["ids"][0]]]
        idle = self.render_mask(ass, 1.8)
        peak = self.render_mask(ass, 2.6)
        y0, x0 = np.where(idle)
        y1, x1 = np.where(peak)
        left, right = x0.min()-x1.min(), x1.max()-x0.max()
        top, bottom = y0.min()-y1.min(), y1.max()-y0.max()
        self.assertGreater(min(left, right, top, bottom), 5)
        self.assertLessEqual(abs(left-right), 2)
        self.assertLessEqual(abs(top-bottom), 2)
        print(f"FX5 libass pop peak growth: left/right={left}/{right} px, top/bottom={top}/{bottom} px")
        for mode, moment in (("拉伸", 2.6), ("挤压", 2.6), ("波浪", 2.55)):
            doc.karaoke["k_fx"]["effect"] = {"preset": "morph", "params": {"mode": mode}}
            morph, report = self.export(doc, tpl)
            morph.events = [e for e, o in zip(morph.events, report.owners)
                            if o["kind"] == "base" and o["ids"] == [rep.owners[0]["ids"][0]]]
            changed = self.render_mask(morph, moment)
            yy, xx = np.where(changed)
            self.assertLessEqual(abs(yy.max()-y0.max()), 1, mode)
            self.assertGreater(abs((xx.max()-xx.min())-(x0.max()-x0.min())), 3, mode)
            print(f"FX5 libass morph {mode}: bottom drift={yy.max()-y0.max()} px, width={xx.max()-xx.min()+1} px")

    def test_ktv_inline_timing_translation_placeholder_and_zero_report(self):
        doc, tpl = ktv_fixture(starts=(1000, 2000, 2100))
        original = copy.deepcopy(doc)
        ass, rep = self.export(doc, tpl, ruby=False)
        events = {o["member"]: e for e, o in zip(ass.events, rep.owners)}
        self.assertEqual([events[f"JP_{i}"].start for i in range(3)], [40, 140, 220])
        self.assertEqual([events[f"JP_{i}"].end for i in range(3)], [220, 320, 330])
        self.assertIn("{\\kf60}", events["JP_0"].text)
        self.assertIn("\\an1", events["JP_0"].text)
        self.assertIn("\\an3", events["JP_1"].text)
        self.assertIn("\\pos(860,490)", events["JP_1"].text)
        self.assertIn("\\pos(860,430)", events["CN_1"].text)
        self.assertEqual(events["CN_1"].start, events["JP_1"].start)
        self.assertIn("{\\kf60}{\\kf120}Hi", events["CN_1"].text)
        self.assertIn("0:00:02.10 双行卡拉 OK：换行提前已缩短为 0 毫秒", rep.warnings)
        self.assertEqual(doc, original)
        # An untapped original follows the same slot with placeholder beats.
        del doc.karaoke["JP_1"]
        placeholder, report = self.export(doc, tpl, ruby=False)
        e = next(e for e, o in zip(placeholder.events, report.owners) if o["member"] == "JP_1")
        self.assertEqual(e.start, 140)
        self.assertIn("{\\kf60}{\\kf120}HH", e.text)
        self.assertTrue(report.karaoke_unfinished)

    def test_ktv_furigana_effects_geometry_and_window(self):
        for name in REGISTRY.keys():
            with self.subTest(effect=name):
                doc, tpl = ktv_fixture(name, starts=(1000, 2000, 2100), ruby=True)
                ass, rep = self.export(doc, tpl)
                self.assertTrue(any(o["kind"] == "ruby" for o in rep.owners))
                self.assertEqual(len(ass.events), len(rep.owners))
                if name == "fly":
                    self.assertTrue(any("\\move" in e.text and e.start == 40 and e.end == 100
                                        for e,o in zip(ass.events,rep.owners) if o["member"] == "JP_0"))
                for e, owner in zip(ass.events, rep.owners):
                    i = int(owner["member"].rsplit("_", 1)[1])
                    self.assertGreaterEqual(e.start, (40, 140, 220)[i])
                    self.assertLessEqual(e.end, (220, 320, 330)[i])
                li = LayoutIndex(ass, rep.owners)
                try:
                    geometry = li.geometry(li.active(1100))
                    upper = [g for g in geometry if g.member == "JP_0" and g.kind == "base"]
                    ruby = [g for g in geometry if g.member == "JP_0" and g.kind == "ruby"]
                    lower = [g for g in li.geometry(li.active(2100)) if g.member == "JP_1" and g.kind == "base"]
                    self.assertTrue(upper and lower)
                    self.assertTrue(ruby)
                    self.assertLess(min(g.rect.y0 for g in ruby), min(g.rect.y0 for g in upper))
                    self.assertLess(min(g.rect.y0 for g in upper), 250)
                    self.assertGreater(min(g.rect.y0 for g in lower), 350)
                    self.assertLess(min(g.rect.x0 for g in upper), 200)
                    self.assertGreater(min(g.rect.x0 for g in lower), 650)
                finally:
                    li.close()

    def test_ktv_cache_dependencies_and_off_identity(self):
        for name in REGISTRY.keys():
            doc, tpl = ktv_fixture(name, "off")
            with_off, _ = self.export(doc, tpl, ruby=False)
            doc.karaoke["JP_0"]["effect"]["params"] = {}
            without, _ = self.export(doc, tpl, ruby=False)
            self.assertEqual(with_off.serialize(), without.serialize(), name)
        doc, tpl = ktv_fixture()
        cache = ExportCache()
        def export(mode="release"):
            return build_ass(doc, {}, tpl, self.path, ExportOptions(mode=mode, playres=(960,540)), cache=cache)
        first, _ = export()
        preview, _ = export("preview")
        self.assertEqual(first.serialize(), preview.serialize())
        same, report = export("preview")
        self.assertEqual(report.laid_out, 0)
        doc.members["JP_0"].end = 2800
        changed, report = export("preview")
        self.assertEqual(next(e.start for e,o in zip(changed.events,report.owners) if o["member"] == "JP_2"), 280)
        self.assertGreaterEqual(report.laid_out, 3)
        doc.karaoke["JP_0"]["effect"]["params"]["ktv"] = "stacked"
        centred, _ = export("preview")
        self.assertTrue(all("\\an2\\pos(480," in e.text for e in centred.events))

    def test_ktv_translation_own_beats_fallback_and_untimed(self):
        doc, tpl = ktv_fixture("glow", ruby=True)
        cn = doc.members["CN_0"]
        bs = kara.derive(doc, cn)
        for b in bs:
            b.t0, b.t1, b.by = 1100, 1900, "manual"
        doc.karaoke[cn.id] = kara.beats_to_track(cn, bs, effect={"preset": "kf"})
        advanced, rep = self.export(doc, tpl)
        self.assertEqual(rep.karaoke_fx_lines, 6)
        self.assertTrue(any(e.start == 110 for e,o in zip(advanced.events,rep.owners) if o["member"] == cn.id))
        with patch.object(get_effect("glow"), "generate", side_effect=EventLimit):
            fallback, report = self.export(doc, tpl)
        for t in doc.karaoke.values():
            t["effect"]["preset"] = "kf"
        inline, _ = self.export(doc, tpl)
        self.assertEqual(fallback.serialize(), inline.serialize())
        self.assertTrue(any("已改用渐变扫过" in w for w in report.warnings))
        doc.members["JP_0"].end = doc.members["JP_0"].start
        untimed, report = self.export(doc, tpl)
        self.assertEqual(next(e.kind for e,o in zip(untimed.events,report.owners) if o["member"] == "JP_0"), "Comment")
        self.assertTrue(report.not_timed)

    def test_ktv_libass_positions_predisplay_and_colours(self):
        for name, mode in (("kf", "corners"), ("kf", "stacked"), ("glow", "corners"), ("particles", "stacked")):
            with self.subTest(effect=name, mode=mode):
                doc, tpl = ktv_fixture(name, mode)
                ass, rep = self.export(doc, tpl, ruby=False)
                (self.path/"ktv.ass").write_text(ass.serialize(), encoding="utf-8-sig")
                proc = run(["ffmpeg", "-hide_banner", "-loglevel", "verbose", "-f", "lavfi", "-i",
                            "color=c=black:s=960x540:r=20:d=4.3", "-vf", "ass=ktv.ass,format=rgb24",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], cwd=self.path,
                           capture_output=True, timeout=30)
                self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
                frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1,540,960,3)
                upper_x = slice(60, 250) if mode == "corners" else slice(380, 580)
                lower_x = slice(650, 900) if mode == "corners" else slice(380, 580)
                self.assertFalse(np.any(frames[7]))  # before first pre-display
                upper = frames[16, :250, upper_x]
                self.assertGreater(np.count_nonzero(upper), 0)
                # Before the beat, the red unsung fill dominates the white outline.
                self.assertGreater(np.count_nonzero(upper[:,:,0].astype(int) > upper[:,:,1].astype(int)+10), 20)
                self.assertGreater(np.count_nonzero(frames[32, 350:, lower_x]), 0)
                self.assertFalse(np.array_equal(frames[32,350:,lower_x], frames[43,350:,lower_x]))
                self.assertGreater(np.count_nonzero(frames[52,:250,upper_x]), 0)  # third line reuses A
                print(f"FX3 libass {name}/{mode}: A/B/A positions, pre-display and beat colour; events={len(ass.events)}")
                print(next((s for s in proc.stderr.decode('utf-8','replace').splitlines() if 'libass API' in s), ''))

    def test_preview_cache_params_and_generated_hit_ownership(self):
        doc, tpl = fixture()
        cache = ExportCache()
        def export(mode):
            return build_ass(doc, {}, tpl, self.path, ExportOptions(mode=mode, playres=(960, 540), ruby=False), cache=cache)
        ass, rep = export("release")
        preview, _ = export("preview")
        self.assertEqual(ass.serialize(), preview.serialize())
        same, report = export("preview")
        self.assertEqual(report.laid_out, 0)
        self.assertEqual(preview.serialize(), same.serialize())
        doc.karaoke["k_fx"]["effect"]["params"] = {"scale": 200}
        changed, report = export("preview")
        self.assertEqual(report.laid_out, 1)
        self.assertNotEqual(changed.serialize(), same.serialize())
        self.assertEqual((rep.karaoke_fx_lines, rep.karaoke_fx_events), (1, len(ass.events)))
        self.assertIn(f"卡拉 OK 特效：1 行生成了 {len(ass.events)} 条字幕行", rep.warnings)
        li = LayoutIndex(ass, rep.owners)
        try:
            gs = li.geometry(li.active(800))
            self.assertTrue(gs)
            g = gs[0]
            hit = li.hit(gs, (g.rect.x0+g.rect.x1)/2, (g.rect.y0+g.rect.y1)/2)
            self.assertEqual((hit.line.member, hit.line.group), ("m_fx", "g_fx"))
            self.assertIsNotNone(hit.identity)
        finally:
            li.close()
        doc, tpl = fixture("pop", ruby=True)
        cache = ExportCache()
        first, _ = build_ass(doc, {}, tpl, self.path, ExportOptions(playres=(960,540)), cache=cache)
        cached, report = build_ass(doc, {}, tpl, self.path, ExportOptions(playres=(960,540)), cache=cache)
        self.assertEqual(report.laid_out, 0)
        self.assertEqual(first.serialize(), cached.serialize())
        self.assertTrue(all(cached.style(e.style) is not None for e in cached.events))

    def test_ruby_follow_translation_and_cap_fallback(self):
        doc, tpl = fixture("pop", ruby=True)
        ass, rep = self.export(doc, tpl)
        self.assertTrue(any(o["kind"] == "ruby" for o in rep.owners))
        self.assertEqual(len(ass.events), len(rep.owners))
        self.assertTrue(all(o["member"] == "m_fx" for o in rep.owners))
        self.assertTrue(all("\\kf" not in e.text for e in ass.events))
        doc.karaoke["k_fx"]["effect"]["params"] = {"follow_furigana": False}
        ass, _ = self.export(doc, tpl)
        self.assertTrue(any("\\kf" in e.text for e in ass.events if e.style.endswith(".Ruby")))
        cn = Member("m_cn", "g_fx", "zh-Hans", "translation", "song", 1000, 3000, style="Default",
                    runs=[Run("Hi", ins="i_cn", b=2)])
        doc.members[cn.id] = cn; doc.groups[0].members.append(cn.id)
        bs = kara.derive(doc, cn)
        for b in bs: b.t0,b.t1,b.by=1100,1900,"manual"
        doc.karaoke["k_cn"] = kara.beats_to_track(cn, bs, effect={"preset": "k"})
        _, rep = self.export(doc, tpl)
        self.assertEqual(rep.karaoke_fx_lines, 2)
        doc.karaoke["k_fx"]["effect"]["params"]["follow_translation"] = False
        _, rep = self.export(doc, tpl)
        self.assertEqual(rep.karaoke_fx_lines, 1)
        doc, tpl = fixture("pop", ruby=True)
        original = copy.deepcopy(doc)
        with patch.object(get_effect("pop"), "generate", side_effect=EventLimit):
            fallback, report = self.export(doc, tpl)
        original.karaoke["k_fx"]["effect"] = {"preset": "kf"}
        normal, _ = self.export(original, tpl)
        self.assertEqual(normal.serialize(), fallback.serialize())
        self.assertIn("0:00:01.00 这一行特效太复杂，已改用渐变扫过", report.warnings)

    def test_libass_before_during_after_all_effects(self):
        for name in ("pop", "glow", "color", "fade", "wipe") + MOTION:
            with self.subTest(effect=name):
                doc, tpl = fixture(name)
                ass, _ = self.export(doc, tpl, ruby=False)
                (self.path / "fx.ass").write_text(ass.serialize(), encoding="utf-8-sig")
                proc = run(["ffmpeg", "-hide_banner", "-loglevel", "verbose", "-f", "lavfi", "-i",
                            "color=c=black:s=960x540:r=20:d=3.2", "-vf", "ass=fx.ass,format=rgb24",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], cwd=self.path,
                           capture_output=True, timeout=30)
                self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
                frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1,540,960,3)
                # fly keeps sung text to line end, then exits at 2.85..3.00 s.
                moments = (17, 23, 59 if name == "fly" else 39)
                regions = [frames[k, 160:300, 180:310] for k in moments]
                self.assertGreater(np.count_nonzero(regions[0]), 0)
                self.assertFalse(np.array_equal(regions[0], regions[1]), name)
                self.assertFalse(np.array_equal(regions[1], regions[2]), name)
                print(f"FX libass {name}: before/during/after changed; events={len(ass.events)}")
                print(next((s for s in proc.stderr.decode('utf-8','replace').splitlines() if 'libass API' in s), ''))

    def test_libass_particle_shapes_and_fly_morph_variants(self):
        variants = [("particles", {"shape": s}) for s in ("星", "心", "圆", "花瓣")]
        variants += [("fly", {"direction": d, "after": a}) for d, a in
                     (("左", "散开"), ("右", "落下"), ("上", "淡出"), ("下", "散开"))]
        variants += [("morph", {"mode": m}) for m in ("拉伸", "挤压", "波浪")]
        shape_frames = []
        for name, params in variants:
            with self.subTest(effect=name, params=params):
                doc, tpl = fixture(name, params)
                ass, _ = self.export(doc, tpl, ruby=False)
                if name == "particles":
                    # Render only the generated drawings: text cannot mask an invalid path.
                    ass.events = [e for e in ass.events if "\\p1" in e.text]
                (self.path / "variant.ass").write_text(ass.serialize(), encoding="utf-8-sig")
                proc = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "color=c=black:s=960x540:r=20:d=3.2", "-vf", "ass=variant.ass,format=rgb24",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], cwd=self.path,
                           capture_output=True, timeout=30)
                self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
                frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1,540,960,3)
                region = frames[23, 100:340, 80:380]
                self.assertGreater(np.count_nonzero(region), 0)
                if name == "particles":
                    self.assertFalse(np.any(frames[17]))
                    self.assertFalse(np.any(frames[39]))
                    shape_frames.append(region.copy())
                elif name == "fly":
                    self.assertFalse(np.array_equal(frames[23], frames[59]))
                else:
                    self.assertFalse(np.array_equal(frames[23], frames[39]))
        for i, frame in enumerate(shape_frames):
            for other in shape_frames[i+1:]:
                self.assertFalse(np.array_equal(frame, other))
        print("FX2 libass: four isolated vector paths, four fly directions/three exits, three morph modes rendered")

    def test_motion_deterministic_preview_furigana_and_default_op_counts(self):
        for name in MOTION:
            with self.subTest(effect=name):
                doc, tpl = fixture(name, ruby=True)
                cache = ExportCache()
                def export(mode="release"):
                    return build_ass(doc, {}, tpl, self.path, ExportOptions(mode=mode, playres=(960, 540)), cache=cache)
                first, rep = export()
                second, _ = export("preview")
                # Release may regroup generated annotation lines; all event fields and styles remain identical.
                self.assertEqual(first.script_info, second.script_info)
                self.assertEqual(first.event_format, second.event_format)
                self.assertEqual([s.to_line(first.style_format) for s in first.styles],
                                 [s.to_line(second.style_format) for s in second.styles])
                # Fully positioned effects may need derived Layer numbers to
                # retain their old compositing after annotation rows move.
                self.assertCountEqual([e.copy(layer=0).to_line(first.event_format) for e in first.events],
                                      [e.copy(layer=0).to_line(second.event_format) for e in second.events])
                cache.clear()
                regenerated, _ = export()
                self.assertEqual(first.serialize(), regenerated.serialize())
                self.assertEqual(len(first.events), len(rep.owners))
                ruby = [e for e, o in zip(first.events, rep.owners) if o["kind"] == "ruby"]
                self.assertTrue(ruby)
                expected = {"bounce": "\\move", "shake": "\\org", "particles": "\\1c",
                            "fly": "\\move", "morph": "\\fscx"}[name]
                self.assertTrue(any(expected in e.text for e in ruby))
                doc.karaoke["k_fx"]["effect"]["params"] = {"follow_furigana": False}
                unfollowed, _ = self.export(doc, tpl)
                self.assertTrue(any("\\kf" in e.text for e in unfollowed.events if e.style.endswith(".Ruby")))
                doc, tpl = fixture(name)
                m = doc.members["m_fx"]
                m.runs = [Run("生"*12, ins="i_fx", b=12)]
                m.end = 5000
                for i in range(12):
                    aid = f"a_{i}"
                    doc.annotations[aid] = Annotation(aid, "ruby", m.id, [{"ins": "i_fx", "a": i, "b": i+1}],
                        "生", "い", status="confirmed", parts=[[0, 1, "い"]])
                beats = kara.derive(doc, m)
                self.assertEqual(len(beats), 12)
                for i, b in enumerate(beats):
                    b.t0, b.t1, b.by = 1000+i*300, 1300+i*300, "manual"
                doc.karaoke["k_fx"] = kara.beats_to_track(m, beats, effect={"preset": name})
                op, report = self.export(doc, tpl)
                self.assertTrue(all(e.layer >= 0 for e in op.events))
                self.assertLess(len(op.events), 2000)
                self.assertEqual(report.karaoke_fx_lines, 1)
                self.assertEqual(report.karaoke_fx_events, len(op.events))
                print(f"FX2 OP 12 beats + 12 ruby {name}: {len(op.events)} events / cap 2000")


if __name__ == "__main__":
    unittest.main()
