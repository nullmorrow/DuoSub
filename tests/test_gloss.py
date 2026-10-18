"""Gloss syntax, isolated export events and real libass target measurements."""

import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subkit.ass.document import AssDocument, Style
from subkit.ass.text import VisibleText, line_tag
from subkit.ass.state import state_at
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.layout_index import LayoutIndex
from subkit.core.readings import anchor_of, visible_map
from subkit.gloss import gloss_anchor, parse_glosses
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.model.validate import validate
from subkit.ruby.measure import Box, LineRequest, Measurer
from subkit.ruby.generate import RubyGenerator
from subkit.proc import run
from subkit_app.views.check_result import issue_lines


def fixture(raw="A国B{@注:国=Country}", lane="original", kind="dialogue"):
    m = Member("m", "g", "ja" if lane == "original" else "zh-Hans", lane, kind, 1000, 3000,
               style="Default", runs=[Run(raw, ins="i", b=len(raw))])
    doc = Document("gloss", groups=[Group("g", kind, [m.id])], members={m.id: m})
    template = AssDocument()
    template.styles = [Style.new("Default", Fontname="Arial", Fontsize=60, Outline=4, Shadow=2,
                                Alignment=7, MarginL=200, MarginV=100, PrimaryColour="&H00112233")]
    return doc, template


class Parser(unittest.TestCase):
    def test_aliases_fullwidth_and_option_order(self):
        for name in ("注", "g", "gloss", "GLOSS"):
            g = parse_glosses(f"国{{@{name}：国＝Country；color＝#12aBcD；font＝Missing Font；9；-2.5；size＝17}}")[0]
            self.assertEqual((g.start, g.end, g.position, g.distance, g.font, g.size, g.color),
                             (0, 1, 9, -2.5, "Missing Font", 17, "#12aBcD"))

    def test_legacy_empty_target_and_multiple_marks(self):
        gs = parse_glosses("国{@注:国=Country,80}{@g:国=Again;2}{@gloss:=Whole}")
        self.assertEqual([(g.start, g.end, g.scale, g.position) for g in gs], [(0, 1, 80, 8), (0, 1, None, 2), (0, 1, None, 8)])
        g = parse_glosses("{@注:=First}AB\\NCD")[0]
        self.assertEqual((g.start, g.end, g.text), (0, 4, "First"))

    def test_nearest_occurrence_before_tie_and_overlapping(self):
        self.assertEqual(parse_glosses("国{@注:国=X}国")[0].start, 0)
        self.assertEqual(parse_glosses("国xx国{@注:国=X}")[0].start, 3)
        self.assertEqual(parse_glosses("国xx{@注:国=X}国")[0].start, 3)
        self.assertEqual(parse_glosses("aaa{@注:aa=X}")[0].start, 1)
        self.assertEqual(parse_glosses("{@注:国=X}xx国")[0].start, 2)

    def test_errors_verbatim_and_nonfinite(self):
        cases = [("无=X", "找不到被注的字「无」"), ("国=X;0", "位置要写 1–9"),
                 ("国=X;8;no", "距离要是数字"), ("国=X;8;nan", "距离要是数字"),
                 ("国=X;size=0", "字号要是正数"), ("国=X;size=inf", "字号要是正数"),
                 ("国=X;color=red", "颜色要写成 #RRGGBB"), ("国=X;what=1", "不认识的选项「what」")]
        for body, reason in cases:
            mark = "{@注:" + body + "}"
            warnings = []
            self.assertEqual(parse_glosses("国"+mark, warnings), [])
            self.assertEqual(warnings, [f"旁注写法有误：{mark}（{reason}）"])

    def test_nine_grid_signed_distance(self):
        expected = [(8, 42), (20, 42), (32, 42), (8, 30), (20, 30), (32, 30), (8, 18), (20, 18), (32, 18)]
        ink = Box(-13, 7, -5, 11)
        for pos, point in enumerate(expected, 1):
            x, y = gloss_anchor(Box(10, 30, 20, 40), ink, pos, 2)
            column, row = (pos - 1) % 3, (pos - 1) // 3
            edge = (x + (ink.x1, ink.cx, ink.x0)[column],
                    y + (ink.y0, (ink.y0 + ink.y1) / 2, ink.y1)[row])
            self.assertEqual(edge, point)
        x, y = gloss_anchor(Box(10, 30, 20, 40), ink, 9, -3)
        self.assertEqual((x + ink.x0, y + ink.y1), (27, 23))


class Export(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def export(self, doc, template, **kw):
        return build_ass(doc, {}, template, Path(self.tmp.name), **kw)

    def test_inherited_explicit_and_existing_style(self):
        for options, font, size, color in (("", "Arial", 40, "&H665544&"),
                                           (";font=Missing Font;size=22;color=#AABBCC", "Missing Font", 22, "&HCCBBAA&")):
            doc, tpl = fixture(r"A{\fnArial\fs80\1c&H665544&\b1\i1\bord6}国{@注:国=Country"+options+"}")
            tpl.styles.append(Style.new("Default.Gloss", Fontname="Other", Fontsize=11, PrimaryColour="&H00ABCDEF"))
            ass, rep = self.export(doc, tpl)
            event = next(e for e in ass.events if e.style.endswith(".Gloss"))
            self.assertEqual(state_at(event.text, 0, ("fn",))[0]["fn"].arg, font)
            self.assertEqual(line_tag(event.text, "fs").num(), size)
            colors = [t for tok in VisibleText(event.text).tokens for t in tok.tags if t.name == "1c"]
            self.assertEqual(colors[-1].arg, color)
            self.assertIn("\\b1\\i1", event.text)
            self.assertIn(f"\\xbord{6*size/80:g}", event.text)
            self.assertEqual((event.start, event.end, event.style), (100, 300, "Default.Gloss"))
            self.assertEqual(rep.owners[-1]["kind"], "gloss")

    def test_edge_from_gloss_style_without_line_override(self):
        doc, tpl = fixture()
        tpl.styles.append(Style.new("Default.Gloss", Fontname="Arial", Fontsize=30, Outline=2.5, Shadow=1))
        ass, _ = self.export(doc, tpl)
        event = next(e for e in ass.events if e.style == "Default.Gloss")
        self.assertNotIn("bord", event.text)
        self.assertNotIn("shad", event.text)
        self.assertEqual(ass.style("Default.Gloss").outline, 2.5)
        doc, tpl = fixture()                                  # no .Gloss style: derived like furigana
        ass, _ = self.export(doc, tpl)
        self.assertEqual((ass.style("Default.Gloss").outline, ass.style("Default.Gloss").shadow), (2.4, 1))

    def test_measured_anchor_pos_move_rotation_fades(self):
        for motion in (r"\pos(500,300)", r"\move(500,300,600,320,100,900)"):
            for fade in (r"\fad(100,200)", r"\fade(255,0,255,0,100,1800,2000)"):
                doc, tpl = fixture("{"+motion+r"\org(400,200)\frz15"+fade+"}A国B{@注:国=Country;8;12}")
                ass, _ = self.export(doc, tpl)
                base, gloss = ass.events
                with Measurer(ass, raster_scale=4) as measurer:
                    gen = RubyGenerator(ass, measurer)
                    an, x = gen._anchor(base)
                    box = measurer.line_char_boxes([LineRequest(base.style, base.text, an, x)])[0][1]
                    target_top = gen.block_top(base, 1) + box.y0
                    px = ass.playres[0] / 2
                    measured = measurer.line_char_boxes([LineRequest(gloss.style, gloss.text + r"\N" + gloss.text, 2, px)])[0]
                    count = len(VisibleText(gloss.text).text)
                    height = measured[count].y0 - measured[0].y0
                    ink = [b for b in measured[:count] if b is not None]
                    bottom = max(b.y1 for b in ink) - height
                    center = (min(b.x0 for b in ink) + max(b.x1 for b in ink)) / 2 - px
                tag = line_tag(gloss.text, "move" if "move" in motion else "pos")
                args = tag.float_args()
                self.assertEqual(line_tag(gloss.text, "an").num(), 2)
                self.assertAlmostEqual(args[0] + center, box.cx, delta=0.011)
                self.assertAlmostEqual(args[1] + bottom, target_top - 14, delta=0.011)
                if "move" in motion:
                    self.assertAlmostEqual(args[2]-args[0], 100)
                    self.assertAlmostEqual(args[3]-args[1], 20)
                    self.assertEqual(args[4:], [100, 900])
                self.assertIn(r"\org(400,200)", gloss.text)
                self.assertEqual(state_at(gloss.text, 0, ("frz",))[0]["frz"].num(), 15)
                self.assertIn(fade, gloss.text)

    def test_real_ink_edges_all_positions_and_signed_project_gap(self):
        doc, tpl = fixture()
        doc.groups, doc.members = [], {}
        tpl.styles[0].set("Outline", 0)
        tpl.styles[0].set("Shadow", 0)
        tpl.styles[0].set("PrimaryColour", "&H0000FF00")
        cases = [(p, d) for d in (0, 3, -4) for p in range(1, 10)]
        for i, (pos, distance) in enumerate(cases):
            mid, gid = f"m{i}", f"g{i}"
            text = (r"{\an5\pos(480,270)\fnArial\fs60\fscx120\fscy130\i1}Hg"
                    + f"{{@注:Hg=Agj;{pos};{distance};font=Times New Roman;size=24;color=#FF0000}}")
            doc.members[mid] = Member(mid, gid, "en", "original", "dialogue", (i+1)*1000, (i+1)*1000+800,
                                      style="Default", runs=[Run(text, ins=mid, b=len(text))])
            doc.groups.append(Group(gid, "dialogue", [mid]))
        ass, _ = self.export(doc, tpl, opts=ExportOptions(playres=(960, 540), ruby_gap=5))
        glosses = [e for e in ass.events if e.style.endswith(".Gloss")]
        self.assertEqual([line_tag(e.text, "an").num() for e in glosses], [10-p for p, _d in cases])
        path = Path(self.tmp.name)
        ass.save(path / "ink.ass")
        for scale in (1, 2):
            proc = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"color=c=black:s={960*scale}x{540*scale}:r=1:d={len(cases)}", "-vf",
                        "setpts=PTS+1.4/TB,ass=ink.ass,format=rgb24", "-frames:v", str(len(cases)),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], cwd=path, capture_output=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
            frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(len(cases), 540*scale, 960*scale, 3)
            for frame, (pos, distance) in zip(frames, cases):
                with self.subTest(scale=scale, position=pos, distance=distance):
                    def bounds(channel):
                        mask = (frame[:, :, channel] >= 128) & (frame[:, :, 1-channel] < 64)
                        yy, xx = np.where(mask)
                        self.assertGreater(len(xx), 0)
                        return Box(xx.min()/scale, (xx.max()+1)/scale, yy.min()/scale, (yy.max()+1)/scale)
                    base, gloss = bounds(1), bounds(0)
                    col, row = (pos-1) % 3, (pos-1) // 3
                    dx = (base.x0-gloss.x1, gloss.cx-base.cx, gloss.x0-base.x1)[col]
                    dy = (gloss.y0-base.y1, (gloss.y0+gloss.y1-base.y0-base.y1)/2, base.y0-gloss.y1)[row]
                    # At 1x libass 0.17.4 hinting and the 50% ink threshold differ
                    # from 4x probes by up to 2 px; 2x verifies the actual placement.
                    tolerance = 2.25 if scale == 1 and col == 2 else 1.25
                    self.assertAlmostEqual(dx, 0 if col == 1 else 5+distance, delta=tolerance)
                    self.assertAlmostEqual(dy, 0 if row == 1 else 5+distance, delta=1.25)

    def test_furigana_only_excluded_above_and_data_preserved(self):
        for pos in range(1, 10):
            doc, tpl = fixture(f"国{{@注:国=Country;{pos}}}")
            ids = visible_map(doc.members["m"])[1]
            doc.annotations["a"] = Annotation("a", "ruby", "m", anchor_of(ids), "国", "くに", status="confirmed")
            before = copy.deepcopy(doc.annotations)
            ass, rep = self.export(doc, tpl)
            self.assertEqual(rep.ruby_lines, 0 if pos >= 7 else 1)
            self.assertEqual(doc.annotations, before)
            self.assertEqual(len([e for e in ass.events if e.style.endswith(".Gloss")]), 1)

    def test_warnings_stripping_check_report_and_cache(self):
        doc, tpl = fixture("国{@注:国=X;0}{@private:data}", lane="translation")
        cache = ExportCache()
        ass, rep = self.export(doc, tpl, cache=cache)
        expected = "旁注写法有误：{@注:国=X;0}（位置要写 1–9）"
        self.assertEqual(rep.warnings, [expected])
        self.assertEqual(ass.events[0].text, "国")
        check = validate(doc, {})
        self.assertTrue(check.ok)
        self.assertEqual(issue_lines(check, doc, {}), [expected])
        self.assertEqual(self.export(doc, tpl, cache=cache)[1].laid_out, 0)
        doc.members["m"].runs[-1].text = "国{@注:国=X}"
        self.assertEqual(self.export(doc, tpl, cache=cache)[1].laid_out, 1)
        self.assertEqual(len(self.export(doc, tpl, cache=cache)[0].events), 2)

    def test_display_conversion_scope_for_label_and_maps_target(self):
        from subkit.core.display import DisplayRules
        doc, tpl = fixture("学校{@注:学校=学术};下一行", lane="translation")
        ass, rep = self.export(doc, tpl, opts=ExportOptions(display=DisplayRules(zh_traditional_enabled=True)))
        self.assertIn("學校", ass.events[0].text)
        self.assertTrue(ass.events[1].text.endswith("學術"))
        ass, _ = self.export(doc, tpl, opts=ExportOptions(display=DisplayRules(
            zh_traditional_enabled=True, zh_scope={"gloss": False})))
        self.assertIn("學校", ass.events[0].text)
        self.assertTrue(ass.events[1].text.endswith("学术"))

    def test_gloss_unselectable_and_cached_preview_release_equal(self):
        doc, tpl = fixture()
        cache = ExportCache()
        ass, rep = self.export(doc, tpl, cache=cache)
        same, _ = self.export(doc, tpl, cache=cache)
        self.assertEqual(ass.serialize(), same.serialize())
        index = LayoutIndex(ass, rep.owners)
        self.addCleanup(index.close)
        self.assertEqual(index.active(1500), [0])
        self.assertEqual(index.events_of_group("g"), [0])
        preview, _ = self.export(doc, tpl, opts=ExportOptions(mode="preview"))
        self.assertEqual([e.text for e in ass.events], [e.text for e in preview.events])

    def test_style_reset_scaling_margins_and_implicit_rotation_origin(self):
        doc, tpl = fixture(r"A{\rOther\fscx120\fscy130}国{@注:国=Country}")
        tpl.styles.append(Style.new("Other", Fontname="Arial", Fontsize=80, Angle=12, Outline=6))
        doc.members["m"].margins = [123, 80, 45]
        ass, _ = self.export(doc, tpl)
        gloss = ass.events[1]
        state, _ = state_at(gloss.text, 0, ("fs", "frz", "fscx", "fscy"))
        self.assertEqual([state[n].num() for n in ("fs", "frz", "fscx", "fscy")], [40, 12, 120, 130])
        self.assertEqual(line_tag(gloss.text, "org").float_args(), [123, 45])
        self.assertEqual(ass.style("Default.Gloss").fontsize, 30)

    def test_whole_multiline_gloss_tag_run_and_merged_display(self):
        from subkit.core.display import DisplayRules
        doc, tpl = fixture("AB\\NCD")
        mark = "{@注:=Whole}"
        doc.members["m"].runs.append(Run(mark, kind="tag", ins="mark", b=len(mark)))
        ass, _ = self.export(doc, tpl, opts=ExportOptions(display=DisplayRules(merge_lines=(("original", "dialogue"),))))
        self.assertEqual(ass.events[0].text, "AB  CD")
        self.assertTrue(ass.events[1].text.endswith("Whole"))

    def test_static_song_gloss_not_in_fx_glyphs(self):
        from tests.test_karaoke_fx import fixture as fx_fixture
        doc, tpl = fx_fixture("pop")
        m = doc.members["m_fx"]
        mark = "{@注:H=Static}"
        m.runs.append(Run(mark, kind="tag", ins="gloss", b=len(mark)))
        ass, rep = self.export(doc, tpl)
        glosses = [e for e, owner in zip(ass.events, rep.owners) if owner["kind"] == "gloss"]
        self.assertEqual(len(glosses), 1)
        self.assertEqual((glosses[0].start, glosses[0].end), (100, 300))
        self.assertTrue(glosses[0].text.endswith("Static"))
        self.assertTrue(all("{@" not in e.text and "Static" not in e.text for e in ass.events if e not in glosses))


if __name__ == "__main__":
    unittest.main()
