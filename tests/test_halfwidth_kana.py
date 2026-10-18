"""Display-only kana widening: source clusters, karaoke, ruby, profiles and real rendering."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

import numpy as np
from subkit.ass.document import AssDocument, Style
from subkit.ass.state import state_at
from subkit.ass.text import VisibleText, tokenize
from subkit.core import karaoke as kara
from subkit.core.display import DisplayRules, PunctuationRules, punctuation_layout, punctuation_text
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.punctuation_profiles import ProfileError, PunctuationProfileStore, create_profile, matches, validate_profile
from subkit.core.readings import anchor_of, visible_map
from subkit.karaoke_fx import get_effect
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.model.serialize import document_to_json
from subkit.proc import run
from subkit.styles_default import default_styles_ass


def fixture(text="ｶﾞｰ国", *, song=False):
    m = Member("m_kana", "g_kana", "ja", "original", "song" if song else "dialogue", 1000, 3000,
               style="JP", runs=[Run(t.raw, "tag" if t.kind == "tags" else "text", ins="i_kana",
                                    a=t.start, b=t.end) for t in tokenize(text)])
    doc = Document("kana", groups=[Group(m.group, "song" if song else "dialogue", [m.id])], members={m.id: m})
    template = AssDocument.parse(default_styles_ass())
    template.style("JP").set("Fontname", "MS Gothic")
    template.style("JP").set("Fontsize", 72)
    return doc, template


def display(lane="original", **options):
    return DisplayRules.from_settings({"display": {"punctuation": {lane: {"halfwidth_kana": True, **options}}}})


class KanaUnit(unittest.TestCase):
    def show(self, text, **options):
        return punctuation_text(text, PunctuationRules(halfwidth_kana=True, **options))[0]

    def test_plain_voiced_small_kana_and_long_vowels(self):
        for raw, expected in (("ｶﾞﾊﾟｳﾞ", "ガパヴ"), ("ﾃｨｰｼｬﾂ", "ティーシャツ"),
                              ("海底帝国ｻﾞﾌﾞｰﾝ", "海底帝国ザブーン"), ("超神ﾈｲｶﾞｰ", "超神ネイガー"),
                              ("ｦﾞﾜﾞ", "ヺヷ"), ("カﾞかﾞ", "ガが"), ("ﾞﾟﾝﾞ", "゛゜ン゛")):
            with self.subTest(raw=raw):
                self.assertEqual(self.show(raw), expected)
                self.assertEqual(self.show(expected), expected)
        self.assertEqual(self.show("ｰ", dash="half"), "ー")
        self.assertEqual(self.show("ﾈｲｶﾞｰ", dash="half"), "ネイガー")

    def test_non_targets_raw_drawings_and_breaks(self):
        untouched = "ＡＢＣ１２３ ①Ⅳ㎏　｡｢｣､･ - ~ ! カタカナ ひらがな"
        self.assertEqual(self.show(untouched), untouched)
        raw = r"ｶﾞ{@Raw=ｶﾞｰ}ﾊﾟ{\p1}m 0 0 l 10 10{\p0}ｳﾞ\Nｶ\Nﾞ"
        self.assertEqual(self.show(raw), r"ガ{@Raw=ｶﾞｰ}パ{\p1}m 0 0 l 10 10{\p0}ヴ\Nカ\N゛")
        for text in (raw, "ｻﾞﾌﾞｰﾝ"):
            self.assertEqual(punctuation_text(text, PunctuationRules())[0], text)

    def test_source_start_and_end_maps_cover_both_halves(self):
        text, starts, ends, categories, hidden = punctuation_layout("ｶﾞﾊﾟｰ国", PunctuationRules(halfwidth_kana=True))
        self.assertEqual(text, "ガパー国")
        self.assertEqual(starts, [0, 0, 1, 1, 2, 3, 4])
        self.assertEqual(ends, [0, 1, 1, 2, 2, 3, 4])
        self.assertEqual(categories["long_vowel"], {2})
        self.assertEqual(hidden, 0)
        self.assertTrue(all(starts[i] < ends[i+1] for i in range(6)))

    def test_interior_override_keeps_base_style_and_order(self):
        raw = r"{\fnMS Gothic\1c&H0000FF&}ｶ{\fnArial\rOther}ﾞ次"
        shown = self.show(raw)
        self.assertEqual(shown, r"{\fnMS Gothic\1c&H0000FF&}ガ{\fnArial\rOther}次")
        self.assertEqual([t.raw for t in tokenize(raw) if t.kind == "tags"],
                         [t.raw for t in tokenize(shown) if t.kind == "tags"])
        self.assertEqual(state_at(shown, 0, ["fn"])[0]["fn"].arg, "MS Gothic")
        self.assertEqual(self.show(r"ｶ{\k20}ﾞ{\b1}"), r"ガ{\k20}{\b1}")

    def test_defaults_settings_scope_hash_and_profile_validation(self):
        self.assertFalse(PunctuationRules.from_settings({}).halfwidth_kana)
        self.assertFalse(PunctuationRules().active)
        rule = PunctuationRules(halfwidth_kana=True, scope_chars=("dialogue", "song", "other"))
        self.assertTrue(rule.active)
        self.assertEqual(PunctuationRules.from_settings(rule.to_settings()), rule)
        self.assertNotEqual(ExportOptions().key(), ExportOptions(display=display()).key())
        profile = create_profile("假名", PunctuationRules(), rule)
        self.assertTrue(profile["original"]["halfwidth_kana"])
        old = copy.deepcopy(profile)
        old["original"].pop("halfwidth_kana")
        self.assertFalse(validate_profile(old)["original"]["halfwidth_kana"])
        profile["original"]["halfwidth_kana"] = "yes"
        with self.assertRaises(ProfileError):
            validate_profile(profile)


class KanaExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="subkit-kana-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def build(self, doc, tpl, rules=None, **kw):
        return build_ass(doc, {}, tpl, self.path, ExportOptions(display=rules or DisplayRules()),
                         measure_cache=self.path/"measure.sqlite3", **kw)

    def test_roles_lanes_off_bytes_and_document_unchanged(self):
        doc, tpl = fixture("ﾈｲｶﾞｰｰｰ国")
        saved = document_to_json(doc)
        for lane in ("original", "translation"):
            m = doc.members["m_kana"]
            m.lane = lane
            for kind, role in (("dialogue", "dialogue"), ("song", "song"), ("other", "sign")):
                m.role = role
                for included in (False, True):
                    ass, _ = self.build(doc, tpl, display(lane, scope_chars=[kind] if included else []))
                    self.assertEqual(VisibleText(ass.events[0].text).text, "ネイガーーー国" if included else "ﾈｲｶﾞｰｰｰ国")
        m.lane, m.role = "original", "dialogue"
        plain, _ = self.build(doc, tpl)
        off, _ = self.build(doc, tpl, display(halfwidth_kana=False))
        self.assertEqual(plain.serialize(), off.serialize())
        self.assertEqual(document_to_json(doc), saved)

    def test_cross_runs_ruby_gloss_long_vowel_fonts_and_primary_identity(self):
        doc, tpl = fixture("ｶﾞｰ国{@注:国=くに}")
        m = doc.members["m_kana"]
        m.runs = [Run("ｶ", src="original", e=0, a=0, b=1), Run("ﾞｰ国", ins="typed", b=3),
                  Run("{@注:国=くに}", ins="gloss", b=11)]
        ids = visible_map(m)[1]
        # Use a ruby on the preceding word instead of competing with this gloss.
        doc.annotations["a_kana"] = Annotation("a_kana", "ruby", m.id, anchor_of(ids[:2]), "ｶﾞ", "が",
                                               status="confirmed", parts=[[0, 2, "が"]])
        m.format = [{"anchor": anchor_of(ids[:1]), "props": {"c1": "&H0000FF&"}},
                    {"anchor": anchor_of(ids[1:2]), "props": {"c1": "&HFF0000&"}}]
        saved = copy.deepcopy(doc)
        ass, report = self.build(doc, tpl, display(fonts={"long_vowel": "Arial"}))
        base = next((e, o) for e, o in zip(ass.events, report.owners) if o["kind"] == "base")
        self.assertEqual(VisibleText(base[0].text).text, "ガー国")
        self.assertEqual(base[1]["ids"][0], ids[0])
        self.assertEqual(base[1]["ids"][2], ids[3])
        self.assertEqual(state_at(base[0].text, 0, ["1c"])[0]["1c"].arg, "&H0000FF&")
        self.assertEqual(state_at(base[0].text, 1, ["fn"])[0]["fn"].arg, "Arial")
        self.assertEqual(report.ruby_lines, 1)
        self.assertTrue(any(o["kind"] == "gloss" for o in report.owners))
        self.assertEqual(doc, saved)

    def test_inline_karaoke_combines_display_cluster_without_changing_beats(self):
        doc, tpl = fixture("ｶﾞｰ国", song=True)
        m = doc.members["m_kana"]
        beats = kara.derive(doc, m, "mora")
        for i, b in enumerate(beats):
            b.t0, b.t1, b.by = 1000+i*200, 1200+i*200, "manual"
        doc.karaoke["k"] = kara.beats_to_track(m, beats, unit="mora", effect={"preset": "kf"})
        original = copy.deepcopy(doc)
        ass, _ = self.build(doc, tpl, display(scope_chars=["song"]))
        runs = kara.parse_k_runs(ass.events[0].text)
        self.assertEqual("".join(s for s, d in runs), "ガー国")
        onset, times = 1000, {}
        for s, duration in runs:
            if s:
                times[s] = onset
            onset += duration*10
        self.assertEqual(times["ガ"], 1000)
        self.assertEqual(times["国"], next(b.t0 for b in beats if "国" in m.text[b.a:b.b]))
        self.assertEqual(onset, max(b.t1 for b in beats))
        self.assertEqual(doc, original)

    def test_advanced_effect_receives_both_original_beats_and_anchors(self):
        doc, tpl = fixture("ｶﾞ国", song=True)
        m = doc.members["m_kana"]
        beats = kara.derive(doc, m, "mora")
        for i, b in enumerate(beats):
            b.t0, b.t1, b.by = 1000+i*200, 1200+i*200, "manual"
        doc.karaoke["k"] = kara.beats_to_track(m, beats, unit="mora", effect={"preset": "tremble"})
        ids = visible_map(m)[1]
        contexts = []
        fx = get_effect("tremble")
        generate = fx.generate
        with patch.object(fx, "generate", side_effect=lambda ctx: (contexts.append(ctx), generate(ctx))[1]):
            self.build(doc, tpl, display(scope_chars=["song"]))
        glyph = next(g for g in contexts[0].glyphs if g.text == "ガ")
        self.assertEqual(set(glyph.anchor_ids), set(ids[:2]))
        self.assertEqual([(a, z) for i, a, z in glyph.beats], [(b.t0, b.t1) for b in beats[:2]])

    def test_preview_cache_and_profile_roundtrip(self):
        doc, tpl = fixture()
        cache = ExportCache()
        original, _ = self.build(doc, tpl, cache=cache)
        changed, rep = self.build(doc, tpl, display(), cache=cache)
        self.assertEqual(rep.laid_out, 1)
        self.assertNotEqual(original.serialize(), changed.serialize())
        again, rep = self.build(doc, tpl, display(), cache=cache)
        self.assertEqual(rep.laid_out, 0)
        self.assertEqual(changed.serialize(), again.serialize())
        store = PunctuationProfileStore(self.path/"profiles")
        profile = create_profile("宽假名", PunctuationRules(), PunctuationRules(halfwidth_kana=True))
        store.save(profile)
        store.export_to("宽假名", self.path/"width.json")
        other = PunctuationProfileStore(self.path/"other")
        other.import_from(self.path/"width.json")
        self.assertEqual(other.load("宽假名"), profile)
        self.assertTrue(matches({"display": {"punctuation": {"original": {"halfwidth_kana": True}}}}, profile))

    def test_libass_ruby_position_matches_fullwidth_reference(self):
        doc, tpl = fixture("ｻﾞﾌﾞｰﾝ国")
        m = doc.members["m_kana"]
        ids = visible_map(m)[1]
        doc.annotations["a_country"] = Annotation("a_country", "ruby", m.id, anchor_of(ids[-1:]), "国", "くに",
                                                  status="confirmed", parts=[[0, 1, "くに"]])
        actual, _ = self.build(doc, tpl, display())
        reference, _ = fixture("ザブーン国")
        rm = reference.members["m_kana"]
        rids = visible_map(rm)[1]
        reference.annotations["a_country"] = Annotation("a_country", "ruby", rm.id, anchor_of(rids[-1:]), "国", "くに",
                                                        status="confirmed", parts=[[0, 1, "くに"]])
        expected, _ = self.build(reference, tpl)
        frames = []
        for name, ass in (("actual", actual), ("expected", expected)):
            (self.path/f"{name}.ass").write_text(ass.serialize(), encoding="utf-8-sig")
            proc = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "color=c=black:s=1920x1080:r=25:d=0.04", "-vf",
                        f"setpts=PTS+1.5/TB,ass={name}.ass,format=rgb24", "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], cwd=self.path, capture_output=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
            frames.append(proc.stdout)
        self.assertEqual(frames[0], frames[1])
        self.assertGreater(np.frombuffer(frames[0], np.uint8).max(), 0)
