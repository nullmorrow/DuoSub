"""Punctuation display rules on Windows host libass and offscreen Qt."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import unittest
from itertools import combinations
from unittest.mock import patch

from tests.test_pinyin_export import document
from subkit.ass.document import AssDocument, Style
from subkit.ass.state import state_at
from subkit.ass.text import VisibleText, line_tag, split_karaoke_overrides, tokenize
from subkit.core import karaoke as kara
from subkit.core.display import (DisplayRules, PunctuationRules, PUNCTUATION_CATEGORIES,
                                 punctuation_text, punctuation_text_categories)
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.layout_index import LayoutIndex
from subkit.core.readings import anchor_of, visible_map
from subkit.core.workspace import Project
from subkit.model.document import Annotation, Run
from subkit.ruby.measure import Measurer
from subkit.styles_default import default_styles_ass


def rules(lane="original", **kw):
    return DisplayRules.from_settings({"display": {"punctuation": {lane: kw}}})


class PunctuationUnit(unittest.TestCase):
    def show(self, raw, **kw):
        return punctuation_text(raw, PunctuationRules.from_settings(kw))[0]

    def test_defaults_lanes_and_cache_key(self):
        self.assertEqual(DisplayRules(), DisplayRules.from_settings({}))
        self.assertFalse(DisplayRules.from_settings({}).active)
        self.assertFalse(rules(ellipsis="bad").active)
        self.assertTrue(rules(font="Arial").active)
        m = document("字", lang="ja", lane="original").members["m_1"]
        self.assertTrue(rules(comma=True).punctuation_layers_for(m)[0].comma)
        self.assertFalse(rules("translation", comma=True).punctuation_layers_for(m)[0].active)
        hash(ExportOptions(display=rules(comma=True)).key())

    def test_ellipsis_modes_before_period(self):
        for raw in ("甲…乙", "甲……乙", "甲...乙", "甲。。。乙", "甲・・・乙", "甲......乙"):
            self.assertEqual(self.show(raw, ellipsis="single", period=True), "甲…乙")
            self.assertEqual(self.show(raw, ellipsis="double", period=True), "甲……乙")
            self.assertEqual(self.show(raw), raw)
        self.assertEqual(self.show("甲......乙……丙・・・丁", ellipsis="keep_count"), "甲……乙……丙…丁")

    def test_period_comma_and_numeric_latin_exceptions(self):
        for text in ("中文。次句．末句.", "日本。次の文．終わり."):
            self.assertEqual(self.show(text, period=True), text.replace("。", "  ").replace("．", "  "))
        self.assertEqual(self.show("3.5 ３．５ ａ．ｍ． a.m. U.S.A. x.y z.", period=True),
                         "3.5 ３．５ ａ．ｍ． a.m. U.S.A. x.y z.")
        # A half-width "." is English and never a sentence period; "..." is still an ellipsis.
        self.assertEqual(self.show("能取走Lv.4冒險者的性命。No.1、Ｌｖ．５ ver2.0 文.", period=True),
                         "能取走Lv.4冒險者的性命  No.1、Ｌｖ．５ ver2.0 文.")
        self.assertEqual(self.show("等級4．文...", period=True, ellipsis="single"), "等級4  文…")
        for text in ("甲，乙,丙、丁", "私，君,彼、彼女"):
            self.assertEqual(self.show(text, comma=True), text.replace("，", "  ").replace(",", "  ").replace("、", "  "))
        self.assertEqual(self.show("1,000，2，000、終", comma=True), "1,000  2，000  終")

    def test_comma_dunhao_combinations_and_legacy(self):
        raw = "甲，乙,丙、丁 1,000 ２，０００"
        for comma in ("off", "spaces", "ideographic"):
            for dunhao in (False, True):
                with self.subTest(comma=comma, dunhao=dunhao):
                    expected = "甲，乙,丙、丁"
                    if comma != "off":
                        expected = expected.replace("，", "、" if comma == "ideographic" else "  ").replace(",", "、" if comma == "ideographic" else "  ")
                    if dunhao:
                        expected = expected.replace("、", "  ")
                    self.assertEqual(self.show(raw, comma=comma, dunhao=dunhao), expected + " 1,000 ２，０００")
                    for lane in ("original", "translation"):
                        m = document(raw, lane=lane).members["m_1"]
                        rule = rules(lane, comma=comma, dunhao=dunhao).punctuation_layers_for(m)[0]
                        self.assertEqual(punctuation_text(raw, rule)[0], expected + " 1,000 ２，０００")
        # Settings saved before scopes existed keep display fonts on dialogue only.
        legacy = dict(scope_fonts=("dialogue",))
        self.assertEqual(PunctuationRules.from_settings({"comma": True}), PunctuationRules(comma="spaces", dunhao=True, **legacy))
        self.assertEqual(PunctuationRules.from_settings({"comma": False}), PunctuationRules(**legacy))
        self.assertEqual(PunctuationRules.from_settings({"comma": "bad"}), PunctuationRules(**legacy))
        self.assertEqual(self.show(r"，甲、\N、乙，", comma="ideographic", dunhao=True), r"甲\N乙")
        self.assertEqual(self.show("甲{@Raw=，,、}，乙", comma="ideographic", dunhao=True), "甲{@Raw=，,、}  乙")

    def test_spaces_edges_fullwidth_and_latin_words(self):
        self.assertEqual(self.show(r" 甲　乙   丙\N  丁　 \N　戊 ", spaces=True), r"甲  乙  丙\N丁\N戊")
        self.assertEqual(self.show("Non Player Character 1 2 A　B A   B", spaces=True), "Non Player Character 1 2 A  B A  B")
        self.assertEqual(self.show(r"甲， 、 。乙。\N，丙。", comma=True, period=True, spaces=True), r"甲  乙\N丙")
        self.assertEqual(self.show(r"甲\h\h乙\n丙", spaces=True), r"甲  乙\n丙")

    def test_exclaim_question_zh_ja(self):
        for text in ("何!?！！？？", "什么!?！！？？"):
            self.assertEqual(self.show(text, exclaim="full", question="full"), text.replace("!", "！").replace("?", "？"))
            self.assertEqual(self.show(text, exclaim="half", question="half"), text.replace("！", "!").replace("？", "?"))

    def test_nested_unmatched_quotes_and_apostrophes(self):
        for raw in ('“甲‘乙’丙”', '「甲『乙』丙」', '"甲‘乙’丙"'):
            self.assertEqual(self.show(raw, quotes="nested"), "「甲『乙』丙」")
        self.assertEqual(self.show("“甲‘乙“丙”丁’戊”", quotes="nested"), "「甲『乙「丙」丁』戊」")
        for text in ("“孤立", "孤立”", "don't"):
            self.assertEqual(self.show(text, quotes="nested"), text)
        self.assertEqual(self.show('“甲\'乙\'丙”', quotes="nested"), "「甲『乙』丙」")

    def test_quote_modes_and_old_settings(self):
        raw = '“甲‘乙’丙”'
        self.assertEqual(self.show(raw, quotes="corner"), "「甲「乙」丙」")
        self.assertEqual(self.show(raw, quotes="white_corner"), "『甲『乙』丙』")
        self.assertEqual(self.show(raw, quotes="off"), raw)
        for value, expected in ((True, "nested"), (False, "off"), ("bad", "off")):
            rule = PunctuationRules.from_settings({"quotes": value})
            self.assertEqual(rule.quotes, expected)
            self.assertEqual(punctuation_text(raw, rule)[0], "「甲『乙』丙」" if value is True else raw)

    def test_separator_before_period_and_cjk_context(self):
        for sep in "·‧•･⋅∙・．.":
            self.assertEqual(self.show("勞爾" + sep + "諾德", separator=True), "勞爾・諾德")
            self.assertEqual(self.show("勞爾" + sep + "諾德", separator=True, period=True), "勞爾・諾德")
        for raw in ("a.m.", "3.5", "文。", "A.文", "文.A", ".文", "文."):
            self.assertEqual(self.show(raw, separator=True), raw)
        for raw in ("あ.カ", "𠀀．字", "ｶ.ﾅ"):
            self.assertEqual(self.show(raw, separator=True), raw.replace(".", "・").replace("．", "・"))
        self.assertEqual(self.show("a.m. 3.5 文。次", separator=True, period=True), "a.m. 3.5 文  次")
        self.assertTrue(rules(separator=True).punctuation_layers_for(document("字", lang="ja", lane="original").members["m_1"])[0].separator)
        self.assertFalse(rules("translation", separator=True).punctuation_layers_for(document("字", lang="ja", lane="original").members["m_1"])[0].separator)

    def test_independent_scope_matrix_matches_merge_kinds_and_lanes(self):
        kinds = ("dialogue", "song", "other")
        scopes = [list(c) for n in range(4) for c in combinations(kinds, n)]
        roles = {"dialogue": ("dialogue", "top", "alt"), "song": ("song", "song_jp", "song_cn"),
                 "other": ("sign", "info", "comment", "unknown")}
        for lane in ("translation", "original"):
            m = document("!", lane=lane).members["m_1"]
            for chars_scope in scopes:
                for fonts_scope in scopes:
                    display = rules(lane, exclaim="full", fonts={"exclaim": "Punct"},
                                    scope_chars=chars_scope, scope_fonts=fonts_scope)
                    display.merge_lines = tuple((lane, kind) for kind in chars_scope)
                    for kind, group_roles in roles.items():
                        for role in group_roles:
                            with self.subTest(lane=lane, chars=chars_scope, fonts=fonts_scope, role=role):
                                m.role = role
                                chars, fonts = display.punctuation_layers_for(m)
                                self.assertEqual(chars.exclaim, "full" if kind in chars_scope else "off")
                                self.assertFalse(chars.fonts)
                                self.assertEqual(fonts, (("exclaim", "Punct"),) if kind in fonts_scope else ())
                                self.assertEqual(display.merge_for(m), kind in chars_scope)
                                m.lane = "original" if lane == "translation" else "translation"
                                inactive, fonts = display.punctuation_layers_for(m)
                                self.assertFalse(inactive.active)
                                self.assertFalse(fonts)
                                m.lane = lane

    def test_legacy_scope_defaults_and_scope_settings_round_trip(self):
        legacy = PunctuationRules.from_settings({"exclaim": "full", "font": "Legacy"})
        self.assertEqual(legacy.scope_chars, ("dialogue",))
        self.assertEqual(legacy.scope_fonts, ("dialogue",))      # old export unchanged
        self.assertEqual(legacy.to_settings()["scope_chars"], ["dialogue"])
        self.assertEqual(legacy.to_settings()["scope_fonts"], ["dialogue"])
        self.assertEqual(PunctuationRules.from_settings({"exclaim": "full", "tilde": "off"}).scope_fonts,
                         ("dialogue", "song", "other"))
        for chars in ([], ["song"], ["other", "dialogue", "song", "song", "unknown"]):
            for fonts in ([], ["other"], ["song", "dialogue"]):
                rule = PunctuationRules.from_settings({"scope_chars": chars, "scope_fonts": fonts,
                                                      "fonts": {"long_vowel": "Long"}})
                self.assertEqual(PunctuationRules.from_settings(rule.to_settings()), rule)
                hash(ExportOptions(display=rules(**rule.to_settings())).key())
        self.assertEqual(PunctuationRules.from_settings({"scope_chars": None, "scope_fonts": "song"}),
                         PunctuationRules())

    def test_nested_inverse_depth_and_settings(self):
        self.assertEqual(self.show("“甲‘乙“丙”丁’戊”", quotes="nested_inverse"), "『甲「乙『丙』丁」戊』")
        self.assertEqual(PunctuationRules.from_settings({"quotes": "nested_inverse"}).quotes, "nested_inverse")
        self.assertEqual(self.show("“孤立", quotes="nested_inverse"), "“孤立")

    def test_tags_drawings_and_split_tokens(self):
        raw = r"{\fnA}甲.{\b1}..乙{\p1}m 0 0 l 10 10{\p0}丙，丁"
        text, bounds, fonts, hidden = punctuation_text(raw, PunctuationRules(ellipsis="single", comma="spaces", dunhao=True))
        self.assertEqual(text, r"{\fnA}甲{\b1}…乙{\p1}m 0 0 l 10 10{\p0}丙  丁")
        self.assertEqual(bounds[-1], len(VisibleText(text)))
        self.assertEqual((fonts, hidden), ({1}, 0))


class PunctuationCategoriesUnit(unittest.TestCase):
    show = PunctuationUnit.show

    def categories(self, text, **kw):
        return punctuation_text_categories(text, PunctuationRules.from_settings(kw))[2]

    def test_every_category_and_character_set(self):
        samples = {
            "comma": "，,", "dunhao": "、", "period": "。 ．","colon": "：:；;",
            "ellipsis": "… ... ．．． 。。。 ・・・ ･･･", "exclaim": "！!", "question": "？?",
            "quotes": "「」『』“”‘’\"'", "brackets": "（）()【】《》〈〉［］[]",
            "separator": "·‧•･⋅∙・", "tilde": "~～〜∼", "dash": "-–—―─━‐‒－ー"}
        for category, text in samples.items():
            with self.subTest(category=category):
                positions = self.categories(text)
                self.assertEqual(positions[category], {i for i, ch in enumerate(text) if ch != " "})
                self.assertFalse(any(positions[c] for c in PUNCTUATION_CATEGORIES if c != category))
        self.assertEqual(self.categories("漢.字 あ．カ ｶ.ﾅ")["separator"], {1, 5, 9})

    def test_category_context_exclusions(self):
        text = "ハロー すごーい ハーー ぁー ㇰー ｶｰー T-shirt 2-3 Re- 1,000 ２，０００ 3.14 ３．１４ U.S.A. a.m. x.y Lv.4 Ｌｖ．５ don't"
        categories = self.categories(text)
        self.assertTrue(categories["long_vowel"])
        self.assertFalse(any(positions for category, positions in categories.items() if category != "long_vowel"))
        positions = self.categories("漢ーー。ー ー\\Nー")
        self.assertEqual(positions["dash"], {1, 2, 4, 6, 7})
        self.assertEqual(self.categories("漢字...")["ellipsis"], {2, 3, 4})
        self.assertFalse(self.categories("漢字...")["period"])

    def test_long_vowels_are_display_only_and_exclusive_with_dashes(self):
        for text in ("すごーい", "ハロー", "やったかー", "ぁー", "ㇰー", "ｶｰーｰ", "ハーーｰ",
                     r"カ{\k10}ー{\b1}ｰ"):
            with self.subTest(text=text):
                shown, _bounds, categories, _hidden = punctuation_text_categories(text, PunctuationRules(dash="half"))
                self.assertEqual(shown, text)
                self.assertEqual(categories["long_vowel"],
                                 {i for i, ch in enumerate(VisibleText(text).text) if ch in "ーｰ"})
                self.assertFalse(categories["dash"])
                self.assertEqual(sum(map(len, categories.values())), len(categories["long_vowel"]))
        for text in ("ー", "漢ー", "漢ーー", "。ー", " ー", r"カ\Nー"):
            self.assertFalse(self.categories(text)["long_vowel"])
            self.assertEqual(len(self.categories(text)["dash"]), text.count("ー"))
        categories = self.categories("カーー漢ー{@Raw=カーーｰ}")
        self.assertEqual(categories["long_vowel"], {1, 2})
        self.assertEqual(categories["dash"], {4})

    def test_tilde_every_option_count_and_elongation(self):
        text = "危な〜い~～∼〜〜"
        self.assertEqual(self.show(text, tilde="off"), text)
        for option, glyph in (("full", "～"), ("half", "~"), ("wave", "〜")):
            with self.subTest(option=option):
                shown, bounds, positions, hidden = punctuation_text_categories(text, PunctuationRules(tilde=option))
                self.assertEqual(shown, "危な" + glyph + "い" + glyph * 5)
                self.assertEqual(bounds, list(range(len(text) + 1)))
                self.assertEqual(positions["tilde"], {2, 4, 5, 6, 7, 8})
                self.assertEqual(hidden, 0)

    def test_dash_every_option_and_run_counts(self):
        text = "漢-–—―─━‐‒－ーー字"
        self.assertEqual(self.show(text, dash="off"), text)
        for option, glyph in (("half", "-"), ("em", "—"), ("choon", "ー"), ("bar", "―"),
                              ("fullwidth_hyphen", "－")):
            with self.subTest(option=option):
                shown, bounds, positions, hidden = punctuation_text_categories(text, PunctuationRules(dash=option))
                self.assertEqual(shown, "漢" + glyph * 11 + "字")
                self.assertEqual(bounds, list(range(len(text) + 1)))
                self.assertEqual(positions["dash"], set(range(1, 12)))
                self.assertEqual(hidden, 0)
        self.assertEqual(self.show("——", dash="half"), "--")
        self.assertEqual(self.show("A--B", dash="em"), "A——B")

    def test_dash_exclusions_and_transformed_choon_fonts(self):
        text = "ハロー すごーい ハーー ぁー ㇰー ｶｰー T-shirt 2-3 Re-"
        for option in ("half", "em", "choon", "bar", "fullwidth_hyphen"):
            self.assertEqual(self.show(text, dash=option), text)
        for source, expected in (("漢ーー", "漢--"), ("。ー", "。-"), (" ー", " -"),
                                 (r"\Nー", r"\N-"), ("ーー", "--")):
            self.assertEqual(self.show(source, dash="half"), expected)
        shown, _bounds, positions, _hidden = punctuation_text_categories("ハ—ー", PunctuationRules(dash="choon"))
        self.assertEqual(shown, "ハーー")
        self.assertEqual(positions["dash"], {1, 2})

    def test_raw_tags_drawings_breaks_and_positions(self):
        text = r"!{@Raw=!?~—...}？\N{\p1}—~{\p0}～{\b1}—"
        shown, bounds, positions, hidden = punctuation_text_categories(text,
            PunctuationRules(exclaim="full", question="half", tilde="half", dash="bar"))
        self.assertEqual(shown, r"！{@Raw=!?~—...}?\N{\p1}—~{\p0}~{\b1}―")
        self.assertEqual(positions["exclaim"], {0})
        self.assertEqual(positions["question"], {8})
        self.assertEqual(positions["tilde"], {9})
        self.assertEqual(positions["dash"], {10})
        self.assertEqual(bounds[-1], 11)
        self.assertEqual(hidden, 0)

    def test_categories_after_transform_and_final_period_hiding(self):
        rule = PunctuationRules(comma="ideographic", ellipsis="double", exclaim="full", quotes="corner")
        shown, bounds, positions, hidden = punctuation_text_categories('甲,乙...!"丙"。', rule, hide_final_period=True)
        self.assertEqual(shown, '甲、乙……！「丙」')
        self.assertEqual(positions["dunhao"], {1})
        self.assertFalse(positions["comma"])
        self.assertEqual(positions["ellipsis"], {3, 4})
        self.assertEqual(positions["exclaim"], {5})
        self.assertEqual(positions["quotes"], {6, 8})
        self.assertFalse(positions["period"])
        self.assertEqual((bounds[-1], hidden), (9, 1))

    def test_settings_legacy_migration_canonical_round_trip(self):
        legacy = PunctuationRules.from_settings({"font": "Punct"})
        self.assertEqual(dict(legacy.fonts), {c: "Punct" for c in ("ellipsis", "exclaim", "question", "quotes")})
        # Saved before scopes existed: the font kept its dialogue-only reach, so old exports do not change.
        self.assertEqual(legacy, PunctuationRules(font="Punct", scope_fonts=("dialogue",)))
        self.assertEqual(PunctuationRules.from_settings({}).scope_fonts, ("dialogue", "song", "other"))
        self.assertNotIn("font", legacy.to_settings())
        self.assertEqual(PunctuationRules.from_settings(legacy.to_settings()), legacy)
        settings = {"tilde": "wave", "dash": "choon", "font": "Ignored",
                    "fonts": {c: c + " Font" for c in PUNCTUATION_CATEGORIES}}
        rule = PunctuationRules.from_settings(settings)
        self.assertTrue(rule.active)
        self.assertEqual(PunctuationRules.from_settings(rule.to_settings()), rule)
        self.assertIsInstance(rule.fonts, tuple)
        hash(ExportOptions(display=rules(fonts=settings["fonts"])).key())
        for fonts in ({}, None, {"exclaim": None}, {"unknown": "A", "tilde": 2}):
            self.assertFalse(PunctuationRules.from_settings({"fonts": fonts, "font": "Ignored"}).active)
        self.assertEqual(PunctuationRules.from_settings({"tilde": "bad", "dash": "bad"}), PunctuationRules())
        self.assertEqual(PunctuationRules().to_settings()["fonts"], {c: None for c in PUNCTUATION_CATEGORIES})


class PunctuationExport(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp(prefix="punctuation-", dir=os.environ["TEMP"]))
        self.template = AssDocument.parse(default_styles_ass())
        self.template.styles.append(Style.new("Other", Fontname="Courier New"))

    def build(self, doc, display=None, cache=None, **kw):
        return build_ass(doc, {}, self.template, self.path, ExportOptions(display=display or DisplayRules(), **kw),
                         measure_cache=self.path / "render.sqlite3", cache=cache)

    def test_lanes_roles_and_off_bytes(self):
        for lang, lane, style in (("ja", "original", "JP"), ("zh-Hans", "translation", "CN")):
            doc = document("甲，乙。...!?“丙”", lang=lang, lane=lane, style=style)
            before = copy.deepcopy(doc)
            off, _ = self.build(doc)
            explicit, _ = self.build(doc, rules(lane, comma=False, period=False, font=None))
            self.assertEqual(off.serialize(), explicit.serialize())
            for role in ("dialogue", "top", "alt", "song", "song_jp", "song_cn", "sign", "info"):
                doc.members["m_1"].role = role
                on, _ = self.build(doc, rules(lane, comma=True))
                self.assertEqual(VisibleText(on.events[0].text).text,
                    "甲  乙。...!?“丙”" if role in ("dialogue", "top", "alt") else before.members["m_1"].text)
            doc.members["m_1"].role = "dialogue"
            self.assertEqual(doc, before)

    def test_independent_layers_export_scope_matrix_and_raw(self):
        from subkit.core.fontcheck import font_runs

        for lane, lang, style in (("original", "ja", "JP"), ("translation", "zh-Hans", "CN")):
            for kind, role in (("dialogue", "alt"), ("song", "song_jp"), ("other", "sign")):
                for chars in (False, True):
                    for fonts in (False, True):
                        with self.subTest(lane=lane, kind=kind, chars=chars, fonts=fonts):
                            doc = document("カｰ!~{@Raw=カｰ!~}", lane=lane, lang=lang, style=style)
                            doc.members["m_1"].role = role
                            saved = copy.deepcopy(doc)
                            display = rules(lane, exclaim="full", tilde="wave", dash="half",
                                scope_chars=[kind] if chars else [], scope_fonts=[kind] if fonts else [],
                                fonts={"exclaim": "Punct", "tilde": "Punct", "long_vowel": "Long"})
                            ass, _report = self.build(doc, display)
                            text = ass.events[0].text
                            self.assertEqual(VisibleText(text).text, ("カｰ！〜" if chars else "カｰ!~") + "カｰ!~")
                            runs = list(font_runs(ass.events[0], ass))
                            self.assertEqual(any(font == "Long" for font, _text in runs), fonts)
                            self.assertEqual(any(font == "Punct" for font, _text in runs), fonts)
                            if fonts:
                                self.assertIn(("Long", "ｰ"), runs)
                                self.assertIn(("Punct", "！〜" if chars else "!~"), runs)
                                self.assertEqual(state_at(text, 5, ["fn"])[0]["fn"].arg,
                                                 self.template.style(style).fontname)
                            self.assertEqual(doc, saved)

    def test_song_source_k_tags_long_vowel_font_and_measured_ruby(self):
        self._assert_song_font_layout(stored=False, vertical=False)

    def test_song_stored_k_tags_long_vowel_font_and_measured_ruby(self):
        self._assert_song_font_layout(stored=True, vertical=False)

    def test_vertical_song_k_tags_long_vowel_font_and_measured_ruby(self):
        self._assert_song_font_layout(stored=True, vertical=True)

    def test_vertical_song_leading_punctuation_font_does_not_restyle_ruby(self):
        for font in ("@MS PGothic", "MS PGothic"):
            with self.subTest(font=font):
                doc = document("!カー日", lang="ja", lane="original", style="JP")
                m = doc.members["m_1"]
                m.role = "song"
                doc.groups[0].kind = "song"
                ids = visible_map(m)[1]
                doc.annotations = {"a": Annotation("a", "ruby", m.id, anchor_of(ids[3:4]),
                                                   "日", "にーち", parts=[[0, 1, "にーち"]])}
                style = self.template.style("JP")
                style.set("Fontname", "@MS Gothic")
                style.set("Angle", 270)
                style.set("Alignment", 5)
                ass, report = self.build(doc, rules(fonts={"exclaim": font}))
                self.assertEqual(report.ruby_lines, 1)
                self.assertEqual(state_at(ass.events[0].text, 0, ["fn"])[0]["fn"].arg, font)
                self.assertEqual(line_tag(ass.events[1].text, "fn").arg, "@MS Gothic")
                self.assertEqual(line_tag(ass.events[1].text, "frz").num(), 270)

    def _assert_song_font_layout(self, *, stored, vertical):
        doc = document("カー日!" if stored else r"{\k20}カ{\k30}ー{\k40}日!",
                       lang="ja", lane="original", style="JP")
        m = doc.members["m_1"]
        m.role = "song_jp"
        doc.groups[0].kind = "song"
        if not stored:
            m.runs = [Run(t.raw, "tag" if t.kind == "tags" else "text", ins="i_1", a=t.start, b=t.end)
                      for t in tokenize(m.text)]
        ids = visible_map(m)[1]
        doc.annotations = {"a_day": Annotation("a_day", "ruby", m.id, anchor_of(ids[2:3]),
                                              "日", "にーち", status="confirmed", parts=[[0, 1, "にーち"]])}
        if stored:
            beats = kara.derive(doc, m)
            for i, beat in enumerate(beats):
                beat.t0, beat.t1, beat.by = i * 400, (i + 1) * 400, "manual"
            doc.karaoke["k_song"] = kara.beats_to_track(m, beats, effect={"preset": "k"})
        style = self.template.style("JP")
        style.set("Fontname", "@MS Gothic" if vertical else "MS Gothic")
        style.set("Angle", 270 if vertical else 0)
        style.set("Alignment", 5)
        font = "@MS PGothic" if vertical else "MS PGothic"
        saved = copy.deepcopy(doc)
        captured = []
        original_measure = Measurer.line_char_boxes

        def measured(measurer, requests):
            captured.extend(copy.deepcopy(requests))
            return original_measure(measurer, requests)

        display = rules(scope_chars=["song"], dash="half", exclaim="full", fonts={"long_vowel": font})
        with patch.object(Measurer, "line_char_boxes", measured):
            ass, report = self.build(doc, display)
        base = ass.events[0]
        self.assertEqual(VisibleText(base.text).text, "カー日！")
        self.assertEqual(state_at(base.text, 1, ["fn"])[0]["fn"].arg, font)
        self.assertEqual(report.ruby_lines, 1)
        self.assertFalse(report.warnings)
        # 0.20.0: the export splits a beat at the font switch after measuring (karaoke tags do not move glyphs)
        self.assertTrue(any(split_karaoke_overrides(r.text) == base.text and "\\fn" + font in r.text for r in captured))
        ruby = ass.events[1]
        self.assertNotIn("\\fn" + font, ruby.text)
        self.assertEqual(VisibleText(ruby.text).text, "にーち")

        # Compare real libass geometry with the same font explicitly applied to the glyph.
        reference = copy.deepcopy(doc)
        reference.members[m.id].format = [{"anchor": anchor_of(ids[1:2]), "props": {"fn": font}}]
        expected, expected_report = self.build(reference, rules(scope_chars=["song"], dash="half", exclaim="full"))
        self.assertEqual(report.owners[1], expected_report.owners[1])
        self.assertEqual(ruby, expected.events[1])

        def timings(text):
            at, tags = 0, []
            for token in tokenize(text):
                if token.kind == "tags":
                    tags.extend((at, tag.name, tag.arg) for tag in token.tags if tag.name in ("k", "kf", "ko", "K"))
                elif token.kind in ("text", "hardspace", "softbreak"):
                    at += len(token.raw) if token.kind == "text" else 1
            return tags

        self.assertEqual(timings(base.text), timings(expected.events[0].text))
        if not stored:
            self.assertEqual(timings(base.text), [(0, "k", "20"), (1, "k", "30"), (2, "k", "40")])
        else:
            def onsets(text):
                """(\\k) Every visible character turns at the start of its syllable; splits keep these instants."""
                now, start, out = 0, 0, []
                for token in tokenize(text):
                    if token.kind == "tags":
                        for tag in token.tags:
                            if tag.name in ("k", "kf", "ko", "K"):
                                start, now = now, now + int(tag.arg)
                    elif token.kind in ("text", "hardspace", "softbreak"):
                        out.extend([start] * (len(token.raw) if token.kind == "text" else 1))
                return out, now

            plain, _plain_report = self.build(doc)
            # 0.20.0: the long-vowel font switch splits カー into \k0 + \k40; the timing it shows is unchanged.
            self.assertEqual(onsets(base.text), onsets(plain.events[0].text))
            self.assertTrue(timings(ruby.text))
        if vertical:
            self.assertEqual(line_tag(ruby.text, "frz").num(), 270)
        ass.save(self.path / ("song-vertical.ass" if vertical else "song-k-ruby.ass"))
        self.assertEqual(doc, saved)

    def test_scope_change_invalidates_export_cache(self):
        doc = document("カーー!", lang="ja", lane="original", style="JP")
        doc.members["m_1"].role = "song"
        cache = ExportCache()
        off = rules(exclaim="full", fonts={"long_vowel": "Long"}, scope_fonts=[])
        _ass, _report = self.build(doc, off, cache=cache)
        on = rules(exclaim="full", fonts={"long_vowel": "Long"}, scope_chars=["song"], scope_fonts=["song"])
        ass, report = self.build(doc, on, cache=cache)
        self.assertEqual(report.laid_out, 1)
        self.assertEqual(VisibleText(ass.events[0].text).text, "カーー！")
        self.assertIn(r"{\fnLong}ーー{\fn", ass.events[0].text)
        _same, report = self.build(doc, on, cache=cache)
        self.assertEqual(report.laid_out, 0)

    def test_font_restores_spans_line_and_named_reset(self):
        doc = document(r"甲!乙{\fnSource}丙!丁{\rOther}戊!己{\r}庚!辛", lang="ja", lane="original", style="JP")
        m = doc.members["m_1"]
        m.overrides = {"fn": "Line", "b": 1}
        m.format = [{"anchor": [{"ins": "i_1", "a": 0, "b": 3}], "props": {"fn": "Span", "i": 1}}]
        ass, _ = self.build(doc, rules(exclaim="full", font="Punct"))
        text = ass.events[0].text
        for i, ch in enumerate(VisibleText(text).text):
            if ch == "！":
                self.assertEqual(state_at(text, i, ["fn"])[0]["fn"].arg, "Punct")
        for expected in (r"{\fnPunct}！{\fnSpan}乙", r"{\fnPunct}！{\fnLine}丁",
                         r"{\fnPunct}！{\fnCourier New}己"):
            self.assertIn(expected, text)
        self.assertIn("{\\fnPunct}！{\\fn" + self.template.style("JP").fontname + "}辛", text)
        self.assertIn(r"\b1", text)
        self.assertIn(r"\i1", text)

    def test_missing_punctuation_font_in_export_check(self):
        from subkit.core.fontcheck import check_fonts, font_runs

        font = "DuoSub Punctuation Missing Font 20261012"
        doc = document("甲!乙", lang="ja", lane="original", style="JP")
        ass, _ = self.build(doc, rules(exclaim="full", font=font))
        self.assertIn((font, "！"), list(font_runs(ass.events[0], ass)))
        checked = check_fonts(ass)
        self.assertTrue(checked.ran, checked.error)
        self.assertIn(font, checked.families)
        self.assertIn(font, checked.missing)
        self.assertTrue(checked.missing[font])
        unused, _ = self.build(doc, rules(font=font))
        # Legacy font now covers unchanged glyphs even with every normalization off.
        self.assertIn((font, "!"), list(font_runs(unused.events[0], unused)))

    def test_every_category_font_export_unchanged_and_raw(self):
        from subkit.core.fontcheck import font_runs, probe_ass

        samples = {"comma": "，", "dunhao": "、", "period": "。", "colon": "：", "ellipsis": "…",
                   "exclaim": "！", "question": "？", "quotes": "「", "brackets": "（",
                   "separator": "・", "tilde": "〜", "dash": "—"}
        for lane, lang, style in (("original", "ja", "JP"), ("translation", "zh-Hans", "CN")):
            for category, glyph in samples.items():
                with self.subTest(lane=lane, category=category):
                    font = category + " Punct"
                    doc = document("甲" + glyph + "乙{@Raw=" + glyph + "}丙", lang=lang, lane=lane, style=style)
                    saved = copy.deepcopy(doc)
                    ass, _ = self.build(doc, rules(lane, fonts={category: font}))
                    text = ass.events[0].text
                    self.assertIn("{\\fn" + font + "}" + glyph + "{\\fn" + self.template.style(style).fontname + "}乙", text)
                    self.assertEqual(VisibleText(text).text, "甲" + glyph + "乙" + glyph + "丙")
                    self.assertEqual(list(font_runs(ass.events[0], ass)).count((font, glyph)), 1)
                    self.assertIn("\\fn" + font, probe_ass(ass))
                    self.assertEqual(doc, saved)

    def test_legacy_font_all_four_unchanged_categories_and_default_scope(self):
        from subkit.core.fontcheck import font_runs

        doc = document('甲...!?“乙”丙', lang="ja", lane="original", style="JP")
        for role in ("dialogue", "top", "alt", "song", "song_jp", "song_cn", "sign", "info"):
            doc.members["m_1"].role = role
            ass, _ = self.build(doc, rules(font="Legacy"))
            runs = list(font_runs(ass.events[0], ass))
            if role in ("dialogue", "top", "alt"):
                self.assertIn(("Legacy", '...!?“'), runs)
                self.assertIn(("Legacy", '”'), runs)
            else:
                # A legacy single font only ever reached dialogue lines; songs and signs stay as exported before.
                self.assertFalse(any(font == "Legacy" for font, _text in runs))

    def test_category_fonts_changed_glyphs_adjacent_merge_and_restoration(self):
        from subkit.core.fontcheck import font_runs

        doc = document('甲,乙...!?“丙”~——丁', lang="ja", lane="original", style="JP")
        ass, _ = self.build(doc, rules(comma="ideographic", ellipsis="single", tilde="wave", dash="half",
            fonts={"comma": "Unused", "dunhao": "Dunhao", "ellipsis": "Dots", "exclaim": "Shared",
                   "question": "Shared", "quotes": "Quotes", "tilde": "Tilde", "dash": "Dash"}))
        text = ass.events[0].text
        self.assertIn(r"{\fnShared}!?{\fn", text)
        self.assertEqual(text.count(r"{\fnShared}"), 1)
        self.assertIn(r"{\fnDash}--{\fn", text)
        self.assertNotIn(r"\fnUnused", text)
        for item in (("Dunhao", "、"), ("Dots", "…"), ("Shared", "!?"), ("Tilde", "〜"), ("Dash", "--")):
            self.assertIn(item, list(font_runs(ass.events[0], ass)))

    def test_adjacent_fonts_across_inline_reset_and_font_boundaries(self):
        doc = document(r"甲!{\fnSource}?{\rOther}!乙", lang="ja", lane="original", style="JP")
        ass, _ = self.build(doc, rules(fonts={"exclaim": "Same", "question": "Same"}))
        text = ass.events[0].text
        for i in (1, 2, 3):
            self.assertEqual(state_at(text, i, ["fn"])[0]["fn"].arg, "Same")
        self.assertIn(r"{\fnSame}!{\fnCourier New}乙", text)
        different, _ = self.build(document("!?字", lang="ja", lane="original", style="JP"),
                                  rules(fonts={"exclaim": "A", "question": "B"}))
        self.assertEqual(state_at(different.events[0].text, 0, ["fn"])[0]["fn"].arg, "A")
        self.assertEqual(state_at(different.events[0].text, 1, ["fn"])[0]["fn"].arg, "B")
        self.assertEqual(state_at(different.events[0].text, 2, ["fn"])[0]["fn"].arg, self.template.style("JP").fontname)

    def test_category_fonts_merge_bounds_and_cache(self):
        doc = document(r"甲!\N~——乙", lang="ja", lane="original", style="JP")
        rule = rules(tilde="wave", dash="half", fonts={"exclaim": "A", "tilde": "B", "dash": "C"})
        rule.merge_lines = (("original", "dialogue"),)
        cache = ExportCache()
        ass, rep = self.build(doc, rule, cache=cache)
        self.assertEqual(VisibleText(ass.events[0].text).text, "甲!  〜--乙")
        for i, font in ((1, "A"), (4, "B"), (5, "C"), (6, "C")):
            self.assertEqual(state_at(ass.events[0].text, i, ["fn"])[0]["fn"].arg, font)
        _same, same_rep = self.build(doc, rule, cache=cache)
        self.assertEqual(same_rep.laid_out, 0)
        other = rules(tilde="wave", dash="half", fonts={"exclaim": "Changed"})
        other.merge_lines = rule.merge_lines
        changed, changed_rep = self.build(doc, other, cache=cache)
        self.assertEqual(changed_rep.laid_out, 1)
        self.assertIn(r"\fnChanged", changed.events[0].text)

    def test_furigana_identities_hit_testing_preview(self):
        doc = document("日，本。日...本", lang="ja", lane="original", style="JP")
        doc.annotations = {str(i): Annotation(str(i), "ruby", "m_1", [{"ins": "i_1", "a": i, "b": i+1}],
            ch, "にち" if ch == "日" else "ほん", parts=[[0, 1, "にち" if ch == "日" else "ほん"]])
            for i, ch in enumerate(doc.members["m_1"].text) if ch in "日本"}
        saved = copy.deepcopy(doc)
        rule = rules(comma=True, period=True, ellipsis="single")
        ass, rep = self.build(doc, rule)
        self.assertEqual(VisibleText(ass.events[0].text).text, "日  本  日…本")
        self.assertEqual(rep.ruby_lines, 4)
        self.assertFalse(rep.warnings)
        ids = rep.owners[0]["ids"]
        self.assertEqual(ids[:4], [("i", "i_1", 0), ("i", "i_1", 1), ("i", "i_1", 1), ("i", "i_1", 2)])
        preview, prep = self.build(doc, rule, mode="preview")
        self.assertEqual(preview.events, ass.events)
        self.assertEqual(prep.owners, rep.owners)
        index = LayoutIndex(ass, rep.owners, measure_cache=self.path / "render.sqlite3")
        try:
            base = next(g for g in index.geometry(index.events_of_group("g_1")) if g.kind == "base")
            for i in (0, 3, 6, 8):
                box = base.chars[i]
                self.assertEqual(index.hit([base], box.cx, box.cy, pad=0).identity, ids[i])
        finally:
            index.close()
        self.assertEqual(doc, saved)
        ass.save(self.path / "mapped-ruby.ass")

    def test_traditional_hide_merge_pinyin(self):
        doc = document(r"互联网，行。\N绿。")
        rule = rules("translation", comma=True, period=True, spaces=True)
        rule.zh_script = "traditional"
        rule.zh_traditional_phrases = True
        rule.merge_lines = (("translation", "dialogue"),)
        ass, rep = self.build(doc, rule, pinyin_mode="PY")
        self.assertEqual(VisibleText(ass.events[0].text).text, "網際網路  行  綠")
        self.assertFalse(any("找不到底字" in w for w in rep.warnings))
        self.assertEqual(rep.pinyin_lines, 5)
        ja = document(r"日。\N本｡", lang="ja", lane="original", style="JP")
        rule = rules(comma=True)
        rule.hide_final_period = True
        rule.merge_lines = (("original", "dialogue"),)
        ass, rep = self.build(ja, rule)
        self.assertEqual(VisibleText(ass.events[0].text).text, "日  本")
        self.assertEqual(rep.hidden_by_rules, 2)

    def test_project_persistence_and_cache_toggle(self):
        project = Project.create(self.path / "project", "test", template_text=default_styles_ass())
        ep = project.add_episode("E1")
        ep.document = document("甲，乙", lang="ja", lane="original", style="JP")
        project.data.setdefault("settings", {}).setdefault("display", {}).setdefault("punctuation", {})["original"] = {"comma": True}
        project.save()
        self.assertTrue(DisplayRules.from_settings(Project.open(project.root).settings()).active)
        ass, _ = ep.build_export()
        self.assertEqual(ass.events[0].text, "甲  乙")
        cache = ExportCache()
        off, _ = self.build(ep.document, cache=cache)
        on, rep = self.build(ep.document, rules(comma=True), cache=cache)
        self.assertEqual(rep.laid_out, 1)
        same, rep = self.build(ep.document, rules(comma=True), cache=cache)
        self.assertEqual(rep.laid_out, 0)
        self.assertEqual(same.serialize(), on.serialize())
        after, _ = self.build(ep.document, cache=cache)
        self.assertEqual(after.serialize(), off.serialize())
