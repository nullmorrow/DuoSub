"""Paired dialogue fades are derived, cache-aware and inherited by generated ruby."""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.ass.document import AssDocument
from subkit.ass.text import line_tag
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.fades import inherited_fades, query_fade_inheritance
from subkit.core.readings import anchor_of, visible_map
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.styles_default import default_styles_ass


def member(mid, lane, text, *, start=159510, end=163020):
    return Member(mid, "g", "ja" if lane == "original" else "zh-Hant", lane, "dialogue", start, end,
                  style="JP" if lane == "original" else "CN", runs=[Run(text, ins="i_" + mid, b=len(text))])


class FadeInheritanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="duosub_fade_")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.cn = member("cn", "translation", r"{\fad(0,720)}沒錯 整個世界都將為之驚恐的怪人")
        self.jp = member("jp", "original", "恐ろしい怪人")
        self.doc = Document("E1", groups=[Group("g", members=["cn", "jp"])], members={"cn": self.cn, "jp": self.jp})
        self.template = AssDocument.parse(default_styles_ass())

    def export(self, *, cache=None, ruby=False, mode="release"):
        return build_ass(self.doc, {}, self.template, self.path, ExportOptions(ruby=ruby, mode=mode),
                         cache=cache, measure_cache=self.path / "render.sqlite3")

    @staticmethod
    def text(ass, style="JP"):
        return next(e.text for e in ass.events if e.style == style)

    def add_translation(self, text, *, mid="cn2", start=159510, end=163020):
        m = member(mid, "translation", text, start=start, end=end)
        self.doc.members[mid] = m
        self.doc.groups[0].members.append(mid)
        return m

    def test_fad_and_fade_copy_only_fade_without_changing_sources_or_overrides(self):
        for raw in (r"\fad(0,720)", r"\fade(255,0,255,0,200,2790,3510)"):
            with self.subTest(raw=raw):
                self.cn.runs[0].text = "{" + raw + r"\blur3\clip(1,2,3,4)\frz25}" + "翻譯"
                self.jp.overrides = {"raw": r"\blur1", "b": 1}
                before = copy.deepcopy(self.doc)
                decision = query_fade_inheritance(self.doc, "jp")
                self.assertEqual((decision.raw, decision.donors, decision.reason), (raw, ("cn",), "inherited"))
                for mode in ("release", "preview", "work"):
                    ass, _ = self.export(mode=mode)
                    text = self.text(ass)
                    self.assertIn(raw, text)
                    self.assertIn(r"\blur1", text)
                    self.assertNotIn(r"\blur3", text)
                    self.assertNotIn(r"\clip", text)
                    self.assertNotIn(r"\frz25", text)
                self.assertEqual(self.doc, before)

    def test_original_source_and_manual_fades_take_priority_including_explicit_off(self):
        for own in (r"\fad(100,200)", r"\fade(255,0,255,0,50,3000,3510)", r"\fad(0,0)"):
            for where in ("source", "override"):
                with self.subTest(own=own, where=where):
                    self.jp.runs[0].text = ("{" + own + "}" if where == "source" else "") + "恐ろしい怪人"
                    self.jp.overrides = {"raw": own} if where == "override" else {}
                    self.assertEqual(query_fade_inheritance(self.doc, "jp").reason, "own_fade")
                    ass, _ = self.export()
                    self.assertIn(own, self.text(ass))
                    self.assertNotIn(r"\fad(0,720)", self.text(ass))

    def test_translation_override_replaces_its_source_fade_and_keeps_other_tags_local(self):
        self.cn.overrides["raw"] = r"\blur3\fade(255,0,255,0,100,3000,3510)"
        ass, _ = self.export()
        text = self.text(ass)
        self.assertIn(r"\fade(255,0,255,0,100,3000,3510)", text)
        self.assertNotIn(r"\fad(0,720)", text)
        self.assertNotIn(r"\blur3", text)

    def test_multiple_translations_require_one_consistent_fade_and_time(self):
        other = self.add_translation(r"{\fad(0,720.0)}第二行")
        self.assertEqual(query_fade_inheritance(self.doc, "jp").donors, ("cn", "cn2"))
        for text, end, reason in ((r"{\fad(0,500)}第二行", 163020, "conflicting_translation_fades"),
                                   ("沒有淡出", 163020, "missing_or_invalid_translation_fade"),
                                   (r"{\fad(0,720)}第二行", 163500, "different_times")):
            with self.subTest(text=text, end=end):
                other.runs[0].text, other.end = text, end
                self.assertEqual(query_fade_inheritance(self.doc, "jp").reason, reason)
                self.assertNotIn(r"\fad", self.text(self.export()[0]))

    def test_final_ass_times_match_but_different_timelines_are_never_retimed(self):
        self.jp.start, self.jp.end = 159511, 163024
        self.assertTrue(query_fade_inheritance(self.doc, "jp").raw)
        self.jp.end = 163025
        before = (self.jp.start, self.jp.end)
        self.assertEqual(query_fade_inheritance(self.doc, "jp").reason, "different_times")
        self.assertNotIn(r"\fad", self.text(self.export()[0]))
        self.assertEqual((self.jp.start, self.jp.end), before)
        self.jp.end = self.jp.start
        self.assertFalse(query_fade_inheritance(self.doc, "jp").raw)

    def test_top_level_only_and_invalid_or_mixed_tags_fail_closed(self):
        for tags in (r"\t(0,500,\fad(0,720))", r"\fad(0,720)\fade(255,0,255,0,0,3000,3510)",
                     r"\fad(0,720)\fad(0,500)", r"\fad(0)", r"\fad(0,720", r"\fad(-1,720)", r"\fad(nan,720)",
                     r"\fade(256,0,255,0,0,3000,3510)", r"\fade(255,0,255,200,100,3000,3510)"):
            with self.subTest(tags=tags):
                self.cn.runs[0].text = "{" + tags + "}翻譯"
                self.assertFalse(query_fade_inheritance(self.doc, "jp").raw)
                self.assertNotIn(r"\fad", self.text(self.export()[0]))
        self.cn.runs[0].text = r"\fad(0,720) is plain text"
        self.assertFalse(query_fade_inheritance(self.doc, "jp").raw)

    def test_hidden_deleted_comments_empty_and_other_groups_do_not_donate(self):
        for attr, value in (("role", "comment"), ("source_kind", "Comment")):
            with self.subTest(attr=attr):
                previous = getattr(self.cn, attr)
                setattr(self.cn, attr, value)
                self.assertFalse(query_fade_inheritance(self.doc, "jp").raw)
                setattr(self.cn, attr, previous)
        self.cn.runs[0].text = r"{\fad(0,720)}"
        self.assertFalse(query_fade_inheritance(self.doc, "jp").raw)
        self.cn.runs[0].text += "翻譯"
        self.doc.groups[0].flags = ["hidden"]
        self.assertFalse(query_fade_inheritance(self.doc, "jp").raw)
        self.assertEqual(self.export()[0].events, [])
        self.doc.groups[0].flags = []
        self.doc.groups[0].members.remove("cn")
        del self.doc.members["cn"]
        self.assertFalse(query_fade_inheritance(self.doc, "jp").raw)
        self.doc.groups.append(Group("other", members=["other-cn"]))
        other = self.add_translation(r"{\fad(0,720)}無關翻譯", mid="other-cn")
        self.doc.groups[0].members.remove("other-cn")
        other.group = "other"
        self.assertFalse(query_fade_inheritance(self.doc, "jp").raw)

    def test_song_sign_and_non_japanese_originals_are_unchanged(self):
        for kind in ("song", "sign", "info"):
            with self.subTest(kind=kind):
                self.doc.groups[0].kind = kind
                self.assertFalse(query_fade_inheritance(self.doc, "jp").raw)
        self.doc.groups[0].kind = "dialogue"
        self.jp.lang = "en"
        self.assertNotIn("jp", inherited_fades(self.doc))
        self.assertNotIn(r"\fad", self.text(self.export()[0]))

    def test_cache_follows_translation_fade_removal_manual_override_and_regrouping(self):
        cache = ExportCache()
        self.assertIn(r"\fad(0,720)", self.text(self.export(cache=cache)[0]))
        self.assertEqual(self.export(cache=cache)[1].laid_out, 0)
        self.cn.runs[0].text = r"{\fad(0,500)}翻譯"
        self.assertIn(r"\fad(0,500)", self.text(self.export(cache=cache)[0]))
        self.cn.overrides["raw"] = r"\fad(0,250)"
        self.assertIn(r"\fad(0,250)", self.text(self.export(cache=cache)[0]))
        self.doc.groups[0].members.remove("cn")
        self.doc.groups.append(Group("other", members=["cn"]))
        self.cn.group = "other"
        self.doc.structure_changed()
        self.assertNotIn(r"\fad", self.text(self.export(cache=cache)[0]))
        self.doc.groups[0].members.append("cn")
        self.doc.groups.pop()
        self.cn.group = "g"
        self.doc.structure_changed()
        self.assertIn(r"\fad(0,250)", self.text(self.export(cache=cache)[0]))
        self.jp.overrides["raw"] = r"\fad(0,0)"
        self.assertIn(r"\fad(0,0)", self.text(self.export(cache=cache)[0]))
        del self.jp.overrides["raw"]
        del self.cn.overrides["raw"]
        self.cn.runs[0].text = "翻譯"
        cached, _ = self.export(cache=cache)
        self.assertNotIn(r"\fad", self.text(cached))
        self.assertEqual(cached.serialize(), self.export()[0].serialize())

    def test_many_originals_each_keep_their_own_fade_and_other_groups_stay_clean(self):
        jp2 = member("jp2", "original", r"{\fad(0,0)}二人目")
        self.doc.members[jp2.id] = jp2
        self.doc.groups[0].members.append(jp2.id)
        ass, report = self.export()
        by_member = {owner["member"]: event.text for owner, event in zip(report.owners, ass.events) if owner.get("kind") == "base"}
        self.assertIn(r"\fad(0,720)", by_member["jp"])
        self.assertIn(r"\fad(0,0)", by_member["jp2"])

    def test_generated_ruby_follows_final_fade_and_cache_changes(self):
        ids = visible_map(self.jp)[1]
        self.doc.annotations["a"] = Annotation("a", "ruby", "jp", anchor_of(ids[4:6]), "怪人", "かいじん")
        cache = ExportCache()
        before = copy.deepcopy(self.doc)
        for value in (720, 300):
            self.cn.runs[0].text = rf"{{\fad(0,{value})}}翻譯"
            ass, report = self.export(cache=cache, ruby=True)
            events = [(event, owner) for event, owner in zip(ass.events, report.owners) if owner.get("member") == "jp"]
            self.assertTrue(any(owner["kind"] == "ruby" for _event, owner in events))
            for event, _owner in events:
                self.assertEqual(line_tag(event.text, "fad").arg, f"0,{value}")
                self.assertEqual((event.start, event.end), (15951, 16302))
        self.assertEqual(self.jp, before.members["jp"])
        self.assertEqual(self.doc.annotations, before.annotations)


if __name__ == "__main__":
    unittest.main()
