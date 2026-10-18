"""Translation beat links: stored identity keys, following and single-step undo."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.core import karaoke as kara
from subkit.core.ops import Editor
from subkit.model.document import Document, Group, Member, Run
from subkit.model.serialize import document_from_json, document_to_json


class KaraokeLinks(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = Document("links", groups=[Group("g", "song", [])])
        self.ed = Editor(self.doc, {})
        self.jp = self.member("jp", "かなよ", "original")
        self.cn = self.member("cn", "你好世界", "translation")
        self.write(self.jp, [(1000, 1400), (1400, 1800), (1800, 2200)])
        self.write(self.cn, [(1000, 1200), (1200, 1600), (1600, 1900), (1900, 2200)])

    def member(self, mid, text, lane, *, gid="g"):
        m = Member(mid, gid, "ja" if lane == "original" else "zh-Hans", lane, "song", 900, 4000,
                   runs=[Run(text, ins="i_" + mid, a=0, b=len(text))])
        self.doc.members[mid] = m
        self.doc.group(gid).members.append(mid)
        self.doc.structure_changed()
        return m

    def beats(self, m):
        return kara.effective(self.doc, self.doc.members[m.id]).beats

    def track(self, m):
        return kara.tracks_of(self.doc, m.id, dormant=True)[0]

    def times(self, m):
        return [(b.t0, b.t1) for b in self.beats(m)]

    def write(self, m, times, *, by="tap", twins=False):
        beats = self.beats(m)
        for b, (a, z) in zip(beats, times):
            b.t0, b.t1, b.by = a, z, by
        self.ed.set_karaoke(m.id, beats, twins=twins)

    def link(self, *, twins=False):
        beats = self.beats(self.cn)
        originals = self.beats(self.jp)
        beats[0].link["start"] = kara.beat_link(originals[0], "start")
        beats[-1].link["end"] = kara.beat_link(originals[-1], "end")
        self.ed.set_karaoke(self.cn.id, beats, twins=twins)

    def test_links_roundtrip_copy_and_use_stored_timing_keys(self):
        self.link()
        link = self.track(self.cn)["beats"][0]["link"]["start"]
        self.assertEqual(link, {**{k: self.track(self.jp)["beats"][0][k] for k in ("core", "k", "sung")},
                                "edge": "start"})
        reopened = document_from_json(json.loads(json.dumps(document_to_json(self.doc), ensure_ascii=False)))
        self.assertEqual(reopened.karaoke, self.doc.karaoke)
        copied = kara.effective(reopened, reopened.members[self.cn.id]).beats
        self.assertEqual(copied[0].link, self.beats(self.cn)[0].link)
        copied[0].link.clear()
        self.assertTrue(self.track(self.cn)["beats"][0]["link"])

    def test_move_all_boundaries_and_twins_one_undo_redo(self):
        twin = self.member("glow", "你好世界", "translation")
        self.link(twins=True)
        before = copy.deepcopy(self.doc.karaoke)
        count = len(self.ed.history.done)
        self.write(self.jp, [(1100, 1500), (1500, 1900), (1900, 2300)], by="drag")
        expected = [(1100, 1300), (1300, 1700), (1700, 2000), (2000, 2300)]
        self.assertEqual(self.times(self.cn), expected)
        self.assertEqual(self.times(twin), expected)
        self.assertEqual(self.track(twin)["beats"][0]["core"][0]["ins"], "i_glow")
        self.assertEqual(len(self.ed.history.done), count + 1)
        self.assertTrue(self.ed.history.done[-1].keeps_text)
        after = copy.deepcopy(self.doc.karaoke)
        self.ed.history.undo()
        self.assertEqual(self.doc.karaoke, before)
        self.ed.history.redo()
        self.assertEqual(self.doc.karaoke, after)

    def test_scale_uses_time_ratios_not_beat_counts(self):
        self.link()
        self.write(self.jp, [(1200, 2000), (2000, 2800), (2800, 3600)])
        self.assertEqual(self.times(self.cn), [(1200, 1600), (1600, 2400), (2400, 3000), (3000, 3600)])

    def test_one_link_moves_shared_edge_only_outside_interpolation(self):
        beats = self.beats(self.cn)
        ref = kara.beat_link(self.beats(self.jp)[1], "start")
        beats[0].t1 = beats[1].t0 = 1400
        beats[0].link["end"] = copy.deepcopy(ref)
        beats[1].link["start"] = copy.deepcopy(ref)
        self.ed.set_karaoke(self.cn.id, beats)
        self.write(self.jp, [(1000, 1450), (1450, 1800), (1800, 2200)])
        self.assertEqual(self.times(self.cn), [(1000, 1450), (1450, 1600), (1600, 1900), (1900, 2200)])

    def test_rounding_and_rest_boundaries_interpolate(self):
        self.write(self.cn, [(1000, 1101), (1200, 1600), (1700, 1900), (1900, 2200)])
        self.link()
        self.write(self.jp, [(1100, 1500), (1500, 1900), (1900, 2900)])
        self.assertEqual(self.times(self.cn), [(1100, 1252), (1400, 2000), (2150, 2450), (2450, 2900)])

    def test_zero_time_span_between_links_does_not_divide_by_zero(self):
        self.write(self.cn, [(1000, 1000)] * 4)
        self.link()
        self.write(self.jp, [(1100, 1500), (1500, 1900), (1900, 2500)])
        self.assertEqual(self.times(self.cn), [(1100, 1300), (1500, 1700), (1900, 2100), (2300, 2500)])

    def test_split_and_merge_drop_missing_keys_report_issue_and_undo(self):
        self.jp.runs = [Run("ガーなよ", ins="i_jp", a=0, b=4)]
        self.ed.clear_karaoke(self.jp.id)
        self.write(self.jp, [(1000, 1400), (1400, 1800), (1800, 2200)])
        self.link()
        before = copy.deepcopy(self.doc.karaoke)
        at = kara.boundary_key(self.doc, self.jp, 1)
        self.ed.split_beat(self.jp.id, at, 1200)
        self.assertNotIn("link", self.track(self.cn)["beats"][0])
        self.assertIn(kara.LINK_MISSING, kara.effective(self.doc, self.cn).issues)
        self.assertTrue(self.track(self.cn)["beats"][-1]["link"])
        self.assertIn(kara.LINK_MISSING, dict(kara.unfinished(self.doc))[self.cn.id])
        self.ed.history.undo()
        self.assertEqual(self.doc.karaoke, before)
        # Merge removes the linked last beat's key too, including its end edge.
        self.ed.merge_beats(self.jp.id, kara.boundary_key(self.doc, self.jp, 3))
        self.assertNotIn("link", self.track(self.cn)["beats"][-1])
        self.assertIn(kara.LINK_MISSING, self.track(self.cn)["issues"])

    def test_surviving_link_follows_after_split_merge(self):
        self.jp.runs = [Run("ガーなよ", ins="i_jp", a=0, b=4)]
        self.ed.clear_karaoke(self.jp.id)
        self.write(self.jp, [(1000, 1400), (1400, 1800), (1800, 2200)])
        beats = self.beats(self.cn)
        beats[-1].link["end"] = kara.beat_link(self.beats(self.jp)[-1], "end")
        self.ed.set_karaoke(self.cn.id, beats)
        at = kara.boundary_key(self.doc, self.jp, 1)
        self.ed.split_beat(self.jp.id, at, 1200)
        self.ed.merge_beats(self.jp.id, at)
        self.write(self.jp, [(1000, 1400), (1400, 1800), (1800, 2500)])
        self.assertEqual(self.beats(self.cn)[-1].t1, 2500)
        self.assertNotIn(kara.LINK_MISSING, kara.effective(self.doc, self.cn).issues)

    def test_shift_has_no_double_movement_and_is_one_undo_step(self):
        self.link()
        before = copy.deepcopy(document_to_json(self.doc))
        count = len(self.ed.history.done)
        self.ed.shift_group("g", 200)
        self.assertEqual(self.times(self.cn), [(1200, 1400), (1400, 1800), (1800, 2100), (2100, 2400)])
        self.assertEqual(len(self.ed.history.done), count + 1)
        self.ed.history.undo()
        after = document_to_json(self.doc)
        after["revision"] = before["revision"]
        self.assertEqual(after, before)

    def test_convert_back_revives_links_and_follows_in_same_step(self):
        self.link()
        self.ed.convert_to_dialogue(["g"])
        # Dormant tracks can have moved independently while their links were asleep.
        for b in self.track(self.jp)["beats"]:
            b["t0"] += 150
            b["t1"] += 150
        before = copy.deepcopy(self.doc.karaoke)
        count = len(self.ed.history.done)
        self.ed.convert_to_song(["g"], "OP")
        self.assertEqual(self.times(self.cn), [(1150, 1350), (1350, 1750), (1750, 2050), (2050, 2350)])
        self.assertEqual(len(self.ed.history.done), count + 1)
        self.ed.history.undo()
        self.assertEqual(self.doc.karaoke, before)

    def test_deleted_text_key_drops_and_issue_survives_rewrite(self):
        self.link()
        self.ed.replace_text(self.jp.id, "なよ")
        self.ed.set_karaoke(self.jp.id, self.beats(self.jp))
        self.assertNotIn("link", self.track(self.cn)["beats"][0])
        self.ed.set_karaoke(self.cn.id, self.beats(self.cn))
        self.assertIn(kara.LINK_MISSING, kara.effective(self.doc, self.cn).issues)

    def test_dormant_translation_and_other_group_do_not_follow(self):
        self.link()
        self.track(self.cn)["dormant"] = True
        self.doc.groups.append(Group("other", "song", []))
        self.doc.structure_changed()
        other = self.member("other", "你好世界", "translation", gid="other")
        self.write(other, [(1000, 1200), (1200, 1600), (1600, 1900), (1900, 2200)])
        # A foreign key is deliberately not followed across group boundaries.
        self.track(other)["beats"][0]["link"] = copy.deepcopy(self.track(self.cn)["beats"][0]["link"])
        before = [copy.deepcopy(self.track(m)) for m in (self.cn, other)]
        self.write(self.jp, [(1100, 1500), (1500, 1900), (1900, 2300)])
        self.assertEqual([self.track(m) for m in (self.cn, other)], before)

    def test_translation_split_merge_preserve_outer_links(self):
        self.cn.runs = [Run("hello world", ins="i_cn", a=0, b=11)]
        self.ed.clear_karaoke(self.cn.id)
        self.write(self.cn, [(1000, 1600), (1600, 2200)])
        self.link()
        at = kara.boundary_key(self.doc, self.cn, 2)
        self.ed.split_beat(self.cn.id, at)
        self.ed.merge_beats(self.cn.id, at)
        self.write(self.jp, [(1100, 1500), (1500, 1900), (1900, 2300)])
        self.assertEqual(self.times(self.cn), [(1100, 1700), (1700, 2300)])

    def test_import_tag_times_follows_links_in_one_step(self):
        self.link()
        self.jp.runs = [Run("{\\kf50}", "tag"), self.jp.runs[0].slice(0, 1),
                        Run("{\\kf60}", "tag"), self.jp.runs[0].slice(1, 2),
                        Run("{\\kf70}", "tag"), self.jp.runs[0].slice(2, 3)]
        before = copy.deepcopy(self.doc.karaoke)
        count = len(self.ed.history.done)
        self.ed.import_karaoke(self.jp.id)
        self.assertEqual(self.times(self.cn), [(900, 1200), (1200, 1800), (1800, 2250), (2250, 2700)])
        self.assertEqual(len(self.ed.history.done), count + 1)
        self.ed.history.undo()
        self.assertEqual(self.doc.karaoke, before)

    def test_unit_change_drops_reference_to_changed_beat(self):
        self.jp.runs = [Run("ガーなよ", ins="i_jp", a=0, b=4)]
        self.ed.clear_karaoke(self.jp.id)
        self.write(self.jp, [(1000, 1400), (1400, 1800), (1800, 2200)])
        self.link()
        self.ed.set_karaoke_state([self.jp.id], unit="mora")
        self.assertNotIn("link", self.track(self.cn)["beats"][0])
        self.assertIn(kara.LINK_MISSING, kara.effective(self.doc, self.cn).issues)

    def test_clearing_original_edits_drops_split_beat_link_and_undo_restores_it(self):
        self.jp.runs = [Run("ガーなよ", ins="i_jp", a=0, b=4)]
        self.ed.clear_karaoke(self.jp.id)
        self.write(self.jp, [(1000, 1400), (1400, 1800), (1800, 2200)])
        self.ed.split_beat(self.jp.id, kara.boundary_key(self.doc, self.jp, 1), 1200)
        self.link()
        before = copy.deepcopy(self.doc.karaoke)
        self.ed.clear_karaoke(self.jp.id)
        self.assertNotIn("link", self.track(self.cn)["beats"][0])
        self.assertIn(kara.LINK_MISSING, kara.effective(self.doc, self.cn).issues)
        self.ed.history.undo()
        self.assertEqual(self.doc.karaoke, before)


if __name__ == "__main__":
    unittest.main()
