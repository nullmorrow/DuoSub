"""Identity-keyed manual karaoke boundaries and their release output (KR1)."""

from __future__ import annotations

import copy
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.ass import AssDocument
from subkit.ass.text import insert_at_visible
from subkit.core import karaoke as kara
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.ops import Editor
from subkit.core.readings import visible_map
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.model.serialize import document_from_json, document_to_json
from subkit.styles_default import default_styles_ass


class KaraokeEdits(unittest.TestCase):
    def setup_line(self, text: str, *, reading: str = "", unit: str = "syllable", durations=()) -> None:
        self.m = Member("m_song", "g_song", "ja", "original", "song", 1000, 3000, style="JP",
                        runs=[Run(text, ins="i_song", a=0, b=len(text))])
        self.doc = Document("E1", groups=[Group("g_song", "song", [self.m.id])], members={self.m.id: self.m})
        if reading:
            self.doc.annotations["a_song"] = Annotation("a_song", "ruby", self.m.id,
                [{"ins": "i_song", "a": 0, "b": len(text)}], text, reading, status="confirmed",
                parts=[[0, len(text), reading]])
        self.ed = Editor(self.doc, {})
        if durations:
            beats = kara.derive(self.doc, self.m, unit)
            self.assertEqual(len(beats), len(durations))
            t = 1100
            for b, d in zip(beats, durations):
                b.t0, b.t1, b.by = t, t + d, "tap"
                t += d
            self.ed.set_karaoke(self.m.id, beats, unit=unit)

    def view(self):
        return kara.effective(self.doc, self.doc.members[self.m.id])

    def key(self, at: int, offset=None):
        return kara.boundary_key(self.doc, self.doc.members[self.m.id], at, offset)

    def times(self):
        return [(b.t0, b.t1, b.by) for b in self.view().beats]

    def base(self):
        view = self.view()
        vt, ids = visible_map(self.doc.members[self.m.id])
        return insert_at_visible(vt.text, kara.base_inserts(view, ids, 100))

    def test_split_kana_clamps_time_and_is_one_undo_step(self):
        self.setup_line("ガー", durations=[400])
        before = copy.deepcopy(self.doc.karaoke)
        self.assertTrue(self.ed.split_beat(self.m.id, self.key(1), -500))
        self.assertEqual([b.sung for b in self.view().beats], ["が", "ー"])
        self.assertEqual(self.times(), [(1100, 1110, "manual"), (1110, 1500, "manual")])
        self.assertEqual(len(self.ed.history.done), 2)
        self.assertTrue(self.ed.history.done[-1].keeps_text)
        after = copy.deepcopy(self.doc.karaoke)
        self.ed.history.undo()
        self.assertEqual(self.doc.karaoke, before)
        self.ed.history.redo()
        self.assertEqual(self.doc.karaoke, after)
        self.assertEqual(self.m.text, "ガー")

    def test_split_high_clamp_and_minimum_span(self):
        self.setup_line("ガー", durations=[20])
        self.ed.split_beat(self.m.id, self.key(1), 99999)
        self.assertEqual(self.times(), [(1100, 1110, "manual"), (1110, 1120, "manual")])
        self.setup_line("ガー", durations=[19])
        before = copy.deepcopy(self.doc.karaoke)
        with self.assertRaisesRegex(ValueError, "20 毫秒"):
            self.ed.split_beat(self.m.id, self.key(1))
        self.assertEqual(self.doc.karaoke, before)
        self.assertEqual(len(self.ed.history.done), 1)

    def test_word_split_uses_character_proportion(self):
        self.setup_line("Just!", durations=[400])
        self.ed.split_beat(self.m.id, self.key(1))
        self.assertEqual(self.times(), [(1100, 1200, "manual"), (1200, 1500, "manual")])
        self.assertEqual(self.base(), "{\\kf10}{\\kf10}J{\\kf30}ust!")

    def test_ruby_split_and_merge_tags(self):
        self.setup_line("学", reading="がっ", durations=[400])
        key = self.key(0, 1)
        self.ed.split_beat(self.m.id, key, 1250)
        self.assertEqual(self.base(), "{\\kf10}{\\kf40}学")
        self.assertEqual(kara.ruby_text(self.view(), "a_song", 0, "がっ", 100),
                         "{\\kf10}{\\kf15}が{\\kf25}っ")
        before = copy.deepcopy(self.doc.karaoke)
        self.ed.merge_beats(self.m.id, key)
        self.assertEqual(self.times(), [(1100, 1500, "manual")])
        self.assertEqual(kara.ruby_text(self.view(), "a_song", 0, "がっ", 100), "{\\kf10}{\\kf40}がっ")
        self.assertEqual(len(self.ed.history.done), 3)
        self.ed.history.undo()
        self.assertEqual(self.doc.karaoke, before)

    def test_merge_kana_slots_and_resplit(self):
        self.setup_line("メッキ", unit="mora", durations=[100, 200, 300])
        key = self.key(1)
        self.ed.merge_beats(self.m.id, key)
        self.assertEqual([b.sung for b in self.view().beats], ["めっ", "き"])
        self.assertEqual(self.base(), "{\\kf10}{\\kf30}メッ{\\kf30}キ")
        self.ed.split_beat(self.m.id, key, 1200)
        self.assertEqual(self.times(), [(1100, 1200, "manual"), (1200, 1400, "manual"), (1400, 1700, "tap")])
        self.assertEqual(len(next(iter(self.doc.karaoke.values()))["edits"]), 1)

    def test_merge_across_ruby_and_kana_slots(self):
        self.setup_line("学か", durations=())
        self.doc.annotations["a_song"] = Annotation("a_song", "ruby", self.m.id,
            [{"ins": "i_song", "a": 0, "b": 1}], "学", "がっ", status="confirmed", parts=[[0, 1, "がっ"]])
        beats = kara.derive(self.doc, self.m)
        for b, a, z in zip(beats, [1100, 1300], [1300, 1400]):
            b.t0, b.t1, b.by = a, z, "tap"
        self.ed.set_karaoke(self.m.id, beats)
        self.ed.merge_beats(self.m.id, self.key(1))
        self.assertEqual(self.base(), "{\\kf10}{\\kf30}学か")
        self.assertEqual(kara.ruby_text(self.view(), "a_song", 0, "がっ", 100), "{\\kf10}{\\kf20}がっ")

    def test_merge_two_ruby_pieces_waits_in_order(self):
        self.setup_line("学火")
        for aid, pos, reading in [("a_one", 0, "がっ"), ("a_two", 1, "か")]:
            self.doc.annotations[aid] = Annotation(aid, "ruby", self.m.id,
                [{"ins": "i_song", "a": pos, "b": pos + 1}], self.m.text[pos], reading,
                status="confirmed", parts=[[0, 1, reading]])
        beats = kara.derive(self.doc, self.m)
        for b, a, z in zip(beats, [1100, 1300], [1300, 1400]):
            b.t0, b.t1, b.by = a, z, "tap"
        self.ed.set_karaoke(self.m.id, beats)
        self.ed.merge_beats(self.m.id, self.key(1, 0))
        self.assertEqual(self.base(), "{\\kf10}{\\kf30}学火")
        self.assertEqual(kara.ruby_text(self.view(), "a_one", 0, "がっ", 100), "{\\kf10}{\\kf20}がっ")
        self.assertEqual(kara.ruby_text(self.view(), "a_two", 0, "か", 100), "{\\kf30}{\\kf10}か")

    def test_split_inside_cross_slot_merge_preserves_other_ruby_beats(self):
        self.setup_line("学か", reading="", durations=())
        self.doc.annotations["a_song"] = Annotation("a_song", "ruby", self.m.id,
            [{"ins": "i_song", "a": 0, "b": 1}], "学", "あがっ", status="confirmed", parts=[[0, 1, "あがっ"]])
        beats = kara.derive(self.doc, self.m)
        for b, a, z in zip(beats, [1100, 1200, 1400], [1200, 1400, 1500]):
            b.t0, b.t1, b.by = a, z, "tap"
        self.ed.set_karaoke(self.m.id, beats)
        self.ed.merge_beats(self.m.id, self.key(1))
        self.assertEqual(kara.ruby_text(self.view(), "a_song", 0, "あがっ", 100),
                         "{\\kf10}{\\kf10}あ{\\kf20}がっ")
        self.ed.split_beat(self.m.id, self.key(0, 2), 1300)
        self.assertEqual([b.sung for b in self.view().beats], ["あ", "が", "っか"])
        self.assertEqual(kara.ruby_text(self.view(), "a_song", 0, "あがっ", 100),
                         "{\\kf10}{\\kf10}あ{\\kf10}が{\\kf10}っ")

    def test_save_reopen_and_text_identity_survives(self):
        self.setup_line("ガーな", durations=[400, 300])
        self.ed.split_beat(self.m.id, self.key(1), 1300)
        before = self.times()
        tmp = Path(tempfile.mkdtemp())
        try:
            path = tmp / "document.json"
            path.write_text(json.dumps(document_to_json(self.doc), ensure_ascii=False), encoding="utf-8")
            reopened = document_from_json(json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(reopened.karaoke, self.doc.karaoke)
            self.assertEqual([(b.t0, b.t1, b.by) for b in kara.effective(reopened, reopened.members[self.m.id]).beats], before)
        finally:
            for p in tmp.rglob("*"):
                if p.is_file():
                    os.chmod(p, stat.S_IREAD | stat.S_IWRITE)
            shutil.rmtree(tmp)
        self.ed.replace_text(self.m.id, "ガーなよ")
        self.assertEqual(self.times()[:3], before)
        self.assertEqual(self.view().beats[-1].by, "placeholder")

    def test_deleted_key_reports_issue_and_drops_effect(self):
        self.setup_line("ガーな", durations=[400, 300])
        self.ed.split_beat(self.m.id, self.key(1))
        self.ed.replace_text(self.m.id, "ガな")
        self.assertIn(kara.EDIT_MISSING, self.view().issues)
        self.assertEqual([b.sung for b in self.view().beats], ["が", "な"])
        self.assertIn(kara.EDIT_MISSING, kara.unfinished(self.doc)[0][1])

    def test_changed_reading_does_not_reuse_offset(self):
        self.setup_line("学", reading="がっ", durations=[400])
        self.ed.split_beat(self.m.id, self.key(0, 1))
        self.ed.set_reading("a_song", "きゃ")
        self.assertEqual([b.sung for b in self.view().beats], ["きゃ"])
        self.assertIn(kara.EDIT_MISSING, self.view().issues)
        self.assertFalse(self.view().complete)

    def test_unit_switch_keeps_valid_splits_and_merges(self):
        self.setup_line("ガーな", durations=[400, 300])
        self.ed.split_beat(self.m.id, self.key(1))
        self.ed.set_karaoke_state([self.m.id], unit="mora")
        self.assertEqual([b.sung for b in self.view().beats], ["が", "ー", "な"])
        self.assertTrue(self.view().complete)
        self.ed.merge_beats(self.m.id, self.key(1))
        self.ed.set_karaoke_state([self.m.id], unit="syllable")
        self.assertEqual([b.sung for b in self.view().beats], ["がー", "な"])
        self.assertNotIn(kara.EDIT_MISSING, self.view().issues)
        self.assertTrue(self.view().complete)

    def test_twins_follow_in_same_transaction_with_own_keys(self):
        self.setup_line("学", reading="がっ", durations=[400])
        twin = copy.deepcopy(self.m)
        twin.id, twin.layer = "m_glow", 1
        twin.runs = [Run("学", ins="i_glow", a=0, b=1)]
        ann = copy.deepcopy(self.doc.annotations["a_song"])
        ann.id, ann.member, ann.anchor = "a_glow", twin.id, [{"ins": "i_glow", "a": 0, "b": 1}]
        self.doc.annotations[ann.id] = ann
        self.doc.members[twin.id] = twin
        self.doc.groups[0].members.append(twin.id)
        self.ed.split_beat(self.m.id, self.key(0, 1), 1250)
        tv = kara.effective(self.doc, twin)
        self.assertEqual([(b.t0, b.t1, b.by) for b in tv.beats], self.times())
        tr = kara.tracks_of(self.doc, twin.id)[0]
        self.assertEqual(tr["edits"][0]["at"]["core"][0]["ins"], "i_glow")
        self.assertEqual(kara.ruby_text(tv, ann.id, 0, "がっ", 100), "{\\kf10}{\\kf15}が{\\kf25}っ")
        self.ed.merge_beats(self.m.id, self.key(0, 1))
        self.assertEqual(len(kara.effective(self.doc, twin).beats), 1)
        self.ed.history.undo()
        self.assertEqual(len(kara.effective(self.doc, twin).beats), 2)

    def test_placeholder_edits_export_and_timing_rewrite_preserves_edits(self):
        self.setup_line("ガー")
        self.ed.split_beat(self.m.id, self.key(1))
        self.assertEqual(self.view().state, "placeholder")
        self.assertFalse(self.view().complete)
        self.assertEqual(self.base(), "{\\kf100}ガ{\\kf100}ー")
        edits = copy.deepcopy(kara.tracks_of(self.doc, self.m.id)[0]["edits"])
        beats = self.view().beats
        for i, b in enumerate(beats):
            b.t0, b.t1, b.by = 1000 + i * 400, 1400 + i * 400, "tap"
        self.ed.set_karaoke(self.m.id, beats)
        self.assertEqual(kara.tracks_of(self.doc, self.m.id)[0]["edits"], edits)
        self.assertTrue(self.view().complete)

    def test_edit_only_track_exports_placeholders(self):
        self.setup_line("ガー")
        self.doc.karaoke["k_song"] = {"id": "k_song", "member": self.m.id, "unit": "syllable", "beats": [],
                                      "edits": [{"op": "split", "at": self.key(1)}]}
        self.assertEqual(self.base(), "{\\kf100}ガ{\\kf100}ー")
        self.assertEqual(self.view().state, "placeholder")

    def test_merge_with_placeholder_keeps_result_unfinished(self):
        self.setup_line("メッキ", unit="mora", durations=[100, 200, 300])
        tr = kara.tracks_of(self.doc, self.m.id)[0]
        tr["beats"].pop(1)
        self.ed.merge_beats(self.m.id, self.key(1))
        self.assertEqual(self.times()[0], (1100, 1400, "placeholder"))
        self.assertFalse(self.view().complete)

    def test_translation_track_is_independent_of_original_edits(self):
        self.setup_line("ガー", durations=[400])
        cn = Member("m_cn", "g_song", "zh-Hant", "translation", "song", 1000, 3000, style="CN",
                    runs=[Run("你好", ins="i_cn", a=0, b=2)])
        self.doc.members[cn.id] = cn
        self.doc.groups[0].members.append(cn.id)
        beats = kara.derive(self.doc, cn)
        for b, a, z in zip(beats, [1000, 1400], [1400, 1600]):
            b.t0, b.t1, b.by = a, z, "tap"
        self.ed.set_karaoke(cn.id, beats)
        before = copy.deepcopy(kara.tracks_of(self.doc, cn.id))
        self.ed.split_beat(self.m.id, self.key(1))
        self.assertEqual(kara.tracks_of(self.doc, cn.id), before)
        cv = kara.effective(self.doc, cn)
        self.assertEqual(insert_at_visible("你好", kara.base_inserts(cv, visible_map(cn)[1], 100)),
                         "{\\kf40}你{\\kf20}好")

    def test_edits_follow_characters_moved_to_another_member(self):
        self.setup_line("ガーな", durations=[400, 300])
        self.ed.split_beat(self.m.id, self.key(1))
        moved = copy.deepcopy(self.m)
        moved.id = "m_moved"
        moved.runs = [self.m.runs[0].slice(0, 2)]
        self.m.runs = [self.m.runs[0].slice(2, 3)]
        self.doc.members[moved.id] = moved
        v = kara.effective(self.doc, moved)
        self.assertEqual([b.sung for b in v.beats], ["が", "ー"])
        self.assertTrue(v.complete)

    def test_invalid_actions_are_noops_and_dormant_tracks_are_ignored(self):
        self.setup_line("ガー", durations=[400])
        self.assertFalse(self.ed.split_beat(self.m.id, self.key(0)))
        self.assertFalse(self.ed.merge_beats(self.m.id, self.key(1)))
        self.assertEqual(len(self.ed.history.done), 1)
        self.ed.split_beat(self.m.id, self.key(1))
        tr = kara.tracks_of(self.doc, self.m.id)[0]
        tr["dormant"] = True
        self.assertFalse(kara.KIndex(self.doc).has(self.m))
        self.assertEqual([b.sung for b in self.view().beats], ["がー"])

    def test_release_export_and_cache_include_edited_base_and_issues(self):
        self.setup_line("ガーな", durations=[400, 300])
        template = AssDocument.parse(default_styles_ass())
        cache = ExportCache()
        def export():
            return build_ass(self.doc, {}, template, Path(os.environ["TEMP"]),
                             ExportOptions(mode="release", ruby=False), cache=cache)
        before, _ = export()
        self.assertEqual(before.events[0].text, "{\\kf10}{\\kf40}ガー{\\kf30}な")
        self.ed.split_beat(self.m.id, self.key(1), 1250)
        after, rep = export()
        self.assertEqual(after.events[0].text, "{\\kf10}{\\kf15}ガ{\\kf25}ー{\\kf30}な")
        self.assertEqual(rep.laid_out, 1)
        self.assertEqual(rep.karaoke_unfinished, [])
        self.ed.replace_text(self.m.id, "ガな")
        _, rep = export()
        self.assertIn(kara.EDIT_MISSING, rep.karaoke_unfinished[0])

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg/libass for real ruby layout")
    def test_actual_release_ruby_layout_carries_split_tags(self):
        self.setup_line("学", reading="がっ", durations=[400])
        self.ed.split_beat(self.m.id, self.key(0, 1), 1250)
        ass, rep = build_ass(self.doc, {}, AssDocument.parse(default_styles_ass()), Path(os.environ["TEMP"]),
                             ExportOptions(mode="release", ruby=True))
        self.assertEqual(ass.events[0].text, "{\\kf10}{\\kf40}学")
        self.assertTrue(any(e.text.endswith("{\\kf10}{\\kf15}が{\\kf25}っ") for e in ass.events[1:]))
        self.assertEqual(rep.karaoke_unfinished, [])


if __name__ == "__main__":
    unittest.main()
