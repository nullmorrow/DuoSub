"""S3a whole-note redistribution and bounded automatic timing, without models/media."""
import copy
import dataclasses
import hashlib
import json
import unittest

from tests.test_lyrics_auto import KNOW, LOVE, TOMORROW, MORNING, fixture
from tests.test_lyrics_map import document
from subkit.core import karaoke as kara, lyrics_auto as la, lyrics_map as lm
from subkit.model.document import Group, Member, Run
from subkit.model.serialize import document_to_json, document_from_json


class WholeNotes(unittest.TestCase):
    def prepared(self, translation="向前", durations=(300, 300, 300)):
        return document("てこう", translation, sung=list(zip("てこう", durations)))

    def test_whole_note_default_callers_and_proposal(self):
        doc, ed = self.prepared()
        originals = kara.effective(doc, doc.members["jp"]).beats
        self.assertEqual(lm.auto_mapping(originals, "向前"), "向（て）前（こう）")
        ed.redistribute_translation(["g"])
        beats = kara.effective(doc, doc.members["cn"]).beats
        self.assertEqual([(b.t0, b.t1) for b in beats], [(1000, 1300), (1300, 1900)])
        self.assertEqual(beats[-1].link, {"start": kara.beat_link(originals[1], "start"),
                                         "end": kara.beat_link(originals[2], "end")})
        self.assertTrue(all("auto_split" not in b.flags for b in beats))

    def test_equal_internal_boundary_is_linked_and_roundtrips(self):
        doc, ed = document("こう", "向前", sung=[("こ", 300), ("う", 300)])
        ed.redistribute_translation(["g"])
        beats = kara.effective(doc, doc.members["cn"]).beats
        self.assertEqual([(b.t0, b.t1) for b in beats], [(1000, 1300), (1300, 1600)])
        self.assertTrue(all(len(b.link) == 2 and "auto_split" not in b.flags for b in beats))
        loaded = document_from_json(json.loads(json.dumps(document_to_json(doc))))
        self.assertEqual(loaded.karaoke, doc.karaoke)

    def test_fractional_note_cut_maps_by_time_and_follows(self):
        doc, ed = document("こう", "向前", sung=[("こ", 200), ("う", 400)])
        ed.redistribute_translation(["g"])
        beats = kara.effective(doc, doc.members["cn"]).beats
        self.assertEqual([(b.t0, b.t1) for b in beats], [(1000, 1300), (1300, 1600)])
        self.assertNotIn("end", beats[0].link)
        self.assertNotIn("start", beats[1].link)
        self.assertTrue(all("auto_split" in b.flags for b in beats))
        originals = kara.effective(doc, doc.members["jp"]).beats
        originals[-1].t1 = 1800
        ed.set_karaoke("jp", originals)
        self.assertEqual([(b.t0, b.t1) for b in kara.effective(doc, doc.members["cn"]).beats],
                         [(1000, 1400), (1400, 1800)])

    def test_chained_continuations_and_zero_duration(self):
        for durations in ((200, 300, 400), (0, 0, 0)):
            doc, ed = document("こうー", "前", sung=list(zip("こうー", durations)))
            ed.redistribute_translation(["g"])
            beats = kara.effective(doc, doc.members["cn"]).beats
            self.assertEqual(len(beats), 1)
            self.assertEqual((beats[0].t0, beats[0].t1), (1000, 1000 + sum(durations)))

    def test_excess_translation_units_preserve_all_text(self):
        doc, ed = document("こう", "一二三四五", sung=[("こ", 200), ("う", 400)])
        ed.redistribute_translation(["g"])
        beats = kara.effective(doc, doc.members["cn"]).beats
        self.assertLessEqual(len(beats), 3)
        self.assertEqual("".join(b.sung for b in beats), "一二三四五")
        self.assertEqual([(b.t0, b.t1) for b in beats], [(1000, 1200), (1200, 1400), (1400, 1600)])

    def test_phrase_excess_does_not_split_when_whole_line_has_enough_notes(self):
        beats = [kara.Beat(i, i + 1, i, sung=s, t0=1000 + i * 300, t1=1300 + i * 300, base_text=s)
                 for i, s in enumerate(["こ", "う ", "か", "き", "く", "け"])]
        segments = la.redistribute(beats, "你好啊 世", follow_notes=True)
        self.assertEqual("".join(s.text for s in segments), "你好啊 世")
        self.assertTrue(all(s.start.fraction == 0 and s.end.fraction == 1 for s in segments))
        self.assertEqual(segments[0].text, "你好啊 ")

    def test_off_matches_original_serialized_outputs(self):
        # SHA-256 of the exact pre-S3a Segment JSON, both fractional and whole proposals.
        expected = (
            "1c850c58d6ede968ce3bd09af994c9f48f49d30800be0c4ea6221ca1c23b0537",
            "462b82f99ce77181b90323e86c7da44f4006ccf66ed5bec097dc41ef581cfcd5",
            "d8de422c6bf2e1cb9874c8b431dfa0048c44d716f72a684138c632fabd10ba21",
            "8b166fbc31b92d2ad8d7e0890de4ef0af5fac3e9dd0609a5d18b3b3ad4457bc0",
            "b24cbf6ab30f6f8349e2b737d0479e8d24df037e95791b459f64773e2a3cb72a",
            "35a403a165c5bb13a02f587d3cbd459ac4535354ec15ba683c404a1463f1cab1",
            "ac5e4e114a2ac01b8d0f555ddcd1971d5efa032862cd81cecf4c5a2c50707959",
            "ac5e4e114a2ac01b8d0f555ddcd1971d5efa032862cd81cecf4c5a2c50707959")
        actual = []
        for values in (KNOW, LOVE, TOMORROW, MORNING):
            beats, _ = fixture(values)
            for splits in (True, False):
                segments = la.redistribute(beats, values[1], original_text=values[0],
                                           splits=splits, follow_notes=False)
                payload = json.dumps([dataclasses.asdict(s) for s in segments], ensure_ascii=False, sort_keys=True)
                actual.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
        self.assertEqual(tuple(actual), expected)

    def test_apply_and_adopt_thread_option(self):
        for follow in (True, False):
            for adopt in (True, False):
                doc, ed = self.prepared()
                originals = kara.effective(doc, doc.members["jp"]).beats
                for b in originals:
                    b.t0 += 10; b.t1 += 10
                if adopt:
                    ed.adopt_auto_karaoke("jp", originals, "syllable", True, follow_notes=follow)
                else:
                    ed.apply_auto_karaoke([{"mid": "jp", "beats": [(b.t0, b.t1, 1, False) for b in originals]}],
                                         translations=True, overwrite=True, follow_notes=follow)
                expected = la.redistribute(originals, "向前", follow_notes=follow)
                mapped, _, _ = la.mapped_beats(doc, doc.members["cn"], originals, expected)
                self.assertEqual([(b.t0, b.t1) for b in kara.effective(doc, doc.members["cn"]).beats],
                                 [(b.t0, b.t1) for b in mapped])


class BoundedApply(unittest.TestCase):
    def prepared(self):
        doc, ed = document("かな", "你好", sung=[("か", 400), ("な", 600)])
        for m in doc.members.values():
            m.start, m.end = 1000, 2000
        return doc, ed

    def neighbour(self, doc, gid, start, end, kind="dialogue", hidden=False):
        mid = gid + "jp"
        doc.groups.append(Group(gid, kind, [mid], flags=["hidden"] if hidden else []))
        doc.members[mid] = Member(mid, gid, "ja", "original", kind, start, end,
                                  runs=[Run("かな", ins="i_" + mid, a=0, b=2)])
        doc.structure_changed()
        return mid

    def apply(self, ed, values, mid="jp", translations=False):
        return ed.apply_auto_karaoke([{"mid": mid, "beats": [(a, z, 1, False) for a, z in values]}],
                                    translations=translations, overwrite=True)

    def test_both_edges_dialogue_and_song_sorted_by_time_undo(self):
        doc, ed = self.prepared()
        self.neighbour(doc, "later", 2050, 3000, "song")
        self.neighbour(doc, "earlier", 100, 950)
        before = copy.deepcopy(document_to_json(doc)); before.pop("revision")
        report = self.apply(ed, [(800, 970), (1980, 2200)], translations=True)
        originals = kara.effective(doc, doc.members["jp"]).beats
        # Squeezed against both neighbours: each gets 100 ms out of the gap between them (the gap stays a gap).
        self.assertEqual([(b.t0, b.t1) for b in originals], [(950, 1050), (1950, 2050)])
        self.assertTrue(all("check" in b.flags for b in originals))
        self.assertEqual(report["clipped"], 2)
        self.assertEqual(report["lines_with_check"], 1)
        self.assertEqual(doc.group_time(doc.group("g")), (950, 2050))
        translated = kara.effective(doc, doc.members["cn"]).beats
        for b in translated:
            self.assertTrue(950 <= b.t0 <= b.t1 <= 2050)
        after = copy.deepcopy(document_to_json(doc)); after.pop("revision")
        ed.history.undo()
        undone = document_to_json(doc); undone.pop("revision")
        self.assertEqual(undone, before)
        ed.history.redo()
        redone = document_to_json(doc); redone.pop("revision")
        self.assertEqual(redone, after)
        stored = copy.deepcopy(doc.karaoke)
        ed.history.run("follow", lambda tx: ed._follow_karaoke_links(tx, ["g"]))
        self.assertEqual(doc.karaoke, stored)

    def test_simultaneous_widening_uses_shared_boundary(self):
        doc, ed = self.prepared()
        mid = self.neighbour(doc, "next", 2200, 3200, "song")
        report = ed.apply_auto_karaoke([
            {"mid": "jp", "beats": [(900, 1400, 1, False), (1400, 2600, 1, False)]},
            {"mid": mid, "beats": [(1800, 2500, 1, False), (2500, 3400, 1, False)]}],
            translations=False, overwrite=True)
        self.assertEqual(doc.group_time(doc.group("g")), (900, 2000))
        self.assertEqual(doc.group_time(doc.group("next")), (2000, 3400))
        self.assertEqual(report["clipped"], 2)

    def test_existing_overlap_stays_and_hidden_signs_do_not_bound(self):
        for start, end, kind, hidden in ((1900, 3000, "song", False),
                                         (2050, 3000, "song", True),
                                         (2050, 3000, "sign", False)):
            doc, ed = self.prepared()
            self.neighbour(doc, "next", start, end, kind, hidden)
            report = self.apply(ed, [(900, 1400), (1400, 2300)])
            self.assertEqual(doc.group_time(doc.group("g")), (900, 2300))
            self.assertEqual(report["clipped"], 0)

    def test_clipping_exact_100ms_and_collapsed_beat(self):
        doc, ed = self.prepared()
        self.neighbour(doc, "next", 2100, 3000)
        report = self.apply(ed, [(2000, 2300), (2400, 2600)])
        beats = kara.effective(doc, doc.members["jp"]).beats
        # A beat squeezed onto the bound shares the span with its neighbour instead of vanishing (0 ms).
        self.assertEqual([(b.t0, b.t1) for b in beats], [(2000, 2050), (2050, 2100)])
        self.assertIn("check", beats[0].flags)
        self.assertIn("check", beats[1].flags)
        self.assertEqual(report["clipped"], 1)

    def test_squeezed_first_beat_gets_room_after_previous_line(self):
        doc, ed = self.prepared()
        self.neighbour(doc, "prev", 0, 1000)
        report = self.apply(ed, [(600, 900), (1300, 1600)])        # first beat inside the previous line
        beats = kara.effective(doc, doc.members["jp"]).beats
        self.assertEqual([(b.t0, b.t1) for b in beats], [(1000, 1100), (1300, 1600)])
        self.assertIn("check", beats[0].flags)
        self.assertNotIn("check", beats[1].flags)
        self.assertEqual(report["clipped"], 1)

    def test_independently_clipped_translation_link_removed(self):
        doc, ed = self.prepared()
        self.neighbour(doc, "next", 2100, 3000)
        # A second, unselected original carries a wider beat; the group itself did not overlap.
        extra = copy.deepcopy(doc.members["jp"])
        extra.id = "other"; extra.runs[0].ins = "i_other"
        doc.members[extra.id] = extra; doc.groups[0].members.append(extra.id); doc.structure_changed()
        orig = kara.effective(doc, extra).beats
        orig[-1].t1 = 2500; orig[-1].by = "tap"
        ed.set_karaoke(extra.id, orig)
        cn = kara.effective(doc, doc.members["cn"]).beats
        for b in cn: b.by = "tap"
        cn[-1].t1 = 2500; cn[-1].link["end"] = kara.beat_link(orig[-1], "end")
        ed.set_karaoke("cn", cn)
        report = ed.apply_auto_karaoke([{"mid": "jp", "beats": [(1000, 1500, 1, False), (1500, 2000, 1, False)]}],
                                      translations=False, overwrite=True, requested_ids={"jp"})
        last = kara.effective(doc, doc.members["cn"]).beats[-1]
        self.assertEqual(last.t1, 2100)
        self.assertNotIn("end", last.link)
        self.assertEqual(report["clipped"], 1)
