"""Synthetic Qt-free request contracts; no model or media access."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from dataclasses import asdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subkit.core import karaoke
from subkit.core.ops import Editor
from subkit.model.document import Document, Group, Member, Run
from subkit.kalign.request import build_dialogue_request, build_song_request, line_signature
from subkit.kalign import worker


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.doc = Document("request")
        for i in range(3):
            gid, mid = f"g{i}", f"m{i}"
            m = Member(mid, gid, "ja", "original", "song", 1000+i*3000, 3000+i*3000,
                       style="ED.JP", runs=[Run("かな", ins=mid, a=0, b=2)])
            self.doc.members[mid] = m
            self.doc.groups.append(Group(gid, "song", [mid]))
        self.ed = Editor(self.doc, {})

    def build(self, **options):
        return build_song_request(self.doc, "g1", video="fake.mkv", audio_track=2, cache_dir="cache", options=options)

    def test_every_scope_aligns_whole_song_but_records_requested(self):
        for requested in (["g1"], ["g0", "g2"], ["g0", "g1", "g2"]):
            request, meta = self.build(requested_gids=requested)
            self.assertEqual([l.mid for l in request.lines], ["m0", "m1", "m2"])
            self.assertEqual(meta["requested_ids"], ["m"+g[1:] for g in requested])
            self.assertEqual(request.clip, (0, 11000))

    def test_protected_fixed_hand_calibration_and_overwrite(self):
        for mid, by, state in (("m0", "tap", "timed"), ("m1", "manual", "timed"), ("m2", "align", "confirmed")):
            beats = karaoke.derive(self.doc, self.doc.members[mid])
            for i,b in enumerate(beats): b.t0,b.t1,b.by = 1200+i*300,1500+i*300,by
            self.ed.set_karaoke(mid, beats, state=state)
        request, _ = self.build(stored_habit=12)
        self.assertTrue(all(b.fixed is not None for l in request.lines for b in l.beats))
        self.assertEqual({b["by"] for b in request.options["hand_beats"]}, {"tap"})
        self.assertEqual(request.options["stored_habit"], 12)
        request, _ = self.build(overwrite=True)
        self.assertTrue(all(b.fixed is None for l in request.lines for b in l.beats))

    def test_redo_of_one_line_is_free_and_placeholders_are_never_anchors(self):
        # m0, m1 hand-tuned; each has one placeholder beat (e.g. a kanji whose reading changed).
        for mid in ("m0", "m1"):
            beats = karaoke.derive(self.doc, self.doc.members[mid])
            for i, b in enumerate(beats):
                b.t0, b.t1, b.by = 1200 + i * 300, 1500 + i * 300, ("drag" if i == 0 else "placeholder")
            self.ed.set_karaoke(mid, beats, state="timed")
        request, _ = self.build(requested_gids=["g1"], free_requested=True)
        lines = {l.mid: l for l in request.lines}
        self.assertTrue(all(b.fixed is None for b in lines["m1"].beats))           # the line being redone
        self.assertIsNotNone(lines["m0"].beats[0].fixed)                           # hand-tuned context beat
        self.assertIsNone(lines["m0"].beats[1].fixed)                              # placeholder: no real time
        self.assertNotIn("free_requested", request.options)
        request, _ = self.build(requested_gids=["g1"])                              # 选中的句 / 整首歌 semantics
        self.assertIsNotNone({l.mid: l for l in request.lines}["m1"].beats[0].fixed)

    def test_signature_detects_text_reading_and_split_edits(self):
        before = line_signature(self.doc, "m1")
        self.doc.members["m1"].runs[0].text = "かに"
        self.assertNotEqual(before, line_signature(self.doc, "m1"))
        self.assertIsNone(line_signature(self.doc, "missing"))

    def test_non_song_has_no_request(self):
        self.doc.groups[1].kind = "dialogue"
        with self.assertRaises(ValueError): self.build()


class DialogueRequestTests(unittest.TestCase):
    def setUp(self):
        self.doc = Document("dialogue")

    def add(self, gid, mid, start, *, lane="original", lang="ja", text="かな", role="dialogue"):
        member = Member(mid, gid, lang, lane, role, start, start + 1000,
                        runs=[Run(text, ins=mid, a=0, b=len(text))])
        self.doc.members[mid] = member
        group = next((g for g in self.doc.groups if g.id == gid), None)
        if group is None:
            group = Group(gid)
            self.doc.groups.append(group)
        group.members.append(mid)
        return group, member

    def build(self):
        return build_dialogue_request(self.doc, video="fake.mkv", audio_track=2, cache_dir="cache")

    def test_group_times_order_multiple_members_and_metadata_roundtrip(self):
        self.add("later", "late", 5000)
        self.add("early", "jp1", 2000)
        self.add("early", "jp2", 2200)
        self.add("early", "cn", 1000, lane="translation", lang="zh")
        with patch("subkit.kalign.tokens.particle_positions_many", return_value=[set()]):
            request, meta = self.build()
        self.assertEqual([line.mid for line in request.lines], ["jp1", "jp2", "late"])
        self.assertEqual([(l.start, l.end) for l in request.lines], [(1000, 3200), (1000, 3200), (5000, 6000)])
        self.assertEqual(meta["members"], {"later": ["late"], "early": ["jp1", "jp2"]})
        self.assertEqual(meta["signatures"], {l.mid: l.signature for l in request.lines})
        self.assertEqual((request.kind, request.clip), ("dialogue", None))
        self.assertEqual(worker.request_from_json(asdict(request)), request)
        self.assertEqual(self.doc.revision, 0)

    def test_hidden_non_dialogue_non_japanese_comment_empty_rows_skipped(self):
        group, _ = self.add("hidden", "hide", 0)
        group.flags.append("hidden")
        group, _ = self.add("song", "song", 1000)
        group.kind = "song"
        self.add("foreign", "en", 2000, lang="en")
        self.add("empty", "empty", 3000, text=" ")
        self.add("comment", "note", 4000, role="comment")
        _, member = self.add("sourcecomment", "sourcecomment", 5000)
        member.source_kind = "Comment"
        self.add("translation", "cn", 6000, lane="translation", lang="zh")
        with patch("subkit.kalign.dialogue.spoken_kana") as reading:
            with self.assertRaises(ValueError):
                self.build()
        reading.assert_not_called()

    def test_readings_signature_and_manual_edges_are_preserved(self):
        _, member = self.add("g", "jp", 1000)
        member.extra["time_src"] = {"start": "manual", "end": "adopted"}
        with patch("subkit.kalign.dialogue.spoken_kana", return_value=("よみ", "signature")) as reading:
            request, meta = self.build()
        reading.assert_called_once_with(self.doc, member)
        self.assertEqual((request.lines[0].kana, meta["signatures"]["jp"]), ("よみ", "signature"))
        self.assertEqual(member.extra["time_src"], {"start": "manual", "end": "adopted"})
