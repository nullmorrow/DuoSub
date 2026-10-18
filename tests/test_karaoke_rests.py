"""KD rests: existing timing storage, undo, string export and linked lyric edges."""
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subkit.ass.text import insert_at_visible
from subkit.core import karaoke as kara, lyrics_map as lm, oped_import as oi
from subkit.core.ops import Editor
from subkit.core.readings import visible_map
from subkit.model.document import Document, Group, Member, Run
from subkit.model.serialize import document_from_json, document_to_json


class RestTests(unittest.TestCase):
    def setUp(self):
        self.doc = Document("rests", groups=[Group("g", "song", ["jp", "cn"])])
        for mid, text, lane in (("jp", "かなよ", "original"), ("cn", "你好呀", "translation")):
            self.doc.members[mid] = Member(mid, "g", "ja" if mid == "jp" else "zh-Hant", lane,
                "song", 1000, 3000, runs=[Run(text, ins=mid, a=0, b=len(text))])
        self.ed = Editor(self.doc, {})
        self.m = self.doc.members["jp"]
        beats = kara.derive(self.doc, self.m)
        for i, b in enumerate(beats):
            b.t0, b.t1, b.by = 1000 + i * 400, 1400 + i * 400, "tap"
        self.ed.set_karaoke("jp", beats)

    def view(self):
        return kara.effective(self.doc, self.doc.members["jp"])

    def test_add_remove_one_undo_and_save_gap(self):
        before, count = copy.deepcopy(self.doc.karaoke), len(self.ed.history.done)
        keys = [b.key for b in self.view().beats]
        self.assertTrue(self.ed.add_rest("jp", keys[0]))
        self.assertEqual(kara.rests_of(self.view().beats), [kara.Rest(0, 1200, 1400)])
        self.assertEqual(len(self.ed.history.done), count + 1)
        reopened = document_from_json(json.loads(json.dumps(document_to_json(self.doc))))
        self.assertEqual(kara.rests_of(kara.effective(reopened, reopened.members["jp"]).beats),
                         [kara.Rest(0, 1200, 1400)])
        added = copy.deepcopy(self.doc.karaoke)
        self.ed.history.undo()
        self.assertEqual(self.doc.karaoke, before)
        self.ed.history.redo()
        self.assertEqual(self.doc.karaoke, added)
        self.assertTrue(self.ed.remove_rest("jp", keys[0], keys[1]))
        self.assertEqual(kara.rests_of(self.view().beats), [])
        self.assertEqual(len(self.ed.history.done), count + 2)
        self.ed.history.undo()
        self.assertEqual(self.doc.karaoke, added)

    def test_short_beat_third_and_no_successor_or_stale_selection(self):
        beats = self.view().beats
        beats[0].t1 = 1300
        self.ed.set_karaoke("jp", beats)
        self.assertTrue(self.ed.add_rest("jp", beats[0].key))
        self.assertEqual(self.view().beats[0].t1, 1200)
        count = len(self.ed.history.done)
        self.assertFalse(self.ed.add_rest("jp", beats[-1].key))
        self.assertFalse(self.ed.remove_rest("jp", beats[1].key, beats[2].key))
        self.assertEqual(len(self.ed.history.done), count)

    def test_independent_edges_clamp_and_shared_edge_with_undo(self):
        self.ed.add_rest("jp", self.view().beats[0].key)
        original = copy.deepcopy(self.doc.karaoke)
        beats = self.view().beats
        moved = kara.move_edge(beats, 0, "end", 1250)
        self.assertEqual((moved[0].t1, moved[1].t0), (1250, 1400))
        self.ed.set_karaoke("jp", moved)
        self.ed.history.undo()
        self.assertEqual(self.doc.karaoke, original)
        moved = kara.move_edge(beats, 1, "start", 1450, shift=True)
        self.assertEqual((moved[0].t1, moved[1].t0, moved[1].t1, moved[2].t0),
                         (1200, 1450, 1800, 1800))
        self.assertEqual(kara.move_edge(beats, 1, "start", 1100)[1].t0, 1200)
        self.assertEqual(kara.move_edge(beats, 0, "end", 1700)[0].t1, 1400)
        self.ed.remove_rest("jp", beats[0].key, beats[1].key)
        shared = kara.move_edge(self.view().beats, 1, "start", 1500)
        self.assertEqual((shared[0].t1, shared[1].t0), (1500, 1500))
        shared_end = kara.move_edge(self.view().beats, 0, "end", 1500)
        self.assertEqual([(b.t0, b.t1) for b in shared_end], [(b.t0, b.t1) for b in shared])

    def test_string_export_no_rest_bytes_and_empty_wait_rounding(self):
        def export():
            return insert_at_visible(self.m.text, kara.base_inserts(self.view(), visible_map(self.m)[1], 100))
        expected = "{\\kf40}か{\\kf40}な{\\kf40}よ".encode("utf-8")
        self.assertEqual(export().encode("utf-8"), expected)
        self.ed.add_rest("jp", self.view().beats[0].key)
        self.assertEqual(export(), "{\\kf20}か{\\kf20}{\\kf40}な{\\kf40}よ")
        beats = self.view().beats
        beats[0].t1 = 1205
        self.ed.set_karaoke("jp", beats)
        self.assertEqual(export(), "{\\kf21}か{\\kf19}{\\kf40}な{\\kf40}よ")
        self.ed.remove_rest("jp", beats[0].key, beats[1].key)
        self.assertEqual(export().encode("utf-8"), expected)
        self.assertTrue(self.view().complete)

    def test_mapping_end_follows_previous_end_and_start_follows_next_start(self):
        self.ed.apply_lyrics_mapping({"g": "你（か）好（な）呀（よ）"})
        self.ed.add_rest("jp", self.view().beats[0].key)
        translated = kara.effective(self.doc, self.doc.members["cn"]).beats
        self.assertEqual((translated[0].t1, translated[1].t0), (1200, 1400))
        self.assertEqual(translated[0].link["end"]["edge"], "end")
        self.ed.set_karaoke("jp", kara.move_edge(self.view().beats, 1, "start", 1500))
        translated = kara.effective(self.doc, self.doc.members["cn"]).beats
        self.assertEqual((translated[0].t1, translated[1].t0), (1200, 1500))
        self.assertEqual(lm.current_mapping(self.doc, self.m, self.doc.members["cn"]), "你（か）好（な）呀（よ）")
        self.assertNotIn("jp", [mid for mid, _issues in kara.unfinished(self.doc)])

    def test_ruby_internal_rest_exports_empty_wait_without_changing_no_rest_bytes(self):
        beats = [kara.Beat(0, 1, 0, k=i, n=2, ann="ruby", piece=0, r=(i, i + 1),
                          t0=1000 + i * 400, t1=1400 + i * 400, by="tap") for i in range(2)]
        view = kara.KView("jp", beats, "timed", "syllable", "kf")
        self.assertEqual(kara.ruby_text(view, "ruby", 0, "かな", 100).encode("utf-8"),
                         "{\\kf40}か{\\kf40}な".encode("utf-8"))
        beats[0].t1 = 1200
        self.assertEqual(kara.ruby_text(view, "ruby", 0, "かな", 100),
                         "{\\kf20}か{\\kf20}{\\kf40}な")

    def test_legacy_pause_import_is_rest(self):
        text = (r"Comment: 0,0:00:01.00,0:00:03.00,K轴,,0,0,0,,{\kf20}か{\kf20}{\kf40}な"
                r"\N{KaraOKMapping:你（か）好（な）}{CNOrigin:你好,JPOrigin:かな,KanaOrigin:かな}")
        parsed = oi.parse_text(text)
        self.assertEqual(parsed.unrecognised, [])
        self.assertEqual(parsed.lines[0].beats, [("か", 1000, 1200), ("", 1200, 1400), ("な", 1400, 1800)])
        from subkit.core import passages

        with patch.object(passages, "snapshot", wraps=passages.snapshot) as snapshot:
            oi.build_passage(parsed.songs[0], "test", "OP", "TV")
        doc = snapshot.call_args.args[0]
        original = next(m for m in doc.members.values() if m.lane == "original")
        self.assertEqual(kara.rests_of(kara.effective(doc, original).beats), [kara.Rest(0, 1200, 1400)])
