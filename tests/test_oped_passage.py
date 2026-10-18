"""L3b blocks, passage conversion, optional audio and real read-only matching."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subkit.core import oped_import as oi, passages
from subkit.core.ops import Editor
from subkit.core.workspace import Project
from subkit.audio_match import EpisodeIndex, Match
from subkit.model.document import Document
from subkit.model.validate import validate

MATERIAL = Path('./sample-media/少女革命ウテナ')
BLOCK = """# 歌名：夢
# 类别：片头曲
时间：0:00:01.00 - 0:00:05.00
原文：夢を
读音：夢（ゆめ）
假名拍：{\\kf10}ゆ{\\kf20}{\\kf30}め{\\kf40}を
翻译：夢想吧
对应：夢想（ゆめ）吧（を）
备注：检查特殊读音
"""


class BlockPassageTests(unittest.TestCase):
    def test_fields_headers_mixed_events_notes_and_multiple_songs(self):
        old = ("Comment: 0,0:00:06.00,0:00:07.00,K轴,,0,0,0,,"
               r"{\kf50}ゆ{\kf50}め\N{KaraOKMapping:夢想（ゆめ）}"
               "{CNOrigin:夢想,JPOrigin:夢,KanaOrigin:ゆめ}")
        text = BLOCK + "\n" + old + "\n\n# 歌名:English\n# 类别:片尾曲\n" + (
            "时间:0:00:08.00 - 0:00:12.00\n翻译:{\\kf100}revo{\\kf200}lution\n备注:暂定\n")
        r = oi.parse_text(text, "pasted.txt")
        self.assertEqual(r.unrecognised, [])
        self.assertEqual([(s.family, s.category, len(s.lines)) for s in r.songs],
                         [("夢", "OP", 2), ("English", "ED", 1)])
        self.assertEqual(r.lines[0].notes, ["检查特殊读音"])
        self.assertEqual(r.lines[0].readings, [(0, 1, "ゆめ")])
        self.assertEqual(r.lines[-1].translation_beats, [("revo", 8000, 9000), ("lution", 9000, 11000)])
        p = oi.build_passage(r.songs[0], "夢", "OP", "TV")
        self.assertEqual((p["start"], p["duration"]), (1000, 6000))
        original, translation = p["groups"][0]["members"]
        self.assertEqual([(b["t0"], b["t1"], b["by"]) for b in original["karaoke"]["beats"]],
                         [(0, 100, "memory"), (300, 600, "memory"), (600, 1000, "memory")])
        self.assertTrue(all(b["link"]["start"]["member_index"] == 0 for b in translation["karaoke"]["beats"]))
        self.assertNotIn("检查特殊读音", json.dumps(p, ensure_ascii=False))
        english = oi.build_passage(r.songs[-1], "English", "ED", "TV")
        self.assertEqual([b["sung"] for b in english["groups"][0]["members"][0]["karaoke"]["beats"]],
                         ["revo", "lution"])

    def test_exact_errors_reject_only_bad_block(self):
        cases = [(BLOCK.replace("时间：0:00:01.00 - 0:00:05.00\n", ""), "第 1 块没有“时间”"),
                 (BLOCK.replace("0:00:01.00 - 0:00:05.00", "bad"), "第 1 块的时间写法不对：bad"),
                 (BLOCK.replace("夢（ゆめ）", "光（ひかり）"), "第 1 块“读音”里的「光」在原文里找不到"),
                 (BLOCK.replace("\\kf10}ゆ", "\\kf10}よ"), "第 1 块“假名拍”的假名和原文的读音对不上"),
                 (BLOCK.replace("吧（を）", "吧（を"), "括号不成对"),
                 (BLOCK.replace("夢想（ゆめ）", "夢想（よめ）"), "括号里的假名和原声拍对不上：第 1 拍应该是「ゆ」"),
                 (BLOCK.replace("夢想（ゆめ）", "夢（ゆめ）"), "翻译文字对不上：少了「想」")]
        for text, message in cases:
            with self.subTest(message=message):
                r = oi.parse_text(text + "\n时间:0:00:08.00 - 0:00:12.00\n翻译:hello")
                self.assertEqual(len(r.lines), 1)
                self.assertEqual(r.unrecognised[0].reason, message)
                self.assertEqual(r.unrecognised[0].block_number, 1)

    def test_store_conflicts_optional_audio_and_no_mutation(self):
        song = oi.parse_text(BLOCK).songs[0]
        p = oi.build_passage(song, "夢", "OP", "TV")
        before = copy.deepcopy(p)
        with tempfile.TemporaryDirectory(prefix="l3b-store-") as folder:
            store = passages.PassageStore(Project.create(Path(folder), "test"))
            first = oi.store_passage(store, p)
            self.assertIsNone(oi.store_passage(store, p))
            self.assertEqual(oi.store_passage(store, p, conflict="replace"), first)
            second = oi.store_passage(store, p, conflict="new")
            self.assertNotEqual(first, second)
            self.assertEqual(store.get(second)["version"], "TV (2)")
            self.assertEqual(len(store.list()), 2)
            with self.assertRaises(FileNotFoundError):
                store.template(first)
        self.assertEqual(p, before)
        self.assertIsNone(oi.build_audio_template(song))

    def test_audio_range_rounding_metadata_and_crop(self):
        from subkit.audio import AudioFeatures
        a = np.zeros((1000, 128), np.uint8)
        features = AudioFeatures(np.zeros(1000), np.zeros(1000), a, np.zeros(1000, np.float32))
        song = oi.parse_text("时间:0:00:02.001 - 0:00:06.001\n翻译:hello").songs[0]
        with patch.object(oi, "decode_audio_features", return_value=features):
            t = oi.build_audio_template(song, "episode.mkv")
            self.assertEqual(len(t.spectrogram), 400)
            with self.assertRaisesRegex(ValueError, "outside audio"):
                oi.build_audio_template(song, "episode.mkv", audio_origin_ms=3000)
        with tempfile.TemporaryDirectory(prefix="l3b-audio-") as folder:
            path = Path(folder) / "test.song"
            path.with_suffix(".wav").touch()
            np.savez(path.with_suffix(".npz"), start_sec=1.0, end_sec=11.0, unused=np.array([object()]))
            song.lines[0].source_file = str(path)
            with patch.object(oi, "decode_audio_features", return_value=features):
                self.assertEqual(len(oi.build_audio_template(song).spectrogram), 400)
            np.savez(path.with_suffix(".npz"), start_sec=1.0, end_sec=12.0)
            with patch.object(oi, "decode_audio_features", return_value=features), self.assertRaisesRegex(ValueError, "时长"):
                oi.build_audio_template(song)


@unittest.skipUnless(MATERIAL.exists(), "real Utena files unavailable")
class RealPassageTests(unittest.TestCase):
    def test_real_snapshots_readings_beats_links_and_source_hashes(self):
        paths = [MATERIAL / "OPED.txt", next(MATERIAL.glob("S02E003*.zh-Hant.ass"))]
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        captured = []
        real_snapshot = passages.snapshot
        def snapshot(doc, *args, **kwargs):
            self.assertTrue(validate(doc, {}).ok)
            self.assertTrue(all(r.ins and not r.src for m in doc.members.values() for r in m.runs))
            captured.append(doc)
            return real_snapshot(doc, *args, **kwargs)
        with patch.object(passages, "snapshot", side_effect=snapshot):
            for path in paths:
                song = oi.parse(path).songs[0]
                p = oi.build_passage(song, "輪舞-revolution", "OP", "TV")
                self.assertEqual(p["summary"]["groups"], len(song.lines))
                self.assertGreater(p["summary"]["readings"], 20)
                self.assertGreater(p["summary"]["karaoke"]["done"], 10)
                self.assertTrue(any(a["text"] == "ガーデン" for g in p["groups"] for m in g["members"] for a in m["annotations"]))
                self.assertTrue(all(b["by"] == "memory" for g in p["groups"] for m in g["members"]
                                    if m["lane"] == "original" for b in (m["karaoke"] or {}).get("beats", [])))
                if path.suffix == ".ass":
                    self.assertTrue(any(b.get("link") for g in p["groups"] for m in g["members"]
                                        for b in (m["karaoke"] or {}).get("beats", [])))
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

    def test_real_op_wav_found_in_e17_and_application_plan(self):
        begin = time.perf_counter()
        source = MATERIAL / "OPED-DB/OP.song"
        video = next(MATERIAL.glob("S02E004*.mkv"))
        paths = [source, source.with_suffix(".wav"), source.with_suffix(".npz"), video]
        def hashes():
            out = {}
            for path in paths:
                with path.open("rb") as fh:
                    out[str(path)] = hashlib.file_digest(fh, "sha256").hexdigest()
            return out
        before = hashes()
        song = oi.parse(source).songs[0]
        passage = oi.build_passage(song, "輪舞-revolution", "OP", "TV")
        template = oi.build_audio_template(song)
        features = oi.decode_audio_features(video)
        result = EpisodeIndex(features.spectrogram).find(template)
        target = next(MATERIAL.glob("S02E004*.zh-Hant.ass"))
        from subkit.ass import AssDocument
        from subkit.ass.text import plain_text
        expected = next(e.start * 10 for e in AssDocument.parse(target.read_text(encoding="utf-8-sig")).events
                        if e.style == "OPCH" and "".join(plain_text(e.text).split()) == "".join(song.lines[0].translation.split()))
        match = min(result.matches, key=lambda m: abs(m.start_ms - expected))
        print("OP match / subtitle anchor", match.start_ms, match.score, expected)
        # OP-DB audio has its own crop origin; fansub lyric times are not audio ground truth.
        self.assertLessEqual(abs(match.start_ms - expected), 1500)
        self.assertGreater(match.score, .9)
        with tempfile.TemporaryDirectory(prefix="l3b-real-project-") as folder:
            store = passages.PassageStore(Project.create(Path(folder), "Utena check"))
            id = oi.store_passage(store, passage, template)
            self.assertEqual(len(store.template(id).spectrogram), 7914)
            doc = Document("E17")
            plan = passages.plan(doc, store.get(id), match)
            actions = {a.id: a for l in plan.lines for a in l.actions}
            self.assertEqual(len(actions), len(song.lines))
            self.assertTrue(all(a.kind == "insert_group" for a in actions.values()))
            summary = Editor(doc, {}).apply_passage(plan, actions)
            self.assertEqual(summary.inserted_lines, passage["summary"]["lines"])
            self.assertTrue(validate(doc, {}).ok)
        after = hashes()
        self.assertEqual(before, after)
        evidence = dict(start_ms=match.start_ms, score=match.score, expected_ms=expected, covered=match.covered,
                        actions=len(actions), action_kinds=sorted({a.kind for a in actions.values()}),
                        source_start_ms=song.start_ms, source_end_ms=song.end_ms,
                        template_frames=len(template.spectrogram), seconds=time.perf_counter()-begin,
                        source_sha256=after, unchanged=True)
        path = Path(os.environ["TEMP"]) / "l3b-evidence/real-match.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        print("L3b real match", json.dumps(evidence, ensure_ascii=False), "evidence", path)


if __name__ == "__main__":
    unittest.main()
