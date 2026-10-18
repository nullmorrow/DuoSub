"""L3a read-only parsing, conservative timing recovery, and real legacy files."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.core import oped_import as oi
from subkit.reading.engine import ReadingEngine
from subkit.textutil import is_kanji

MATERIAL = Path('./sample-media/少女革命ウテナ')


def event(style, text, start="0:00:01.00", end="0:00:02.00", kind="Dialogue"):
    return f"{kind}: 0,{start},{end},{style},,0,0,0,,{text}"


def canonical(original="夢", kana="ゆめ", translation="夢想", mapping="夢想（ゆめ）"):
    return r"{\kf20}ゆ{\kf30}め{\kf0}\N" + (
        f"{{KaraOKMapping:{mapping}}}{{CNOrigin:{translation},JPOrigin:{original},KanaOrigin:{kana}}}")


class OpedImportTests(unittest.TestCase):
    def test_canonical_precedence_mapping_pauses_and_provenance(self):
        text = "\n".join([event("CNOPED", "old translation"), event("JPOPED", "old original"),
                          event("FuriganaOPED", "old reading"),
                          event("K轴", canonical(mapping="夢想(ゆめ)"), kind="Comment")])
        result = oi.parse_text(text, "finished.ass")
        self.assertEqual(len(result.lines), 1)
        line = result.lines[0]
        self.assertEqual((line.start_ms, line.end_ms, line.original, line.translation),
                         (1000, 2000, "夢", "夢想"))
        self.assertEqual((line.source_file, line.line_number, line.mapping),
                         ("finished.ass", 4, "夢想(ゆめ)"))
        self.assertEqual(line.readings, [(0, 1, "ゆめ")])
        self.assertEqual(line.beats, [("ゆ", 1000, 1200), ("め", 1200, 1500), ("", 1500, 1500)])
        self.assertEqual(result.unrecognised, [])

    def test_explicit_okurigana_scale_and_precedence(self):
        result = oi.parse_text("\n".join([
            event("songJP", "潔く 生きて夢"), event("songCN", "正文"),
            event("振假名", "潔く（いさぎよく,75） 生き(いき，90.5) 夢（ドリーム）", kind="Comment"),
            event("K轴测试", r"{\kf100}いさぎよくいきてゆめ")]))
        self.assertEqual(result.lines[0].readings, [(0, 1, "いさぎよ"), (3, 4, "い"), (6, 7, "ドリーム")])
        self.assertEqual(result.unrecognised, [])

    def test_english_hint_invisible_and_repeated_explicit_offsets(self):
        result = oi.parse_text("\n".join([
            event("songJP", "校庭{English:校庭,ガーデン} 手と手"),
            event("振假名", "手（て） 手（て）", kind="Comment"),
            event("K轴测试", r"{\kf50}ガーデン{\kf20}て{\kf10}と{\kf20}て")]))
        self.assertEqual(result.lines[0].original, "校庭 手と手")
        self.assertEqual(result.lines[0].readings, [(0, 2, "ガーデン"), (3, 4, "て"), (5, 6, "て")])
        self.assertEqual(result.unrecognised, [])

    def test_engine_consistent_split_peel_and_unsplittable_gap(self):
        class Engine:
            def analyze(self, text):
                return [SimpleNamespace(parts=[SimpleNamespace(start=0, end=2, reading="こうてい"),
                                               SimpleNamespace(start=3, end=4, reading="て")])]
        engine = Engine()
        self.assertEqual(oi.align_readings("校庭 手を", "こうていてを", engine=engine),
                         [(0, 2, "こうてい"), (3, 4, "て")])
        self.assertEqual(oi.align_readings("校庭 手を", "ガーデンてを", engine=engine),
                         [(0, 2, "ガーデン"), (3, 4, "て")])
        self.assertEqual(oi.align_readings("校庭 手を", "おもいでを", engine=engine), [(0, 4, "おもいで")])
        self.assertEqual(oi.align_readings("光差す校庭 手をとり合い", "ひかりさすガーデンてをとりあい"),
                         [(0, 1, "ひかり"), (1, 2, "さ"), (3, 5, "ガーデン"), (6, 7, "て"), (10, 11, "あ")])

    def test_ruby_shared_window_multikanji_internal_pause_and_okurigana(self):
        result = oi.parse_text("\n".join([
            event("JPOPED", r"{\kf40}二{\kf60}人{\kf0}"),
            event("FuriganaOPED", r"{\kf20}ふ{\kf10}{\kf30}た{\kf40}り"),
            event("JPOPED", r"{\kf100}合っ", "0:00:03.00", "0:00:04.00"),
            event("FuriganaOPED", r"{\kf100}あ", "0:00:03.00", "0:00:04.00")]))
        self.assertEqual(result.lines[0].readings, [(0, 2, "ふたり")])
        self.assertEqual(result.lines[0].beats, [("ふ", 1000, 1200), ("", 1200, 1300),
                                              ("た", 1300, 1600), ("り", 1600, 2000), ("", 2000, 2000)])
        self.assertEqual(result.lines[1].readings, [(0, 1, "あ")])
        self.assertEqual(result.lines[1].beats, [("あっ", 3000, 4000)])
        self.assertEqual(result.unrecognised, [])

    def test_timing_mismatch_is_reported_and_never_emits_kanji_as_kana(self):
        result = oi.parse_text("\n".join([event("JPOPED", r"{\kf100}夢"),
                                           event("FuriganaOPED", r"{\kf99}ゆめ")]))
        self.assertEqual(result.lines[0].beats, [])
        self.assertEqual([u.line_number for u in result.unrecognised], [1, 2])
        self.assertIn("缺少同演唱时间", result.unrecognised[0].reason)
        self.assertIn("不能唯一匹配", result.unrecognised[1].reason)

    def test_empty_tags_overrun_and_shared_visible_offsets(self):
        chunks = oi._timed(r"{\kf0}{\kf10}か\Nな\hよ\nね{\kf30}{\kf0}", 1000)
        self.assertEqual(chunks, [("", 1000, 1000, 0, 0), ("かな よ ね", 1000, 1100, 0, 6),
                                  ("", 1100, 1400, 6, 6), ("", 1400, 1400, 6, 6)])
        line = oi.parse_text(event("songCN", r"{\kf300}English", end="0:00:01.01")).lines[0]
        self.assertEqual(line.beats, [("English", 1000, 4000)])
        self.assertEqual(line.end_ms, 1010)

    def test_malformed_or_orphan_lines_have_source_reasons(self):
        text = "\n".join(["not an event", "Dialogue: 0,broken", event("unknown", "正文"),
                          event("songJP", "夢", start="bad"),
                          event("songJP", "夢", start="0:00:03.00"),
                          event("振假名", "夢（ゆめ）", start="0:00:05.00", end="0:00:06.00")])
        result = oi.parse_text(text, "broken.txt")
        self.assertEqual(result.lines, [])
        self.assertEqual([u.line_number for u in result.unrecognised], list(range(1, 7)))
        self.assertTrue(all(u.reason and u.source_file == "broken.txt" for u in result.unrecognised))
        for tag in (r"{\kf-1}か", r"{\kf1.5}か", r"{\kf12junk}か"):
            with self.subTest(tag=tag):
                r = oi.parse_text(event("songCN", tag))
                self.assertEqual(r.lines[0].beats, [])
                self.assertEqual(r.unrecognised[0].reason, "K 轴时长不是非负整数")

    def test_family_ambiguity_duplicates_and_helper_only_source_order(self):
        result = oi.parse_text("\n".join([
            event("K轴", canonical(), kind="Comment"),
            event("secondCN", "English", "0:00:03.00", "0:00:04.00")]))
        self.assertEqual([s.family for s in result.songs], ["OPED", "second"])
        result = oi.parse_text("\n".join([
            event("aJP", "夢"), event("bJP", "夢"), event("振假名", "夢（ゆめ）", kind="Comment")]))
        self.assertEqual(len(result.lines), 2)
        self.assertIn("多个歌词类别", result.unrecognised[0].reason)
        result = oi.parse_text("\n".join([event("aCN", "first"), event("aCN", "second")]))
        self.assertEqual(result.lines[0].translation, "first")
        self.assertIn("重复歌词行", result.unrecognised[0].reason)

    def test_grouping_shared_15_second_boundary_overlap_family_and_file(self):
        def line(a, b, family="OP", source="a"):
            return oi.SongLine(a, b, family, "", "text", [], [], None, source, 1)
        lines = [line(0, 1000), line(16000, 17000), line(16000, 18000),
                 line(33001, 34000), line(34000, 35000, "ED"),
                 line(35000, 36000, "OP"), line(36000, 37000, "OP", "b")]
        self.assertEqual([len(s.lines) for s in oi.group_songs(lines)], [3, 1, 1, 1, 1])

    def test_existing_ass_rounding_reordered_format_and_duplicate_ruby(self):
        text = "[Events]\nFormat: Start, End, Style, Text\n" + (
            "Dialogue: 0:00:01.005,0:00:02.015,songCN,English")
        line = oi.parse_text(text).lines[0]
        self.assertEqual((line.start_ms, line.end_ms), (1000, 2020))
        text = "\n".join([event("JPOPED", r"{\kf100}夢"),
                          event("FuriganaOPED", r"{\kf100}ゆめ"),
                          event("FuriganaOPED", r"{\kf100}ドリーム")])
        result = oi.parse_text(text)
        self.assertEqual(result.lines[0].readings, [(0, 1, "ゆめ")])
        self.assertEqual(result.lines[0].beats, [("ゆめ", 1000, 2000)])
        self.assertIn("重复或重叠", result.unrecognised[0].reason)


@unittest.skipUnless(MATERIAL.exists(), "real Utena legacy material unavailable")
class RealOpedImportTests(unittest.TestCase):
    def test_all_real_sources_text_readings_counts_mapping_timing_and_hashes(self):
        import numpy as np

        finished = next(MATERIAL.glob("S02E003*.zh-Hant.ass"))
        paths = [MATERIAL / "OPED.txt", *sorted((MATERIAL / "OPED-DB").glob("*.song")),
                 finished, *sorted((MATERIAL / "Done/Publish Candidate").glob("*.zh-Hant.ass"))]
        npzs = sorted((MATERIAL / "OPED-DB").glob("*.npz"))
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths + npzs}
        expected = [(50, 3), (17, 1), (22, 1), (34, 2), (34, 2), (56, 3), (34, 2),
                    (34, 2), (56, 3), (56, 3), (34, 2), (53, 3), (56, 3), (56, 3)]
        self.assertEqual(len(paths), len(expected))
        engine = ReadingEngine()
        results = [oi.parse(p, engine=engine) for p in paths]
        evidence = []
        for p, r, counts in zip(paths, results, expected):
            with self.subTest(file=p.name):
                self.assertEqual((len(r.lines), len(r.songs)), counts)
                self.assertEqual([l for s in r.songs for l in s.lines], r.lines)
                self.assertTrue(all(l.start_ms <= l.end_ms for l in r.lines))
                self.assertTrue(all(not any(is_kanji(c) for c in text)
                                    for l in r.lines if l.original for text, _, _ in l.beats))
                self.assertTrue(all(0 <= a < b <= len(l.original) and reading
                                    for l in r.lines for a, b, reading in l.readings))
                stats = dict(file=str(p), songs=len(r.songs), lines=len(r.lines),
                             beats=sum(len(l.beats) for l in r.lines),
                             pauses=sum(not b[0] for l in r.lines for b in l.beats),
                             mapped=sum(l.mapping is not None for l in r.lines),
                             reasons=dict(Counter(u.reason for u in r.unrecognised)), sha256=before[str(p)])
                evidence.append(stats)
                print("Utena L3a", json.dumps(stats, ensure_ascii=False))
        for r in results:
            first = r.lines[0]
            self.assertEqual(first.original, "潔く カッコよく 生きて行こう…" if "潔" in first.original
                             else "絶対運命黙示録")
            if "潔" in first.original:
                self.assertIn((0, 1, "いさぎよ"), first.readings)
                garden = next(l for l in r.lines if "校庭" in l.original)
                at = garden.original.index("校庭")
                self.assertIn((at, at + 2, "ガーデン"), garden.readings)
                pair = next(l for l in r.lines if "二人" in l.original)
                self.assertIn((3, 5, "ふたり"), pair.readings)
        txt = results[0]
        self.assertEqual(txt.lines[0].translation, "果敢地瀟灑地活下去吧")
        self.assertEqual(len(txt.lines[0].beats), 15)
        self.assertEqual(txt.lines[1].beats, [("Just", 12980, 13160), ("a", 13160, 13210),
                                            ("long", 13210, 13540), ("long", 13540, 13930),
                                            ("time", 13930, 14650), ("", 14650, 14830)])
        self.assertEqual(txt.unrecognised, [])
        self.assertEqual(len(results[1].lines[0].beats), 16)
        self.assertIn((3, 5, "ふたり"), results[1].lines[2].readings)
        self.assertEqual(results[1].lines[5].beats[4][0], "あっ")
        raw = finished.read_text(encoding="utf-8-sig").splitlines()
        canonical_lines = [l for l in results[3].lines if l.mapping is not None]
        self.assertGreater(len(canonical_lines), 20)
        for l in canonical_lines:
            source = raw[l.line_number - 1]
            self.assertIn(",K轴,", source)
            self.assertEqual(l.mapping, re.search(r"\{KaraOKMapping:(.*?)\}", source).group(1))
            origins = re.search(r"\{CNOrigin:(.*?),JPOrigin:(.*?),KanaOrigin:(.*?)\}", source)
            self.assertEqual((l.translation, l.original), origins.groups()[:2])
            expected_beats, at = [], l.start_ms
            for duration, text in re.findall(r"\{\\kf(\d+)\}([^{}]*)", source.split(r"\N")[0]):
                end = at + int(duration) * 10
                expected_beats.append((text.strip(), at, end))
                at = end
            self.assertEqual(l.beats, expected_beats)
        self.assertTrue(any("（" in l.mapping for l in canonical_lines))
        self.assertTrue(any("(" in l.mapping for l in canonical_lines))
        for p, r in zip(paths[1:3], results[1:3]):
            with np.load(p.with_suffix(".npz"), allow_pickle=False) as feature:
                self.assertEqual((r.songs[0].start_ms, r.songs[0].end_ms),
                                 (round(float(feature["start_sec"]) * 1000),
                                  round(float(feature["end_sec"]) * 1000)))
                print("NPZ absolute episode time", p.name, r.songs[0].start_ms, r.songs[0].end_ms)
        self.assertEqual((results[1].songs[0].start_ms, results[1].songs[0].end_ms), (9130, 88270))
        self.assertEqual((results[2].songs[0].start_ms, results[2].songs[0].end_ms), (834730, 927230))
        self.assertEqual(sum("不能唯一匹配" in u.reason for u in results[1].unrecognised), 2)
        self.assertEqual(sum("不能唯一匹配" in u.reason for u in results[2].unrecognised), 11)
        with tempfile.TemporaryDirectory(prefix="l3a-copy-") as scratch:
            copy = Path(scratch) / "OP.song"
            shutil.copyfile(paths[1], copy)
            copied = oi.parse(copy, engine=engine)
            self.assertEqual([(l.original, l.readings, l.beats) for l in copied.lines],
                             [(l.original, l.readings, l.beats) for l in results[1].lines])
        after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths + npzs}
        self.assertEqual(after, before)
        evidence_path = Path(os.environ.get("TEMP", tempfile.gettempdir())) / "l3a-real.json"
        evidence_path.write_text(json.dumps(dict(files=evidence, sources_sha256=after,
                                                unchanged=True), ensure_ascii=False, indent=2), encoding="utf-8")
        print("SHA256 unchanged:", len(after), "source files; evidence", evidence_path)


if __name__ == "__main__":
    unittest.main()
