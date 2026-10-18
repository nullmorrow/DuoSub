"""Native-reference discovery uses immutable events, not edited subtitle times."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opencc import OpenCC
from subkit.core.reference_sources import ReferenceEvent, resolve_reference
from subkit.model.source import Source, SourceEvent


_TEXT = ["這是第一段測試", "學校的大門開了", "請先回到這裡", "我們一起走吧", "時鐘已經響起", "最後一個問題"]


def _stamp(ms):
    cs = ms // 10
    return f"{cs // 360000}:{cs // 6000 % 60:02}:{cs // 100 % 60:02}.{cs % 100:02}"


def _events(texts=None):
    return [ReferenceEvent(i, 1000 + i * 3000, 3000 + i * 3000, text)
            for i, text in enumerate(texts or _TEXT)]


class ReferenceSourcesTests(unittest.TestCase):
    def test_normalizer_cache_does_not_retain_finished_episode(self):
        import gc
        import weakref
        from subkit.core.reference_sources import _Normalizer
        normalizer = _Normalizer([])
        normalizer.key("不保留已完成的集")
        reference = weakref.ref(normalizer)
        del normalizer
        gc.collect()
        self.assertIsNone(reference())

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="duosub_reference_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.to_s = OpenCC("t2s").convert

    def ass(self, name, events=None, *, encoding="utf-8-sig", header="", wrap=0):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = ["[Script Info]", header, f"WrapStyle: {wrap}", "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
        rows.extend(f"{e.kind}: 0,{_stamp(e.start)},{_stamp(e.end)},Default,,0,0,0,,{e.text}"
                    for e in events or _events())
        path.write_bytes(("\r\n".join(rows) + "\r\n").encode(encoding))
        return path

    def source(self, texts=None, *, name="Show.S01E001.cht.ass", events=None, encoding="utf-8-sig"):
        evs = events or _events(texts)
        path = self.ass(name, evs, encoding=encoding)
        raw = path.read_bytes()
        source = Source("t1", "zh-Hant", "translation", "ass", str(path), "sources/t1.ass",
                        hashlib.sha256(raw).hexdigest(), len(raw), encoding, "crlf")
        source.events = [SourceEvent(11 + e.i * 7, e.kind, e.start, e.end, e.text) for e in evs]
        return source

    def simplified(self, events=None):
        return [ReferenceEvent(e.i, e.start, e.end, self.to_s(e.text), e.kind) for e in events or _events()]

    def target(self, events=None, *, name="Show.S01E01.chs.ass", **kw):
        return self.ass(name, self.simplified() if events is None else events, **kw)

    def selected(self, source=None, reference=None, target="zh-Hans"):
        return resolve_reference(source or self.source(), target, reference, episode_id="S01E001")

    def test_auto_discovers_same_episode_and_normalizes_script_alias(self):
        source = self.source()
        path = self.target()
        result = self.selected(source, target="simplified")
        self.assertEqual(result.target_script, "zh-Hans")
        self.assertEqual(result.target.path, str(path))
        self.assertEqual(result.target.script_hint, "zh-Hans")
        self.assertEqual(result.target_matches, {11 + i * 7: i for i in range(6)})
        self.assertIsNone(result.baseline)

    def test_reference_event_values_are_actual_file_ids_including_comments(self):
        source = self.source()
        evs = self.simplified()
        evs.insert(0, ReferenceEvent(100, 0, 100, "製作備註", "Comment"))
        result = self.selected(source, self.target(evs))
        self.assertEqual(result.target_matches[11], 1)
        self.assertEqual(result.target.events[0].kind, "Comment")
        self.assertEqual(result.target.events[0].text, "製作備註")

    def test_parser_keeps_newlines_strips_ass_tags_notes_and_drawings(self):
        texts = [r"{\b1}這是{@note=私有}第一段\N測試{\p1}m 0 0 l 1 1{\p0}", *_TEXT[1:]]
        source = self.source(texts)
        path = self.target(self.simplified(_events(texts)))
        before = path.read_bytes()
        result = self.selected(source, path)
        self.assertEqual(result.target.events[0].text, "这是第一段\n测试")
        self.assertEqual(result.target.sha256, hashlib.sha256(before).hexdigest())
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(result.target.encoding, "utf-8")

    def test_srt_keeps_millisecond_precision_and_visible_lines(self):
        source = self.source()
        path = self.root / "Show.S01E01.chs.srt"
        path.write_text("1\n00:00:01,123 --> 00:00:03,456\n<i>这是第一段测试</i>\n第二行<br>第三行\n\n" +
                        "\n\n".join(f"{i + 1}\n00:00:{1 + i * 3:02},000 --> 00:00:{3 + i * 3:02},000\n{self.to_s(t)}"
                                      for i, t in enumerate(_TEXT) if i), encoding="utf-8")
        result = self.selected(source, path)
        event = result.target.events[0]
        self.assertEqual((event.start, event.end), (1123, 3456))
        self.assertEqual(event.text, "这是第一段测试\n第二行\n第三行")
        self.assertNotIn(11, result.target_matches)
        self.assertEqual(len(result.target_matches), 5)

    def test_utf16_reference_preserves_encoding_and_bytes(self):
        source = self.source()
        path = self.target(encoding="utf-16")
        raw = path.read_bytes()
        result = self.selected(source, path)
        self.assertEqual(result.target.encoding, "utf-16-le")
        self.assertEqual(path.read_bytes(), raw)

    def test_changed_translation_needs_original_exact_time_and_file_anchors(self):
        source = self.source()
        evs = self.simplified()
        evs[2].text = "请到大厅集合"
        result = self.selected(source, self.target(evs))
        self.assertEqual(result.target_matches[25], 2)
        self.assertEqual(result.target.events[2].text, "请到大厅集合")

    def test_changed_translation_with_changed_time_stays_unmatched(self):
        source = self.source()
        evs = self.simplified()
        evs[2].text, evs[2].start = "请到大厅集合", evs[2].start + 10
        result = self.selected(source, self.target(evs))
        self.assertNotIn(25, result.target_matches)
        self.assertEqual(len(result.target_matches), 5)

    def test_same_text_does_not_require_identical_times(self):
        source = self.source()
        evs = self.simplified()
        for event in evs:
            event.start += 1500
            event.end += 1500
        result = self.selected(source, self.target(evs))
        self.assertEqual(len(result.target_matches), 6)

    def test_target_with_only_matching_times_and_order_is_rejected(self):
        source = self.source()
        evs = _events([f"完全不同的第{i}行" for i in range(6)])
        result = self.selected(source, self.target(evs))
        self.assertIsNone(result.target)
        self.assertTrue(any("证据不足" in issue for issue in result.issues))

    def test_repeated_lines_use_unique_original_times(self):
        texts = list(_TEXT)
        texts[2] = texts[3] = "這是重複的歌詞"
        source = self.source(texts)
        result = self.selected(source, self.target(self.simplified(_events(texts))))
        self.assertEqual(result.target_matches[25], 2)
        self.assertEqual(result.target_matches[32], 3)

    def test_repeated_lines_with_ambiguous_times_are_not_paired_by_index(self):
        evs = _events()
        evs[2].text = evs[3].text = "這是重複的歌詞"
        evs[3].start, evs[3].end = evs[2].start, evs[2].end
        source = self.source(events=evs)
        result = self.selected(source, self.target(self.simplified(evs)))
        self.assertNotIn(25, result.target_matches)
        self.assertNotIn(32, result.target_matches)

    def test_split_event_is_not_matched_to_first_fragment(self):
        source = self.source()
        evs = self.simplified()
        original = evs.pop(2)
        evs[2:2] = [ReferenceEvent(90, original.start, original.start + 1000, "请先回到"),
                     ReferenceEvent(91, original.start + 1000, original.end, "这里")]
        result = self.selected(source, self.target(evs))
        self.assertNotIn(25, result.target_matches)
        self.assertEqual(result.target_matches[32], 4)

    def test_merge_does_not_match_two_events_to_one_reference(self):
        source = self.source()
        evs = self.simplified()
        evs[2:4] = [ReferenceEvent(90, evs[2].start, evs[3].end, evs[2].text + evs[3].text)]
        result = self.selected(source, self.target(evs))
        self.assertNotIn(25, result.target_matches)
        self.assertNotIn(32, result.target_matches)
        self.assertEqual(len(set(result.target_matches.values())), len(result.target_matches))

    def test_parallel_unmatched_line_blocks_changed_time_only_match(self):
        source = self.source()
        evs = self.simplified()
        evs[2].text = "重新改译的一句"
        evs.insert(3, ReferenceEvent(90, evs[2].start + 100, evs[2].end - 100, "额外的拆句"))
        result = self.selected(source, self.target(evs))
        self.assertNotIn(25, result.target_matches)

    def test_reordered_unique_lines_are_not_arbitrarily_chosen(self):
        source = self.source()
        evs = self.simplified()
        evs[2], evs[3] = evs[3], evs[2]
        result = self.selected(source, self.target(evs))
        self.assertNotIn(25, result.target_matches)
        self.assertNotIn(32, result.target_matches)

    def test_wrong_episode_is_rejected_even_when_song_and_times_match(self):
        source = self.source()
        path = self.target(name="Show.S01E02.chs.ass")
        result = self.selected(source, path)
        self.assertIsNone(result.target)
        self.assertTrue(any("集号" in issue for issue in result.issues))
        self.assertIn(str(path), result.protected_input_paths)

    def test_bracket_episode_matches_padded_episode_metadata(self):
        source = self.source()
        path = self.target(name="[Publisher] Show [01][1080p].SC.ass")
        self.assertEqual(self.selected(source, path).target.path, str(path))

    def test_automatic_discovery_never_recurses_or_visits_neighbours(self):
        source = self.source(name="current/Show.S01E001.cht.ass")
        nested = self.target(name="current/DuoSub/Show.S01E01.chs.ass")
        sibling = self.target(name="original/Show.S01E01.chs.ass")
        result = self.selected(source)
        self.assertIsNone(result.target)
        self.assertNotIn(str(nested), result.protected_input_paths)
        self.assertNotIn(str(sibling), result.protected_input_paths)
        explicit = self.selected(source, sibling.parent)
        self.assertEqual(explicit.target.path, str(sibling))

    def test_explicit_directory_only_selects_targets_from_that_directory(self):
        source = self.source(name="current/Show.S01E001.cht.ass")
        self.target(name="current/Show.S01E01.chs.ass")
        selected = self.target(name="chosen/Show.S01E01.chs.ass")
        result = self.selected(source, selected.parent)
        self.assertEqual(result.target.path, str(selected))

    def test_different_valid_editions_require_explicit_file(self):
        source = self.source()
        first = self.target()
        evs = self.simplified()
        evs[2].text = "另一个版本的翻译"
        second = self.target(evs, name="Other.S01E01.chs.ass")
        result = self.selected(source)
        self.assertIsNone(result.target)
        self.assertTrue(any("多个不同" in issue for issue in result.issues))
        self.assertEqual(self.selected(source, second).target.path, str(second))
        self.assertIn(str(first), result.protected_input_paths)

    def test_identical_visible_copies_are_not_a_translation_ambiguity(self):
        source = self.source()
        self.target()
        self.target(name="Copy.S01E01.zh-Hans.ass", encoding="utf-8")
        self.assertIsNotNone(self.selected(source).target)

    def test_generated_duosub_output_is_not_a_native_reference(self):
        source = self.source()
        path = self.target(header="; 双语字幕工坊（DuoSub）— 由工程导出，不要直接编辑")
        result = self.selected(source, path)
        self.assertIsNone(result.target)
        self.assertTrue(any("工程生成" in issue for issue in result.issues))

    def test_script_mismatch_and_unprovable_script_are_reported(self):
        source = self.source()
        traditional = self.ass("Show.S01E01.cht.ass")
        result = self.selected(source, traditional)
        self.assertIsNone(result.target)
        self.assertTrue(any("字形标签" in issue for issue in result.issues))
        neutral = self.ass("Show.S01E01.ass", _events(["天地人", "山川日月", "森林大地"]))
        self.assertIsNone(self.selected(source, neutral).target)

    def test_mislabelled_script_cannot_supply_wrong_glyphs(self):
        source = self.source()
        path = self.ass("Show.S01E01.chs.ass")
        result = self.selected(source, path)
        self.assertIsNone(result.target)
        self.assertTrue(any("标签与正文不一致" in issue for issue in result.issues))

    def test_unlabelled_target_uses_actual_script_not_year(self):
        source = self.source()
        path = self.target(name="2025/S01E01.ass")
        result = self.selected(source, path.parent)
        self.assertEqual(result.target.path, str(path))
        self.assertEqual(result.target.script_hint, "zh-Hans")

    def test_paired_native_companion_provides_original_same_script_baseline(self):
        texts = list(_TEXT)
        texts[2] = "這是匯入前已改過的一句"
        source = self.source(texts, name="edited/Edited.S01E001.zh-Hant.ass")
        baseline = self.ass("native/Show.S01E01.TC.ass")
        target = self.target(name="native/Show.S01E01.SC.ass")
        result = self.selected(source, target.parent)
        self.assertEqual(result.baseline.path, str(baseline))
        self.assertEqual(result.target.path, str(target))
        self.assertEqual(result.baseline_matches[25], 2)
        self.assertEqual(result.baseline.events[2].text, _TEXT[2])

    def test_arbitrary_same_script_file_is_not_declared_true_original(self):
        source = self.source(name="edited/Edited.S01E001.zh-Hant.ass")
        self.ass("native/Unrelated.S01E01.TC.ass")
        target = self.target(name="native/Show.S01E01.SC.ass")
        result = self.selected(source, target.parent)
        self.assertIsNone(result.baseline)

    def test_native_companion_equal_to_source_bytes_is_still_a_proven_baseline(self):
        source = self.source(name="edited/Edited.S01E001.zh-Hant.ass")
        baseline = self.ass("native/Show.S01E01.TC.ass")
        target = self.target(name="native/Show.S01E01.SC.ass")
        self.assertEqual(baseline.read_bytes(), Path(source.path).read_bytes())
        result = self.selected(source, target.parent)
        self.assertEqual(result.baseline.path, str(baseline))
        self.assertEqual(result.baseline.sha256, source.sha256)
        self.assertEqual(len(result.baseline_matches), 6)

    def test_all_inputs_and_enumerated_candidates_are_protected(self):
        source = self.source()
        selected = self.target()
        other = self.target(name="Show.S01E02.chs.ass")
        japanese = self.ass("Show.S01E01.ja-JP.ass")
        result = self.selected(source, self.root)
        for path in (Path(source.path), self.root, selected, other, japanese):
            self.assertIn(str(path), result.protected_input_paths)
        self.assertEqual(len(result.protected_input_paths), len(set(result.protected_input_paths)))

    def test_resolution_does_not_mutate_source_or_any_file(self):
        source = self.source()
        path = self.target()
        source_before = asdict(source)
        originals = {p: p.read_bytes() for p in self.root.iterdir() if p.is_file()}
        self.selected(source, path)
        self.assertEqual(asdict(source), source_before)
        self.assertEqual({p: p.read_bytes() for p in originals}, originals)

    def test_invalid_missing_or_empty_files_fail_closed(self):
        source = self.source()
        missing = self.root / "missing.ass"
        self.assertIsNone(self.selected(source, missing).target)
        binary = self.root / "Show.S01E01.chs.ass"
        binary.write_bytes(b"\x00\x01\x02")
        result = self.selected(source, binary)
        self.assertIsNone(result.target)
        self.assertTrue(any("无法读取" in issue for issue in result.issues))
        binary.write_text("[Script Info]\nTitle: Empty\n", encoding="utf-8")
        self.assertIsNone(self.selected(source, binary).target)

    def test_cancel_is_propagated_without_becoming_an_issue(self):
        source = self.source()
        self.target()

        class Cancelled(Exception):
            pass

        class Runner:
            def __init__(self):
                self.calls = 0

            def check_cancelled(self):
                self.calls += 1
                if self.calls >= 5:
                    raise Cancelled("stop")

        runner = Runner()
        with self.assertRaises(Cancelled):
            resolve_reference(source, "zh-Hans", runner=runner)
        self.assertEqual(runner.calls, 5)

    def test_large_repeated_input_does_not_guess_or_require_quadratic_alignment(self):
        texts = [f"這是唯一的第{i}個原始句子" if i % 4 == 0 else "重複的共同歌詞" for i in range(1200)]
        source = self.source(texts)
        path = self.target(self.simplified(_events(texts)))
        result = self.selected(source, path)
        self.assertEqual(len(result.target_matches), 1200)

    def test_tiny_exact_file_is_safe_but_tiny_retranslation_is_not(self):
        source = self.source(_TEXT[:2])
        path = self.target(self.simplified(_events(_TEXT[:2])))
        self.assertEqual(len(self.selected(source, path).target_matches), 2)
        evs = self.simplified(_events(_TEXT[:2]))
        evs[1].text = "完全不同的译文"
        self.target(evs)
        self.assertIsNone(self.selected(source, path).target)

    def test_tiny_duplicate_file_with_ambiguous_times_is_not_paired_by_index(self):
        evs = [ReferenceEvent(i, 1000, 2000, "重複的共同歌詞") for i in range(2)]
        source = self.source(events=evs)
        result = self.selected(source, self.target(self.simplified(evs)))
        self.assertIsNone(result.target)

    def test_changed_wording_with_globally_ambiguous_timestamp_is_not_paired(self):
        evs = _events()
        evs[2].start, evs[2].end = evs[0].start, evs[0].end
        source = self.source(events=evs)
        translated = self.simplified(evs)
        translated[2].text = "改译后的新句子"
        result = self.selected(source, self.target(translated))
        self.assertNotIn(25, result.target_matches)


if __name__ == "__main__":
    unittest.main()
