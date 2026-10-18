"""Model-free numeric song fixtures and karaoke transaction integration."""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.kalign import song, tokens
from subkit.core import karaoke as kara
from subkit.core.ops import Editor
from subkit.model.document import Document, Group, Member, Run
from subkit.model.serialize import document_from_json, document_to_json


def emission(name, frames, peaks, vocab=None, *, blank=0, star=None, confidence=.95):
    vocab = vocab or {"a" if name == "mms" else "あ": 1}
    probs = np.full((frames, max([blank, *(vocab.values()), star or 0]) + 1), .001)
    probs[:, blank] = .95
    for frame, token in peaks:
        probs[frame:frame + 2, :] = (1 - confidence) / (probs.shape[1] - 1)
        probs[frame:frame + 2, token] = confidence
    probs /= probs.sum(axis=1, keepdims=True)
    return song.ModelEmission(name, np.log(probs), 20, vocab, blank, star)


class SongNumerics(unittest.TestCase):
    def test_voice_gate_bins_strict_threshold_nonzero_blank_star_and_copy(self):
        line = song.LineInput("jp", 0, 100, [song.BeatInput("あ")])
        model = emission("mms", 10, [], {"a": 0, "*": 2}, blank=1, star=2)
        model.frame_ms = 15
        before = model.log_probs.copy()
        rms = np.zeros(30)
        rms[3], rms[6] = -40, -18
        def align(lp, targets, windows, blank, hold):
            expected = before.copy()
            expected[2, 0] -= song.VOICE_GATE_PENALTY  # frame 30 ms -> bin 3
            np.testing.assert_array_equal(lp, expected)
            return np.array([0, 2, 4]), np.array([0, 2, 4]), 0
        with patch.object(song.viterbi, "align", side_effect=align):
            song.align_model([line], model, 1000, rms)
        np.testing.assert_array_equal(model.log_probs, before)

    def test_voice_gate_failure_retries_original_and_flags(self):
        line = song.LineInput("jp", 0, 100, [song.BeatInput("あ")])
        model = emission("kana", 10, [(2, 1)])
        rms = np.zeros(20)
        rms[:4] = -40
        valid = (np.array([2]), np.array([3]), 0)
        with patch.object(song.viterbi, "align", side_effect=[ValueError("gate"), ValueError("gate"), valid]) as call:
            result = song.align_model([line], model, 0, rms)
        self.assertEqual(call.call_count, 3)
        np.testing.assert_array_equal(call.call_args.args[0], model.log_probs)
        self.assertEqual(result.flags, ["voice_gate_fallback:kana"])
        self.assertTrue(result.lines[0][0].check)

    def test_continuation_targets_empty_and_not_unread_check(self):
        line = song.LineInput("jp", 0, 1000, [song.BeatInput("こ"), song.BeatInput("う")])
        model = emission("kana", 50, [(10, 1)], {"こ": 1, "う": 2})
        def align(lp, targets, windows, blank, hold):
            self.assertEqual(targets, [[1], []])
            return np.array([10, 20]), np.array([11, 21]), 0
        with patch.object(song.viterbi, "align", side_effect=align):
            result = song.align_model([line], model, 0)
        self.assertEqual(result.lines[0][1].start, result.lines[0][0].start)
        self.assertFalse(result.lines[0][1].check)

    def test_continuation_ratio_samples_fallback_and_clipping(self):
        self.assertEqual(song.continuation_ratio([.7] * 4), (.5, "default", 4))
        self.assertEqual(song.continuation_ratio([.7] * 4, .65), (.65, "stored", 4))
        self.assertEqual(song.continuation_ratio([.9] * 5, .65), (.8, "habit", 5))
        self.assertEqual(song.continuation_ratio([.1] * 5), (.3, "habit", 5))
        self.assertEqual(song.continuation_ratio([0, 1, np.nan, np.inf]), (.5, "default", 0))

    def test_continuation_split_silence_bound_minimum_and_tiny_check(self):
        model = emission("kana", 100, [], {"こ": 1, "う": 2, "お": 3})
        rms = np.zeros(200)
        rms[70:80] = -40
        for limit, expected, check in ((900, 460, False), (200, 150, True)):
            starts, ends = [100, 140, limit], [700, 140, 1000]
            beats = [song.ModelBeat(a, .9) for a in starts]
            song.split_continuations(starts, ends, beats, ["こ", "う", "か"], [False, True, False],
                                     rms, 0, 1000, model, .6, 0)
            self.assertEqual(starts[1], expected)
            self.assertEqual(ends[0], expected)
            self.assertEqual(ends[1], min(700, limit))
            self.assertEqual(beats[1].check, check)
            self.assertEqual(beats[0].check, check)

    def test_changed_vowel_requires_sustained_nonblank_evidence(self):
        model = emission("kana", 50, [], {"お": 1, "う": 2})
        # Tiny probabilities under blank must not look like a vowel change.
        model.log_probs[:, 2] += .1
        self.assertIsNone(song._vowel_cut(model, "こ", "う", 0, 800, 0, 0))

        model.log_probs[20:25, 2] = 0
        self.assertEqual(song._vowel_cut(model, "こ", "う", 0, 800, 0, 0), 350)
        model.log_probs[24, 2] = -20
        self.assertIsNone(song._vowel_cut(model, "こ", "う", 0, 800, 0, 0))

    def test_run_learns_only_matching_original_hand_pairs_and_eval_override(self):
        lines = [song.LineInput(str(i), i * 1000, (i + 1) * 1000,
                               [song.BeatInput("こ"), song.BeatInput("う")]) for i in range(5)]
        models = [emission("mms", 300, [], {"k": 1, "o": 2, "u": 3}),
                  emission("kana", 300, [], {"こ": 1, "お": 2, "う": 3})]
        aligned = song.ModelResult([[song.ModelBeat(i * 1000 + 100, .9),
                                     song.ModelBeat(i * 1000 + 100, .9)] for i in range(5)])
        hand = [dict(mid=str(i), index=j, t0=i * 1000 + t, by="tap")
                for i in range(5) for j, t in enumerate((100, 730))]
        hand += [dict(mid="missing", index=1, t0=9000, by="tap"),
                 dict(mid="0", index=1, t0=999, by="drag", lane="translation")]
        with patch.object(song, "align_model", return_value=aligned):
            result = song.run(lines, models, np.zeros(600), 0, {"hand_beats": hand, "offset": "none", "continuation_mode": "habit"})
            fixed = song.run(lines, models, np.zeros(600), 0, {"hand_beats": hand, "offset": "none", "continuation_ratio": .6})
            even = song.run(lines, models, np.zeros(600), 0, {"hand_beats": hand, "offset": "none", "stored_continuation": .8})
        self.assertEqual((result.continuation_ratio, result.continuation_source, result.continuation_count), (.7, "habit", 5))
        self.assertEqual((fixed.continuation_ratio, fixed.continuation_source, fixed.continuation_count), (.6, "default", 0))
        self.assertEqual((even.continuation_ratio, even.continuation_source, even.continuation_count), (.5, "even", 0))
        with self.assertRaisesRegex(ValueError, "continuation_mode"):
            song.run(lines, models, [], 0, {"continuation_mode": "invalid"})

    def test_synthetic_run_prototype_fusion_ends_and_calibration_numbers(self):
        # Prototype corrections -10/-50 -> mean -30, then learned +12 ms.
        frames = [10, 40] + list(range(60, 240, 10))
        line = song.LineInput("jp", 1000, 6200, [song.BeatInput("ん") for _ in frames])
        models = [emission(name, 300, [(f, 1) for f in frames], {"n" if name == "mms" else "ん": 1}) for name in ("mms", "kana")]
        rms = np.zeros(600)
        rms[40:50] = rms[480:490] = -40
        hand = [{"mid": "jp", "index": i, "t0": 1000 + f * 20 - 30 + 12, "by": "tap"}
                for i, f in enumerate(frames)]
        with patch.object(tokens, "particle_positions_many", return_value=[set()]) as particles:
            result = song.run([line], models, rms, 1000, {"hand_beats": hand})
        particles.assert_called_once()
        self.assertEqual((result.offset_ms, result.offset_source, result.offset_count), (12, "habit", 20))
        beats = result.lines[0].beats
        self.assertEqual([b[0] for b in beats], [1000 + f * 20 - 18 for f in frames])
        self.assertEqual(beats[0][1], 1400)
        self.assertEqual(beats[1][1], beats[2][0])
        self.assertEqual(beats[-1][1], 5800)
        self.assertTrue(all(b[2] > .9 and not b[3] for b in beats))
        self.assertEqual(result.flags, [])
        self.assertEqual(json.loads(json.dumps(asdict(result)))["lines"][0]["mid"], "jp")

    def test_fusion_200ms_boundary_confidence_empty_and_order(self):
        a = [song.ModelBeat(101, .4), song.ModelBeat(100, .9), song.ModelBeat(90, 0, True)]
        b = [song.ModelBeat(301, .8), song.ModelBeat(301, .8), song.ModelBeat(90, .6)]
        result = song.fuse(a, b)
        self.assertEqual([x.start for x in result], [201, 301, 401])
        self.assertEqual([x.check for x in result], [False, True, True])
        self.assertEqual([x.conf for x in result], [.8, .9, .6])
        with self.assertRaises(ValueError):
            song.fuse(a, b[:1])

    def test_line_start_moves_past_breath_after_previous_held_note(self):
        # 10 ms bins from clip 0: voice 0-500 ms (previous line's held vowel), breath 500-800, voice from 800.
        silent = np.array([False] * 50 + [True] * 30 + [False] * 70)
        self.assertEqual(song.after_breath(100, 1200, silent, 0, held_from=0), 800)
        self.assertEqual(song.after_breath(100, 1200, silent, 0), 100)          # first line of the song
        self.assertEqual(song.after_breath(100, 1200, silent, 1000, held_from=900), 100)  # breath outside [start, limit)
        self.assertEqual(song.after_breath(100, 840, silent, 0, held_from=0), 100)      # < 50 ms would remain
        self.assertEqual(song.after_breath(550, 1200, silent, 0, held_from=0), 800)     # starts inside that breath
        short = np.array([False] * 50 + [True] * 9 + [False] * 70)              # 90 ms is not a breath
        self.assertEqual(song.after_breath(100, 1200, short, 0, held_from=0), 100)
        self.assertEqual(song.after_breath(100, 1200, np.zeros(0, dtype=bool), 0, held_from=0), 100)
        # The first syllable has its own breath before it (the previous note ended): sung, then a breath — stays.
        own = np.array([False] * 20 + [True] * 15 + [False] * 15 + [True] * 30 + [False] * 70)
        self.assertEqual(song.after_breath(350, 1200, own, 0, held_from=0), 350)

    def test_silent_start_100ms_partial_bin_threshold_and_missing_rms(self):
        silent = np.ones(40, dtype=bool)
        silent[19] = False
        self.assertFalse(song.silent_start(100, silent, 0))  # last bin is voiced
        self.assertTrue(song.silent_start(200, silent, 0))
        self.assertFalse(song.silent_start(0, [], 0))
        self.assertFalse(song.silent_start(1095, silent, 1000))

    def test_all_trusted_equals_round_one_and_no_local_ctc(self):
        line = song.LineInput("jp", 0, 1000, [song.BeatInput("ん")] * 3)
        models = [emission(n, 60, [(10, 1), (25, 1), (40, 1)],
                           {"n" if n == "mms" else "ん": 1}) for n in ("mms", "kana")]
        with patch.object(song, "MAX_ALIGNMENT_ROUNDS", 1):
            first = song.run([line], models, np.zeros(120), 0, {"offset": "none"})
        with patch.object(song, "_refine_model", side_effect=AssertionError("trusted song refined")):
            final = song.run([line], models, np.zeros(120), 0, {"offset": "none"})
        self.assertEqual(final, first)

    def test_local_refinement_preserves_both_trusted_anchors(self):
        line = song.LineInput("jp", 0, 1000, [song.BeatInput("ん")] * 3)
        models = [song.ModelResult([[song.ModelBeat(a, .9) for a in row]])
                  for row in ([100, 300, 700], [100, 450, 700])]
        fused = [[song.ModelBeat(a, .9) for a in (100, 375, 700)]]
        emissions = [emission(n, 60, [], {"n" if n == "mms" else "ん": 1}) for n in ("mms", "kana")]
        def local(*args, **kwargs):
            self.assertEqual(args[-4:], (0, 2, 100, 700))
            self.assertTrue(kwargs["anchored"])
            return {0: song.ModelBeat(110, .9), 1: song.ModelBeat(400, .9)}
        with patch.object(song, "_refine_model", side_effect=local) as call:
            rows, _ = song._refine([line], models, fused, [["ん"] * 3], [[False] * 3],
                                  emissions, np.zeros(120), 0, {}, None, None, 0)
        self.assertEqual(rows[0][0], [100, 400, 700])
        self.assertEqual(call.call_count, 2)

    def test_continuation_anchor_remains_fixed_when_parent_changes(self):
        line = song.LineInput("jp", 0, 1000, [song.BeatInput("こ"), song.BeatInput("う"), song.BeatInput("か")])
        fused = [[song.ModelBeat(a, .9) for a in (100, 100, 800)]]
        kana = emission("kana", 60, [], {"こ": 1, "お": 2, "う": 3, "か": 4})
        rows = song._render([line], fused, [["こ", "う", "か"]], [[False, True, False]],
                            np.zeros(120), 0, {}, kana, .5, 0, anchors={0: {1: 500}})
        self.assertEqual(rows[0][0][1], 500)
        self.assertEqual(rows[0][1][0], 500)

    def test_local_first_beat_anchor_uses_frame_shift_and_100ms_ceiling(self):
        line = song.LineInput("jp", 0, 1000, [song.BeatInput("ん")] * 2)
        model = emission("kana", 80, [], {"ん": 1}); model.frame_ms = 30
        def align(lp, targets, windows, blank, **kwargs):
            self.assertEqual(kwargs["min_frames"], [4, 4])
            # 110+50 ms -> round(160/30)=5; cropped lo=4 -> anchor 1 +/-1.
            self.assertEqual(kwargs["start_windows"][0], (0, 2))
            return np.array([1, 6]), np.array([1, 6]), 0
        with patch.object(song.viterbi, "align", side_effect=align):
            song._refine_model(line, ["ん", "ん"], [False, False], model, np.zeros(240),
                               0, 0, [110, 400], 0, 2, 110, 700, anchored=True)

    def test_infeasible_proportional_span_preserves_anchor_and_flags(self):
        beats = [song.ModelBeat(a, .9) for a in (100, 140, 180)]
        song._proportional(beats, ["か", "きゃ", "ん"], [140, 180, 220], 0, 3, 100, 220, 0)
        self.assertEqual([b.start for b in beats], [100, 140, 180])
        self.assertTrue(all(b.check for b in beats))

    def test_unsorted_lines_bound_last_end_and_keep_existing_overlap(self):
        lines = [song.LineInput("later", 800, 1200, [song.BeatInput("ん")]),
                 song.LineInput("early", 0, 800, [song.BeatInput("ん")])]
        fused = [[song.ModelBeat(900, .9)], [song.ModelBeat(100, .9)]]
        args = (lines, fused, [["ん"], ["ん"]], [[False], [False]], np.zeros(300), 0,
                {"last_end": "line"}, None, None, 0)
        self.assertEqual(song._render(*args)[1][1], [800])
        lines[1].end = 1000  # existing line overlap: next lyric start only
        self.assertEqual(song._render(*args)[1][1], [900])

    def test_rms_100ms_boundary_60ms_minimum_and_strict_threshold(self):
        rms = np.zeros(200)
        rms[5:15] = -40  # only 90 ms remain after +60, so cannot finish here
        rms[30:40] = -18  # equal threshold is not silence
        rms[50:60] = -40
        self.assertEqual(song.ends_from_rms([1000, 1800], rms, 1000, 2000), [1500, 2000])

    def test_last_voice_line_and_fallback_late_start(self):
        rms = np.zeros(300)
        rms[120:130] = -40
        self.assertEqual(song.ends_from_rms([1800], rms, 1000, 2500, "voice"), [2200])
        self.assertEqual(song.ends_from_rms([1800], rms, 1000, 2500, "line"), [2500])
        self.assertEqual(song.ends_from_rms([1800], rms, 1000, 2000, "line"), [2200])
        self.assertEqual(song.ends_from_rms([2500], [], 1000, 2000), [2500])

    def test_last_silence_search_stops_at_line_end_plus_1000(self):
        rms = np.zeros(400)
        rms[201:211] = -40
        self.assertEqual(song.ends_from_rms([1000], rms, 1000, 2000), [2000])

    def test_calibration_filters_19_20_median_clip_and_stored(self):
        pred = {("jp", i): 1000 for i in range(21)}
        hand = [{"mid": "jp", "index": i, "t0": 1250, "by": "drag"} for i in range(20)]
        self.assertEqual(song.calibration(pred, hand, -20), (120, "habit", 20))
        for b in hand:
            b["t0"] = 750
        self.assertEqual(song.calibration(pred, hand), (-120, "habit", 20))
        self.assertEqual(song.calibration(pred, hand[:19], -20), (-20, "stored", 19))
        self.assertEqual(song.calibration(pred, hand[:19]), (0, "none", 19))
        hand += [{"mid": "jp", "index": 20, "t0": 9999, "by": "tap", "lane": "translation"},
                 {"mid": "jp", "index": 20, "t0": 9999, "by": "manual"}]
        self.assertEqual(song.calibration(pred, hand)[2], 20)

    def test_run_offset_modes_and_empty_song(self):
        line = song.LineInput("jp", 1000, 2000, [song.BeatInput("あ")])
        models = [emission(name, 80, [(10, 1)]) for name in ("mms", "kana")]
        for mode, expected, source in (("none", 1170, "none"), (50, 1220, "custom"), ("habit", 1185, "stored")):
            result = song.run([line], models, [], 1000, {"offset": mode, "stored_habit": 15})
            self.assertEqual(result.lines[0].beats[0][0], expected)
            self.assertEqual(result.offset_source, source)
        self.assertEqual(song.run([], models, [], 0).lines, [])
        with self.assertRaises(ValueError):
            song.run([line], models[:1], [], 1000)
        with self.assertRaises(ValueError):
            song.run([line], models, [], 1000, {"offset": "invalid"})

    def test_empty_tokens_flagged_in_real_run(self):
        line = song.LineInput("jp", 0, 1000, [song.BeatInput("漢")])
        result = song.run([line], [emission(n, 50, []) for n in ("mms", "kana")], [], 0)
        self.assertTrue(result.lines[0].beats[0][3])
        self.assertEqual(result.lines[0].beats[0][2], 0)

    def test_mms_stars_particles_ruby_and_fixed_clipped_windows(self):
        lines = [song.LineInput("a", 1020, 1100, [song.BeatInput("は", fixed=(1040, 1080), a=1)], "猫は"),
                 song.LineInput("b", 1200, 1500, [song.BeatInput("へ", True, True)])]
        model = emission("mms", 50, [], {"w": 1, "a": 2, "h": 3, "e": 4}, star=5)
        def align(lp, targets, windows, blank, hold):
            self.assertEqual(targets, [[5], [1, 2], [5], [3, 4], [5]])
            self.assertEqual(windows[1], (0, 8))
            self.assertEqual(windows[2], (0, 49))
            self.assertEqual(hold, 3)
            return np.arange(5), np.arange(5), 0
        with patch.object(tokens, "particle_positions_many", return_value=[{1}, set()]), \
                patch.object(song.viterbi, "align", side_effect=align):
            result = song.align_model(lines, model, 1000)
        self.assertEqual([len(b) for b in result.lines], [1, 1])
        self.assertEqual(result.lines[0][0].start, 1010)

    def test_kana_merge_into_nonzero_blank_does_not_mutate_input(self):
        line = song.LineInput("jp", 0, 100, [song.BeatInput("あー")])
        model = emission("kana", 10, [], {"あ": 0, "|": 1, "ー": 2}, blank=3)
        before = model.log_probs.copy()
        def align(lp, targets, windows, blank, hold):
            self.assertEqual(blank, 3)
            self.assertEqual(targets, [[0]])
            np.testing.assert_allclose(lp[:, 3], np.logaddexp.reduce(before[:, [3, 1, 2]], axis=1))
            self.assertTrue(np.isneginf(lp[:, 1:3]).all())
            return np.array([2]), np.array([3]), 0
        with patch.object(song.viterbi, "align", side_effect=align):
            result = song.align_model([line], model, 0)
        np.testing.assert_array_equal(model.log_probs, before)
        self.assertEqual(result.lines[0][0].start, -10)
        self.assertAlmostEqual(result.lines[0][0].conf, np.exp(before[2:4, 0]).mean())

    def test_window_retry_double_once_flag_and_second_failure(self):
        line = song.LineInput("jp", 0, 100, [song.BeatInput("あ", fixed=(400, 400))])
        model = emission("kana", 100, [(20, 1)])
        real = song.viterbi.align
        calls = []
        def align(lp, targets, windows, blank, hold):
            calls.append(windows)
            if len(calls) == 1:
                raise ValueError("no path")
            return real(lp, targets, windows, blank, hold)
        with patch.object(song.viterbi, "align", side_effect=align):
            result = song.run([line], [model, emission("mms", 100, [(20, 1)])], [], 0)
        self.assertEqual(calls[:2], [[(16, 24)], [(11, 28)]])
        self.assertIn("windows_widened:kana", result.flags)
        with patch.object(song.viterbi, "align", side_effect=ValueError("no path")) as align:
            with self.assertRaises(ValueError):
                song.align_model([line], model, 0)
        self.assertEqual(align.call_count, 2)

    def test_batch_particle_dictionary_once_and_closure(self):
        dictionary = Mock()
        morph = SimpleNamespace(surface=lambda: "は", begin=lambda: 1, part_of_speech=lambda: ("助詞",))
        dictionary.create.return_value.tokenize.return_value = [morph]
        fake = SimpleNamespace(Dictionary=Mock(return_value=dictionary), SplitMode=SimpleNamespace(C=0))
        with patch.dict(sys.modules, {"sudachipy": fake}):
            self.assertEqual(tokens.particle_positions_many(["猫は", "君は"]), [{1}, {1}])
        fake.Dictionary.assert_called_once_with(dict="full")
        self.assertEqual(dictionary.create.return_value.tokenize.call_count, 2)
        dictionary.close.assert_called_once()
        with patch.dict(sys.modules, {"sudachipy": None}):
            self.assertEqual(tokens.particle_positions_many(["は", "へ"]), [set(), set()])
        self.assertEqual(tokens.particle_positions_many([]), [])

    def test_batch_particle_closes_on_later_line_failure(self):
        dictionary = Mock()
        dictionary.create.return_value.tokenize.side_effect = [[], RuntimeError("second line")]
        fake = SimpleNamespace(Dictionary=Mock(return_value=dictionary), SplitMode=SimpleNamespace(C=0))
        with patch.dict(sys.modules, {"sudachipy": fake}), self.assertRaises(RuntimeError):
            tokens.particle_positions_many(["猫", "は"])
        dictionary.close.assert_called_once()


class AutoKaraokeStorage(unittest.TestCase):
    def setUp(self):
        self.doc = Document("auto", groups=[Group("g", "song", ["jp", "cn"])])
        for mid, text, lane in (("jp", "あいう", "original"), ("cn", "你好", "translation")):
            self.doc.members[mid] = Member(mid, "g", "ja" if lane == "original" else "zh-Hans", lane,
                                          "song", 1000, 2000, runs=[Run(text, ins="i_" + mid, a=0, b=len(text))])
        self.ed = Editor(self.doc, {})
        self.result = song.LineResult("jp", [(900, 1200, .9, True), (1300, 1700, .8, False), (1800, 2300, .7, True)])

    def view(self, mid="jp"):
        return kara.effective(self.doc, self.doc.members[mid])

    def apply(self, **kwargs):
        return self.ed.apply_auto_karaoke([self.result], translations=kwargs.get("translations", False),
                                          overwrite=kwargs.get("overwrite", False))

    def test_align_timed_check_roundtrip_suggested_and_single_undo(self):
        before = copy.deepcopy(document_to_json(self.doc))
        before.pop("revision", None)  # History revisions advance on undo/redo.
        report = self.apply(translations=True)
        self.assertEqual((report["updated"], report["lines_with_check"], report["widened_lines"]), (1, 1, 2))
        self.assertEqual(report["translation_report"]["updated"], 1)
        self.assertEqual(report["translations_skipped"], "")
        view = self.view()
        self.assertEqual(view.state, "suggested")
        self.assertFalse(view.complete)
        self.assertTrue(all(b.timed and b.by == "align" for b in view.beats))
        self.assertEqual(["check" in b.flags for b in view.beats], [True, False, True])
        self.assertIn("自动建议还没确认", view.issues)
        restored = document_from_json(json.loads(json.dumps(document_to_json(self.doc))))
        self.assertEqual(["check" in b.flags for b in kara.effective(restored, restored.members["jp"]).beats], [True, False, True])
        self.assertEqual(len(self.ed.history.done), 1)
        after = copy.deepcopy(document_to_json(self.doc))
        after.pop("revision", None)
        self.ed.history.undo()
        undone = document_to_json(self.doc)
        undone.pop("revision", None)
        self.assertEqual(undone, before)
        self.ed.history.redo()
        redone = document_to_json(self.doc)
        redone.pop("revision", None)
        self.assertEqual(redone, after)

    def test_confirm_clears_checks_and_is_undoable(self):
        self.apply()
        self.ed.set_karaoke_state(["jp"], state="confirmed")
        self.assertTrue(self.view().complete)
        self.assertTrue(all("check" not in b.flags for b in self.view().beats))
        self.ed.history.undo()
        self.assertIn("check", self.view().beats[0].flags)

    def test_move_shared_edge_shift_and_noop_only_clear_changed_beats(self):
        self.apply()
        beats = self.view().beats
        beats[0].t1 = beats[1].t0
        beats[1].flags.append("check")
        moved = kara.move_edge(beats, 1, "start", 1400)
        self.assertEqual(["check" in b.flags for b in moved], [False, False, True])
        self.assertIn("check", beats[0].flags)
        unchanged = kara.move_edge(beats, 2, "end", beats[2].t1)
        self.assertIn("check", unchanged[2].flags)
        shifted = kara.move_edge(beats, 0, "start", 950, shift=True)
        self.assertTrue(all("check" not in b.flags for b in shifted))

    def test_retap_clears_one_check_preserves_other_and_caller(self):
        self.apply()
        beats = self.view().beats
        beats[0].by = "tap"
        self.ed.set_karaoke("jp", beats, state="suggested")
        self.assertEqual(["check" in b.flags for b in self.view().beats], [False, False, True])
        self.assertIn("check", beats[0].flags)

    def test_protection_tap_drag_manual_confirmed_and_overwrite(self):
        for by in ("tap", "drag", "manual"):
            beats = self.view().beats
            for b in beats:
                b.by = by
            self.ed.set_karaoke("jp", beats)
            count = len(self.ed.history.done)
            self.assertEqual(self.apply()["skipped"], {"jp": "已确认或手调过"})
            self.assertEqual(len(self.ed.history.done), count)
            self.assertEqual(self.apply(overwrite=True)["updated"], 1)
        self.ed.set_karaoke_state(["jp"], state="confirmed")
        self.assertEqual(self.apply()["updated"], 0)

    def test_preserve_unit_edits_effect_and_twins_own_identities(self):
        twin = copy.deepcopy(self.doc.members["jp"])
        twin.id, twin.layer = "glow", 1
        twin.runs = [Run("あいう", ins="i_glow", a=0, b=3)]
        self.doc.members[twin.id] = twin
        self.doc.groups[0].members.append(twin.id)
        edits = [{"op": "merge", "at": kara.boundary_key(self.doc, self.doc.members["jp"], 1)}]
        beats = kara.derive(self.doc, self.doc.members["jp"], "mora", edits)
        for i, b in enumerate(beats):
            b.t0, b.t1, b.by = 1000 + i * 400, 1400 + i * 400, "onset"
        self.ed.set_karaoke("jp", beats, state="suggested", unit="mora", effect={"preset": "ko"}, edits=edits, twins=True)
        self.result.beats = self.result.beats[:2]
        report = self.apply()
        self.assertEqual(report["updated"], 2)
        self.assertEqual(self.view().unit, "mora")
        self.assertEqual(self.view().kind, "ko")
        self.assertEqual(kara.edits_of(self.doc, self.doc.members["jp"]), edits)
        self.assertEqual([(b.t0, b.t1, b.flags) for b in self.view().beats],
                         [(b.t0, b.t1, b.flags) for b in self.view("glow").beats])
        self.assertEqual(self.view("glow").beats[0].core[0][1], "i_glow")

    def test_links_follow_and_member_ranges_never_shrink(self):
        orig = self.view().beats
        cn = self.view("cn").beats
        cn[0].link["start"] = kara.beat_link(orig[0], "start")
        cn[-1].link["end"] = kara.beat_link(orig[-1], "end")
        for b in cn:
            b.by = "tap"
        self.ed.set_karaoke("cn", cn)
        self.doc.members["cn"].start = 800
        self.doc.members["cn"].end = 3000
        report = self.apply()
        self.assertEqual((self.view("cn").beats[0].t0, self.view("cn").beats[-1].t1), (900, 2300))
        self.assertEqual((self.doc.members["cn"].start, self.doc.members["cn"].end), (800, 3000))
        self.assertEqual(report["widened_lines"], 1)

    def test_named_s4_hook_uses_same_transaction(self):
        def hook(tx, gids, *, follow_notes):
            self.assertEqual(gids, {"g"})
            self.assertTrue(follow_notes)
            tx.member("cn").extra["test_redistributed"] = True
            return {"updated": 1}
        with patch.object(self.ed, "_redistribute_auto_karaoke_translations", side_effect=hook):
            report = self.apply(translations=True)
        self.assertEqual(report["translation_report"], {"updated": 1})
        self.assertEqual(len(self.ed.history.done), 1)
        self.ed.history.undo()
        self.assertNotIn("test_redistributed", self.doc.members["cn"].extra)

    def test_single_adoption_label_unit_and_explicit_overwrite(self):
        beats = self.view().beats
        for b, values in zip(beats, self.result.beats):
            b.t0, b.t1, b.by = values[0], values[1], "tap"
            b.flags.append("check")
        self.ed.set_karaoke("jp", beats)
        report = self.ed.adopt_auto_karaoke("jp", beats, "mora", False)
        self.assertEqual(report["updated"], 1)
        self.assertEqual(self.ed.history.done[-1].label, "采用自动 K 轴")
        self.assertEqual(self.view().state, "suggested")
        self.assertEqual(self.view().unit, "mora")
        self.assertTrue(all("check" in b.flags for b in self.view().beats))

    def test_bad_count_prevalidated_no_partial_writes(self):
        before = copy.deepcopy(document_to_json(self.doc))
        self.result.beats.pop()
        with self.assertRaises(ValueError):
            self.apply()
        self.assertEqual(document_to_json(self.doc), before)
        self.assertEqual(len(self.ed.history.done), 0)


if __name__ == "__main__":
    unittest.main()
