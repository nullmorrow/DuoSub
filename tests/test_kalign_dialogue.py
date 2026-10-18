"""Model-free dialogue fixtures and parity against the evaluated algorithms."""

from __future__ import annotations

import ast
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.kalign import dialogue as d, tokens, viterbi
from subkit.kalign.song import ModelEmission
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.reading.engine import Morph


def reference(filename, names, **globals_):
    """Load actual pure functions without the eval scripts' media dependencies."""
    path = ROOT / "tools/eval" / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = dict(np=np, tokens=tokens, viterbi=viterbi, WINDOW_MS=1000, SHIFT={"mms": -10, "kana": -50}, **globals_)
    exec(compile(tree, str(path), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


def line(kana="あい", start=200, end=640, mid="jp"):
    return d.DialogueLine("g", mid, start, end, kana, "sig")


def edge(start=1000, end=1500, conf=.9, mid="jp"):
    return d.DialogueEdge(mid, start, end, conf)


def emission(name="kana", *, frames=100, blank=0, frame_ms=20., peaks=((10, 1), (30, 2))):
    vocab = {"a" if name == "mms" else "あ": 1, "i" if name == "mms" else "い": 2}
    if blank:
        vocab = {key: val - 1 for key, val in vocab.items()}
    star = 3 if name == "mms" else None
    probs = np.full((frames, 4 if star is not None else 3), .005)
    probs[:, blank] = .97
    for at, token in peaks:
        probs[at] = .005
        probs[at, token] = .97
    if star is not None:
        probs[0, :] = probs[-1, :] = .005
        probs[0, star] = probs[-1, star] = .97
    probs /= probs.sum(axis=1, keepdims=True)
    return ModelEmission(name, np.log(probs).astype(np.float32), frame_ms, vocab, blank, star)


class SpokenReadings(unittest.TestCase):
    def setUp(self):
        self.particles = patch.object(tokens, "particle_positions_many", return_value=[set()]).start()
        self.addCleanup(patch.stopall)

    def document(self, text):
        member = Member("jp", "g", "ja", "original", "dialogue", 100, 200,
                        runs=[Run(text, ins="ins", a=0, b=len(text))])
        return Document("ep", groups=[Group("g", members=["jp"])], members={"jp": member}), member

    def ruby(self, doc, start, end, text, *, status="manual"):
        aid = "ruby" + str(start)
        base = doc.members["jp"].runs[0].text[start:end]
        doc.annotations[aid] = Annotation(aid, "ruby", "jp", [{"ins": "ins", "a": start, "b": end}],
                                          base, text, status=status, parts=[[0, len(base), text]])
        return doc.annotations[aid]

    def test_ruby_kana_katakana_latin_and_punctuation(self):
        doc, member = self.document("「私」は カッコいい、ABC！")
        self.ruby(doc, 1, 2, "ワタクシ")
        self.particles.return_value = [{3}]
        before = copy.deepcopy(asdict(doc))
        kana, signature = d.spoken_kana(doc, member, Mock())
        self.assertEqual(kana, "わたくしわかっこいいabc")
        self.assertEqual(len(signature), 64)
        self.particles.assert_called_once_with(["「私」は カッコいい、ABC！"])
        self.assertEqual(asdict(doc), before)

    def test_plain_particles_use_visible_offsets_after_leading_punctuation(self):
        doc, member = self.document("「は」へ はな")
        self.particles.return_value = [{1, 3}]
        self.assertEqual(d.spoken_kana(doc, member)[0], "わえはな")

    def test_explicit_ruby_particles_are_never_rewritten(self):
        doc, member = self.document("はへ")
        self.ruby(doc, 0, 1, "は")
        self.ruby(doc, 1, 2, "へ")
        self.particles.return_value = [{0, 1}]
        self.assertEqual(d.spoken_kana(doc, member)[0], "はへ")

    def test_no_reading_compound_and_okurigana_fallback(self):
        doc, member = self.document("今日へ行く。")
        analyzer = Mock()
        analyzer.tokenize.return_value = [Morph("今日", 0, 2, "キョウ", ("名詞",), "今日"),
                                          Morph("へ", 2, 3, "へ", ("助詞",), "へ"),
                                          Morph("行く", 3, 5, "イク", ("動詞",), "行く"),
                                          Morph("。", 5, 6, "。", ("補助記号",), "。")]
        self.particles.return_value = [{2}]
        self.assertEqual(d.spoken_kana(doc, member, analyzer)[0], "きょうえいく")
        analyzer.tokenize.assert_called_once_with("今日へ行く。")

    def test_hidden_ruby_uses_fallback_but_visible_ruby_does_not(self):
        doc, member = self.document("私")
        ruby = self.ruby(doc, 0, 1, "わたくし", status="hidden")
        analyzer = Mock()
        analyzer.tokenize.return_value = [Morph("私", 0, 1, "わたし", ("名詞",), "私")]
        self.assertEqual(d.spoken_kana(doc, member, analyzer)[0], "わたし")
        ruby.status = "manual"
        self.assertEqual(d.spoken_kana(doc, member, analyzer)[0], "わたくし")
        self.assertEqual(analyzer.tokenize.call_count, 1)

    def test_mixed_ruby_and_unannotated_spans_preserve_readings(self):
        doc, member = self.document("君は今日へ")
        self.ruby(doc, 0, 1, "あなた")
        analyzer = Mock()
        analyzer.tokenize.return_value = [Morph("は", 0, 1, "は", ("助詞",), "は"),
                                          Morph("今日", 1, 3, "きょう", ("名詞",), "今日"),
                                          Morph("へ", 3, 4, "へ", ("助詞",), "へ")]
        self.particles.return_value = [{1, 4}]
        self.assertEqual(d.spoken_kana(doc, member, analyzer)[0], "あなたわきょうえ")
        analyzer.tokenize.assert_called_once_with("は今日へ")

    def test_ruby_with_attached_long_vowel_and_small_kana_is_retained(self):
        doc, member = self.document("兄ーちゃん")
        self.ruby(doc, 0, 1, "にい")
        self.assertEqual(d.spoken_kana(doc, member)[0], "にいーちゃん")

    def test_signature_changes_on_text_and_reading_but_not_timing(self):
        doc, member = self.document("私！")
        ruby = self.ruby(doc, 0, 1, "わたし")
        first = d.spoken_kana(doc, member)[1]
        member.start += 10
        self.assertEqual(d.spoken_kana(doc, member)[1], first)
        member.runs[0].text = "私？"
        second = d.spoken_kana(doc, member)[1]
        self.assertNotEqual(first, second)
        ruby.text = "わたくし"
        ruby.parts[0][2] = "わたくし"
        self.assertNotEqual(second, d.spoken_kana(doc, member)[1])

    def test_empty_and_symbols_do_not_load_reading_engine(self):
        for text in ("", " ！？123"):
            doc, member = self.document(text)
            analyzer = Mock()
            self.assertEqual(d.spoken_kana(doc, member, analyzer)[0], "")
            analyzer.tokenize.assert_not_called()

    def test_default_analyzer_is_lazy_and_borrowed(self):
        from subkit.reading.engine import Analyzer
        doc, member = self.document("猫")
        analyzer = Mock()
        analyzer.tokenize.return_value = [Morph("猫", 0, 1, "ねこ", ("名詞",), "猫")]
        with patch.object(Analyzer, "get", return_value=analyzer) as get:
            self.assertEqual(d.spoken_kana(doc, member)[0], "ねこ")
        get.assert_called_once_with()


class DialogueNumerics(unittest.TestCase):
    def test_json_roundtrips_and_flags_do_not_share_storage(self):
        value = line("かなABC")
        self.assertEqual(d.DialogueLine.from_json(value.to_json()), value)
        value = d.DialogueEdge("jp", 101, 999, .87, ["disagree", "one_model"])
        self.assertEqual(d.DialogueEdge.from_json(value.to_json()), value)
        self.assertEqual(json.loads(value.to_json())["flags"], value.flags)
        a, b = edge(), edge()
        a.flags.append("no_text")
        self.assertEqual(b.flags, [])

    def test_real_ctc_first_last_frames_model_shifts_and_confidence(self):
        for name, start in (("mms", 190), ("kana", 150)):
            model = emission(name)
            result = d.align_dialogue([line()], [model], [])[0]
            self.assertEqual((result.speech_start, result.speech_end), (start, 620))
            self.assertAlmostEqual(result.conf, np.exp(model.log_probs[10:31][:, [1, 2]]).max(axis=1).mean())
            self.assertEqual(result.flags, ["one_model"])

    def test_two_models_fuse_without_one_model_flag(self):
        result = d.align_dialogue([line()], [emission("mms"), emission()], [])[0]
        self.assertEqual((result.speech_start, result.speech_end, result.flags), (170, 620, []))

    def test_clip_offset_shift_override_and_noninteger_frame_rounding(self):
        model = emission(frame_ms=20.25)
        model.shift_ms = -12.25
        result = d.align_dialogue([line(start=1200, end=1800)], [model], [], clip_ms=1000)[0]
        self.assertEqual((result.speech_start, result.speech_end), (1190, 1628))

    def test_pseudo_star_nonzero_blank_and_input_unchanged(self):
        model = emission(blank=2, peaks=((10, 0), (30, 1)))
        before = model.log_probs.copy()
        real = viterbi.align
        def align(lp, beats, windows, blank):
            self.assertEqual(blank, 2)
            np.testing.assert_allclose(lp[:, -1], before[:, :2].max(axis=1) - 1)
            self.assertEqual(beats, [[3], [0, 1], [3]])
            self.assertEqual(windows, [(0, 99)] * 3)
            return real(lp, beats, windows, blank=blank)
        with patch.object(viterbi, "align", side_effect=align):
            result = d.align_dialogue([line(end=2000)], [model], [])[0]
        self.assertEqual(result.speech_start, 150)
        np.testing.assert_array_equal(model.log_probs, before)

    def test_mms_own_star_window_truncation_and_inclusive_end(self):
        model = emission("mms", frames=200)
        def align(lp, beats, windows, blank):
            self.assertEqual(lp.shape, (51, 4))  # [int(205/20), int(1205/20)]
            self.assertEqual(beats, [[3], [1, 2], [3]])
            self.assertEqual(windows, [(0, 50)] * 3)
            return np.array([0, 1, 49]), np.array([0, 2, 50]), 0.
        with patch.object(viterbi, "align", side_effect=align):
            result = d.align_dialogue([line(start=705, end=705)], [model], [], window_ms=500)[0]
        self.assertEqual((result.speech_start, result.speech_end), (210, 260))

    def test_rms_150ms_strict_threshold_and_first_qualifying_run(self):
        rms = np.zeros(300)
        rms[64:78] = -40  # 140 ms does not qualify
        rms[80:96] = -20  # exactly percentile80 - 20 does not qualify
        rms[100:115] = -40
        rms[120:140] = -40
        result = d.align_dialogue([line()], [emission()], rms)[0]
        self.assertEqual(result.speech_end, 1000)

    def test_rms_800ms_search_boundary_and_clip_offset(self):
        for begin, expected in ((127, 1270), (128, 620)):
            rms = np.zeros(300)
            rms[begin:begin + 15] = -40
            result = d.align_dialogue([line(start=1200, end=1640)], [emission()], rms, 1000)[0]
            self.assertEqual(result.speech_end, 1000 + expected)

    def test_silence_does_not_end_before_fractional_last_token(self):
        model = emission(frame_ms=20.25)
        rms = np.zeros(300)
        rms[62:77] = -40
        result = d.align_dialogue([line()], [model], rms)[0]
        self.assertEqual(result.speech_end, 628)

    def test_empty_targets_short_windows_and_impossible_paths_abstain(self):
        models = [emission("mms"), emission()]
        self.assertEqual(d.align_dialogue([line("！？漢字"), line("")], models, []), [None, None])
        self.assertEqual(d.align_dialogue([line(start=9000, end=10000)], models, []), [None])
        self.assertEqual(d.align_dialogue([line()], [emission(frames=5, peaks=())], []), [None])
        with patch.object(viterbi, "align", side_effect=ValueError("no valid path")):
            self.assertEqual(d.align_dialogue([line()], models, []), [None])
        self.assertEqual(d.align_dialogue([line()], [], []), [None])

    def test_latin_targets_only_reach_mms(self):
        result = d.align_dialogue([line("AI")], [emission("mms"), emission()], [])[0]
        self.assertEqual((result.speech_start, result.flags), (190, ["one_model"]))

    def test_fusion_boundaries_floor_mean_tie_and_nonmutation(self):
        a, b = edge(101, 999, .4), edge(302, 1200, .8)
        result = d.fuse(a, b)
        self.assertEqual((result.speech_start, result.speech_end, result.conf, result.flags), (302, 1200, .8, ["disagree"]))
        b.speech_start, b.speech_end = 301, 1199
        result = d.fuse(a, b)
        self.assertEqual((result.speech_start, result.speech_end, result.conf, result.flags), (201, 1099, .8, []))
        b.speech_start, b.conf = 302, .4
        self.assertEqual(d.fuse(a, b).speech_start, 101)
        self.assertEqual(d.fuse(None, a).flags, ["one_model"])
        self.assertEqual(d.fuse(a, None).flags, ["one_model"])
        self.assertEqual((a.flags, b.flags), ([], []))
        self.assertIsNone(d.fuse(None, None))

    def test_alignment_and_fusion_parity_with_eval(self):
        ref = reference("dialogue_align_eval.py", ["align_line", "fuse"])
        rng = np.random.default_rng(42)
        for _ in range(20):
            a, z = sorted(rng.integers(0, 1500, size=2))
            current = line(start=int(a), end=int(z))
            rms = np.zeros(300)
            silent = int(rng.integers(50, 170))
            rms[silent:silent + 15] = -40
            expected = []
            for name in ("mms", "kana"):
                model = emission(name)
                model.log_probs = rng.dirichlet(np.ones(model.log_probs.shape[1]), len(model.log_probs)).astype(np.float32)
                model.log_probs = np.log(model.log_probs)
                value = ref.align_line(dict(name=name, lp=model.log_probs, frame_ms=model.frame_ms,
                                            vocab=model.vocab, blank=model.blank, star=model.star),
                                       current.kana, current.start, current.end, rms, -20.)
                result = d.align_dialogue([current], [model], rms)[0]
                actual = None if result is None else (result.speech_start, result.speech_end, result.conf)
                self.assertEqual(actual, value)
                expected.append(value)
            values = [None if value is None else edge(*value) for value in expected]
            fused = d.fuse(*values)
            self.assertEqual(None if fused is None else (fused.speech_start, fused.speech_end, fused.conf), ref.fuse(*expected))

    def test_malformed_emissions_fail_before_alignment(self):
        model = emission()
        for field, value in (("frame_ms", 0), ("blank", 9), ("name", "unknown"), ("star", 0)):
            bad = copy.copy(model)
            setattr(bad, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                d.align_dialogue([line()], [bad], [])
        with self.assertRaises(ValueError):
            d.align_dialogue([line()], [model], [np.nan])

    def test_fresh_import_is_qt_and_model_free(self):
        code = "\n".join(("import sys", f"sys.path.insert(0, {str(ROOT / 'src')!r})",
                          "from subkit.kalign import dialogue",
                          "heavy = ('torch', 'torchaudio', 'transformers', 'demucs', 'sudachipy', 'PySide6')",
                          "assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in heavy)"))
        proc = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", code], capture_output=True,
                              text=True, encoding="utf-8", timeout=20,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.assertEqual(proc.returncode, 0, proc.stderr)


class DialogueSuggestions(unittest.TestCase):
    def test_previous_modes_current_suggested_and_unclamped(self):
        edges = [edge(500, 900), edge(1300, 1700, mid="next")]
        times = [(0, 1600), (1700, 2200)]
        self.assertEqual(d.suggest_times(edges, times, []), [(335, 1095), (1640, 1950)])
        self.assertEqual(d.suggest_times(edges, times, [], previous="current"), [(335, 1095), (1640, 1950)])
        self.assertEqual(d.suggest_times(edges, times, [], previous="suggested"), [(335, 1095), (1135, 1950)])
        self.assertEqual(d.suggest_times(edges, times, [], previous=None), [(335, 1095), (1135, 1950)])

    def test_suggested_previous_falls_back_after_missing_or_collapsed_row(self):
        for first in (None, edge(1000, 500)):
            self.assertEqual(d.suggest_times([first, edge()], [(0, 1000), (1100, 1800)], [],
                                            previous="suggested"), [None, (1040, 1750)])
        with self.assertRaises(ValueError):
            d.suggest_times([], [], [], previous="invalid")

    def test_unclamped_mode_matches_reference_on_overlapping_rows(self):
        ref = reference("dialogue_tpp_eval.py", ["snap", "post_process"])
        times = [(900, 2300), (1100, 1800), (1600, 2100)]
        edges = [edge(), edge(1200, 1700, mid="two"), None]
        params = dict(chain=None, end_before=300, end_after=250, start_before=0,
                      start_after=0, keep_start=0, keep_end=0)
        expected = ref.post_process([dict(start=s, end=e) for s, e in times],
                       [None if e is None else (e.speech_start, e.speech_end) for e in edges],
                       [1300, 1800], 165, 250, 40, params)
        self.assertEqual(d.suggest_times(edges, times, [1300, 1800], previous=None), expected)

    def test_default_padding_and_no_start_snapping(self):
        self.assertEqual(d.suggest_times([edge()], [(900, 1800)], [900, 1800]), [(835, 1800)])
        self.assertEqual(d.suggest_times([edge(50, 300)], [(0, 1000)], []), [(0, 550)])

    def test_keyframe_window_bounds_inclusive_outside_and_ties(self):
        for keyframe, expected in ((1450, 1450), (2000, 2000), (1449, 1750), (2001, 1750)):
            self.assertEqual(d.suggest_times([edge()], [(500, 2000)], [keyframe]), [(835, expected)])
        self.assertEqual(d.suggest_times([edge()], [(500, 2000)], [1700, 1800]), [(835, 1700)])

    def test_next_suggested_start_overlap_clamp_and_chained_end_not_snapped(self):
        edges = [edge(), edge(1800, 2300, mid="next")]
        times = [(900, 1400), (1700, 2500)]
        result = d.suggest_times(edges, times, [1500, 1600, 2000])
        self.assertEqual(result, [(835, 1595), (1635, 2550)])

    def test_next_current_start_when_edge_missing_and_snap_rejected(self):
        times = [(900, 1600), (1800, 2300)]
        self.assertEqual(d.suggest_times([edge(), None], times, [1900]), [(835, 1750), None])
        self.assertEqual(d.suggest_times([edge(end=1700), None], times, [1700]), [(835, 1760), None])

    def test_previous_current_end_plus_gap_even_without_edge(self):
        times = [(0, 1000), (1100, 1800)]
        self.assertEqual(d.suggest_times([None, edge()], times, []), [None, (1040, 1750)])
        self.assertEqual(d.suggest_times([None, edge()], [(0, 1800), times[1]], []), [None, None])

    def test_no_edges_and_length_mismatch(self):
        self.assertEqual(d.suggest_times([], [], []), [])
        self.assertEqual(d.suggest_times([None, None], [(0, 100), (200, 300)], []), [None, None])
        with self.assertRaises(ValueError):
            d.suggest_times([edge()], [], [])

    def test_random_episode_parity_with_eval_and_previous_safety(self):
        ref = reference("dialogue_tpp_eval.py", ["snap", "post_process"])
        rng = np.random.default_rng(713)
        params = dict(chain=None, end_before=300, end_after=250, start_before=0, start_after=0, keep_start=0, keep_end=0)
        for trial in range(20):
            starts = np.cumsum(rng.integers(1800, 3000, 50)).tolist()
            times = [(int(start), int(start + rng.integers(500, 1800))) for start in starts]
            edges = [None if rng.random() < .2 else edge(int(start + rng.integers(-300, 400)),
                                                       int(end + rng.integers(-400, 600)), mid=str(i))
                     for i, (start, end) in enumerate(times)]
            ks = np.sort(rng.integers(0, times[-1][1] + 1000, 300))
            raw = ref.post_process([dict(start=s, end=e) for s, e in times],
                                   [None if e is None else (e.speech_start, e.speech_end) for e in edges],
                                   ks, 165, 250, 40, params)
            expected = []
            for i, value in enumerate(raw):
                if value is None:
                    expected.append(None)
                else:
                    s, e = value
                    s = max(s, times[i - 1][1] + 40) if i else s
                    expected.append((s, e) if e > s else None)
            with self.subTest(trial=trial):
                self.assertEqual(d.suggest_times(edges, times, ks), expected)


class DialogueHabit(unittest.TestCase):
    def samples(self, count=20):
        edges, rows = [], []
        for i in range(count):
            start = i * 3000 + 1000
            edges.append(edge(start, start + 500, mid=str(i)))
            rows.append(dict(mid=str(i), start=start - 165, end=start + 750,
                             time_src=dict(start="manual", end="manual")))
        return edges, rows

    def test_19_20_threshold_and_stored_fallback(self):
        edges, rows = self.samples()
        self.assertEqual(d.learn_habit(edges, rows), (165, 250, "habit", 20, 20))
        self.assertEqual(d.learn_habit(edges, rows[:19]), (165, 250, "default", 19, 19))
        self.assertEqual(d.learn_habit(edges, rows[:19], (80, 90)), (80, 90, "stored", 19, 19))

    def test_both_edges_need_minimum_and_adopted_autotime_do_not_train(self):
        edges, rows = self.samples()
        rows[-1]["time_src"]["end"] = "adopted"
        rows[0]["time_src"]["start"] = "autotime"
        self.assertEqual(d.learn_habit(edges, rows, (0, 100)), (0, 100, "stored", 19, 19))
        rows[0]["time_src"]["start"] = "manual"
        self.assertEqual(d.learn_habit(edges, rows), (165, 250, "default", 20, 19))

    def test_chained_ends_50ms_inclusive_excluded_next_row_without_edge(self):
        edges = [edge(1000, 1500)]
        for gap, n_end in ((-1, 0), (0, 0), (50, 0), (51, 1)):
            rows = [dict(mid="jp", start=835, end=1750), dict(mid="missing", start=1750 + gap, end=2300)]
            self.assertEqual(d.learn_habit(edges, rows, minimum=1)[-2:], (1, n_end))

    def test_explicit_next_start_episode_boundaries_and_partial_manual_edges(self):
        rows = [dict(mid="jp", start=800, end=1800, next_start=None),
                dict(mid="other", start=0, end=100)]
        self.assertEqual(d.learn_habit([edge()], rows, minimum=1), (200, 300, "habit", 1, 1))
        rows[0].update(start=None, next_start=1830)
        self.assertEqual(d.learn_habit([edge()], rows, minimum=1), (165, 250, "default", 0, 0))

    def test_medians_truncate_and_clip_both_directions(self):
        edges, rows = self.samples(2)
        rows[0].update(start=834, end=1750)
        self.assertEqual(d.learn_habit(edges, rows, minimum=2), (165, 250, "habit", 2, 2))
        for onset, offset, expected in ((1000, 1000, (600, 800)), (-1000, -1000, (-100, -200))):
            for e, row in zip(edges, rows):
                row.update(start=e.speech_start - onset, end=e.speech_end + offset, next_start=None)
            self.assertEqual(d.learn_habit(edges, rows, minimum=2)[:2], expected)

    def test_member_matching_missing_edges_and_empty_input(self):
        edges, rows = self.samples(2)
        self.assertEqual(d.learn_habit(list(reversed(edges)), rows, minimum=2), (165, 250, "habit", 2, 2))
        self.assertEqual(d.learn_habit([None, edges[0]], rows, minimum=2)[-2:], (1, 1))
        self.assertEqual(d.learn_habit([], []), (165, 250, "default", 0, 0))

    def test_gold_learning_parity_with_chain_eval(self):
        ref = reference("dialogue_chain_eval.py", ["learn"], ev=SimpleNamespace(DEFAULT_PADDING=(165, 250)))
        rng = np.random.default_rng(86)
        edges, rows = self.samples(100)
        for i, (e, row) in enumerate(zip(edges, rows)):
            row["start"] = e.speech_start - int(rng.integers(-120, 700))
            row["end"] = e.speech_end + int(rng.integers(-250, 900))
            if i % 3 == 0 and i + 1 < len(rows):
                row["end"] = rows[i + 1]["start"] - 40
            row.pop("time_src")
        data = dict(gold={"A": rows}, mapping={"A": list(range(len(rows)))},
                    edges={"A": [(e.speech_start, e.speech_end) for e in edges]})
        self.assertEqual(d.learn_habit(edges, rows)[:2], ref.learn([data], "A")[:2])


if __name__ == "__main__":
    unittest.main()
