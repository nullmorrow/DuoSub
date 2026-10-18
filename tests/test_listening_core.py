"""Listening proposals and explicit adoption; local fixtures, no ASR or separation models."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subkit.core.listening import adopt, target_signature, unavailable_reason
from subkit.listening import Candidate, ListeningRequest, ListeningResult, assemble, clip_window, generate, model_file
from subkit.model.document import Group, Member, Run
from subkit.model.serialize import document_from_json, document_to_json
from subkit.model.validate import validate
from subkit.proc import CancelledError
from tests.test_text_edit_rendering import fixture, snapshot


def request(**changes):
    return replace(ListeningRequest("never-decoded.mkv", 2, 1000, 2000, padding_ms=1000), **changes)


def segment(text="風船はいかがですか", start=1.1, end=1.9, **changes):
    return dict(dict(text=text, start=start, end=end, avg_logprob=-.3, no_speech_prob=.05, words=[]), **changes)


def payload(runs, **changes):
    return dict(dict(clips={"original": "original.wav", "side": "side.wav"}, runs=runs,
                     device="cpu", duration_ms=3000, diagnostics=[]), **changes)


def run(*segments, source="原声混音", clip="original", distorted=False):
    return dict(source=source, clip=clip, distorted=distorted, segments=list(segments))


class ListeningAssembly(unittest.TestCase):
    def test_clip_window_clamps_zero_and_ensures_minimum_context(self):
        self.assertEqual(clip_window(request()), (0, 3000))
        self.assertEqual(clip_window(request(start_ms=100, end_ms=200, padding_ms=0)), (0, 3000))
        self.assertEqual(clip_window(request(start_ms=10000, end_ms=12000, padding_ms=0)), (9500, 12500))
        self.assertEqual(clip_window(request(start_ms=10000, end_ms=12000, padding_ms=5000)), (5000, 17000))

    def test_invalid_time_values_are_rejected_without_decoding(self):
        for change in ({"start_ms": -1}, {"end_ms": 1000}, {"end_ms": 1001.0},
                       {"start_ms": True}, {"end_ms": 61001}, {"padding_ms": -1},
                       {"padding_ms": 5001}, {"padding_ms": False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                clip_window(request(**change))

    def test_nearby_word_excerpt_uses_absolute_media_clock(self):
        words = [{"word": "前景の声", "start": .1, "end": .6},
                 {"word": "風船は", "start": 1.1, "end": 1.4},
                 {"word": "いかがですか", "start": 1.4, "end": 1.9}]
        data = payload([run(segment("前景の声 風船はいかがですか", .1, 2.5, words=words))])
        result = assemble(request(start_ms=11000, end_ms=12000), data, Path("owned-output"))
        selected = next(c for c in result.candidates if c.text == "風船はいかがですか")
        self.assertEqual((result.window_start_ms, result.window_end_ms), (10000, 13000))
        self.assertEqual((selected.start_ms, selected.end_ms), (11100, 11900))
        self.assertTrue(selected.evidence[0]["partial"])
        self.assertTrue(any("逐词时间" in text for text in selected.warnings))
        self.assertEqual(selected.audio, str(Path("owned-output") / "original.wav"))

    def test_duplicate_normalized_words_merge_evidence_but_not_different_occurrences(self):
        data = payload([run(segment("風船 はいかがですか？"), segment("風船はいかがですか", .1, .4)),
                        run(segment("風船はいかがですか！"), source="左声道")])
        result = assemble(request(), data, Path("out"))
        self.assertEqual(len(result.candidates), 2)
        near, previous = result.candidates
        self.assertEqual(near.sources, ["原声混音", "左声道"])
        self.assertEqual(len(near.evidence), 2)
        self.assertEqual((previous.start_ms, previous.end_ms), (100, 400))
        self.assertTrue(any("相邻一句" in text for text in previous.warnings))

    def test_nonfinite_far_empty_and_damaged_common_hallucination_are_discarded(self):
        bad = [segment("遠い", 6, 7), segment("！？　"), segment("bad", math.nan, 2),
               segment("bad", 1, 2, avg_logprob=math.inf)]
        data = payload([run(*bad), run(segment("ご視聴ありがとうございました"),
                                       source="侧向消声", clip="side", distorted=True)])
        result = assemble(request(), data, Path("out"))
        self.assertEqual(result.candidates, [])

    def test_low_evidence_and_damaged_hypotheses_are_marked_and_ranked_after_original(self):
        data = payload([run(segment("別の候補", avg_logprob=-2, no_speech_prob=.9), source="侧向消声", clip="side", distorted=True),
                        run(segment())])
        result = assemble(request(), data, Path("out"))
        self.assertEqual(result.candidates[0].text, "風船はいかがですか")
        warning = "；".join(result.candidates[1].warnings)
        self.assertIn("失真", warning)
        self.assertIn("依据较弱", warning)

    def test_candidates_are_limited_and_audio_end_is_clamped_to_real_duration(self):
        rows = [segment(f"候補{i}", 1.05, 1.8) for i in range(30)]
        data = payload([run(*rows)], duration_ms=1500)
        result = assemble(request(), data, Path("out"))
        self.assertEqual(len(result.candidates), 24)
        self.assertTrue(all(c.end_ms == 1500 for c in result.candidates))
        self.assertEqual(result.window_end_ms, 1500)

    def test_model_lookup_explicit_local_directory_does_not_download(self):
        with tempfile.TemporaryDirectory(prefix="listening-model-lookup-") as root:
            with patch.dict(os.environ, {"SUBKIT_ASR_MODEL_DIR": root}):
                with self.assertRaisesRegex(ValueError, "不会自动下载"):
                    model_file("medium")
                local = Path(root) / "medium.pt"
                local.write_bytes(b"not loaded by this test")
                self.assertEqual(model_file("medium"), local.resolve())
                with self.assertRaisesRegex(ValueError, "medium 或 large-v3"):
                    model_file("../../other")

    def test_missing_media_and_pre_cancelled_runner_do_not_start_inference(self):
        runner = Mock()
        with tempfile.TemporaryDirectory(prefix="listening-no-inference-") as root:
            with self.assertRaisesRegex(ValueError, "找不到这一集"):
                generate(request(video=str(Path(root) / "missing.mkv")), Path(root) / "out", runner=runner, progress=Mock())
            runner.run.assert_not_called()
            video = Path(root) / "video.mkv"
            video.write_bytes(b"not decoded")
            runner.check_cancelled.side_effect = CancelledError
            with patch("subkit.listening.model_file", return_value=Path(root) / "medium.pt"):
                with self.assertRaises(CancelledError):
                    generate(request(video=str(video)), Path(root) / "out", runner=runner, progress=Mock())
            runner.run.assert_not_called()

    def test_existing_output_file_is_never_overwritten_or_decoded(self):
        runner = Mock()
        with tempfile.TemporaryDirectory(prefix="listening-existing-output-") as root:
            video = Path(root) / "video.mkv"
            video.write_bytes(b"identity only")
            output = Path(root) / "out"
            output.mkdir()
            original = output / "side.wav"
            original.write_bytes(b"existing artifact")
            with patch("subkit.listening.model_file", return_value=Path(root) / "medium.pt"):
                with self.assertRaisesRegex(ValueError, "新的临时目录"):
                    generate(request(video=str(video)), output, runner=runner, progress=Mock())
            self.assertEqual(original.read_bytes(), b"existing artifact")
            runner.run.assert_not_called()


class ListeningAdoption(unittest.TestCase):
    def make(self):
        doc, sources, editor = fixture()
        cn = doc.members["m"]
        cn.start, cn.end = 1000, 2000
        cn.layer, cn.margins, cn.effect = 7, [1, 2, 3], "existing-effect"
        cn.overrides = {"raw": r"\fad(0,720)"}
        cn.extra["unknown_preserved"] = {"key": [1, 2]}
        group = doc.group("g")
        group.state, group.flags = "confirmed", ["marked"]
        group.origin = {"existing": True}
        candidate = Candidate("風船はいかがですか", 1100, 1900, ["原声混音"], "temporary/original.wav",
                              ["请试听"], [{"partial": False, "distorted": False}])
        result = ListeningResult([candidate], {"original": candidate.audio}, 0, 3000, "cpu", ["候选不是定论"],
                                 {"model": "large-v3", "torch": "test"})
        return doc, sources, editor, candidate, result

    def test_sign_adoption_preserves_chinese_attributes_and_source_and_creates_ruby(self):
        doc, sources, editor, candidate, result = self.make()
        before = snapshot(doc)
        old_member = copy.deepcopy(doc.members["m"])
        old_group = asdict(doc.group("g"))
        old_sources = {k: copy.deepcopy(s.events_json()) for k, s in sources.items()}
        mid = adopt(editor, "g", candidate, "風船はいかがですか？", 1050, 1950, result,
                    styles={"original/dialogue": "My.JP"})
        self.assertEqual(doc.members["m"], old_member)
        after_group = asdict(doc.group("g"))
        old_group.pop("members")
        after_group.pop("members")
        self.assertEqual(after_group, old_group)
        self.assertEqual({k: s.events_json() for k, s in sources.items()}, old_sources)
        jp = doc.members[mid]
        self.assertEqual((jp.lane, jp.lang, jp.role, jp.style, jp.layer, jp.start, jp.end),
                         ("original", "ja", "dialogue", "My.JP", 0, 1050, 1950))
        self.assertEqual(doc.group("g").members, [mid, "m"])
        self.assertTrue(all(r.ins and r.src is None and r.render_src is None for r in jp.runs))
        self.assertTrue(any(a.kind == "ruby" and a.member == mid and a.base == "風船" and a.text
                            for a in doc.annotations.values()))
        self.assertTrue(validate(doc, sources).ok)
        self.assertEqual(len(editor.history.done), 1)
        after = snapshot(doc)
        editor.history.undo()
        self.assertEqual(snapshot(doc), before)
        editor.history.redo()
        self.assertEqual(snapshot(doc), after)

    def test_evidence_is_saved_without_dead_temporary_paths_and_survives_roundtrip(self):
        doc, sources, editor, candidate, result = self.make()
        mid = adopt(editor, "g", candidate, "風船", 1000, 2000, result)
        evidence = doc.members[mid].extra["asr"]
        self.assertEqual(evidence["adopted_by"], "user")
        self.assertEqual(evidence["selected"]["text"], candidate.text)
        self.assertEqual(evidence["adopted_text"], "風船")
        self.assertNotIn("audio", evidence["selected"])
        self.assertNotIn("audio", evidence["candidates"][0])
        self.assertNotIn("temporary/", str(evidence))
        candidate.evidence[0]["partial"] = True
        result.metadata["model"] = "changed-after-adoption"
        self.assertFalse(evidence["selected"]["evidence"][0]["partial"])
        self.assertEqual(evidence["model"]["model"], "large-v3")
        self.assertEqual(snapshot(document_from_json(document_to_json(doc))), snapshot(doc))

    def test_existing_original_is_never_overwritten_or_duplicated(self):
        doc, sources, editor, candidate, result = self.make()
        mid = adopt(editor, "g", candidate, candidate.text, 1100, 1900, result)
        before = snapshot(doc)
        with self.assertRaisesRegex(ValueError, "不会覆盖"):
            adopt(editor, "g", candidate, "別の文字", 1100, 1900, result)
        self.assertEqual(snapshot(doc), before)
        self.assertEqual(len(editor.history.done), 1)
        self.assertEqual(doc.members[mid].text, candidate.text)

    def test_invalid_plain_speech_or_times_never_mutates_document_or_history(self):
        for text in ("", "　", r"{\pos(1,2)}風船", "風\x00船", r"風\N船", r"風\n船", "風\n船", "風\r船"):
            doc, sources, editor, candidate, result = self.make()
            before = snapshot(doc)
            with self.subTest(text=text), self.assertRaises(ValueError):
                adopt(editor, "g", candidate, text, 1000, 2000, result)
            self.assertEqual(snapshot(doc), before)
            self.assertEqual(editor.history.done, [])
        for start, end in ((-1, 2000), (1000, 3001), (2000, 1000), (1000, 1000), (True, 2000), (1000., 2000)):
            doc, sources, editor, candidate, result = self.make()
            before = snapshot(doc)
            with self.subTest(times=(start, end)), self.assertRaises(ValueError):
                adopt(editor, "g", candidate, "風船", start, end, result)
            self.assertEqual(snapshot(doc), before)
            self.assertEqual(editor.history.done, [])

    def test_manual_input_is_identified_and_declined_reading_generation_is_atomic(self):
        doc, sources, editor, candidate, result = self.make()
        manual = Candidate("風船", 1000, 2000, ["manual"], candidate.audio)
        result.candidates = []
        mid = adopt(editor, "g", manual, "風船", 1000, 2000, result)
        self.assertEqual(doc.members[mid].extra["asr"]["selected"]["sources"], ["manual"])
        editor.history.undo()
        before = snapshot(doc)
        editor.reading_engine_factory = Mock(side_effect=ValueError("reading initialization failed"))
        with self.assertRaisesRegex(ValueError, "reading initialization"):
            adopt(editor, "g", candidate, "風船", 1000, 2000, result)
        self.assertEqual(snapshot(doc), before)
        self.assertEqual(editor.history.done, [])

    def test_availability_includes_sign_and_rejects_song_multiselect_non_japanese_or_comment_only(self):
        doc, sources, editor, candidate, result = self.make()
        self.assertEqual(unavailable_reason(doc, ["g"], {"original": "ja"}), "")
        self.assertIn("只选择一组", unavailable_reason(doc, ["g", "g"], {"original": "ja"}))
        self.assertIn("日语", unavailable_reason(doc, ["g"], {"original": "en"}))
        self.assertIn("找不到", unavailable_reason(doc, ["unknown"], {"original": "ja"}))
        doc.group("g").kind = "song"
        self.assertIn("对白和屏幕字", unavailable_reason(doc, ["g"], {"original": "ja"}))
        doc.group("g").kind = "sign"
        doc.members["m"].source_kind = "Comment"
        self.assertIn("没有可参考", unavailable_reason(doc, ["g"], {"original": "ja"}))

    def test_target_signature_detects_target_media_track_changes_not_unrelated_groups(self):
        doc, sources, editor, candidate, result = self.make()
        languages = {"original": "ja", "translation": "zh-Hant"}
        with tempfile.TemporaryDirectory(prefix="listening-signature-") as root:
            video = Path(root) / "video.mkv"
            video.write_bytes(b"identity only, never decoded")
            signature = target_signature(doc, "g", video, 2, languages)
            other = Member("unrelated", "other", "ja", "original", "dialogue", 5000, 6000,
                           runs=[Run("別の台詞", ins="i_unrelated", b=4)])
            doc.members[other.id] = other
            doc.groups.append(Group("other", "dialogue", [other.id]))
            doc.revision += 1
            self.assertEqual(target_signature(doc, "g", video, 2, languages), signature)
            self.assertNotEqual(target_signature(doc, "g", video, 1, languages), signature)
            doc.members["m"].end += 1
            self.assertNotEqual(target_signature(doc, "g", video, 2, languages), signature)
            doc.members["m"].end -= 1
            video.write_bytes(b"changed media identity")
            self.assertNotEqual(target_signature(doc, "g", video, 2, languages), signature)


if __name__ == "__main__":
    unittest.main()
