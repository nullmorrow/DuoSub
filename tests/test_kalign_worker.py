"""Worker/cache contract tests without downloads, model loading or user media."""
from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit import proc
from subkit.kalign import cache, dialogue, models, song, worker
from tools.eval import kalign_eval
from tools.eval import dialogue_product_eval as product_eval


class Tensor(np.ndarray):
    """NumPy test double for inference plumbing; never imports torch."""
    def to(self, device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def float(self):
        return self

    def std(self, *args, **kwargs):
        return np.asarray(self).std(ddof=1)


def tensor(value):
    return np.asarray(value).view(Tensor)


def fake_torch(cuda=False):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda), from_numpy=tensor,
                           inference_mode=nullcontext,
                           log_softmax=lambda x, dim: tensor(x - np.logaddexp.reduce(x, axis=dim, keepdims=True)))


class WorkerTests(unittest.TestCase):
    def setUp(self):
        # Deliberately keep scratch for the bridge; close all owned handles.
        self.root = Path(tempfile.mkdtemp(prefix="kalign-test-"))
        self.video = self.root / "fake-video.bin"
        self.video.write_bytes(b"fake media, never decoded by a real tool")

    def request(self):
        return worker.Request(str(self.video), 2, (1000, 3000), str(self.root / "cache"),
                              [song.LineInput("jp", 1200, 2500,
                                              [song.BeatInput("あ", True, True, (1300, 1400), 0)], "は")],
                              {"offset": "none", "last_end": "voice"})

    def emission(self, name):
        return song.ModelEmission(name, np.zeros((100, 3), np.float32), 20.,
                                  {"a" if name == "mms" else "あ": 1, "*": 2}, 0,
                                  2 if name == "mms" else None)

    def dialogue_request(self):
        return worker.Request(str(self.video), 2, None, str(self.root / "cache"),
                              [dialogue.DialogueLine("g", "jp", 100, 900, "あ", "signature"),
                               dialogue.DialogueLine("h", "empty", 1000, 1200, "", "empty-signature")],
                              {}, "dialogue")

    def test_dialogue_request_validation_and_song_default_compatibility(self):
        request = self.dialogue_request()
        self.assertEqual(worker.request_from_json(asdict(request)), request)
        for field, value in (("clip", [0, 1000]), ("lines", []), ("kind", "unknown")):
            data = asdict(request)
            data[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                worker.request_from_json(data)
        for start in (True, -1, 1.5):
            data = asdict(request)
            data["lines"][0]["start"] = start
            with self.assertRaises(ValueError):
                worker.request_from_json(data)
        data = asdict(self.request())
        del data["kind"]
        self.assertEqual(worker.request_from_json(data).kind, "song")

    def test_dialogue_whole_episode_cache_progress_and_realign_edits(self):
        request = self.dialogue_request()
        vocals = np.zeros((44100, 2), np.float32)
        events = []
        notify = lambda *args: events.append(args)
        with patch.object(proc, "run", return_value=SimpleNamespace(stdout=vocals.astype("<f4").tobytes())) as extract, \
                patch.object(models, "separate_vocals", return_value=vocals) as separate, \
                patch.object(models, "mms_emission", return_value=self.emission("mms")) as mms, \
                patch.object(models, "kana_emission", return_value=self.emission("kana")) as kana:
            result = worker.run_request(request, notify)
            self.assertEqual(worker.run_request(request, notify), result)
            request.lines[0].signature = "edited"
            edited = worker.run_request(request, notify)
        extract.assert_called_once()
        separate.assert_called_once()
        mms.assert_called_once()
        kana.assert_called_once()
        cmd = extract.call_args.args[0]
        self.assertNotIn("-ss", cmd)
        self.assertNotIn("-t", cmd)
        self.assertEqual(cmd[cmd.index("-map") + 1], "0:a:1")
        directory = Path(request.cache_dir) / "kalign" / cache.cache_key(self.video, 1, None)
        for name in ("mms", "kana"):
            self.assertEqual(cache.load_emission(directory, name).log_probs.dtype, np.float16)
        self.assertFalse((directory / "vocals.npy").exists())
        self.assertTrue((directory / "rms.npy").is_file())
        self.assertEqual(json.loads((directory / "dialogue.json").read_text(encoding="utf-8")), edited)
        self.assertEqual(edited[0]["signature"], "edited")
        self.assertEqual(result[1]["flags"], ["no_text"])
        self.assertIsNone(result[1]["speech_start"])
        self.assertIn(("align", 0, 2), events)
        self.assertIn(("align", 1, 2), events)
        self.assertIn(("align", 2, 2), events)

    def test_dialogue_cache_identity_ignores_rows_but_separates_song_track_and_models(self):
        key = cache.cache_key(self.video, 1, None)
        self.assertNotEqual(key, cache.cache_key(self.video, 1, (0, 1000)))
        self.assertNotEqual(key, cache.cache_key(self.video, 0, None))
        for field in ("KALIGN_VERSION", "KANA_REVISION", "MMS_ID", "DEMUCS_ID"):
            with patch.object(cache, field, "changed"):
                self.assertNotEqual(key, cache.cache_key(self.video, 1, None))

    def test_dialogue_main_publishes_list_result_and_done(self):
        path, output = self.root / "request.json", self.root / "result.json"
        path.write_text(json.dumps(asdict(self.dialogue_request())), encoding="utf-8")
        result = [dict(mid="jp", signature="sig", speech_start=100, speech_end=900, conf=.9, flags=[])]
        out = io.StringIO()
        with patch.object(worker, "run_request", return_value=result), redirect_stdout(out):
            self.assertEqual(worker.main([str(path), str(output)]), 0)
        self.assertEqual(worker.result_from_json(json.loads(output.read_text(encoding="utf-8"))), result)
        self.assertEqual(worker.parse_progress(out.getvalue())["stage"], "done")

    def test_product_eval_import_args_and_rough_rows_without_models(self):
        self.assertEqual(product_eval.parse_args(["--episodes", "6", "7"]).episodes, [6, 7])
        source = {"A": [dict(id="cn", start=100, end=900, text="译文"),
                         dict(id="missing", start=5000, end=6000, text="译文")],
                  "B": [dict(id="jp", start=200, end=1000, text="かな")]}
        with patch.object(dialogue, "spoken_kana", side_effect=lambda doc, m: (m.runs[0].text, m.id)), \
                patch.object(models, "mms_emission") as mms, patch.object(models, "kana_emission") as kana:
            lines = product_eval.build_lines(source, 6)
        self.assertEqual([l.kana for l in lines["A"]], ["かな", ""])
        self.assertEqual(lines["A"][0].start, 100)
        self.assertNotEqual(lines["A"][0].mid, lines["B"][0].mid)
        mms.assert_not_called()
        kana.assert_not_called()

    def test_product_eval_research_cache_skips_worker_and_preserves_metadata(self):
        directory = self.root / "research"
        directory.mkdir()
        self.assertIsNone(product_eval.research_audio(directory))
        for name in ("mms", "kana"):
            e = self.emission(name)
            np.savez(directory / f"{name}.npz", lp=e.log_probs.astype(np.float16), frame_ms=e.frame_ms,
                     vocab=json.dumps(e.vocab), blank=e.blank, star=-1 if e.star is None else e.star)
        np.save(directory / "vocals16k.npy", np.zeros(16000, np.float16))
        emissions, rms = product_eval.research_audio(directory)
        self.assertEqual([(e.name, e.blank, e.star, e.frame_ms) for e in emissions],
                         [("mms", 0, 2, 20.), ("kana", 0, None, 20.)])
        self.assertEqual(len(rms), 101)
        self.assertTrue(all(e.log_probs.dtype == np.float32 for e in emissions))

    def test_product_eval_loo_gold_gap_and_all_six_count_lines(self):
        data = {}
        for ep in (6, 7):
            lines = [dialogue.DialogueLine("g", f"{ep}/{i}", i*3000, i*3000+900, "あ", "sig") for i in range(20)]
            edges = [dialogue.DialogueEdge(l.mid, l.start+100, l.end-250, .9) for l in lines]
            rows = [dict(id=l.mid, start=l.start, end=l.end, text="あ") for l in lines]
            data[ep] = dict(lines={s: lines for s in ("A", "B")}, edges={s: edges for s in ("A", "B")},
                            source={s: rows for s in ("A", "B")}, gold={s: rows for s in ("A", "B")},
                            mapping={s: list(range(20)) for s in ("A", "B")}, keyframes=[], audio_source="synthetic")
        data[6]["edges"]["A"][0].speech_start = 9000
        self.assertEqual(product_eval.learn_leave_one_out(data, 6, "A"), (100, 250, "habit", 20, 20))
        data[7]["gold"]["A"][0]["end"] = 3000-50
        self.assertEqual(product_eval.learn_leave_one_out(data, 6, "A")[-2:], (20, 19))
        out = io.StringIO()
        with redirect_stdout(out):
            report = product_eval.evaluate(data)
        self.assertEqual(len(out.getvalue().splitlines()), 6)
        self.assertEqual(set(report["results"]), {f"{s}/previous={p}" for s in ("A", "B") for p in ("current", "suggested", "none")})
        self.assertTrue(all(v["start"]["denominator"] == 40 for v in report["results"].values()))

    def test_product_eval_episode_cached_and_worker_fallback_branches(self):
        source = {s: [dict(id=s, start=100, end=900, text="あ")] for s in ("A", "B")}
        lines = {s: [dialogue.DialogueLine(s, s, 100, 900, "あ", "sig")] for s in ("A", "B")}
        edge = dialogue.DialogueEdge("A", 120, 700, .8)
        records = [dict(mid=s, speech_start=120, speech_end=700, conf=.8, flags=[], signature="sig") for s in ("A", "B")]
        with patch.object(product_eval, "inputs", return_value=(self.video, "cn", "jp", "final")), \
                patch.object(product_eval, "read_rows", return_value=(source, source)), \
                patch.object(product_eval, "pair_rows", return_value=[0]), \
                patch.object(product_eval, "build_lines", return_value=lines), \
                patch.object(product_eval, "keyframes", return_value=[]), \
                patch.object(product_eval, "research_audio", return_value=([self.emission("mms"), self.emission("kana")], [])) as audio, \
                patch.object(dialogue, "align_dialogue", return_value=[edge, None]) as align, \
                patch.object(worker, "run_request", return_value=records) as run:
            cached = product_eval.episode_data(6, self.root)
            run.assert_not_called()
            align.assert_called_once()
            self.assertEqual(cached["edges"]["B"], [None])
            audio.return_value = None
            fallback = product_eval.episode_data(6, self.root)
            run.assert_called_once()
            request = run.call_args.args[0]
            self.assertEqual((request.kind, request.clip, request.audio_track), ("dialogue", None, 1))
            self.assertEqual([l.mid for l in request.lines], ["A", "B"])
            self.assertEqual(fallback["audio_source"], "product worker")

    def test_product_eval_fresh_import_and_cli_help_never_load_models(self):
        code = ("import sys; from tools.eval import dialogue_product_eval as e; "
                "assert not any(m in sys.modules for m in ('torch','torchaudio','transformers','demucs')); "
                "e.parse_args(['--help'])")
        completed = proc.run([sys.executable, "-B", "-X", "utf8", "-c", code],
                             capture_output=True, text=True, encoding="utf-8", timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--cache-dir", completed.stdout)

    def test_dialogue_partial_cache_only_recomputes_missing_emission(self):
        request = self.dialogue_request()
        directory = Path(request.cache_dir) / "kalign" / cache.cache_key(self.video, 1, None)
        cache.write_emission(directory, self.emission("mms"), dtype=np.float16)
        cache.write_array(directory / "rms.npy", np.zeros(100))
        vocals = np.zeros((44100, 2), np.float32)
        with patch.object(proc, "run", return_value=SimpleNamespace(stdout=vocals.astype("<f4").tobytes())), \
                patch.object(models, "separate_vocals", return_value=vocals), \
                patch.object(models, "mms_emission") as mms, \
                patch.object(models, "rms_db_10ms") as rms, \
                patch.object(models, "kana_emission", return_value=self.emission("kana")) as kana:
            worker.run_request(request, lambda *args: None)
        mms.assert_not_called()
        rms.assert_not_called()
        kana.assert_called_once()

    def test_request_round_trip_dataclasses_all_optional_fields(self):
        request = self.request()
        restored = worker.request_from_json(json.loads(json.dumps(asdict(request))))
        self.assertEqual(restored, request)
        self.assertIsInstance(restored.lines[0].beats[0].fixed, tuple)

    def test_result_round_trip_dataclasses(self):
        result = song.SongResult([song.LineResult("jp", [(1200, 1400, .9, True)])], 12, "habit", 20, ["flag"])
        self.assertEqual(worker.result_from_json(json.loads(json.dumps(asdict(result)))), result)

    def test_request_defaults(self):
        data = asdict(self.request())
        del data["options"]
        del data["audio_track"]
        restored = worker.request_from_json(data)
        self.assertEqual(restored.options, {})
        self.assertIsNone(restored.audio_track)

    def test_request_rejects_invalid_clip_lines_options(self):
        for field, value in (("clip", [1000, 1000]), ("clip", [-10, 1000]), ("clip", [True, 1000]),
                             ("clip", [1.5, 1000]), ("clip", [1000]), ("lines", []),
                             ("options", {"offset": True}), ("options", {"last_end": "invalid"})):
            with self.subTest(field=field, value=value):
                data = asdict(self.request())
                data[field] = value
                with self.assertRaises(ValueError):
                    worker.request_from_json(data)

    def test_clip_rounding_and_key_match_decode_boundaries(self):
        data = asdict(self.request())
        data["clip"] = [1005, 3015]
        self.assertEqual(worker.request_from_json(data).clip, (1000, 3020))
        self.assertEqual(cache.cache_key(self.video, 1, [1001, 3001]),
                         cache.cache_key(self.video, 1, [1004, 3004]))
        self.assertNotEqual(cache.cache_key(self.video, 1, [1004, 3004]),
                            cache.cache_key(self.video, 1, [1006, 3006]))

    def test_cache_key_path_track_clip_version_revision(self):
        key = cache.cache_key(self.video, 1, (1000, 3000))
        self.assertEqual(len(key), 40)
        self.assertEqual(key, cache.cache_key(self.video.parent / "." / self.video.name, 1, (1000, 3000)))
        self.assertNotEqual(key, cache.cache_key(self.video, 0, (1000, 3000)))
        for field in ("KALIGN_VERSION", "KANA_REVISION", "MMS_ID", "DEMUCS_ID"):
            with patch.object(cache, field, "changed"):
                self.assertNotEqual(key, cache.cache_key(self.video, 1, (1000, 3000)))

    def test_cache_key_invalidates_size_and_mtime(self):
        key = cache.cache_key(self.video, 0, (0, 1000))
        original = self.video.stat()
        self.video.write_bytes(b"different size")
        os.utime(self.video, ns=(original.st_atime_ns, original.st_mtime_ns))
        changed = cache.cache_key(self.video, 0, (0, 1000))
        self.assertNotEqual(key, changed)
        os.utime(self.video, ns=(original.st_atime_ns, original.st_mtime_ns + 10_000_000))
        self.assertNotEqual(changed, cache.cache_key(self.video, 0, (0, 1000)))

    def test_atomic_array_and_utf8_json(self):
        directory = self.root / "cache/kalign/key"
        cache.write_array(directory / "rms.npy", np.arange(5))
        np.testing.assert_equal(np.load(directory / "rms.npy", allow_pickle=False), np.arange(5))
        cache.write_json(directory / "result.json", {"text": "あ中文"})
        self.assertEqual(json.loads((directory / "result.json").read_text(encoding="utf-8")), {"text": "あ中文"})
        self.assertFalse(list(directory.glob("*.tmp")))

    def test_atomic_failure_retains_previous_cache_and_cleans_staging(self):
        path = self.root / "cache/kalign/key/rms.npy"
        cache.write_array(path, np.arange(3))
        def fail(stream):
            stream.write(b"partial")
            raise OSError("write failed")
        with self.assertRaises(OSError):
            cache.atomic_write(path, fail)
        np.testing.assert_equal(np.load(path), np.arange(3))
        self.assertFalse(list((self.root / "cache/.kalign-staging").glob("*.tmp")))

    @unittest.skipUnless(os.name == "nt", "Windows worker kill cleanup")
    def test_killed_cache_writer_leaves_no_partial_files(self):
        path = self.root / "cache/kalign/key/rms.npy"
        ready = self.root / "writer-ready"
        code = ("import sys,time; from pathlib import Path; from subkit.kalign.cache import atomic_write; "
                "writer=lambda f: (f.write(b'partial'), f.flush(), Path(sys.argv[2]).write_text('ready'), time.sleep(30)); "
                "atomic_write(sys.argv[1],writer)")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        child = subprocess.Popen([sys.executable, "-B", "-X", "utf8", "-c", code, str(path), str(ready)],
                                 env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **proc.hidden())
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(ready.exists(), "writer failed to reach cache staging")
            child.kill()
            child.wait(timeout=5)
            self.assertFalse(path.exists())
            self.assertFalse(list((self.root / "cache/.kalign-staging").glob("*.tmp")))
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)

    def test_emission_cache_round_trip_and_incomplete_marker(self):
        directory = self.root / "cache/kalign/key"
        original = self.emission("kana")
        cache.write_emission(directory, original)
        restored = cache.load_emission(directory, "kana")
        np.testing.assert_equal(restored.log_probs, original.log_probs)
        self.assertEqual(restored.vocab, original.vocab)
        self.assertEqual((restored.blank, restored.frame_ms, restored.star, restored.shift_ms), (0, 20., None, None))
        (directory / "kana.json").unlink()
        self.assertIsNone(cache.load_emission(directory, "kana"))

    def test_bad_emission_cache_is_a_miss(self):
        directory = self.root / "cache/kalign/key"
        directory.mkdir(parents=True)
        (directory / "mms.json").write_text("{broken", encoding="utf-8")
        self.assertIsNone(cache.load_emission(directory, "mms"))

    def test_progress_parsing_exact_stages_and_errors(self):
        for stage in worker.STAGES:
            value = {"stage": stage, "done": 0, "total": 2}
            self.assertEqual(worker.parse_progress(json.dumps(value)), value)
        self.assertEqual(worker.parse_progress('{"error":"中文错误"}'), {"error": "中文错误"})

    def test_progress_ignores_log_and_malformed_fields(self):
        for line in ("loading model", "[]", "null", "{}", '{"stage":"other","done":0,"total":1}',
                     '{"stage":"align","done":true,"total":1}', '{"stage":"align","done":2,"total":1}',
                     '{"stage":"align","done":0,"total":0}', b"\xff"):
            self.assertIsNone(worker.parse_progress(line))

    def test_progress_stdout_json_line(self):
        out = io.StringIO()
        with redirect_stdout(out):
            worker.progress("model1", 2, 4)
        self.assertEqual(out.getvalue(), '{"stage": "model1", "done": 2, "total": 4}\n')

    def test_worker_pipeline_maps_track_cache_hit_and_song_run(self):
        request = self.request()
        vocals = np.zeros((44100, 2), np.float32)
        result = song.SongResult([song.LineResult("jp", [(1200, 1300, .8, False)])], 0, "none", 0)
        events = []
        with patch.object(proc, "run", return_value=SimpleNamespace(stdout=vocals.astype("<f4").tobytes())) as extract, \
                patch.object(models, "separate_vocals", return_value=vocals) as separate, \
                patch.object(models, "mms_emission", return_value=self.emission("mms")) as mms, \
                patch.object(models, "kana_emission", return_value=self.emission("kana")) as kana, \
                patch.object(song, "run", return_value=result) as assemble:
            notify = lambda *args: events.append(args)
            self.assertEqual(worker.run_request(request, notify), result)
            self.assertEqual(worker.run_request(request, notify), result)
        extract.assert_called_once()
        cmd = extract.call_args.args[0]
        self.assertEqual(cmd[cmd.index("-map") + 1], "0:a:1")
        self.assertEqual(cmd[cmd.index("-ss") + 1], "1.000")
        self.assertEqual(cmd[cmd.index("-t") + 1], "2.000")
        separate.assert_called_once()
        mms.assert_called_once()
        kana.assert_called_once()
        self.assertEqual(assemble.call_count, 2)
        self.assertEqual(assemble.call_args.args[3:], (1000, request.options))
        self.assertEqual([x.name for x in assemble.call_args.args[1]], ["mms", "kana"])
        self.assertIn(("align", 1, 1), events)

    def test_empty_extraction_fails_before_models(self):
        with patch.object(proc, "run", return_value=SimpleNamespace(stdout=b"")), \
                patch.object(models, "separate_vocals") as separate:
            with self.assertRaisesRegex(ValueError, "empty audio"):
                worker.run_request(self.request(), lambda *args: None)
        separate.assert_not_called()

    def test_worker_missing_video_error_real_subprocess(self):
        request = asdict(self.request())
        request["video"] = str(self.root / "missing.mkv")
        path, result = self.root / "request.json", self.root / "result.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        completed = proc.run([sys.executable, "-B", "-X", "utf8", "-m", "subkit.kalign.worker", str(path), str(result)],
                             env=env, capture_output=True, text=True, encoding="utf-8", timeout=15)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("video not found", worker.parse_progress(completed.stdout)["error"])
        self.assertFalse(result.exists())

    def test_worker_main_writes_result_before_done(self):
        path, output = self.root / "request.json", self.root / "result.json"
        path.write_text(json.dumps(asdict(self.request())), encoding="utf-8")
        result = song.SongResult([], 0, "none", 0)
        out = io.StringIO()
        with patch.object(worker, "run_request", return_value=result), redirect_stdout(out):
            self.assertEqual(worker.main([str(path), str(output)]), 0)
        self.assertEqual(worker.result_from_json(json.loads(output.read_text(encoding="utf-8"))), result)
        self.assertEqual(worker.parse_progress(out.getvalue()), {"stage": "done", "done": 1, "total": 1})

    def test_lazy_module_imports_real_subprocess(self):
        code = ("import sys; from subkit.kalign import models, cache, worker; "
                "assert not any(m in sys.modules for m in ('torch','torchaudio','transformers','demucs','soundfile')); print('lazy OK')")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        completed = proc.run([sys.executable, "-B", "-X", "utf8", "-c", code], env=env,
                             capture_output=True, text=True, encoding="utf-8", timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "lazy OK")

    def test_chunk_coverage_and_context_boundaries(self):
        for n in (400, 16000 * 30, 16000 * 30 + 400, 16000 * 65):
            chunks = list(models.chunk_ranges(n))
            self.assertEqual(chunks[0][0], 0)
            self.assertEqual(chunks[-1][1], n)
            self.assertEqual(sum(b - a for a, b, lo, hi in chunks), n)
            for a, b, lo, hi in chunks:
                self.assertLessEqual(a - lo, 16000)
                self.assertLessEqual(hi - b, 16000)
                frames = (hi - lo - 400) // 320 + 1
                s, e = models.chunk_crop(a, b, lo, hi, frames)
                self.assertGreater(e, s)
                self.assertLessEqual(e, frames)

    def test_rms_identical_to_prototype_stereo_and_silence(self):
        audio = np.random.default_rng(9).normal(size=(2003, 2)).astype(np.float32)
        mono, hop = audio.mean(axis=1), 441
        frames = np.lib.stride_tricks.sliding_window_view(np.pad(mono, (hop * 2, hop * 3)), hop * 5)[::hop][:len(mono) // hop + 1]
        expected = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
        np.testing.assert_equal(models.rms_db_10ms(audio), expected)
        np.testing.assert_allclose(models.rms_db_10ms(np.zeros((441, 2))), -180, atol=.001)

    def test_mms_inference_normalization_star_and_context_crop(self):
        calls, devices, events = [], [], []
        class Model:
            def eval(self):
                return self
            def to(self, device):
                devices.append(device)
                return self
            def __call__(self, x):
                calls.append(x.copy())
                frames = (x.shape[1] - 400) // 320 + 1
                return tensor(np.broadcast_to([1., 2., 3.], (1, frames, 3)).copy()), None
        bundle = SimpleNamespace(get_dict=lambda **kw: {"a": 1, "*": 2},
                                 get_model=lambda **kw: Model())
        audio = np.random.default_rng(2).normal(size=16000 * 31).astype(np.float32)
        with patch.dict(sys.modules, {"torch": fake_torch(), "torchaudio": SimpleNamespace(pipelines=SimpleNamespace(MMS_FA=bundle))}):
            result = models.mms_emission(audio, lambda i, n: events.append((i, n)))
        self.assertEqual(devices, ["cpu"])
        self.assertEqual(events, [(1, 2), (2, 2)])
        np.testing.assert_allclose(np.asarray(calls[0]).std(ddof=1), .1, atol=1e-6)
        self.assertAlmostEqual(float(calls[0].mean()), 0, places=6)
        self.assertEqual((result.name, result.blank, result.star), ("mms", 0, 2))
        np.testing.assert_allclose(np.exp(result.log_probs).sum(axis=1), 1, atol=1e-6)
        self.assertAlmostEqual(result.frame_ms * len(result.log_probs), 31000)

    def test_download_decision_before_pinned_kana_retry_and_checkpoints(self):
        events = []
        processor = Mock()
        model = Mock()
        processor.from_pretrained.side_effect = [OSError("not cached"), "processor"]
        model.from_pretrained.return_value = "model"
        self.assertEqual(models.load_kana(processor, model, lambda: events.append("download")), ("processor", "model"))
        self.assertEqual(events, ["download"])
        self.assertEqual(processor.from_pretrained.call_args.kwargs,
                         {"revision": models.KANA_REVISION, "local_files_only": False})
        self.assertEqual(model.from_pretrained.call_args.kwargs, processor.from_pretrained.call_args.kwargs)
        fake = SimpleNamespace(hub=SimpleNamespace(get_dir=lambda: str(self.root)))
        for filename in ("model.pt", "955717e8-8726e21a.th"):
            events.clear()
            self.assertTrue(models.checkpoint_download_needed(fake, filename, lambda: events.append("download")))
            self.assertEqual(events, ["download"])
            path = self.root / "checkpoints" / filename
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"fake")
            events.clear()
            self.assertFalse(models.checkpoint_download_needed(fake, filename, lambda: events.append("download")))
            self.assertEqual(events, [])

    def test_kana_pinned_local_only_nonzero_blank_and_chunking(self):
        loads, calls, devices = [], [], []
        class Model:
            def eval(self):
                return self
            def to(self, device):
                devices.append(device)
                return self
            def __call__(self, x):
                calls.append(x.shape[1])
                frames = (x.shape[1] - 400) // 320 + 1
                return SimpleNamespace(logits=tensor(np.broadcast_to([1., 2., 3.], (1, frames, 3)).copy()))
        class Processor:
            tokenizer = SimpleNamespace(pad_token="<pad>", get_vocab=lambda: {"あ": 0, "|": 1, "<pad>": 2})
            def __call__(self, x, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(input_values=tensor(x[None]))
        processor = Processor()
        def load_processor(model_id, **kwargs):
            loads.append((model_id, kwargs))
            return processor
        def load_model(model_id, **kwargs):
            loads.append((model_id, kwargs))
            return Model()
        transformers = SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
                                       AutoModelForCTC=SimpleNamespace(from_pretrained=load_model))
        with patch.dict(sys.modules, {"torch": fake_torch(cuda=True), "transformers": transformers}):
            result = models.kana_emission(np.zeros(16000 * 21, np.float32))
        self.assertEqual(len(calls), 2)
        self.assertEqual(devices, ["cuda"])
        self.assertEqual(loads, [(models.KANA_ID, {"revision": models.KANA_REVISION, "local_files_only": True})] * 2)
        self.assertEqual(processor.kwargs, {"sampling_rate": 16000, "return_tensors": "pt"})
        self.assertEqual(result.blank, 2)
        self.assertAlmostEqual(result.frame_ms * len(result.log_probs), 21000)

    def test_demucs_normalization_restore_parameters_and_silence(self):
        devices, applied = [], []
        class Model:
            sources = ["vocals", "other"]
            def eval(self):
                return self
            def to(self, device):
                devices.append(device)
                return self
        def apply_model(model, x, **kwargs):
            applied.append(kwargs)
            return tensor(np.stack([x, x], axis=1))
        modules = {"torch": fake_torch(), "demucs.apply": SimpleNamespace(apply_model=apply_model),
                   "demucs.pretrained": SimpleNamespace(get_model=lambda name: Model())}
        audio = np.random.default_rng(3).normal(size=(1000, 2)).astype(np.float32)
        with patch.dict(sys.modules, modules):
            vocals = models.separate_vocals(audio)
            silence = models.separate_vocals(np.zeros((1000, 2), np.float32))
        np.testing.assert_allclose(vocals, audio, atol=1e-6)
        np.testing.assert_equal(silence, 0)
        self.assertEqual(applied, [{"device": "cpu", "shifts": 1, "split": True, "overlap": .25, "progress": False}])

    def test_eval_build_request_fixture_keeps_readings_gold_out_of_windows(self):
        from subkit.model.document import Document, Group, Member, Run
        from subkit.model.serialize import save_document
        # Use only an owned synthetic project, never user media in default tests.
        project = self.root / "project"
        ep = project / "episodes/S01E001"
        ep.mkdir(parents=True)
        doc = Document("S01E001")
        member = Member("jp", "g", "ja", "original", "song", 2000, 3000, runs=[Run("あ", ins="i")])
        doc.members[member.id] = member
        doc.groups = [Group("g", kind="song", members=["jp"])]
        save_document(doc, ep / "document.json")
        (ep / "episode.json").write_text(json.dumps({"media": {"video": {"path": str(self.video)}, "audio_track": 2}}), encoding="utf-8")
        (project / "subkit-project.json").write_text('{"settings":{}}', encoding="utf-8")
        request, refs = kalign_eval.build_request(project, "S01E001", self.root / "cache")
        self.assertEqual((request.clip, request.audio_track, request.options["offset"]), ((0, 5000), 2, "none"))
        self.assertEqual(request.lines[0].beats[0].sung, "あ")
        self.assertIsNone(request.lines[0].beats[0].fixed)
        self.assertEqual(refs[0]["mid"], "jp")
        self.assertEqual(request.options["continuation_mode"], "even")

    def test_eval_arguments_and_exact_reference_exclusions(self):
        args = kalign_eval.argument_parser().parse_args([])
        self.assertEqual((args.episode, args.offset, args.cache_dir), ("S01E001", "none", None))
        refs = [{"mid": "jp", "start": 1000, "beats": [{"t0": 1000, "by": "drag"},
                  {"t0": 1200, "by": "manual"}, {"t0": 1300, "by": "onset"},
                  {"t0": 1400, "by": "tap"}, {"t0": 1500, "by": "drag"}]}]
        self.assertEqual(kalign_eval.metric_line("fused", [[9999, 1250, 1400, 9999, 1700]], refs),
                         "fused: n=3 median=100 ms <=50 ms=33% <=100 ms=67% <=200 ms=100%")

    @unittest.skipUnless(os.environ.get("SUBKIT_KALIGN_TESTS") == "1", "heavy real-media pipeline opt-in")
    def test_heavy_real_media_eval(self):
        self.assertEqual(kalign_eval.main(["--cache-dir", str(self.root / "heavy-cache")]), 0)


if __name__ == "__main__":
    unittest.main()


class ContinuationContracts(unittest.TestCase):
    def test_result_old_and_new_json_fields(self):
        result = song.SongResult([], 0, "none", 0, [], .62, "habit", 8)
        self.assertEqual(worker.result_from_json(json.loads(json.dumps(asdict(result)))), result)
        old = {"lines": [], "offset_ms": 0, "offset_source": "none", "offset_count": 0}
        result = worker.result_from_json(old)
        self.assertEqual((result.continuation_ratio, result.continuation_source, result.continuation_count), (None, "none", 0))
        even = song.SongResult([], 0, "none", 0, [], .5, "even", 0)
        self.assertEqual(worker.result_from_json(json.loads(json.dumps(asdict(even)))), even)

    def test_evaluation_anomalies_without_reference(self):
        line = song.LineInput("jp", 0, 600, [song.BeatInput("か"), song.BeatInput("う"), song.BeatInput("っ")])
        result = song.SongResult([song.LineResult("jp", [(0, 100, 1, False), (250, 290, 1, False), (400, 600, 1, False)])], 0, "none", 0)
        rms = np.zeros(100); rms[25] = -40
        self.assertEqual(kalign_eval.anomaly_line(result, [line], rms, 0), 'fused anomalies: short=1 silent-start-100ms=0 suspicious-gap=1')
        rms[25:35] = -40
        self.assertEqual(kalign_eval.anomaly_line(result, [line], rms, 0), 'fused anomalies: short=1 silent-start-100ms=1 suspicious-gap=1')
        references = [{"start": 0, "beats": [{"t0": 0, "by": "align"}] * 3}]
        self.assertEqual(kalign_eval.metric_line('fused', [[0,250,400]], references), 'fused: no comparable hand-timed beats')
