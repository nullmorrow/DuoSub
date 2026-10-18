"""Translation-only work must not extract audio, replace results, or outlive cancellation."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subkit.listening_refresh import generate_semantic_batch
from subkit.proc import CancelledError
from subkit_app.listening_job import SemanticJob


class RefreshPipeline(unittest.TestCase):
    def request(self, **changes):
        return SimpleNamespace(**(dict(translation="要氣球嗎", python="chosen-python") | changes))

    def call(self, request, output, runner, **changes):
        return generate_semantic_batch(request, output, runner=runner, progress=Mock(),
            **(dict(existing=("風船、いる？",), round_index=1, device="cuda") | changes))

    def test_only_disposable_translation_worker_receives_source_existing_and_round(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            runner = Mock()
            expected = {"texts": ["風船、いかがですか？"], "model": "test"}
            def run(command, timeout):
                self.assertEqual(command[:4], ["chosen-python", "-B", "-X", "utf8"])
                self.assertEqual(Path(command[4]).name, "listening_semantic_worker.py")
                self.assertEqual(timeout, 270)
                self.assertEqual(json.loads(Path(command[5]).read_text(encoding="utf-8")),
                    dict(translation="要氣球嗎", existing=["風船、いる？"], round_index=1, device="cuda"))
                (output / "result.json").write_text(json.dumps(expected), encoding="utf-8")
                return SimpleNamespace(returncode=0, stderr=b"")
            runner.run.side_effect = run
            self.assertEqual(self.call(self.request(), output, runner), expected)
            runner.run.assert_called_once()
            self.assertEqual(runner.check_cancelled.call_count, 2)
            self.assertFalse(any(p.suffix == ".wav" for p in output.iterdir()))

    def test_validation_rejects_empty_long_input_and_invalid_round_before_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            for changes in (dict(translation=""), dict(translation="中" * 1001)):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    self.call(self.request(**changes), Path(temp), Mock())
            for value in (0, -1, True, "1"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    self.call(self.request(), Path(temp), Mock(), round_index=value)

    def test_existing_output_is_preserved_and_never_launched(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "keep.wav").write_bytes(b"audio")
            runner = Mock()
            with self.assertRaises(ValueError):
                self.call(self.request(), output, runner)
            runner.run.assert_not_called()
            self.assertEqual((output / "keep.wav").read_bytes(), b"audio")

    def test_worker_failure_or_invalid_json_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            for i, (code, value) in enumerate(((1, None), (0, []), (0, {"texts": []}))):
                output = Path(temp) / str(i)
                output.mkdir()
                runner = Mock()
                def run(*args, **kwargs):
                    if value is not None:
                        (output / "result.json").write_text(json.dumps(value), encoding="utf-8")
                    return SimpleNamespace(returncode=code, stderr=b"model busy")
                runner.run.side_effect = run
                with self.subTest(code=code, value=value), self.assertRaises((RuntimeError, ValueError)):
                    self.call(self.request(), output, runner)

    def test_cancelled_worker_result_is_not_read(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = Mock()
            runner.run.return_value = SimpleNamespace(returncode=0, stderr=b"")
            runner.check_cancelled.side_effect = [None, CancelledError()]
            with self.assertRaises(CancelledError):
                self.call(self.request(), Path(temp), runner)

    def test_semantic_job_calls_only_refresh_and_discards_late_result_after_cancel(self):
        with tempfile.TemporaryDirectory() as temp:
            for mode in ("success", "cancel", "error", "cancel-before"):
                job = SemanticJob(self.request(), Path(temp), existing=("風船、いる？",), round_index=2, device="cpu")
                def generate(*args, **kwargs):
                    self.assertEqual(kwargs["round_index"], 2)
                    if mode == "error":
                        raise ValueError("busy")
                    if mode == "cancel":
                        job.cancel()
                    return {"texts": ["風船、いかがですか？"], "model": "test"}
                if mode == "cancel-before":
                    job.cancel()
                with patch("subkit.listening_refresh.generate_semantic_batch", side_effect=generate) as refresh, \
                        patch("subkit.listening.generate") as audio:
                    job.run()
                    audio.assert_not_called()
                    self.assertEqual(refresh.call_count, int(mode != "cancel-before"))
                self.assertTrue(job.finished.is_set())
                self.assertEqual(job.result is not None, mode == "success")
                self.assertEqual(job.cancelled, mode in ("cancel", "cancel-before"))
                self.assertEqual(job.error is not None, mode == "error")


if __name__ == "__main__":
    unittest.main()
