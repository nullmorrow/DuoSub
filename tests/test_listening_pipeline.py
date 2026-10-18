"""Killable external pipeline, offline models, exact track and window handling."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from subkit.listening import ListeningRequest, generate
from subkit.proc import CancelledError, KillableRun


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.video = self.path / "input.mkv"
        self.video.write_bytes(b"immutable video fixture")
        self.model = self.path / "large-v3.pt"
        self.model.write_bytes(b"local model fixture")
        self.request = ListeningRequest(str(self.video), 3, 10000, 11000)
        self.output = self.path / "fresh"

    def test_extract_selected_track_then_single_direct_worker_and_local_model(self):
        commands, labels = [], []
        outer = self
        class Runner(KillableRun):
            def run(self, cmd, **kwargs):
                commands.append(cmd)
                if cmd[0] == "ffmpeg":
                    (outer.output / "original.wav").write_bytes(b"PCM fixture")
                else:
                    request = json.loads((outer.output / "request.json").read_text(encoding="utf-8"))
                    outer.assertEqual(request["model_file"], str(outer.model))
                    outer.assertEqual(request["window_start_ms"], 8800)
                    data = {"clips": {"original": "original.wav"}, "duration_ms": 3400,
                            "runs": [], "diagnostics": [], "device": "cpu"}
                    (outer.output / "result.json").write_text(json.dumps(data), encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, b"", b"")
        with patch("subkit.listening.model_file", return_value=self.model):
            result = generate(self.request, self.output, runner=Runner(), progress=labels.append)
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][commands[0].index("-map") + 1], "0:a:2")
        self.assertEqual(commands[0][commands[0].index("-ss") + 1], "8.800")
        self.assertEqual(commands[0][commands[0].index("-t") + 1], "3.400")
        self.assertIn("-n", commands[0])
        self.assertTrue(commands[1][4].endswith("listening_worker.py"))
        self.assertEqual(result.device, "cpu")
        self.assertEqual(self.video.read_bytes(), b"immutable video fixture")
        self.assertEqual(len(labels), 3)

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        keep = self.output / "vocals.wav"
        keep.write_bytes(b"do not overwrite")
        with patch("subkit.listening.model_file", return_value=self.model), self.assertRaisesRegex(ValueError, "新"):
            generate(self.request, self.output, runner=KillableRun(), progress=lambda _: None)
        self.assertEqual(keep.read_bytes(), b"do not overwrite")

    def test_user_heard_hint_is_optional_and_invalid_hint_never_starts_tools(self):
        self.assertEqual(self.request.hint, "")
        runner = KillableRun()
        with patch.object(runner, "run") as run:
            for hint in ("a" * 81, "風船\n別の句", None):
                with self.subTest(hint=hint), self.assertRaisesRegex(ValueError, "80"):
                    generate(replace(self.request, hint=hint), self.output, runner=runner, progress=lambda _: None)
            run.assert_not_called()

    def test_worker_error_visible_without_partial_result(self):
        runner = KillableRun()
        def run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, int(cmd[0] != "ffmpeg"), b"", "缺少 Torch".encode())
        runner.run = run
        with patch("subkit.listening.model_file", return_value=self.model), self.assertRaisesRegex(RuntimeError, "缺少 Torch"):
            generate(self.request, self.output, runner=runner, progress=lambda _: None)

    def test_explicit_python_override_and_cancel_between_extract_and_inference(self):
        runner = KillableRun()
        commands = []
        def run(cmd, **kwargs):
            commands.append(cmd)
            runner.cancel()
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        # The real runner rejects the next start even if cancellation happens
        # while serializing the worker's request after extraction.
        original = runner.run
        def first(cmd, **kwargs):
            runner.run = original
            return run(cmd, **kwargs)
        runner.run = first
        with patch("subkit.listening.model_file", return_value=self.model), self.assertRaises(CancelledError):
            generate(replace(self.request, python="custom-python"), self.output, runner=runner, progress=lambda _: None)
        self.assertEqual(len(commands), 1)

    def test_live_direct_child_cancel_is_reaped_before_completion(self):
        runner = KillableRun()
        ready = self.path / "ready"
        command = [sys.executable, "-c", "from pathlib import Path;import sys,time;Path(sys.argv[1]).write_text('ok');time.sleep(60)", str(ready)]
        done, cancelled = threading.Event(), []
        def run():
            try:
                runner.run(command, timeout=10)
            except CancelledError:
                cancelled.append(True)
            finally:
                done.set()
        thread = threading.Thread(target=run)
        thread.start()
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(ready.is_file())
        process = runner._process
        runner.cancel()
        self.assertTrue(done.wait(3))
        thread.join(1)
        self.assertIsNotNone(process.poll())
        self.assertEqual(cancelled, [True])
        self.assertIsNone(runner._process)


if __name__ == "__main__":
    unittest.main()
