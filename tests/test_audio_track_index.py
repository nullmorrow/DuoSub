"""W1 stored mpv ids, decode commands and offscreen failure/retry behaviour."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PySide6.QtWidgets import QApplication
from subkit import audio
from subkit.core import oped_import
from subkit.reading import voice
from subkit_app.views import waveform
from subkit_app import help as help_ui
import tests.test_passages_ui as fixtures


class AudioTrackIndex(unittest.TestCase):
    def test_stored_aid_conversion(self):
        for aid, index in ((None, 0), (0, 0), (1, 0), (2, 1)):
            with self.subTest(aid=aid):
                self.assertEqual(audio.ffmpeg_audio_index(aid), index)

    def test_all_decode_commands_use_stored_aid(self):
        raw = np.zeros(1600, dtype="<i2").tobytes()
        for aid, index in ((None, 0), (0, 0), (1, 0), (2, 1)):
            with self.subTest(aid=aid):
                runner = Mock()
                runner.run.return_value = SimpleNamespace(stdout=raw)
                waveform._decode_samples(runner, Path("episode.mkv"), aid)
                cmd = runner.run.call_args.args[0]
                self.assertEqual(cmd[cmd.index("-map") + 1], f"0:a:{index}")
                oped_import.decode_audio_features("episode.mkv", audio_track=aid, runner=runner)
                cmd = runner.run.call_args.args[0]
                self.assertEqual(cmd[cmd.index("-map") + 1], f"0:a:{index}")
                with patch("subkit.proc.run", return_value=SimpleNamespace(stdout=raw)) as run:
                    voice.load_audio(Path("episode.mkv"), 0, 1, audio_track=aid)
                cmd = run.call_args.args[0]
                self.assertEqual(cmd[cmd.index("-map") + 1], f"0:a:{index}")
                self.assertTrue(run.call_args.kwargs["check"])

    def test_error_last_line_limit_and_full_details(self):
        stderr = "first line\n" + "错" * 100 + "\n"
        error = subprocess.CalledProcessError(1, ["ffmpeg", "-i", "file with spaces.mkv"], stderr=stderr.encode("utf-8"))
        reason, details = waveform._audio_error(error)
        self.assertEqual(reason, "错" * 80)
        self.assertIn('"file with spaces.mkv"', details)
        self.assertIn(stderr, details)


class AudioReadUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.video = self.root / "episode.mkv"
        self.video.write_bytes(b"test media")
        self.view = waveform.WaveformView()
        self.view.resize(640, 240)
        self.view.show()
        self.app.processEvents()

    def tearDown(self):
        self.view.close()
        self.app.processEvents()
        self.tmp.cleanup()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertTrue(predicate())

    def test_peak_failure_and_retry_launches_new_decode(self):
        calls = []
        def run(runner, cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return SimpleNamespace(stdout="")
            calls.append(cmd)
            if len(calls) == 1:
                raise subprocess.CalledProcessError(1, cmd, stderr=b"header\nStream map '0:a:0' matches no streams\n")
            return SimpleNamespace(stdout=np.zeros(1600, dtype="<i2").tobytes())
        with patch.object(waveform.proc.KillableRun, "run", run):
            self.view.load(self.video, 1, self.root)
            self.wait_for(lambda: self.view.status_text.startswith("读取音频失败："))
            self.assertEqual(self.view.status_text, "读取音频失败：Stream map '0:a:0' matches no streams")
            self.assertIn("-map 0:a:0", self.view.toolTip())
            self.assertIn("header", self.view.toolTip())
            self.assertTrue(self.view.btn_retry.isVisible())
            self.assertEqual(self.view.btn_retry.help_anchor, "audio-read")
            self.view.btn_retry.click()
            self.wait_for(lambda: self.view.audio_features() is not None)
            self.assertEqual(len(calls), 2)
            self.assertFalse(self.view.btn_retry.isVisible())
            self.assertEqual(self.view.toolTip(), "")

    def test_retry_clears_only_current_waveform_cache(self):
        cache = waveform._waveform_cache_file(self.video, 1, self.root)
        cache.write_bytes(b"stale features")
        other = waveform._waveform_cache_file(self.video, 2, self.root)
        other.write_bytes(b"other track")
        self.view._audio_source = (self.video, 1)
        self.view._audio_cache = self.root
        with patch.object(self.view, "load") as load:
            self.view.retry_audio()
        self.assertFalse(cache.exists())
        self.assertTrue(other.exists())
        load.assert_called_once_with(self.video, 1, self.root)

    def test_cache_uses_converted_index_and_new_version(self):
        first = waveform._waveform_cache_file(self.video, None, self.root)
        self.assertEqual(first, waveform._waveform_cache_file(self.video, 0, self.root))
        self.assertEqual(first, waveform._waveform_cache_file(self.video, 1, self.root))
        self.assertNotEqual(first, waveform._waveform_cache_file(self.video, 2, self.root))
        with patch.object(audio, "CACHE_VERSION", "audio-v3"):
            self.assertNotEqual(first, waveform._waveform_cache_file(self.video, 1, self.root))

    def test_fine_sample_failure_preserves_coarse_features(self):
        features = audio.compute_features(np.zeros(1600, np.float32))
        self.view._loaded(features)
        messages = []
        self.view.audioStatus.connect(messages.append)
        error = subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"fine read failed")
        self.view._samples_fail(error, self.view._load_generation - 1)
        self.assertEqual(messages, [])
        self.view._samples_fail(error, self.view._load_generation)
        self.assertEqual(messages, ["读取放大细节失败，显示粗波形"])
        self.assertIs(self.view.audio_features(), features)
        self.assertIs(self.view.mins, features.mins)
        self.assertIs(self.view.spectrogram, features.spectrogram)
        self.assertEqual(self.view.status_text, "")
        self.assertTrue(self.view._samples_failed)

    def test_cached_audio_fine_job_failure_keeps_waveform_and_spectrum(self):
        features = audio.compute_features(np.zeros(1600, np.float32))
        cache = waveform._waveform_cache_file(self.video, 2, self.root)
        np.savez_compressed(cache, mins=features.mins, maxs=features.maxs,
                            spectrogram=features.spectrogram, onset=features.onset)
        self.view.span = 1000
        messages, maps = [], []
        self.view.audioStatus.connect(messages.append)
        def run(runner, cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return SimpleNamespace(stdout="")
            maps.append(cmd[cmd.index("-map") + 1])
            raise subprocess.CalledProcessError(1, cmd, stderr=b"fine job failed")
        with patch.object(waveform.proc.KillableRun, "run", run):
            self.view.load(self.video, 2, self.root)
            self.wait_for(lambda: self.view._samples_failed)
        self.assertEqual(maps, ["0:a:1"])
        self.assertEqual(messages, ["读取放大细节失败，显示粗波形"])
        np.testing.assert_array_equal(self.view.mins, features.mins)
        np.testing.assert_array_equal(self.view.spectrogram, features.spectrogram)
        self.assertIn("fine job failed", self.view.toolTip())

    def test_hint_failure_keeps_loaded_audio(self):
        features = audio.compute_features(np.zeros(1600, np.float32))
        with patch.object(waveform, "_snap_activity", side_effect=ValueError("hint failure")):
            self.view._loaded(features)
        self.assertIs(self.view.audio_features(), features)
        self.assertIs(self.view.spectrogram, features.spectrogram)
        self.assertEqual(self.view._nearby_hints(), [])
        self.assertEqual(self.view.status_text, "")

    def test_fixed_toolbar_retry_is_available_and_documented(self):
        panel = waveform.WaveformPanel(self.view)
        try:
            self.assertEqual(panel.btn_retry.text(), "重新读取音频")
            self.assertEqual(panel.btn_retry.help_anchor, "audio-read")
            with patch.object(self.view, "load") as load:
                self.view._audio_source = (self.video, 1)
                self.view._audio_cache = self.root
                panel.btn_retry.click()
                load.assert_called_once_with(self.video, 1, self.root)
        finally:
            panel.close()


class AudioReadMainWindow(unittest.TestCase):
    setUpClass = classmethod(fixtures.PassageUI.setUpClass.__func__)
    setUp = fixtures.PassageUI.setUp
    _dispose = fixtures.PassageUI._dispose
    select = fixtures.PassageUI.select

    def test_fine_failure_reaches_status_bar_and_retry_f1_in_song_layout(self):
        view = self.window.wave
        before = view.audio_features()
        view._samples_fail(subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"failed"),
                           view._load_generation)
        self.assertEqual(self.window.statusBar().currentMessage(), "读取放大细节失败，显示粗波形")
        self.assertIs(view.audio_features(), before)
        self.assertEqual(help_ui.resolve_anchor(self.window, active=self.window,
                                               focus=self.window.wave_panel.btn_retry), "audio-read")
        self.assertEqual(help_ui.resolve_anchor(self.window, active=self.window,
                                               focus=view.btn_retry), "audio-read")

    def test_group_hint_failure_only_hides_hints(self):
        view = self.window.wave
        before = view.audio_features()
        self.doc.groups[0].kind = "dialogue"
        view.hint_lines = True
        with patch.object(waveform, "speech_segments", side_effect=ValueError("hints failed")):
            view._refresh_hints()
        self.assertEqual(view._nearby_hints(), [])
        self.assertIsNone(view.suggested_times())
        self.assertIs(view.audio_features(), before)
        self.assertIs(view.spectrogram, before.spectrogram)
        self.assertEqual(view.status_text, "")


if __name__ == "__main__":
    unittest.main()
