"""Offscreen lifecycle regressions with project copies and a fake player."""

from __future__ import annotations

import copy
import contextlib
import io
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests import test_transform_ui as transform_ui
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from subkit.core.workspace import Project
from subkit.model.serialize import document_to_json
from subkit_app.session import EpisodeSession, PreviewJob
from subkit import proc
from subkit.ruby.measure import Measurer
from subkit_app.watch import WatchController

REQUEST_PREVIEW = EpisodeSession.request_preview


class LifecycleUI(unittest.TestCase):
    setUpClass = classmethod(transform_ui.TransformUI.setUpClass.__func__)
    group = transform_ui.TransformUI.group
    key = transform_ui.TransformUI.key
    enter = transform_ui.TransformUI.enter
    drag = transform_ui.TransformUI.drag

    def setUp(self):
        transform_ui.TransformUI.setUp(self)
        old = self.session
        self.window.close_session()
        old._autosave.stop()
        old._debounce.stop()
        old.deleteLater()
        root = Path(__file__).resolve().parents[1]
        shutil.copytree(root / "test_projects" / "soma", self.path / "project",
                        copy_function=shutil.copyfile, ignore=shutil.ignore_patterns("cache"))
        self.project = Project.open(self.path / "project")
        self.episode = self.project.add_episode("LIFE_A")
        self.episode.document = self.doc
        self.episode.save()
        other = self.project.add_episode("LIFE_B")
        other.document = copy.deepcopy(self.doc)
        other.document.id = "lifecycle-b"
        other.document.members[self.cn.id].lane = "original"
        other.save()
        self.window.project = self.project
        p = patch.object(EpisodeSession, "request_preview", Mock())
        p.start()
        self.addCleanup(p.stop)
        self.window._load_media = Mock()
        self.window._start_session(self.episode)
        self.session = self.window.session
        self.session.engine = SimpleNamespace(analyze=Mock(return_value=[]))
        self.session.editor.set_overrides = Mock(wraps=self.session.editor.set_overrides)
        self.window.watch_mode = True
        self.player.setFocus()
        self.app.processEvents()
        self.before = copy.deepcopy(document_to_json(self.doc))
        self.history_before = list(self.session.editor.history.done)

    def _dispose(self):
        current = getattr(self, "window", None) and self.window.session
        transform_ui.TransformUI._dispose(self)
        if current is not None and current is not self.session:
            current._autosave.stop()
            current._debounce.stop()
            current.deleteLater()
            self.app.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def assert_discarded(self):
        self.assertEqual(document_to_json(self.project.episode("LIFE_A").document), self.before)
        self.assertEqual(document_to_json(self.doc), self.before)
        self.assertEqual(self.session.editor.history.done, self.history_before)
        self.assertEqual(self.session.editor.history.undone, [])
        self.assertIsNone(self.watch.transform)

    def test_close_discards_unconfirmed_transform_before_saving(self):
        # An unrelated dirty edit still needs saving; only the draft transform is discarded.
        self.session.dirty = True
        self.enter()
        self.drag()
        self.window.close()
        self.assert_discarded()

    def test_switch_episode_discards_unconfirmed_transform_before_saving(self):
        self.session.dirty = True
        self.enter()
        self.drag()
        self.window._switch_episode("LIFE_B")
        self.assertEqual(self.window.session.episode.id, "LIFE_B")
        self.assert_discarded()

    def test_user_watch_exit_applies_transform_with_one_undo(self):
        for key in (Qt.Key.Key_F10, Qt.Key.Key_F11, Qt.Key.Key_Tab):
            with self.subTest(key=key):
                self.window.watch_mode = True
                self.window.wave_panel.btn_pin.blockSignals(True)
                self.window.wave_panel.btn_pin.setChecked(key == Qt.Key.Key_F10)
                self.window.wave_panel.btn_pin.blockSignals(False)
                self.watch.set_selected("g")
                self.player.setFocus()
                self.enter()
                self.drag()
                self.key(key)
                self.assertFalse(self.window.watch_mode)
                self.assertEqual(self.doc.members[self.cn.id].overrides["pos"], [115, 112])
                self.assertEqual(len(self.session.editor.history.done), 1)
                self.session.save()
                saved = self.project.episode("LIFE_A").document
                self.assertEqual(saved.members[self.cn.id].overrides["pos"], [115, 112])
                self.assertEqual(self.session.editor.history.undo().label, "自由变换")
                self.assertEqual(self.doc.members[self.cn.id].overrides, self.initial[self.cn.id])

    def test_late_preview_after_switch_does_not_change_current_episode(self):
        old = self.session
        job = PreviewJob(old, old.doc.revision, old.preview_path)
        job.signals.done.connect(old._preview_done)
        old._preview_running = True
        old.previewReady.connect(self.window._on_preview, Qt.ConnectionType.QueuedConnection)
        self.player.load_subtitle = Mock()
        old.previewReady.emit("queued-before-switch.ass", self.ass, self.report)
        self.window._switch_episode("LIFE_B")
        current = self.window.session
        current.warmed = True
        self.player.load_subtitle = Mock()
        current.previewReady.emit("current.ass", self.ass, self.report)
        self.player.load_subtitle.assert_called_once_with("current.ass")
        self.player.load_subtitle.reset_mock()
        ass, owners, lanes = self.watch.ass, copy.deepcopy(self.watch.owners), dict(self.watch.member_lane)
        doc = copy.deepcopy(document_to_json(current.doc))
        disk = (current.episode.dir / "document.json").read_bytes()
        job.signals.done.emit(job.revision, "old.ass", self.ass.copy(), self.report)
        old.previewReady.emit("already-queued-old.ass", self.ass.copy(), self.report)
        self.app.processEvents()
        self.player.load_subtitle.assert_not_called()
        self.assertIs(self.watch.ass, ass)
        self.assertEqual(self.watch.owners, owners)
        self.assertEqual(self.watch.member_lane, lanes)
        self.assertEqual(document_to_json(current.doc), doc)
        self.assertEqual((current.episode.dir / "document.json").read_bytes(), disk)

    def test_closed_session_stops_timers_and_ignores_worker_completion(self):
        old = self.session
        old._debounce.start()
        old._preview_running = old._preview_wanted = True
        ready, busy, status = Mock(), Mock(), Mock()
        old.previewReady.connect(ready)
        old.busy.connect(busy)
        old.status.connect(status)
        self.window._switch_episode("LIFE_B")
        self.assertFalse(old._debounce.isActive())
        self.assertFalse(old._autosave.isActive())
        with patch.object(old.pool, "start") as start:
            old._preview_done(old.doc.revision, "old.ass", self.ass, self.report)
            old._preview_failed(old.doc.revision, "old error")
            old._start_preview()
            REQUEST_PREVIEW(old)
            old.dirty = True
            old.autosave()
            start.assert_not_called()
        self.assertFalse(old._debounce.isActive())
        ready.assert_not_called()
        busy.assert_not_called()
        status.assert_not_called()

    def test_only_current_session_updates_status_busy_and_views(self):
        old = self.session
        old.status.connect(self.window._on_session_status, Qt.ConnectionType.QueuedConnection)
        old.busy.connect(self.window._on_busy, Qt.ConnectionType.QueuedConnection)
        old.changed.connect(self.window._on_changed, Qt.ConnectionType.QueuedConnection)
        old.status.emit("old status")
        old.busy.emit("old busy", True)
        old.changed.emit(set())
        self.window._switch_episode("LIFE_B")
        current = self.window.session
        current.status.emit("current status")
        current.busy.emit("current busy", True)
        with patch.object(self.window.model, "reset") as reset:
            self.app.processEvents()
            # Also cover a late signal that still has an identifiable old sender.
            old.previewReady.connect(self.window._on_preview)
            self.player.load_subtitle = Mock()
            old.previewReady.emit("old.ass", self.ass, self.report)
            self.player.load_subtitle.assert_not_called()
            reset.assert_not_called()
        self.assertEqual(self.window.statusBar().currentMessage(), "current status")
        self.assertEqual(self.window.lbl_task.text(), "current busy…")
        self.assertFalse(self.window.progress.isHidden())

    def _cancel_slow_warm(self, switch=False, precise=False, geometry=False, concurrent=False):
        started = threading.Event()
        children, probes = [], []
        real_popen = subprocess.Popen

        def slow_child(cmd, **kwargs):
            if "-frames:v" in cmd:
                probes.append(Path(kwargs["cwd"]))
                cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
            child = real_popen(cmd, **kwargs)
            children.append(child)
            if len(children) >= (3 if concurrent else 2 if geometry else 1):
                started.set()
            return child

        if precise:
            self.watch.ass.events[0].text = r"{\frz30}" + self.watch.ass.events[0].text
        errors = io.StringIO()
        with patch.object(proc.subprocess, "Popen", side_effect=slow_child), contextlib.redirect_stderr(errors):
            self.watch.warm_up()
            if geometry:
                self.watch._measuring_stopped = False
                self.watch.geo = {}
                WatchController.request_geometry(self.watch)
            if concurrent:
                self.watch._measuring_stopped = False
                self.watch.geo = {}
                self.watch.selected = None
                with patch.object(self.watch, "active", return_value=[]), patch.object(self.watch, "upcoming", return_value=[1]):
                    WatchController.request_geometry(self.watch)
                with patch.object(self.watch, "active", return_value=[0]), patch.object(self.watch, "upcoming", return_value=[1]):
                    WatchController.request_geometry(self.watch)
            try:
                self.assertTrue(started.wait(5), "measurement jobs did not start their children")
                start = time.monotonic()
                if switch:
                    self.window._switch_episode("LIFE_B")
                else:
                    self.window.close()
                elapsed = time.monotonic() - start
                self.assertLess(elapsed, 3)
                self.assertTrue(all(child.poll() is not None for child in children), "measurement child survived cancellation")
                self.assertTrue(all(not path.exists() for path in probes), "probe directory survived cancellation")
                # Process queued signals after shutdown as well.
                self.app.processEvents()
                self.assertEqual(errors.getvalue(), "")
                print(f"warm {'switch' if switch else 'close'} precise={precise} geometry={geometry}: {elapsed:.3f}s; child gone; probes removed")
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=3)
                from PySide6.QtCore import QThreadPool
                QThreadPool.globalInstance().waitForDone(3000)

    def test_close_kills_inflight_warm_measurement(self):
        self._cancel_slow_warm()

    def test_episode_switch_kills_inflight_precise_warm_measurement(self):
        self._cancel_slow_warm(switch=True, precise=True)

    def test_close_kills_both_warm_and_geometry_measurements(self):
        self._cancel_slow_warm(geometry=True)

    def test_close_kills_warm_and_two_concurrent_geometry_measurements(self):
        self._cancel_slow_warm(concurrent=True)

    def test_cancel_before_process_start_does_not_spawn(self):
        runner = proc.KillableRun()
        runner.cancel()
        with patch.object(proc.subprocess, "Popen") as spawn:
            with self.assertRaises(proc.CancelledError):
                runner.run([sys.executable, "-c", "pass"], cwd=str(self.path))
        spawn.assert_not_called()

    def test_abandoned_probe_cleanup_retried_by_next_measurer(self):
        import tempfile
        probe = Path(tempfile.mkdtemp(prefix="subkit_probe_"))
        self.addCleanup(lambda: shutil.rmtree(probe, ignore_errors=True))
        (probe / ".cleanup-pending").touch()
        # Simulate a transient sharing failure; never alter permissions.
        with patch("subkit.ruby.measure.shutil.rmtree", return_value=None):
            with Measurer(self.ass):
                pass
        self.assertTrue(probe.exists())
        with Measurer(self.ass):
            pass
        self.assertFalse(probe.exists())

    def _cancel_slow_audio(self, action):
        from PySide6.QtCore import QThreadPool
        children = []
        started = threading.Event()
        real_popen = subprocess.Popen
        video = self.path / "slow.wav"
        video.write_bytes(b"audio")
        cache = self.path / "wave-cache"
        def slow_child(cmd, **kwargs):
            child = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
            children.append(child)
            if len(children) == 2:
                started.set()
            return child
        errors = io.StringIO()
        with patch.object(proc.subprocess, "Popen", side_effect=slow_child), contextlib.redirect_stderr(errors):
            self.window.wave.load(video, None, cache)
            jobs = list(self.window.wave._jobs)
            try:
                self.assertTrue(started.wait(5), "decode/ffprobe did not start")
                before = time.monotonic()
                if action == "close":
                    self.window.close()
                elif action == "switch":
                    self.window._switch_episode("LIFE_B")
                else:
                    self.window.wave.load(None, None, cache)
                elapsed = time.monotonic() - before
                self.assertLess(elapsed, 3)
                self.assertTrue(all(child.poll() is not None for child in children))
                self.assertTrue(all(job.finished.is_set() for job in jobs))
                self.assertEqual(list(cache.glob("*")), [])
                self.app.processEvents()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                self.assertEqual(errors.getvalue(), "")
                print(f"audio {action}: {elapsed:.3f}s; ffmpeg/ffprobe child gone; no cache/error")
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=3)
                QThreadPool.globalInstance().waitForDone(3000)

    def test_close_kills_audio_decode_and_keyframe_probe(self):
        self._cancel_slow_audio("close")

    def test_switch_kills_audio_decode_and_keyframe_probe(self):
        self._cancel_slow_audio("switch")

    def test_reload_cancels_previous_audio_jobs(self):
        self._cancel_slow_audio("reload")

    def test_cancel_during_cache_write_leaves_no_partial_cache(self):
        import numpy as np
        from subkit_app.views.waveform import PeakJob
        video = self.path / "cache.wav"
        video.write_bytes(b"audio")
        cache = self.path / "wave-cache"
        job = PeakJob(video, None, cache)
        done, failed = Mock(), Mock()
        job.signals.done.connect(done)
        job.signals.failed.connect(failed)
        def interrupted_write(fh, **kwargs):
            fh.write(b"partial")
            job.cancel()
        with patch.object(job.runner, "run", return_value=SimpleNamespace(stdout=np.zeros(160, "<i2").tobytes())), \
                patch("subkit_app.views.waveform.np.savez_compressed", side_effect=interrupted_write):
            job.run()
        self.assertTrue(job.finished.is_set())
        self.assertEqual(list(cache.iterdir()), [])
        done.assert_not_called()
        failed.assert_not_called()

    def test_feature_computation_cancels_between_fft_chunks(self):
        import numpy as np
        from subkit.audio import compute_features
        check = Mock(side_effect=[None, proc.CancelledError()])
        with self.assertRaises(proc.CancelledError):
            compute_features(np.zeros(160 * 4096, np.float32), check_cancelled=check)
        self.assertEqual(check.call_count, 2)

    def _cancel_other_worker(self, template=False):
        from PySide6.QtCore import QThreadPool
        from PySide6.QtWidgets import QDialog
        from subkit_app.views.template_editor import TemplateEditor
        children = []
        started = threading.Event()
        real_popen = subprocess.Popen
        def slow_child(cmd, **kwargs):
            child = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
            children.append(child)
            started.set()
            return child
        dialog = None
        errors = io.StringIO()
        with patch.object(proc.subprocess, "Popen", side_effect=slow_child), contextlib.redirect_stderr(errors), contextlib.ExitStack() as stack:
            if not template:
                def slow_build(*args, runner, **kwargs):
                    runner.run(["ffmpeg", "fake-preview"], cwd=str(self.path))
                stack.enter_context(patch("subkit.core.export.build_ass", side_effect=slow_build))
            if template:
                dialog = QDialog()
                editor = TemplateEditor(dialog)
                editor._timer.stop()
                editor._render()
            else:
                job = PreviewJob(self.session, self.doc.revision, self.session.preview_path)
                self.session._preview_job = job
                QThreadPool.globalInstance().start(job)
            try:
                self.assertTrue(started.wait(5))
                before = time.monotonic()
                if template:
                    dialog.reject()
                else:
                    self.window.close()
                elapsed = time.monotonic() - before
                self.assertLess(elapsed, 3)
                self.assertTrue(all(child.poll() is not None for child in children))
                self.app.processEvents()
                self.assertEqual(errors.getvalue(), "")
                print(f"{'template' if template else 'preview'} close: {elapsed:.3f}s; child gone; no error")
            finally:
                if dialog is not None:
                    editor.cancel_jobs()
                    dialog.deleteLater()
                for child in children:
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=3)
                QThreadPool.globalInstance().waitForDone(3000)

    def test_close_cancels_preview_measurement_child(self):
        self._cancel_other_worker()

    def test_template_dialog_reject_cancels_measurement_child(self):
        self._cancel_other_worker(template=True)


if __name__ == "__main__":
    unittest.main()
