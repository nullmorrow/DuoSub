"""F4: offscreen Qt with player double, real files and cooperative slow jobs."""
import copy
import itertools
import shutil
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests import test_convert_ui as fixtures
from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QMessageBox
from subkit.core.workspace import Project, Episode
from subkit.proc import CancelledError
from subkit_app.episode_job import EpisodeJob


class ReviewF4(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__['setUpClass']
    _dispose = fixtures.ConversionUI._dispose

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.window.close_session()
        self.project = Project.create(self.path / 'project', 'test', original='ja', translation='zh-Hans')
        self.ep = self.project.add_episode('E01')
        for lane, text in [('translation', 'hello'), ('original', 'こんにちは')]:
            source = self.path / (lane + '.srt')
            source.write_text('1\n00:00:01,000 --> 00:00:02,000\n' + text + '\n', encoding='utf-8')
            self.ep.add_source(source, lane=lane, lang='ja' if lane == 'original' else 'zh-Hans')
        self.ep.merge()
        self.before = (self.ep.dir / 'document.json').read_bytes()
        self.window.project = self.project
        self.window.ep_box.addItem('E01')
        self.window._start_session = Mock()
        self.window._load_media = Mock()
        for method in ['information', 'warning', 'question']:
            p = patch.object(QMessageBox, method, return_value=QMessageBox.StandardButton.Yes)
            mock = p.start(); self.addCleanup(p.stop)
            setattr(self, method, mock)
        self.dialog = patch('subkit_app.mainwindow.CheckResultDialog').start()
        self.addCleanup(patch.stopall)
        self.addCleanup(self._stop_job)   # runs first, while the message boxes are still patched

    def _stop_job(self):
        job = self.window._episode_job
        if job is not None:
            self.window._cancel_episode_job()
            job.finished.wait(5)
            self.app.processEvents()

    def wait(self, job):
        deadline = time.monotonic() + 5
        while self.window._episode_job is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertTrue(job.finished.wait(2))
        self.app.processEvents()
        self.assertIsNone(self.window._episode_job)

    def test_responsive_cancel_preserves_bytes_and_restores_menus(self):
        ticks = []
        timer = QTimer(); timer.setInterval(5); timer.timeout.connect(lambda: ticks.append(1)); timer.start()
        real = Episode.merge
        def slowed(ep, **kwargs):
            for _ in range(30):
                time.sleep(.01); kwargs['cancel_check']()
            return real(ep, **kwargs)
        with patch.object(Episode, 'merge', slowed):
            job = EpisodeJob(self.ep, 'merge', settings={})
            self.window._start_episode_job(job)
            self.assertFalse(self.window.ep_box.isEnabled())
            for action in self.window._episode_actions:
                self.assertFalse(action.isEnabled()); self.assertIn('（正在添加一集）', action.text())
            until = time.monotonic() + .1
            while time.monotonic() < until:
                self.app.processEvents(); time.sleep(.005)
            self.assertGreater(len(ticks), 3)
            self.assertEqual(self.window._episode_progress.step.text(), '正在对齐…')
            self.window._cancel_episode_job(); self.wait(job)
        timer.stop()
        self.assertEqual((self.ep.dir / 'document.json').read_bytes(), self.before)
        self.assertTrue(self.window.ep_box.isEnabled())
        self.assertTrue(all(a.isEnabled() for a in self.window._episode_actions))
        self.assertIn('已取消，这一集没有改动。', self.information.call_args.args)
        self.window._start_session.assert_not_called()

    def test_success_session_and_result_on_gui_thread(self):
        threads = []
        self.window._start_session.side_effect = lambda ep: threads.append(QThread.currentThread())
        job = EpisodeJob(self.ep, 'merge', settings={})
        self.window._start_episode_job(job); self.wait(job)
        self.assertEqual(threads, [self.app.thread()])
        self.dialog.assert_called_once()
        self.warning.assert_not_called()

    def test_error_restores_ui_without_writing(self):
        with patch.object(Episode, 'merge', side_effect=RuntimeError('merge error')):
            job = EpisodeJob(self.ep, 'merge', settings={})
            self.window._start_episode_job(job); self.wait(job)
        self.assertIn('合并失败', self.warning.call_args.args[2])
        self.assertTrue(self.window.ep_box.isEnabled())
        self.assertEqual((self.ep.dir / 'document.json').read_bytes(), self.before)

    def test_close_asks_and_cancels(self):
        def slow(ep, **kwargs):
            while True:
                time.sleep(.01); kwargs['cancel_check']()
        with patch.object(Episode, 'merge', slow):
            job = EpisodeJob(self.ep, 'merge', settings={})
            self.window._start_episode_job(job)
            self.question.return_value = QMessageBox.StandardButton.No
            self.window.close(); self.assertFalse(job._cancel.is_set())
            self.question.return_value = QMessageBox.StandardButton.Yes
            start = time.monotonic(); self.window.close()
            self.assertLess(time.monotonic()-start, 2.2)
            self.wait(job)
        self.assertEqual((self.ep.dir / 'document.json').read_bytes(), self.before)
        self.window._start_session.assert_not_called()

    def test_close_during_source_import(self):
        ep = self.project.add_episode('E02')
        imports = dict(cn=str(self.path/'translation.srt'), jp=str(self.path/'original.srt'), roles={}, video='', audio=None)
        entered = threading.Event()
        real = Episode.add_source
        def slow(ep, *args, **kwargs):
            entered.set(); time.sleep(.08)
            return real(ep, *args, **kwargs)
        with patch.object(Episode, 'add_source', slow):
            job = EpisodeJob(ep, 'import', imports=imports)
            self.window._start_episode_job(job)
            self.assertTrue(entered.wait(1))
            self.window.close(); self.wait(job)
        self.assertFalse((ep.dir/'document.json').exists())
        self.assertEqual((self.ep.dir/'document.json').read_bytes(), self.before)
        self.window._start_session.assert_not_called()

    def test_semantic_factory_is_worker_owned_and_closed(self):
        threads = []
        factory = Mock(return_value=None); factory.close = Mock()
        def make(project, **kwargs):
            threads.append(QThread.currentThread()); return factory
        job = EpisodeJob(self.ep, 'merge', settings={'semantic': True}, semantic_factory=make)
        self.window._start_episode_job(job); self.wait(job)
        self.assertNotEqual(threads[0], self.app.thread())
        factory.close.assert_called_once()

    def test_import_steps_and_no_document_until_merge(self):
        ep = self.project.add_episode('E02')
        imports = dict(cn=str(self.path/'translation.srt'), jp=str(self.path/'original.srt'), roles={}, video='', audio=None)
        self.window._offer_merge = Mock()
        job = EpisodeJob(ep, 'import', imports=imports)
        progress = []; job.signals.progress.connect(progress.append)
        self.window._start_episode_job(job); self.wait(job)
        self.assertEqual(progress, ['正在读取翻译（简中）字幕…', '正在读取原声（日语）字幕…'])
        self.assertFalse((ep.dir/'document.json').exists())
        self.assertEqual(len(self.project.episode('E02').sources), 2)
        self.window._offer_merge.assert_called_once()

    def test_reanalysis_runs_in_worker_then_offers_merge(self):
        self.window._reanalyzed = Mock()
        original = Episode.reanalyze_sources
        threads = []
        def analyze(ep, **kwargs):
            threads.append(QThread.currentThread())
            return original(ep, **kwargs)
        with patch.object(Episode, 'reanalyze_sources', analyze):
            job = EpisodeJob(self.ep, 'reanalyze')
            self.window._start_episode_job(job); self.wait(job)
        self.assertNotEqual(threads[0], self.app.thread())
        self.window._reanalyzed.assert_called_once()
        self.assertEqual((self.ep.dir/'document.json').read_bytes(), self.before)

    def test_cancel_before_save_is_not_written(self):
        job = EpisodeJob(self.ep, 'merge', settings={})
        real = Episode.merge
        def cancel_after_merge(ep, **kwargs):
            result = real(ep, **kwargs)
            job.cancel()
            return result
        with patch.object(Episode, 'merge', cancel_after_merge):
            self.window._start_episode_job(job); self.wait(job)
        self.assertEqual((self.ep.dir/'document.json').read_bytes(), self.before)
        self.window._start_session.assert_not_called()

    def test_core_cancel_during_dp_row(self):
        from subkit.align.dp2 import Aligner2, Row
        calls = []
        def check():
            calls.append(1)
            if len(calls) == 2: raise CancelledError()
        with self.assertRaises(CancelledError):
            Aligner2([Row(0, 0, 1000, 'a', 0), Row(1, 1000, 2000, 'b', 1)], [], cancel_check=check).run()
        self.assertEqual(len(calls), 2)

    def test_soma_background_document_byte_identical_to_sync(self):
        root = Path(__file__).resolve().parents[1]
        shutil.copytree(root/'test_projects'/'soma', self.path/'soma', copy_function=shutil.copyfile,
                        ignore=shutil.ignore_patterns('cache'))
        ep = Project.open(self.path/'soma').episode('S01E001')
        ep.document = None
        ep.data['media'] = {'video': None, 'audio_track': None}
        original = copy.deepcopy(ep)
        def ids():
            counter = itertools.count()
            return lambda prefix: prefix + str(next(counter))
        def patches(fn):
            from contextlib import ExitStack
            stack = ExitStack()
            for target in ['subkit.core.merge.new_id', 'subkit.core.readings.new_id']:
                stack.enter_context(patch(target, fn))
            return stack
        with patches(ids()): ep.merge(settings={})
        expected = (ep.dir/'document.json').read_bytes()
        job = EpisodeJob(original, 'merge', settings={})
        with patches(ids()):
            self.window._start_episode_job(job)
            deadline = time.monotonic()+60
            while self.window._episode_job is not None and time.monotonic()<deadline:
                self.app.processEvents(); time.sleep(.01)
            self.wait(job)
        self.assertEqual((ep.dir/'document.json').read_bytes(), expected)
