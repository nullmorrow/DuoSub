"""Foreground geometry must not wait for look-ahead; Qt signals and held jobs."""
import copy
import unittest
from unittest.mock import Mock, patch

from tests import test_convert_ui as fixtures
from subkit.ass import AssDocument, Event
from subkit_app.watch import WatchController


class WatchGeometry(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]

    def setUp(self):
        self.player = fixtures.PlayerDouble()
        self.watch = WatchController(self.player)
        self.watch.ass = AssDocument()
        self.watch.ass.events = [Event(text="a"), Event(text="b"), Event(text="c")]
        self.watch.owners = [{"group": str(i), "member": str(i), "kind": "base"} for i in range(3)]
        self.watch.redraw = Mock()
        self.watch._video_height = Mock(return_value=None)
        self.watch.active = Mock(return_value=[])
        self.watch.upcoming = Mock(return_value=[1])
        self.pool = Mock()
        self.pool.tryTake.return_value = True
        p = patch("subkit_app.watch.QThreadPool.globalInstance", return_value=self.pool)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.dispose)

    def dispose(self):
        self.watch.stop_warm()
        self.watch.deleteLater()
        self.player.deleteLater()
        self.app.processEvents()

    def jobs(self):
        self.watch.request_geometry()
        ahead = self.pool.start.call_args.args[0]
        self.watch.active.return_value = [0]
        self.watch.request_geometry()
        foreground = self.pool.start.call_args.args[0]
        return ahead, foreground

    def complete(self, job, geo, anchors):
        job.finished.set()
        job.signals.done.emit(job.serial, geo, anchors)
        self.app.processEvents()

    def test_active_request_starts_before_lookahead_finishes_and_is_not_duplicated(self):
        ahead, foreground = self.jobs()
        self.assertFalse(ahead.finished.is_set())
        self.assertEqual(ahead.indices, [1])
        self.assertEqual(foreground.indices, [0])
        self.assertEqual(self.pool.start.call_count, 2)
        self.assertEqual(self.pool.start.call_args.args[1], 1)
        self.watch.request_geometry()
        self.assertEqual(self.pool.start.call_count, 2)

    def test_both_concurrent_results_apply_in_either_order(self):
        for foreground_first in (True, False):
            with self.subTest(foreground_first=foreground_first):
                self.watch.geo.clear()
                self.watch.anchors.clear()
                self.watch.active.return_value = []
                ahead, foreground = self.jobs()
                order = (foreground, ahead) if foreground_first else (ahead, foreground)
                for job in order:
                    i = job.indices[0]
                    self.complete(job, {i: f"geometry {i}"}, {i: {"x": i}})
                self.assertEqual(self.watch.geo, {0: "geometry 0", 1: "geometry 1"})
                self.assertEqual(self.watch.anchors, {0: {"x": 0}, 1: {"x": 1}})
                self.assertFalse(self.watch._job_running)
                self.assertEqual(self.watch._geometry_jobs, {})

    def test_new_serial_starts_without_waiting_and_rejects_old_results(self):
        ahead, foreground = self.jobs()
        self.watch.serial += 1
        self.watch.upcoming.return_value = []
        self.watch.request_geometry()
        current = self.pool.start.call_args.args[0]
        self.assertEqual(current.serial, self.watch.serial)
        self.complete(ahead, {1: "stale"}, {1: {"x": -1}})
        self.complete(foreground, {0: "stale"}, {0: {"x": -1}})
        self.assertEqual(self.watch.geo, {})
        self.assertEqual(self.watch.anchors, {})
        self.assertTrue(self.watch._job_running)
        self.complete(current, {0: "current"}, {0: {"x": 0}})
        self.assertEqual(self.watch.geo, {0: "current"})
        self.assertFalse(self.watch._job_running)

    def test_stop_cancels_all_jobs_and_rejects_late_signals_after_restart(self):
        ahead, foreground = self.jobs()
        self.watch._pending = ("redraw", ())
        self.watch.stop_warm()
        self.assertTrue(all(j.cancelled and j.finished.is_set() for j in (ahead, foreground)))
        self.assertEqual(self.watch._geometry_jobs, {})
        self.assertFalse(self.watch._job_running)
        self.assertFalse(self.watch._job_wanted)
        self.assertIsNone(self.watch._pending)
        self.watch._measuring_stopped = False
        self.complete(ahead, {1: "late"}, {})
        self.complete(foreground, {0: "late"}, {})
        self.assertEqual(self.watch.geo, {})

    def test_unknown_sender_cannot_publish_current_serial_geometry(self):
        self.watch._geometry_done(self.watch.serial, {0: "unknown"}, {})
        self.assertEqual(self.watch.geo, {})

    def test_coarse_keeps_job_registered_until_precise_and_rejects_stale(self):
        ahead, foreground = self.jobs()
        foreground.signals.coarse.emit(foreground.serial, {0: "coarse"}, {})
        self.assertEqual(self.watch.geo[0], "coarse")
        self.assertIn(foreground.signals, self.watch._geometry_jobs)
        self.assertTrue(self.watch._job_running)
        self.complete(foreground, {0: "precise"}, {})
        self.assertEqual(self.watch.geo[0], "precise")
        self.watch.serial += 1
        ahead.signals.coarse.emit(ahead.serial, {1: "stale coarse"}, {})
        self.assertNotIn(1, self.watch.geo)
        self.complete(ahead, {1: "stale precise"}, {})
        self.assertNotIn(1, self.watch.geo)

    def preview(self, ass=None, owners=None):
        self.watch.set_preview(ass if ass is not None else copy.deepcopy(self.watch.ass),
                               owners if owners is not None else copy.deepcopy(self.watch.owners), {}, None)

    def test_unchanged_old_jobs_publish_coarse_then_precise_without_duplicate_work(self):
        ahead, foreground = self.jobs()
        self.preview()
        self.assertEqual(self.pool.start.call_count, 2)
        self.assertTrue(all(not j.cancelled and j.target_serial == self.watch.serial for j in (ahead, foreground)))
        foreground.signals.coarse.emit(foreground.serial, {0: "coarse"}, {0: {"x": 2}})
        self.assertEqual(self.watch.geo, {0: "coarse"})
        self.assertIn(foreground.signals, self.watch._geometry_jobs)
        self.complete(foreground, {0: "precise"}, {0: {"x": 3}})
        self.complete(ahead, {1: "ahead"}, {})
        self.assertEqual(self.watch.geo, {0: "precise", 1: "ahead"})
        self.assertEqual(self.watch.anchors[0], {"x": 3})
        self.assertFalse(self.watch._job_running)

    def test_mixed_job_reuses_only_unchanged_events_and_visible_change_overtakes(self):
        self.watch.upcoming.return_value = [0, 1]
        self.watch.request_geometry()
        old = self.pool.start.call_args.args[0]
        ass = copy.deepcopy(self.watch.ass)
        ass.events[0].text = "edited"
        self.watch.active.return_value = [0]
        self.preview(ass)
        current = self.pool.start.call_args.args[0]
        self.assertEqual(old.reusable_indices, {1})
        self.assertFalse(old.cancelled)
        self.assertEqual(current.indices, [0])
        self.assertEqual(self.pool.start.call_args.args[1], 1)
        self.watch.request_geometry()
        self.assertEqual(self.pool.start.call_count, 2)
        old.signals.coarse.emit(old.serial, {0: "stale coarse", 1: "reused coarse"}, {0: {"x": -1}, 1: {"x": 1}})
        self.assertEqual(self.watch.geo, {1: "reused coarse"})
        self.complete(current, {0: "new"}, {0: {"x": 2}})
        self.complete(old, {0: "stale precise", 1: "reused precise"}, {0: {"x": -1}, 1: {"x": 3}})
        self.assertEqual(self.watch.geo, {0: "new", 1: "reused precise"})
        self.assertEqual(self.watch.anchors, {0: {"x": 2}, 1: {"x": 3}})

    def test_repeated_previews_remap_transitively_without_reviving_changed_events(self):
        self.watch.upcoming.return_value = [0, 1]
        self.watch.request_geometry()
        old = self.pool.start.call_args.args[0]
        original = copy.deepcopy(self.watch.ass)
        changed = copy.deepcopy(original)
        changed.events[0].text = "edited"
        self.preview(changed)
        self.preview(original)
        self.assertEqual(old.reusable_indices, {1})
        self.assertEqual(old.target_serial, self.watch.serial)
        self.complete(old, {0: "revived", 1: "still unchanged"}, {})
        self.assertEqual(self.watch.geo, {1: "still unchanged"})

    def test_layout_owner_and_deleted_event_cancel_jobs(self):
        for change in ("styles", "playres", "owner", "delete"):
            with self.subTest(change=change):
                self.watch.upcoming.return_value = [1]
                self.watch.request_geometry()
                old = self.pool.start.call_args.args[0]
                ass, owners = copy.deepcopy(self.watch.ass), copy.deepcopy(self.watch.owners)
                if change == "styles":
                    ass.styles = copy.deepcopy(fixtures.AssDocument.parse(fixtures.default_styles_ass()).styles)
                elif change == "playres":
                    ass.set_info("PlayResX", str(ass.playres[0] + 100))
                elif change == "owner":
                    owners[1]["member"] = "new member"
                else:
                    ass.events = ass.events[:1]
                    owners = owners[:1]
                    self.watch.upcoming.return_value = []
                self.preview(ass, owners)
                self.assertTrue(old.cancelled)
                self.assertTrue(old.finished.is_set())
                self.assertNotIn(old.signals, self.watch._geometry_jobs)
                self.complete(old, {1: "stale"}, {1: {"x": -1}})
                self.assertNotIn(1, self.watch.geo)
                self.watch.stop_warm()
                self.watch._measuring_stopped = False

    def test_continuous_edits_cancel_queued_jobs_keep_only_latest(self):
        self.watch.active.return_value = [0]
        self.watch.upcoming.return_value = []
        self.watch.request_geometry()
        old_jobs = []
        for i in range(5):
            old_jobs.append(self.pool.start.call_args.args[0])
            ass = copy.deepcopy(self.watch.ass)
            ass.events[0].text = f"edit {i}"
            self.preview(ass)
            self.assertEqual(len(self.watch._geometry_jobs), 1)
        self.assertTrue(all(j.cancelled and j.finished.is_set() for j in old_jobs))
        newest = self.pool.start.call_args.args[0]
        self.complete(newest, {0: "newest"}, {})
        self.assertEqual(self.watch.geo, {0: "newest"})

    def test_cancel_running_job_retains_stop_tracking_until_resource_closure(self):
        self.watch.request_geometry()
        old = self.pool.start.call_args.args[0]
        self.pool.tryTake.return_value = False
        ass = copy.deepcopy(self.watch.ass)
        ass.events[1].text = "changed"
        self.preview(ass)
        self.assertTrue(old.cancelled)
        self.assertFalse(old.finished.is_set())
        self.assertIn(old.signals, self.watch._geometry_jobs)
        old.finished.set()
        self.watch.request_geometry()
        self.assertNotIn(old.signals, self.watch._geometry_jobs)
        self.pool.tryTake.return_value = True

    def test_reused_coarse_result_satisfies_pending_click(self):
        self.watch.active.return_value = [0]
        self.watch.upcoming.return_value = []
        self.watch.request_geometry()
        old = self.pool.start.call_args.args[0]
        self.watch._deferred_click = Mock()
        self.watch._pending = ("_deferred_click", ("click",))
        self.preview()
        old.signals.coarse.emit(old.serial, {0: "coarse"}, {})
        self.watch._deferred_click.assert_called_once_with("click")
        self.assertIsNone(self.watch._pending)

    def test_done_queued_before_preview_can_still_publish_unchanged_result(self):
        self.watch.request_geometry()
        old = self.pool.start.call_args.args[0]
        old.finished.set()
        self.preview()
        self.complete(old, {1: "finished before preview"}, {})
        self.assertEqual(self.watch.geo, {1: "finished before preview"})
