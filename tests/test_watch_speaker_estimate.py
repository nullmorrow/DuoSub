"""Cold speaker presentation and phased geometry (offscreen; real libass probes)."""
import math
import unittest
from unittest.mock import Mock, patch
from PySide6.QtCore import QPointF, Qt
from tests import test_convert_ui as fixtures
from tests.test_transform import document
from subkit.ass import Event
from subkit.core.layout_index import LayoutIndex, LineGeom, Rect
from subkit_app.watch import GeometryJob, WatchController, OV_SPEAKER


class SpeakerEstimate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ConversionUI.setUpClass.__func__(cls)
        from PySide6.QtGui import QFontDatabase
        if not QFontDatabase.families():
            if QFontDatabase.addApplicationFont("C:/Windows/Fonts/arial.ttf") < 0:
                raise RuntimeError("Offscreen font setup failed")

    def setUp(self):
        self.player = fixtures.PlayerDouble()
        self.player.osd = dict(w=960, h=540)
        self.watch = WatchController(self.player)
        self.watch.ass = document()
        self.watch.playres = self.watch.ass.playres
        self.watch.ass.styles[0].set("Fontsize", 40)
        self.watch.ass.styles[0].set("Alignment", 2)
        self.watch.ass.events = [Event(text="Hello World", start=0, end=200)]
        self.watch.owners = [dict(group="g", member="m", kind="base")]
        self.watch.active = Mock(return_value=[0])
        self.watch._speakers_enabled = Mock(return_value=True)
        self.watch._doc = Mock(return_value=Mock(members={"m": Mock(lane="translation")}))
        self.watch._doc.return_value.members_of.return_value = []
        self.watch._speaker_text = Mock(return_value=("? Alice", "&H0000FF&"))
        self.settings = Mock()
        self.settings.value.return_value = "original"
        self.parent_patch = patch.object(self.watch, "parent", return_value=Mock(settings=self.settings))
        self.parent_patch.start()
        self.addCleanup(self.parent_patch.stop)
        self.addCleanup(self.dispose)

    def dispose(self):
        self.watch.stop_warm()
        self.watch.deleteLater()
        self.player.deleteLater()
        self.app.processEvents()

    def test_draws_before_geometry_then_only_one_measured_jump(self):
        with patch.object(self.watch, "_layout_index", side_effect=AssertionError("GUI probe")):
            self.watch._draw_speakers()
        initial = self.player.overlay.call_args.args[1]
        self.assertIn(r"\1c&H0000FF&}?", initial)
        self.assertEqual(self.watch.geo, {})
        self.assertEqual(self.watch.speaker_hits, [])
        coarse = LineGeom(0, "m", "g", "base", 0, Rect(300, 340, 500, 375), chars=[Rect(300, 340, 320, 375)])
        self.watch.geo[0] = coarse
        self.watch._draw_speakers()
        measured = self.player.overlay.call_args.args[1]
        self.assertNotEqual(initial, measured)
        self.assertEqual(len(self.watch.speaker_hits), 1)
        self.watch.geo[0] = LineGeom(0, "m", "g", "base", 0, Rect(301, 341, 501, 376),
                                    layout_origins=((301, 335),), line_height=40)
        self.watch._draw_speakers()
        self.assertEqual(self.player.overlay.call_args.args[1], measured)

    def test_released_cold_click_does_not_leave_a_drag(self):
        self.watch.request_geometry = Mock()
        p, button, mods = QPointF(400, 350), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
        self.watch._press(p, button, mods)
        self.watch._release(p, button, mods)
        self.assertEqual(self.watch._pending[0], "_deferred_click")
        self.watch.geo[0] = LineGeom(0, "m", "g", "base", 0, Rect(350, 320, 500, 380))
        # Isolate the subtitle selection from speaker picking.
        self.watch._speakers_enabled.return_value = False
        picked = Mock()
        self.watch.groupPicked.connect(picked)
        kind, args = self.watch._pending
        self.watch._pending = None
        getattr(self.watch, kind)(*args)
        picked.assert_called_once_with("g")
        self.assertIsNone(self.watch.drag)

    def test_estimate_matches_real_libass_first_line_styles(self):
        for text, alignment in (("Hello World", 2), ("Hello World", 1),
                                (r"{\an7\pos(300,100)}Hello World", 2),
                                (r"{\an2\pos(480,350)\fscx140\fscy80\fsp2\fs50}Hello World", 2),
                                (r"Short\NA much longer second line", 2)):
            with self.subTest(text=text, alignment=alignment):
                self.watch.ass.styles[0].set("Alignment", alignment)
                self.watch.ass.events[0].margin_l = 160
                self.watch.ass.events[0].margin_r = 80
                self.watch.ass.events[0].text = text
                estimate = self.watch._estimate_speaker_geometry(0)
                li = LayoutIndex(self.watch.ass, self.watch.owners)
                try:
                    measured = li.precise_geometry([0])[0]
                finally:
                    li.close()
                ink = measured._frames[0][0]
                distance = math.hypot(estimate.rect.x0 - ink.x0,
                                      estimate.layout_origins[0][1] + estimate.line_height -
                                      measured.layout_origins[0][1] - measured.line_height)
                self.assertLess(distance, 20, (text, distance))

    def test_cold_click_and_double_click_wait_without_gui_probes(self):
        self.watch.request_geometry = Mock()
        self.watch._press_now = Mock()
        self.watch._layout_index = Mock(side_effect=AssertionError("GUI probe"))
        self.watch._press(QPointF(400, 350), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.assertEqual(self.watch._pending[0], "_press")
        self.watch._press_now.assert_not_called()
        self.watch._double(QPointF(400, 350), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.assertEqual(self.watch._pending[0], "_double")
        self.watch._layout_index.assert_not_called()
        self.assertTrue(self.watch.request_geometry.called)


class GeometryPhases(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]

    def test_coarse_published_before_precise_and_cleanup(self):
        job = GeometryJob(7, document(), [dict(kind="base")], [0], None)
        coarse, precise = Mock(index=0), Mock(index=0)
        li = Mock()
        li.geometry.return_value = [coarse]
        calls = []
        job.signals.coarse.connect(lambda serial, geo, anchors: calls.append(("coarse", serial, geo[0])))
        def measure(indices):
            self.assertEqual(calls, [("coarse", 7, coarse)])
            return [precise]
        li.precise_geometry.side_effect = measure
        job.signals.done.connect(lambda serial, geo, anchors: calls.append(("precise", serial, geo[0])))
        with patch("subkit_app.watch.LayoutIndex", return_value=li):
            job.run()
        self.assertEqual(calls, [("coarse", 7, coarse), ("precise", 7, precise)])
        li.close.assert_called_once()
        self.assertTrue(job.finished.is_set())

    def test_cancel_between_phases_closes_without_second_signal(self):
        job = GeometryJob(7, document(), [dict(kind="base")], [0], None)
        li = Mock()
        li.geometry.return_value = [Mock(index=0)]
        job.signals.coarse.connect(lambda *_args: job.cancel())
        done = Mock()
        job.signals.done.connect(done)
        with patch("subkit_app.watch.LayoutIndex", return_value=li):
            job.run()
        li.precise_geometry.assert_not_called()
        done.assert_not_called()
        li.close.assert_called_once()
        self.assertTrue(job.finished.is_set())
