"""Real software-render probes and deterministic cancellation/fallback coverage."""
import os
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from subkit.ass import AssDocument, Style
from subkit.proc import CancelledError, KillableRun
from subkit.ruby.measure import LineRequest, Measurer
from subkit_app.player.measure import ProbeRenderer


def document():
    doc = AssDocument()
    doc.set_info("PlayResX", 640)
    doc.set_info("PlayResY", 360)
    doc.styles = [Style.new("Default", Fontname="Arial", Fontsize="36")]
    return doc


# The libmpv measurer is not wired into the app (kept as a backup). Since 2026-10-11 this module
# intermittently crashes the full suite natively (access violation; never when run alone); run it on demand.
@unittest.skipUnless(os.environ.get("SUBKIT_MPV_TESTS") == "1", "set SUBKIT_MPV_TESTS=1 to run the libmpv measurer tests")
class SoftwareMeasurement(unittest.TestCase):
    def setUp(self):
        self.renderer = ProbeRenderer()
        self.addCleanup(self.renderer.close)

    def test_real_repeated_multiframe_resize_and_empty_probes(self):
        doc = document()
        # The small bands force multiple probe timestamps; repeat with a new ASS.
        for scale, text in ((1, "字幕 Hello"), (4, r"{\fs52}川いハ"), (1, "Goodbye"), (1, "   ")):
            with self.subTest(scale=scale, text=text):
                reqs = [LineRequest("Default", text + str(i), 8, 320) for i in range(20)]
                with Measurer(doc, raster_scale=scale) as old, Measurer(doc, raster_scale=scale, renderer=self.renderer) as new:
                    expected = old.line_char_boxes(reqs)
                    with patch.object(new, "_render_ffmpeg", side_effect=AssertionError("unexpected fallback")):
                        actual = new.line_char_boxes(reqs)
                    for a, b in zip(expected, actual):
                        self.assertEqual([x is None for x in a], [x is None for x in b])
                        for x, y in zip(a, b):
                            if x is not None:
                                self.assertLessEqual(max(abs(getattr(x, k) - getattr(y, k)) for k in ("x0", "x1", "y0", "y1")), 1)
        self.assertIsNone(self.renderer.failure)

    def test_cache_identity_and_ffmpeg_fallback(self):
        doc = document()
        failed = Mock(cache_id="mpv-sw1")
        failed.render.side_effect = TimeoutError("native timeout")
        with Measurer(doc) as old, Measurer(doc, renderer=failed) as new:
            self.assertNotEqual(old._key("Default", "abc", 1, 320), new._key("Default", "abc", 1, 320))
            self.assertEqual(old.text_boxes([("Default", "Hello")]), new.text_boxes([("Default", "Hello")]))
        failed.render.assert_called_once()

    def test_cancellation_never_falls_back(self):
        renderer = Mock(cache_id="mpv-sw1")
        renderer.render.side_effect = CancelledError()
        with Measurer(document(), renderer=renderer) as m, patch.object(m, "_render_ffmpeg") as ffmpeg:
            with self.assertRaises(CancelledError):
                m.text_boxes([("Default", "Hello")])
            ffmpeg.assert_not_called()

    def test_native_owner_serializes_and_closes_on_same_thread(self):
        calls = []
        class Native:
            def __init__(self):
                calls.append(("create", threading.get_ident()))
            def render(self, *args):
                calls.append(("render", threading.get_ident()))
                return np.zeros((1, 2, 2), dtype=np.uint8), set()
            def close(self):
                calls.append(("close", threading.get_ident()))
        renderer = ProbeRenderer(factory=Native)
        threads = [threading.Thread(target=lambda: renderer.render("", 2, 2, 1, KillableRun())) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
            self.assertFalse(thread.is_alive())
        renderer.close()
        self.assertEqual([c[0] for c in calls], ["create", "render", "render", "render", "close"])
        self.assertEqual(len({c[1] for c in calls}), 1)
        self.assertNotEqual(calls[0][1], threading.get_ident())

    def test_close_cancels_inflight_and_queued_requests(self):
        started = threading.Event()
        closed = threading.Event()
        class Native:
            def render(self, content, w, h, n, check):
                started.set()
                while True:
                    check()
                    time.sleep(.001)
            def close(self):
                closed.set()
        renderer = ProbeRenderer(factory=Native)
        errors = []
        def request():
            try:
                renderer.render("", 2, 2, 1, KillableRun())
            except CancelledError:
                errors.append("cancelled")
        threads = [threading.Thread(target=request) for _ in range(2)]
        for thread in threads:
            thread.start()
        self.assertTrue(started.wait(1))
        renderer.close()
        for thread in threads:
            thread.join(1)
            self.assertFalse(thread.is_alive())
        self.assertTrue(closed.is_set())
        self.assertEqual(errors, ["cancelled", "cancelled"])

    def test_timeout_disables_native_and_releases_owner(self):
        closed = Mock()
        class Native:
            def render(self, content, w, h, n, check):
                while True:
                    check()
                    time.sleep(.001)
            close = closed
        renderer = ProbeRenderer(factory=Native, timeout=.02)
        try:
            with self.assertRaises(TimeoutError):
                renderer.render("", 2, 2, 1, KillableRun())
            with self.assertRaises(RuntimeError):
                renderer.render("", 2, 2, 1, KillableRun())
        finally:
            renderer.close()
        closed.assert_called_once()

    def test_cancelled_before_start_owns_no_thread(self):
        runner = KillableRun()
        runner.cancel()
        with self.assertRaises(CancelledError):
            self.renderer.render("", 2, 2, 1, runner)
        self.assertIsNone(self.renderer.thread)
