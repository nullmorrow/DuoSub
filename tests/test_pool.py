"""Background jobs are released after they run, and queued jobs can still be taken back."""

from __future__ import annotations

import gc
import os
from pathlib import Path
import sys
import threading
import unittest
import weakref

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QRunnable, QThreadPool
from PySide6.QtWidgets import QApplication

from subkit_app import pool as job_pool


class Job(QRunnable):
    def __init__(self, gate: threading.Event | None = None) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.gate = gate
        self.ran = threading.Event()
        self.payload = bytearray(1_000_000)

    def run(self) -> None:
        if self.gate is not None:
            self.gate.wait(5)
        self.ran.set()


class PoolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)
        self.addCleanup(self.pool.waitForDone, 5000)

    def test_finished_jobs_are_not_kept_by_the_pool(self):
        refs = []
        for _ in range(10):
            job = Job()
            refs.append(weakref.ref(job))
            job_pool.start(job, pool=self.pool)
            self.assertTrue(job.ran.wait(5))
            del job
        self.assertTrue(self.pool.waitForDone(5000))
        self.app.processEvents()
        gc.collect()
        self.assertEqual(sum(r() is not None for r in refs), 0)

    def test_plain_start_of_a_kept_job_does_leak_on_this_pyside(self):
        # Documents why the helper exists; if PySide stops keeping these, the helper is harmless.
        job = Job()
        ref = weakref.ref(job)
        self.pool.start(job)
        self.assertTrue(job.ran.wait(5))
        del job
        self.pool.waitForDone(5000)
        gc.collect()
        if ref() is None:
            self.skipTest("this PySide releases finished non-auto-delete runnables")
        self.assertIsNotNone(ref())

    def test_queued_job_can_be_taken_back_and_never_runs(self):
        gate = threading.Event()
        blocker, queued = Job(gate), Job()
        job_pool.start(blocker, pool=self.pool)
        job_pool.start(queued, pool=self.pool)
        self.assertTrue(job_pool.try_take(queued, self.pool))
        self.assertFalse(job_pool.try_take(queued, self.pool))      # already taken
        gate.set()
        self.assertTrue(blocker.ran.wait(5))
        self.assertTrue(self.pool.waitForDone(5000))
        self.assertFalse(queued.ran.is_set())
        ref = weakref.ref(queued)
        del queued
        gc.collect()
        self.assertIsNone(ref())

    def test_running_or_finished_jobs_cannot_be_taken(self):
        gate = threading.Event()
        job = Job(gate)
        job_pool.start(job, pool=self.pool)
        while not job._pool_carrier.started:
            pass
        self.assertFalse(job_pool.try_take(job, self.pool))
        gate.set()
        self.assertTrue(job.ran.wait(5))
        self.assertTrue(self.pool.waitForDone(5000))
        self.assertFalse(job_pool.try_take(job, self.pool))        # the carrier is deleted by now
        self.assertFalse(job_pool.try_take(Job(), self.pool))       # never started

    def test_priority_orders_the_queue(self):
        gate, order = threading.Event(), []

        class Tagged(Job):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def run(self):
                order.append(self.name)

        job_pool.start(Job(gate), pool=self.pool)
        low, high = Tagged("low"), Tagged("high")
        job_pool.start(low, -1, pool=self.pool)
        job_pool.start(high, 1, pool=self.pool)
        gate.set()
        self.assertTrue(self.pool.waitForDone(5000))
        self.assertEqual(order, ["high", "low"])


if __name__ == "__main__":
    unittest.main()
