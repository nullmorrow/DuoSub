"""Crash/health log: session markers, uncaught errors, stall notes, rotation and clean restore of global hooks."""

from __future__ import annotations

import faulthandler
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication

from subkit_app import crash_log


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class CrashLogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)
        self.parent = QObject()
        self.addCleanup(self.parent.deleteLater)
        self.fault_was_enabled = faulthandler.is_enabled()
        self.thread_hook = threading.excepthook

    def start(self, **kwargs) -> crash_log.CrashLog:
        log = crash_log.CrashLog(self.parent, self.folder, version="9.9.9", **kwargs)
        self.addCleanup(log.close)
        return log

    def text(self) -> str:
        return (self.folder / crash_log.LOG_NAME).read_text(encoding="utf-8")

    def test_folder_defaults_to_local_app_data_and_honours_override(self):
        with patch.dict(os.environ, {"LOCALAPPDATA": str(self.folder)}, clear=False):
            os.environ.pop("SUBKIT_LOG_DIR", None)
            self.assertEqual(crash_log.log_dir(), self.folder / "DuoSub" / "logs")
            with patch.dict(os.environ, {"SUBKIT_LOG_DIR": str(self.folder / "x")}):
                self.assertEqual(crash_log.log_dir(), self.folder / "x")

    def test_clean_session_has_start_and_end_and_next_start_does_not_complain(self):
        self.start().close()
        log = self.start()
        log.close()
        text = self.text()
        self.assertEqual(text.count("session start: DuoSub 9.9.9"), 2)
        self.assertTrue(text.rstrip().endswith(crash_log.END_MARK))
        self.assertNotIn("did not end cleanly", text)

    def test_cut_off_session_is_reported_with_its_last_record(self):
        log = self.start()
        log.write("health: private_mb 1234")
        log.close()
        path = self.folder / crash_log.LOG_NAME     # a hard stop leaves no end line
        path.write_text("".join(path.read_text(encoding="utf-8").splitlines(keepends=True)[:-1]), encoding="utf-8")
        self.start().close()
        self.assertIn("previous session did not end cleanly; its last record: ", self.text())
        self.assertIn("health: private_mb 1234", self.text().split("did not end cleanly")[1])

    def test_uncaught_errors_from_gui_and_worker_threads_are_logged(self):
        log = self.start()
        try:
            raise ValueError("gui boom")
        except ValueError:
            log.exception(*sys.exc_info())
        with patch.object(log, "_previous_thread_hook") as previous:
            worker = threading.Thread(target=lambda: 1 / 0, name="preview-worker")
            worker.start()
            worker.join()
            previous.assert_called_once()
        text = self.text()
        self.assertIn("uncaught error (GUI thread):", text)
        self.assertIn("ValueError: gui boom", text)
        self.assertIn("uncaught error (thread preview-worker):", text)
        self.assertIn("ZeroDivisionError", text)

    def test_stalls_are_measured_noted_and_reported_in_health(self):
        clock = Clock()
        log = self.start(clock=clock)
        clock.now += 30
        log.heartbeat()                        # first beat: start-up time is not a stall
        clock.now += 1.0
        log.heartbeat()                        # on time
        clock.now += 1.0 + 7.5
        log.heartbeat()                        # 7.5 s late
        clock.now += 1.0 + 40
        log.heartbeat()                        # long enough for the faulthandler dump
        with patch.object(crash_log, "process_health", return_value={"private_mb": 2000, "handles": 5300}):
            log.health()
            log.health()
        lines = self.text().splitlines()
        self.assertEqual(sum("did not respond" in line for line in lines), 2)
        self.assertTrue(any("GUI thread did not respond for 7.5 s" in line and "fault.log" not in line for line in lines))
        self.assertTrue(any("did not respond for 40.0 s; stacks in fault.log" in line for line in lines))
        health = [line for line in lines if "health:" in line]
        self.assertIn("private_mb 2000, handles 5300", health[0])
        self.assertIn("longest stall 40000 ms", health[0])
        self.assertIn("longest stall 0 ms", health[1])     # the maximum restarts each minute

    @unittest.skipUnless(os.name == "nt", "Windows process counters")
    def test_process_health_reads_real_counters(self):
        values = crash_log.process_health()
        self.assertGreater(values["private_mb"], 0)
        self.assertGreater(values["handles"], 0)
        self.assertGreaterEqual(values["gdi"], 0)

    def test_large_logs_rotate_to_one_old_generation(self):
        for name in (crash_log.LOG_NAME, crash_log.FAULT_NAME):
            (self.folder / name).write_bytes(b"x" * (crash_log.ROTATE_BYTES + 1))
        self.start().close()
        for name in (crash_log.LOG_NAME, crash_log.FAULT_NAME):
            self.assertEqual((self.folder / (name + ".1")).stat().st_size, crash_log.ROTATE_BYTES + 1)
            self.assertLess((self.folder / name).stat().st_size, 1000)

    def test_close_restores_global_hooks(self):
        log = self.start()
        self.assertTrue(faulthandler.is_enabled())
        self.assertNotEqual(threading.excepthook, self.thread_hook)
        log.close()
        self.assertEqual(faulthandler.is_enabled(), self.fault_was_enabled and sys.stderr is not None)
        self.assertEqual(threading.excepthook, self.thread_hook)
        log.close()                            # closing twice is harmless

    def test_unwritable_folder_never_blocks_start_up(self):
        blocker = self.folder / "file"
        blocker.write_text("not a folder")
        log = crash_log.CrashLog(self.parent, blocker / "logs")
        log.write("ignored")
        log.health()
        log.close()
        self.assertEqual(threading.excepthook, self.thread_hook)

    def test_install_is_once_per_application(self):
        with patch.dict(os.environ, {"SUBKIT_LOG_DIR": str(self.folder)}):
            log = crash_log.install(self.app, "1.0")
            self.addCleanup(lambda: (log.close(), delattr(self.app, "_crash_log")))
            self.assertIs(crash_log.install(self.app, "1.0"), log)
        self.assertEqual(self.text().count("session start: DuoSub 1.0"), 1)


if __name__ == "__main__":
    unittest.main()
