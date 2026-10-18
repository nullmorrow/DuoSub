"""Real offscreen Qt local sockets in one process; startup/window doubles."""
import os
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from PySide6.QtNetwork import QLocalSocket, QLocalServer
from subkit_app.single_instance import SingleInstance, activate_window, prepare_instance, server_name
from subkit_app import app as entry


class SingleInstanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.name = "DuoSub-test-" + uuid.uuid4().hex
        self.server = SingleInstance(self.name)
        self.client = SingleInstance(self.name)
        self.addCleanup(self.server.deleteLater)
        self.addCleanup(self.client.deleteLater)
        self.addCleanup(self.server.close)
        self.addCleanup(self.client.close)

    def test_real_handshake_second_exits_and_queued_unicode_arguments(self):
        self.assertTrue(self.server.acquire([]))
        args = ["C:/工程/字幕", "--option"]
        self.assertFalse(self.client.acquire(args))
        handler = Mock()
        self.server.set_handler(handler)
        self.app.processEvents()
        handler.assert_called_once_with(args)
        self.assertFalse(self.client.server.isListening())

    def test_real_handshake_running_handler_and_restart(self):
        self.assertTrue(self.server.acquire([]))
        handler = Mock()
        self.server.set_handler(handler)
        self.assertFalse(self.client.acquire([]))
        self.app.processEvents()
        handler.assert_called_once_with([])
        self.server.close()
        with patch.object(QLocalServer, "removeServer", wraps=QLocalServer.removeServer) as remove:
            self.assertTrue(self.client.acquire([]))
            remove.assert_called_once_with(self.name)

    def wait_until(self, condition):
        deadline = time.monotonic() + 2
        while not condition() and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(condition(), "Qt local socket delivery did not complete within 2 seconds")

    def test_fragmented_payload_and_invalid_request_close(self):
        self.assertTrue(self.server.acquire([]))
        socket = QLocalSocket()
        self.addCleanup(socket.close)
        socket.connectToServer(self.name)
        self.assertTrue(socket.waitForConnected(300))
        socket.write(b'["--flag"')
        socket.flush()
        self.app.processEvents()
        self.assertEqual(self.server.pending, [])
        socket.write(b']\n')
        socket.flush()
        self.wait_until(lambda: self.server.pending == [["--flag"]])
        self.assertEqual(self.server.pending, [["--flag"]])
        socket.abort()
        socket.connectToServer(self.name)
        self.assertTrue(socket.waitForConnected(300))
        socket.write(b'{"bad": 1}\n')
        socket.flush()
        self.wait_until(lambda: socket.state() == QLocalSocket.LocalSocketState.UnconnectedState)
        self.assertEqual(self.server.pending, [["--flag"]])

    def test_smoke_bypass_never_constructs_guard(self):
        with patch("subkit_app.single_instance.SingleInstance", side_effect=AssertionError("must bypass")):
            self.assertEqual(prepare_instance(["--smoke-test"]), (None, True))

    def test_user_name_and_relative_folder_forwarded_absolute(self):
        with patch("subkit_app.single_instance.getpass.getuser", return_value="test-user"):
            self.assertIn("test-user", server_name())
        with patch("subkit_app.single_instance.SingleInstance") as guard:
            guard.return_value.acquire.return_value = False
            prepare_instance(["--flag", "."])
            guard.return_value.acquire.assert_called_once_with(["--flag", str(Path.cwd())])

    def test_restore_activate_open_only_without_project(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TEMP"]) as folder:
            w = Mock()
            w.project = None
            w.isMinimized.return_value = True
            activate_window(w, [folder])
            w.showNormal.assert_called_once()
            w.raise_.assert_called_once()
            w.activateWindow.assert_called_once()
            w.open_project.assert_called_once_with(Path(folder).resolve())
            w.reset_mock()
            w.project = object()
            w.isMinimized.return_value = False
            activate_window(w, [folder])
            w.open_project.assert_not_called()
            w.showNormal.assert_not_called()
            self.assertEqual(w.statusBar().showMessage.call_args.args[0], "双语字幕工坊已经开着：只能开一个，避免两份同时改同一个工程")

    def _main(self, args, guard, primary):
        old_hook = sys.excepthook
        self.addCleanup(setattr, sys, "excepthook", old_hook)
        with patch("subkit_app.mainwindow.MainWindow") as window, \
             patch("subkit_app.single_instance.prepare_instance", return_value=(guard, primary)), \
             patch("subkit_app.gc_guard.install"), \
             patch("PySide6.QtWidgets.QApplication", return_value=self.app), \
             patch.object(self.app, "exec", return_value=0), \
             patch("PySide6.QtCore.QTimer.singleShot"):
            result = entry.main(["DuoSub.pyw"] + args)
        return result, window

    def test_main_forwarded_launch_returns_zero_without_window(self):
        guard = Mock()
        result, window = self._main([], guard, False)
        self.assertEqual(result, 0)
        window.assert_not_called()
        guard.close.assert_called_once()

    def test_main_primary_constructs_window_and_closes_server(self):
        guard = Mock()
        result, window = self._main([], guard, True)
        self.assertEqual(result, 0)
        window.assert_called_once()
        window.return_value.show.assert_called_once()
        guard.set_handler.assert_called_once()
        guard.close.assert_called_once()
        self.app.aboutToQuit.disconnect(guard.close)

    def test_main_smoke_accepts_bypassed_guard(self):
        result, window = self._main(["--smoke-test"], None, True)
        self.assertEqual(result, 0)
        window.assert_called_once()
