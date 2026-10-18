"""Per-user local launch forwarding; smoke runs never acquire the endpoint."""
from __future__ import annotations

import getpass
import json
import time
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QObject, QTimer
from PySide6.QtNetwork import QLocalServer, QLocalSocket


def server_name():
    return "DuoSub-" + getpass.getuser()


def _allow_foreground():
    """The launch the user just started may give the foreground to the running instance; without this Windows only
    flashes its taskbar button instead of bringing the window up."""
    import sys

    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.AllowSetForegroundWindow(-1)      # ASFW_ANY
        except (AttributeError, OSError):
            pass


def project_argument(arguments):
    args = [a for a in arguments if not a.startswith("-")]
    return Path(args[0]).resolve() if args and Path(args[0]).is_dir() else None


class SingleInstance(QObject):
    def __init__(self, name=None, parent=None):
        super().__init__(parent)
        self.name = name or server_name()
        self.server = QLocalServer(self)
        self.server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self.server.newConnection.connect(self._accept)
        self.handler = None
        self.pending = []
        self.clients = set()

    def forward(self, arguments):
        socket = QLocalSocket()
        try:
            socket.connectToServer(self.name)
            if not socket.waitForConnected(300):
                return False
            _allow_foreground()

            socket.write(json.dumps(arguments, ensure_ascii=False).encode("utf-8") + b"\n")
            socket.flush()
            deadline = time.monotonic() + 1.5
            reply = bytearray()
            while time.monotonic() < deadline:
                QCoreApplication.processEvents()  # also supports in-process handshake tests
                socket.waitForReadyRead(20)
                reply.extend(bytes(socket.readAll()))
                if b"\n" in reply:
                    return bytes(reply) == b"ok\n"
            # A connected endpoint belongs to a live instance even if it is busy.
            raise RuntimeError("双语字幕工坊已经开着，但没有回应启动请求")
        finally:
            socket.abort()
            socket.close()

    def acquire(self, arguments):
        if self.forward(arguments):
            return False
        QLocalServer.removeServer(self.name)
        if not self.server.listen(self.name):
            if self.forward(arguments):  # another launch may have won the race
                return False
            raise RuntimeError(self.server.errorString())
        return True

    def set_handler(self, handler):
        self.handler = handler
        pending, self.pending = self.pending, []
        for arguments in pending:
            handler(arguments)

    def _accept(self):
        while self.server.hasPendingConnections():
            socket = self.server.nextPendingConnection()
            self.clients.add(socket)
            buffer = bytearray()
            timer = QTimer(socket)
            timer.setSingleShot(True)
            timer.timeout.connect(socket.abort)
            timer.start(3000)

            def finished(s=socket):
                self.clients.discard(s)
                s.deleteLater()

            def read(s=socket, data=buffer, timeout=timer):
                data.extend(bytes(s.readAll()))
                if b"\n" not in data:
                    return
                timeout.stop()
                try:
                    arguments = json.loads(data.split(b"\n", 1)[0].decode("utf-8"))
                    if not isinstance(arguments, list) or any(not isinstance(a, str) for a in arguments):
                        raise ValueError("invalid startup arguments")
                except (ValueError, UnicodeError):
                    s.abort()
                    return
                # Queue before acknowledging, so the second process can exit promptly.
                self.pending.append(arguments)
                s.write(b"ok\n")
                s.flush()
                s.disconnectFromServer()
                if self.handler is not None:
                    QTimer.singleShot(0, lambda: self.set_handler(self.handler))

            socket.disconnected.connect(finished)
            socket.readyRead.connect(read)
            if socket.bytesAvailable():
                read()

    def close(self):
        for socket in list(self.clients):
            socket.abort()
        self.server.close()


def prepare_instance(arguments, parent=None):
    if "--smoke-test" in arguments:
        return None, True
    guard = SingleInstance(parent=parent)
    forwarded = list(arguments)
    folder = project_argument(arguments)
    if folder is not None:
        index = next(i for i, arg in enumerate(arguments) if not arg.startswith("-"))
        forwarded[index] = str(folder)
    return guard, guard.acquire(forwarded)


def activate_window(window, arguments):
    if window.isMinimized():
        window.showNormal()
    window.raise_()
    window.activateWindow()
    folder = project_argument(arguments)
    if window.project is None:
        if folder is not None:
            window.open_project(folder)
    else:
        window.statusBar().showMessage("双语字幕工坊已经开着：只能开一个，避免两份同时改同一个工程", 10000)
