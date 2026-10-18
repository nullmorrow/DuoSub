"""One isolated alignment process at a time; Qt objects stay on the GUI thread."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal
from subkit.kalign.worker import parse_progress, result_from_json


class KalignRunner(QObject):
    progress = Signal(dict)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.directory = None
        self._buffer = b""
        self._error = ""
        self._cancelled = False

    @property
    def running(self):
        return self.process is not None

    def start(self, request):
        if self.running:
            raise RuntimeError("自动 K 轴正在处理")
        self.directory = Path(tempfile.mkdtemp(prefix="subkit-kalign-"))
        try:
            (self.directory / "request.json").write_text(json.dumps(asdict(request), ensure_ascii=False), encoding="utf-8")
            self._buffer, self._error, self._cancelled = b"", "", False
            process = self.process = QProcess(self)
            process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            env = QProcessEnvironment.systemEnvironment()
            source = str(Path(__file__).resolve().parents[1])
            env.insert("PYTHONPATH", source + os.pathsep + env.value("PYTHONPATH"))
            env.insert("PYTHONIOENCODING", "utf-8")
            process.setProcessEnvironment(env)
            program = sys.executable
            if os.name == "nt":
                modifier = getattr(process, "setCreateProcessArgumentsModifier", None)
                if modifier is not None:
                    def hide_console(args):
                        args.flags |= 0x08000000  # CREATE_NO_WINDOW
                    modifier(hide_console)
                else:
                    # PySide6 6.11 omits the native modifier binding. The matching
                    # windowless interpreter uses the same environment and packages.
                    windowless = Path(sys.executable).with_name("pythonw.exe")
                    if not windowless.is_file():
                        raise RuntimeError("找不到无控制台 Python（pythonw.exe）")
                    program = str(windowless)
            process.readyReadStandardOutput.connect(self._read)
            process.finished.connect(self._done)
            process.errorOccurred.connect(self._process_error)
            paths = [str(self.directory / "request.json"), str(self.directory / "result.json")]
            if request.kind == "dialogue":
                # Probe actual CUDA availability in the worker process, never the GUI.
                bootstrap = ("import json,runpy,sys,torch; "
                    "print(json.dumps({'dialogue_device': 'cuda' if torch.cuda.is_available() else 'cpu'}),flush=True); "
                    "sys.argv=['subkit.kalign.worker']+sys.argv[1:]; "
                    "runpy.run_module('subkit.kalign.worker',run_name='__main__')")
                arguments = ["-B", "-X", "utf8", "-c", bootstrap, *paths]
            else:
                arguments = ["-B", "-X", "utf8", "-m", "subkit.kalign.worker", *paths]
            process.start(program, arguments)
        except Exception:
            self._release()
            raise

    def _read(self):
        self._buffer += bytes(self.process.readAllStandardOutput())
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            try:
                device = json.loads(line).get("dialogue_device")
            except (ValueError, AttributeError):
                device = None
            if device in ("cpu", "cuda"):
                self.progress.emit({"stage": "extract", "device": device})
                continue
            value = parse_progress(line)
            if value and "error" in value:
                self._error = value["error"]
            elif value:
                self.progress.emit(value)

    def _process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            message = self.process.errorString()
            self._release()
            self.failed.emit(message)

    def _done(self, code, status):
        self._read()
        cancelled = self._cancelled
        result, error = None, self._error
        if not cancelled and code == 0 and status == QProcess.ExitStatus.NormalExit:
            try:
                result = result_from_json(json.loads((self.directory / "result.json").read_text(encoding="utf-8")))
            except Exception as exc:
                error = str(exc)
        self._release()
        if cancelled:
            self.cancelled.emit()
        elif result is not None:
            self.finished.emit(result)
        else:
            self.failed.emit(error or f"处理进程退出（{code}）")

    def cancel(self):
        if self.process is not None:
            self._cancelled = True
            self.process.kill()
            self.process.waitForFinished(2000)

    def _release(self):
        if self.process is not None:
            self.process.deleteLater()
            self.process = None
        if self.directory is not None:
            for name in ("request.json", "result.json"):
                (self.directory / name).unlink(missing_ok=True)
            try:
                self.directory.rmdir()
            except OSError:
                pass  # Leave any unexpected files for the owning cleanup policy.
            self.directory = None
