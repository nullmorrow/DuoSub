"""One detached listening pipeline; Qt polls completion instead of worker signals."""
from __future__ import annotations

from pathlib import Path
import threading
import time

from PySide6.QtCore import QRunnable

from subkit.proc import CancelledError, KillableRun
from subkit.listening_progress import read_progress


class ListeningJob(QRunnable):
    def __init__(self, request, output: Path):
        super().__init__()
        self.setAutoDelete(False)
        self.request, self.output = request, Path(output)
        self.runner = KillableRun()
        self.finished = threading.Event()
        self.result = self.error = None
        self.cancelled = False
        self._progress = "正在准备音频…"
        self._progress_at_ns = time.time_ns()
        self._lock = threading.Lock()

    def progress_text(self) -> str:
        with self._lock:
            text, updated = self._progress, self._progress_at_ns
        # A later parent callback (e.g. assembling candidates) supersedes the
        # last child stage. Finished jobs retain the final parent message.
        worker = None if self.finished.is_set() else read_progress(self.output, newer_than_ns=updated)
        return worker or text

    def _set_progress(self, text: str) -> None:
        with self._lock:
            self._progress = str(text)
            self._progress_at_ns = time.time_ns()

    def cancel(self) -> None:
        self.cancelled = True
        self.runner.cancel()

    def run(self) -> None:
        try:
            from subkit.listening import generate

            self.runner.check_cancelled()
            result = generate(self.request, self.output, runner=self.runner, progress=self._set_progress)
            self.runner.check_cancelled()
            self.result = result
        except CancelledError:
            self.cancelled = True
        except Exception as exc:
            self.error = exc
        finally:
            self.finished.set()


class SemanticJob(ListeningJob):
    """A translation-only task; existing audio stays owned by the dialog."""

    def __init__(self, request, output: Path, *, existing: tuple[str, ...], round_index: int, device: str):
        super().__init__(request, output)
        self.existing, self.round_index, self.device = existing, round_index, device
        self._set_progress("正在准备更多翻译建议…")

    def run(self) -> None:
        try:
            from subkit.listening_refresh import generate_semantic_batch

            self.runner.check_cancelled()
            result = generate_semantic_batch(self.request, self.output, existing=self.existing,
                                            round_index=self.round_index, device=self.device,
                                            runner=self.runner, progress=self._set_progress)
            self.runner.check_cancelled()
            self.result = result
        except CancelledError:
            self.cancelled = True
        except Exception as exc:
            self.error = exc
        finally:
            self.finished.set()
