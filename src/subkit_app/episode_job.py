"""Episode import/analysis/merge workers own detached data, never a Qt session."""
from __future__ import annotations

import copy
import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from PySide6.QtCore import Qt

from subkit.languages import lane_label
from subkit.proc import CancelledError

from .help import dialog_buttons


class _Signals(QObject):
    progress = Signal(str)
    done = Signal(object, object)
    failed = Signal(str)
    cancelled = Signal()


class EpisodeProgress(QDialog):
    help_anchor = "add-episode"
    cancelRequested = Signal()

    def __init__(self, parent):
        super().__init__(parent, Qt.WindowType.Tool)
        self.setWindowTitle("正在添加这一集…")
        self.setModal(False)
        self.setMinimumWidth(330)
        layout = QVBoxLayout(self)
        self.step = QLabel("正在对齐…")
        layout.addWidget(self.step)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.cancelRequested)
        buttons = QHBoxLayout()
        buttons.addWidget(self.cancel_button)
        dialog_buttons(self, buttons)
        layout.addLayout(buttons)

    def reject(self):
        self.cancelRequested.emit()

    def closeEvent(self, event):
        self.cancelRequested.emit()
        event.ignore()


class EpisodeJob(QRunnable):
    def __init__(self, episode, mode, *, imports=None, settings=None, semantic_factory=None):
        super().__init__()
        self.setAutoDelete(False)
        self.episode = copy.deepcopy(episode)
        self.mode, self.imports, self.settings = mode, imports, settings
        self.semantic_factory = semantic_factory
        self.signals = _Signals()
        self.finished = threading.Event()
        self._cancel = threading.Event()
        self._commit_lock = threading.Lock()
        self._committing = False

    def cancel(self):
        # The atomic save is the point of no return; a late click reports success.
        with self._commit_lock:
            if not self._committing:
                self._cancel.set()

    def check(self):
        if self._cancel.is_set():
            raise CancelledError()

    def run(self):
        factory = None
        try:
            self.check()
            ep = self.episode
            langs = ep.project.languages
            if self.mode == "import":
                data = self.imports
                if data.get("video"):
                    ep.set_video(Path(data["video"]), data["audio"])
                for lane, key in (("translation", "cn"), ("original", "jp")):
                    self.signals.progress.emit(f"正在读取{lane_label(lane, langs)}字幕…")
                    self.check()
                    ep.add_source(Path(data[key]), lang=langs[lane], lane=lane,
                                  cn_cfg=data["roles"] if lane == "translation" else None)
                self.check()
                result = None
            elif self.mode == "reanalyze":
                self.signals.progress.emit(f"正在读取{lane_label('original', langs)}字幕…")
                result = ep.reanalyze_sources(cancel_check=self.check)
                self.check()
            else:
                self.signals.progress.emit("正在对齐…")
                previous = ep.document
                confirmed = previous is not None and any(g.state == "confirmed" for g in previous.groups)
                if confirmed:
                    ep.document = None
                if self.settings.get("semantic"):
                    self.signals.progress.emit("正在用语义模型对齐（第一次较慢）…")
                    self.check()
                    factory = self.semantic_factory(ep.project, cancel_check=self.check)
                ep.merge(settings=self.settings, sim_factory=factory, cancel_check=self.check, persist=False)
                self.signals.progress.emit("正在保存…")
                with self._commit_lock:
                    self.check()
                    self._committing = True
                if previous is not None:
                    ep.backup("before-remerge" if confirmed else "before-merge")
                ep.save()
                result = ep.validate()
            self.signals.done.emit(ep, result)
        except CancelledError:
            self.signals.cancelled.emit()
        except Exception as exc:
            self.signals.failed.emit(f"{exc}\n\n{traceback.format_exc()}")
        finally:
            if factory is not None and hasattr(factory, "close"):
                factory.close()
            self.finished.set()
