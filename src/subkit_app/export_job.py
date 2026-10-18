"""Detached release build and font probe; the GUI owns all output writes."""
from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from subkit.proc import CancelledError, KillableRun

from .help import dialog_buttons


def _reference_html(report, runner):
    data = getattr(report, "reference", None)
    if not isinstance(data, dict):
        return
    try:
        from subkit.core.reference_report import render_reference_report
        data["html"] = render_reference_report(data, runner=runner)
    except CancelledError:
        raise
    except Exception as exc:
        data["html_error"] = str(exc) or type(exc).__name__
    runner.check_cancelled()


class ExportProgress(QDialog):
    cancelRequested = Signal()

    def __init__(self, parent, *, fonts_only=False):
        super().__init__(parent)
        self.help_anchor = "fonts" if fonts_only else "export"
        text = "正在检查字体…" if fonts_only else "正在导出…"
        self.setWindowTitle(text)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(280)
        layout = QVBoxLayout(self)
        self.label = QLabel(text)
        layout.addWidget(self.label)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addWidget(self.cancel_button)
        dialog_buttons(self, buttons)
        layout.addLayout(buttons)

    def reject(self):
        self.cancel_button.setEnabled(False)
        self.cancelRequested.emit()

    def closeEvent(self, event):
        self.reject()
        event.ignore()


class ExportJob(QRunnable):
    def __init__(self, episode, check_fonts, **options):
        super().__init__()
        self.setAutoDelete(False)
        # Constructed under the session lock on the GUI thread.
        self.episode = copy.deepcopy(episode)
        self.check_fonts = check_fonts
        self.zh_both = options.pop("zh_both", False)
        options.setdefault("annotation_order", "end")
        self.options = options
        self.runner = KillableRun()
        self.finished = threading.Event()
        self.result = None
        self.error = None
        self.cancelled = False

    def cancel(self):
        self.cancelled = True
        self.runner.cancel()

    def run(self):
        try:
            self.runner.check_cancelled()
            results = []
            for direction in (("simplified", "traditional") if self.zh_both else (None,)):
                options = dict(self.options)
                if direction is not None:
                    options["zh_direction"] = direction
                ass, report = self.episode.build_export(runner=self.runner, **options)
                self.runner.check_cancelled()
                _reference_html(report, self.runner)
                fonts = self.check_fonts(ass, runner=self.runner)
                self.runner.check_cancelled()
                results.append((ass, report, fonts))
            self.result = results if self.zh_both else results[0]
        except CancelledError:
            self.cancelled = True
        except Exception as exc:
            self.error = exc
        finally:
            self.finished.set()


class _SeasonSignals(QObject):
    progress = Signal(int, int, str, str)          # episode number, count, episode id, step


@dataclass
class SeasonResult:
    eid: str
    target: Path
    episode: object = None
    ass: object = None
    report: object = None
    fonts: object = None
    validation: object = None
    unfinished: list = field(default_factory=list)
    error: str = ""


class SeasonExportJob(QRunnable):
    """Build every chosen episode's release ASS and probe its fonts; the GUI writes afterwards.

    ``current`` is a copy of the open episode (taken under the session lock on the GUI thread), so its
    unsaved edits are exported; the others are opened from their saved documents.  One episode failing
    does not stop the others; cancelling stops at the next step and nothing is written."""

    def __init__(self, project, items, current, check_fonts, *, include_review=False, **options):
        super().__init__()
        self.setAutoDelete(False)
        self.project, self.items, self.current = project, list(items), current
        self.include_review = include_review
        self.review_packets: dict[str, tuple[str, str]] = {}
        self.review_errors: dict[str, str] = {}
        options.setdefault("annotation_order", "end")
        self.check_fonts, self.options = check_fonts, options
        self.signals = _SeasonSignals()
        self.runner = KillableRun()
        self.finished = threading.Event()
        self.results: list[SeasonResult] = []
        self.error = None
        self.cancelled = False

    def cancel(self):
        self.cancelled = True
        self.runner.cancel()

    def run(self):
        from subkit.core import karaoke as kara

        try:
            episode_ids = list(dict.fromkeys(item[0] for item in self.items))
            episodes = {}
            for item in self.items:
                eid, target = item[:2]
                i, count = episode_ids.index(eid) + 1, len(episode_ids)
                direction = item[2] if len(item) > 2 else None
                result = SeasonResult(eid, target)
                try:
                    self.runner.check_cancelled()
                    self.signals.progress.emit(i, count, eid, "正在生成")
                    if eid not in episodes:
                        episodes[eid] = (self.current if self.current is not None and self.current.id == eid
                                         else self.project.episode(eid))
                    ep = episodes[eid]
                    if ep.document is None:
                        raise ValueError("还没有合并")
                    result.episode = ep
                    options = dict(self.options)
                    if direction is not None:
                        options["zh_direction"] = direction
                    result.ass, result.report = ep.build_export(runner=self.runner, **options)
                    self.runner.check_cancelled()
                    if isinstance(getattr(result.report, "reference", None), dict):
                        self.signals.progress.emit(i, count, eid, "正在生成原配对照报告")
                        _reference_html(result.report, self.runner)
                    self.signals.progress.emit(i, count, eid, "正在检查字体")
                    result.fonts = self.check_fonts(result.ass, runner=self.runner)
                    self.runner.check_cancelled()
                    result.validation = ep.validate()
                    result.unfinished = kara.unfinished(ep.document)
                    if self.include_review and eid not in self.review_packets and eid not in self.review_errors:
                        self.signals.progress.emit(i, count, eid, "正在生成审议包")
                        try:
                            from subkit.core.display import DisplayRules
                            from subkit.core.review import build_packet

                            # Match the independent review export/import, not an ASS-only script override.
                            project = ep.project
                            langs = project.languages
                            self.review_packets[eid] = build_packet(
                                ep.document, episode=ep.id, orig_lang=langs.get("original", "ja"),
                                trans_lang=langs.get("translation", "zh"), chars=project.character_book(),
                                pinyin_mode=project.pinyin_mode,
                                display=DisplayRules.from_settings(project.settings()),
                                check_cancelled=self.runner.check_cancelled)
                            self.runner.check_cancelled()
                        except CancelledError:
                            raise
                        except Exception as exc:
                            self.review_errors[eid] = str(exc) or type(exc).__name__
                except CancelledError:
                    raise
                except Exception as exc:
                    result.error = str(exc) or type(exc).__name__
                self.results.append(result)
        except CancelledError:
            self.cancelled = True
        except Exception as exc:
            self.error = exc
        finally:
            self.finished.set()
