"""Private season batch processing with step-boundary cancellation."""

from __future__ import annotations

from pathlib import Path
import subprocess
import threading
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from subkit.core.batch import BatchOptions, BatchResult, run_season, scan_season, song_families
from subkit.ass.document import AssDocument
from .. import help as app_help

MODE_LABELS = {
    "5.1": "自动导入整季…",
    "5.2": "自动导入整季（双语字幕）…",
}
TIMING_CHOICES = (
    ("audio", "按视频的声音（推荐）",
     "和「时间轴 → 自动调轴（按视频音频）」一样：按视频里的人声找出整集或分段的偏差，两种语言一起挪；没有可信偏差时不改。需要视频。"),
    ("original", "按原声字幕的时间",
     "对白用原声字幕文件里的时间，翻译跟着；没配对的翻译行按附近的时间差挪。只在原声字幕和视频同步时用。"),
    ("translation", "按翻译字幕的时间（不另外调轴）",
     "合并时本来就用翻译字幕的时间，原声跟着；不再另外调整。"),
)


class _Signals(QObject):
    progress = Signal(int, int, str, str)
    completed = Signal(int, object, str)


class _BatchJob(QRunnable):
    def __init__(self, serial: int, season: Path, options: BatchOptions, *, project_root: Path | None = None, scan: bool = False):
        super().__init__()
        self.setAutoDelete(False)
        self.finished = threading.Event()
        self.cancelled = threading.Event()
        self.signals = _Signals()
        self.serial, self.season, self.options, self.scan = serial, season, options, scan
        self.project_root = project_root

    def run(self) -> None:
        result, error = None, ""
        try:
            if self.scan:
                episodes = scan_season(self.season, self.options)
                first = next((e for e in episodes if e.status == "就绪"), None)
                families = song_families(AssDocument.load(first.translation)) if first else []
                result = (episodes, families)
            else:
                batch = run_season(self.season, self.project_root, self.options, progress=self.signals.progress.emit,
                                   cancel_check=self.cancelled.is_set)
                result = (batch, batch.report.read_text(encoding="utf-8"))
        except Exception as exc:
            error = str(exc)
        try:
            self.signals.completed.emit(self.serial, result, error)
        finally:
            self.finished.set()


class BatchReportDialog(QDialog):
    help_anchor = "batch"

    def __init__(self, result: BatchResult, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量处理报告")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(850, 600)
        self.result = result
        layout = QVBoxLayout(self)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlainText(text)
        layout.addWidget(self.text)
        buttons = QHBoxLayout()
        self.folder_button = QPushButton("打开输出文件夹")
        self.folder_button.clicked.connect(self.show_folder)
        buttons.addWidget(self.folder_button)
        close = QPushButton("关闭")
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        app_help.dialog_buttons(self, buttons, auxiliary=(self.folder_button,))
        layout.addLayout(buttons)

    def show_folder(self) -> None:
        try:
            subprocess.Popen(["explorer.exe", str(self.result.report.parent.resolve())])
        except OSError as exc:
            QMessageBox.warning(self, "批量处理报告", f"打开输出文件夹失败：{exc}")


class BatchDialog(QDialog):
    help_anchor = "batch"

    def __init__(self, mode: str, settings, padding: Callable[[], tuple[int, int]],
                 project_root: Path, prepare: Callable[[], None], reload_project: Callable[[], None], parent=None):
        super().__init__(parent)
        self.setWindowTitle(MODE_LABELS[mode])
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(1000, 560)
        self.mode, self.settings, self.padding = mode, settings, padding
        self.project_root, self.prepare, self.reload_project = project_root, prepare, reload_project
        self._ready = False
        self._serial = 0
        self._jobs: dict[int, _BatchJob] = {}
        self._pending = {}
        self._run_job = None
        self._closing = False
        self._asking_close = False
        self.report_dialog = None
        layout = QVBoxLayout(self)
        form = QFormLayout()
        row = QHBoxLayout()
        self.season = QLineEdit()
        self.choose_button = QPushButton("选择文件夹…")
        self.choose_button.clicked.connect(self.choose_season)
        row.addWidget(self.season)
        row.addWidget(self.choose_button)
        form.addRow("季目录", row)
        layout.addLayout(form)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["集", "视频", "中文字幕" if mode == "5.1" else "双语字幕",
                                              "日文字幕", "状态"])
        self.table.setColumnHidden(3, mode == "5.2")
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        layout.addWidget(QLabel("歌曲"))
        self.song_table = QTableWidget(0, 5)
        self.song_table.setHorizontalHeaderLabels(["家族", "样式", "句数", "建议", "处理方式"])
        self.song_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.song_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.song_table.horizontalHeader().setStretchLastSection(True)
        self.song_table.setMaximumHeight(150)
        layout.addWidget(self.song_table)
        timing_row = QHBoxLayout()
        timing_row.addWidget(QLabel("调轴"))
        self.timing = QComboBox()
        choices = (TIMING_CHOICES if mode == "5.1" else TIMING_CHOICES[:1] + (("translation", "不调轴", ""),))
        for value, text, tip in choices:
            self.timing.addItem(text, value)
            self.timing.setItemData(self.timing.count() - 1, tip or "两种语言都用这个双语文件里的时间。",
                                    Qt.ItemDataRole.ToolTipRole)
        index = self.timing.findData(settings.value(f"batch/timing/{mode}", "audio"))
        self.timing.setCurrentIndex(max(0, index))
        timing_row.addWidget(self.timing)
        timing_row.addStretch(1)
        layout.addLayout(timing_row)
        self.adopt_times = QCheckBox("全季自动采用建议时间")
        self.adopt_times.setToolTip("实测（少女革命 10 集）：字幕组中文轴的开始时间明显更准；"
                                    "已经较准的日文轴整季采用后不会更准，还会改动一部分原本正确的时间。")
        layout.addWidget(self.adopt_times)
        self.hide_interjections = QCheckBox("没有中文对照的叫喊和语气词不显示")
        self.hide_interjections.setChecked(settings.value("batch/hide_unpaired_interjections", True, type=bool))
        self.hide_interjections.toggled.connect(
            lambda checked: self.settings.setValue("batch/hide_unpaired_interjections", checked))
        layout.addWidget(self.hide_interjections)
        options = QFormLayout()
        self.embed_fonts = QCheckBox("用 assfonts 嵌入字体")
        self.embed_fonts.setChecked(True)
        fonts = QHBoxLayout()
        self.assfonts = QLineEdit(str(settings.value("batch/assfonts", str(BatchOptions().assfonts))))
        self.fonts_button = QPushButton("更改…")
        self.fonts_button.clicked.connect(self.choose_assfonts)
        fonts.addWidget(self.assfonts)
        fonts.addWidget(self.fonts_button)
        options.addRow(self.embed_fonts, fonts)
        layout.addLayout(options)
        self.progress = QLabel("")
        layout.addWidget(self.progress)
        buttons = QHBoxLayout()
        self.start_button, self.cancel_button, self.close_button = (QPushButton(t) for t in ("开始", "取消", "关闭"))
        self.start_button.setEnabled(False)
        self.cancel_button.hide()
        self.start_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.cancel)
        self.close_button.clicked.connect(self.reject)
        for button in (self.start_button, self.cancel_button, self.close_button):
            buttons.addWidget(button)
        app_help.dialog_buttons(self, buttons)
        layout.addLayout(buttons)
        self.season.textChanged.connect(self.scan)
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._reap)

    def choose_season(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择季目录", self.season.text())
        if path:
            self.season.setText(path)

    def choose_assfonts(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 assfonts.exe", self.assfonts.text(), "程序 (*.exe)")
        if path:
            self.assfonts.setText(path)
            self.settings.setValue("batch/assfonts", path)

    def options(self) -> BatchOptions:
        lead_in, lead_out = self.padding()
        return BatchOptions(mode=self.mode, timing=self.timing.currentData(),
                            adopt_suggested_times=self.adopt_times.isChecked(),
                            hide_unpaired_interjections=self.hide_interjections.isChecked(),
                            lead_in_ms=lead_in, lead_out_ms=lead_out,
                            song_choices={self.song_table.item(row, 0).text(): self.song_table.cellWidget(row, 4).currentData()
                                          for row in range(self.song_table.rowCount())},
                            assfonts=Path(self.assfonts.text()) if self.embed_fonts.isChecked() else None)

    def _launch(self, *, scan: bool) -> None:
        job = _BatchJob(self._serial, Path(self.season.text()), self.options(), project_root=self.project_root, scan=scan)
        self._jobs[job.serial] = job
        job.signals.completed.connect(self._completed)
        job.signals.progress.connect(self._progress)
        if not scan:
            self._run_job = job
        self._timer.start()
        QThreadPool.globalInstance().start(job)

    def scan(self) -> None:
        if self._run_job is not None:
            return
        self.song_table.setRowCount(0)
        self._serial += 1
        self._ready = False
        self.start_button.setEnabled(False)
        self.table.setRowCount(0)
        if not self.season.text().strip():
            self.progress.clear()
            return
        self.progress.setText("正在识别文件…")
        self._launch(scan=True)

    def start(self) -> None:
        if self._jobs or not self.start_button.isEnabled():
            return
        try:
            self.prepare()
        except Exception as exc:
            QMessageBox.warning(self, "批量处理", f"保存工程失败：{exc}")
            return
        self.settings.setValue("batch/assfonts", self.assfonts.text())
        self.settings.setValue(f"batch/timing/{self.mode}", self.timing.currentData())
        self._serial += 1
        self._set_running(True)
        self.progress.setText("正在开始…")
        self._launch(scan=False)

    def _set_running(self, running: bool) -> None:
        for widget in (self.season, self.choose_button, self.timing, self.adopt_times, self.hide_interjections,
                       self.embed_fonts, self.assfonts, self.fonts_button, self.start_button, self.song_table):
            widget.setEnabled(not running)
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(running)

    def cancel(self) -> None:
        if self._run_job is not None:
            self._run_job.cancelled.set()
            self.cancel_button.setEnabled(False)
            self.progress.setText("正在取消，当前步骤结束后停止…")

    @Slot(int, int, str, str)
    def _progress(self, i: int, n: int, key: str, step: str) -> None:
        if self._run_job is not None and not self._run_job.cancelled.is_set():
            self.progress.setText(f"第 {i}/{n} 集 {key}：{step}")

    @Slot(int, object, str)
    def _completed(self, serial: int, result, error: str) -> None:
        self._pending[serial] = (result, error)

    def _reap(self) -> None:
        if self._asking_close:
            return
        for serial, job in list(self._jobs.items()):
            if not job.finished.is_set() or serial not in self._pending:
                continue
            result, error = self._pending.pop(serial)
            del self._jobs[serial]
            if job.scan:
                if serial != self._serial or self._closing:
                    continue
                self.progress.setText(error)
                if not error:
                    result, families = result
                    self.song_table.setRowCount(len(families))
                    for row, family in enumerate(families):
                        label = ("转写为竖排并加振假名（右侧）" if family.default == "transcribe_vertical" else
                                 "原样保留" if family.default == "keep" else "转写并加振假名")
                        for col, value in enumerate([family.family, "、".join(family.styles), str(family.line_count), label]):
                            item = QTableWidgetItem(value)
                            item.setToolTip(f"karaoke：{family.karaoke_lines}；fx：{family.fx_count}；绘图：{family.has_drawing}；自带振假名：{family.own_furigana}；竖排：{family.vertical}")
                            self.song_table.setItem(row, col, item)
                        combo = QComboBox()
                        combo.addItem("转写并加振假名", "transcribe")
                        combo.addItem("原样保留", "keep")
                        if family.vertical:
                            combo.addItem("转写为竖排并加振假名（右侧）", "transcribe_vertical")
                        combo.setCurrentIndex(combo.findData(family.default))
                        self.song_table.setCellWidget(row, 4, combo)
                    self.table.setRowCount(len(result))
                    for row, episode in enumerate(result):
                        values = [episode.key, episode.video, episode.translation, episode.original, episode.status]
                        for col, value in enumerate(values):
                            item = QTableWidgetItem(value.name if isinstance(value, Path) else str(value or ""))
                            item.setToolTip(str(value or ""))
                            self.table.setItem(row, col, item)
                    self._ready = any(e.status == "就绪" for e in result)
            else:
                self._run_job = None
                self._set_running(False)
                try:
                    self.reload_project()
                except Exception as exc:
                    QMessageBox.warning(self, "批量处理", f"刷新工程失败：{exc}")
                self.progress.setText("批量处理失败：" + error if error else "已取消" if result[0].cancelled else "批量处理完成")
                if error:
                    QMessageBox.warning(self, "批量处理", f"批量处理失败：{error}")
                else:
                    self.report_dialog = BatchReportDialog(result[0], result[1], self)
                    self.report_dialog.show()
        self.start_button.setEnabled(self._ready and not self._jobs)
        if not self._jobs:
            self._timer.stop()
            if self._closing:
                # The report remains available after cancelling a run via Close.
                if self.report_dialog is not None and self.report_dialog.isVisible():
                    self.report_dialog.finished.connect(lambda _: QDialog.reject(self))
                else:
                    QDialog.reject(self)

    def reject(self) -> None:
        if self._run_job is not None:
            self._asking_close = True
            try:
                answer = QMessageBox.question(self, "批量处理", "正在批量处理，取消后关闭吗？")
            finally:
                self._asking_close = False
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._closing = True
            self.cancel()
        elif self._jobs:
            self._closing = True
        else:
            super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802
        event.ignore()
        self.reject()
