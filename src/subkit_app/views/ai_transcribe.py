"""Review local listening hypotheses and audition their audio before adoption."""
from __future__ import annotations

from dataclasses import replace
import copy
from pathlib import Path
import tempfile
import time

from PySide6.QtCore import QThreadPool, QTimer, Qt, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                              QDialogButtonBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                              QLineEdit, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QTreeWidget,
                              QTreeWidgetItem, QVBoxLayout)

from .. import pool as job_pool
from ..help import dialog_buttons
from ..listening_job import ListeningJob, SemanticJob
from .group_table import fmt_time


class ListeningDialog(QDialog):
    help_anchor = "ai-transcribe"

    def __init__(self, request, cn_text: str, parent=None, *, job_factory=ListeningJob,
                 semantic_job_factory=SemanticJob, before_adopt=None):
        super().__init__(parent, Qt.WindowType.Tool)
        self.setWindowTitle("AI听译")
        self.resize(940, min(760, max(610, self.screen().availableGeometry().height() - 80)))
        self.cn_text = cn_text
        self.request, self.job_factory = request, job_factory
        self.semantic_job_factory = semantic_job_factory
        self.before_adopt = before_adopt
        self.job = None
        self.result = None
        self._directory = None
        self._directory_identity = None
        self._refresh_directory = self._refresh_identity = None
        self._refreshing = False
        self._semantic_round = 0
        self._refresh_draft = None
        self._usable_result = False
        self._completed = False
        self._retired = False
        self._invalid = ""
        self._pending_request = None
        self._pending_done = None
        self._adopted = None
        self._audio_bounds = None
        self._audio_pending = False
        self._started_at = None
        self._elapsed_seconds = 0

        layout = QVBoxLayout(self)
        self.translation = QLabel("译文：" + cn_text)
        self.translation.setTextFormat(Qt.TextFormat.PlainText)
        self.translation.setWordWrap(True)
        self.translation.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.translation)
        self.range_label = QLabel(f"当前句：{fmt_time(request.start_ms)}–{fmt_time(request.end_ms)}")
        layout.addWidget(self.range_label)
        progress_box = QGroupBox("听译状态")
        progress_layout = QVBoxLayout(progress_box)
        state_row = QHBoxLayout()
        self.state_label = QLabel("正在准备听译…")
        state_font = self.state_label.font()
        state_font.setBold(True)
        state_font.setPointSizeF(max(11., state_font.pointSizeF() + 2))
        self.state_label.setFont(state_font)
        self.state_label.setTextFormat(Qt.TextFormat.PlainText)
        state_row.addWidget(self.state_label, 1)
        self.elapsed_label = QLabel("已用时 0:00")
        state_row.addWidget(self.elapsed_label)
        progress_layout.addLayout(state_row)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        progress_layout.addWidget(self.progress_bar)
        self.status = QLabel("正在准备音频…")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        progress_layout.addWidget(self.status)
        layout.addWidget(progress_box)
        options = QHBoxLayout()
        options.addWidget(QLabel("前后各扩展"))
        self.padding = QSpinBox()
        self.padding.setRange(0, 5000)
        self.padding.setSuffix(" ms")
        self.padding.setValue(request.padding_ms)
        options.addWidget(self.padding)
        self.separate = QCheckBox("同时尝试人声分离")
        self.separate.setChecked(request.separate)
        options.addWidget(self.separate)
        self.model = QComboBox()
        for text, model in (("large-v3（较慢）", "large-v3"), ("medium（较快）", "medium")):
            self.model.addItem(text, model)
        index = self.model.findData(request.model)
        if index < 0:
            self.model.addItem(request.model, request.model)
            index = self.model.count() - 1
        self.model.setCurrentIndex(index)
        options.addWidget(self.model)
        self.rerun_button = QPushButton("重新听译")
        self.rerun_button.clicked.connect(self.rerun)
        options.addWidget(self.rerun_button)
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.cancel)
        options.addWidget(self.stop_button)
        layout.addLayout(options)
        hint_row = QHBoxLayout()
        hint_row.addWidget(QLabel("听到的词（可选）"))
        self.hint = QLineEdit()
        self.hint.setMaxLength(80)
        self.hint.setPlaceholderText("例如：ふうせん 或 風船；仅填写实际听到的词")
        self.hint.setText(getattr(request, "hint", ""))
        hint_row.addWidget(self.hint, 1)
        layout.addLayout(hint_row)
        hint_note = QLabel("提示词可能被模型复述，仍须试听核对。")
        hint_note.setWordWrap(True)
        layout.addWidget(hint_note)
        semantic_row = QHBoxLayout()
        self.semantic = QCheckBox("根据中文另列日语候选（语义推测）")
        self.semantic.setChecked(getattr(request, "semantic", False))
        semantic_row.addWidget(self.semantic)
        self.more_button = QPushButton("更多翻译建议")
        self.more_button.setToolTip("保留已有候选，根据中文尝试新的日语说法；不重新识别音频。")
        self.more_button.clicked.connect(self.more_translations)
        semantic_row.addWidget(self.more_button)
        semantic_row.addStretch(1)
        semantic_row.addWidget(QLabel("查看："))
        self.kind_filter = QComboBox()
        for label, kind in (("全部候选", ""), ("音频识别候选", "asr"), ("中文语义推测", "semantic")):
            self.kind_filter.addItem(label, kind)
        self.kind_filter.currentIndexChanged.connect(self._filter_candidates)
        semantic_row.addWidget(self.kind_filter)
        layout.addLayout(semantic_row)
        self.usage_note = QLabel("中文字幕用于独立生成日语语义候选；Whisper 只识别音频和可选的「听到的词」。语义候选未经音频确认。")
        self.usage_note.setWordWrap(True)
        layout.addWidget(self.usage_note)
        self.candidates = QTreeWidget()
        self.candidates.setHeaderLabels(["日文候选", "类型", "来源", "需要留意"])
        self.candidates.setColumnWidth(0, 300)
        self.candidates.setRootIsDecorated(False)
        self.candidates.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.candidates.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.candidates.currentItemChanged.connect(self._candidate_changed)
        layout.addWidget(self.candidates, 1)
        self.warning = QLabel("候选可能把前景对白或背景杂音听成日文，请先试听并核对。")
        self.warning.setTextFormat(Qt.TextFormat.PlainText)
        self.warning.setWordWrap(True)
        layout.addWidget(self.warning)
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText("选择候选后可修改；没有可靠候选时，也可以在试听后手动输入。")
        self.text.setMaximumHeight(95)
        self.text.textChanged.connect(self._enabled)
        layout.addWidget(self.text)
        times = QGridLayout()
        self.start, self.end = QSpinBox(), QSpinBox()
        for column, (label, spin, value) in enumerate((("开始", self.start, request.start_ms),
                                                      ("结束", self.end, request.end_ms))):
            spin.setRange(0, 2_147_483_647)
            spin.setSuffix(" ms")
            spin.setValue(value)
            spin.valueChanged.connect(self._enabled)
            times.addWidget(QLabel(label), 0, column * 2)
            times.addWidget(spin, 0, column * 2 + 1)
        self.original_button = QPushButton("试听原声（扩展范围）")
        self.original_button.clicked.connect(self.play_original)
        self.candidate_button = QPushButton("试听所选候选")
        self.candidate_button.clicked.connect(self.play_candidate)
        self.audio_stop = QPushButton("停止试听")
        self.audio_stop.clicked.connect(self._stop_audio)
        for column, button in enumerate((self.original_button, self.candidate_button, self.audio_stop)):
            times.addWidget(button, 1, column, 1, 1)
        layout.addLayout(times)
        self.buttons = QDialogButtonBox()
        self.adopt_button = self.buttons.addButton("补入原文", QDialogButtonBox.ButtonRole.AcceptRole)
        self.close_button = self.buttons.addButton("关闭", QDialogButtonBox.ButtonRole.RejectRole)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        dialog_buttons(self, self.buttons, default=False)
        layout.addWidget(self.buttons)
        # Enter in an option or time field must never adopt an unchecked hypothesis.
        self._clear_defaults()
        self.audio_output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.mediaStatusChanged.connect(self._audio_status)
        self.player.positionChanged.connect(self._audio_position)
        self.player.errorOccurred.connect(self._audio_error)
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        self._enabled()
        QTimer.singleShot(0, self._initial_start)

    def _clear_defaults(self):
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)

    def showEvent(self, event):
        super().showEvent(event)
        # QDialogButtonBox assigns the first accept button on show even when
        # autoDefault was disabled during construction.
        self._clear_defaults()
        QTimer.singleShot(0, self._clear_defaults)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    def _initial_start(self):
        if not self._retired and self._pending_done is None and self.job is None and not self._invalid:
            self._launch(self._options_request())

    def _options_request(self):
        return replace(self.request, padding_ms=self.padding.value(), separate=self.separate.isChecked(),
                       model=self.model.currentData(), hint=self.hint.text().strip(),
                       translation=self.cn_text, semantic=self.semantic.isChecked())

    def _state(self, label: str, *, busy: bool, detail: str | None = None):
        self.state_label.setText(label)
        self.setWindowTitle("AI听译 — " + label.rstrip("…"))
        self.progress_bar.setVisible(busy)
        if detail is not None:
            self.status.setText(detail)
            self.status.setMinimumHeight(self.status.fontMetrics().height() * (detail.count("\n") + 1) + 8)
        self._update_elapsed(busy=busy)

    def _update_elapsed(self, *, busy: bool):
        if self._started_at is not None:
            self._elapsed_seconds = max(0, int(time.monotonic() - self._started_at))
            if not busy:
                self._started_at = None
        minutes, seconds = divmod(self._elapsed_seconds, 60)
        self.elapsed_label.setText(f"已用时 {minutes}:{seconds:02d}")

    def _launch(self, request):
        if self._retired:
            return
        self._stop_audio(clear=True)
        self._cleanup_output()
        self.request = request
        self.result = None
        self._completed = False
        self._usable_result = False
        self._refreshing = False
        self._semantic_round = 0
        self._refresh_draft = None
        self._adopted = None
        self.candidates.clear()
        self.text.clear()
        self._directory = Path(tempfile.mkdtemp(prefix="duosub-listening-"))
        self._directory_identity = self._directory.resolve()
        self.job = self.job_factory(request, self._directory)
        self._started_at = time.monotonic()
        self._elapsed_seconds = 0
        self._state("正在听译…", busy=True, detail="正在准备音频；首次加载模型可能较慢，请稍候。")
        self._enabled()
        job_pool.start(self.job, pool=QThreadPool.globalInstance())

    def more_translations(self):
        running = self.job is not None and not self.job.finished.is_set()
        if (running or not self._usable_result or self.result is None or self._invalid
                or self._pending_done is not None or self._retired):
            return
        self._cleanup_directory("_refresh_directory", "_refresh_identity")
        self._stop_audio()
        self.semantic.setChecked(True)
        self._semantic_round += 1
        self._refresh_draft = (self.text.toPlainText(), self.start.value(), self.end.value(),
                               self.candidates.currentItem())
        self._refresh_directory = Path(tempfile.mkdtemp(prefix="duosub-more-translations-"))
        self._refresh_identity = self._refresh_directory.resolve()
        existing = tuple(c.text for c in self.result.candidates if getattr(c, "kind", "asr") == "semantic")
        request = replace(self.request, translation=self.cn_text, semantic=True)
        self.job = self.semantic_job_factory(request, self._refresh_directory, existing=existing,
                                             round_index=self._semantic_round, device=self.result.device)
        self._refreshing = True
        self._completed = False
        self._started_at = time.monotonic()
        self._state("正在生成更多翻译建议…", busy=True,
                    detail="保留原候选和编辑文字；只生成新的说法，不重新识别音频。")
        self._enabled()
        job_pool.start(self.job, pool=QThreadPool.globalInstance())

    def rerun(self):
        if self._invalid or self._pending_done is not None:
            return
        request = self._options_request()
        if self.job is not None and not self.job.finished.is_set():
            self._pending_request = request
            self.job.cancel()
            self._state("正在停止…", busy=True, detail="正在停止上一次听译，随后重新开始…")
            self._enabled()
            return
        self._launch(request)

    def cancel(self):
        self._pending_request = None
        if self.job is not None and not self.job.finished.is_set():
            self.job.cancel()
            self._state("正在停止…", busy=True, detail="正在停止听译，请等待后台处理退出…")
            self._enabled()

    def _poll(self):
        job = self.job
        if job is None:
            return
        if not job.finished.is_set():
            self._update_elapsed(busy=True)
            if not job.cancelled and not self._invalid:
                self.status.setText(job.progress_text())
            return
        if self._pending_done is not None:
            code, self._pending_done = self._pending_done, None
            self.done(code)
            return
        if self._pending_request is not None:
            request, self._pending_request = self._pending_request, None
            self._launch(request)
            return
        if self._completed:
            return
        if self._refreshing:
            self._finish_more_translations(job)
            return
        self._completed = True
        self.result = job.result
        self._usable_result = bool(self.result is not None and not self._invalid
                                   and not job.cancelled and job.error is None)
        if self._invalid:
            self._state("结果已失效", busy=False, detail=self._invalid)
        elif job.cancelled:
            self._state("听译已停止", busy=False, detail="听译已停止；可以重新听译。")
        elif job.error is not None:
            self._state("听译失败", busy=False, detail="听译失败：" + str(job.error))
        else:
            self._show_result()
        self._enabled()

    def _show_result(self):
        result = self.result
        if result is None:
            self._state("未获得结果", busy=False, detail="没有获得听译结果，可以重新听译。")
            return
        self.start.setRange(result.window_start_ms, result.window_end_ms)
        self.end.setRange(result.window_start_ms, result.window_end_ms)
        self.start.setValue(self.request.start_ms)
        self.end.setValue(self.request.end_ms)
        self.range_label.setText(f"搜索范围：{fmt_time(result.window_start_ms)}–{fmt_time(result.window_end_ms)}")
        self._add_candidate_items(0)
        if result.candidates:
            self.candidates.setCurrentItem(self.candidates.topLevelItem(0))
        self._summarize_result()
        self._filter_candidates()

    def _add_candidate_items(self, start):
        for index in range(start, len(self.result.candidates)):
            candidate = self.result.candidates[index]
            kind = getattr(candidate, "kind", "asr")
            item = QTreeWidgetItem([candidate.text, "语义推测（未确认）" if kind == "semantic" else "音频识别",
                                    "／".join(candidate.sources), "；".join(candidate.warnings)])
            item.setData(0, Qt.ItemDataRole.UserRole, index)
            for column in range(4):
                item.setToolTip(column, "\n".join(candidate.warnings))
            self.candidates.addTopLevelItem(item)
        self.candidates.resizeColumnToContents(1)
        self.candidates.resizeColumnToContents(2)
        self.candidates.setColumnWidth(2, min(220, self.candidates.columnWidth(2)))

    def _summarize_result(self, prefix=""):
        result = self.result
        count = len(result.candidates)
        status = f"已生成 {count} 个候选（{result.device}）；请试听、修改，再点「补入原文」。" if count else (
            "没有可靠候选。可以扩大搜索范围重新听译，或试听后手动输入。")
        if result.diagnostics:
            status += "\n" + "；".join(result.diagnostics)
        semantic_count = sum(getattr(c, "kind", "asr") == "semantic" for c in result.candidates)
        if semantic_count:
            status += f"\n音频识别 {count - semantic_count} 个；中文语义推测 {semantic_count} 个（未经音频确认）。可用「查看」筛选。"
        self._state("听译完成 — 请核对候选" if count else "听译完成 — 无可靠候选", busy=False,
                    detail=(prefix + "\n" if prefix else "") + status)

    def _finish_more_translations(self, job):
        self._refreshing = False
        self._completed = True
        if self._invalid:
            self._state("结果已失效", busy=False, detail=self._invalid)
        elif job.cancelled:
            self._state("翻译建议已停止", busy=False, detail="原候选和编辑文字已保留，可以再次生成更多建议。")
        elif job.error is not None:
            self._state("翻译建议生成失败", busy=False, detail="原候选仍可使用：" + str(job.error))
        else:
            try:
                from subkit.listening_semantics import add_semantic_candidates

                batch = job.result
                metadata = copy.deepcopy({k: v for k, v in batch.items() if k != "texts"})
                old_count = len(self.result.candidates)
                added = add_semantic_candidates(self.result, batch["texts"], translation=self.cn_text,
                    start=max(self.result.window_start_ms, self.request.start_ms),
                    end=min(self.result.window_end_ms, self.request.end_ms), origin="本地中译日模型",
                    model=batch["model"], provenance=metadata)
                batches = self.result.metadata.setdefault("semantic_batches", [])
                if not batches and self.result.metadata.get("semantic"):
                    batches.append(copy.deepcopy(self.result.metadata["semantic"]))
                batches.append(metadata)
                self.result.metadata["semantic"] = copy.deepcopy(metadata)
                self._add_candidate_items(old_count)
                self._filter_candidates()
                if self._refresh_draft is not None:
                    text, start, end, item = self._refresh_draft
                    self.candidates.setCurrentItem(item)
                    self.text.setPlainText(text)
                    self.start.setValue(start)
                    self.end.setValue(end)
                prefix = (f"新增 {len(added)} 条翻译建议；原候选和编辑文字已保留。" if added else
                          "这一轮没有不同的新说法；原候选已保留，可以再试一次或手动修改。")
                self._summarize_result(prefix)
            except Exception as exc:
                self._state("翻译建议生成失败", busy=False, detail="原候选仍可使用：" + str(exc))
        self._refresh_draft = None
        self._cleanup_directory("_refresh_directory", "_refresh_identity")
        self._enabled()

    def _filter_candidates(self, *_args):
        if not hasattr(self, "candidates") or self.result is None:
            return
        kind = self.kind_filter.currentData()
        visible = []
        for i in range(self.candidates.topLevelItemCount()):
            item = self.candidates.topLevelItem(i)
            candidate = self.result.candidates[item.data(0, Qt.ItemDataRole.UserRole)]
            hidden = bool(kind and getattr(candidate, "kind", "asr") != kind)
            item.setHidden(hidden)
            if not hidden:
                visible.append(item)
        current = self.candidates.currentItem()
        if current is None or current.isHidden():
            self.candidates.setCurrentItem(visible[0] if visible else None)
            if not visible:
                self.text.clear()
        self._enabled()

    def _selected_candidate(self):
        item = self.candidates.currentItem()
        if item is None or self.result is None:
            return None
        index = item.data(0, Qt.ItemDataRole.UserRole)
        return self.result.candidates[index] if isinstance(index, int) and 0 <= index < len(self.result.candidates) else None

    def _candidate_changed(self, *_args):
        candidate = self._selected_candidate()
        if candidate is not None:
            self.text.setPlainText(candidate.text)
            self.start.setValue(candidate.start_ms)
            self.end.setValue(candidate.end_ms)
            self.warning.setText("；".join(candidate.warnings) or "请确认这是目标背景声，而非附近的前景对白。")
        self._enabled()

    def _enabled(self):
        if not hasattr(self, "adopt_button"):
            return
        running = self.job is not None and not self.job.finished.is_set()
        ready = self._completed and self.result is not None and self._usable_result and not running
        valid_times = bool(self.result is not None and self.result.window_start_ms <= self.start.value()
                           < self.end.value() <= self.result.window_end_ms)
        self.adopt_button.setEnabled(bool(ready and not self._invalid and self.text.toPlainText().strip()
                                          and valid_times))
        self.stop_button.setEnabled(running and not self.job.cancelled)
        self.rerun_button.setEnabled(not self._invalid and self._pending_done is None)
        self.more_button.setEnabled(bool(ready and not self._invalid and self.cn_text.strip()
                                          and self._pending_done is None))
        self.original_button.setEnabled(bool(self.result and self.result.clips.get("original") and not running))
        self.candidate_button.setEnabled(bool(self._selected_candidate() and not running))
        for widget in (self.candidates, self.text, self.start, self.end, self.kind_filter):
            widget.setEnabled(not running)

    def selection(self):
        if self._adopted is not None:
            return self._adopted
        text = self.text.toPlainText().strip()
        if not self.adopt_button.isEnabled() or not text:
            return None
        candidate = self._selected_candidate()
        if candidate is None:
            from subkit.listening import Candidate

            audio = self.result.clips.get("original", "") if self.result is not None else ""
            candidate = Candidate(text, self.start.value(), self.end.value(), ["manual"], audio,
                                  ["没有采用识别候选，由用户手动输入。"])
            candidate.kind = "manual"
        return candidate, text, self.start.value(), self.end.value()

    def invalidate(self, message: str):
        self._invalid = message
        self._pending_request = None
        if self.job is not None and not self.job.finished.is_set():
            self.job.cancel()
        running = self.job is not None and not self.job.finished.is_set()
        self._state("正在停止 — 当前组已变化" if running else "结果已失效", busy=running, detail=message)
        self._enabled()

    def play_original(self):
        if self.result is not None:
            self._play_audio(self.result.clips.get("original", ""), 0,
                             self.result.window_end_ms - self.result.window_start_ms)

    def play_candidate(self):
        candidate = self._selected_candidate()
        if candidate is not None and self.result is not None:
            self._play_audio(candidate.audio, self.start.value() - self.result.window_start_ms,
                             self.end.value() - self.result.window_start_ms)

    def _play_audio(self, path: str, start: int, end: int):
        if not path or not Path(path).is_file() or end <= start:
            self.warning.setText("找不到这份试听音频，请重新听译。")
            return
        self._stop_audio()
        self._audio_bounds = max(0, start), max(0, end)
        self._audio_pending = True
        url = QUrl.fromLocalFile(str(Path(path).resolve()))
        if self.player.source() == url and self.player.mediaStatus() in (
                QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia,
                QMediaPlayer.MediaStatus.EndOfMedia):
            self._audio_status(QMediaPlayer.MediaStatus.LoadedMedia)
        else:
            self.player.setSource(url)

    def _audio_status(self, status):
        if self._audio_pending and self._audio_bounds is not None and status in (
                QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
            self._audio_pending = False
            self.player.setPosition(self._audio_bounds[0])
            self.player.play()

    def _audio_position(self, position):
        if self._audio_bounds is not None and position >= self._audio_bounds[1]:
            self._stop_audio()

    def _audio_error(self, _error, text):
        if text:
            self.warning.setText("试听失败：" + text)

    def _stop_audio(self, *, clear=False):
        if not hasattr(self, "player"):
            return
        self._audio_pending = False
        self._audio_bounds = None
        self.player.stop()
        if clear:
            self.player.setSource(QUrl())

    def _cleanup_output(self):
        self._cleanup_directory("_refresh_directory", "_refresh_identity")
        self._cleanup_directory("_directory", "_directory_identity")

    def _cleanup_directory(self, attribute, identity_attribute):
        """Only direct regular files in our fresh run directory; never traverse subfolders."""
        directory = getattr(self, attribute)
        identity = getattr(self, identity_attribute)
        if directory is None or self.job is not None and not self.job.finished.is_set():
            return
        try:
            if directory.is_symlink() or directory.is_junction() or directory.resolve() != identity:
                return
            for path in directory.iterdir():
                if not path.is_symlink() and not path.is_junction() and path.is_file() and path.resolve().parent == identity:
                    path.unlink()
            directory.rmdir()
        except OSError:
            pass  # Leave unexpected directories or locked files in this run's isolated temp folder.
        setattr(self, attribute, None)
        setattr(self, identity_attribute, None)

    def accept(self):
        selection = self.selection()
        if selection is not None:
            if self.before_adopt is not None:
                try:
                    if not self.before_adopt(selection):
                        return
                except Exception as exc:
                    self.status.setText("补入原文失败：" + str(exc))
                    return
            self._adopted = selection
            self.done(QDialog.DialogCode.Accepted)

    def reject(self):
        self.done(QDialog.DialogCode.Rejected)

    def done(self, code):
        if self.job is not None and not self.job.finished.is_set():
            self._pending_request = None
            self._pending_done = code
            self.job.cancel()
            self._state("正在停止…", busy=True, detail="正在停止听译，完成后关闭…")
            self.adopt_button.setEnabled(False)
            self.rerun_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            return
        self._stop_audio(clear=True)
        self._cleanup_output()
        self.timer.stop()
        self._retired = True
        super().done(code)

    def closeEvent(self, event):
        if self.job is not None and not self.job.finished.is_set():
            self.reject()
            event.ignore()
        else:
            self._stop_audio(clear=True)
            self._cleanup_output()
            self.timer.stop()
            super().closeEvent(event)
