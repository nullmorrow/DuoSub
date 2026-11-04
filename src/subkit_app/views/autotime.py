"""One-click timing result and background, editable timing preview."""

from __future__ import annotations

import copy
import threading
from contextlib import contextmanager
from dataclasses import replace

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QComboBox, QCheckBox, QDialog, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

from subkit.autotime import TimingMap
from subkit.proc import CancelledError
from subkit.core.autotime_doc import Settings, plan
from .. import pool as job_pool
from ..help import dialog_buttons, open_manual
from .group_table import fmt_time
from .passages import _modal, _label

CONFIDENCE = {"high": "把握大", "medium": "把握一般", "low": "把握小"}
GRADE = {"good": "好", "fair": "一般", "bad": "差"}
BAD_QUALITY = "不少句子本身的开始、结束时间就不准（不是整体偏移），自动调轴不改单句；需要的话在列表里逐句调。"


@contextmanager
def busy(window):
    previous = window.statusBar().currentMessage()
    window.statusBar().showMessage("正在自动调轴…")
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        window.statusBar().repaint()
        yield
    finally:
        QApplication.restoreOverrideCursor()
        window.statusBar().showMessage(previous)


def quality_text(preview, *, changed=True, preview_only=False) -> str:
    seconds = preview.quality.max_offset_ms / 1000
    if changed:
        overall = (f"整体：应用后将对齐（调之前最多差 {seconds:.2f} 秒；预览）。" if preview_only else
                   f"整体：已对齐（调之前最多差 {seconds:.2f} 秒）。")
    elif all(s.confidence == "low" for s in preview.segments):
        overall = "整体：没有找到把握大的时间差，没有改动。"
    elif preview.changes or preview.excluded:
        overall = "整体：没有改动。可在逐段查看和调整中检查没动的组。"
    else:
        overall = f"整体：本来就对得上（最多差 {seconds:.2f} 秒）。"
    lines = {"good": "各句的时间都和声音吻合。",
             "fair": "少数句子本身的开始、结束时间不太准，自动调轴不改单句。", "bad": BAD_QUALITY}
    return overall + "\n单句：" + lines[preview.quality.grade]


def segment_shifts(segment):
    # Use this segment at its right edge: TimingMap.map(end) selects the next segment.
    return (segment.map(segment.start_ms) - segment.start_ms,
            segment.map(segment.end_ms) - segment.end_ms)


def segment_text(segment) -> str:
    start, end = segment_shifts(segment)
    def shift(value):
        return f"{'提前' if value < 0 else '推后'} {abs(value) / 1000:.2f} 秒"
    movement = f"整段{shift(start)}" if abs(end - start) < 50 else f"开头{shift(start)}，结尾{shift(end)}"
    confidence = CONFIDENCE[segment.confidence] + ("，没改" if segment.confidence == "low" else "")
    return f"{fmt_time(segment.start_ms)}–{fmt_time(segment.end_ms)}　{movement}（{confidence}）"


def _section(text: str):
    label = _label(text)
    label.setStyleSheet("color: #6e7781; font-weight: bold; margin-top: 6px;")
    return label


def _help(dialog) -> None:
    dialog.help_shortcut = QShortcut(QKeySequence("F1"), dialog)
    dialog.help_shortcut.activated.connect(lambda: open_manual("autotime"))


def is_last(session, transaction) -> bool:
    done = session.editor.history.done
    return transaction is not None and bool(done) and done[-1] is transaction


class EpisodeAutoProcessDialog(QDialog):
    help_anchor = "autotime"

    def __init__(self, offset_ms, segmented: bool, names: list[str], parent) -> None:
        super().__init__(parent)
        _modal(self, "这一集可以自动处理")
        if parent.watch_mode:
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.action = None
        layout = QVBoxLayout(self)
        if offset_ms is not None:
            suffix = "（片中有一处前后差得不一样）" if segmented else ""
            direction = "早" if offset_ms > 0 else "晚"
            layout.addWidget(_label(f"字幕比视频声音{direction}约 {abs(offset_ms) / 1000:.2f} 秒{suffix}。"))
        if names:
            layout.addWidget(_label(f"找到重复段落：{'、'.join(names)}。"))
        layout.addWidget(_label("要现在处理吗？处理后都可以 Ctrl+Z 撤销。"))
        self.mute = QCheckBox("这一集不再询问")
        layout.addWidget(self.mute)
        buttons = QHBoxLayout()
        choices = {}
        for text, action in (("自动调轴", "autotime"), ("套用重复段落…", "passages"), ("先不处理", None)):
            if (action == "autotime" and offset_ms is None) or (action == "passages" and not names):
                continue
            button = QPushButton(text)
            button.clicked.connect(lambda checked=False, action=action: self.choose(action))
            buttons.addWidget(button)
            choices[action] = button
        primary = choices.get("autotime") or choices.get("passages")
        auxiliary = (choices["passages"],) if "autotime" in choices and "passages" in choices else ()
        dialog_buttons(self, buttons, primary=primary, auxiliary=auxiliary)
        layout.addLayout(buttons)
        _help(self)

    def choose(self, action) -> None:
        self.action = action
        self.accept()


class AutoTimeResultDialog(QDialog):
    help_anchor = "autotime"

    def __init__(self, session, preview, counts, transaction, features, parent) -> None:
        super().__init__(parent)
        _modal(self, "自动调轴")
        self.session, self.preview = session, preview
        self.transaction, self.features = transaction, features
        layout = QVBoxLayout(self)
        self.summary = _label(f"已把 {counts[0]} 组（{counts[1]} 行）挪到和视频声音对齐。可以按 Ctrl+Z 撤销。")
        font = self.summary.font()
        font.setBold(True)
        self.summary.setFont(font)
        if not counts[0]:
            if all(s.confidence == "low" for s in preview.segments):
                self.summary.setText("没有找到把握大的时间差，没有改动。")
            elif not preview.changes and not preview.excluded:
                self.summary.setText(f"字幕和视频声音已经对得上（最多差 {preview.quality.max_offset_ms / 1000:.2f} 秒），没有改动。")
            else:
                self.summary.setText("没有改动。可在逐段查看和调整中检查没动的组。")
        layout.addWidget(self.summary)
        layout.addWidget(_section("怎么挪的"))
        for s in preview.segments:
            layout.addWidget(_label(segment_text(s)))
        if preview.protected_count or preview.low_confidence_count:
            layout.addWidget(_section("没动的"))
        if preview.protected_count:
            layout.addWidget(_label(f"{preview.protected_count} 组你确认过或手动调过时间"))
        if preview.low_confidence_count:
            layout.addWidget(_label(f"{preview.low_confidence_count} 组所在的片段把握小"))
        layout.addWidget(_section("对齐结果"))
        self.quality_label = _label(quality_text(preview, changed=bool(counts[0])))
        layout.addWidget(self.quality_label)
        buttons = QHBoxLayout()
        self.undo_button = QPushButton("撤销")
        self.details_button = QPushButton("逐段查看和调整…")
        close = QPushButton("关闭")
        for button in (self.undo_button, self.details_button, close):
            buttons.addWidget(button)
        dialog_buttons(self, buttons, auxiliary=(self.undo_button, self.details_button))
        layout.addLayout(buttons)
        self.undo_button.setEnabled(is_last(session, transaction))
        self.undo_button.clicked.connect(self.undo)
        self.details_button.clicked.connect(self.details)
        close.clicked.connect(self.accept)
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)
        _help(self)

    def undo(self) -> None:
        if is_last(self.session, self.transaction):
            self.session.editor.history.undo()
        self.undo_button.setEnabled(False)

    def details(self) -> None:
        undone = is_last(self.session, self.transaction)
        self.undo()
        self.accept()
        dialog = AutoTimeDetailsDialog(self.session, self.features, self.parent().player,
                                       self.parent().settings, self.parent())
        try:
            applied = dialog.exec() == QDialog.DialogCode.Accepted
        finally:
            dialog.deleteLater()
        history = self.session.editor.history
        # Cancelling the details keeps the one-click result the user already had.
        if undone and not applied and history.undone and history.undone[-1] is self.transaction:
            history.redo()


class _PlanSignals(QObject):
    finished = Signal(int, object, str)


_LIVE: list["_PlanJob"] = []      # running jobs outlive a closed dialog; dropped on the GUI thread once finished


def _prune_live() -> None:
    _LIVE[:] = [job for job in _LIVE if not job.finished.is_set()]


class _PlanJob(QRunnable):
    def __init__(self, generation, doc, sources, features, settings, mapping):
        super().__init__()
        # Like the app's other jobs: Python owns the runnable, so the pool never deletes it under us.
        self.setAutoDelete(False)
        self.finished = threading.Event()
        self.generation = generation
        self.doc, self.sources = copy.deepcopy((doc, sources))
        self.features, self.settings = features, settings
        self.mapping = copy.deepcopy(mapping)
        self.signals = _PlanSignals()
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def check(self) -> None:
        if self._cancel.is_set():
            raise CancelledError()

    def run(self) -> None:
        result, error = None, ""
        try:
            self.check()
            result = plan(self.doc, self.sources, self.features, self.settings,
                          mapping_override=self.mapping, cancel_check=self.check)
            self.check()
        except CancelledError:
            result = None
        except Exception as exc:
            error = str(exc)
        try:
            self.signals.finished.emit(self.generation, result, error)
        finally:
            self.finished.set()


class AutoTimeDetailsDialog(QDialog):
    help_anchor = "autotime"

    def __init__(self, session, features, player, settings, parent) -> None:
        super().__init__(parent)
        _modal(self, "自动调轴 — 逐段查看和调整")
        self.resize(960, 720)
        self.session, self.features, self.player, self.settings = session, features, player, settings
        self.mapping_override = None
        self.current_plan = None
        self.group_items = {}
        self._generation = 0
        self._closed = False
        self._jobs = {}
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(200)
        self._debounce.timeout.connect(self._start_plan)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.reference, self.method, self.snap = QComboBox(), QComboBox(), QComboBox()
        for text, value in (("视频的声音（推荐）", "audio"), ("原声字幕原来的时间", "original"),
                            ("翻译字幕原来的时间", "translation")):
            self.reference.addItem(text, value)
        for text, value in (("整集一起挪", "shift"), ("分段挪", "piecewise"),
                            ("分段挪 + 纠正速度（推荐）", "piecewise_rate")):
            self.method.addItem(text, value)
        self.method.setCurrentIndex(2)
        for text, value in (("关", "off"), ("只调开始", "start"), ("开始和结束", "both"),
                            ("保持原时长", "keep_duration")):
            self.snap.addItem(text, value)
        self.snap.setCurrentIndex(max(0, self.snap.findData(settings.value("autotime/snap", "off"))))
        self.lead_in, self.lead_out = QDoubleSpinBox(), QDoubleSpinBox()
        for spin, key, default in ((self.lead_in, "lead_in", 0.27), (self.lead_out, "lead_out", 0.0)):
            spin.setRange(0, 60)
            spin.setDecimals(3)
            spin.setSingleStep(0.01)
            spin.setSuffix(" 秒")
            spin.setValue(float(settings.value("autotime/" + key, default)))
        for text, widget in (("对齐到", self.reference), ("方式", self.method),
                             ("逐行贴合人声（实验）", self.snap), ("比人声提前开始", self.lead_in),
                             ("比人声延后结束", self.lead_out)):
            form.addRow(text, widget)
        layout.addLayout(form)
        self.protected = QCheckBox("也调整确认过的组和手动调过时间的行")
        layout.addWidget(self.protected)
        layout.addWidget(_label("屏幕字、注释、带位置标签的行、歌曲行、附近没有人声的行，只跟着所在的段一起挪。"))
        self.segments = QTableWidget(0, 5)
        self.segments.setHorizontalHeaderLabels(["开始", "结束", "挪动（秒，负数=提前）", "结尾多挪（秒）", "把握"])
        layout.addWidget(self.segments)
        self.delete_breakpoint = QPushButton("和上一段合并")
        self.delete_breakpoint.clicked.connect(self.remove_breakpoint)
        layout.addWidget(self.delete_breakpoint)
        self.groups = QTableWidget(0, 4)
        self.groups.setHorizontalHeaderLabels(["应用 / 第几组", "原时间 → 新时间（±毫秒）", "内容", "把握 / 没动的原因"])
        layout.addWidget(self.groups, 1)
        self.note = _label("")
        layout.addWidget(self.note)
        buttons = QHBoxLayout()
        self.audition_button, self.apply_button = QPushButton("试听"), QPushButton("应用")
        cancel = QPushButton("取消")
        for button in (self.audition_button, self.apply_button, cancel):
            buttons.addWidget(button)
        dialog_buttons(self, buttons, primary=self.apply_button, auxiliary=(self.audition_button,))
        layout.addLayout(buttons)
        self.audition_button.clicked.connect(self.audition)
        self.apply_button.clicked.connect(self.apply)
        cancel.clicked.connect(self.reject)
        for combo in (self.reference, self.method):
            combo.currentIndexChanged.connect(self.reset_mapping)
        self.snap.currentIndexChanged.connect(self.recompute)
        self.lead_in.valueChanged.connect(self.recompute)
        self.lead_out.valueChanged.connect(self.recompute)
        self.protected.toggled.connect(self.recompute)
        self.segments.itemChanged.connect(self.edit_segment)
        _help(self)
        self.recompute()

    def selected_settings(self) -> Settings:
        return Settings(reference=self.reference.currentData(), method=self.method.currentData(),
                        snap=self.snap.currentData(), lead_in_ms=round(self.lead_in.value() * 1000),
                        lead_out_ms=round(self.lead_out.value() * 1000),
                        include_protected=self.protected.isChecked(), include_low_confidence=True)

    def reset_mapping(self) -> None:
        self.mapping_override = None
        self.recompute()

    def recompute(self) -> None:
        if self._closed:
            return
        self._generation += 1
        for job in self._jobs.values():
            job.cancel()
        self.current_plan = None
        self.apply_button.setEnabled(False)
        self.audition_button.setEnabled(False)
        self.segments.setEnabled(False)
        self.delete_breakpoint.setEnabled(False)
        self.groups.setEnabled(False)
        self.note.setText("正在计算…")
        self._debounce.start()

    def _start_plan(self) -> None:
        job = _PlanJob(self._generation, self.session.doc, self.session.episode.sources,
                       self.features, self.selected_settings(), self.mapping_override)
        self._jobs[job.generation] = job
        _prune_live()
        _LIVE.append(job)
        job.signals.finished.connect(self._plan_finished, Qt.ConnectionType.QueuedConnection)
        job_pool.start(job)

    @Slot(int, object, str)
    def _plan_finished(self, generation, p, error) -> None:
        self._jobs.pop(generation, None)
        _prune_live()
        if self._closed or generation != self._generation:
            return
        if p is None:
            self.note.setText(f"没有生成预览：{error}")
            return
        if p.revision != self.session.doc.revision:
            self.recompute()
            return
        self.audition_button.setEnabled(True)
        self.segments.setEnabled(True)
        self.groups.setEnabled(True)
        self.current_plan = p
        self.settings.setValue("autotime/snap", p.settings.snap)
        self.settings.setValue("autotime/lead_in", self.lead_in.value())
        self.settings.setValue("autotime/lead_out", self.lead_out.value())
        self.segments.blockSignals(True)
        try:
            self.segments.setRowCount(len(p.segments))
            for row, s in enumerate(p.segments):
                start_shift, end_shift = segment_shifts(s)
                for col, text in enumerate((fmt_time(s.start_ms), fmt_time(s.end_ms),
                                             f"{s.offset_ms / 1000:.2f}", f"{(end_shift - start_shift) / 1000:+.2f}", CONFIDENCE[s.confidence])):
                    item = QTableWidgetItem(text)
                    if col != 2:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.segments.setItem(row, col, item)
        finally:
            self.segments.blockSignals(False)
        self.groups.setRowCount(0)
        self.group_items = {}
        exclusions = {g.group_id: g.reason for g in p.excluded}
        changes = {g.group_id: g for g in p.changes}
        for number, g in enumerate(self.session.doc.groups, 1):
            members = self.session.doc.members_of(g)
            if not members:
                continue
            row = self.groups.rowCount()
            self.groups.insertRow(row)
            start, end = self.session.doc.group_time(g)
            segment = next((s for s in p.segments if s.start_ms <= start < s.end_ms),
                           p.segments[0] if start < p.segments[0].start_ms else p.segments[-1])
            low = any(s.confidence == "low" for s in p.segments if s.start_ms < end and s.end_ms > start)
            low = low or segment.confidence == "low"
            change = changes.get(g.id)
            changed_members = {c.member_id: c for c in change.members} if change else {}
            new_start = min(changed_members[m.id].start if m.id in changed_members else m.start for m in members)
            new_end = max(changed_members[m.id].end if m.id in changed_members else m.end for m in members)
            item = QTableWidgetItem(str(number))
            item.setData(Qt.ItemDataRole.UserRole, g.id)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            if change:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked if low else Qt.CheckState.Checked)
                self.group_items[g.id] = item
            self.groups.setItem(row, 0, item)
            reason = exclusions.get(g.id)
            label = "已隐藏，不改时间" if reason == "hidden" else "确认过或手动调过时间" if reason == "protected" else "把握小，没改" if reason else CONFIDENCE[segment.confidence]
            for col, text in enumerate((f"{fmt_time(start)}–{fmt_time(end)} → {fmt_time(new_start)}–{fmt_time(new_end)}"
                                       f"（{new_start - start:+d} / {new_end - end:+d} 毫秒）",
                                       " / ".join(m.text for m in members), label), 1):
                cell = QTableWidgetItem(text)
                cell.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.groups.setItem(row, col, cell)
        self.groups.resizeColumnsToContents()
        self.segments.resizeColumnsToContents()
        self.delete_breakpoint.setEnabled(len(p.segments) > 1)
        self.apply_button.setEnabled(bool(p.changes))
        self.note.setText("对齐结果\n" + quality_text(p, changed=bool(p.changes), preview_only=True))

    def done(self, result) -> None:
        self._closed = True
        self._generation += 1
        self._debounce.stop()
        for job in self._jobs.values():
            job.cancel()
        super().done(result)

    def edit_segment(self, item) -> None:
        if self.current_plan is None or item.column() != 2:
            return
        try:
            segments = list(self.current_plan.segments)
            segments[item.row()] = replace(segments[item.row()], offset_ms=float(item.text()) * 1000)
            self.mapping_override = TimingMap(segments, self.current_plan.mapping.method,
                                              self.current_plan.mapping.windows)
        except ValueError:
            self.recompute()
            return
        self.recompute()

    def remove_breakpoint(self) -> None:
        if self.current_plan is None or len(self.current_plan.segments) < 2:
            return
        segments = list(self.current_plan.segments)
        row = max(1, self.segments.currentRow())
        segments[row - 1] = replace(segments[row - 1], end_ms=segments[row].end_ms)
        del segments[row]
        self.mapping_override = TimingMap(segments, self.current_plan.mapping.method,
                                          self.current_plan.mapping.windows)
        self.recompute()

    def audition(self) -> None:
        row = self.groups.currentRow()
        if row < 0 or self.current_plan is None:
            return
        gid = self.groups.item(row, 0).data(Qt.ItemDataRole.UserRole)
        change = next((g for g in self.current_plan.changes if g.group_id == gid), None)
        changes = {c.member_id: c for c in change.members} if change else {}
        start = min(changes[m.id].start if m.id in changes else m.start
                    for m in self.session.doc.members_of(self.session.doc.group(gid)))
        self.parent()._stop_at = None
        self.player.seek(max(0, start / 1000))
        self.player.play()

    def apply(self) -> None:
        if self.current_plan is None:
            return
        chosen = [gid for gid, item in self.group_items.items() if item.checkState() == Qt.CheckState.Checked]
        try:
            with busy(self.parent()):
                self.session.editor.apply_autotime(self.current_plan, chosen)
        except Exception as exc:
            QMessageBox.warning(self, "自动调轴", f"没有应用：{exc}")
            return
        self.accept()
