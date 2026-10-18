"""Review a frozen whole-episode timing proposal before any document edit."""
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox,
                              QHBoxLayout, QLabel, QSpinBox, QTableWidget,
                              QTableWidgetItem, QVBoxLayout)

from subkit.kalign.dialogue import DialogueEdge, suggest_times
from ..dialogue_alignment import manual_edges, valid_edges
from ..help import dialog_buttons


@dataclass
class ApplyRow:
    gid: str
    text: str
    current: tuple[int, int]
    suggested: tuple[int, int] | None
    manual: tuple[bool, bool]
    flags: list[str]


def episode_suggestions(doc, records, keyframes, padding):
    """Compute all proposals once against current times, including unsuggested neighbours."""
    groups = sorted((g for g in doc.groups if g.members and "hidden" not in g.flags
                     and g.kind in ("dialogue", "song")), key=doc.group_time)
    times = [doc.group_time(g) for g in groups]
    edges, changed = [], []
    for group in groups:
        valid, stale = valid_edges(doc, group, records)
        changed.append(stale)
        edges.append(DialogueEdge(valid[0].mid, min(e.speech_start for e in valid),
                     max(e.speech_end for e in valid), max(e.conf for e in valid),
                     list(dict.fromkeys(f for e in valid for f in e.flags)))
                     if valid and not stale else None)
    suggestions = suggest_times(edges, times, keyframes if keyframes is not None else [],
                                lead_in_ms=padding[0], lead_out_ms=padding[1], previous="current")
    rows = []
    for group, current, edge, suggestion, stale in zip(groups, times, edges, suggestions, changed):
        if suggestion is None and not stale:
            continue
        manual = manual_edges(doc, group)
        flags = ["文字改过"] if stale else []
        if edge is not None and "disagree" in edge.flags:
            flags.append("两种识别不一致")
        if "manual" in manual.values():
            flags.append("手调过")
        text = " / ".join("".join(r.text for _, r in m.visible_runs()) for m in doc.members_of(group)
                          if m.lane == "original" and m.lang == "ja")
        rows.append(ApplyRow(group.id, text[:60], current, suggestion,
                            (manual["start"] == "manual", manual["end"] == "manual"), flags))
    return rows, sum(changed)


def _time(ms):
    seconds, millis = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


class DialogueAlignApplyDialog(QDialog):
    help_anchor = "dialogue-align"

    def __init__(self, doc, records, keyframes, padding, parent=None):
        super().__init__(parent)
        self.setWindowTitle("精确对话轴（整集采用）")
        self.resize(1050, 520)
        self.rows, stale = episode_suggestions(doc, records, keyframes, padding)
        layout = QVBoxLayout(self)
        applicable = sum(row.suggested is not None for row in self.rows)
        self.summary = QLabel(f"{applicable} 句有建议；{stale} 句文字改过，需要重新分析")
        layout.addWidget(self.summary)
        filters = QHBoxLayout()
        self.filter = QCheckBox("只显示相差超过")
        self.threshold = QSpinBox()
        self.threshold.setRange(0, 3600000)
        self.threshold.setValue(100)
        filters.addWidget(self.filter)
        filters.addWidget(self.threshold)
        filters.addWidget(QLabel("ms 的"))
        filters.addStretch()
        layout.addLayout(filters)
        self.table = QTableWidget(len(self.rows), 7)
        self.table.setHorizontalHeaderLabels(["时间", "日文开头", "开始", "当前开始→建议开始（差值）",
                                              "结束", "当前结束→建议结束（差值）", "标记"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        for i, row in enumerate(self.rows):
            values = [_time(row.current[0]), row.text, "", "", "", "", "／".join(row.flags)]
            for j, value in enumerate(values):
                self.table.setItem(i, j, QTableWidgetItem(value))
            for edge, check_col, time_col in ((0, 2, 3), (1, 4, 5)):
                item = self.table.item(i, check_col)
                if row.suggested is None:
                    item.setFlags(Qt.ItemFlag.ItemIsSelectable)
                    self.table.item(i, time_col).setText(f"{_time(row.current[edge])} → —")
                    self.table.item(i, time_col).setForeground(QBrush(QColor("gray")))
                    continue
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked if row.manual[edge] else Qt.CheckState.Checked)
                delta = row.suggested[edge] - row.current[edge]
                self.table.item(i, time_col).setText(
                    f"{_time(row.current[edge])} → {_time(row.suggested[edge])} ({delta:+d} ms)")
                if row.manual[edge]:
                    item.setForeground(QBrush(QColor("gray")))
                    item.setText("手调过")
                    self.table.item(i, time_col).setForeground(QBrush(QColor("gray")))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        buttons = QDialogButtonBox()
        self.apply_button = buttons.addButton("采用勾选的", QDialogButtonBox.ButtonRole.AcceptRole)
        self.apply_button.setEnabled(bool(applicable))
        buttons.addButton("取消", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        dialog_buttons(self, buttons)
        layout.addWidget(buttons)
        self.filter.toggled.connect(self._filter_rows)
        self.threshold.valueChanged.connect(self._filter_rows)

    def _filter_rows(self):
        for i, row in enumerate(self.rows):
            if row.suggested is None:
                self.table.setRowHidden(i, False)  # Keep the stale warning visible, never applicable.
                continue
            difference = max(abs(a - b) for a, b in zip(row.current, row.suggested))
            self.table.setRowHidden(i, self.filter.isChecked() and difference <= self.threshold.value())

    def checked_changes(self):
        """Hidden rows retain their selections and still participate in application."""
        changes = []
        for i, row in enumerate(self.rows):
            if row.suggested is None:
                continue
            start = row.suggested[0] if self.table.item(i, 2).checkState() == Qt.CheckState.Checked else None
            end = row.suggested[1] if self.table.item(i, 4).checkState() == Qt.CheckState.Checked else None
            if start is not None or end is not None:
                changes.append((row.gid, start, end))
        return changes
