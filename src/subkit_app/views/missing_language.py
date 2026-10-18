"""L1 missing-language input windows and ordered clipboard distribution."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QHeaderView, QLabel,
                               QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from subkit.ass.text import plain_text
from subkit.languages import lane_label
from .group_table import fmt_time
from ..help import dialog_buttons


def distribute_lines(text: str, count: int) -> tuple[list[str], str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    note = "" if len(lines) == count else f"粘贴了 {len(lines)} 行，选中了 {count} 组：" + (
        "多的没有用" if len(lines) > count else "少的空着")
    return (lines[:count] + [""] * max(0, count - len(lines))), note


class PasteEdit(QLineEdit):
    def __init__(self, paste, parent=None):
        super().__init__(parent)
        self.paste_lines = paste

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Paste) and len(QApplication.clipboard().text().splitlines()) > 1:
            self.paste_lines()
            event.accept()
            return
        super().keyPressEvent(event)


class MissingLanguageDialog(QDialog):
    help_anchor = "review"

    def __init__(self, doc, gids, lane, languages, parent=None):
        super().__init__(parent, Qt.WindowType.Tool)
        self.setWindowTitle("补上" + lane_label(lane, languages))
        self.lane = lane
        selected = set(gids)
        groups = [g for g in doc.groups if g.id in selected]
        self.groups = [g for g in groups if not any(m.lane == lane and m.role != "comment" for m in doc.members_of(g))]
        self.inputs = []
        layout = QVBoxLayout(self)
        skipped = len(groups) - len(self.groups)
        if skipped:
            layout.addWidget(QLabel(f"{skipped} 组已经有{lane_label(lane, languages)}，不列出"))
        self.table = None
        if len(gids) == 1:
            edit = QLineEdit()
            edit.returnPressed.connect(self.accept)
            self.inputs.append(edit)
            layout.addWidget(edit)
            self.resize(480, 120)
        else:
            other = "translation" if lane == "original" else "original"
            self.table = QTableWidget(len(self.groups), 3)
            self.table.setHorizontalHeaderLabels(["时间", lane_label(other, languages), lane_label(lane, languages) + "（输入）"])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            for row, g in enumerate(self.groups):
                start, end = doc.group_time(g)
                text = " ／ ".join(plain_text(m.text) for m in doc.members_of(g) if m.lane == other)
                for column, value in enumerate((fmt_time(start) + "–" + fmt_time(end), text)):
                    item = QTableWidgetItem(value)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(row, column, item)
                edit = PasteEdit(self.paste) if row == 0 else QLineEdit()
                self.inputs.append(edit)
                self.table.setCellWidget(row, 2, edit)
            layout.addWidget(self.table)
            paste = QPushButton("从剪贴板粘贴")
            paste.clicked.connect(self.paste)
            layout.addWidget(paste)
            self.resize(850, 360)
        self.note = QLabel()
        layout.addWidget(self.note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.add_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.add_button.setText("补上")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        dialog_buttons(self, buttons)
        layout.addWidget(buttons)
        for edit in self.inputs:
            edit.textChanged.connect(self._enabled)
        self._enabled()

    def _enabled(self):
        self.add_button.setEnabled(any(edit.text().strip() for edit in self.inputs))

    def accept(self):
        if any(edit.text().strip() for edit in self.inputs):
            super().accept()

    def texts(self):
        return {g.id: edit.text() for g, edit in zip(self.groups, self.inputs) if edit.text().strip()}

    def paste(self):
        lines, note = distribute_lines(QApplication.clipboard().text(), len(self.inputs))
        for edit, text in zip(self.inputs, lines):
            edit.setText(text)
        self.note.setText(note)
