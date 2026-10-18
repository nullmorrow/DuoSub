"""Whole-song lyric mapping editor (LYRICS 2.3)."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHeaderView, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from subkit.core import karaoke as kara, lyrics_map
from subkit.core.readings import visible_map
from subkit.core.songs import song_span
from subkit.languages import lane_label
from .group_table import fmt_time
from ..help import dialog_buttons


class LyricsMapDialog(QDialog):
    help_anchor = "karaoke"
    playRequested = Signal(int, int)

    def __init__(self, doc, gid, languages, parent=None, *, follow_notes=True):
        super().__init__(parent, Qt.WindowType.Tool)
        self.setWindowTitle("歌词对应")
        self.doc, self.languages = doc, languages
        self.follow_notes = follow_notes
        self.groups = [doc.group(g) for g in (song_span(doc, gid) or [gid])]
        self.rows, self.suggestions, self.valid, self.initial = [], set(), {}, {}
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("翻译的一段（原声的几拍）。灰色是自动建议，需要检查；改动或点「采用」后变黑。"
                                "「应用」只改黑色且有改动的行，灰色建议和没改的行保持原来的轴。"))
        self.table = QTableWidget(len(self.groups), 4)
        self.table.setHorizontalHeaderLabels(["时间", lane_label("original", languages) + "拍", "对应（可以直接改）", "状态"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 175)
        self.table.setColumnWidth(1, 230)
        self.table.setColumnWidth(3, 275)
        for row, group in enumerate(self.groups):
            original = lyrics_map.member(doc, group, "original")
            translation = lyrics_map.member(doc, group, "translation")
            originals = kara.effective(doc, original).beats if original else []
            text = visible_map(translation)[0].text if translation else ""
            current = lyrics_map.current_mapping(doc, original, translation) if original and translation else ""
            suggestion = lyrics_map.auto_mapping(originals, text, follow_notes=self.follow_notes) if original and translation else ""
            self.rows.append((originals, text))
            self.initial[row] = current
            start, end = doc.group_time(group)
            for col, value in enumerate((fmt_time(start) + "–" + fmt_time(end), "|".join(b.sung for b in originals),
                                         current or suggestion, "")):
                item = QTableWidgetItem(value)
                if col != 2 or not (original and translation):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, col, item)
            if not current and suggestion:
                self.suggestions.add(row)
                self.table.item(row, 2).setForeground(QColor("#888888"))
            self._validate(row)
        self.table.itemChanged.connect(self._changed)
        self.table.setCurrentCell(next(i for i, g in enumerate(self.groups) if g.id == gid), 2)
        layout.addWidget(self.table)
        actions = QHBoxLayout()
        self.auto_button = QPushButton("自动对应")
        self.adopt_button = QPushButton("采用")
        self.play_button = QPushButton("试听这一行")
        for button, slot in ((self.auto_button, self.auto), (self.adopt_button, self.adopt), (self.play_button, self.play)):
            button.clicked.connect(slot)
            actions.addWidget(button)
        actions.addStretch()
        layout.addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.setText("应用")
        self.apply_button.clicked.connect(self.accept)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.rejected.connect(self.reject)
        dialog_buttons(self, buttons)
        layout.addWidget(buttons)
        self.table.currentCellChanged.connect(self._selected)
        self._selected()
        self.resize(1120, 480)

    def _validate(self, row):
        reason = lyrics_map.disabled_reason(self.doc, self.groups[row], self.languages)
        try:
            if reason:
                raise ValueError(reason)
            lyrics_map.parse(self.table.item(row, 2).text(), *self.rows[row])
            status = "✓ 建议，没采用（应用时不改）" if row in self.suggestions else "✓"
            self.valid[row] = self.table.item(row, 2).text()
        except ValueError as exc:
            status = str(exc)
            self.valid.pop(row, None)
        item = self.table.item(row, 3)
        item.setText(status)
        item.setToolTip(status)
        item.setForeground(QColor("#d33") if not status.startswith("✓")
                           else QColor("#888888") if row in self.suggestions else self.palette().text().color())

    def _changed(self, item):
        if item.column() != 2:
            return
        self.table.blockSignals(True)
        try:
            self.suggestions.discard(item.row())
            item.setForeground(self.palette().text().color())
            self._validate(item.row())
        finally:
            self.table.blockSignals(False)
        self._selected()

    def _selected(self, *_args):
        row = self.table.currentRow()
        enabled = row >= 0 and not lyrics_map.disabled_reason(self.doc, self.groups[row], self.languages)
        self.auto_button.setEnabled(enabled)
        self.adopt_button.setEnabled(row in self.suggestions)
        self.play_button.setEnabled(row >= 0)
        self.apply_button.setEnabled(bool(self.mappings()))

    def auto(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.item(row, 2).setText(lyrics_map.auto_mapping(*self.rows[row], follow_notes=self.follow_notes))
            self.adopt()

    def adopt(self):
        row = self.table.currentRow()
        if row >= 0:
            self._changed(self.table.item(row, 2))

    def play(self):
        row = self.table.currentRow()
        if row >= 0:
            self.playRequested.emit(*self.doc.group_time(self.groups[row]))

    def mappings(self):
        """Adopted or edited rows that differ from the current timing; grey suggestions never
        overwrite a line (its timing may be finer than this text can express)."""
        return {self.groups[row].id: text for row, text in self.valid.items()
                if row not in self.suggestions and text != self.initial.get(row)}

    def accept(self):
        # Commit any active table editor before collecting the final validated rows.
        self.apply_button.setFocus()
        if not self.mappings():
            return
        invalid = len(self.groups) - len(self.valid)
        if invalid and QMessageBox.question(self, "歌词对应", f"{invalid} 行还没对上，这些行不改。继续吗？",
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                           QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        super().accept()
