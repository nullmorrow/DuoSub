"""「集 → 原字幕副本（检查和修复）…」: the project's copies of the imported subtitles, put back when missing."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from subkit.core.importer import SourceChanged
from subkit.core.source_copies import CopyProblem, find_in_folder, fix_from_file, fix_from_original, scan_project
from subkit.languages import lane_label
from ..help import dialog_buttons

STATE = {"missing": "找不到副本", "changed": "副本内容和导入时不同"}


class SourceCopiesDialog(QDialog):
    help_anchor = "source-copies"
    fixed = Signal(list)                   # episode ids whose copies were put back

    def __init__(self, project, parent=None, *, problems: Optional[list[CopyProblem]] = None):
        super().__init__(parent)
        self.setWindowTitle("原字幕副本")
        self.resize(980, 420)
        self.project = project
        self.problems = scan_project(project) if problems is None else problems
        layout = QVBoxLayout(self)
        intro = QLabel("工程只用自己保存的原字幕副本（每集的 sources 文件夹），不读导入时的原路径。"
                       "副本找不到或被改过时，只能用和导入时一字节不差的文件放回：先试导入时的位置，"
                       "也可以选一个文件夹，在里面按内容查找（文件名改过也能找到）。")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["集", "语言", "副本", "状态", "导入时的位置"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        header = self.table.horizontalHeader()
        for col in range(4):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        layout.addWidget(self.table, 1)
        self.message = QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QHBoxLayout()
        self.original_button = QPushButton("从导入时的位置恢复")
        self.folder_button = QPushButton("在文件夹里查找…")
        self.file_button = QPushButton("选择文件…")
        self.file_button.setToolTip("给选中的这一行选一个文件")
        close = QPushButton("关闭")
        self.original_button.clicked.connect(self.fix_from_originals)
        self.folder_button.clicked.connect(self.search_folder)
        self.file_button.clicked.connect(self.choose_file)
        close.clicked.connect(self.accept)
        for button in (self.original_button, self.folder_button, self.file_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(close)
        dialog_buttons(self, buttons, auxiliary=(self.original_button, self.folder_button, self.file_button))
        layout.addLayout(buttons)
        self.refresh()

    def open_problems(self) -> list[CopyProblem]:
        return [p for p in self.problems if p.fixed_from is None]

    def refresh(self, note: str = "") -> None:
        self.table.setRowCount(len(self.problems))
        languages = getattr(self.project, "languages", {})
        for row, p in enumerate(self.problems):
            state = f"已放回（来自 {p.fixed_from.name}）" if p.fixed_from else STATE.get(p.state, p.state)
            values = (p.eid, lane_label(p.lane, languages) if p.lane else p.sid, p.copy.name, state,
                      str(p.original) if p.original.name else "（没有记录）")
            for col, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setToolTip(str(p.fixed_from) if col == 3 and p.fixed_from else value)
                if col == 3:
                    cell.setForeground(QColor("#1a7f37" if p.fixed_from else "#cf222e"))
                self.table.setItem(row, col, cell)
        remaining = len(self.open_problems())
        summary = ("整个工程的原字幕副本都完整。" if not self.problems else
                   f"还有 {remaining} 份副本需要放回；放回前这些集仍可编辑，但导出会缺原字幕里的样式和特效行等。"
                   if remaining else "全部副本都已放回。")
        self.message.setText(summary + (("\n" + note) if note else ""))
        for button in (self.original_button, self.folder_button, self.file_button):
            button.setEnabled(bool(remaining))

    def _done(self, fixed: list[CopyProblem], note: str) -> None:
        if fixed:
            self.fixed.emit(sorted({p.eid for p in fixed}))
        self.refresh(note)

    def fix_from_originals(self) -> None:
        fixed = [p for p in self.open_problems() if fix_from_original(p)]
        left = len(self.open_problems())
        self._done(fixed, f"从导入时的位置放回 {len(fixed)} 份。" +
                   (f"其余 {left} 份导入时的文件已不在或被改过，可以在文件夹里查找。" if left else ""))

    def search_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "在这个文件夹（含子文件夹）里查找原字幕")
        if not folder:
            return
        problems = self.open_problems()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            found = find_in_folder(Path(folder), problems)
            fixed, failed = [], []
            for i, path in found.items():
                try:
                    fix_from_file(problems[i], path)
                    fixed.append(problems[i])
                except (OSError, SourceChanged) as exc:
                    failed.append(f"{problems[i].eid}：{exc}")
        finally:
            QApplication.restoreOverrideCursor()
        self._done(fixed, f"在「{folder}」里找到并放回 {len(fixed)} 份。" + "\n".join(failed))

    def choose_file(self) -> None:
        row = self.table.currentRow()
        if not 0 <= row < len(self.problems) or self.problems[row].fixed_from is not None:
            QMessageBox.information(self, "原字幕副本", "先在表格里选一行还没放回的副本。")
            return
        p = self.problems[row]
        start = str(p.original.parent) if p.original.name and p.original.parent.exists() else ""
        path, _ = QFileDialog.getOpenFileName(self, f"选择 {p.eid} 的原字幕（{p.copy.name}）", start,
                                              "字幕 (*.ass *.ssa *.srt *.vtt *.txt);;所有文件 (*)")
        if not path:
            return
        try:
            fix_from_file(p, Path(path))
        except (OSError, SourceChanged) as exc:
            QMessageBox.warning(self, "原字幕副本", f"没有放回：{exc}")
            return
        self._done([p], f"已放回 {p.eid} 的 {p.copy.name}。")
