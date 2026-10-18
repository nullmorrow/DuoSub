"""「集 → 视频文件位置…」: point episodes at videos that were moved or renamed."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from subkit.core.video_link import RULES, VIDEO_SUFFIXES, match_folder, relink, video_entries
from ..help import dialog_buttons

VIDEO_FILTER = "视频 (" + " ".join("*" + s for s in sorted(VIDEO_SUFFIXES)) + ");;所有文件 (*)"


class VideoLinksDialog(QDialog):
    """After 应用, :attr:`applied` lists ``(episode id, new video)``."""

    help_anchor = "video-links"

    def __init__(self, project, parent=None, *, focus_eid: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("视频文件位置")
        self.resize(1000, 460)
        self.project = project
        self.entries = video_entries(project)
        self.pending: dict[str, tuple[Path, str]] = {}
        self.applied: list[tuple[str, Path]] = []
        layout = QVBoxLayout(self)
        intro = QLabel("工程只记住视频在哪里（字幕在工程里另有副本）。视频挪过位置或改过名字时，在这里重新指定："
                       "选一个文件夹自动配对（先找同名文件，再找大小相同的文件，再按集号），或给某一集选文件，最后点「应用」。")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["集", "记录的视频", "状态", "新位置", "依据"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        header = self.table.horizontalHeader()
        for col in (0, 2, 4):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        self.message = QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QHBoxLayout()
        self.folder_button = QPushButton("在文件夹里查找…")
        self.file_button = QPushButton("选择文件…")
        self.file_button.setToolTip("给选中的这一集选视频；同一个文件夹里其他找不到视频的集会一起配对")
        self.apply_button = QPushButton("应用")
        close = QPushButton("关闭")
        self.folder_button.clicked.connect(self.search_folder)
        self.file_button.clicked.connect(self.choose_file)
        self.apply_button.clicked.connect(self.apply)
        close.clicked.connect(self.reject)
        for button in (self.folder_button, self.file_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(self.apply_button)
        buttons.addWidget(close)
        dialog_buttons(self, buttons, primary=self.apply_button, auxiliary=(self.folder_button, self.file_button))
        layout.addLayout(buttons)
        self.refresh()
        row = next((i for i, e in enumerate(self.entries) if e.eid == focus_eid), None)
        if row is not None:
            self.table.selectRow(row)

    def missing(self):
        return [e for e in self.entries if e.missing and e.eid not in self.pending]

    def refresh(self, note: str = "") -> None:
        self.table.setRowCount(len(self.entries))
        for row, e in enumerate(self.entries):
            state = ("没有视频" if e.recorded is None else "找不到" if e.missing else "找得到")
            new, rule = self.pending.get(e.eid, (None, ""))
            values = (e.eid, str(e.recorded) if e.recorded else "", state, str(new) if new else "", RULES.get(rule, rule))
            for col, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setToolTip(value)
                if col == 2:
                    cell.setForeground(QColor("#cf222e" if e.missing else "#8c959f" if e.recorded is None else "#1a7f37"))
                self.table.setItem(row, col, cell)
        left = len(self.missing())
        text = (f"{len(self.pending)} 集准备改到新位置，点「应用」生效。" if self.pending else "")
        text += (f"还有 {left} 集找不到视频。" if left else "" if self.pending else "所有集的视频都找得到。")
        self.message.setText(text + (("\n" + note) if note else ""))
        self.apply_button.setEnabled(bool(self.pending))

    def _match(self, folder: Path) -> int:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            found = match_folder(self.missing(), folder)
        finally:
            QApplication.restoreOverrideCursor()
        taken = {path for path, _rule in self.pending.values()}
        added = 0
        for eid, (path, rule) in found.items():
            if path not in taken:
                self.pending[eid] = (path, rule)
                added += 1
        return added

    def search_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "在这个文件夹（含子文件夹）里找视频")
        if folder:
            added = self._match(Path(folder))
            self.refresh(f"在「{folder}」里配上 {added} 集。")

    def choose_file(self) -> None:
        row = self.table.currentRow()
        if not 0 <= row < len(self.entries):
            QMessageBox.information(self, "视频文件位置", "先在表格里选一集。")
            return
        entry = self.entries[row]
        start = str(entry.recorded.parent) if entry.recorded and entry.recorded.parent.exists() else ""
        path, _ = QFileDialog.getOpenFileName(self, f"选择 {entry.eid} 的视频", start, VIDEO_FILTER)
        if not path:
            return
        self.pending = {eid: value for eid, value in self.pending.items() if value[0] != Path(path)}
        self.pending[entry.eid] = (Path(path), "你选的")
        added = self._match(Path(path).parent)
        self.refresh(f"同一个文件夹里另外配上 {added} 集。" if added else "")

    def apply(self) -> None:
        errors = []
        for entry in self.entries:
            if entry.eid not in self.pending:
                continue
            path = self.pending[entry.eid][0]
            try:
                relink(entry, path, self.project.cache_dir)
                self.applied.append((entry.eid, path))
            except (OSError, ValueError) as exc:
                errors.append(f"{entry.eid}：{exc}")
        self.pending.clear()
        if errors:
            QMessageBox.warning(self, "视频文件位置", "有几集没改成：\n" + "\n".join(errors))
        self.accept()
