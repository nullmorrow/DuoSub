"""「文件 → 导出 → 整季发布用 ASS…」: output folder, language suffix, names and the overwrite question."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

from subkit.core.season_export import SUFFIXES, SeasonItem, check_suffix, plan, season_entries
from ..help import dialog_buttons


class SeasonExportDialog(QDialog):
    """After Accepted, :meth:`chosen` lists ``(episode id, output path)`` in episode order."""

    help_anchor = "export"

    def __init__(self, project, settings, current_eid: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导出整季发布用 ASS")
        self.resize(900, 520)
        self.project, self.settings = project, settings
        self.entries = season_entries(project)
        self.items: list[SeasonItem] = []
        self.overwrite = True
        self._review_stamp = time.strftime("%Y%m%d-%H%M%S")
        key = project.data.get("id", "")
        self._folder_key = f"export/season_folder/{key}"
        videos = [e.video for e in self.entries if e.video is not None]
        folder = settings.value(self._folder_key, "") or (str(videos[0].parent) if videos else "")
        suffix = settings.value("export/season_suffix", "") or project.languages.get("translation", "zh-Hant")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        row = QHBoxLayout()
        self.folder = QLineEdit(folder)
        self.folder.setPlaceholderText("选择输出文件夹")
        choose = QPushButton("选择…")
        choose.clicked.connect(self._choose)
        row.addWidget(self.folder, 1)
        row.addWidget(choose)
        form.addRow("输出文件夹", row)
        self.suffix = QComboBox()
        self.suffix.setEditable(True)
        self.suffix.addItems(list(dict.fromkeys([suffix, *SUFFIXES])))
        self.suffix.setCurrentText(suffix)
        self.suffix.setToolTip("写在视频文件名和 .ass 之间，例如 zh-Hant → 第05集.zh-Hant.ass")
        form.addRow("语言后缀", self.suffix)
        self.form = form
        self.zh_both = QCheckBox("同时导出简体和繁体两份")
        self.zh_both.setChecked(settings.value("export/zh_both", False, type=bool))
        self.simplified_suffix = QLineEdit(settings.value("export/season_simplified_suffix", "zh-Hans"))
        self.traditional_suffix = QLineEdit(settings.value("export/season_traditional_suffix", "zh-Hant"))
        form.addRow(self.zh_both)
        form.addRow("简体后缀", self.simplified_suffix)
        form.addRow("繁体后缀", self.traditional_suffix)
        self.annotation_order = QComboBox()
        self.annotation_order.addItem("统一放在文件末尾", "end")
        self.annotation_order.addItem("紧跟对应正文行", "after_parent")
        self.annotation_order.setToolTip("对白和歌曲的 Ruby / Furigana / Pinyin 注音行；歌曲有多条特效事件时，跟在对应正文的整块事件之后。")
        form.addRow("注音行顺序", self.annotation_order)
        layout.addLayout(form)
        cleanup_note = QLabel("发布 ASS 自动清理每行首尾半角空格并跳过空白字幕，不改工程原文。")
        cleanup_note.setWordWrap(True)
        layout.addWidget(cleanup_note)
        self.review_packet = QCheckBox("同时导出审议包（发布前给大模型复查）")
        self.review_packet.setChecked(settings.value("export/season_review", False, type=bool))
        self.review_packet.setToolTip("每个实际导出的集另存 Markdown 和 JSONL；沿用单独导出审议包的显示规则，简繁两份 ASS 共用一份包。")
        layout.addWidget(self.review_packet)
        self.review_note = QLabel("审议包放在输出文件夹，按集名和时间另存，不覆盖旧包。每集一份，沿用工程当前显示规则，供模型复查后导回意见。")
        self.review_note.setWordWrap(True)
        self.review_note.setStyleSheet("color:#57606a;")
        self.review_note.setVisible(self.review_packet.isChecked())
        self.review_packet.toggled.connect(self.review_note.setVisible)
        layout.addWidget(self.review_note)
        self.reference = QCheckBox("参考原配简繁字幕，附对照报告")
        self.reference_path = QLineEdit()
        self.reference_path.setPlaceholderText("留空：自动在各集原字幕目录查找原配简繁字幕")
        reference_row = QHBoxLayout()
        reference_row.addWidget(self.reference_path, 1)
        self.reference_choose = QPushButton("选择参考文件夹…")
        reference_row.addWidget(self.reference_choose)
        self.reference_choose.clicked.connect(self._choose_reference)
        self.reference_path.setEnabled(False)
        self.reference_choose.setEnabled(False)
        self.reference.toggled.connect(self.reference_path.setEnabled)
        self.reference.toggled.connect(self.reference_choose.setEnabled)
        layout.addWidget(self.reference)
        layout.addLayout(reference_row)
        reference_note = QLabel("自动导出，不逐条确认。缺少参考或匹配不唯一时回退自动简繁转换；每个 ASS 附一份 HTML 对照报告，与审议包分别生成。")
        reference_note.setWordWrap(True)
        layout.addWidget(reference_note)
        note = QLabel("文件名 = 视频文件名（去掉 .mkv）+ .语言后缀 + .ass。" +
                      (f"正在编辑的 {current_eid} 按窗口里现在的内容导出（包括还没保存的改动），"
                       "其他集按已保存的内容。" if current_eid else "各集按已保存的内容导出。"))
        note.setWordWrap(True)
        note.setStyleSheet("color:#57606a;")
        layout.addWidget(note)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["集", "输出文件", "状态"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)
        self.reason = QLabel()
        self.reason.setStyleSheet("color:#cf222e;")
        layout.addWidget(self.reason)
        buttons = QDialogButtonBox()
        self.export_button = buttons.addButton("导出", QDialogButtonBox.ButtonRole.AcceptRole)
        cancel = buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        cancel.setText("取消")
        self.export_button.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        dialog_buttons(self, buttons)
        layout.addWidget(buttons)
        self.folder.textChanged.connect(self.refresh)
        self.suffix.currentTextChanged.connect(self.refresh)
        self.zh_both.toggled.connect(self.refresh)
        self.simplified_suffix.textChanged.connect(self.refresh)
        self.traditional_suffix.textChanged.connect(self.refresh)
        self.refresh()

    def _choose(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹", self.folder.text())
        if path:
            self.folder.setText(path)

    def _choose_reference(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择原配字幕文件夹", self.reference_path.text())
        if path:
            self.reference_path.setText(path)

    def refresh(self) -> None:
        both = self.zh_both.isChecked()
        self.form.setRowVisible(self.suffix, not both)
        self.form.setRowVisible(self.simplified_suffix, both)
        self.form.setRowVisible(self.traditional_suffix, both)
        suffix, error = check_suffix(self.simplified_suffix.text() if both else self.suffix.currentText())
        traditional, other_error = check_suffix(self.traditional_suffix.text()) if both else (None, "")
        error = error or other_error
        if both and not error and suffix.casefold() == traditional.casefold():
            error = "简体后缀和繁体后缀不能相同"
        suffix_error = error
        folder = Path(self.folder.text().strip()) if self.folder.text().strip() else None
        if not error and (folder is None or not folder.is_dir()):
            error = "请选择已存在的输出文件夹"
        self.items = plan(self.entries, folder, suffix, traditional_suffix=traditional) if not error else []
        # Without a usable folder the names are still shown, without checking what exists.
        rows = self.items or plan(self.entries, Path(""), suffix or "?", check_files=False,
                                  traditional_suffix=traditional if both and not suffix_error else None)
        self.table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            for col, value in enumerate((item.eid, item.target.name if item.target else "", self._status(item))):
                cell = QTableWidgetItem(value)
                cell.setToolTip(str(item.target) if col == 1 and item.target and self.items else value)
                if col == 2:
                    colour = ("#cf222e" if item.duplicate_of else "#8c959f" if item.target is None
                              else "#9a6700" if item.exists else None)
                    if colour:
                        cell.setForeground(QColor(colour))
                self.table.setItem(row, col, cell)
        duplicates = [i for i in self.items if i.duplicate_of]
        if not error and duplicates:
            error = "有几集输出文件名相同：" + "、".join(f"{i.eid} 和 {i.duplicate_of}" for i in duplicates)
        if not error and not any(i.target for i in self.items):
            error = "没有可以导出的集（都还没有合并）"
        self.reason.setText(error)
        self.export_button.setEnabled(not error)

    @staticmethod
    def _status(item: SeasonItem) -> str:
        if item.duplicate_of:
            return f"和 {item.duplicate_of} 同名"
        if item.target is None:
            return item.note
        return "；".join([item.note] * bool(item.note) + ["已存在（导出时询问）" if item.exists else "将导出"])

    def accept(self) -> None:
        self.refresh()
        if not self.export_button.isEnabled():
            return
        existing = [i for i in self.items if i.target is not None and i.exists]
        self.overwrite = True
        if existing:
            box = QMessageBox(QMessageBox.Icon.Question, "导出整季发布用 ASS",
                              f"输出文件夹里已经有 {len(existing)} 个同名文件：\n"
                              + "\n".join(i.target.name for i in (existing if self.zh_both.isChecked() else existing[:10]))
                              + (f"\n……共 {len(existing)} 个" if len(existing) > 10 and not self.zh_both.isChecked() else ""), parent=self)
            overwrite = box.addButton("覆盖", QMessageBox.ButtonRole.AcceptRole)
            skip = box.addButton("跳过这些文件" if self.zh_both.isChecked() else "跳过这些集", QMessageBox.ButtonRole.ActionRole)
            box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            box.help_anchor = "export-season"
            dialog_buttons(box)
            box.exec()
            if box.clickedButton() not in (overwrite, skip):
                return
            self.overwrite = box.clickedButton() is overwrite
            if not self.overwrite and len(existing) == sum(1 for i in self.items if i.target is not None):
                QMessageBox.information(self, "导出整季发布用 ASS", "所有要导出的集都已存在，跳过后没有要导出的。")
                return
        self.settings.setValue("export/season_review", self.review_packet.isChecked())
        self.settings.setValue("export/zh_both", self.zh_both.isChecked())
        self.settings.setValue("export/season_simplified_suffix", check_suffix(self.simplified_suffix.text())[0])
        self.settings.setValue("export/season_traditional_suffix", check_suffix(self.traditional_suffix.text())[0])
        self.settings.setValue(self._folder_key, self.folder.text().strip())
        self.settings.setValue("export/season_suffix", check_suffix(self.suffix.currentText())[0])
        super().accept()

    def chosen(self) -> list[tuple[str, Path] | tuple[str, Path, str]]:
        return [(i.eid, i.target, i.direction) if i.direction else (i.eid, i.target) for i in self.items
                if i.target is not None and not i.duplicate_of and (self.overwrite or not i.exists)]

    def review_targets(self) -> dict[str, tuple[Path, Path]]:
        from subkit.core.review import available_packet_paths

        if not self.review_packet.isChecked():
            return {}
        folder = Path(self.folder.text().strip())
        return {eid: available_packet_paths(folder, episode=eid, stamp=self._review_stamp)
                for eid in dict.fromkeys(item[0] for item in self.chosen())}

    def skipped(self) -> list[tuple[str, str]]:
        """(episode id, reason) for episodes this export leaves out."""
        return list(dict.fromkeys(
            (i.eid, i.note if i.target is None else (i.target.name + "：" if i.direction else "") + "已存在，按你的选择跳过")
            for i in self.items if i.target is None or (i.exists and not self.overwrite)))


class SeasonReportDialog(QDialog):
    help_anchor = "export"

    def __init__(self, text: str, folder: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("整季导出报告")
        self.resize(820, 560)
        self.folder = folder
        layout = QVBoxLayout(self)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlainText(text)
        layout.addWidget(self.text)
        buttons = QHBoxLayout()
        show = QPushButton("打开输出文件夹")
        show.clicked.connect(self.show_folder)
        close = QPushButton("关闭")
        close.clicked.connect(self.accept)
        buttons.addWidget(show)
        buttons.addStretch(1)
        buttons.addWidget(close)
        dialog_buttons(self, buttons, auxiliary=(show,))
        layout.addLayout(buttons)

    def show_folder(self) -> None:
        try:
            subprocess.Popen(["explorer.exe", str(self.folder.resolve())])
        except OSError as exc:
            QMessageBox.warning(self, "整季导出报告", f"打开输出文件夹失败：{exc}")
