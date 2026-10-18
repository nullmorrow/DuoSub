"""Shared release/check font results with private library file location."""

from __future__ import annotations

from pathlib import Path
import subprocess

from PySide6.QtCore import Qt

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from subkit.core.fontcheck import FontCheck, FontLibrary, report_lines
from .. import help as app_help


class FontCheckDialog(QDialog):
    help_anchor = "fonts"

    def __init__(self, check: FontCheck, parent=None, *, library: FontLibrary | None = None,
                 exporting: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("检查字体（缺字体、缺字）")
        self.resize(700, 420)
        layout = QVBoxLayout(self)
        area = QScrollArea()
        area.setWidgetResizable(True)
        contents = QWidget()
        rows = QVBoxLayout(contents)
        summary = QLabel("\n".join(report_lines(check)))
        summary.setWordWrap(True)
        summary.setTextInteractionFlags(summary.textInteractionFlags() | Qt.TextInteractionFlag.TextSelectableByMouse)
        rows.addWidget(summary)
        if library is not None and check.missing:
            label = QLabel(f"字体库位置：{library.path}")
            label.setWordWrap(True)
            rows.addWidget(label)
            for family in check.missing:
                paths = library.lookup(family)
                if not paths:
                    rows.addWidget(QLabel(f"{family}：字体库里也没有"))
                for value in paths:
                    path = Path(value)
                    row = QHBoxLayout()
                    label = QLabel(f"{family}：{path}")
                    label.setWordWrap(True)
                    row.addWidget(label, 1)
                    if path.is_file():
                        button = QPushButton("在资源管理器中显示")
                        button.clicked.connect(lambda _checked=False, p=path: self.show_file(p))
                        row.addWidget(button)
                    else:
                        label.setText(f"{family}：字体库里记的文件已经不在了：{path}。可以用 assfonts 重新建库")
                    rows.addLayout(row)
            if library.error:
                rows.addWidget(QLabel(f"字体库读取失败：{library.error}"))
        rows.addStretch()
        area.setWidget(contents)
        layout.addWidget(area)
        buttons = QDialogButtonBox()
        if exporting:
            buttons.addButton("仍然导出", QDialogButtonBox.ButtonRole.AcceptRole)
            buttons.addButton("取消", QDialogButtonBox.ButtonRole.RejectRole)
        else:
            buttons.addButton("关闭", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        app_help.dialog_buttons(self, buttons)
        layout.addWidget(buttons)

    def show_file(self, path: Path) -> None:
        if not path.is_file():
            QMessageBox.warning(self, "检查字体", f"字体库里记的文件已经不在了：{path}。可以用 assfonts 重新建库")
            return
        try:
            # Not subkit.proc.run: that hides windows, and explorer exits with 1 even when it worked.
            subprocess.Popen(["explorer.exe", "/select,", str(path.resolve())])
        except OSError as exc:
            QMessageBox.warning(self, "检查字体", f"无法在资源管理器中显示：{exc}")
