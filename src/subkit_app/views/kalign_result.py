"""Reusable non-modal summary of automatic K timing writes."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QTreeWidget,
                              QTreeWidgetItem, QVBoxLayout)

from .. import help as app_help


class KalignResultDialog(QDialog):
    help_anchor = "karaoke-auto"
    groupRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("自动 K 轴")
        self.setModal(False)
        self.resize(660, 400)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.offset = QLabel()
        layout.addWidget(self.offset)
        self.rows = QTreeWidget()
        self.rows.setHeaderLabels(["句子", "结果", "需复查拍数"])
        self.rows.setRootIsDecorated(False)
        self.rows.setColumnWidth(0, 280)
        self.rows.itemDoubleClicked.connect(self._jump)
        layout.addWidget(self.rows)
        buttons = QDialogButtonBox()
        buttons.addButton("关闭", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.rejected.connect(self.close)
        app_help.dialog_buttons(self, buttons)
        layout.addWidget(buttons)

    def set_report(self, report, rows):
        text = (f"已更新 {report['updated']} 句，跳过 {len(report['skipped'])} 句（已确认或手调过），"
                f"{report['lines_with_check']} 句有拍需要复查")
        if report.get("stale"):
            text += f"；{len(report['stale'])} 句在处理期间改过，没有写入，请重新运行"
        if report.get("clipped", 0) > 0:
            text += f"\n{report['clipped']} 句放宽时碰到相邻的句，拍已截短（橙色框）"
        self.summary.setText(text)
        self.offset.clear()
        self.rows.clear()
        for gid, text, status, checks in rows:
            item = QTreeWidgetItem([text, status, str(checks)])
            item.setData(0, Qt.ItemDataRole.UserRole, gid)
            self.rows.addTopLevelItem(item)

    def _jump(self, item, _column):
        gid = item.data(0, Qt.ItemDataRole.UserRole)
        if gid:
            self.groupRequested.emit(gid)
