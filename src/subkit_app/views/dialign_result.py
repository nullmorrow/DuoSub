"""Non-modal dialogue analysis summary; applying timings is a separate action."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from ..help import dialog_buttons


class DialogueAlignResultDialog(QDialog):
    help_anchor = "dialogue-align"
    applyRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("精确对话轴")
        self.setModal(False)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.offset = QLabel()
        layout.addWidget(self.offset)
        self.stale = QLabel()
        layout.addWidget(self.stale)
        buttons = QDialogButtonBox()
        self.apply_button = buttons.addButton("整集采用…", QDialogButtonBox.ButtonRole.ActionRole)
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.applyRequested)
        buttons.addButton("关闭", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.rejected.connect(self.close)
        dialog_buttons(self, buttons, primary=self.apply_button)
        layout.addWidget(buttons)

    def set_report(self, aligned, fallback, disagree, stale, values, source):
        self.summary.setText(f"已对齐 {aligned} 句，{fallback} 句没有日文或对不上（用能量判断），{disagree} 句两种识别不一致")
        self.offset.setText(f"提前 {values[0]} ms，延后 {values[1]} ms（{source}）")
        self.stale.setText(f"{stale} 句文字改过，需要重新分析" if stale else "")
        self.apply_button.setEnabled(bool(aligned))
