"""One undoable time shift for the selected groups."""

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                               QRadioButton, QSpinBox, QVBoxLayout)

from .. import help as app_help


class ShiftGroupsDialog(QDialog):
    help_anchor = "timing"

    def __init__(self, doc, gids, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.earliest = min((m.start for gid in gids for m in doc.members_of(doc.group(gid))), default=0)
        self.setWindowTitle("整体平移")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"把选中的 {len(gids)} 组"))
        row = QHBoxLayout()
        self.earlier = QRadioButton("提前")
        self.later = QRadioButton("推后")
        self.earlier.setChecked(True)
        row.addWidget(self.earlier)
        row.addWidget(self.later)
        self.milliseconds = QSpinBox()
        self.milliseconds.setRange(0, 600000)
        self.milliseconds.setSingleStep(10)
        self.milliseconds.setValue(settings.value("timing/shift_ms", 100, type=int))
        row.addWidget(self.milliseconds)
        row.addWidget(QLabel("毫秒"))
        layout.addLayout(row)
        self.hint = QLabel()
        self.hint.setStyleSheet("color:#cf222e;")
        layout.addWidget(self.hint)
        self.buttons = QDialogButtonBox()
        self.ok = self.buttons.addButton("确定", QDialogButtonBox.ButtonRole.AcceptRole)
        self.buttons.addButton("取消", QDialogButtonBox.ButtonRole.RejectRole)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        app_help.dialog_buttons(self, self.buttons)
        layout.addWidget(self.buttons)
        self.earlier.toggled.connect(self._validate)
        self.milliseconds.valueChanged.connect(self._validate)
        self._validate()

    def delta(self):
        value = self.milliseconds.value()
        return -value if self.earlier.isChecked() else value

    def _validate(self, *_args):
        self.hint.setText("" if self.earliest + self.delta() >= 0 else "开始和结束都夹到 0")

    def accept(self):
        if self.ok.isEnabled():
            self.settings.setValue("timing/shift_ms", self.milliseconds.value())
            super().accept()
