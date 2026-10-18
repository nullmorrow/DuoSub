"""Exact start/end inputs for one group, with the waveform's lane targets."""

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFormLayout, QLabel, QLineEdit, QVBoxLayout)

from subkit.core import karaoke as kara
from subkit.core.time_input import TIME_FORMAT, format_time, parse_time

from .. import help as app_help


class GroupTimesDialog(QDialog):
    help_anchor = "timing"

    def __init__(self, doc, gid, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置开始和结束时间")
        group = doc.group(gid)
        members = [m for m in doc.members_of(group) if m.role != "comment"]
        start, end = doc.group_time(group)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.start = QLineEdit(format_time(start))
        self.end = QLineEdit(format_time(end))
        for edit in (self.start, self.end):
            edit.setPlaceholderText("0:01:23.45")
            edit.setToolTip("时:分:秒.厘秒，也可填毫秒（0:01:23.456）、分:秒（1:23.45）或纯数字毫秒")
        form.addRow("开始", self.start)
        form.addRow("结束", self.end)
        self.target = QComboBox()
        for label, key in (("整组", "group"), ("只改翻译行", "translation"), ("只改原声行", "original")):
            self.target.addItem(label, key)
            self.target.model().item(self.target.count() - 1).setEnabled(
                any(key == "group" or m.lane == key for m in members))
        form.addRow("作用于", self.target)
        layout.addLayout(form)
        self.shift_beats = QCheckBox("K 轴拍跟着开始时间一起挪")
        self.shift_beats.setChecked(True)
        self.shift_beats.setVisible(group.kind == "song" and any(
            kara.has_karaoke(doc, m) or kara.has_k_tags(m) for m in members))
        layout.addWidget(self.shift_beats)
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
        self.start.textChanged.connect(self._validate)
        self.end.textChanged.connect(self._validate)
        self._validate()
        self.start.selectAll()

    def values(self):
        return parse_time(self.start.text()), parse_time(self.end.text()), self.target.currentData(), self.shift_beats.isChecked()

    def _validate(self, *_args):
        try:
            start, end, *_ = self.values()
            error = "结束要晚于开始" if end <= start else ""
        except ValueError:
            error = TIME_FORMAT
        self.hint.setText(error)
        self.ok.setEnabled(not error)

    def accept(self):
        self._validate()
        if self.ok.isEnabled():
            super().accept()
