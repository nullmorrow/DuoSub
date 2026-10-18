"""Fade in / out of the lyric lines of a song (K toolbar 「更多 ▾ → 歌词 → 淡入淡出」)."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLabel, QSpinBox, QVBoxLayout

from ..help import dialog_buttons


class SongFadeDialog(QDialog):
    """``fade`` after Accepted: ``(in, out)`` milliseconds, or None to go back to the source's own fade."""

    help_anchor = "karaoke"

    def __init__(self, groups: int, current: Optional[tuple[int, int]], overridden: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("淡入淡出")
        self.fade: Optional[tuple[int, int]] = None
        layout = QVBoxLayout(self)
        now = (f"淡入 {current[0]} 毫秒、淡出 {current[1]} 毫秒" if current is not None else "没有淡入淡出")
        intro = QLabel(f"{groups} 组歌词的原声和翻译一起改，原字幕不变。\n"
                       f"现在：{now}（{'改过' if overridden else '原字幕'}）")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self.fade_in, self.fade_out = QSpinBox(), QSpinBox()
        for box, value in ((self.fade_in, current[0] if current else 0), (self.fade_out, current[1] if current else 0)):
            box.setRange(0, 5000)
            box.setSingleStep(50)
            box.setSuffix(" 毫秒")
            box.setValue(value)
        form.addRow("淡入", self.fade_in)
        form.addRow("淡出", self.fade_out)
        layout.addLayout(form)
        hint = QLabel("0 = 不淡入／不淡出。翻译和原声的结束时间不同时淡出也不同步，可先用「翻译时间对齐原声」。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#57606a;")
        layout.addWidget(hint)
        buttons = QDialogButtonBox()
        self.apply_button = buttons.addButton("应用", QDialogButtonBox.ButtonRole.AcceptRole)
        self.restore_button = buttons.addButton("恢复原字幕的淡入淡出", QDialogButtonBox.ButtonRole.ResetRole)
        self.restore_button.setEnabled(overridden)
        self.restore_button.setToolTip("" if overridden else "还没有改过，正在用原字幕的淡入淡出")
        cancel = buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        cancel.setText("取消")
        self.apply_button.clicked.connect(self._apply)
        self.restore_button.clicked.connect(self._restore)
        buttons.rejected.connect(self.reject)
        dialog_buttons(self, buttons)
        layout.addWidget(buttons)

    def _apply(self) -> None:
        self.fade = (self.fade_in.value(), self.fade_out.value())
        self.accept()

    def _restore(self) -> None:
        self.fade = None
        self.accept()
