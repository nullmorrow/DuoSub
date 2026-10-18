"""Project-wide width limits for dialogue and song furigana."""

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QGroupBox,
                               QHBoxLayout, QInputDialog, QLabel, QVBoxLayout)


class RubySettingsDialog(QDialog):
    help_anchor = "readings"

    def __init__(self, settings: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("振假名设置")
        layout = QVBoxLayout(self)
        self.controls = {}
        values = settings.get("max_width", {})
        for kind, title in (("dialogue", "台词"), ("song", "歌词")):
            group = QGroupBox(title)
            row = QHBoxLayout(group)
            enabled = QCheckBox("限制振假名宽度")
            value = values.get(kind, {})
            enabled.setChecked(value.get("enabled", True))
            combo = QComboBox()
            for percent, label in ((100, "100%（不超出汉字）"), (120, "120%（紧凑）"),
                                   (150, "150%（推荐）"), (200, "200%（宽松）")):
                combo.addItem(label, percent)
            percent = int(value.get("percent", 150))
            combo.addItem("自定义…", percent)
            index = combo.findData(percent)
            combo.setCurrentIndex(index)
            combo.setToolTip(f"{percent}%")
            combo.setEnabled(enabled.isChecked())
            combo.setProperty("previous_index", index)
            enabled.toggled.connect(combo.setEnabled)
            combo.activated.connect(lambda i, c=combo: self._choose_percent(c, i))
            row.addWidget(enabled)
            row.addWidget(combo)
            self.controls[kind] = enabled, combo
            layout.addWidget(group)
        min_scale = float(settings.get("min_scale", 60))
        explanation = QLabel(f"按所注汉字的排版占位宽度限制振假名，计入字体、字号、字距和横向缩放；超过上限时横向压缩，最多压到 {min_scale:g}%。保留居中、相邻振假名避让和人工缩放，F11 与 ASS 导出使用相同规则。")
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #888;")
        layout.addWidget(explanation)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        from ..help import dialog_buttons

        dialog_buttons(self, buttons)
        layout.addWidget(buttons)

    def _choose_percent(self, combo, index) -> None:
        if index == 4:
            percent, ok = QInputDialog.getInt(self, "自定义…", "限制振假名宽度（%）：",
                                              int(combo.itemData(4)), 100, 300)
            if not ok:
                combo.setCurrentIndex(combo.property("previous_index"))
                return
            combo.setItemData(4, percent)
        combo.setProperty("previous_index", index)
        combo.setToolTip(f"{combo.currentData()}%")

    def values(self) -> dict:
        return {kind: {"enabled": enabled.isChecked(), "percent": int(combo.currentData())}
                for kind, (enabled, combo) in self.controls.items()}
