"""Schema-driven non-modal tool window (Windows IME must not use Qt.Popup)."""
from __future__ import annotations

import copy

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QInputDialog, QLabel, QListWidget, QMessageBox,
                               QPushButton, QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget)

from subkit.karaoke_fx import get_effect
from subkit.karaoke_fx.ktv import PARAMETERS
from .. import help as app_help

CHOICE_LABELS = {"off": "关闭", "corners": "左上 / 右下", "stacked": "上下两行居中",
                 "right": "左→右", "up": "下→上"}


class ColorList(QWidget):
    changed = Signal()

    def __init__(self, values, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.items = QListWidget()
        self.items.addItems(values)
        self.items.setMaximumHeight(80)
        layout.addWidget(self.items)
        row = QHBoxLayout()
        for label, callback in (("添加颜色…", self.add), ("修改颜色…", self.edit), ("移除颜色", self.remove)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)

    def value(self):
        return [self.items.item(i).text() for i in range(self.items.count())]

    def add(self):
        color = QColorDialog.getColor(QColor("#FFFFFF"), self, "颜色")
        if color.isValid():
            self.items.addItem(color.name().upper())
            self.changed.emit()

    def edit(self):
        item = self.items.currentItem()
        if item is not None:
            color = QColorDialog.getColor(QColor(item.text()), self, "颜色")
            if color.isValid():
                item.setText(color.name().upper())
                self.changed.emit()

    def remove(self):
        if self.items.currentRow() >= 0:
            self.items.takeItem(self.items.currentRow())
            self.changed.emit()


class KaraokeFxPanel(QWidget):
    help_anchor = "karaoke-fx"
    changed = Signal(str, object, str)
    presetsChanged = Signal()

    def __init__(self, store, parent=None):
        super().__init__(parent, Qt.WindowType.Tool)
        self.setWindowTitle("卡拉 OK 特效参数")
        self.store, self.effect, self.params = store, None, {}
        self.controls = {}
        layout = QVBoxLayout(self)
        self.title = QLabel()
        layout.addWidget(self.title)
        row = QHBoxLayout()
        row.addWidget(QLabel("应用到："))
        self.scope = QComboBox()
        for label, key in (("这一行", "line"), ("选中的组", "selection"), ("整首歌", "song")):
            self.scope.addItem(label, key)
        row.addWidget(self.scope)
        apply = QPushButton("应用")
        apply.clicked.connect(self._changed)
        row.addWidget(apply)
        layout.addLayout(row)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)
        row = QHBoxLayout()
        for label, callback in (("存为预设…", self.save), ("重命名预设…", self.rename), ("删除预设", self.delete)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        self.presets = QComboBox()
        layout.addWidget(self.presets)
        help_button = QPushButton("帮助")
        help_button.clicked.connect(lambda: app_help.open_manual(self.help_anchor))
        layout.addWidget(help_button)
        self.reload_presets()
        self.resize(570, 650)

    def reload_presets(self):
        current = self.presets.currentText()
        self.presets.clear()
        self.presets.addItems([e["name"] for e in self.store.list()])
        self.presets.setCurrentIndex(max(0, self.presets.findText(current)))

    def set_effect(self, effect, params):
        if effect == self.effect and params == self.params:
            return
        self.effect, self.params = effect, copy.deepcopy(params)
        fx = get_effect(effect)
        self.title.setText(fx.display_name)
        body = QWidget()
        form = QFormLayout(body)
        self.controls = {}
        schema = {**fx.schema, **PARAMETERS}
        for key, spec in schema.items():
            value = params.get(key, spec.default)
            if spec.type in ("int", "float"):
                control = QSpinBox() if spec.type == "int" else QDoubleSpinBox()
                control.setRange(spec.minimum, spec.maximum)
                if spec.type == "float":
                    control.setDecimals(2)
                control.setValue(value)
                slider = QSlider(Qt.Orientation.Horizontal)
                factor = 1 if spec.type == "int" else 100
                slider.setRange(round(spec.minimum * factor), round(spec.maximum * factor))
                slider.setValue(round(value * factor))
                slider.valueChanged.connect(lambda v, c=control, f=factor: c.setValue(v if f == 1 else v/f))
                control.valueChanged.connect(lambda v, s=slider, f=factor: s.setValue(round(v*f)))
                row = QWidget()
                boxes = QHBoxLayout(row)
                boxes.setContentsMargins(0, 0, 0, 0)
                boxes.addWidget(slider)
                boxes.addWidget(control)
                form.addRow(spec.label, row)
                control.valueChanged.connect(self._changed)
            elif spec.type == "bool":
                control = QCheckBox()
                control.setChecked(value)
                control.toggled.connect(self._changed)
                form.addRow(spec.label, control)
            elif spec.type == "choice":
                control = QComboBox()
                for choice in spec.choices:
                    control.addItem(CHOICE_LABELS.get(choice, choice), choice)
                control.setCurrentIndex(control.findData(value))
                control.currentIndexChanged.connect(self._changed)
                form.addRow(spec.label, control)
            elif spec.type == "color":
                control = QPushButton(value)
                control.clicked.connect(lambda _=False, c=control: self._color(c))
                form.addRow(spec.label, control)
            else:
                control = ColorList(value)
                control.changed.connect(self._changed)
                form.addRow(spec.label, control)
            self.controls[key] = control
        old = self.scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self.scroll.setWidget(body)

    def values(self):
        values = {}
        for key, control in self.controls.items():
            if isinstance(control, (QSpinBox, QDoubleSpinBox, ColorList)):
                values[key] = control.value()
            elif isinstance(control, QCheckBox):
                values[key] = control.isChecked()
            elif isinstance(control, QComboBox):
                values[key] = control.currentData()
            else:
                values[key] = control.text()
        return values

    def _changed(self, *_):
        self.params = self.values()
        self.changed.emit(self.effect, copy.deepcopy(self.params), self.scope.currentData())

    def _color(self, button):
        color = QColorDialog.getColor(QColor(button.text()), self, "颜色")
        if color.isValid():
            button.setText(color.name().upper())
            self._changed()

    def _mutate(self, callback):
        try:
            callback()
            self.reload_presets()
            self.presetsChanged.emit()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, self.windowTitle(), str(exc))

    def save(self):
        name, ok = QInputDialog.getText(self, "存为预设…", "名称")
        if ok:
            self._mutate(lambda: self.store.save(name, self.effect, self.values()))

    def rename(self):
        old = self.presets.currentText()
        if old:
            name, ok = QInputDialog.getText(self, "重命名预设…", "名称", text=old)
            if ok:
                self._mutate(lambda: self.store.rename(old, name))

    def delete(self):
        name = self.presets.currentText()
        if name and QMessageBox.question(self, "删除预设", f"删除预设「{name}」？") == QMessageBox.StandardButton.Yes:
            self._mutate(lambda: self.store.delete(name))
