"""Compact numeric positioning controls; the owner supplies preview coordinates."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, QTimer, Signal
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout,
                               QLabel, QSizePolicy, QToolButton, QWidget)

GRID_TIP = "对齐点：单行字不动，只换 X/Y 指的那个点和旋转中心；多行字幕按新的对齐方式重新排列各行"


class PositionBar(QWidget):
    help_anchor = "position"
    xyRequested = Signal(object, object)  # x / y; None leaves that axis alone
    alignmentRequested = Signal(int)
    actionRequested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.settings = None
        self._shown = (0.0, 0.0)
        from .waveform import FlowLayout

        lay = FlowLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        lay.addWidget(QLabel("位置"))
        lab = QLabel("对齐点")
        lab.setToolTip(GRID_TIP)
        lay.addWidget(lab)
        alignment = self.alignment_grid = QWidget()
        grid = QGridLayout(alignment)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(1)
        self.alignments: dict[int, QToolButton] = {}
        self.buttons = QButtonGroup(self)
        self.buttons.setExclusive(True)
        for row, values in enumerate(((7, 8, 9), (4, 5, 6), (1, 2, 3))):
            for col, an in enumerate(values):
                b = QToolButton(text=str(an))
                b.setCheckable(True)
                b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)   # follows the current font
                b.setToolTip(GRID_TIP)
                self.buttons.addButton(b, an)
                self.alignments[an] = b
                grid.addWidget(b, row, col)
        # The section starts collapsed; initialise the grid before its first show.
        alignment.resize(grid.sizeHint())
        grid.setGeometry(alignment.rect())
        self.buttons.idClicked.connect(self.alignmentRequested)
        lay.addWidget(alignment)
        self.x, self.y = QDoubleSpinBox(), QDoubleSpinBox()
        for label, box in (("X", self.x), ("Y", self.y)):
            box.setRange(-1000000, 1000000)
            box.setDecimals(1)
            box.setSingleStep(0.1)
            box.setKeyboardTracking(False)
            box.setMaximumWidth(105)
            lay.addWidget(QLabel(label))
            lay.addWidget(box)
        self.x.valueChanged.connect(lambda v: self._coordinate(0, v))
        self.y.valueChanged.connect(lambda v: self._coordinate(1, v))
        self.scope_box = QComboBox()
        self.scope_box.addItem("整组", "group")
        self.scope_box.addItem("只这一行", "line")
        self.scope_box.currentIndexChanged.connect(self._remember_scope)
        lay.addWidget(self.scope_box)
        self.presets: dict[str, QToolButton] = {}
        for action, label, tip in (
            ("restore", "恢复原位置", "去掉手动位置，回到原字幕自带的位置（没有就用样式位置）"),
            ("left", "左侧", "左对齐贴左边距，每行高度不变（两人同时说话时用）"),
            ("right", "右侧", "右对齐贴右边距，每行高度不变"),
            ("top", "顶部", "换成模板的顶部样式（CN.Top / JP.Top）"),
            ("bottom", "底部", "回到普通的底部样式"),
            ("alt", "底部第二人", "换成模板的第二人样式（CN.Alt / JP.Alt）：底部居中，在普通底部字幕上方（两人同时说话时用）"),
        ):
            b = QToolButton(text=label)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, a=action: self.actionRequested.emit(a))
            self.presets[action] = b
            lay.addWidget(b)
        self.setEnabled(False)

    def set_settings(self, settings) -> None:
        self.settings = settings
        value = settings.value("position/scope", "group")
        self.scope_box.setCurrentIndex(1 if value == "line" else 0)

    def sizeHint(self) -> QSize:  # noqa: N802
        hint = super().sizeHint()
        return QSize(hint.width(), self.layout().heightForWidth(self.width()))

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            QTimer.singleShot(0, self, self._layout_alignment)

    def _layout_alignment(self) -> None:
        grid = self.alignment_grid.layout()
        grid.invalidate()
        self.alignment_grid.resize(grid.sizeHint())
        grid.setGeometry(self.alignment_grid.rect())

    def scope(self) -> str:
        return self.scope_box.currentData()

    def _remember_scope(self, _index: int) -> None:
        if self.settings is not None:
            self.settings.setValue("position/scope", self.scope())

    def show_anchor(self, anchor: dict | None) -> None:
        self.setEnabled(anchor is not None)
        if anchor is None:
            return
        self._shown = (float(anchor["x"]), float(anchor["y"]))
        for box, value in zip((self.x, self.y), self._shown):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
            box.setToolTip("脚本像素，0.1 px；显示起点，修改时两个端点一起移动" if anchor.get("move")
                           else "脚本像素，0.1 px；回车、离开输入框或按箭头生效")
        self.alignments[int(anchor["an"])].setChecked(True)

    def _coordinate(self, axis: int, value: float) -> None:
        if abs(value - self._shown[axis]) < 0.05:
            return
        shown = list(self._shown)
        shown[axis] = value
        self._shown = tuple(shown)
        self.xyRequested.emit(round(value, 1) if axis == 0 else None,
                              round(value, 1) if axis == 1 else None)
