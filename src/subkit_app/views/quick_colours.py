"""Project-owned quick colours shared by the two text editors."""
from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QHBoxLayout, QMenu, QToolButton, QWidget

from subkit.core.workspace import DEFAULT_QUICK_COLORS


class QuickColourModel(QObject):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None

    def set_project(self, project):
        self.project = project
        self.changed.emit()

    @property
    def colours(self):
        return self.project.quick_colors if self.project is not None else list(DEFAULT_QUICK_COLORS)

    def save(self, colours):
        if self.project is None:
            return
        self.project.set_quick_colors(colours)
        self.changed.emit()


class QuickColours(QWidget):
    help_anchor = "format"
    formatRequested = Signal(dict)

    def __init__(self, current_colour, parent=None, *, height=24):
        super().__init__(parent)
        self.current_colour = current_colour
        self.model = None
        self.button_height = height
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(3)
        self.buttons = []
        self.refresh()

    def set_model(self, model):
        if self.model is not None:
            self.model.changed.disconnect(self.refresh)
        self.model = model
        model.changed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        while self.row.count():
            item = self.row.takeAt(0)
            item.widget().hide()
            item.widget().deleteLater()
        colours = self.model.colours if self.model else list(DEFAULT_QUICK_COLORS)
        self.buttons = []
        for i, colour in enumerate(colours):
            button = QToolButton(self)
            button.setFixedSize(self.button_height, self.button_height)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setStyleSheet(f"QToolButton {{ background: {colour}; border: 1px solid #8B949E; padding: 0; }}")
            button.setToolTip(f"{colour}：点一下给选中的字（没选就是整行）上色；右键改颜色或删除")
            button.clicked.connect(lambda _=False, c=colour: self._apply(c))
            button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            button.customContextMenuRequested.connect(lambda pos, n=i, b=button: self._menu(n, b.mapToGlobal(pos)))
            self.row.addWidget(button)
            self.buttons.append(button)
        self.plus = QToolButton(text="＋", parent=self)
        self.plus.setFixedSize(self.button_height, self.button_height)
        self.plus.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.plus.setStyleSheet("QToolButton:disabled { color: #8B949E; }")
        self.plus.setToolTip("最多 12 个" if len(colours) >= 12 else "添加快速颜色")
        self.plus.setEnabled(len(colours) < 12)
        self.plus.clicked.connect(self.add_colour)
        self.row.addWidget(self.plus)
        self.setEnabled(self.model is not None and self.model.project is not None)

    def _apply(self, colour):
        from .format_bar import ass_colour
        self.formatRequested.emit({"c1": ass_colour(QColor(colour))})

    def add_colour(self):
        if self.model is None or self.model.project is None or len(self.model.colours) >= 12:
            return
        colour = QColorDialog.getColor(self.current_colour(), self, "字的颜色",
                                          options=QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if colour.isValid():
            self.model.save(self.model.colours + [colour.name().upper()])

    def edit_colour(self, index):
        colour = QColorDialog.getColor(QColor(self.model.colours[index]), self, "字的颜色",
                                          options=QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if colour.isValid():
            colours = self.model.colours
            colours[index] = colour.name().upper()
            self.model.save(colours)

    def remove_colour(self, index):
        colours = self.model.colours
        del colours[index]
        self.model.save(colours)

    def move_colour(self, index, delta):
        colours = self.model.colours
        other = index + delta
        if 0 <= other < len(colours):
            colours[index], colours[other] = colours[other], colours[index]
            self.model.save(colours)

    def _menu(self, index, pos):
        menu = QMenu(self)
        edit = menu.addAction("改颜色…")
        remove = menu.addAction("删除")
        left = menu.addAction("往左移")
        right = menu.addAction("往右移")
        left.setEnabled(index > 0)
        right.setEnabled(index < len(self.model.colours) - 1)
        action = menu.exec(pos)
        if action is edit:
            self.edit_colour(index)
        elif action is remove:
            self.remove_colour(index)
        elif action is left:
            self.move_colour(index, -1)
        elif action is right:
            self.move_colour(index, 1)
        menu.deleteLater()
