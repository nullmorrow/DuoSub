"""Explicit dock controls while leaving unused mouse events to Qt docking."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QScrollArea, QWidget


class DockTitleBar(QWidget):
    def __init__(self, dock, dock_back):
        super().__init__(dock)
        self.dock, self.dock_back = dock, dock_back
        self.help_anchor = getattr(dock, "help_anchor", "review")
        content = dock.widget()
        if content.layout() is not None:
            # The people controls keep their readable minimum inside a scrolling dock.
            scroll = QScrollArea(dock)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setMinimumSize(360, 200)
            scroll.setWidget(content)
            dock.setWidget(scroll)
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 2, 4, 2)
        title = QLabel(dock.windowTitle())
        title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        row.addWidget(title, 1)
        self.toggle = QPushButton()
        self.close_button = QPushButton("关闭")
        for button in (self.toggle, self.close_button):
            button.setAutoDefault(False)
            row.addWidget(button)
        self.toggle.clicked.connect(self.toggle_floating)
        self.close_button.clicked.connect(dock.close)
        self.close_button.setToolTip("关闭")
        dock.topLevelChanged.connect(self.update_toggle)
        self.update_toggle()
        # Fill both the window and its content even after reparenting / restoreState.
        for widget in (dock, dock.widget(), content, self):
            palette = widget.palette()
            color = QApplication.palette().color(QPalette.ColorRole.Window)
            color.setAlpha(255)
            palette.setColor(QPalette.ColorRole.Window, color)
            widget.setPalette(palette)
            widget.setBackgroundRole(QPalette.ColorRole.Window)
            widget.setAutoFillBackground(True)
            widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

    def update_toggle(self, _floating=False):
        text = "放回右侧" if self.dock.isFloating() else "浮动"
        self.toggle.setText(text)
        self.toggle.setToolTip(text)

    def toggle_floating(self):
        if self.dock.isFloating():
            self.dock_back(self.dock)
        else:
            self.dock.setFloating(True)
            self.dock.show()
            self.dock.activateWindow()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_floating()
            event.accept()
        else:
            event.ignore()

    def mousePressEvent(self, event):
        event.ignore()

    def mouseMoveEvent(self, event):
        event.ignore()

    def mouseReleaseEvent(self, event):
        event.ignore()
