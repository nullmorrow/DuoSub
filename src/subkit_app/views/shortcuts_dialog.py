"""Scene-grouped shortcut editor; changes stay private until accepted."""

from __future__ import annotations

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHeaderView, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from .. import help as app_help
from ..actions import SCENES, SCENE_PRIORITY, ActionRegistry, is_reserved


class KeyRecorder(QLineEdit):
    recorded = Signal(str)
    cancelled = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("shortcut_recording", True)
        self.setReadOnly(True)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)
        self.setPlaceholderText("请按新的快捷键…")
        self.composing = False

    def inputMethodEvent(self, event) -> None:  # noqa: N802
        self.composing = bool(event.preeditString())
        event.accept()

    def event(self, event) -> bool:
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Type.ShortcutOverride:
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self.composing or event.isAutoRepeat():
            return
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
        elif event.key() == Qt.Key.Key_Backspace:
            self.recorded.emit("")
        elif event.key() not in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt,
                                 Qt.Key.Key_Meta, Qt.Key.Key_AltGr, Qt.Key.Key_unknown):
            self.recorded.emit(QKeySequence(event.keyCombination()).toString(QKeySequence.SequenceFormat.PortableText))


class ShortcutsDialog(QDialog):
    help_anchor = "keys"

    def __init__(self, registry: ActionRegistry, parent=None, *, names=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("快捷键")
        self.setModal(True)
        self.resize(860, 620)
        self.registry = registry.copy()
        self.defaults = ActionRegistry(registry)
        self.names = names or {}
        self.rows = {}
        self.recorder = None
        self.recording_item = None
        layout = QVBoxLayout(self)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("搜索命令或按键")
        layout.addWidget(self.search)
        self.table = QTreeWidget(self)
        self.table.setHeaderLabels(["命令", "快捷键", "默认"])
        header = self.table.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setResizeContentsPrecision(-1)
        layout.addWidget(self.table)
        for scene, title in SCENES.items():
            group = QTreeWidgetItem(self.table, ["弹窗内部（不能改）" if scene == "popup" else title])
            group.setFlags(Qt.ItemFlag.ItemIsEnabled)
            for spec in registry:
                if scene not in spec.keys and not (scene == "watch" and "main" in spec.keys):
                    continue
                item = QTreeWidgetItem(group, [self.names.get(spec.id, spec.text)])
                item.setData(0, Qt.ItemDataRole.UserRole, (spec.id, scene))
                self.rows[spec.id, scene] = item
            group.setExpanded(True)
        self.message = QLabel(self)
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        controls = QHBoxLayout()
        layout.addLayout(controls)
        self.rebind = QPushButton("改键…", self)
        self.reset_one = QPushButton("恢复这一项", self)
        self.reset_all = QPushButton("全部恢复默认", self)
        self.manual = QPushButton("打开快捷键一览", self)
        for button in (self.rebind, self.reset_one):
            button.setAutoDefault(False)
            controls.addWidget(button)
        self.buttons = QDialogButtonBox(self)
        self.ok = self.buttons.addButton("确定", QDialogButtonBox.ButtonRole.AcceptRole)
        self.buttons.addButton("取消", QDialogButtonBox.ButtonRole.RejectRole)
        self.buttons.addButton(self.reset_all, QDialogButtonBox.ButtonRole.ResetRole)
        self.buttons.addButton(self.manual, QDialogButtonBox.ButtonRole.ActionRole)
        app_help.dialog_buttons(self, self.buttons, auxiliary=(self.reset_all, self.manual))
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.manual.clicked.connect(lambda: app_help.open_manual("keys"))
        self.search.textChanged.connect(self.filter_rows)
        self.table.currentItemChanged.connect(self.selection_changed)
        self.table.itemDoubleClicked.connect(lambda item, column: self.start_recording() if column == 1 else None)
        self.rebind.clicked.connect(self.start_recording)
        self.reset_one.clicked.connect(self.restore_one)
        self.reset_all.clicked.connect(self.restore_all)
        self.refresh()

    @staticmethod
    def menu_names(menu_bar) -> dict[str, str]:
        result = {}

        def visit(menu, path):
            for action in menu.actions():
                if action.menu() is not None:
                    title = re.sub(r"\(&.\)", "", action.text()).split("\t")[0]
                    visit(action.menu(), path + [title] if menu is not menu_bar else [])
                elif action.property("action_id"):
                    action_id = action.property("action_id")
                    # Scene headings already identify the K-axis commands.
                    prefix = [part for part in path if part != "K 轴"]
                    result.setdefault(action_id, " → ".join(prefix))

        visit(menu_bar, [])
        return result

    def current(self):
        item = self.table.currentItem()
        return item.data(0, Qt.ItemDataRole.UserRole) if item else None

    def selection_changed(self, *_args) -> None:
        # Focusing an embedded recorder can change the current column only.
        if self.recorder is not None and self.table.currentItem() is self.recording_item:
            return
        self.stop_recording()
        self.refresh()

    def stop_recording(self) -> None:
        if self.recorder is not None:
            self.table.removeItemWidget(self.recording_item, 1)
            self.recorder.deleteLater()
            self.recorder = self.recording_item = None

    def start_recording(self) -> None:
        binding = self.current()
        if not binding or not self.registry.lookup(binding[0]).customizable:
            return
        self.stop_recording()
        item = self.rows[binding]
        recorder = self.recorder = KeyRecorder(self.table)
        self.recording_item = item
        self.table.setItemWidget(item, 1, recorder)
        self.ok.setEnabled(False)
        recorder.recorded.connect(self.record_key)
        recorder.cancelled.connect(lambda: (self.stop_recording(), self.refresh()))
        recorder.setFocus()

    def record_key(self, key: str) -> None:
        binding = self.current()
        if not binding:
            return
        try:
            if key and is_reserved(key):
                raise ValueError(f"{key} 是保留键，不能分配")
            self.registry.set_keys(*binding, key)
        except ValueError as error:
            self.message.setText(str(error))
            return
        self.stop_recording()
        self.refresh()

    def restore_one(self) -> None:
        binding = self.current()
        if binding and self.registry.lookup(binding[0]).customizable:
            self.stop_recording()
            self.registry.reset(*binding)
            self.refresh()

    def restore_all(self) -> None:
        if QMessageBox.question(self, "快捷键", "把所有快捷键恢复为默认？") == QMessageBox.StandardButton.Yes:
            self.stop_recording()
            self.registry.reset_all()
            self.refresh()

    def refresh(self) -> None:
        conflicts = self.registry.conflicts()
        red = {(action_id, conflict.scene) for conflict in conflicts for action_id in conflict.action_ids}
        for binding, item in self.rows.items():
            action_id, scene = binding
            spec = self.registry.lookup(action_id)
            item.setText(1, "；".join(self.registry.effective_keys(*binding)))
            item.setText(2, "；".join(self.defaults.effective_keys(*binding)))
            color = QColor("#bd2020" if binding in red else "#888888" if not spec.customizable else self.palette().text().color())
            for column in range(3):
                item.setForeground(column, color)
        binding = self.current()
        editable = bool(binding and self.registry.lookup(binding[0]).customizable)
        self.rebind.setEnabled(editable)
        self.reset_one.setEnabled(editable)
        self.ok.setEnabled(not conflicts and self.recorder is None)
        messages = []
        if conflicts:
            conflict = conflicts[0]
            name = self.rows[conflict.action_ids[0], conflict.scene].text(0)
            messages.append(f"{conflict.key} 已用于「{name}」")
        if binding:
            spec = self.registry.lookup(binding[0])
            keys = self.registry.effective_keys(*binding)
            if not spec.typing and any(not QKeySequence(key)[0].keyboardModifiers() & (
                    Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier) for key in keys):
                messages.append("这个键在输入框里打字时会被当成文字，不会触发")
            scenes = {scene for (action_id, scene) in self.rows if scene != binding[1]
                      and scene != "popup" and set(keys) & set(self.registry.effective_keys(action_id, scene))}
            if scenes:
                priority = next(scene for scene in SCENE_PRIORITY if scene in scenes | {binding[1]})
                if priority == "k" or (priority == "watch" and binding[1] == "main"
                                        and any("watch" in s.keys and set(keys) & set(self.registry.effective_keys(s.id, "watch")) for s in self.registry)):
                    messages.append(f"{SCENES[priority]}里优先")
        self.message.setStyleSheet("color: #bd2020" if conflicts else "color: #777777")
        self.message.setText("\n".join(messages))
        self.filter_rows()

    def filter_rows(self, *_args) -> None:
        needle = self.search.text().strip().casefold()
        for item in self.rows.values():
            item.setHidden(bool(needle) and not any(needle in item.text(i).casefold() for i in range(3)))
        for i in range(self.table.topLevelItemCount()):
            group = self.table.topLevelItem(i)
            group.setHidden(all(group.child(j).isHidden() for j in range(group.childCount())))

    def accept(self) -> None:
        if not self.registry.conflicts() and self.recorder is None:
            super().accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_F1:
            app_help.open_manual("keys")
            event.accept()
        else:
            super().keyPressEvent(event)
